"""Singularity-hosted vLLM backend.

Spawns ``vllm serve`` inside a Singularity container as a sibling subprocess
and talks to it over HTTP via the vLLM OpenAI-compatible server. The calling
Python process stays on bare metal — this is the only place in llenvs that
knows about Singularity.

Use this backend when the cluster's native ``vllm`` is too old for a given
model (e.g. gemma-4 needs vllm>=0.19 + transformers 5.x). For anything the
native vllm can serve, prefer
:class:`llenvs.inference.backends.vllm.VLLMBackend` — it's cheaper
(in-process, no HTTP round-trip, no subprocess lifecycle).

Per-cluster defaults for ``LLENVS_SIF`` / ``LLENVS_BINDS`` / ``LLENVS_HF_HOME``
live in ``llenvs/bin/_cluster.sh``; the backend is cluster-agnostic.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import IO, Any

from llenvs.inference.backends.api import OpenAIBackend, _run_concurrent
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    ScoringResult,
)
from llenvs.inference.scoring_utils import (
    build_scoring_inputs,
    parse_prompt_logprobs_http,
)

_log = logging.getLogger(__name__)


def _with_request_thinking_toggle(params: SamplingParams) -> SamplingParams:
    """Map ``params.disable_thinking`` onto vLLM's ``chat_template_kwargs``.

    The server renders the chat template itself, so the only way to switch a
    hybrid-reasoning model (e.g. Qwen3) out of thinking mode per request is
    the ``chat_template_kwargs`` request field. Caller-supplied values in
    ``params.extra["extra_body"]["chat_template_kwargs"]`` win on key
    collisions — the explicit escape hatch has priority, matching the
    ``extra_body`` merge convention of the API backends.
    """
    if not params.disable_thinking:
        return params
    extra = dict(params.extra) if params.extra else {}
    extra_body = dict(extra.get("extra_body") or {})
    extra_body["chat_template_kwargs"] = {
        "enable_thinking": False,
        **(extra_body.get("chat_template_kwargs") or {}),
    }
    extra["extra_body"] = extra_body
    return dataclasses.replace(params, extra=extra)


def _load_hf_tokenizer(model_path: str) -> Any:
    """Load the HF tokenizer for scoring-prompt rendering on the host.

    Only the tokenizer is fetched (no weights); it honors ``HF_HOME`` /
    ``HF_HUB_OFFLINE`` from the ambient environment. Kept module-level so
    tests can substitute it without a real download.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def _pick_free_port() -> int:
    """Pick a currently-free TCP port on 127.0.0.1.

    Classic bind-to-0 trick. A small TOCTOU race exists between close() and
    vllm serve's bind() — acceptable for single-node use. If the race bites
    in practice, we'll add a retry loop that detects "address already in use"
    in the server's stderr.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_default(explicit: str | None, env_var: str) -> str | None:
    if explicit is not None:
        return explicit
    val = os.environ.get(env_var, "")
    return val or None


class SingularityVLLMBackend(ModelBackend):
    """Run ``vllm serve`` inside a Singularity container, talk to it over HTTP.

    Composition over inheritance: once the vLLM OpenAI server is up, this
    class delegates all inference to an internally-held :class:`OpenAIBackend`
    pointing at ``http://127.0.0.1:<port>/v1``. The only extra responsibility
    is the subprocess lifecycle (spawn → health-check → teardown).

    Attributes:
        model_path: HF repo id or local path to the model.
        sif: Path to the Singularity .sif image.
        port: TCP port the vLLM server is bound to.
        pid: PID of the Singularity subprocess (None after close).
    """

    def __init__(
        self,
        model_path: str,
        *,
        sif: str | None = None,
        singularity_binds: tuple[str, ...] = (),
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
        served_model_name: str | None = None,
        port: int | None = None,
        startup_timeout: float = 900.0,
        health_poll_interval: float = 2.0,
        max_concurrency: int = 64,
        chat_template_kwargs: dict | None = None,
        extra_vllm_args: tuple[str, ...] = (),
        extra_singularity_env: dict[str, str] | None = None,
        hf_home: str | None = None,
        hf_offline: bool | None = None,
        cuda_visible_devices: str | None = None,
        server_log_path: str | None = None,
    ) -> None:
        """Spawn ``vllm serve`` in the container and wait until it's healthy.

        Args:
            model_path: HF repo id (e.g. ``google/gemma-4-31B-it``) or an
                absolute local path. On offline GPU nodes, prefer the local
                path (the container will not hit the network).
            sif: Path to the Singularity ``.sif`` image. Defaults to the
                ``LLENVS_SIF`` env var (typically set by
                ``source bin/_cluster.sh``).
            singularity_binds: ``--bind`` specs forwarded to ``singularity
                exec``. Defaults to splitting ``LLENVS_BINDS`` on whitespace.
                ``hf_home`` is added automatically if not already covered.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length. ``None`` lets vLLM pick.
            dtype: ``"bfloat16"``, ``"float16"``, or ``"auto"``.
            served_model_name: Short alias the HTTP client passes as ``model=``.
                Defaults to ``Path(model_path).name``.
            port: TCP port for the server. ``None`` picks a free port
                dynamically.
            startup_timeout: Seconds to wait for ``/health`` before giving up.
                Large VLMs with torch.compile warmup can take 10+ minutes cold.
            health_poll_interval: Seconds between ``/health`` polls.
            max_concurrency: Forwarded to the inner ``OpenAIBackend`` for
                batched chat calls.
            extra_vllm_args: Extra CLI flags passed to ``vllm serve``
                (e.g. ``("--swap-space", "4")``).
            extra_singularity_env: Extra env vars to forward into the
                container (each gets ``SINGULARITYENV_`` prefix). ``HF_HOME``
                and ``HF_HUB_OFFLINE`` are handled automatically.
            hf_home: HuggingFace cache root. Defaults to ``$HF_HOME`` or
                ``$LLENVS_HF_HOME``. Bound into the container so prefetched
                weights are visible.
            hf_offline: Set ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``
                inside the container. Defaults from ``$LLENVS_HF_OFFLINE``
                (the per-cluster profile in ``bin/_cluster.sh`` typically
                sets this to 1 when compute nodes are offline). Set
                explicitly to override.
            cuda_visible_devices: Value for ``CUDA_VISIBLE_DEVICES`` inside
                the container (e.g. ``"0,1"``). Lets you pin ``vllm serve``
                to a subset of the job's GPUs so the remaining ones stay
                free for other workloads on the same node. ``None`` inherits
                whatever CUDA_VISIBLE_DEVICES the parent process sees (or
                all GPUs, if unset).
            server_log_path: File to tee server stdout+stderr to. Defaults
                to ``./.llenvs-vllm-serve-<port>-<pid>.log`` in the current
                working directory.
            chat_template_kwargs: Extra kwargs for chat-template rendering
                during continuation scoring (e.g. reasoning toggles). Scoring
                always forces ``enable_thinking=False`` on top of these.

        Raises:
            RuntimeError: if the ``.sif`` can't be located, the subprocess
                dies during startup, or the health check times out.
        """
        self._closed = False
        self._model_path = model_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: IO[bytes] | None = None
        self._openai: OpenAIBackend | None = None
        self._tokenizer: Any = None
        self._chat_template_kwargs = chat_template_kwargs or {}
        self._max_concurrency = max_concurrency

        # --- resolve sif ---
        sif_resolved = _resolve_default(sif, "LLENVS_SIF") or _resolve_default(None, "LLENVS_VLLM_SIF")
        if not sif_resolved:
            raise RuntimeError(
                "SingularityVLLMBackend: no .sif path provided. Pass "
                "sif=... or set $LLENVS_SIF (e.g. by sourcing "
                "llenvs/bin/_cluster.sh). See docs/guides/singularity.md."
            )
        if not Path(sif_resolved).is_file():
            raise RuntimeError(
                f"SingularityVLLMBackend: .sif not found at {sif_resolved}. "
                "Build it with bin/build_container.sh."
            )
        self._sif = sif_resolved

        # --- resolve binds ---
        if singularity_binds:
            binds = list(singularity_binds)
        else:
            env_binds = os.environ.get("LLENVS_BINDS", "").split()
            binds = [b for b in env_binds if b]

        # --- resolve hf_home (and add to binds) ---
        hf_home_resolved = (
            hf_home
            or os.environ.get("HF_HOME")
            or os.environ.get("LLENVS_HF_HOME")
        )
        if hf_home_resolved and hf_home_resolved not in binds:
            binds.append(hf_home_resolved)
        self._hf_home = hf_home_resolved

        # --- resolve offline mode ---
        if hf_offline is None:
            hf_offline = os.environ.get("LLENVS_HF_OFFLINE", "0") in ("1", "true", "True")
        self._hf_offline = bool(hf_offline)

        # --- port + served model name ---
        self._port = port if port is not None else _pick_free_port()
        self._served_model_name = served_model_name or Path(model_path).name

        # --- build argv ---
        bind_flags: list[str] = []
        for b in binds:
            if Path(b.split(":")[0]).exists():
                bind_flags.extend(["--bind", b])

        vllm_serve_args: list[str] = [
            "vllm",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--dtype",
            dtype,
            "--served-model-name",
            self._served_model_name,
        ]
        if max_model_len is not None:
            vllm_serve_args.extend(["--max-model-len", str(max_model_len)])
        vllm_serve_args.extend(extra_vllm_args)

        argv = [
            "singularity",
            "exec",
            "--nv",
            *bind_flags,
            self._sif,
            *vllm_serve_args,
        ]

        # --- build env (SINGULARITYENV_* forwarding) ---
        env = os.environ.copy()
        if hf_home_resolved:
            env["SINGULARITYENV_HF_HOME"] = hf_home_resolved
        if self._hf_offline:
            env["SINGULARITYENV_HF_HUB_OFFLINE"] = "1"
            env["SINGULARITYENV_TRANSFORMERS_OFFLINE"] = "1"
        if cuda_visible_devices is not None:
            # Pin vllm-serve to a subset of the job's GPUs. The parent
            # process keeps its own CUDA_VISIBLE_DEVICES untouched, so other
            # code on the same node can use the remaining devices.
            env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self._cuda_visible_devices = cuda_visible_devices
        for k, v in (extra_singularity_env or {}).items():
            env[f"SINGULARITYENV_{k}"] = v

        # --- log file ---
        if server_log_path is None:
            server_log_path = f".llenvs-vllm-serve-{self._port}.log"
        self._log_path = server_log_path
        self._log_fh = open(server_log_path, "ab", buffering=0)
        self._log_fh.write(
            f"# singularity exec cmd: {shlex.join(argv)}\n".encode()
        )

        _log.info(
            "SingularityVLLMBackend: spawning vllm serve "
            "(model=%s, port=%d, tp=%d, sif=%s, log=%s)",
            model_path,
            self._port,
            tensor_parallel_size,
            self._sif,
            server_log_path,
        )

        # --- spawn ---
        try:
            self._proc = subprocess.Popen(
                argv,
                env=env,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group for killpg
            )
        except FileNotFoundError as e:
            self._close_log_fh()
            raise RuntimeError(
                f"SingularityVLLMBackend: singularity binary not found. "
                f"Is singularity installed on PATH? ({e})"
            ) from e

        atexit.register(self._atexit_close)

        # --- wait for /health ---
        try:
            self._wait_for_health(timeout=startup_timeout, interval=health_poll_interval)
        except Exception:
            self.close()
            raise

        # --- instantiate inner OpenAI client ---
        self._openai = OpenAIBackend(
            model=self._served_model_name,
            api_key="EMPTY",
            base_url=f"http://127.0.0.1:{self._port}/v1",
            max_concurrency=max_concurrency,
        )

        _log.info(
            "SingularityVLLMBackend: ready (model=%s, served_as=%s, pid=%d)",
            model_path,
            self._served_model_name,
            self._proc.pid,
        )

    # ---------- ModelBackend interface ----------

    @property
    def capabilities(self) -> BackendCapabilities:
        base = self._openai.capabilities if self._openai else None
        return BackendCapabilities(
            # vllm serve returns logprobs over its OpenAI-compatible API, and all
            # chat calls proxy to the inner OpenAIBackend that parses them; surface
            # whatever that client reports (False only when closed).
            supports_logprobs=bool(base and base.supports_logprobs),
            supports_prefix_continuation=False,
            supports_batching=True,
            supports_streaming=False,
            supports_chat=True,
            supports_function_calling=False,
            supports_vision=True,
            # Scoring rides the same server via /v1/completions prompt_logprobs
            # (see score_chat_batch); available only while the server is live.
            supports_full_scoring=self._openai is not None,
            max_batch_size=None,
            max_context_length=base.max_context_length if base else None,
            max_concurrency=base.max_concurrency if base else None,
        )

    @property
    def model_name(self) -> str:
        return self._model_path

    def generate(
        self, prompts: list[str], params: SamplingParams
    ) -> list[GenerationResult]:
        if self._openai is None:
            raise RuntimeError("SingularityVLLMBackend is closed")
        return self._openai.generate(prompts, params)

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        if self._openai is None:
            raise RuntimeError("SingularityVLLMBackend is closed")
        return self._openai.generate_chat(
            messages, _with_request_thinking_toggle(params)
        )

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        if self._openai is None:
            raise RuntimeError("SingularityVLLMBackend is closed")
        return self._openai.generate_chat_batch(
            messages_batch, _with_request_thinking_toggle(params)
        )

    def score_chat(
        self, messages: list[ChatMessage], continuation: str
    ) -> ScoringResult:
        return self.score_chat_batch([messages], [continuation])[0]

    def score_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> list[ScoringResult]:
        """Score continuations via the server's ``prompt_logprobs``.

        Renders each prompt with the chat template (thinking disabled),
        appends the continuation as raw text, then requests per-position
        prompt logprobs from ``/v1/completions`` with a token-id prompt —
        the HTTP analog of the in-process ``VLLMBackend`` scoring path.
        """
        if self._openai is None:
            raise RuntimeError("SingularityVLLMBackend is closed")

        # Scored continuations are answer text, never reasoning: a forced pass
        # has nowhere to put a thinking block, so render with thinking off.
        template_kwargs = {**self._chat_template_kwargs, "enable_thinking": False}
        _full_texts, prompt_lengths, full_token_ids, empty_results = build_scoring_inputs(
            self._get_tokenizer(), template_kwargs, messages_batch, continuations
        )
        if not full_token_ids:
            return [empty_results[i] for i in range(len(messages_batch))]

        scored = _run_concurrent(
            lambda item: self._score_one_async(item[0], item[1]),
            list(zip(prompt_lengths, full_token_ids)),
            self._max_concurrency,
        )

        results: list[ScoringResult] = []
        scored_iter = iter(scored)
        for index in range(len(messages_batch)):
            if index in empty_results:
                results.append(empty_results[index])
            else:
                results.append(next(scored_iter))
        return results

    async def _score_one_async(
        self, prompt_len: int, full_ids: list[int]
    ) -> ScoringResult:
        # Ride the inner OpenAI client's async connection; vLLM's completions
        # endpoint returns prompt logprobs via the extra_body passthrough.
        response = await self._openai._async_client.completions.create(
            model=self._served_model_name,
            prompt=full_ids,
            max_tokens=1,
            temperature=0,
            extra_body={"prompt_logprobs": 0},
        )
        choice = response.choices[0]
        return parse_prompt_logprobs_http(
            prompt_len=prompt_len,
            full_ids=full_ids,
            prompt_logprobs=getattr(choice, "prompt_logprobs", None),
            model_name=self._model_path,
        )

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = _load_hf_tokenizer(self._model_path)
        return self._tokenizer

    # ---------- lifecycle ----------

    def _wait_for_health(self, timeout: float, interval: float) -> None:
        """Poll GET /health until 200, or the subprocess dies, or timeout."""
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                rc = self._proc.returncode if self._proc else None
                raise RuntimeError(
                    f"vllm serve exited during startup (rc={rc}); "
                    f"see {self._log_path} for details"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_error = e
            time.sleep(interval)

        raise RuntimeError(
            f"vllm serve health check timed out after {timeout}s "
            f"(last error: {last_error}); see {self._log_path} for details"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._openai is not None:
            with contextlib.suppress(Exception):
                self._openai.close()
            self._openai = None

        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as e:
                _log.warning("SIGTERM to vllm serve pgroup failed: %s", e)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _log.warning("vllm serve did not exit within 30s, escalating to SIGKILL")
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
        self._proc = None

        self._close_log_fh()

    def _close_log_fh(self) -> None:
        fh = self._log_fh
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()
        self._log_fh = None

    def _atexit_close(self) -> None:
        """atexit safety net; must never raise."""
        with contextlib.suppress(Exception):
            self.close()

    def __del__(self) -> None:
        # __del__ must not raise; close() is already crash-safe.
        with contextlib.suppress(Exception):
            self.close()

    # ---------- dev / debug helpers ----------

    @property
    def port(self) -> int:
        return self._port

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def server_log_path(self) -> str:
        return self._log_path
