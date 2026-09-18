"""CPU-importable acceptance process/checkpoint checks; no native runtime imports."""

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from llenvs.integrations.skyrl.data import _unique_object


def training_process(artifact, mode, estimator, dp, *, resume):
    """A fresh process and owned process group; never attach to another Ray job."""
    command = [
        sys.executable,
        "-m",
        "tests.skyrl_acceptance_run",
        mode,
        estimator,
        str(artifact),
        "--dp",
        str(dp),
    ] + (["--resume"] if resume else [])
    phase = "resumed" if resume else "fresh"
    log = artifact / f"{phase}.log"
    _owned_process(command, log)
    return _completed_report(artifact / phase / "logs/llenvs-acceptance.json", log)


def native_probe_process(artifact, scenario, *, estimator="llenvs_token_rtg", packed=False):
    assert scenario in ("dp1", "dp2", "cache_keep", "cache_clear")
    command = [
        sys.executable,
        "-m",
        "tests.skyrl_native_probe_run",
        scenario,
        str(artifact),
        "--estimator",
        estimator,
    ] + (["--packed"] if packed else [])
    log = artifact / f"{scenario}.log"
    _owned_process(command, log)
    return _completed_report(artifact / f"{scenario}.json", log)


def _owned_process(command, log):
    timeout = int(os.environ.get("LLENVS_SKYRL_TIMEOUT_SECONDS", "1800"))
    assert 1 <= timeout <= 7200, "acceptance timeout must be 1..7200 seconds per subprocess"
    with log.open("x") as stream:
        with subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env={
                **os.environ,
                "RAY_ADDRESS": "local",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only this owned session. Detached Ray actors still require
                # the allocated job's teardown/lease, never a broad ray stop.
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Child exited between poll and kill.
                process.wait()
                pytest.fail(f"native child timed out; inspect {log} and the allocated job")
    assert code == 0, f"native child failed (exit {code}); inspect {log}"


def _completed_report(path, log):
    assert path.is_file(), f"driver exited without its audit; inspect {log}"

    def invalid_constant(value):
        raise ValueError(f"non-finite audit value: {value}")

    try:
        result = json.loads(
            path.read_text(), object_pairs_hook=_unique_object, parse_constant=invalid_constant
        )
        assert isinstance(result, dict) and result.get("completed") is True
        assert result.get("failure") is None and result.get("pending_batch") is None
    except (ValueError, AssertionError) as error:
        raise AssertionError(f"invalid/incomplete native audit {path}; inspect {log}") from error
    return result


def _checkpoint_state(root, step, asynchronous):
    import torch

    checkpoint = root / f"global_step_{step}"
    assert (checkpoint / "policy").is_dir(), str(checkpoint)
    assert any(p.is_file() and p.stat().st_size for p in (checkpoint / "policy").rglob("*")), str(
        checkpoint
    )
    for name in ("data.pt", "trainer_state.pt") + (
        ("fully_async_state.pt",) if asynchronous else ()
    ):
        path = checkpoint / name
        assert path.is_file(), str(path)
        try:
            # Only locally generated, owned acceptance artifacts are loaded.
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise AssertionError(f"unreadable checkpoint evidence: {path}") from error
        assert isinstance(state, dict), str(path)
        if name == "trainer_state.pt":
            assert state["global_step"] == step, str(path)
        elif name == "fully_async_state.pt":
            assert not state["filtered_uids"], str(path)
            return state
    return None


def check_smoke_progress(artifact, phase, result, expected_steps, *, asynchronous):
    assert [b["global_step"] for b in result["batches"]] == expected_steps, str(artifact)
    assert result["evaluation_steps"] == expected_steps, str(artifact)
    root = artifact / phase / "checkpoints"
    for batch in result["batches"]:
        assert len(batch["uids"]) == 4 and len(set(batch["uids"])) == 2
        assert batch["policy_metrics"], "a converted batch is not evidence of an optimizer step"
        assert all(math.isfinite(v) for v in batch["policy_metrics"].values())
        assert any(
            d["llenvs/end_reason"] == "terminated" and len(d["llenvs/reward_components"]) == 2
            for d in batch["diagnostics"]
        ), str(artifact)
        state = _checkpoint_state(root, batch["global_step"], asynchronous)
        if asynchronous:
            assert set(batch["uids"]) <= set(state["consumed_uids"]), str(root)
    latest = int((root / "latest_ckpt_global_step.txt").read_text())
    # Native async final counter means NEXT step, not completed updates. These
    # are content checks, not certification of that checkpoint's resume safety.
    assert latest == result["final_counter"], str(root)
    _checkpoint_state(root, latest, asynchronous)
