"""Provider-specific usage accounting and stable, credential-free identities."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypedDict

USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "cache_write_prompt_tokens",
    "reasoning_tokens",
)


class UsageCounts(TypedDict):
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_prompt_tokens: int | None
    cache_write_prompt_tokens: int | None
    reasoning_tokens: int | None


def sum_known(values: Iterable[int | None]) -> int | None:
    """Sum disjoint counts, propagating missing observations."""
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def credential_free(value: Any) -> Any:
    """Remove credentials from descriptors without discarding model input fields."""
    private = {
        "api_key",
        "authorization",
        "proxy-authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "hf_token",
        "cookie",
        "x-api-key",
    }
    if isinstance(value, Mapping):
        return {
            key: credential_free(item)
            for key, item in value.items()
            if str(key).lower() not in private
        }
    if isinstance(value, (list, tuple)):
        return [credential_free(item) for item in value]
    if isinstance(value, str) and "://" in value:
        from llenvs.inference.sampling import _record_value

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return _record_value({"url": value})["url"]
        return re.sub(
            r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"\]\)<>]+",
            lambda match: _record_value({"url": match.group()})["url"],
            value,
        )
    return value


def plain(value: Any) -> Any:
    """Detach provider data without retaining SDK response objects."""
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return plain({key: item for key, item in vars(value).items() if not key.startswith("_")})
    return value


def normalize_usage(usage: Any, *, provider: str = "openai") -> UsageCounts:
    """Normalize total input/output; cache and reasoning counts are subsets.

    Anthropic's ordinary input excludes cache categories. An absent category
    remains unknown; callers must not turn an omitted field into a cache miss.
    """
    data = plain(usage) or {}
    if provider == "codex_cli":
        data = dict(data)
        if "cached_input_tokens" in data:
            details = dict(data.get("input_tokens_details") or {})
            details["cached_tokens"] = data["cached_input_tokens"]
            data["input_tokens_details"] = details
        provider = "openai"

    def count(mapping, key):
        value = mapping.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Invalid usage count {key}: {value!r}")
        return value

    if provider == "anthropic":
        read = count(data, "cache_read_input_tokens")
        write = count(data, "cache_creation_input_tokens")
        return dict(
            prompt_tokens=sum_known([count(data, "input_tokens"), read, write]),
            completion_tokens=count(data, "output_tokens"),
            cached_prompt_tokens=read,
            cache_write_prompt_tokens=write,
            reasoning_tokens=None,
        )
    if provider != "openai":
        raise ValueError(f"No usage accounting mapping for {provider!r}")
    input_key = "prompt_tokens" if "prompt_tokens" in data else "input_tokens"
    output_key = "completion_tokens" if "completion_tokens" in data else "output_tokens"
    inputs = data.get(input_key + "_details") or {}
    outputs = data.get(output_key + "_details") or {}
    return dict(
        prompt_tokens=count(data, input_key),
        completion_tokens=count(data, output_key),
        cached_prompt_tokens=count(inputs, "cached_tokens"),
        cache_write_prompt_tokens=count(inputs, "cache_write_tokens"),
        reasoning_tokens=count(outputs, "reasoning_tokens"),
    )


def dependency_versions(*packages: str) -> dict[str, str | None]:
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def content_manifest(root: str | Path) -> dict[str, str]:
    """Identify a local artifact by content, independent of its installation path."""
    root = Path(root)
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def model_artifact(model: str, revision: str | None) -> dict[str, Any]:
    """Require immutable weight identity before a local backend can be replayed."""
    local = Path(model).is_dir()
    if not local and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("A local backend descriptor requires a resolved immutable model revision")
    return {
        "model": "local_content" if local else model,
        "revision": None if local else revision,
        "version_pinned": True,
        "model_manifest": content_manifest(model) if local else None,
    }


def resolve_model_revision(model: str, revision: str | None = None, *, offline=False) -> str | None:
    """Resolve a mutable Hub revision once, or identify a local directory separately."""
    if Path(model).is_dir():
        if revision is not None:
            raise ValueError("Local model directories use a content manifest, not a Hub revision")
        return None
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if offline:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model, "config.json", revision=revision, local_files_only=True))
        for parent in path.parents:
            if re.fullmatch(r"[0-9a-f]{40}", parent.name):
                return parent.name
        raise ValueError("Offline model configuration has no immutable snapshot revision")
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(model, revision=revision).sha
    if resolved is None or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("Hub did not resolve an immutable model revision")
    return resolved


def tokenizer_identity(tokenizer) -> dict[str, Any]:
    native = getattr(tokenizer, "backend_tokenizer", None)
    serialized = (
        native.to_str() if native is not None else json.dumps(tokenizer.get_vocab(), sort_keys=True)
    )
    template = getattr(tokenizer, "chat_template", None)
    return {
        "tokenizer_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "template_sha256": hashlib.sha256(
            json.dumps(template, sort_keys=True).encode()
        ).hexdigest(),
    }


def processor_identity(processor) -> dict[str, Any]:
    """Hash the loaded processor and its modality-specific configurations."""
    if processor is None:
        return {"processor_sha256": None}
    config = {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "config": plain(processor.to_dict()),
        "chat_template": getattr(processor, "chat_template", None),
    }
    for name in ("image_processor", "video_processor", "feature_extractor"):
        component = getattr(processor, name, None)
        if component is not None:
            config[name] = plain(component.to_dict())
    return {
        "processor_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    }


def hardware_identity(torch, *, device_indices=None, device_type=None) -> dict[str, Any]:
    """Describe numerical hardware without device ordinals or host names."""
    import platform

    result = {"machine": platform.machine(), "accelerator": "cpu", "devices": []}
    if device_type in {None, "cuda"} and torch.cuda.is_available():
        indices = range(torch.cuda.device_count()) if device_indices is None else device_indices
        result.update(
            accelerator="rocm" if torch.version.hip else "cuda",
            cuda_runtime=torch.version.cuda,
            hip_runtime=torch.version.hip,
            devices=[
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in indices
            ],
        )
    elif device_type in {None, "mps"} and torch.backends.mps.is_available():
        result.update(
            accelerator="mps", devices=[{"name": platform.processor() or platform.machine()}]
        )
    return result


def distribution_identity(package: str) -> dict[str, Any]:
    """Hash installed runtime content, independently of its installation path."""
    from importlib.metadata import distribution

    dist = distribution(package)
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    if (source.get("dir_info") or {}).get("editable"):
        raise ValueError("Runtime identity requires a non-editable wheel or source build")
    manifest = {}
    for entry in dist.files or ():
        if ".." in entry.parts or entry.name in {
            "RECORD",
            "INSTALLER",
            "REQUESTED",
            "direct_url.json",
        }:
            continue
        path = Path(str(dist.locate_file(entry)))
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            manifest[str(entry)] = file_digest(path)
    return {
        "version": dist.version,
        "content_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }


def native_runtime_identity(
    executable: str, package: str, *, launcher_prefix=(), env=None
) -> dict[str, Any]:
    """Inspect the executable's Python environment without importing model code."""
    import shlex
    import shutil
    import subprocess

    script = """import hashlib, importlib.metadata as m, json, pathlib, sys
package = sys.argv[1]
d = m.distribution(package)
source = json.loads(d.read_text("direct_url.json") or "{}")
if (source.get("dir_info") or {}).get("editable"):
    raise ValueError("Runtime identity requires a non-editable wheel or source build")
if source.get("url", "").startswith("file:"):
    source["url"] = "local_build"
files = {}
for entry in d.files or ():
    if ".." in entry.parts or entry.name in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}:
        continue
    p = pathlib.Path(d.locate_file(entry))
    if p.is_file() and p.suffix not in {".pyc", ".pyo"}:
        with p.open("rb") as f:
            files[str(entry)] = hashlib.file_digest(f, "sha256").hexdigest()
import platform
hardware = {"machine": platform.machine(), "accelerator": "cpu", "devices": []}
try:
    import torch
    if torch.cuda.is_available():
        hardware["accelerator"] = "rocm" if torch.version.hip else "cuda"
        hardware["cuda_runtime"] = torch.version.cuda
        hardware["hip_runtime"] = torch.version.hip
        hardware["devices"] = [{"name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
except ImportError:
    pass
print(json.dumps({"hardware": hardware, "package": package, "version": d.version,
    "content_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    "source": source or None,
    "dependencies": {x.metadata["Name"]: x.version for x in m.distributions()}}))
"""
    if launcher_prefix:
        bootstrap = """import os, pathlib, shlex, shutil, sys
path = shutil.which(sys.argv[1])
if path is None:
    raise FileNotFoundError(sys.argv[1])
first = pathlib.Path(path).open("rb").readline(4096)
if not first.startswith(b"#!"):
    raise ValueError("Runtime entry point has no Python interpreter")
argv = shlex.split(first[2:].decode().strip())
os.execvp(argv[0], [*argv, "-c", sys.argv[2], sys.argv[3]])
"""
        command = [*launcher_prefix, "python3", "-c", bootstrap, executable, script, package]
    else:
        path = shutil.which(executable)
        if path is None:
            raise FileNotFoundError(executable)
        with Path(path).open("rb") as handle:
            first_line = handle.readline(4096)
        if not first_line.startswith(b"#!"):
            raise ValueError("Native runtime entry point must identify its Python interpreter")
        interpreter = shlex.split(first_line[2:].decode().strip())
        command = [*interpreter, "-c", script, package]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    from llenvs.inference.sampling import _record_value

    return _record_value(json.loads(completed.stdout))


def sanitized_argv(argv) -> list[str]:
    """Retain launch semantics while removing credential arguments and URL secrets."""
    secret_flags = {
        "--api-key",
        "--token",
        "--password",
        "--authorization",
        "--hf-token",
        "--extra-headers",
        "--headers",
    }
    result, sensitive = [], False
    for value in map(str, argv):
        flag, separator, _ = value.partition("=")
        if flag in secret_flags:
            result.append(flag + "=<redacted>" if separator else flag)
            sensitive = not separator
        elif value.startswith("--"):
            sensitive = False
            result.append(credential_free(value))
        elif sensitive:
            result.append("<redacted>")
        else:
            result.append(credential_free(value))
    return result


def vllm_engine_config(log: str) -> dict[str, Any]:
    """Read the last pinned engine configuration without executing its repr."""
    import ast

    lines = [
        line.split("with config: ", 1)[1]
        for line in log.splitlines()
        if "Initializing a V1 LLM engine" in line and "with config: " in line
    ]
    if not lines:
        raise ValueError("The vLLM server did not report its effective engine configuration")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", lines[-1])
    entries, start, depth, quote, escaped = [], 0, 0, None, False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            entries.append(text[start:index].strip())
            start = index + 1
    entries.append(text[start:].strip())
    result = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or key in {"model", "tokenizer", "download_dir", "served_model_name"}:
            continue
        value = re.sub(
            r"\b(cache_dir|debug_dump_path|cudagraph_debug_dump_path)=('[^']*'|\"[^\"]*\")",
            r"\1='<runtime-path>'",
            value,
        )
        try:
            result[key] = credential_free(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            result[key] = credential_free(value)
    if type(result.get("enable_prefix_caching")) is not bool:
        raise ValueError("The vLLM server did not report effective prefix caching")
    return result
