"""Provider usage is nullable and follows each provider's input accounting."""

import pytest

from llenvs.inference.protocol import GenerationResult
from llenvs.inference.records import normalize_usage, sum_known


def test_missing_usage_is_unknown_including_total():
    result = GenerationResult()
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
    assert normalize_usage(None) == dict.fromkeys(
        (
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "cache_write_prompt_tokens",
            "reasoning_tokens",
        )
    )


@pytest.mark.parametrize(
    "input_name,output_name",
    [("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens")],
)
def test_openai_cache_and_reasoning_are_subsets(input_name, output_name):
    usage = {
        input_name: 100,
        output_name: 30,
        input_name + "_details": {"cached_tokens": 70, "cache_write_tokens": 10},
        output_name + "_details": {"reasoning_tokens": 20},
    }
    normalized = normalize_usage(usage)
    assert normalized == dict(
        prompt_tokens=100,
        completion_tokens=30,
        cached_prompt_tokens=70,
        cache_write_prompt_tokens=10,
        reasoning_tokens=20,
    )
    assert GenerationResult(**normalized).total_tokens == 130


def test_anthropic_total_includes_disjoint_cache_categories():
    normalized = normalize_usage(
        {
            "input_tokens": 20,
            "output_tokens": 3,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 10,
        },
        provider="anthropic",
    )
    assert normalized["prompt_tokens"] == 100
    assert normalized["completion_tokens"] == 3
    assert normalized["cached_prompt_tokens"] == 70
    assert normalized["cache_write_prompt_tokens"] == 10
    assert normalized["reasoning_tokens"] is None


def test_partial_anthropic_categories_do_not_claim_a_known_total():
    assert normalize_usage({"input_tokens": 20}, provider="anthropic")["prompt_tokens"] is None


def test_reported_zero_is_distinct_from_absent():
    assert normalize_usage({"prompt_tokens": 0, "completion_tokens": 0})["prompt_tokens"] == 0
    assert sum_known([0, 3]) == 3
    assert sum_known([0, None]) is None
    assert sum_known([]) == 0


@pytest.mark.parametrize("value", [-1, True, 1.5, float("nan")])
def test_invalid_provider_counts_raise(value):
    with pytest.raises(ValueError):
        normalize_usage({"prompt_tokens": value})


def test_descriptor_redaction_preserves_token_ids_and_noncredential_input_fields():
    from llenvs.inference.records import credential_free, sanitized_argv

    descriptor = {
        "input": "dataset-id",
        "token_ids": [1, 2],
        "api_key": "secret",
        "url": "https://user:pass@example.com/v1?api_key=secret&mode=chat",
    }
    assert credential_free(descriptor) == {
        "input": "dataset-id",
        "token_ids": [1, 2],
        "url": "https://example.com/v1?mode=chat",
    }
    assert sanitized_argv(["vllm", "--api-key", "a", "b", "--revision", "abc"]) == [
        "vllm",
        "--api-key",
        "<redacted>",
        "<redacted>",
        "--revision",
        "abc",
    ]


def test_effective_vllm_config_comes_from_the_final_engine_record():
    from llenvs.inference.records import vllm_engine_config

    text = "Initializing a V1 LLM engine (v0.28.0) with config: model='old', dtype=torch.float16, enable_prefix_caching=True\n"
    text += "Initializing a V1 LLM engine (v0.28.0) with config: model='/new/path', revision=abcdef0123456789abcdef0123456789abcdef0123, tokenizer_revision=abcdef0123456789abcdef0123456789abcdef0123, dtype=torch.bfloat16, seed=42, tensor_parallel_size=2, enable_prefix_caching=False, compilation_config=CompilationConfig(cudagraph_capture_sizes=[1, 2, 4], cache_dir='/tmp/one')\n"
    result = vllm_engine_config(text)
    assert result["enable_prefix_caching"] is False
    assert result["dtype"] == "torch.bfloat16" and result["tensor_parallel_size"] == 2
    assert "model" not in result
    assert "[1, 2, 4]" in result["compilation_config"]
    assert "/tmp/one" not in result["compilation_config"]
    with pytest.raises(ValueError):
        vllm_engine_config("No engine configuration was reported")


def test_offline_revision_uses_snapshot_path_before_following_blob_symlink(monkeypatch, tmp_path):
    import huggingface_hub

    from llenvs.inference.records import resolve_model_revision

    revision = "a" * 40
    blob = tmp_path / "blobs" / ("b" * 40)
    blob.parent.mkdir()
    blob.write_text("{}")
    snapshot = tmp_path / "snapshots" / revision / "config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(snapshot))
    assert resolve_model_revision("org/model", offline=True) == revision


def test_processor_digest_changes_with_preprocessing_configuration():
    from llenvs.inference.records import processor_identity

    class Processor:
        chat_template = "template"

        def __init__(self, size):
            self.size = size

        def to_dict(self):
            return {"image_size": self.size}

    assert processor_identity(None) == {"processor_sha256": None}
    assert processor_identity(Processor(224)) != processor_identity(Processor(448))


def test_runtime_descriptor_rejects_editable_loader_metadata_as_code_identity(monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace

    from llenvs.inference.records import distribution_identity

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda package: SimpleNamespace(
            version="1", files=[], read_text=lambda name: '{"dir_info":{"editable":true}}'
        ),
    )
    with pytest.raises(ValueError, match="editable"):
        distribution_identity("runtime")


def test_hardware_identity_records_selected_classes_without_ordinals():
    from types import SimpleNamespace

    from llenvs.inference.records import hardware_identity

    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None, cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
            get_device_name=lambda index: "H100",
            get_device_properties=lambda index: SimpleNamespace(total_memory=80),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    result = hardware_identity(torch, device_indices=[1, 3])
    assert result["accelerator"] == "cuda"
    assert result["devices"] == [{"name": "H100", "memory_bytes": 80}] * 2
