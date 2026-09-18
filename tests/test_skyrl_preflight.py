"""Initial execution gates on native-shaped fixtures, without GPU imports."""

import copy
import importlib
from types import SimpleNamespace as Namespace

import pytest

from llenvs.integrations.skyrl._config import LlenvsConfig


def set_field(root, path, value):
    *parents, name = path.split(".")
    for parent in parents:
        if not hasattr(root, parent):
            setattr(root, parent, Namespace())
        root = getattr(root, parent)
    setattr(root, name, value)


@pytest.fixture
def config():
    cfg = Namespace(llenvs=LlenvsConfig())
    fields = {
        "trainer.strategy": "fsdp",
        "trainer.bf16": True,
        "trainer.flash_attn": True,
        "trainer.gradient_checkpointing_use_reentrant": False,
        "trainer.recompute_old_logprobs_per_minibatch": True,
        "trainer.fused_lm_head_logprob": False,
        "trainer.gradient_checkpointing": True,
        "trainer.remove_microbatch_padding": True,
        "trainer.update_ref_every_epoch": False,
        "trainer.max_prompt_length": 512,
        "trainer.mtp.enabled": False,
        "trainer.critic.model.path": None,
        "trainer.max_tokens_per_microbatch": -1,
        "trainer.train_batch_size": 4,
        "trainer.policy_mini_batch_size": 4,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.micro_forward_batch_size_per_gpu": 1,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.ref_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.ref_num_gpus_per_node": 1,
        "trainer.placement.colocate_all": True,
        "trainer.placement.colocate_policy_ref": True,
        "trainer.fully_async.enabled": False,
        "trainer.fully_async.simulate_training": False,
        "trainer.fully_async.sample_full_batch": False,
        "trainer.fully_async.clear_kv_cache_on_weight_sync": False,
        "trainer.fully_async.num_parallel_generation_workers": 4,
        "trainer.fully_async.max_staleness_steps": 1,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.policy_loss_type": "regular",
        "trainer.algorithm.use_entropy_loss": False,
        "trainer.algorithm.use_kl_in_reward": False,
        "trainer.algorithm.use_kl_loss": True,
        "trainer.algorithm.eps_clip_low": 0.2,
        "trainer.algorithm.eps_clip_high": 0.2,
        "trainer.algorithm.temperature": 1.0,
        "trainer.algorithm.gamma": 1.0,
        "trainer.algorithm.grpo_norm_by_std": True,
        "trainer.algorithm.advantage_batch_normalize": False,
        "trainer.algorithm.zero_variance_filter": False,
        "trainer.algorithm.dynamic_sampling.type": None,
        "trainer.algorithm.max_seq_len": None,
        "trainer.algorithm.loss_reduction": "token_mean",
        "trainer.algorithm.use_tis": False,
        "trainer.algorithm.off_policy_correction.tis_ratio_type": None,
        "trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high": 2.0,
        "trainer.algorithm.off_policy_correction.sequence_mask_metric": None,
        "trainer.algorithm.off_policy_correction.outlier_token_is_threshold_low": None,
        "trainer.algorithm.off_policy_correction.outlier_token_is_threshold_high": None,
        "trainer.algorithm.off_policy_correction.token_mask_is_threshold_low": None,
        "trainer.algorithm.off_policy_correction.token_mask_is_threshold_high": None,
        "generator.batched": False,
        "generator.use_cache_salt": True,
        "generator.append_eos_token_after_stop_str_in_multi_turn": True,
        "generator.max_turns": 1,
        "generator.max_input_length": 512,
        "generator.eval_n_samples_per_prompt": 1,
        "generator.sampling_params.max_generate_length": 1024,
        "generator.eval_sampling_params.max_generate_length": 1024,
        "generator.step_wise_trajectories": False,
        "generator.merge_stepwise_output": False,
        "generator.vision_language_generator": False,
        "generator.use_conversation_multi_turn": True,
        "generator.chat_template.name_or_path": None,
        "generator.chat_template_kwargs": {},
        "generator.n_samples_per_prompt": 2,
        "generator.zero_reward_on_non_stop": False,
        "generator.apply_overlong_filtering": False,
        "generator.sampling_params.repetition_penalty": 1.0,
        "generator.eval_sampling_params.repetition_penalty": 1.0,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.model_dtype": "bfloat16",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.weight_sync_backend": "nccl",
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": 1,
        "generator.inference_engine.pipeline_parallel_size": 1,
        "generator.inference_engine.data_parallel_size": 1,
        "generator.inference_engine.expert_parallel_size": 1,
        "generator.inference_engine.enable_return_routed_experts": False,
        "generator.inference_engine.enable_pd": False,
        "generator.inference_engine.speculative_config": None,
        "generator.inference_engine.fp8_weight_sync_mode": None,
        "generator.inference_engine.delta_weight_sync": None,
        "generator.inference_engine.offload_kv_for_weight_sync": False,
        "generator.inference_engine.external_proxy_url": None,
        "generator.inference_engine.external_server_urls": None,
        "generator.inference_engine.prefill_init_kwargs": {},
        "generator.inference_engine.decode_init_kwargs": {},
        "generator.inference_engine.router_init_kwargs": {},
        "generator.inference_engine.engine_init_kwargs": {},
        "generator.inference_engine.language_model_only": False,
        "generator.inference_engine.enable_prefix_caching": True,
        "generator.inference_engine.enforce_eager": False,
        "generator.inference_engine.enable_chunked_prefill": True,
        "environment.env_class": "llenvs",
        "environment.skyrl_gym.max_env_workers": 32,
    }
    for role in ("policy", "ref"):
        for name, value in {
            "model.path": "/models/qwen",
            "sequence_parallel_size": 1,
            "model.lora.rank": 0,
            "model.fake_int4_qat.enabled": False,
            "model_config_kwargs": {},
            "language_model_only": False,
            "fsdp_config.fsdp_size": -1,
            "fsdp_config.mixed_precision": None,
        }.items():
            fields[f"trainer.{role}.{name}"] = value
    fields["trainer.policy.use_torch_compile"] = False
    fields["trainer.policy.inference_only_init"] = False
    for name, value in fields.items():
        set_field(cfg, name, value)
    return cfg


@pytest.fixture
def preflight():
    return importlib.import_module("llenvs.integrations.skyrl._preflight")


def test_native_profile_is_not_mutated(preflight, config):
    before = copy.deepcopy(config)
    preflight.validate_profile(config)
    assert config == before


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("trainer.strategy", "megatron"),
        ("trainer.algorithm.use_entropy_loss", "false"),
        ("generator.max_turns", 0),
        ("generator.use_cache_salt", "false"),
        ("trainer.bf16", False),
        ("trainer.policy.sequence_parallel_size", 2),
        ("trainer.policy.sequence_parallel_size", True),
        ("trainer.policy.use_torch_compile", True),
        ("trainer.policy.model.lora.rank", 16),
        ("trainer.ref.model.path", "/models/different-tokenizer"),
        ("trainer.policy.model_config_kwargs", {"vocab_size": 1}),
        ("trainer.policy.model.fake_int4_qat.enabled", True),
        ("trainer.critic.model.path", "/critic"),
        ("trainer.placement.policy_num_nodes", 2),
        ("trainer.mtp.enabled", True),
        ("trainer.gradient_checkpointing_use_reentrant", True),
        ("trainer.recompute_old_logprobs_per_minibatch", False),
        ("trainer.micro_forward_batch_size_per_gpu", 2),
        ("trainer.algorithm.eps_clip_low", 0.1),
        ("trainer.algorithm.policy_loss_type", "rollout_is"),
        ("generator.vision_language_generator", True),
        ("generator.batched", True),
        ("generator.step_wise_trajectories", True),
        ("generator.use_conversation_multi_turn", False),
        ("generator.chat_template.name_or_path", "replacement"),
        ("generator.chat_template_kwargs", {"continue_final_message": True}),
        ("generator.inference_engine.run_engines_locally", False),
        ("generator.inference_engine.tensor_parallel_size", 2),
        ("generator.inference_engine.weight_sync_backend", "delta"),
        ("generator.inference_engine.speculative_config", {"method": "mtp"}),
        ("generator.inference_engine.enable_return_routed_experts", True),
        ("generator.inference_engine.engine_init_kwargs", {"hf_overrides": {"vocab_size": 1}}),
        ("generator.inference_engine.engine_init_kwargs", {"logits_processors": ["custom"]}),
        ("generator.inference_engine.engine_init_kwargs", {"model": "/different"}),
        ("generator.inference_engine.engine_init_kwargs", {"unknown_flag": True}),
        ("generator.inference_engine.enable_chunked_prefill", False),
        ("generator.sampling_params.repetition_penalty", 1.2),
        ("environment.skyrl_gym.max_env_workers", 0),
    ],
)
def test_unsupported_or_inert_requested_features_fail(preflight, config, path, value):
    set_field(config, path, value)
    with pytest.raises(ValueError):
        preflight.validate_profile(config)


def test_custom_credit_profile_and_entropy_microbatch_boundary(preflight, config):
    config.trainer.algorithm.advantage_estimator = "llenvs_turn_grpo"
    config.llenvs.sampling_contract = "unmodified"
    preflight.validate_profile(config)
    config.trainer.algorithm.use_entropy_loss = True
    preflight.validate_profile(config)
    config.trainer.max_tokens_per_microbatch = 1024
    with pytest.raises(ValueError, match="entropy"):
        preflight.validate_profile(config)
    config.trainer.algorithm.use_entropy_loss = False
    preflight.validate_profile(config)


def test_real_dp_rows_must_be_divisible_even_when_native_floor_division_passes(preflight, config):
    config.trainer.placement.policy_num_gpus_per_node = 2
    config.trainer.placement.ref_num_gpus_per_node = 2
    config.generator.inference_engine.num_engines = 2
    config.trainer.train_batch_size = config.trainer.policy_mini_batch_size = 3
    config.generator.n_samples_per_prompt = 1
    with pytest.raises(ValueError, match="divisible"):
        preflight.validate_profile(config)
    config.generator.n_samples_per_prompt = 2
    preflight.validate_profile(config)


def test_fully_async_requires_bounded_workers_and_single_correction(preflight, config):
    config.trainer.fully_async.enabled = True
    config.trainer.placement.colocate_all = False
    config.trainer.algorithm.policy_loss_type = "rollout_is"
    config.llenvs.sampling_contract = "unmodified"
    preflight.validate_profile(config)
    config.trainer.fully_async.num_parallel_generation_workers = 768
    with pytest.raises(ValueError, match="workers"):
        preflight.validate_profile(config)


def test_sampling_helper_override_is_train_only_and_preserves_native_inputs(preflight):
    native = {"temperature": 1.0, "max_tokens": 32, "min_tokens": 1, "logprobs": 1}
    assert preflight.training_request(native, contract="unmodified")["min_tokens"] == 0
    assert preflight.training_request(native, contract="native") == native
    assert native["min_tokens"] == 1


def test_resolved_engine_validates_probability_and_identity_fields(preflight, config):
    resolved = dict(
        model="/models/qwen",
        tokenizer=None,
        dtype="bfloat16",
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_lora=False,
        speculative_config=None,
        quantization=None,
        kv_cache_dtype="auto",
        hf_overrides={},
        override_generation_config={},
        logits_processors=None,
        max_model_len=None,
        enable_chunked_prefill=None,
        enable_prefix_caching=True,
    )
    preflight.validate_engine_args(config, Namespace(**resolved))
    for name, value in {
        "generation_config": "auto",
        "tokenizer": "/other",
        "logprobs_mode": "raw_logits",
        "quantization": "fp8",
        "tensor_parallel_size": 2,
    }.items():
        with pytest.raises(ValueError, match=name):
            preflight.validate_engine_args(config, Namespace(**(resolved | {name: value})))


def test_runtime_control_identity_accounts_for_native_injected_defaults(preflight):
    head = preflight.runtime_controls({}, driver=False)
    worker = preflight.runtime_controls(
        {
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "SKYRL_LOG_FILE": "/native/generated.log",
        },
        driver=True,
    )
    assert head == worker


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VLLM_PLUGINS", "custom"),
        ("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"),
        ("SKYRL_UNKNOWN_OPTIMIZATION", "1"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1"),
        ("NCCL_ALGO", "Tree"),
        ("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0"),
        ("VLLM_USE_V1", "1"),
    ],
)
def test_unreviewed_ambient_execution_overrides_fail(preflight, name, value):
    with pytest.raises(ValueError, match=name):
        preflight.runtime_controls({name: value}, driver=False)
