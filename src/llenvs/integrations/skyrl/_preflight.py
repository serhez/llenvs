"""Bounded FSDP/vLLM text recipe checks, before native resource allocation.

Passing these checks is not installed-runtime or numerical certification.
Native validation runs too; this module never rewrites native configuration.
"""

import copy
from collections.abc import Mapping
from typing import Any

from llenvs.integrations.skyrl._checks import finite_number, integer
from llenvs.integrations.skyrl._validation import validate_credit_recipe

# Only individually reviewed engine overrides. Model/tokenizer/processor,
# speculation, quantization, and arbitrary server/sampler plugins are excluded.
_ENGINE_OVERRIDES = {
    "max_model_len",
    "enable_chunked_prefill",
    "enforce_eager",
    "enable_prefix_caching",
    "logprobs_mode",
    "generation_config",
    "trust_remote_code",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
}


def runtime_controls(environ: Mapping[str, str], *, driver: bool) -> dict[str, str]:
    """Bound ambient execution knobs too, without logging credential values.

    The initial launch uses SkyRL's absent-VLLM_USE_V1 branch, which injects
    both V1=1 and multiprocessing=0. Do not assume explicit shell overrides
    are forwarded by the native helper (it does not forward these two).
    """
    generated = {
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "NCCL_CUMEM_ENABLE": "0",
    }
    for name, expected in generated.items():
        if (
            not driver
            and name in {"VLLM_USE_V1", "VLLM_ENABLE_V1_MULTIPROCESSING"}
            and name in environ
        ):
            raise ValueError(f"unset {name}; the initial recipe uses native runtime injection")
        if name in environ and environ[name] != expected:
            raise ValueError(f"{name} differs from the native execution recipe")
    ignored = {
        "SKYRL_LOG_FILE",
        "SKYRL_DUMP_INFRA_LOG_TO_STDOUT",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_DEBUG",
        "TORCH_SHOW_CPP_STACKTRACES",
        "TORCH_USE_CUDA_DSA",
    }
    admitted = {
        "SKYRL_RAY_PG_TIMEOUT_IN_S",
        "SKYRL_WORKER_NCCL_TIMEOUT_IN_S",
        "SKYRL_VLLM_DP_PORT_OFFSET",
        "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S",
        "SKYRL_HTTP_CONNECTION_LIMIT",
        "SKYRL_GENERATE_CONCURRENCY_PER_ENGINE",
        "SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC",
        "SKYRL_DISABLE_FA4",
        "VLLM_DISABLE_COMPILE_CACHE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    relevant = {
        name
        for name in environ
        if name.startswith(("VLLM_", "SKYRL_", "NCCL_", "TORCH_", "PYTORCH_", "NVTE_", "CUBLAS_"))
    }
    unknown = relevant - generated.keys() - ignored - admitted
    if unknown:
        raise ValueError(f"unreviewed runtime controls: {', '.join(sorted(unknown))}")
    if driver and any(name not in environ for name in generated):
        raise ValueError("Ray driver is missing native-injected execution controls")
    return generated | {name: environ[name] for name in sorted(admitted & relevant)}


def _value(root: Any, path: str) -> Any:
    for name in path.split("."):
        if not hasattr(root, name):
            raise ValueError(f"native configuration is missing {path}")
        root = getattr(root, name)
    return root


def _expect(root: Any, path: str, expected: Any) -> None:
    value = _value(root, path)
    # Python considers True == 1; configuration provenance does not.
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path} is outside the initial SkyRL text execution recipe")


def training_request(parameters: dict[str, Any], *, contract: str) -> dict[str, Any]:
    """The sole sampler override, after native construction and only for training."""
    if contract not in ("native", "unmodified"):
        raise ValueError("unknown sampling contract")
    result = copy.deepcopy(parameters)
    if contract == "unmodified":
        result["min_tokens"] = 0
    return result


def validate_profile(cfg: Any) -> None:
    trainer, generator = cfg.trainer, cfg.generator
    algorithm, engine, placement = trainer.algorithm, generator.inference_engine, trainer.placement
    for path in (
        "trainer.gradient_checkpointing",
        "trainer.remove_microbatch_padding",
        "trainer.update_ref_every_epoch",
        "trainer.algorithm.use_kl_in_reward",
        "trainer.algorithm.use_kl_loss",
        "trainer.algorithm.use_entropy_loss",
        "trainer.algorithm.advantage_batch_normalize",
        "trainer.algorithm.zero_variance_filter",
        "trainer.algorithm.grpo_norm_by_std",
        "trainer.fully_async.clear_kv_cache_on_weight_sync",
        "generator.zero_reward_on_non_stop",
        "generator.apply_overlong_filtering",
        "generator.use_cache_salt",
        "generator.append_eos_token_after_stop_str_in_multi_turn",
        "generator.inference_engine.enable_prefix_caching",
        "generator.inference_engine.enforce_eager",
    ):
        if type(_value(cfg, path)) is not bool:
            raise ValueError(f"{path} must be a boolean")
    for path in (
        "trainer.max_prompt_length",
        "generator.max_turns",
        "generator.max_input_length",
        "generator.eval_n_samples_per_prompt",
        "generator.sampling_params.max_generate_length",
        "generator.eval_sampling_params.max_generate_length",
    ):
        integer(_value(cfg, path), path, minimum=1)
    validate_credit_recipe(
        algorithm,
        generator,
        sampling_contract=cfg.llenvs.sampling_contract,
        turn_weighting=cfg.llenvs.turn_weighting,
        token_scorer=cfg.llenvs.token_scorer,
    )
    for path, expected in {
        "trainer.strategy": "fsdp",
        "trainer.bf16": True,
        "trainer.flash_attn": True,
        "trainer.gradient_checkpointing_use_reentrant": False,
        "trainer.recompute_old_logprobs_per_minibatch": True,
        "trainer.fused_lm_head_logprob": False,
        "trainer.mtp.enabled": False,
        "trainer.critic.model.path": None,
        "trainer.policy.use_torch_compile": False,
        "trainer.policy.inference_only_init": False,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.ref_num_nodes": 1,
        "trainer.placement.colocate_policy_ref": True,
        "trainer.fully_async.simulate_training": False,
        "trainer.fully_async.sample_full_batch": False,
        "generator.batched": False,
        "generator.step_wise_trajectories": False,
        "generator.merge_stepwise_output": False,
        "generator.vision_language_generator": False,
        "generator.use_conversation_multi_turn": True,
        "generator.chat_template.name_or_path": None,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.model_dtype": "bfloat16",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.weight_sync_backend": "nccl",
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
        "generator.inference_engine.language_model_only": False,
        "environment.env_class": "llenvs",
    }.items():
        _expect(cfg, path, expected)
    for role in ("policy", "ref"):
        for name, value in {
            "sequence_parallel_size": 1,
            "model.lora.rank": 0,
            "model.fake_int4_qat.enabled": False,
            "model_config_kwargs": {},
            "language_model_only": False,
        }.items():
            _expect(cfg, f"trainer.{role}.{name}", value)
        model = getattr(trainer, role)
        if type(model.fsdp_config.fsdp_size) is not int or model.fsdp_config.fsdp_size not in (
            -1,
            placement.policy_num_gpus_per_node,
        ):
            raise ValueError("fsdp_size must span the single-node data-parallel group")
        precision = model.fsdp_config.mixed_precision
        if precision is not None:
            for name, value in {
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "buffer_dtype": "fp32",
            }.items():
                _expect(precision, name, value)
    if trainer.ref.model.path != trainer.policy.model.path:
        raise ValueError("ref model must use the same local snapshot/tokenizer as policy")
    for name in ("eps_clip_low", "eps_clip_high"):
        if finite_number(getattr(algorithm, name), name) != 0.2:
            raise ValueError(f"{name} must retain the reviewed native 0.2 clip boundary")
    if (
        cfg.llenvs.sampling_contract == "unmodified"
        and finite_number(algorithm.temperature, "algorithm.temperature") != 1
    ):
        raise ValueError("unmodified sampling requires algorithm.temperature=1")
    if algorithm.max_seq_len is not None:
        integer(algorithm.max_seq_len, "algorithm.max_seq_len", minimum=1)
    if set(generator.chat_template_kwargs) - {"enable_thinking"} or any(
        type(value) is not bool for value in generator.chat_template_kwargs.values()
    ):
        raise ValueError("chat_template_kwargs only admits boolean enable_thinking")
    for phase in ("sampling_params", "eval_sampling_params"):
        if (
            finite_number(
                getattr(generator, phase).repetition_penalty, f"{phase}.repetition_penalty"
            )
            != 1
        ):
            raise ValueError(
                f"typed {phase}.repetition_penalty is inert; use reviewed additional_kwargs"
            )
    if (
        not isinstance(engine.engine_init_kwargs, dict)
        or set(engine.engine_init_kwargs) - _ENGINE_OVERRIDES
    ):
        raise ValueError("unsupported inference engine_init_kwargs")
    # This typed field is not forwarded by the pinned builder. Never claim its
    # requested false value took effect; the actual engine override is supported.
    if engine.enable_chunked_prefill is not True:
        raise ValueError(
            "typed enable_chunked_prefill is inert; set engine_init_kwargs.enable_chunked_prefill"
        )
    integer(cfg.environment.skyrl_gym.max_env_workers, "max_env_workers", minimum=1)
    dp = integer(placement.policy_num_gpus_per_node, "policy GPUs", minimum=1)
    if (
        dp not in (1, 2)
        or type(placement.ref_num_gpus_per_node) is not int
        or placement.ref_num_gpus_per_node != dp
    ):
        raise ValueError("initial recipe requires matching policy/ref DP of 1 or 2")
    samples = integer(generator.n_samples_per_prompt, "n_samples_per_prompt", minimum=1)
    batch = integer(trainer.train_batch_size, "train_batch_size", minimum=1)
    mini = integer(trainer.policy_mini_batch_size, "policy_mini_batch_size", minimum=1)
    if batch % mini or mini * samples % dp:
        raise ValueError(
            "real minibatch rows must be divisible by DP and train batches by minibatches"
        )
    micro = integer(
        trainer.micro_train_batch_size_per_gpu, "micro_train_batch_size_per_gpu", minimum=1
    )
    forward = integer(
        trainer.micro_forward_batch_size_per_gpu, "micro_forward_batch_size_per_gpu", minimum=1
    )
    if micro != forward:
        raise ValueError("old-forward/update microbatch settings must match")
    budget = trainer.max_tokens_per_microbatch
    if type(budget) is not int or (budget != -1 and budget <= 0):
        raise ValueError("max_tokens_per_microbatch must be -1 or a positive integer")
    if budget == -1 and (mini * samples // dp) % micro:
        raise ValueError("per-rank real minibatch rows must be divisible by microbatch count")
    if budget > 0 and algorithm.use_entropy_loss:
        raise ValueError(
            "native entropy weighting is not invariant to token-budget microbatch regrouping"
        )
    asynchronous = trainer.fully_async
    if type(asynchronous.enabled) is not bool:
        raise ValueError("fully_async.enabled must be a boolean")
    _expect(placement, "colocate_all", not asynchronous.enabled)
    _expect(algorithm, "policy_loss_type", "rollout_is" if asynchronous.enabled else "regular")
    _expect(engine, "num_engines", 1 if asynchronous.enabled else dp)
    if asynchronous.enabled:
        if (
            batch != mini
            or algorithm.dynamic_sampling.type is not None
            or algorithm.zero_variance_filter
        ):
            raise ValueError(
                "fully async requires equal prompt batches and no outcome-group filtering"
            )
        staleness = integer(asynchronous.max_staleness_steps, "max_staleness_steps")
        workers = integer(
            asynchronous.num_parallel_generation_workers, "generation workers", minimum=1
        )
        if not mini <= workers <= mini * (staleness + 1):
            raise ValueError("fully async generation workers exceed the reviewed staleness window")
        if asynchronous.clear_kv_cache_on_weight_sync and not engine.engine_init_kwargs.get(
            "enable_prefix_caching", engine.enable_prefix_caching
        ):
            raise ValueError("running KV reset requires effective prefix caching enabled")


def validate_engine_args(cfg: Any, args: Any) -> None:
    """Check the actual native vLLM namespace after last-applied overrides."""
    for name, expected in {
        "model": cfg.trainer.policy.model.path,
        "dtype": "bfloat16",
        "generation_config": "vllm",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_lora": False,
        "speculative_config": None,
        "quantization": None,
        "kv_cache_dtype": "auto",
        "hf_overrides": {},
        "override_generation_config": {},
        "logits_processors": None,
    }.items():
        _expect(args, name, expected)
    if args.tokenizer not in (None, cfg.trainer.policy.model.path):
        raise ValueError("resolved engine tokenizer differs from policy snapshot")
    if args.logprobs_mode != "raw_logprobs":
        raise ValueError("resolved logprobs_mode must retain raw chosen-token probabilities")
    if args.max_model_len is not None:
        integer(args.max_model_len, "resolved max_model_len", minimum=1)
    for name in ("enable_prefix_caching", "enable_chunked_prefill"):
        if getattr(args, name) is not None and type(getattr(args, name)) is not bool:
            raise ValueError(f"resolved {name} must be a boolean or native automatic default")
