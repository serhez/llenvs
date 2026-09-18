"""Thin native bindings. Import only from the staged SkyRL runtime.

Concrete root annotations are required by SkyRL's nested dataclass builder.
No trainer, packing, distributed layout, or checkpoint loop is reimplemented.
"""

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self, cast

from skyrl.train.config import EnvironmentConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.generators.base import GeneratorInterface
from skyrl.train.generators.utils import get_rollout_metrics
from skyrl.train.trainer import RayPPOTrainer

from llenvs.integrations.skyrl._checks import integer
from llenvs.integrations.skyrl._config import LlenvsConfig, forward_environment
from llenvs.integrations.skyrl._generator import EpisodeGenerator
from llenvs.integrations.skyrl._manifest import check_run, local_path, run_identity
from llenvs.integrations.skyrl._preflight import (
    training_request,
    validate_engine_args,
    validate_profile,
)
from llenvs.integrations.skyrl._preparation import (
    PreparedInputs,
    execution_recipe,
    prepare_inputs,
    runtime_identity,
)
from llenvs.integrations.skyrl._registry import register_estimators
from llenvs.integrations.skyrl._rendering import TextRenderer
from llenvs.integrations.skyrl._trainer import LlenvsTrainerMixin
from llenvs.integrations.skyrl._validation import validate_sampling_request


@dataclass
class LlenvsSkyRLTrainConfig(SkyRLTrainConfig):
    environment: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(env_class="llenvs")
    )
    llenvs: LlenvsConfig = field(default_factory=LlenvsConfig)

    @classmethod
    def from_cli_overrides(cls, args: list[str] | dict[str, Any]) -> Self:
        return cast(Self, super().from_cli_overrides(args))


@dataclass(frozen=True)
class PreparedRun:
    inputs: PreparedInputs
    identity: dict[str, Any]
    engine: dict[str, Any]
    train_sampling: dict[str, Any]
    eval_sampling: dict[str, Any]


def _json_config(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_config(asdict(value))
    if isinstance(value, Enum):
        return _json_config(value.value)
    if isinstance(value, dict):
        return {key: _json_config(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_config(child) for child in value]
    return value


def prepare(cfg: LlenvsSkyRLTrainConfig, *, driver: bool = False) -> PreparedRun:
    """Validate and fingerprint on both the launching host and the Ray driver."""
    import skyrl
    from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
        get_sampling_params_for_backend,
    )
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
    from skyrl.train.utils import validate_cfg
    from vllm import AsyncEngineArgs

    register_estimators()  # Actual functions must exist before native validation.
    validate_cfg(cfg)
    validate_profile(cfg)
    runtime = runtime_identity(Path(skyrl.__file__).resolve().parent.parent, driver=driver)
    inputs = prepare_inputs(cfg)
    cfg.llenvs.config = str(inputs.selected.path)
    for name in ("ckpt_path", "export_path", "log_path"):
        setattr(cfg.trainer, name, str(local_path(getattr(cfg.trainer, name), f"trainer.{name}")))
    if cfg.trainer.resume_path is not None:
        cfg.trainer.resume_path = str(local_path(cfg.trainer.resume_path, "trainer.resume_path"))
    forward_environment(cfg.llenvs)  # Required presence only; values never enter identity.
    args = build_vllm_cli_args(cfg)
    validate_engine_args(cfg, args)
    # Reads the already-staged local model config; does not create an engine or
    # load model weights. Do not infer these facts from len(tokenizer).
    model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
    if model_config.get_vocab_size() != inputs.model["vocab_size"]:
        raise ValueError("resolved engine/output vocabulary differs from the policy model")
    engine = {
        "vocab_size": model_config.get_vocab_size(),
        "model_context_length": model_config.max_model_len,
        "logprobs_mode": model_config.logprobs_mode,
    }
    train_sampling = training_request(
        get_sampling_params_for_backend("vllm", cfg.generator.sampling_params),
        contract=cfg.llenvs.sampling_contract,
    )
    eval_sampling = get_sampling_params_for_backend("vllm", cfg.generator.eval_sampling_params)
    validate_sampling_request(
        train_sampling,
        contract=cfg.llenvs.sampling_contract,
        phase="train",
        logprobs_mode=engine["logprobs_mode"],
    )
    identity = run_identity(
        datasets={
            "train": inputs.train._rows,
            "eval": inputs.evaluation._rows if inputs.evaluation else [],
        },
        environment={
            "config": asdict(inputs.selected.environment_config),
            "system_prompt": inputs.selected.system_prompt,
            "fingerprint": inputs.selected.fingerprint,
        },
        rewards={
            "algorithm": asdict(cfg.trainer.algorithm),
            "llenvs": asdict(cfg.llenvs) | {"config": None, "check_only": None},
            "judges": [asdict(judge) for judge in inputs.selected.judges],
        },
        execution={
            "configuration": execution_recipe(asdict(cfg)),
            "engine_args": _json_config(vars(args)),
            "resolved_model": engine,
            "train_sampling": train_sampling,
            "eval_sampling": eval_sampling,
        },
        models=inputs.model,
        runtime=runtime,
    )
    check_run(
        cfg.trainer.ckpt_path,
        identity,
        resume_mode=cfg.trainer.resume_mode,
        resume_path=cfg.trainer.resume_path,
    )
    return PreparedRun(inputs, identity, engine, train_sampling, eval_sampling)


class NativeGenerator(EpisodeGenerator, GeneratorInterface):
    eval_sampling_params: dict[str, Any]

    async def generate(self, input_batch: Any) -> Any:
        try:
            phase = input_batch["batch_metadata"].training_phase
            request = dict(input_batch)
            defaults = self.train_sampling_params if phase == "train" else self.eval_sampling_params
            parameters = copy.deepcopy(input_batch.get("sampling_params") or defaults)
            request["sampling_params"] = (
                training_request(parameters, contract=self.cfg.llenvs.sampling_contract)
                if phase == "train"
                else parameters
            )
            output = await super().generate(request)
            # Structured per-row attribution/diagnostics are transported for
            # credit/eval dumps, never passed to the numeric environment reducer.
            output["rollout_metrics"] = get_rollout_metrics(
                output["response_ids"],
                output["rewards"],
                loss_masks=output["loss_masks"],
                trajectory_completion_times=output["trajectory_generation_times"],
                trajectory_time_splits=output["trajectory_time_splits"],
            )
            return output
        except BaseException:
            await self.aclose()
            raise


class NativeTrainer(LlenvsTrainerMixin, RayPPOTrainer):
    pass


class NativeAsyncTrainer(LlenvsTrainerMixin, FullyAsyncRayPPOTrainer):
    pass


class LlenvsPPOExp(BasePPOExp):
    def __init__(self, cfg: LlenvsSkyRLTrainConfig, prepared: PreparedRun):
        self.prepared = prepared
        super().__init__(cfg)

    def _validate_rendered_inputs(self, dataset: Any) -> Any:
        if dataset is None:
            return None
        renderer = TextRenderer(
            self.tokenizer,
            vocab_size=self.prepared.engine["vocab_size"],
            chat_template_kwargs=self.cfg.generator.chat_template_kwargs,
        )
        maximum = min(
            integer(self.cfg.trainer.max_prompt_length, "max_prompt_length", minimum=1),
            integer(self.cfg.generator.max_input_length, "max_input_length", minimum=1),
        )
        sequence = self.prepared.engine["model_context_length"]
        for row in dataset._rows:
            length = len(renderer.initial(row["prompt"]))
            if length > maximum or length >= sequence:
                raise ValueError(
                    "prepared prompt exceeds the actual rendered budget; no task filtering is applied"
                )
        return dataset

    def get_train_dataset(self) -> Any:
        return self._validate_rendered_inputs(self.prepared.inputs.train)

    def get_eval_dataset(self) -> Any:
        return self._validate_rendered_inputs(self.prepared.inputs.evaluation)

    def get_generator(
        self, cfg: Any, tokenizer: Any, inference_engine_client: Any
    ) -> NativeGenerator:
        from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name

        model_identity = self.prepared.inputs.model["content_hash"]
        generator = NativeGenerator(
            cfg,
            self.prepared.inputs.selected,
            tokenizer,
            inference_engine_client,
            train_sampling_params=self.prepared.train_sampling,
            provenance={"model": model_identity, "tokenizer": model_identity, "processor": None},
            policy_model_name=resolve_policy_model_name(cfg),
            **self.prepared.engine,
        )
        generator.eval_sampling_params = copy.deepcopy(self.prepared.eval_sampling)
        return generator

    def get_trainer(
        self,
        cfg: Any,
        tracker: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any,
        inference_engine_client: Any,
        generator: Any,
        colocate_pg: Any,
    ) -> Any:
        trainer = NativeAsyncTrainer if cfg.trainer.fully_async.enabled else NativeTrainer
        return trainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


def run_experiment(cfg: LlenvsSkyRLTrainConfig, expected_identity: dict[str, Any]) -> None:
    prepared = prepare(cfg, driver=True)
    if prepared.identity != expected_identity:
        raise ValueError(
            "Ray driver code/data/model/runtime identity differs from launch preflight"
        )
    check_run(
        cfg.trainer.ckpt_path,
        prepared.identity,
        resume_mode=cfg.trainer.resume_mode,
        resume_path=cfg.trainer.resume_path,
        write=True,
    )
    LlenvsPPOExp(cfg, prepared).run()


def launch(cfg: LlenvsSkyRLTrainConfig, identity: dict[str, Any]) -> None:
    import ray
    from skyrl.train.utils.utils import initialize_ray

    if ray.is_initialized():
        raise ValueError(
            "launch requires a fresh native Ray initialization, not an existing driver session"
        )
    forwarded = forward_environment(cfg.llenvs)
    initialize_ray(cfg)
    try:
        driver = ray.remote(num_cpus=1)(run_experiment)
        ray.get(driver.options(runtime_env={"env_vars": forwarded}).remote(cfg, identity))
    finally:
        ray.shutdown()
