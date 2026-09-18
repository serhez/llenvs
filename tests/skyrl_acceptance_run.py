"""Opt-in subprocess entry for real native training/resume with audit-only hooks.

No trainer loop is copied. Run via test_skyrl_gpu.py, not as a generic launcher.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path


def _diagnostic_value(value):
    """Retain corrupt non-finite inputs as labeled strings in valid failure JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"nonfinite:{value}"
    if isinstance(value, dict):
        return {key: _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    return value


class AuditTrainerMixin:
    async def eval(self):
        self._audit_stage = "evaluation"
        result = await super().eval()
        assert result
        self._audit_evaluations.append(self.global_step)
        return result

    def convert_to_training_input(self, output, uids):
        import torch

        from tests.test_skyrl_credit_oracles import reference_credit

        self._audit_batches = getattr(self, "_audit_batches", [])
        recorded = copy.deepcopy(output)
        self._audit_stage = "conversion"
        self._audit_pending_batch = dict(
            global_step=self.global_step,
            uids=list(uids),
            prompts=recorded["prompt_token_ids"],
            responses=recorded["response_ids"],
            rollout_logprobs=recorded["rollout_logprobs"],
            loss_masks=recorded["loss_masks"],
            rewards=recorded["rewards"],
            diagnostics=recorded["env_metrics"],
        )
        batch = super().convert_to_training_input(output, uids)
        assert output == recorded
        real = len(uids)
        for i, (prompt, response) in enumerate(
            zip(output["prompt_token_ids"], output["response_ids"], strict=True)
        ):
            assert (
                batch["sequences"][i, -(len(prompt) + len(response)) :].tolist()
                == prompt + response
            )
            torch.testing.assert_close(
                batch["rollout_logprobs"][i, -len(response) :],
                torch.tensor(output["rollout_logprobs"][i]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                batch["rewards"][i, -len(response) :],
                torch.tensor(output["rewards"][i], dtype=batch["rewards"].dtype),
                rtol=0,
                atol=0,
            )
            assert batch["loss_mask"][i, -len(response) :].tolist() == output["loss_masks"][i]
            length = len(prompt) + len(response)
            assert batch["attention_mask"][i].tolist() == (
                [0] * (batch["sequences"].shape[1] - length) + [1] * length
            )
            assert batch["response_mask"][i].tolist() == (
                [0] * (batch["rewards"].shape[1] - len(response)) + [1] * len(response)
            )
        assert "env_metrics" not in batch and "llenvs/attribution" not in batch.metadata
        estimator = self.cfg.trainer.algorithm.advantage_estimator
        if estimator != "grpo":
            ledger = [m["llenvs/attribution"] for m in output["env_metrics"]]
            expected = reference_credit(
                batch["rewards"].tolist(),
                ledger,
                estimator=estimator,
                gamma=self.cfg.trainer.algorithm.gamma,
                normalize=self.cfg.trainer.algorithm.grpo_norm_by_std,
                weighting=self.cfg.llenvs.turn_weighting,
            )
            for key in ("advantages", "returns"):
                torch.testing.assert_close(
                    batch[key], torch.tensor(expected, dtype=batch[key].dtype)
                )
            assert torch.count_nonzero(batch["advantages"][real:]) == 0
        self._audit_batches.append(self._audit_pending_batch)
        self._audit_pending_batch = None
        return batch

    def fwd_logprobs_values_reward(self, batch):
        import torch

        self._audit_stage = "forward"
        preserved = {
            key: batch[key].clone()
            for key in ("advantages", "returns")
            if batch.get(key) is not None
        }
        result = super().fwd_logprobs_values_reward(batch)
        for key, value in preserved.items():
            assert torch.equal(result[key], value)
        logprobs = result.get("action_log_probs")
        if logprobs is not None:
            mask = result["loss_mask"].bool()
            delta = (logprobs[mask] - result["rollout_logprobs"][mask]).abs().double()
            assert delta.numel() and bool(torch.isfinite(delta).all())
            self._audit_batches[-1]["rollout_train_delta"] = dict(
                max_abs=delta.max().item(), p99_abs=delta.quantile(0.99).item()
            )
        return result

    def train_critic_and_policy(self, batch):
        self._audit_stage = "policy_update"
        result = super().train_critic_and_policy(batch)
        assert result and all(math.isfinite(float(value)) for value in result.values())
        self._audit_batches[-1]["policy_metrics"] = {
            key: float(value) for key, value in result.items()
        }
        return result

    async def train(self):
        self._audit_batches = []
        self._audit_evaluations = []
        self._audit_pending_batch = None
        self._audit_stage = "training"
        failure = None
        completed = False
        try:
            result = await super().train()
            completed = True
            return result
        except BaseException as error:
            failure = dict(stage=self._audit_stage, type=type(error).__name__, message=str(error))
            raise
        finally:
            path = Path(self.cfg.trainer.log_path) / "llenvs-acceptance.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(
                    _diagnostic_value(
                        dict(
                            completed=completed,
                            batches=self._audit_batches,
                            evaluation_steps=self._audit_evaluations,
                            final_counter=self.global_step,
                            pending_batch=self._audit_pending_batch,
                            failure=failure,
                        )
                    ),
                    stream,
                    allow_nan=False,
                )


def audited_run_experiment(cfg, identity):
    from llenvs.integrations.skyrl import _native as native
    from tests.skyrl_acceptance import register_fixture

    register_fixture()  # The real Ray driver needs its own registry instance.
    if native.run_experiment is audited_run_experiment:
        raise RuntimeError("acceptance driver must import an unmodified native entry point")

    class Trainer(AuditTrainerMixin, native.NativeTrainer):
        pass

    class AsyncTrainer(AuditTrainerMixin, native.NativeAsyncTrainer):
        pass

    original = native.LlenvsPPOExp

    class Experiment(original):
        def get_trainer(
            self,
            cfg,
            tracker,
            tokenizer,
            train_dataset,
            eval_dataset,
            inference_engine_client,
            generator,
            colocate_pg,
        ):
            cls = AsyncTrainer if cfg.trainer.fully_async.enabled else Trainer
            return cls(
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=colocate_pg,
            )

    # Audit-only experiment selection in this test process. Native launch,
    # identity revalidation, manifest publication and training remain real.
    native.LlenvsPPOExp = Experiment
    try:
        native.run_experiment(cfg, identity)
    finally:
        native.LlenvsPPOExp = original


def main():
    if not __debug__:
        raise RuntimeError("acceptance assertions require Python without -O")
    if os.environ.get("LLENVS_SKYRL_GPU") != "1":
        raise RuntimeError("this subprocess requires explicit GPU acceptance opt-in")
    if os.environ.get("RAY_ADDRESS") != "local":
        raise RuntimeError("acceptance requires RAY_ADDRESS=local; do not attach to another job")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sync", "async"))
    parser.add_argument("estimator", choices=("grpo", "llenvs_turn_grpo", "llenvs_token_rtg"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.directory.is_absolute() or not args.directory.is_dir():
        raise ValueError("requires an existing owned absolute acceptance directory")
    from llenvs.integrations.skyrl import _native as native
    from llenvs.integrations.skyrl._manifest import content_hash
    from llenvs.integrations.skyrl.data import export_prompt_data
    from tests import skyrl_acceptance as fixture

    options = fixture.acceptance_options(os.environ)
    fixture.register_fixture()
    config = args.directory / "environment.yaml"
    if not args.resume:
        revision = content_hash(Path(fixture.__file__).read_text())
        with config.open("x") as stream:
            json.dump(
                {
                    "environments": [
                        {
                            "name": "transport",
                            "adapter": "skyrl_acceptance",
                            "size": 6,
                            "seed": 42,
                            "params": {"fixture_revision": revision},
                        }
                    ]
                },
                stream,
            )
        export_prompt_data(config, args.directory / "train.jsonl", indices=range(4))
        export_prompt_data(config, args.directory / "eval.jsonl", indices=range(4, 6))
    raw = fixture.smoke_config(
        args.directory,
        options["model"],
        asynchronous=args.mode == "async",
        estimator=args.estimator,
        dp=args.dp,
        resume=args.resume,
    )
    cfg = native.LlenvsSkyRLTrainConfig.from_cli_overrides(raw)
    prepared = native.prepare(cfg)
    # A canonical importable function is serialized by reference, not as a
    # __main__ closure containing the launching process's patched globals.
    from tests.skyrl_acceptance_run import audited_run_experiment as driver_entry

    original = native.run_experiment
    native.run_experiment = driver_entry
    try:
        native.launch(cfg, prepared.identity)
    finally:
        native.run_experiment = original


if __name__ == "__main__":
    main()
