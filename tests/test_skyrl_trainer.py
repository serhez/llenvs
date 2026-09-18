"""Driver hook contracts with a deliberately small native-boundary test double.

These do not certify the installed native trainer, packing, or GPU forwarding.
"""

import copy
import importlib
from types import SimpleNamespace as Namespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_skyrl_resources import run_async

torch = pytest.importorskip("torch")


class Batch(dict):
    metadata: dict


@pytest.fixture
def output():
    return {
        "prompt_token_ids": [[8, 9], [9]],
        "response_ids": [[1, 2, 7, 3], [4]],
        "rewards": [[0, 1, 0, 2], [0]],
        "loss_masks": [[1, 1, 0, 1], [0]],  # Rejected sample remains in the population.
        "rollout_logprobs": [[-0.1, -0.2, 0, -0.3], [-0.4]],
        "trajectory_ids": [
            Namespace(instance_id="task", repetition_id=0),
            Namespace(instance_id="task", repetition_id=1),
        ],
        "env_metrics": [
            {
                "llenvs/attribution": {
                    "instance_id": "visit",
                    "repetition_id": 0,
                    "response_length": 4,
                    "sampled_spans": [
                        {"generation_id": "a", "start": 0, "end": 2, "sampled_count": 2},
                        {"generation_id": "b", "start": 3, "end": 4, "sampled_count": 1},
                    ],
                }
            },
            {
                "llenvs/attribution": {
                    "instance_id": "visit",
                    "repetition_id": 1,
                    "response_length": 1,
                    "sampled_spans": [
                        {"generation_id": "c", "start": 0, "end": 1, "sampled_count": 1}
                    ],
                }
            },
        ],
    }


@pytest.fixture
def trainer():
    module = importlib.import_module("llenvs.integrations.skyrl._trainer")

    class Native:
        def convert_to_training_input(self, output, uids):
            self.native_conversion(output, uids)
            width = max(map(len, output["response_ids"]))
            rewards = torch.tensor(
                [[0] * (width - len(row)) + row for row in output["rewards"]], dtype=torch.float32
            )
            # Native pad_training_input_batch clones row 0's rewards, not zeros.
            batch = Batch(rewards=torch.cat([rewards, rewards[:1].clone()]))
            batch.metadata = {"pad_size": 1, "uids": [*uids, "pad0"]}
            return batch

        def compute_advantages_and_returns(self, data):
            return self.native_advantages(data)

        async def train(self):
            return await self.native_train()

    class Trainer(module.LlenvsTrainerMixin, Native):
        pass

    instance = Trainer()
    instance.cfg = Namespace(
        trainer=Namespace(
            algorithm=Namespace(
                advantage_estimator="llenvs_turn_grpo", gamma=1, grpo_norm_by_std=False
            )
        ),
        generator=Namespace(n_samples_per_prompt=2),
        llenvs=Namespace(turn_weighting="uniform"),
    )
    instance.tokenizer = range(10)
    instance.generator = Namespace(vocab_size=512)
    instance.native_conversion = Mock()
    instance.native_advantages = Mock(return_value=object())
    instance.all_metrics = {}
    return instance


def test_effective_credit_left_padding_dummy_copy_and_mask_independence(trainer, output):
    original = copy.deepcopy(output)
    batch = trainer.convert_to_training_input(output, ["task", "task"])
    assert torch.allclose(
        batch["advantages"],
        torch.tensor([[4 / 3, 4 / 3, 0, 1 / 3], [0, 0, 0, -5 / 3], [0, 0, 0, 0]]),
    )
    assert not batch["rewards"][-1].any()
    assert batch["advantages"].data_ptr() != batch["returns"].data_ptr()
    assert batch.metadata["uids"] == ["task", "task", "pad0"]
    assert all(
        "llenvs/attribution" not in row
        for row in trainer.native_conversion.call_args.args[0]["env_metrics"]
    )
    assert output == original
    before = batch["advantages"].clone()
    batch["action_log_probs"] = torch.zeros_like(before)  # Native forward adds fields.
    assert trainer.compute_advantages_and_returns(batch) is batch
    assert torch.equal(batch["advantages"], before)
    trainer.native_advantages.assert_not_called()
    # Four original sampled tokens, including the rejected real sample; no
    # observation gap or copied DP dummy enters these diagnostics.
    assert trainer.all_metrics["llenvs/avg_sampled_advantages"] == pytest.approx(1 / 3)
    assert trainer.all_metrics["llenvs/avg_sampled_advantages_abs"] == pytest.approx(7 / 6)
    assert batch.metadata["metrics"]["avg_final_rewards"] == 1.5


def test_repeated_task_visits_use_distinct_native_boundary_keys(trainer, output):
    second = copy.deepcopy(output)
    for row in second["env_metrics"]:
        ledger = row["llenvs/attribution"]
        ledger["instance_id"] = "second-visit"
        for span in ledger["sampled_spans"]:
            span["generation_id"] += "2"
    for name in output:
        output[name].extend(second[name])
    trainer.convert_to_training_input(output, ["task"] * 4)
    assert trainer.native_conversion.call_args.args[1] == [
        "visit",
        "visit",
        "second-visit",
        "second-visit",
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_ledger",
        "ragged_probabilities",
        "context_loss",
        "context_logprob",
        "uid",
        "repetition",
        "row_count",
        "missing_group_member",
        "reward_nan",
        "token_id",
        "bool_mask",
    ],
)
def test_malformed_custom_transport_fails_before_native_conversion(trainer, output, damage):
    if damage == "missing_ledger":
        output["env_metrics"][0] = {}
    elif damage == "ragged_probabilities":
        output["rollout_logprobs"][0].pop()
    elif damage == "context_loss":
        output["loss_masks"][0][2] = 1
    elif damage == "context_logprob":
        output["rollout_logprobs"][0][2] = -0.5
    elif damage == "uid":
        output["trajectory_ids"][0].instance_id = "another"
    elif damage == "repetition":
        output["trajectory_ids"][0].repetition_id = 1
    elif damage == "row_count":
        output["env_metrics"].pop()
    elif damage == "missing_group_member":
        for rows in output.values():
            rows.pop()
    elif damage == "reward_nan":
        output["rewards"][0][-1] = float("nan")
    elif damage == "token_id":
        output["response_ids"][0][0] = -1
    elif damage == "bool_mask":
        output["loss_masks"][0][0] = True
    with pytest.raises(ValueError):
        trainer.convert_to_training_input(output, ["task", "task"])
    trainer.native_conversion.assert_not_called()


def test_native_mode_delegates_without_ledger_or_custom_probability_rules(trainer, output):
    trainer.cfg.trainer.algorithm.advantage_estimator = "grpo"
    del output["env_metrics"]
    del output["trajectory_ids"]
    output["rollout_logprobs"] = None
    batch = trainer.convert_to_training_input(output, ["task", "task"])
    assert "advantages" not in batch
    assert trainer.native_conversion.call_args.args == (output, ["task", "task"])
    assert trainer.compute_advantages_and_returns(batch) is trainer.native_advantages.return_value


@pytest.mark.parametrize("damage", ["missing", "shape", "nan"])
def test_late_hook_never_silently_recomputes_missing_or_damaged_credit(trainer, output, damage):
    batch = trainer.convert_to_training_input(output, ["task", "task"])
    if damage == "missing":
        del batch["advantages"]
    elif damage == "shape":
        batch["advantages"] = batch["advantages"][:1]
    else:
        batch["advantages"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        trainer.compute_advantages_and_returns(batch)
    trainer.native_advantages.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
@run_async
async def test_native_training_exit_always_closes_generator_on_same_loop(trainer, fail):
    import asyncio

    loop = asyncio.get_running_loop()

    async def close():
        assert asyncio.get_running_loop() is loop

    trainer.generator = Namespace(aclose=AsyncMock(side_effect=close))
    trainer.native_train = AsyncMock(
        side_effect=RuntimeError("native training failed") if fail else None
    )
    if fail:
        with pytest.raises(RuntimeError, match="native training failed"):
            await trainer.train()
    else:
        assert await trainer.train() is trainer.native_train.return_value
    trainer.generator.aclose.assert_awaited_once_with()


def test_transport_bounds_use_model_vocabulary_not_tokenizer_length(trainer, output):
    output["response_ids"][0][0] = 300
    batch = trainer.convert_to_training_input(output, ["task", "task"])
    assert torch.isfinite(batch["advantages"]).all()
    output["response_ids"][0][0] = 512
    with pytest.raises(ValueError, match="vocabulary"):
        trainer.convert_to_training_input(output, ["task", "task"])
