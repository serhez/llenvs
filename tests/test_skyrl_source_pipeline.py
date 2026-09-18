"""Production generator -> native collection/tensors -> production credit, CPU.

Environment/inference and worker model forwards are explicit doubles. All
native collection, validation, padding and tensor-container bodies are pinned.
"""

import copy
from types import SimpleNamespace as Namespace
from unittest.mock import Mock

import pytest

from tests import test_skyrl_generator as generator_tests
from tests.skyrl_acceptance_run import AuditTrainerMixin
from tests.skyrl_source import driver_namespace
from tests.test_skyrl_credit_oracles import reference_credit
from tests.test_skyrl_resources import run_async

torch = pytest.importorskip("torch")
setup = generator_tests.setup


@pytest.mark.parametrize("estimator", ["grpo", "llenvs_turn_grpo", "llenvs_token_rtg"])
@pytest.mark.parametrize("dp", [1, 3])
@run_async
async def test_full_cpu_transport_keeps_sampled_coordinates_credit_and_occurrences(
    setup, monkeypatch, estimator, dp
):
    from llenvs.integrations.skyrl._trainer import LlenvsTrainerMixin

    ns = driver_namespace(monkeypatch)
    setup.cfg.trainer.algorithm.advantage_estimator = estimator
    generator = setup.generator()
    try:
        first = await generator.generate(setup.batch())
        # Repeat the same dataset task at a different horizon/response length.
        generator.max_turns = 1
        second = await generator.generate(setup.batch())
    finally:
        await generator.aclose()
    assert all(env.closed for env in setup.created)
    # Custom occurrence IDs support repeated task visits in one batch. Stock
    # scalar conversion requires unique task groups; keep that native contract.
    parts = [first] if estimator == "grpo" else [second, first]
    count = len(parts) * 2
    before = copy.deepcopy(parts)
    output = ns["concatenate_generator_outputs"](parts)
    assert parts == before
    ns["validate_generator_output"](count, output)

    class Native:
        convert_to_training_input = ns["convert_to_training_input"]
        fwd_logprobs_values_reward = ns["fwd_logprobs_values_reward"]
        postprocess_generator_output = ns["postprocess_generator_output"]

    class Trainer(AuditTrainerMixin, LlenvsTrainerMixin, Native):
        pass

    trainer = Trainer()
    trainer.global_step = 1
    trainer.cfg = Namespace(
        trainer=Namespace(
            train_batch_size=len(parts),
            policy_mini_batch_size=1,
            critic=Namespace(model=Namespace(path=None)),
            algorithm=Namespace(
                advantage_estimator=estimator,
                max_seq_len=1,
                gamma=0.5 if estimator == "llenvs_turn_grpo" else 1.0,
                grpo_norm_by_std=False,
                zero_variance_filter=False,
                off_policy_correction=Namespace(tis_ratio_type=None, sequence_mask_metric=None),
            ),
        ),
        generator=Namespace(
            n_samples_per_prompt=2, step_wise_trajectories=False, merge_stepwise_output=False
        ),
        llenvs=Namespace(turn_weighting="uniform"),
    )
    trainer.tokenizer = Namespace(pad_token_id=0)
    trainer.generator = generator
    trainer.dispatch = Namespace(get_lcm_dp_size=Mock(return_value=dp), empty_cache=Mock())
    trainer.all_metrics, trainer.has_critic, trainer.ref_model = {}, False, None
    trainer._skip_policy_forward = Mock(return_value=False)
    uids = [identity.instance_id for identity in output["trajectory_ids"]]
    # This native stage expands scalar outcomes before tensorization. Custom
    # vectors and attribution must survive it without a reward rewrite.
    ledger_before = copy.deepcopy(output["env_metrics"])
    output, uids = trainer.postprocess_generator_output(output, uids)
    assert output["env_metrics"] == ledger_before
    original = copy.deepcopy(output)
    batch = trainer.convert_to_training_input(output, uids)
    assert output == original
    assert batch.metadata["uids"][:count] == uids
    if estimator != "grpo":
        assert batch.metadata["policy_prompt_boundaries"] == [(0, 2), (2, 4)]
        ledger = [m["llenvs/attribution"] for m in output["env_metrics"]]
        expected = torch.tensor(
            reference_credit(
                batch["rewards"].tolist(),
                ledger,
                estimator=estimator,
                gamma=trainer.cfg.trainer.algorithm.gamma,
                normalize=False,
                weighting="uniform",
            ),
            dtype=batch["rewards"].dtype,
        )
        torch.testing.assert_close(batch["advantages"], expected)
        assert torch.count_nonzero(batch["rewards"][4:]) == 0
    else:
        # Scalar mode remains native, including native copied dummy rewards.
        expected_native = Native.convert_to_training_input(trainer, output, uids)
        for key in ("rewards", "loss_mask", "rollout_logprobs", "sequences"):
            assert torch.equal(batch[key], expected_native[key])

    for i, (prompt, response) in enumerate(
        zip(output["prompt_token_ids"], output["response_ids"], strict=True)
    ):
        assert batch["sequences"][i, -(len(prompt) + len(response)) :].tolist() == prompt + response
        assert batch["loss_mask"][i, -len(response) :].tolist() == output["loss_masks"][i]
        torch.testing.assert_close(
            batch["rollout_logprobs"][i, -len(response) :],
            torch.tensor(output["rollout_logprobs"][i]),
        )
    fields = ["rewards", "loss_mask", "rollout_logprobs"] + (
        [] if estimator == "grpo" else ["advantages", "returns"]
    )
    preserved = {key: batch[key].clone() for key in fields}

    def forward(role, data, **kwargs):
        assert set(data) == {"sequences", "attention_mask"}
        assert set(data.metadata) == {"response_length"}
        return torch.zeros_like(batch["rewards"])

    trainer._execute_forward_pass = forward
    trainer.fwd_logprobs_values_reward(batch)
    if estimator != "grpo":
        trainer.compute_advantages_and_returns(batch)
    order = list(reversed(range(count)))
    reordered = type(batch).cat([batch.slice(i, i + 1) for i in order])
    for key, value in preserved.items():
        assert torch.equal(batch[key], value)
        assert torch.equal(reordered[key], value[order])
    assert "env_metrics" not in batch
    assert "llenvs/attribution" not in batch.metadata
    assert trainer._audit_batches[0]["uids"] == uids
    assert "rollout_train_delta" in trainer._audit_batches[0]
