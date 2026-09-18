"""Driver-local credit hooks; native code owns all tensor layouts and forwarding."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch

from llenvs.integrations.skyrl._checks import finite_number, integer, token_ids
from llenvs.integrations.skyrl._credit import AttributionRow, _validate_ledger, compute_credit
from llenvs.integrations.skyrl._trace import validate_generation

ATTRIBUTION_KEY = "llenvs/attribution"


def _attribution(
    output: dict[str, Any], uids: list[str], samples: int, vocab_size: int
) -> list[AttributionRow]:
    """Validate complete unpadded rows before a native zip/tensor conversion."""
    count = len(uids)
    required = (
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "rollout_logprobs",
        "trajectory_ids",
        "env_metrics",
    )
    if not count:
        raise ValueError("custom training batch cannot be empty")
    for name in required:
        if not isinstance(output.get(name), list) or len(output[name]) != count:
            raise ValueError(f"{name} must contain exactly one entry per real row")
    for name, rows in output.items():
        if isinstance(rows, list) and len(rows) != count:
            raise ValueError(f"{name} outer length does not match real rows")
    ledger = []
    for metrics in output["env_metrics"]:
        if not isinstance(metrics, dict) or ATTRIBUTION_KEY not in metrics:
            raise ValueError("missing per-row llenvs attribution ledger")
        ledger.append(copy.deepcopy(metrics[ATTRIBUTION_KEY]))
    width = max(map(len, output["response_ids"]))
    groups = _validate_ledger(ledger, samples, width=width, row_count=count)
    for members in groups.values():
        if members != list(range(members[0], members[-1] + 1)):
            raise ValueError("prompt occurrence rows must be contiguous before native conversion")
        if len({uids[i] for i in members}) != 1:
            raise ValueError("one sampling occurrence cannot mix task identities")
    for i, row in enumerate(ledger):
        prompt, response = output["prompt_token_ids"][i], output["response_ids"][i]
        token_ids(prompt, "prompt_token_ids")
        token_ids(response, "response_ids")
        if not prompt or any(token >= vocab_size for token in (*prompt, *response)):
            raise ValueError("prompt/response token outside vocabulary or empty prompt")
        length = row["response_length"]
        if len(response) != length:
            raise ValueError("ledger response_length differs from recorded response")
        trajectory = output["trajectory_ids"][i]
        if (trajectory.instance_id, trajectory.repetition_id) != (uids[i], row["repetition_id"]):
            raise ValueError("native trajectory identity differs from uid/repetition")
        for name in ("rewards", "loss_masks", "rollout_logprobs"):
            values = output[name][i]
            if not isinstance(values, list) or len(values) != length:
                raise ValueError(f"{name} inner length differs from response")
        for value in output["rewards"][i]:
            finite_number(value, "reward")
        for value in output["loss_masks"][i]:
            if integer(value, "loss mask") > 1:
                raise ValueError("loss mask must be 0 or 1")
        membership = [False] * length
        for span in row["sampled_spans"]:
            start, end = span["start"], span["end"]
            membership[start:end] = [True] * (end - start)
            validate_generation(
                expected_prefix=prompt,
                actual_input_ids=prompt,
                output_ids=response[start:end],
                rollout_logprobs=output["rollout_logprobs"][i][start:end],
                vocab_size=vocab_size,
            )
        for j, sampled in enumerate(membership):
            if not sampled and (
                output["loss_masks"][i][j] != 0 or output["rollout_logprobs"][i][j] != 0
            ):
                raise ValueError("context positions must have zero policy mask and rollout logprob")
    return ledger


class LlenvsTrainerMixin:
    """Use before either native trainer in the MRO; never retain a mutable ledger.

    The class deliberately does not import/instantiate the native training
    stack. Its concrete subclasses belong to the native runtime module.
    """

    cfg: Any
    tokenizer: Any
    generator: Any
    all_metrics: dict[str, float]

    async def train(self) -> Any:
        try:
            return await cast(Any, super()).train()
        finally:
            await self.generator.aclose()

    def convert_to_training_input(self, generator_output: Any, uids: list[str]) -> Any:
        if self.cfg.trainer.algorithm.advantage_estimator == "grpo":
            return cast(Any, super()).convert_to_training_input(generator_output, uids)
        algorithm = self.cfg.trainer.algorithm
        ledger = _attribution(
            generator_output,
            uids,
            self.cfg.generator.n_samples_per_prompt,
            integer(
                getattr(self.generator, "vocab_size", None),
                "verified model vocabulary size",
                minimum=1,
            ),
        )
        native_output = dict(generator_output)
        native_output["env_metrics"] = [
            {key: value for key, value in row.items() if key != ATTRIBUTION_KEY}
            for row in generator_output["env_metrics"]
        ]
        # Native TrajectoryID.instance_id is the stable dataset UID. Boundary
        # construction needs occurrence IDs so two visits cannot merge. Restore
        # task UIDs afterwards; checkpoint consumption remains wholly native.
        occurrence_uids = [row["instance_id"] for row in ledger]
        data = cast(Any, super()).convert_to_training_input(native_output, occurrence_uids)
        rewards = data["rewards"]
        count = len(ledger)
        padding = integer(data.metadata.get("pad_size"), "native pad_size")
        if (
            rewards.ndim != 2
            or len(rewards) != count + padding
            or data.metadata["uids"][:count] != occurrence_uids
        ):
            raise ValueError("native conversion changed real-row identity/count")
        # Native DP padding copies row 0, including its rewards. These known
        # dummy rows must not acquire reward/credit or enter group statistics.
        rewards = rewards.clone()
        rewards[count:] = 0
        advantages, returns = compute_credit(
            rewards,
            attribution=ledger,
            estimator=algorithm.advantage_estimator,
            n_samples_per_prompt=self.cfg.generator.n_samples_per_prompt,
            gamma=algorithm.gamma,
            grpo_norm_by_std=algorithm.grpo_norm_by_std,
            turn_weighting=self.cfg.llenvs.turn_weighting,
        )
        data["rewards"], data["advantages"], data["returns"] = rewards, advantages, returns
        data.metadata["uids"] = [*uids, *data.metadata["uids"][count:]]
        sampled_advantages = torch.cat(
            [
                advantages[
                    i,
                    rewards.shape[1] - row["response_length"] + span["start"] : rewards.shape[1]
                    - row["response_length"]
                    + span["end"],
                ]
                for i, row in enumerate(ledger)
                for span in row["sampled_spans"]
            ]
        ).double()
        metrics = {
            "avg_final_rewards": finite_number(
                rewards[:count].double().sum(-1).mean().item(), "mean reward metric"
            ),
            "avg_response_length": sum(row["response_length"] for row in ledger) / count,
            "avg_advantages": finite_number(sampled_advantages.mean().item(), "mean credit metric"),
            "avg_advantages_abs": finite_number(
                sampled_advantages.abs().mean().item(), "absolute credit metric"
            ),
        }
        data.metadata.setdefault("metrics", {}).update(metrics)
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": metrics["avg_final_rewards"],
                "llenvs/avg_sampled_advantages": metrics["avg_advantages"],
                "llenvs/avg_sampled_advantages_abs": metrics["avg_advantages_abs"],
            }
        )
        return data

    def compute_advantages_and_returns(self, data: Any) -> Any:
        if self.cfg.trainer.algorithm.advantage_estimator == "grpo":
            return cast(Any, super()).compute_advantages_and_returns(data)
        for name in ("advantages", "returns"):
            value = data.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != data["rewards"].shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"precomputed custom {name} missing, misaligned, or non-finite")
        return data
