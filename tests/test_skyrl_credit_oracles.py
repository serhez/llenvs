"""Independent scalar oracles and exact tiny-policy gradients, not GPU parity."""

import importlib
import itertools
import math
import random
import statistics

import pytest

from tests.test_skyrl_credit import row

torch = pytest.importorskip("torch")

compute_credit = importlib.import_module("llenvs.integrations.skyrl._credit").compute_credit


def reference_credit(rewards, ledger, *, estimator, gamma, normalize, weighting):
    """Quadratic forward sums in Python, independent of the tensor recurrence."""
    width = len(rewards[0])
    result = [[0.0] * width for _ in rewards]
    groups = {}
    for index, record in enumerate(ledger):
        offset = width - record["response_length"]
        spans = record["sampled_spans"]
        positions = (
            [j for span in spans for j in range(offset + span["start"], offset + span["end"])]
            if estimator == "llenvs_token_rtg"
            else [offset + span["end"] - 1 for span in spans]
        )
        returns = [
            math.fsum(
                rewards[index][p] * gamma ** (k - t) for k, p in enumerate(positions) if k >= t
            )
            for t in range(len(positions))
        ]
        if estimator == "llenvs_token_rtg":
            for position, value in zip(positions, returns, strict=True):
                result[index][position] = value
        else:
            for span, value in zip(spans, returns, strict=True):
                groups.setdefault(record["instance_id"], []).append((index, offset, span, value))
    for members in groups.values():
        values = [member[-1] for member in members]
        mean = statistics.mean(values)
        scale = statistics.stdev(values) + 1e-6 if normalize and len(values) > 1 else 1.0
        for index, offset, span, value in members:
            credit = (value - mean) / scale
            if weighting == "span_normalized":
                credit /= span["sampled_count"]
            for position in range(offset + span["start"], offset + span["end"]):
                result[index][position] = credit
    return result


RECIPES = [
    ("llenvs_turn_grpo", gamma, normalize, weighting)
    for gamma, normalize, weighting in itertools.product(
        (0.0, 0.5, 1.0), (False, True), ("uniform", "span_normalized")
    )
] + [("llenvs_token_rtg", 1.0, False, "uniform")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("estimator", "gamma", "normalize", "weighting"), RECIPES)
def test_random_credit_matches_independent_oracle_and_layout_invariants(
    dtype, estimator, gamma, normalize, weighting
):
    for seed in range(16):
        rng = random.Random(17000 + seed)
        samples = rng.randint(1, 4)
        ledger, unpadded = [], []
        for group in range(3):
            for repetition in range(samples):
                values, spans = [], []
                for _ in range(rng.randint(1, 5)):
                    values.extend([0.0] * rng.randint(0, 5))
                    start = len(values)
                    count = rng.randint(1, 6)
                    immediate = [rng.randint(-8, 8) / 4 for _ in range(count)]
                    if estimator == "llenvs_turn_grpo":
                        immediate[:-1] = [0.0] * (count - 1)
                    values.extend(immediate)
                    spans.append((start, len(values)))
                values.extend([0.0] * rng.randint(0, 3))
                ledger.append(row(str(group), repetition, len(values), spans))
                unpadded.append(values)
        width = max(map(len, unpadded)) + rng.randint(0, 4)
        values = [[0.0] * (width - len(v)) + v for v in unpadded]
        values.extend([[0.0] * width for _ in range(rng.randint(1, 3))])
        rewards = torch.tensor(values, dtype=dtype)
        options = dict(
            estimator=estimator,
            n_samples_per_prompt=samples,
            gamma=gamma,
            grpo_norm_by_std=normalize,
            turn_weighting=weighting,
        )
        expected = torch.tensor(
            reference_credit(
                values,
                ledger,
                estimator=estimator,
                gamma=gamma,
                normalize=normalize,
                weighting=weighting,
            ),
            dtype=dtype,
        )
        advantages, returns = compute_credit(rewards, attribution=ledger, **options)
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        for actual in (advantages, returns):
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        assert advantages.data_ptr() != returns.data_ptr() != rewards.data_ptr()
        order = list(range(len(ledger)))
        rng.shuffle(order)
        shuffled = compute_credit(
            rewards[order], attribution=[ledger[i] for i in order], **options
        )[0]
        torch.testing.assert_close(shuffled, expected[order], rtol=tolerance, atol=tolerance)
        # A whole other prompt group cannot alter this group's normalization.
        isolated = compute_credit(rewards[:samples], attribution=ledger[:samples], **options)[0]
        torch.testing.assert_close(isolated, expected[:samples], rtol=tolerance, atol=tolerance)
        padded = torch.nn.functional.pad(rewards, (7, 0, 0, 2))
        padded_credit = compute_credit(padded, attribution=ledger, **options)[0]
        torch.testing.assert_close(
            padded_credit[: len(values), 7:], expected, rtol=tolerance, atol=tolerance
        )
        assert torch.count_nonzero(padded_credit[:, :7]) == 0
        assert torch.count_nonzero(padded_credit[len(ledger) :]) == 0


@pytest.mark.parametrize(
    "recipe", ["causal", "delayed_endpoint", "future_redistribution", "final_stop_zeroing"]
)
def test_production_token_credit_has_the_exact_declared_causality_gradient(recipe):
    theta = torch.tensor([-0.4, 0.7], dtype=torch.float64, requires_grad=True)
    probabilities = theta.sigmoid()
    logprobs, outcomes, masses = [], [], []
    for a, b in itertools.product((0, 1), repeat=2):
        chosen = torch.stack(
            [
                probabilities[i] if action else 1 - probabilities[i]
                for i, action in enumerate((a, b))
            ]
        )
        masses.append(chosen.prod())
        logprobs.append(chosen.log())
        if recipe == "causal":
            outcomes.append((a, a * b))
        elif recipe == "delayed_endpoint":
            outcomes.append((0, a * b))
        elif recipe == "future_redistribution":
            outcomes.append((-b, b))  # Always zero total, but earlier label sees the future.
        else:
            outcomes.append((a * b, b))  # A final stop condition erased an earlier reward.
    masses, logprobs = torch.stack(masses), torch.stack(logprobs)
    rewards = torch.zeros((4, 7), dtype=torch.float64)
    rewards[:, [2, 5]] = torch.tensor(outcomes, dtype=torch.float64)
    ledger = [row("enumerated", i, 7, [(2, 3), (5, 6)]) for i in range(4)]
    credit, _ = compute_credit(
        rewards,
        attribution=ledger,
        estimator="llenvs_token_rtg",
        n_samples_per_prompt=4,
        grpo_norm_by_std=False,
    )
    true_objective = (masses * rewards.sum(-1)).sum()
    surrogate = (masses.detach() * (logprobs * credit[:, [2, 5]]).sum(-1)).sum()
    true_gradient = torch.autograd.grad(true_objective, theta, retain_graph=True)[0]
    rtg_gradient = torch.autograd.grad(surrogate, theta)[0]
    if recipe in ("causal", "delayed_endpoint"):
        torch.testing.assert_close(rtg_gradient, true_gradient, rtol=1e-12, atol=1e-12)
    else:
        # These are counterexamples, not promises that the generic scorer can
        # detect causality from a declared string or valid tensor dimensions.
        assert abs((rtg_gradient - true_gradient)[1]) > 0.05
        if recipe == "future_redistribution":
            torch.testing.assert_close(true_gradient, torch.zeros_like(theta))
