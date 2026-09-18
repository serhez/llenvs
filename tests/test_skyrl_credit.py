"""CPU contracts for complete-group credit in response-relative coordinates.

Imports happen in fixtures so all proposed cases collect before implementation.
These tests exercise the connector, not copies of its credit algorithms.
"""

import copy
import importlib
import math

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def credit():
    return importlib.import_module("llenvs.integrations.skyrl._credit")


def row(instance, repetition, length, spans):
    return {
        "instance_id": instance,
        "repetition_id": repetition,
        "response_length": length,
        "sampled_spans": [
            {
                "generation_id": f"{instance}/{repetition}/{i}",
                "start": start,
                "end": end,
                "sampled_count": end - start,
            }
            for i, (start, end) in enumerate(spans)
        ],
    }


@pytest.fixture
def batch():
    # The second response is left-padded by three columns. Columns 2:4 of
    # the first response are observations, not sampled actions.
    rewards = torch.tensor([[0, 1, 0, 0, 3], [0, 0, 0, 0, 2]], dtype=torch.float64)
    ledger = [row("g", 0, 5, [(0, 2), (4, 5)]), row("g", 1, 2, [(0, 2)])]
    return rewards, ledger


def compute(credit, batch, **options):
    rewards, ledger = batch
    kwargs = {
        "estimator": "llenvs_turn_grpo",
        "n_samples_per_prompt": 2,
        "gamma": 1.0,
        "grpo_norm_by_std": False,
        "turn_weighting": "uniform",
    }
    kwargs.update(options)
    return credit.compute_credit(rewards, attribution=ledger, **kwargs)


def assert_credit(result, expected, rewards):
    advantages, returns = result
    target = torch.tensor(expected, dtype=rewards.dtype)
    tolerance = 1e-12 if rewards.dtype == torch.float64 else 1e-6
    torch.testing.assert_close(advantages, target, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(returns, target, rtol=tolerance, atol=tolerance)
    assert advantages.dtype == returns.dtype == rewards.dtype
    assert advantages.device == returns.device == rewards.device


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [
        (0.0, [[-1, -1, 0, 0, 1], [0, 0, 0, 0, 0]]),
        (0.5, [[0, 0, 0, 0, 0.5], [0, 0, 0, -0.5, -0.5]]),
        (1.0, [[1, 1, 0, 0, 0], [0, 0, 0, -1, -1]]),
    ],
)
def test_turn_returns_use_decision_time_and_pool_decisions_once(credit, batch, gamma, expected):
    assert_credit(compute(credit, batch, gamma=gamma), expected, batch[0])


def test_turn_std_is_sample_std_with_the_declared_epsilon(credit, batch):
    # Returns [4, 3, 2] have sample std 1, not population std sqrt(2/3).
    scale = 1 / (1 + 1e-6)
    expected = [[scale, scale, 0, 0, 0], [0, 0, 0, -scale, -scale]]
    assert_credit(compute(credit, batch, grpo_norm_by_std=True), expected, batch[0])


def test_span_normalization_uses_original_sampled_count(credit, batch):
    expected = [[0.5, 0.5, 0, 0, 0], [0, 0, 0, -0.5, -0.5]]
    assert_credit(compute(credit, batch, turn_weighting="span_normalized"), expected, batch[0])


def test_token_rtg_uses_sampled_tokens_without_group_centering(credit, batch):
    expected = [[4, 4, 0, 0, 3], [0, 0, 0, 2, 2]]
    assert_credit(compute(credit, batch, estimator="llenvs_token_rtg"), expected, batch[0])


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [([1, -1], [0, -1]), ([0, 0], [0, 0]), ([0, 2], [2, 2]), ([1, 1], [2, 1])],
)
def test_equal_episode_totals_do_not_erase_token_placement(credit, rewards, expected):
    tensor = torch.tensor([rewards], dtype=torch.float64)
    ledger = [row("g", 0, 2, [(0, 2)])]
    result = compute(credit, (tensor, ledger), estimator="llenvs_token_rtg", n_samples_per_prompt=1)
    assert_credit(result, [expected], tensor)


def test_terminal_only_unequal_turn_counts_are_not_scalar_grpo(credit):
    rewards = torch.tensor([[0, 1], [0, 0]], dtype=torch.float64)
    ledger = [row("g", 0, 2, [(0, 1), (1, 2)]), row("g", 1, 1, [(0, 1)])]
    assert_credit(compute(credit, (rewards, ledger)), [[1 / 3, 1 / 3], [0, -2 / 3]], rewards)


@pytest.mark.parametrize("norm", [False, True])
def test_singleton_and_constant_pools_are_exactly_zero(credit, norm):
    for length in (1, 3):
        rewards = torch.zeros((1, length), dtype=torch.float64)
        rewards[0, -1] = 1e200
        ledger = [row("g", 0, length, [(i, i + 1) for i in range(length)])]
        result = compute(credit, (rewards, ledger), n_samples_per_prompt=1, grpo_norm_by_std=norm)
        assert torch.equal(result[0], torch.zeros_like(rewards))
        assert torch.equal(result[1], torch.zeros_like(rewards))


def test_one_trajectory_with_multiple_decisions_is_a_valid_pool(credit):
    rewards = torch.tensor([[1, 3]], dtype=torch.float64)
    ledger = [row("g", 0, 2, [(0, 1), (1, 2)])]
    result = compute(credit, (rewards, ledger), n_samples_per_prompt=1)
    assert_credit(result, [[0.5, -0.5]], rewards)


def test_group_identity_and_row_permutation_are_respected(credit, batch):
    rewards, ledger = batch
    other = [
        row("other", i, r["response_length"], [(s["start"], s["end"]) for s in r["sampled_spans"]])
        for i, r in enumerate(ledger)
    ]
    joined = torch.cat([rewards, rewards * 10])
    all_rows = ledger + other
    baseline = compute(credit, (joined, all_rows))[0]
    order = [3, 0, 2, 1]
    permuted = compute(credit, (joined[order], [all_rows[i] for i in order]))[0]
    torch.testing.assert_close(permuted, baseline[order])
    torch.testing.assert_close(baseline[:2], compute(credit, batch)[0])


@pytest.mark.parametrize("estimator", ["llenvs_turn_grpo", "llenvs_token_rtg"])
def test_padding_and_dummy_rows_do_not_change_real_credit(credit, batch, estimator):
    rewards, ledger = batch
    baseline = compute(credit, batch, estimator=estimator)[0]
    padded = torch.nn.functional.pad(rewards, (4, 0, 0, 1))
    advantages, returns = compute(credit, (padded, ledger), estimator=estimator)
    torch.testing.assert_close(advantages[:2, 4:], baseline)
    assert torch.count_nonzero(advantages[:, :4]) == 0
    assert torch.count_nonzero(advantages[2]) == 0
    torch.testing.assert_close(returns, advantages)


def test_effective_rewards_are_authoritative_and_inputs_are_unchanged(credit, batch):
    rewards, ledger = batch
    before_rewards = rewards.clone()
    before_ledger = copy.deepcopy(ledger)
    advantages, returns = compute(credit, batch)
    assert torch.equal(rewards, before_rewards)
    assert ledger == before_ledger
    returns[0, 0] = 42
    assert advantages[0, 0] == 1
    assert torch.equal(rewards, before_rewards)
    rewards.zero_()
    assert torch.count_nonzero(compute(credit, batch)[0]) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("start", -1), ("end", 99), ("end", 0), ("sampled_count", 1), ("start", False)],
)
def test_invalid_spans_are_rejected_not_clipped(credit, batch, field, value):
    batch[1][0]["sampled_spans"][0][field] = value
    with pytest.raises(ValueError, match="span|sampled|integer"):
        compute(credit, batch)


@pytest.mark.parametrize(
    "case",
    [
        "overlap",
        "unordered",
        "empty",
        "duplicate_generation",
        "duplicate_row",
        "missing_row",
        "wrong_repetition",
        "response_too_long",
    ],
)
def test_malformed_or_incomplete_ledgers_never_fall_back(credit, batch, case):
    rewards, ledger = batch
    spans = ledger[0]["sampled_spans"]
    if case == "overlap":
        spans[1].update(start=1, end=3, sampled_count=2)
    elif case == "unordered":
        spans.reverse()
    elif case == "empty":
        ledger[0]["sampled_spans"] = []
    elif case == "duplicate_generation":
        spans[1]["generation_id"] = spans[0]["generation_id"]
    elif case == "duplicate_row":
        ledger[1]["repetition_id"] = 0
    elif case == "missing_row":
        ledger.pop()
        rewards = rewards[:1]
    elif case == "wrong_repetition":
        ledger[1]["repetition_id"] = 2
    else:
        ledger[0]["response_length"] = 6
    with pytest.raises(ValueError):
        compute(credit, (rewards, ledger))


@pytest.mark.parametrize("position", [(0, 2), (1, 0), (0, 0)])
def test_turn_rewards_must_be_only_at_sampled_endpoints(credit, batch, position):
    batch[0][position] = 1
    with pytest.raises(ValueError, match="reward|endpoint|sampled|padding"):
        compute(credit, batch)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_rewards_are_rejected(credit, batch, value):
    batch[0][0, 1] = value
    with pytest.raises(ValueError, match="finite"):
        compute(credit, batch)


def test_finite_inputs_that_overflow_returns_are_rejected(credit):
    rewards = torch.tensor([[3e38, 3e38]], dtype=torch.float32)
    ledger = [row("g", 0, 2, [(0, 1), (1, 2)])]
    with pytest.raises(ValueError, match="finite|overflow"):
        compute(credit, (rewards, ledger), estimator="llenvs_token_rtg", n_samples_per_prompt=1)


@pytest.mark.parametrize(
    "options", [{"gamma": 0.5}, {"grpo_norm_by_std": True}, {"turn_weighting": "span_normalized"}]
)
def test_token_recipe_rejects_other_clocks_or_normalizers(credit, batch, options):
    with pytest.raises(ValueError):
        compute(credit, batch, estimator="llenvs_token_rtg", **options)


def test_missing_ledger_is_an_error_not_scalar_grpo(credit, batch):
    with pytest.raises(TypeError, match="attribution"):
        credit.compute_credit(batch[0], estimator="llenvs_turn_grpo", n_samples_per_prompt=2)
