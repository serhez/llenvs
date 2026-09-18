"""Independent prefix-by-prefix chosen-token oracle against the native wrapper.

Connects production credit to native reductions/losses/corrections on a tiny
in-memory CPU model. Does not exercise FA2, BF16, model loading or inference.
"""

import copy
import math
from types import SimpleNamespace as Namespace

import pytest

from tests.skyrl_source import worker_namespace
from tests.test_skyrl_credit import row
from tests.test_skyrl_credit_oracles import compute_credit, reference_credit

torch = pytest.importorskip("torch")

REDUCTIONS = ["token_mean", "sequence_mean", "seq_mean_token_sum_norm", "prompt_mean"]


class PrefixModel(torch.nn.Module):
    """Position-sensitive causal CPU model; observations condition later actions."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(731)
            self.tokens = torch.nn.Embedding(128, 6, dtype=torch.float64)
            self.positions = torch.nn.Embedding(32, 6, dtype=torch.float64)
            self.head = torch.nn.Linear(6, 128, dtype=torch.float64)

    def forward(self, sequences, attention_mask, position_ids):
        active = attention_mask.unsqueeze(-1)
        hidden = (self.tokens(sequences) + self.positions(position_ids)) * active
        context = hidden.cumsum(1) / active.cumsum(1).clamp(min=1)
        return {"logits": self.head(context.tanh())}


@pytest.fixture(scope="module")
def native_policy():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield worker_namespace(monkeypatch)


def credit_batch(ns, estimator="llenvs_token_rtg", weighting="uniform"):
    prompts = [[1, 2], [1, 2], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
    responses = [[11, 12, 70, 71, 13, 14, 127], [15, 127], [16, 72, 73, 74, 17, 18, 19, 127], [127]]
    spans = [[(0, 2), (4, 7)], [(0, 2)], [(0, 1), (4, 8)], [(0, 1)]]
    ledger = [row(str(i // 2), i % 2, len(r), spans[i]) for i, r in enumerate(responses)]
    masks, rewards = [], []
    for i, response in enumerate(responses):
        mask, reward = [0] * len(response), [0.0] * len(response)
        for start, end in spans[i]:
            for j in range(start, end):
                mask[j] = 1
                if estimator == "llenvs_token_rtg":
                    reward[j] = (j % 3 - 1) / 4
            reward[end - 1] += (i + end) / 2
        masks.append(mask)
        rewards.append(reward)
    sequences, attention, response_mask, rewards, loss_mask, _, _ = ns[
        "convert_prompts_responses_to_batch_tensors"
    ](0, prompts, responses, rewards, masks, max_seq_len=1)
    advantages, returns = compute_credit(
        rewards,
        attribution=ledger,
        estimator=estimator,
        n_samples_per_prompt=2,
        gamma=1.0,
        grpo_norm_by_std=False,
        turn_weighting=weighting,
    )
    expected = reference_credit(
        rewards.tolist(),
        ledger,
        estimator=estimator,
        gamma=1.0,
        normalize=False,
        weighting=weighting,
    )
    torch.testing.assert_close(advantages, torch.tensor(expected, dtype=advantages.dtype))
    batch = ns["TrainingInputBatch"](
        dict(
            sequences=sequences,
            attention_mask=attention,
            response_mask=response_mask,
            rewards=rewards,
            loss_mask=loss_mask,
            advantages=advantages,
            returns=returns,
            row_ids=torch.arange(4).reshape(-1, 1),
        )
    )
    batch.metadata = {"response_length": rewards.shape[1]}
    return batch, prompts, responses


def prefix_oracle(model, prompts, responses, width, temperature=1.0):
    """Score each next ID from its actual unpadded prefix; no roll/window slice."""
    rows = []
    for prompt, response in zip(prompts, responses, strict=True):
        prefix, values = list(prompt), []
        for token in response:
            tokens = torch.tensor([prefix])
            positions = torch.arange(len(prefix)).unsqueeze(0)
            logits = model(tokens, torch.ones_like(tokens), positions)["logits"][0, len(prefix) - 1]
            values.append(torch.log_softmax(logits / temperature, dim=0)[token])
            prefix.append(token)
        rows.append(torch.cat([values[0].new_zeros(width - len(values)), torch.stack(values)]))
    return torch.stack(rows)


def algorithm(correction="none"):
    options = dict(
        tis_ratio_type=None,
        token_tis_ratio_clip_high=2.0,
        sequence_tis_ratio_clip_high=5.0,
        sequence_mask_metric=None,
        outlier_token_is_threshold_low=None,
        outlier_token_is_threshold_high=None,
        token_mask_is_threshold_low=None,
        token_mask_is_threshold_high=None,
    )
    if correction in ("token", "sequence"):
        options["tis_ratio_type"] = correction
    elif correction == "token_mask":
        options.update(token_mask_is_threshold_low=0.6, token_mask_is_threshold_high=1.6)
    elif correction == "outlier":
        options["outlier_token_is_threshold_high"] = 2.0
    elif correction in ("geometric", "product"):
        options.update(
            sequence_mask_metric=correction,
            geo_mask_low=0.6,
            geo_mask_high=1.6,
            product_mask_low=0.6,
            product_mask_high=1.6,
        )
    return Namespace(
        policy_loss_type="regular",
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        off_policy_correction=Namespace(**options),
    )


def independent_loss(logprobs, old, rollout, advantages, mask, reduction, correction, loss_type):
    """Scalar objective oracle, independent of native masks/reduction helpers."""
    terms = []
    for i in range(len(mask)):
        indices = [j for j, active in enumerate(mask[i]) if active]
        ratios = [math.exp(float(old[i, j] - rollout[i, j])) for j in indices]
        product = math.prod(ratios)
        geometry = product ** (1 / len(indices))
        denominator = {
            "token_mean": int(mask.sum()),
            "sequence_mean": len(mask) * len(indices),
            "seq_mean_token_sum_norm": len(mask) * 13,
            "prompt_mean": 2 * int(mask[(i // 2) * 2 : (i // 2 + 1) * 2].sum()),
        }[reduction]
        for j, ratio in zip(indices, ratios, strict=True):
            keep = not (
                (correction == "token_mask" and not 0.6 <= ratio <= 1.6)
                or (correction == "outlier" and max(ratios) > 2)
                or (correction == "geometric" and not 0.6 <= geometry <= 1.6)
                or (correction == "product" and not 0.6 <= product <= 1.6)
            )
            weight = (
                min(ratio, 2)
                if correction == "token"
                else min(product, 5)
                if correction == "sequence"
                else 1
            )
            a = advantages[i, j] / denominator
            if loss_type == "rollout_is":
                r = (logprobs[i, j] - rollout[i, j]).float().exp().to(logprobs.dtype)
                term = (
                    -r.detach() * a * logprobs[i, j]
                    if 0.8 < float(r.detach()) < 1.2
                    else logprobs[i, j] * 0
                )
            else:
                r = (logprobs[i, j] - old[i, j]).float().exp().to(logprobs.dtype)
                term = -torch.minimum(r * a, r.clamp(0.8, 1.2) * a)
            terms.append(term * (weight if keep else 0))
    return torch.stack(terms).sum()


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_native_next_token_window_matches_prefix_oracle(native_policy, temperature):
    batch, prompts, responses = credit_batch(native_policy)
    model = PrefixModel()
    wrapper = native_policy["HFModelWrapper"](model, use_torch_compile=False)
    actual = wrapper(
        batch["sequences"],
        batch.metadata["response_length"],
        batch["attention_mask"],
        temperature=temperature,
    )
    expected = prefix_oracle(model, prompts, responses, actual.shape[1], temperature)
    mask = batch["response_mask"].bool()  # Includes observation targets as well as sampled EOS.
    torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)
    # Shape-preserving ±1 target shifts must be detected by this oracle.
    for shift in (-1, 1):
        assert not torch.allclose(actual.roll(shift, dims=1)[mask], expected[mask])
    changed = copy.deepcopy(responses)
    changed[0][2] = 69
    perturbed = prefix_oracle(model, prompts, changed, actual.shape[1], temperature)
    assert torch.equal(perturbed[0, 1:3], expected[0, 1:3])
    assert not torch.equal(perturbed[0, -3:], expected[0, -3:])
    assert torch.equal(perturbed[1:], expected[1:])


@pytest.mark.parametrize(
    "estimator,weighting",
    [
        ("llenvs_token_rtg", "uniform"),
        ("llenvs_turn_grpo", "uniform"),
        ("llenvs_turn_grpo", "span_normalized"),
    ],
)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize(
    "correction,loss_type",
    [
        (c, "regular")
        for c in ("none", "token", "sequence", "token_mask", "outlier", "geometric", "product")
    ]
    + [("none", "rollout_is")],
)
def test_production_credit_native_loss_and_gradients_match_independent_oracle(
    native_policy, estimator, weighting, reduction, correction, loss_type
):
    batch, prompts, responses = credit_batch(native_policy, estimator, weighting)
    model, reference = PrefixModel(), PrefixModel()
    wrapper = native_policy["HFModelWrapper"](model, use_torch_compile=False)
    logprobs = wrapper(
        batch["sequences"], batch.metadata["response_length"], batch["attention_mask"]
    )
    expected = prefix_oracle(reference, prompts, responses, logprobs.shape[1])
    # Synthetic chosen probabilities: exercise clipping/rejection, not provenance.
    old = expected.detach() - torch.tensor([0.0, 0.1, 0.5, -0.5, 0.1, 0.0, 0.5, -0.5])
    rollout = old - torch.tensor([0.0, 0.2, 1.0, 0.0, 0.1, 0.0, 0.8, 0.0])
    scaled = native_policy["apply_loss_reduction_to_advantages_minibatch"](
        batch["advantages"], batch["loss_mask"], reduction, 1, 13, [(0, 2), (2, 4)]
    )
    cfg = algorithm(correction)
    cfg.policy_loss_type = loss_type
    fn = native_policy["rollout_is_policy_loss" if loss_type == "rollout_is" else "ppo_policy_loss"]
    actual, metrics = fn(logprobs, old, scaled, cfg, batch["loss_mask"], rollout)
    target = independent_loss(
        expected,
        old,
        rollout,
        batch["advantages"],
        batch["loss_mask"],
        reduction,
        correction,
        loss_type,
    )
    torch.testing.assert_close(actual, target, rtol=2e-6, atol=1e-7)
    assert all(math.isfinite(v) for v in metrics.values())
    actual.backward()
    target.backward()
    for parameter, independent in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, independent.grad, rtol=3e-6, atol=2e-7)
    if correction in ("none", "token", "sequence"):
        assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
