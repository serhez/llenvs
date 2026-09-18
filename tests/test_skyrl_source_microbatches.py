"""Real native token bins and balancing dummies on CPU, not FA2/DP acceptance."""

from types import SimpleNamespace as Namespace

import pytest

from tests import test_skyrl_source_policy as policy_tests

torch = pytest.importorskip("torch")
native_policy = policy_tests.native_policy


@pytest.mark.parametrize("budget", [0, 10, 20])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("reduction", policy_tests.REDUCTIONS)
def test_native_bins_preserve_every_field_and_policy_gradient(
    native_policy, monkeypatch, budget, padding, reduction
):
    ns = native_policy
    batch, _, _ = policy_tests.credit_batch(ns)
    model = policy_tests.PrefixModel()
    wrapper = ns["HFModelWrapper"](model, use_torch_compile=False)
    with torch.no_grad():
        old = wrapper(
            batch["sequences"], batch.metadata["response_length"], batch["attention_mask"]
        )
    batch["action_log_probs"] = old
    batch["base_action_log_probs"] = old - 0.1
    batch["rollout_logprobs"] = old - 0.15
    batch["values"] = torch.arange(old.numel()).reshape(old.shape).double() / 10
    batch["rollout_expert_indices"] = None
    batch["advantages"] = ns["apply_loss_reduction_to_advantages_minibatch"](
        batch["advantages"], batch["loss_mask"], reduction, 2, 13, [(0, 2), (2, 4)]
    )
    before = {key: value.clone() if value is not None else None for key, value in batch.items()}
    collectives = []

    def all_reduce(tensor, op):
        assert op == "MAX" and tensor.device.type == "cpu"
        collectives.append(int(tensor))
        tensor.add_(padding)  # Explicit other-rank maximum fixture, not a real DP collective.

    monkeypatch.setitem(
        ns,
        "dist",
        Namespace(
            is_initialized=lambda: bool(padding),
            all_reduce=all_reduce,
            ReduceOp=Namespace(MAX="MAX"),
        ),
    )
    # Keep this CPU fixture CPU-only even on a GPU-capable test machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    iterator = ns["get_microbatch_iterator"](
        batch, micro_batch_size=2, max_tokens_per_microbatch=budget
    )
    pieces = list(iterator)
    dummy_count = padding if budget else 0
    assert len(collectives) == int(bool(budget and padding))
    real = pieces[:-dummy_count] if dummy_count else pieces
    order = [int(i) for piece in real for i in piece["row_ids"].flatten()]
    assert sorted(order) == [0, 1, 2, 3]
    if budget:
        assert order != [0, 1, 2, 3]
        assert iterator.num_padding_microbatches == dummy_count
        for piece in real:
            counts = piece["attention_mask"].sum(dim=1)
            assert counts.sum() <= budget or (len(piece) == 1 and counts.item() > budget)
        if budget == 10:
            assert real[0]["row_ids"].tolist() == [[2]]
            assert real[0]["attention_mask"].sum() == 13  # Oversized row is not truncated.
    for piece in real:
        ids = piece["row_ids"].flatten()
        for key, value in before.items():
            if value is None:
                assert piece[key] is None
            else:
                assert torch.equal(piece[key], value[ids]), key
        assert piece.metadata == {"response_length": 8}
    restored = iterator.reorder_and_combine_batches(pieces)
    for key, value in before.items():
        if value is None:
            assert restored[key] is None
        else:
            assert torch.equal(restored[key], value), key
            assert torch.equal(batch[key], value), key
    assert "is_padding_batch" not in batch.metadata

    def loss(piece):
        lp = wrapper(piece["sequences"], piece.metadata["response_length"], piece["attention_mask"])
        return ns["ppo_policy_loss"](
            lp,
            piece["action_log_probs"],
            piece["advantages"],
            policy_tests.algorithm("token"),
            piece["loss_mask"],
            piece["rollout_logprobs"],
        )[0]

    baseline = loss(batch)
    baseline.backward()
    reference_gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    total = 0.0
    for index, piece in enumerate(pieces):
        actual = loss(piece)
        total += actual.item()
        if index >= len(real):
            assert piece.metadata["is_padding_batch"]
            assert piece["attention_mask"].sum() == 1
            assert piece["loss_mask"].count_nonzero() == 0
            assert piece["advantages"].count_nonzero() > 0  # Native dummy values are NOT zero.
            assert actual.item() == 0 and torch.isfinite(actual)
            # The zero mask, not invented zero advantages, must neutralize the dummy.
            gradients = torch.autograd.grad(actual, tuple(model.parameters()), retain_graph=True)
            assert all(torch.count_nonzero(g) == 0 for g in gradients)
        actual.backward()
    assert total == pytest.approx(baseline.item(), rel=1e-12, abs=1e-12)
    for parameter, expected in zip(model.parameters(), reference_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-12, atol=1e-12)
