"""v2 session-sample postprocessor (``--session-sample-postprocessor-path``).

Launch flag::

    --session-sample-postprocessor-path llenvs.integrations.miles.postprocess.postprocess

Runs miles' ``default_postprocess`` unchanged (shared-prefix ownership
masking, agent-metadata merge, trajectory-reward assignment), then joins the
agent's per-turn ``reward_events`` to the session tree's node completion
spans and records the result on each sample:

- ``sample.metadata["decision_spans"]``: response-relative ``[start, end)``
  token spans, one per rewarded decision, in commit order along the leaf's
  path.
- ``sample.metadata["decision_rewards"]``: the matching per-decision rewards.

The Tier-2 miles fork lifts these into ``train_data`` at conversion time and
computes per-decision advantages rollout-side (``metadata`` is on the v2
samples wire; ``train_metadata`` is not). Without reward events (Tier-0
agents, plain RM runs) the join is a no-op, so the postprocessor is always
safe to install.

v2 postprocessor hooks are SYNC and run on the session-server event loop —
keep this fast and never raise (a hook exception becomes an HTTP 422).
"""

from __future__ import annotations

from typing import Any


def _default_postprocess() -> Any:
    """Resolve miles' default postprocessor. Module-level seam for tests."""
    from miles.rollout.session.v2.postprocessor_hub.default_postprocess import (
        default_postprocess,
    )

    return default_postprocess


def attach_decision_data(samples: list[Any], session_metadata: dict[str, Any]) -> list[Any]:
    """Join agent reward events to node completion spans, per leaf sample.

    Pure and import-free; tolerates missing agent/tree/leaf data as a no-op.
    Spans are recorded regardless of shared-prefix ownership — the loss mask
    already zeroes non-owned spans downstream.
    """
    agent_metadata = session_metadata.get("agent") or {}
    reward_events = agent_metadata.get("reward_events") or []
    tree = session_metadata.get("tree") or {}
    nodes_by_id = {node["id"]: node for node in tree.get("nodes", [])}
    if not reward_events or not nodes_by_id:
        return samples

    events_by_response_id = {
        event["response_id"]: event
        for event in reward_events
        if event.get("response_id") is not None
    }

    for sample in samples:
        leaf = (getattr(sample, "metadata", None) or {}).get("leaf")
        if not leaf:
            continue
        # Node spans index all tokens of the path; rebase to response space.
        response_start = len(sample.tokens) - sample.response_length
        spans: list[list[int]] = []
        rewards: list[float] = []
        for node_id in leaf.get("path_node_ids", []):
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            event = events_by_response_id.get(node.get("response_id"))
            if event is None:
                continue
            completion_start, completion_end = node["completion_span"]
            start = max(completion_start - response_start, 0)
            end = min(completion_end - response_start, sample.response_length)
            if end > start:
                spans.append([start, end])
                rewards.append(float(event["value"]))
        sample.metadata["decision_spans"] = spans
        sample.metadata["decision_rewards"] = rewards

    return samples


def postprocess(leaf_samples: list[Any], session_metadata: dict[str, Any]) -> list[Any]:
    """miles' default postprocessing plus per-decision span/reward attachment."""
    samples = _default_postprocess()(leaf_samples, session_metadata)
    return attach_decision_data(samples, session_metadata)
