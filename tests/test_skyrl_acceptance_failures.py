"""Fault injection against the exact helpers used by GPU acceptance, on CPU."""

import asyncio
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace as Namespace

import pytest

from tests import test_skyrl_source_policy as policy_tests
from tests.skyrl_acceptance_run import AuditTrainerMixin
from tests.test_skyrl_credit import row

torch = pytest.importorskip("torch")
native_policy = policy_tests.native_policy


@pytest.mark.parametrize(
    "damage",
    [
        "row_order",
        "advantages",
        "returns",
        "attribution",
        "probability",
        "nan_probability",
        "nan_source_probability",
        "loss_mask",
        "attention_mask",
        "response_mask",
        "consistent_reward_corruption",
    ],
)
def test_audit_rejects_corrupted_transport_and_keeps_failed_input(native_policy, tmp_path, damage):
    batch, prompts, responses = policy_tests.credit_batch(native_policy)
    batch["rollout_logprobs"] = torch.where(batch["loss_mask"].bool(), -0.5, 0.0)
    spans = [[(0, 2), (4, 7)], [(0, 2)], [(0, 1), (4, 8)], [(0, 1)]]
    output = dict(
        prompt_token_ids=prompts,
        response_ids=responses,
        rewards=[batch["rewards"][i, -len(r) :].tolist() for i, r in enumerate(responses)],
        loss_masks=[batch["loss_mask"][i, -len(r) :].tolist() for i, r in enumerate(responses)],
        rollout_logprobs=[
            batch["rollout_logprobs"][i, -len(r) :].tolist() for i, r in enumerate(responses)
        ],
        env_metrics=[
            {"llenvs/attribution": row(str(i // 2), i % 2, len(r), spans[i])}
            for i, r in enumerate(responses)
        ],
    )
    if damage == "attribution":
        output["env_metrics"][0]["llenvs/attribution"]["sampled_spans"][0]["start"] += 1
    elif damage == "nan_source_probability":
        output["rollout_logprobs"][0][-1] = float("nan")
    recorded = copy.deepcopy(output)

    class Native:
        def convert_to_training_input(self, output, uids):
            if damage == "row_order":
                batch["sequences"] = batch["sequences"][[1, 0, 2, 3]]
            elif damage in ("advantages", "returns", "response_mask"):
                batch[damage] = batch[damage].roll(1, dims=1)
            elif damage in ("probability", "nan_probability"):
                batch["rollout_logprobs"][0, -1] = 0.1 if damage == "probability" else float("nan")
            elif damage == "loss_mask":
                batch["loss_mask"][0, -1] = 0
            elif damage == "attention_mask":
                batch["attention_mask"][0, -len(responses[0]) - 1] = 0
            elif damage == "consistent_reward_corruption":
                batch["rewards"][0, -1] += 1
                for key in ("advantages", "returns"):
                    batch[key][0] += batch["loss_mask"][0]
            return batch

        async def train(self):
            self.convert_to_training_input(output, ["a", "a", "b", "b"])

    class Trainer(AuditTrainerMixin, Native):
        pass

    trainer = Trainer()
    trainer.global_step = 1
    trainer.cfg = Namespace(
        trainer=Namespace(
            log_path=tmp_path,
            algorithm=Namespace(
                advantage_estimator="llenvs_token_rtg", gamma=1.0, grpo_norm_by_std=False
            ),
        ),
        llenvs=Namespace(turn_weighting="uniform"),
    )
    with pytest.raises(AssertionError):
        asyncio.run(trainer.train())
    assert output == recorded
    report = json.loads((tmp_path / "llenvs-acceptance.json").read_text())
    assert report["completed"] is False and report["batches"] == []
    assert report["failure"]["stage"] == "conversion"
    assert report["failure"]["type"] == "AssertionError"
    assert report["pending_batch"]["responses"] == responses
    assert report["pending_batch"]["uids"] == ["a", "a", "b", "b"]
    if damage == "nan_source_probability":
        assert report["pending_batch"]["rollout_logprobs"][0][-1] == "nonfinite:nan"


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "exit",
        "timeout",
        "missing",
        "incomplete",
        "malformed",
        "nonfinite",
        "duplicate",
        "truthy",
        "failure",
    ],
)
def test_owned_subprocess_failures_are_visible_and_keep_logs(monkeypatch, tmp_path, case):
    from tests.skyrl_acceptance_checks import training_process

    real_popen, processes = subprocess.Popen, []
    monkeypatch.setenv("LLENVS_SKYRL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("RAY_ADDRESS", "ray://must-not-attach")
    content = {
        "success": '{"completed": true}',
        "incomplete": '{"completed": false}',
        "malformed": "{",
        "nonfinite": '{"completed": true, "value": NaN}',
        "duplicate": '{"completed": false, "completed": true}',
        "truthy": '{"completed": "true"}',
        "failure": '{"completed": true, "failure": {"stage": "conversion"}}',
    }.get(case)
    if content:
        path = tmp_path / "fresh/logs/llenvs-acceptance.json"
        path.parent.mkdir(parents=True)
        path.write_text(content)

    def child(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "tests.skyrl_acceptance_run"]
        assert command[3:5] == ["sync", "grpo"]
        assert kwargs["env"]["RAY_ADDRESS"] == "local"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["start_new_session"] is True
        code = "import os, signal; print('fixture child diagnostic', flush=True); " + (
            "signal.pause()"
            if case == "timeout"
            else "os._exit(7)"
            if case == "exit"
            else "os._exit(0)"
        )
        process = real_popen([sys.executable, "-u", "-c", code], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", child)
    if case == "success":
        assert training_process(tmp_path, "sync", "grpo", 1, resume=False) == {"completed": True}
    else:
        with pytest.raises((AssertionError, pytest.fail.Exception), match="fresh.log"):
            training_process(tmp_path, "sync", "grpo", 1, resume=False)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert "fixture child diagnostic" in (tmp_path / "fresh.log").read_text()
    assert os.environ["RAY_ADDRESS"] == "ray://must-not-attach"


@pytest.fixture
def smoke_artifact(tmp_path):
    root = tmp_path / "fresh/checkpoints"
    for step in (1, 2):
        checkpoint = root / f"global_step_{step}"
        (checkpoint / "policy").mkdir(parents=True)
        torch.save({"fixture": True}, checkpoint / "policy/fixture.pt")
        torch.save({}, checkpoint / "data.pt")
        torch.save({"global_step": step}, checkpoint / "trainer_state.pt")
        torch.save(
            {"consumed_uids": ["a", "b"], "filtered_uids": [], "epoch": 0},
            checkpoint / "fully_async_state.pt",
        )
    (root / "latest_ckpt_global_step.txt").write_text("2")
    result = dict(
        completed=True,
        batches=[
            dict(
                global_step=1,
                uids=["a", "a", "b", "b"],
                policy_metrics={"loss": 0.5},
                diagnostics=[
                    {"llenvs/end_reason": "terminated", "llenvs/reward_components": [1, 2]}
                ],
            )
        ],
        evaluation_steps=[1],
        final_counter=2,
    )
    return tmp_path, root, result


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "policy",
        "data",
        "trainer",
        "async",
        "latest_data",
        "latest_counter",
        "evaluation",
        "metrics",
        "nan_metrics",
        "turns",
        "consumed",
        "unreadable",
    ],
)
def test_smoke_progress_rejects_missing_or_corrupt_evidence(smoke_artifact, damage):
    from tests.skyrl_acceptance_checks import check_smoke_progress

    artifact, root, result = smoke_artifact
    files = {
        "policy": "policy/fixture.pt",
        "data": "data.pt",
        "trainer": "trainer_state.pt",
        "async": "fully_async_state.pt",
    }
    if damage in files:
        (root / "global_step_1" / files[damage]).unlink()
    elif damage == "latest_data":
        (root / "global_step_2/data.pt").unlink()
    elif damage == "latest_counter":
        torch.save({"global_step": 99}, root / "global_step_2/trainer_state.pt")
    elif damage == "unreadable":
        (root / "global_step_1/data.pt").write_bytes(b"not a checkpoint")
    elif damage == "evaluation":
        result["evaluation_steps"] = []
    elif damage in ("metrics", "nan_metrics"):
        result["batches"][0]["policy_metrics"] = (
            {} if damage == "metrics" else {"loss": float("nan")}
        )
    elif damage == "turns":
        result["batches"][0]["diagnostics"][0]["llenvs/reward_components"] = [1]
    elif damage == "consumed":
        torch.save(
            {"consumed_uids": ["a"], "filtered_uids": [], "epoch": 0},
            root / "global_step_1/fully_async_state.pt",
        )
    if damage is None:
        check_smoke_progress(artifact, "fresh", result, [1], asynchronous=True)
    else:
        with pytest.raises(AssertionError):
            check_smoke_progress(artifact, "fresh", result, [1], asynchronous=True)
