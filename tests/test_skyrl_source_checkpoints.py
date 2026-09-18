"""Pinned async control flow and checkpoint IO, with CPU worker boundaries.

These are explicit upstream-bug reproducers, not resume acceptance. Generation
is deterministic/on-demand; no model, Ray scheduler, optimizer or GPU is used.
"""

import asyncio
import copy
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as Namespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.skyrl_source import definitions, driver_namespace

torch = pytest.importorskip("torch")


@dataclass
class CPUConfig:
    trainer: Namespace


class IndexedLoader:
    """Replayable six-task data fixture; native wrapper owns consumed IDs."""

    def __iter__(self):
        return iter([[{"uid": str(i)}] for i in range(6)])

    def __len__(self):
        return 6

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


@pytest.fixture
def trainer_factory(monkeypatch, tmp_path):
    ns = driver_namespace(monkeypatch)
    ns.update(
        asyncio=asyncio,
        dataclass=dataclass,
        asdict=asdict,
        Enum=Enum,
        os=os,
        Path=Path,
        Timer=lambda *args: nullcontext(),
        tqdm=Mock(),
        GLOBAL_STEP_PREFIX="global_step_",
        io=Namespace(
            makedirs=os.makedirs,
            open_file=open,
            exists=os.path.exists,
            isdir=os.path.isdir,
            list_dir=os.listdir,
        ),
    )
    definitions(
        "skyrl/train/utils/trainer_utils.py",
        {
            "ResumeMode",
            "extract_step_from_path",
            "list_checkpoint_dirs",
            "validate_consistency_for_latest_checkpoint",
        },
        ns,
    )
    definitions(
        "skyrl/train/trainer.py",
        {"save_checkpoints", "load_checkpoints"},
        ns,
        owner="RayPPOTrainer",
    )
    ns["RayPPOTrainer"] = type(
        "CheckpointBase", (), {key: ns[key] for key in ("save_checkpoints", "load_checkpoints")}
    )
    definitions(
        "skyrl/train/fully_async_trainer.py",
        {
            "GeneratedOutputGroup",
            "_RolloutStat",
            "_AsyncStalenessManager",
            "_AsyncDataloader",
            "FullyAsyncRayPPOTrainer",
        },
        ns,
    )

    class CPUTrainer(ns["FullyAsyncRayPPOTrainer"]):
        def __init__(self, name, *, maximum=None, epochs=1, resume=None):
            self.cfg = CPUConfig(
                Namespace(
                    ckpt_path=str(tmp_path / name),
                    ckpt_interval=1,
                    hf_save_interval=0,
                    eval_interval=0,
                    eval_before_train=False,
                    update_ref_every_epoch=False,
                    epochs=epochs,
                    max_training_steps=maximum,
                    resume_path=str(resume) if resume else None,
                )
            )
            self.resume_mode = ns["ResumeMode"].FROM_PATH if resume else ns["ResumeMode"].NONE
            self.train_dataloader = IndexedLoader()
            self.mini_batch_size, self.num_steps_per_epoch = 2, 3
            self.total_training_steps = min(3 * epochs, maximum) if maximum else 3 * epochs
            self.async_train_dataloader = ns["_AsyncDataloader"](self.train_dataloader, 2)
            self._staleness_manager = ns["_AsyncStalenessManager"](2, 2, 0)
            # Only generation/model/resource boundaries are replaced. The actual
            # loop, dataloader, staleness manager and checkpoint bodies execute.
            self.num_parallel_generation_workers = 0
            self._gen_buffer_maxsize, self.sample_full_batch = 2, False
            self._ray_gpu_monitor = self._vllm_metrics_scraper = None
            self._phase_gauge = Namespace(timed_phase=lambda *args: nullcontext())
            self._loop_gauges = Mock()
            self._profiler_start = self._profiler_stop = self._profiler_step = Mock()
            self.init_weight_sync_state = self._cleanup_old_checkpoints = Mock()
            self.tokenizer, self.has_critic, self.ref_model = None, False, None
            self.all_metrics, self.all_timings, self.updates, self.saves = {}, {}, [], []
            self.tracker = Mock()
            self.dispatch = Namespace(
                save_weights_for_sampler=AsyncMock(),
                get_timing_metrics=lambda: {},
                save_checkpoint=Mock(),
                load_checkpoint=Mock(),
                finalize_pending_saves=Mock(),
            )

        async def _collect_generation_mini_batch(self, buffer, done):
            groups = []
            for _ in range(self.mini_batch_size):
                prompts = await self.async_train_dataloader.get_next_non_consumed_data()
                assert prompts is not None, "native loop requested an exhausted task batch"
                await self._staleness_manager.acquire_submission_slot()
                await self._staleness_manager.on_rollout_accepted()
                groups.append(Namespace(uid=prompts[0]["uid"]))
            return groups, [], False

        def convert_generation_group_mini_batch_to_training_input(self, groups, dropped):
            return [group.uid for group in groups]

        async def _run_training(self, uids):
            self.updates.append((self.global_step, self.epoch, uids))
            return {}

        def save_checkpoints(self):
            path = super().save_checkpoints()
            self.saves.append(
                (self.global_step, self.epoch, self.async_train_dataloader.num_trained())
            )
            return path

    return CPUTrainer


def run(trainer):
    asyncio.run(asyncio.wait_for(trainer.train(), timeout=5))
    return trainer


def checkpoint(trainer, step):
    return Path(trainer.cfg.trainer.ckpt_path) / f"global_step_{step}"


@pytest.mark.parametrize("maximum", [None, 2, 3])
def test_native_periodic_and_final_saves_use_different_counter_semantics(trainer_factory, maximum):
    trainer = run(trainer_factory("fresh", maximum=maximum))
    steps = maximum or 3
    assert [step for step, _, _ in trainer.updates] == list(range(1, steps + 1))
    assert trainer.saves[:-1] == [(i, 0, 2 * i) for i in range(1, steps + 1)]
    # Known native bug: next-step counter is persisted as completed progress.
    assert trainer.saves[-1] == (
        steps + 1,
        1 if maximum is None else 0,
        0 if maximum is None else 2 * steps,
    )
    state = torch.load(checkpoint(trainer, steps + 1) / "trainer_state.pt", weights_only=False)
    assert state["global_step"] == steps + 1 != len(trainer.updates)
    assert (Path(trainer.cfg.trainer.ckpt_path) / "latest_ckpt_global_step.txt").read_text() == str(
        steps + 1
    )


@pytest.mark.parametrize(
    "step,expected", [(1, [(2, 0, ["2", "3"]), (3, 0, ["4", "5"])]), (2, [(3, 0, ["4", "5"])])]
)
def test_explicit_periodic_resume_consumes_exact_remaining_tasks(trainer_factory, step, expected):
    fresh = run(trainer_factory("fresh", maximum=2))
    resumed = run(trainer_factory("resumed", resume=checkpoint(fresh, step)))
    assert resumed.updates == expected
    assert resumed._staleness_manager._stat.accepted == 6
    assert resumed._staleness_manager._current_global_step == 4
    resumed.dispatch.load_checkpoint.assert_called_once_with(
        "policy",
        str(checkpoint(fresh, step) / "policy"),
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )


@pytest.mark.parametrize("latest", [False, True])
def test_final_checkpoint_resume_silently_skips_remaining_tasks(trainer_factory, latest):
    fresh = run(trainer_factory("fresh", maximum=2))
    resumed = trainer_factory("resumed", resume=checkpoint(fresh, 3))
    if latest:
        # Exercise the actual latest-marker resolver too, within this test's
        # owned directory. No connector manifest/resume policy is substituted.
        resumed.resume_mode = type(resumed.resume_mode).LATEST
        resumed.cfg.trainer.ckpt_path = fresh.cfg.trainer.ckpt_path
    run(resumed)
    assert fresh.updates == [(1, 0, ["0", "1"]), (2, 0, ["2", "3"])]
    assert resumed.updates == []  # Tasks 4 and 5 were never trained.
    assert resumed._staleness_manager._stat.accepted == 6  # Only four were real.
    assert resumed.saves == [(4, 1, 0)]


def test_epoch_final_resume_inflates_accepted_count_and_drops_next_epoch_tail(trainer_factory):
    fresh = run(trainer_factory("fresh"))
    resumed = run(trainer_factory("resumed", epochs=2, resume=checkpoint(fresh, 4)))
    assert resumed.updates == [(5, 1, ["0", "1"]), (6, 1, ["2", "3"])]
    assert resumed._staleness_manager._stat.accepted == 12
    assert len(fresh.updates + resumed.updates) * 2 == 10


def test_resume_at_step_limit_executes_an_extra_update_before_testing_limit(trainer_factory):
    fresh = run(trainer_factory("fresh", maximum=2))
    resumed = run(trainer_factory("resumed", maximum=2, resume=checkpoint(fresh, 2)))
    assert resumed.updates == [(3, 0, ["4", "5"])]
    assert resumed.global_step == 4


def test_consumed_and_filtered_ids_are_distinct_from_regenerated_work(trainer_factory):
    trainer = trainer_factory("state")
    loader = trainer.async_train_dataloader

    async def exercise():
        await loader.mark_consumed_uids(["0", "1"])
        await loader.mark_filtered_uids(["2"])
        # A generated but unconsumed task is deliberately not in a checkpoint.
        assert (await loader.get_next_non_consumed_data())[0]["uid"] == "3"
        saved = copy.deepcopy((loader.get_consumed_uids_list(), loader.get_filtered_uids_list()))
        restored = trainer_factory("restored").async_train_dataloader
        restored.load_state_from_checkpoint(set(saved[0]), set(saved[1]))
        assert restored.num_trained() == 2
        assert [(await restored.get_next_non_consumed_data())[0]["uid"] for _ in range(3)] == [
            "3",
            "4",
            "5",
        ]
        assert await restored.get_next_non_consumed_data() is None
        await restored.reset_at_epoch_end()
        assert restored.num_trained() == 0
        assert (await restored.get_next_non_consumed_data())[0]["uid"] == "0"

    asyncio.run(exercise())
