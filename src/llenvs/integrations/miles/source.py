"""Custom rollout data source (``--data-source-path``).

Launch flags::

    --data-source-path llenvs.integrations.miles.source.LLEnvsDataSource
    --disable-rollout-global-dataset

Serves environment tasks straight from the llenvs ``DatasetProvider``
(discovered via ``LLENVS_MILES_CONFIG``) instead of a pre-exported JSONL
file, so the task set and the episodes always come from the same config.
Rows are built with ``data.build_task_row`` — identical shape to the
exporter's output.

Duck-typed against miles' ``DataSource`` contract (no miles base-class
import at module scope): ``get_samples(num) -> list[list[Sample]]`` returning
groups of exactly ``args.n_samples_per_prompt`` deepcopies, buffer drained
first, seeded per-epoch shuffle, epoch wraparound, and JSON save/load state.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from typing import Any

from llenvs.integrations.miles import config as miles_config
from llenvs.integrations.miles.data import build_task_row

logger = logging.getLogger(__name__)


def _sample_cls() -> Any:
    """Resolve miles' Sample class. Module-level seam for tests."""
    from miles.utils.types import Sample

    return Sample


class LLEnvsDataSource:
    """Rollout data source backed by an llenvs environment's task set."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.buffer: list[list[Any]] = []

        provider = miles_config.get_dataset_provider(None)
        system_prompt = miles_config.resolve_system_prompt_for(None)
        self._origin_rows = [
            build_task_row(item, system_prompt=system_prompt) for item in provider.get_items()
        ]
        self._rows = list(self._origin_rows)
        self._shuffled_epoch = -1
        if self.args.rollout_shuffle:
            self._shuffle(self.epoch_id)

    # -- internal -----------------------------------------------------------

    def _shuffle(self, epoch_id: int) -> None:
        if self._shuffled_epoch == epoch_id:
            return
        rng = random.Random(self.args.rollout_seed + epoch_id)
        permutation = list(range(len(self._origin_rows)))
        rng.shuffle(permutation)
        self._rows = [self._origin_rows[i] for i in permutation]
        self._shuffled_epoch = epoch_id

    def _next_rows(self, num_rows: int) -> list[dict[str, Any]]:
        if self.sample_offset + num_rows <= len(self._rows):
            rows = self._rows[self.sample_offset : self.sample_offset + num_rows]
            self.sample_offset += num_rows
        else:
            rows = self._rows[self.sample_offset :]
            num_rows -= len(rows)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self._shuffle(self.epoch_id)
            rows = rows + self._rows[:num_rows]
            self.sample_offset = num_rows
        return rows

    def _make_group(self, row: dict[str, Any]) -> list[Any]:
        sample_cls = _sample_cls()
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = sample_cls(
                prompt=copy.deepcopy(row["prompt"]),
                label=row["label"],
                metadata=copy.deepcopy(row["metadata"]),
            )
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            self.sample_index += 1
            group.append(sample)
        self.sample_group_index += 1
        return group

    # -- miles DataSource contract -------------------------------------------

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        """Return ``num_samples`` groups, draining the buffer first."""
        groups: list[list[Any]] = []
        while self.buffer and len(groups) < num_samples:
            groups.append(self.buffer.pop(0))
        groups.extend(self._make_group(row) for row in self._next_rows(num_samples - len(groups)))
        return groups

    def add_samples(self, samples: list[list[Any]]) -> None:
        """Queue partially processed groups for the next draw."""
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        for group in samples:
            assert isinstance(group, list), f"groups must be lists, got {type(group)}"
            assert len(group) == self.args.n_samples_per_prompt, (
                f"group size must equal n_samples_per_prompt, "
                f"got {len(group)} != {self.args.n_samples_per_prompt}"
            )
            self.buffer.append(group)

    def get_buffer_length(self) -> int:
        return len(self.buffer)

    def _state_path(self, root: str, rollout_id: Any) -> str:
        return os.path.join(root, f"rollout/llenvs_data_source_{rollout_id}.json")

    def save(self, rollout_id: Any) -> None:
        path = self._state_path(self.args.save, rollout_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "sample_offset": self.sample_offset,
                    "epoch_id": self.epoch_id,
                    "sample_group_index": self.sample_group_index,
                    "sample_index": self.sample_index,
                },
                f,
            )

    def load(self, rollout_id: Any = None) -> None:
        load_root = getattr(self.args, "load", None)
        if load_root is None:
            return
        path = self._state_path(load_root, rollout_id)
        if not os.path.exists(path):
            logger.info("Data-source checkpoint %s does not exist.", path)
            return
        with open(path) as f:
            state = json.load(f)
        self.sample_offset = state.get("sample_offset", 0)
        self.epoch_id = state.get("epoch_id", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        if self.args.rollout_shuffle:
            self._shuffle(self.epoch_id)
