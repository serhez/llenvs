"""Prompt-data JSONL exporter for miles (``--prompt-data`` / ``--eval-prompt-data``).

Exports environment tasks as one JSON object per line::

    {"prompt": [<chat messages>], "label": "<ground truth or ''>",
     "metadata": {"task_index": <int>, ...}}

``prompt`` is always a chat-messages list (the TITO session path requires
lists; miles passes them through untouched). ``metadata.task_index`` is what
the agent function and reward model use to reset the environment to the row's
task.

CLI::

    python -m llenvs.integrations.miles.data --config cfg.yaml --output tasks.jsonl \\
        [--num-tasks N] [--env NAME]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from llenvs.core.config import EnvironmentFactory, EvalConfig
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs.integrations.miles.config import resolve_config_system_prompt, select_environment

logger = logging.getLogger(__name__)


def build_task_row(item: TaskItem, *, system_prompt: str | None = None) -> dict[str, Any]:
    """Convert a TaskItem into a miles prompt-data row."""
    if item.images:
        raise ValueError(
            f"Task {item.task_index} has images; the miles TITO session path is "
            "text-only, so image tasks cannot be exported."
        )
    if item.messages:
        prompt: list[dict[str, Any]] = [dict(m) for m in item.messages]
    else:
        prompt = [{"role": "user", "content": item.prompt}]
    if system_prompt is not None:
        prompt.insert(0, {"role": "system", "content": system_prompt})

    metadata: dict[str, Any] = {"task_index": item.task_index}
    for key, value in item.metadata.items():
        # episode_id is a fresh UUID per reset — dropping it keeps exports
        # deterministic for the same config.
        if key in ("task_index", "episode_id"):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            logger.warning(
                "Dropping non-JSON-serializable metadata key %r on task %d",
                key,
                item.task_index,
            )
            continue
        metadata[key] = value

    return {"prompt": prompt, "label": item.ground_truth or "", "metadata": metadata}


def export_prompt_data(
    config: EvalConfig | str | Path,
    output_path: str | Path,
    *,
    num_tasks: int | None = None,
    indices: list[int] | None = None,
    env_name: str | None = None,
) -> int:
    """Export environment tasks to a miles prompt-data JSONL file.

    Args:
        config: EvalConfig instance or path to its YAML.
        output_path: Where to write the JSONL file.
        num_tasks: Cap on the number of tasks (from the start, or of ``indices``).
        indices: Specific task indices to export. None means all tasks.
        env_name: Environment to export when the config defines several.

    Returns:
        Number of rows written.
    """
    cfg = config if isinstance(config, EvalConfig) else EvalConfig.from_yaml(config)
    env_cfg = select_environment(cfg, env_name)
    provider = DatasetProvider(EnvironmentFactory.create(env_cfg))
    system_prompt = resolve_config_system_prompt(cfg, env_cfg)

    if indices is None:
        count = len(provider)
        if num_tasks is not None:
            count = min(count, num_tasks)
        indices = list(range(count))
    elif num_tasks is not None:
        indices = indices[:num_tasks]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in provider.get_items(indices):
            f.write(json.dumps(build_task_row(item, system_prompt=system_prompt)) + "\n")
    return len(indices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export llenvs environment tasks as miles prompt-data JSONL."
    )
    parser.add_argument("--config", required=True, help="Path to the llenvs EvalConfig YAML.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--num-tasks", type=int, default=None, help="Cap on exported tasks.")
    parser.add_argument(
        "--env", default=None, help="Environment name when the config defines several."
    )
    args = parser.parse_args(argv)
    count = export_prompt_data(
        args.config, args.output, num_tasks=args.num_tasks, env_name=args.env
    )
    print(f"Wrote {count} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
