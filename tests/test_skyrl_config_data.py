"""Task-export and dataset contracts without SkyRL, Ray, or a policy backend."""

import copy
import importlib
import json
import pickle
import re
from base64 import b64encode
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from llenvs.core.config import BackendFactory, EnvironmentFactory, EvalConfig
from llenvs.core.environment import EnvironmentSpec
from llenvs.core.state import ImageContent, Observation, ObservationContent, State, StateMetadata
from llenvs.core.tools import ToolDefinition


@pytest.fixture
def data():
    return importlib.import_module("llenvs.integrations.skyrl.data")


@pytest.fixture
def config():
    return EvalConfig.from_dict(
        {
            "environments": [{"name": "indexed", "adapter": "test", "size": 5, "seed": 42}],
            "system_prompt": "Be precise.",
        }
    )


class IndexedEnvironment:
    """Finite environment fixture; reset exposes deliberately sensitive metadata."""

    spec = EnvironmentSpec(name="indexed", adapter="test", max_steps=1)
    prompts = {}
    available_tools = ()

    def __init__(self, options):
        self.options = options
        self.reset = Mock(side_effect=self._reset)
        self.close = Mock()

    def __len__(self):
        return 5

    def _reset(self, *, options, seed=None):
        index = options["task_index"]
        if index == self.options.get("fail_index"):
            raise RuntimeError("fixture reset failed")
        observation = self.options.get("observation") or Observation(prompt=f"Task {index}")
        return State(
            observation=observation,
            hidden=SimpleNamespace(expected_answer="PRIVATE_ANSWER"),
            metadata=StateMetadata(step=0, episode_id="volatile-session-id"),
        ), {"task_index": index, "secret": "PRIVATE_CREDENTIAL", "live": object()}


@pytest.fixture
def environments(monkeypatch):
    created = []
    options = {}

    def create(_config):
        env = IndexedEnvironment(options)
        created.append(env)
        return env

    factory = Mock(side_effect=create)
    monkeypatch.setattr(EnvironmentFactory, "create", factory)
    monkeypatch.setattr(
        BackendFactory, "create", Mock(side_effect=AssertionError("unexpected model"))
    )
    return SimpleNamespace(created=created, options=options, factory=factory)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def png_data(color):
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1, 1), color).save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


def export(data, config, tmp_path, name="tasks.jsonl", **kwargs):
    path = tmp_path / name
    count = data.export_prompt_data(config, path, **kwargs)
    rows = read_rows(path)
    assert count == len(rows)
    return path, rows


def test_export_schema_order_seed_and_owned_environment_cleanup(
    data, config, environments, tmp_path
):
    path, rows = export(data, config, tmp_path, indices=[3, 1])
    assert [r["llenvs"]["task_index"] for r in rows] == [3, 1]
    for row in rows:
        assert set(row) == {"prompt", "env_class", "data_source", "llenvs"}
        assert row["env_class"] == "llenvs"
        assert row["data_source"] == "test/indexed"
        assert set(row["llenvs"]) == {
            "schema_version",
            "env_fingerprint",
            "task_index",
            "task_id",
            "initial_fingerprint",
        }
        assert row["llenvs"]["schema_version"] == 1
        for key in ("env_fingerprint", "task_id", "initial_fingerprint"):
            assert re.fullmatch("[0-9a-f]{64}", row["llenvs"][key])
    assert rows[0]["prompt"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Task 3"},
    ]
    assert "PRIVATE_" not in path.read_text()
    assert "volatile-session-id" not in path.read_text()
    assert environments.factory.call_args.args[0].seed == 42
    assert [c for env in environments.created for c in env.reset.call_args_list] == [
        call(options={"task_index": 3}),
        call(options={"task_index": 1}),
    ]
    for env in environments.created:
        env.close.assert_called_once_with()


def test_identical_export_is_stable_but_constructor_seed_changes_task_identity(
    data, config, environments, tmp_path
):
    first_path, first = export(data, config, tmp_path, "first.jsonl", indices=[2])
    second_path, second = export(data, copy.deepcopy(config), tmp_path, "second.jsonl", indices=[2])
    assert first_path.read_bytes() == second_path.read_bytes()
    config.environments[0].seed = 43
    _, changed = export(data, config, tmp_path, "changed.jsonl", indices=[2])
    assert first[0]["llenvs"]["task_id"] != changed[0]["llenvs"]["task_id"]
    assert first[0]["llenvs"]["initial_fingerprint"] == changed[0]["llenvs"]["initial_fingerprint"]
    assert first == second


def test_unused_policy_runner_settings_do_not_change_task_identity(
    data, config, environments, tmp_path
):
    _, first = export(data, config, tmp_path, "first.jsonl", indices=[0])
    config.model.model = "unused-policy"
    config.limit = 1
    config.batch_size = 99
    config.inference.temperature = 0.7
    _, second = export(data, config, tmp_path, "second.jsonl", indices=[0])
    assert second == first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indices": []},
        {"indices": [1, 1]},
        {"indices": [-1]},
        {"indices": [5]},
        {"indices": [True]},
        {"num_tasks": 0},
        {"num_tasks": -1},
        {"num_tasks": 6},
        {"num_tasks": 10**20},
        {"indices": [0], "num_tasks": 1},
    ],
)
def test_bad_task_selections_are_rejected_without_publishing_partial_data(
    data, config, environments, tmp_path, kwargs
):
    path = tmp_path / "tasks.jsonl"
    with pytest.raises(ValueError):
        data.export_prompt_data(config, path, **kwargs)
    assert not path.exists()
    for env in environments.created:
        env.close.assert_called_once_with()


def test_export_never_overwrites_existing_data(data, config, environments, tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("existing user data\n")
    with pytest.raises(FileExistsError):
        data.export_prompt_data(config, path)
    assert path.read_text() == "existing user data\n"
    environments.factory.assert_not_called()


def test_reset_failure_cleans_up_and_does_not_publish_partial_data(
    data, config, environments, tmp_path
):
    environments.options["fail_index"] = 1
    with pytest.raises(RuntimeError, match="fixture reset failed"):
        data.export_prompt_data(config, tmp_path / "tasks.jsonl", indices=[0, 1])
    assert not (tmp_path / "tasks.jsonl").exists()
    for env in environments.created:
        env.close.assert_called_once_with()


def test_environment_selection_is_explicit_when_ambiguous(data, config, environments, tmp_path):
    second = copy.deepcopy(config.environments[0])
    second.name = "other"
    config.environments.append(second)
    with pytest.raises(ValueError, match="indexed.*other|other.*indexed"):
        data.export_prompt_data(config, tmp_path / "ambiguous.jsonl")
    environments.factory.assert_not_called()
    _, rows = export(data, config, tmp_path, env_name="other", indices=[0])
    assert rows[0]["data_source"] == "test/other"


def test_initial_history_keeps_roles_and_does_not_mutate_shared_messages(
    data, config, environments, tmp_path
):
    history = (
        {"role": "assistant", "content": "Earlier reply"},
        {"role": "user", "content": "Feedback"},
    )
    environments.options["observation"] = Observation(prompt="Task", messages=history)
    before = copy.deepcopy(history)
    _, rows = export(data, config, tmp_path, indices=[0, 1])
    assert rows[0]["prompt"][1:] == [{"role": "user", "content": "Task"}, *history]
    rows[0]["prompt"][-1]["content"] = "changed"
    assert rows[1]["prompt"][-1]["content"] == "Feedback"
    assert history == before


def test_unknown_history_role_is_not_coerced_to_user(data, config, environments, tmp_path):
    environments.options["observation"] = Observation(
        prompt="Task", messages=({"role": "typo", "content": "x"},)
    )
    with pytest.raises(ValueError, match="role"):
        data.export_prompt_data(config, tmp_path / "tasks.jsonl")


def test_task_then_state_image_order_and_tool_preamble_are_fingerprinted(
    data, config, environments, tmp_path
):
    # Tiny inline PNG fixtures; no renderer or image decoder is being certified.
    task_png, state_png = png_data("black"), png_data("white")
    observation = Observation(
        prompt="Task",
        task=ObservationContent(images=(ImageContent(task_png, source="task"),)),
        state=ObservationContent(images=(ImageContent(state_png, source="state"),)),
        available_tools=(ToolDefinition(name="inspect", description="Inspect the scene"),),
    )
    environments.options["observation"] = observation
    _, rows = export(data, config, tmp_path, indices=[0])
    prompt = rows[0]["prompt"]
    assert "inspect" in prompt[0]["content"]
    images = [
        part["image_url"]["url"]
        for message in prompt
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert images == [f"data:image/png;base64,{task_png}", f"data:image/png;base64,{state_png}"]
    changes = [
        replace(observation, task=observation.state, state=observation.task),
        replace(observation, available_tools=(ToolDefinition(name="other", description="Other"),)),
    ]
    for index, changed_observation in enumerate(changes):
        environments.options["observation"] = changed_observation
        _, changed = export(data, config, tmp_path, f"changed-{index}.jsonl", indices=[0])
        assert (
            rows[0]["llenvs"]["initial_fingerprint"] != changed[0]["llenvs"]["initial_fingerprint"]
        )


@pytest.mark.parametrize("case", ["mime", "base64"])
def test_invalid_inline_images_are_not_silently_dropped(data, config, environments, tmp_path, case):
    image = (
        ImageContent(png_data("black"), media_type="video/mp4")
        if case == "mime"
        else ImageContent("not base64!")
    )
    environments.options["observation"] = Observation(
        prompt="Task", task=ObservationContent(images=(image,))
    )
    with pytest.raises(ValueError, match="image|MIME|base64|media"):
        data.export_prompt_data(config, tmp_path / "tasks.jsonl")


def test_fresh_reset_messages_must_match_prepared_identity(data, config, environments, tmp_path):
    _, rows = export(data, config, tmp_path, indices=[0])
    messages = copy.deepcopy(rows[0]["prompt"])
    data.validate_initial_messages(rows[0], messages)
    messages[-1]["content"] = "a different task after reset"
    with pytest.raises(ValueError, match="initial|fingerprint|reset"):
        data.validate_initial_messages(rows[0], messages)


def test_dataset_preserves_stable_uid_copies_rows_and_is_pickleable(
    data, config, environments, tmp_path
):
    path, rows = export(data, config, tmp_path, indices=[3, 1])
    fingerprint = rows[0]["llenvs"]["env_fingerprint"]
    dataset = data.LlenvsPromptDataset([path], env_fingerprint=fingerprint)
    assert len(dataset) == 2
    messages, env_class, extras, uid = dataset[0]
    assert uid == rows[0]["llenvs"]["task_id"]
    assert env_class == "llenvs"
    assert extras["llenvs"]["task_index"] == 3
    messages[0]["content"] = "mutated"
    extras["llenvs"]["task_index"] = 99
    assert dataset[0][0] == rows[0]["prompt"]
    assert dataset[0][2]["llenvs"]["task_index"] == 3
    assert pickle.loads(pickle.dumps(dataset))[0] == dataset[0]
    collated = dataset.collate_fn([dataset[0], dataset[1]])
    assert [item["uid"] for item in collated] == [r["llenvs"]["task_id"] for r in rows]
    assert set(collated[0]) == {"prompt", "env_class", "env_extras", "uid"}


def test_duplicate_task_identity_across_input_files_is_rejected(
    data, config, environments, tmp_path
):
    first, rows = export(data, config, tmp_path, "first.jsonl", indices=[0])
    second, _ = export(data, config, tmp_path, "second.jsonl", indices=[0])
    with pytest.raises(ValueError, match="duplicate|task_id"):
        data.LlenvsPromptDataset(
            [first, second], env_fingerprint=rows[0]["llenvs"]["env_fingerprint"]
        )


@pytest.mark.parametrize(
    "case", ["duplicate", "fingerprint", "schema", "prompt_tampering", "extra_field"]
)
def test_dataset_rejects_invalid_rows_instead_of_filtering(
    data, config, environments, tmp_path, case
):
    path, rows = export(data, config, tmp_path, indices=[0])
    fingerprint = rows[0]["llenvs"]["env_fingerprint"]
    if case == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "fingerprint":
        fingerprint = "f" * 64
    elif case == "schema":
        rows[0]["llenvs"]["schema_version"] = 2
    elif case == "prompt_tampering":
        rows[0]["prompt"][-1]["content"] = "another task"
    else:
        rows[0]["llenvs_config"] = "/untrusted/override.yaml"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        data.LlenvsPromptDataset([path], env_fingerprint=fingerprint)


@pytest.mark.parametrize("count, expected", [(None, [2, 3, 4]), (2, [2, 3])])
def test_cli_loads_yaml_and_exports_a_range_once(data, environments, tmp_path, count, expected):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "environments:\n  - name: indexed\n    adapter: test\n    size: 5\n    seed: 42\n"
    )
    output = tmp_path / "tasks.jsonl"
    args = ["--config", str(config_path), "--output", str(output), "--start", "2"]
    if count is not None:
        args.extend(["--num-tasks", str(count)])
    assert data.main(args) == 0
    assert [row["llenvs"]["task_index"] for row in read_rows(output)] == expected
    environments.factory.assert_called_once()
    environments.created[0].close.assert_called_once_with()


@pytest.mark.parametrize(
    "args", [["--start", "-1"], ["--start", "5"], ["--start", "4", "--num-tasks", "2"]]
)
def test_cli_rejects_invalid_ranges_without_output(data, environments, tmp_path, args):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("environments:\n  - name: indexed\n    adapter: test\n    size: 5\n")
    output = tmp_path / "tasks.jsonl"
    with pytest.raises(ValueError):
        data.main(["--config", str(config_path), "--output", str(output), *args])
    assert not output.exists()
    for env in environments.created:
        env.close.assert_called_once_with()


def test_task_splits_preserve_order_and_refuse_overlap_or_dropped_tail(
    data, config, environments, tmp_path
):
    train_path, rows = export(data, config, tmp_path, "train.jsonl", indices=[3, 1])
    eval_path, _ = export(data, config, tmp_path, "eval.jsonl", indices=[0])
    fingerprint = rows[0]["llenvs"]["env_fingerprint"]
    train = data.LlenvsPromptDataset([train_path], env_fingerprint=fingerprint)
    evaluation = data.LlenvsPromptDataset([eval_path], env_fingerprint=fingerprint)
    original = train[0]
    data.validate_task_splits(train, evaluation, train_batch_size=2)
    data.validate_task_splits(train, None, train_batch_size=1)
    assert train[0] == original
    with pytest.raises(ValueError, match="overlap"):
        data.validate_task_splits(train, train, train_batch_size=2)
    for batch_size in (0, True, 3):
        with pytest.raises(ValueError, match="batch|divisible"):
            data.validate_task_splits(train, evaluation, train_batch_size=batch_size)
