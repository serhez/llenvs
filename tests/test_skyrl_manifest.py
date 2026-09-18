"""Run identity is a compatibility check, never a second checkpoint engine."""

import copy
import importlib
import json

import pytest


@pytest.fixture
def manifest():
    return importlib.import_module("llenvs.integrations.skyrl._manifest")


@pytest.fixture
def identity(manifest):
    return manifest.run_identity(
        datasets={"train": [{"task_id": "a", "prompt": "private task"}], "eval": []},
        environment={"seed": 42, "api_key": "environment-secret"},
        rewards={"estimator": "llenvs_token_rtg", "judge": "fixed"},
        execution={"seed": 42, "batch_size": 4},
        models={"policy": "model-content-hash"},
        runtime={"skyrl": "source-hash", "torch": "version"},
    )


def test_identity_is_owned_deterministic_and_contains_no_task_or_secret_values(manifest):
    parts = dict(
        datasets={"train": [{"task_id": "a", "prompt": "private task"}], "eval": []},
        environment={"seed": 42, "api_key": "environment-secret"},
        rewards={"estimator": "llenvs_token_rtg"},
        execution={"seed": 42},
        models={"policy": "model-hash"},
        runtime={"torch": "version"},
    )
    result = manifest.run_identity(**parts)
    assert result == manifest.run_identity(**copy.deepcopy(parts))
    assert "private task" not in json.dumps(result)
    assert "environment-secret" not in json.dumps(result)
    parts["environment"]["seed"] = 1
    assert result != manifest.run_identity(**parts)


@pytest.mark.parametrize(
    "section", ["datasets", "environment", "rewards", "execution", "models", "runtime"]
)
def test_each_semantic_section_changes_identity(manifest, section):
    parts = {
        key: {"value": 1}
        for key in ("datasets", "environment", "rewards", "execution", "models", "runtime")
    }
    original = manifest.run_identity(**parts)
    parts[section]["value"] = 2
    changed = manifest.run_identity(**parts)
    assert changed[section] != original[section]
    assert all(changed[key] == original[key] for key in original if key != section)


def test_ordered_dataset_hash_does_not_sort_tasks(manifest):
    assert manifest.content_hash(["a", "b"]) != manifest.content_hash(["b", "a"])
    assert manifest.content_hash({"a": 1, "b": 2}) == manifest.content_hash({"b": 2, "a": 1})


def test_fresh_manifest_is_exclusive_and_latest_resume_is_read_only(manifest, identity, tmp_path):
    root = tmp_path / "run"
    manifest.check_run(root, identity, resume_mode="none", write=True)
    path = root / "llenvs-run.json"
    original = path.read_bytes()
    assert json.loads(original) == identity
    with pytest.raises(FileExistsError):
        manifest.check_run(root, identity, resume_mode="none", write=True)
    manifest.check_run(root, identity, resume_mode="latest", write=True)
    assert path.read_bytes() == original


def test_check_only_never_creates_directories_or_manifest(manifest, identity, tmp_path):
    root = tmp_path / "absent" / "run"
    manifest.check_run(root, identity, resume_mode="none", write=False)
    manifest.check_run(root, identity, resume_mode="latest", write=False)
    assert not root.parent.exists()


def test_latest_does_not_adopt_an_unidentified_existing_run(manifest, identity, tmp_path):
    (tmp_path / "global_step_1").mkdir()
    with pytest.raises(ValueError, match="manifest"):
        manifest.check_run(tmp_path, identity, resume_mode="latest", write=False)
    assert not (tmp_path / "llenvs-run.json").exists()


def test_changed_resume_reports_section_names_without_values(manifest, identity, tmp_path):
    manifest.check_run(tmp_path, identity, resume_mode="none", write=True)
    changed = dict(identity, rewards="f" * 64)
    with pytest.raises(ValueError, match="rewards") as error:
        manifest.check_run(tmp_path, changed, resume_mode="latest", write=True)
    assert identity["rewards"] not in str(error.value)
    assert json.loads((tmp_path / "llenvs-run.json").read_text()) == identity


def test_from_path_checks_source_and_fresh_destination(manifest, identity, tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    manifest.check_run(source, identity, resume_mode="none", write=True)
    checkpoint = source / "global_step_7"
    checkpoint.mkdir()
    manifest.check_run(
        destination, identity, resume_mode="from_path", resume_path=checkpoint, write=True
    )
    assert json.loads((destination / "llenvs-run.json").read_text()) == identity
    with pytest.raises(ValueError, match="rewards"):
        manifest.check_run(
            tmp_path / "other",
            dict(identity, rewards="a" * 64),
            resume_mode="from_path",
            resume_path=checkpoint,
            write=True,
        )
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("value", ["global_step_bad", "policy", "global_step_-1"])
def test_from_path_requires_native_step_directory(manifest, identity, tmp_path, value):
    checkpoint = tmp_path / value
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="global_step"):
        manifest.check_run(
            tmp_path / "destination",
            identity,
            resume_mode="from_path",
            resume_path=checkpoint,
            write=False,
        )


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema_version": 1}',
        '{"schema_version": NaN}',
        "[]",
        '{"schema_version": 1, "schema_version": 2}',
    ],
)
def test_corrupt_manifest_is_never_replaced(manifest, identity, tmp_path, contents):
    path = tmp_path / "llenvs-run.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="manifest"):
        manifest.check_run(tmp_path, identity, resume_mode="latest", write=True)
    assert path.read_text() == contents


def test_tree_identity_uses_content_not_mtime_and_detects_added_files(manifest, tmp_path):
    (tmp_path / "config.json").write_text('{"vocab_size": 100}')
    first = manifest.tree_hash(tmp_path)
    (tmp_path / "config.json").touch()
    assert manifest.tree_hash(tmp_path) == first
    (tmp_path / "weights.safetensors").write_bytes(b"fixture weights")
    assert manifest.tree_hash(tmp_path) != first


def test_manifest_rejects_symlink_and_non_local_roots(manifest, identity, tmp_path):
    target = tmp_path / "target"
    target.write_text("do not overwrite")
    (tmp_path / "llenvs-run.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        manifest.check_run(tmp_path, identity, resume_mode="latest", write=True)
    assert target.read_text() == "do not overwrite"
    with pytest.raises(ValueError, match="absolute"):
        manifest.check_run("s3://bucket/run", identity, resume_mode="latest", write=False)
