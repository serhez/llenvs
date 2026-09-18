"""Real identity construction against explicit OS/Git/package inventories."""

from types import SimpleNamespace as Namespace

import pytest

from llenvs.integrations.skyrl import _preparation as preparation
from llenvs.integrations.skyrl._preflight import runtime_controls


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    versions = {
        "ray": "2.57.0",
        "torch": "2.13.0+cu130",
        "vllm": "0.28.0",
        "transformers": "5.16.1",
        "omegaconf": "2.3.1",
    }
    files = {
        "RECORD": "fixture-wheel-inventory",
        "direct_url.json": '{"url":"https://fixture-secret@example.invalid/code"}',
    }
    distributions = [
        Namespace(metadata={"Name": name}, version=version, read_text=files.get)
        for name, version in versions.items()
    ]
    packages = Namespace(version=versions.__getitem__, distributions=lambda: distributions)
    monkeypatch.setattr(preparation, "importlib", Namespace(metadata=packages))
    platform = Namespace(system=lambda: "Linux", platform=lambda: "fixture-Linux-CUDA-host")
    monkeypatch.setattr(preparation, "platform", platform)
    python = Namespace(version_info=(3, 12), version="fixture-python", executable="/staged/python")
    monkeypatch.setattr(preparation, "sys", python)
    environment = {}
    monkeypatch.setattr(preparation, "os", Namespace(environ=environment))
    git = Namespace(revision=preparation.SKYRL_REVISION, returncode=0)
    calls = []

    def revision(args, **kwargs):
        calls.append(args)
        assert args[-2:] == ["rev-parse", "HEAD"]
        return git.revision

    def diff(args, **kwargs):
        calls.append(args)
        assert args[-3:] == ["skyrl", "uv.lock", "pyproject.toml"]
        return git

    monkeypatch.setattr(preparation, "subprocess", Namespace(check_output=revision, run=diff))
    root = tmp_path / "skyrl"
    (root / "skyrl").mkdir(parents=True)
    (root / "skyrl/fixture.py").write_text("# pinned-code fixture\n")
    return Namespace(
        root=root,
        versions=versions,
        files=files,
        distributions=distributions,
        environment=environment,
        git=git,
        calls=calls,
        platform=platform,
        python=python,
    )


def test_head_and_driver_identities_normalize_native_injection_without_secrets(inventory):
    case = inventory
    head = preparation.runtime_identity(case.root)
    case.environment.update(runtime_controls({}, driver=False))
    driver = preparation.runtime_identity(case.root, driver=True)
    assert head == driver
    assert "fixture-secret" not in repr(driver)
    assert driver["skyrl_revision"] == preparation.SKYRL_REVISION
    assert all(len(p[-1]) == len(p[-2]) == 64 for p in driver["packages"])


@pytest.mark.parametrize("change", ["source", "record", "direct_url", "extra_package", "control"])
def test_each_runtime_semantic_change_affects_identity(inventory, change):
    case = inventory
    baseline = preparation.runtime_identity(case.root)
    if change == "source":
        (case.root / "skyrl/fixture.py").write_text("# changed fixture\n")
    elif change in ("record", "direct_url"):
        case.files["RECORD" if change == "record" else "direct_url.json"] += "changed"
    elif change == "extra_package":
        case.distributions.append(
            Namespace(
                metadata={"Name": "external-scorer"}, version="fixed", read_text=lambda name: None
            )
        )
    else:
        case.environment["SKYRL_DISABLE_FA4"] = "1"
    assert preparation.runtime_identity(case.root) != baseline


def test_package_enumeration_order_and_generated_log_paths_do_not_change_identity(inventory):
    baseline = preparation.runtime_identity(inventory.root)
    inventory.distributions.reverse()
    inventory.environment["SKYRL_LOG_FILE"] = "/different/generated/log"
    assert preparation.runtime_identity(inventory.root) == baseline


@pytest.mark.parametrize(
    "damage",
    ["python", "os", "revision", "dirty", "version", "unknown_control", "missing_driver_injection"],
)
def test_incompatible_runtime_is_rejected(inventory, damage):
    case = inventory
    if damage == "python":
        case.python.version_info = (3, 13)
    elif damage == "os":
        case.platform.system = lambda: "Darwin"
    elif damage == "revision":
        case.git.revision = "different-pin"
    elif damage == "dirty":
        case.git.returncode = 1
    elif damage == "version":
        case.versions["vllm"] = "unreviewed"
    elif damage == "unknown_control":
        case.environment["VLLM_UNREVIEWED_PLUGIN"] = "private-value"
    with pytest.raises(ValueError) as error:
        preparation.runtime_identity(case.root, driver=damage == "missing_driver_injection")
    assert "private-value" not in str(error.value)
    if damage in ("python", "os", "unknown_control", "missing_driver_injection"):
        assert not case.calls
