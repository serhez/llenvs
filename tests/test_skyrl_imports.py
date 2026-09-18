"""Fresh-process import isolation, including installed optional dependencies."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module", ["llenvs", "llenvs.integrations.skyrl.data", "llenvs.integrations.skyrl.scoring"]
)
def test_import_does_not_probe_optional_training_stacks(module):
    code = textwrap.dedent(f"""
        import importlib, sys
        attempted = []
        blocked = {{"datasets", "torch", "ray", "skyrl", "transformers", "vllm"}}
        class Guard:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    attempted.append(fullname)
                    raise ImportError("unexpected optional dependency: " + fullname)
        sys.meta_path.insert(0, Guard())
        importlib.import_module({module!r})
        assert not attempted, attempted
        assert not (blocked & sys.modules.keys())
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_selecting_one_adapter_does_not_probe_others():
    code = textwrap.dedent("""
        import llenvs.adapters as adapters
        from llenvs.core.registry import environment_registry
        adapters.GymnasiumAdapter._get_gymnasium = lambda self: object()
        def unexpected(self):
            raise AssertionError("unrelated HuggingFace probe")
        adapters.HuggingFaceAdapter._get_datasets_library = unexpected
        assert environment_registry.get_adapter("gymnasium").name == "gymnasium"
        import sys
        assert "datasets" not in sys.modules
        assert "torch" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
