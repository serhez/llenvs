"""Launch the bounded llenvs text recipe inside an installed native SkyRL stack."""

import importlib
import sys


def main(args: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if args is None else args
    if arguments in (["--help"], ["-h"]):
        print(
            "Usage: python -m llenvs.integrations.skyrl.entrypoint key.path=value ...\n"
            "Native SkyRL dotlist configuration plus llenvs.config=/absolute/config.yaml.\n"
            "llenvs.check_only=true validates without Ray/model allocation or checkpoint writes.\n"
            "Requires the staged, pinned Linux/Python 3.12 SkyRL environment and local models.\n"
            "See docs/guides/skyrl.md for the execution restrictions and acceptance status."
        )
        return
    try:
        native = importlib.import_module("llenvs.integrations.skyrl._native")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SkyRL training requires its separately staged native environment; "
            "task preparation does not. See docs/guides/skyrl.md."
        ) from error
    cfg = native.LlenvsSkyRLTrainConfig.from_cli_overrides(arguments)
    prepared = native.prepare(cfg)
    if cfg.llenvs.check_only:
        print(
            "Static checks passed: no Ray initialization, model allocation, scoring, or checkpoint writes."
        )
        return
    native.launch(cfg, prepared.identity)


if __name__ == "__main__":
    main()
