"""CLI entrypoint: ``python -m llenvs.container --config '...' --port 8080``."""

import argparse
import logging

from llenvs.container.server import run_server_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an environment server.")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string with EnvironmentConfig fields.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_server_from_config(args.config, port=args.port)


if __name__ == "__main__":
    main()
