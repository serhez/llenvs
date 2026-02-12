"""CLI entry point for running evaluations.

Usage:
    llenvs run config.yaml [--limit N] [--output-dir DIR]
    llenvs run config.yaml --environment leg_counting --limit 10
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from llenvs.core.config import (
    EvalConfig,
    EnvironmentFactory,
    BackendFactory,
    create_sampling_params,
)
from llenvs.evaluation.logging import LogConfig
from llenvs.evaluation.runner import run_evaluation
from llenvs.evaluation.results import (
    create_evaluation_result,
    print_summary,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llenvs",
        description="Run LLM evaluations with MDP-style environments",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks per environment",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Run only this environment (by name)",
    )
    run_parser.add_argument(
        "--no-detailed-results",
        action="store_true",
        help="Don't save per-episode results",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    run_parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Comma-separated log targets: console, file, wandb",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available environments")
    list_parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Optional config file to show environments from",
    )

    return parser


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    config = EvalConfig.from_yaml(config_path)

    # Apply command-line overrides
    if args.limit is not None:
        config.limit = args.limit
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_detailed_results:
        config.save_detailed_results = False

    # Filter environments if specified
    if args.environment:
        config.environments = [
            env for env in config.environments if env.name == args.environment
        ]
        if not config.environments:
            print(f"Error: Environment '{args.environment}' not found in config")
            return 1

    # Create output directory
    output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")
    print(f"Model: {config.model.model}")
    print(f"Backend: {config.model.backend}")
    print(f"Environments: {[env.name for env in config.environments]}")
    print()

    # Create backend
    try:
        backend = BackendFactory.create(config.model)
    except Exception as e:
        print(f"Error creating backend: {e}")
        return 1

    # Create sampling params
    sampling_params = create_sampling_params(config.inference)

    # Progress callback
    def progress_callback(current: int, total: int) -> None:
        if not args.quiet:
            pct = (current / total * 100) if total > 0 else 0
            print(f"\r  Progress: {current}/{total} ({pct:.1f}%)", end="", flush=True)

    # Log config
    log_config: LogConfig | None = None
    if args.log:
        targets = tuple(t.strip() for t in args.log.split(",") if t.strip())
        log_config = LogConfig(targets=targets)

    # Run each environment
    all_results = []
    for env_config in config.environments:
        print(f"\nRunning environment: {env_config.name}")

        start_time = datetime.now()

        # Create environment
        try:
            environment = EnvironmentFactory.create(env_config)
        except Exception as e:
            print(f"  Error creating environment: {e}")
            continue

        # Determine task indices
        env_size = len(environment)
        limit = min(config.limit or env_size, env_size)
        task_indices = list(range(limit))

        print(f"  Tasks: {len(task_indices)}")

        # Run evaluation
        try:
            batch_result = run_evaluation(
                environment=environment,
                backend=backend,
                task_indices=task_indices,
                sampling_params=sampling_params,
                system_prompt=config.system_prompt,
                progress_callback=progress_callback if not args.quiet else None,
                log=log_config,
            )
        except Exception as e:
            print(f"\n  Error running evaluation: {e}")
            continue

        if not args.quiet:
            print()  # Newline after progress

        # Create and save results
        eval_result = create_evaluation_result(
            batch_result=batch_result,
            model_name=config.model.model,
            environment_name=env_config.name,
            start_time=start_time,
            config=config.to_dict(),
            include_detailed_results=config.save_detailed_results,
        )

        # Save results
        result_path = run_dir / f"{env_config.name}_results.json"
        eval_result.save(result_path, include_results=config.save_detailed_results)
        print(f"  Results saved to: {result_path}")

        # Print summary
        print_summary(eval_result)

        all_results.append(eval_result)

    # Summary across all environments
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Overall Summary")
        print("=" * 60)
        for result in all_results:
            acc = result.metrics.get("accuracy")
            acc_str = f"{acc.value:.4f}" if acc else "N/A"
            print(f"  {result.metadata.environment}: accuracy={acc_str}")

    print(f"\nAll results saved to: {run_dir}")
    return 0


def list_command(args: argparse.Namespace) -> int:
    """Execute the list command."""
    if args.config:
        # List environments from config
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            return 1

        config = EvalConfig.from_yaml(config_path)
        print("Environments in configuration:")
        for env in config.environments:
            print(f"  - {env.name} (adapter: {env.adapter})")
    else:
        # List available adapters
        print("Available adapters:")
        print("  - reasoning_gym: Wraps reasoning-gym ProceduralDataset")
        print()
        print("Use 'llenvs list config.yaml' to list environments in a config file")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)
    elif args.command == "list":
        return list_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
