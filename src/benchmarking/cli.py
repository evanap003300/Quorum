"""Command-line interface for benchmarking system."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.benchmarking.config.benchmark_config import BenchmarkConfig
from src.benchmarking.runner.benchmark_runner import BenchmarkRunner
from src.benchmarking.storage.results_storage import ResultsStorage
from src.benchmarking.metrics.reporter import Reporter


def cmd_run(args) -> int:
    """Run a benchmark."""
    try:
        # Load or create config
        if args.config:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = BenchmarkConfig.from_yaml(args.config)
            else:
                config = BenchmarkConfig.from_json(args.config)
        else:
            # Create from command-line arguments
            config = BenchmarkConfig(
                batch_size=args.batch_size,
                random_seed=args.seed,
                output_dir=args.output_dir,
                timeout_per_problem=args.timeout,
            )

        # Override with command-line args if provided
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.seed is not None:
            config.random_seed = args.seed

        # Run benchmark
        runner = BenchmarkRunner(config)
        results, summary = runner.run()

        # Generate report
        reporter = Reporter()
        report_path = str(Path(runner.storage.run_dir) / "report.md")
        reporter.generate_markdown_report(summary, config, results, report_path)

        print(f"\n✓ Report saved to: {report_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_resume(args) -> int:
    """Resume a benchmarkfrom checkpoint."""
    try:
        storage = ResultsStorage(output_dir=args.output_dir, run_name=args.run_name)

        # Load config if available
        config_path = Path(storage.run_dir) / "config.yaml"
        if config_path.exists():
            config = BenchmarkConfig.from_yaml(str(config_path))
        else:
            print(f"Warning: Config not found at {config_path}", file=sys.stderr)
            return 1

        # Run benchmark (will load checkpoint)
        runner = BenchmarkRunner(config)
        results, summary = runner.run()

        # Generate report
        reporter = Reporter()
        report_path = str(Path(runner.storage.run_dir) / "report.md")
        reporter.generate_markdown_report(summary, config, results, report_path)

        print(f"\n✓ Resumed and completed. Report: {report_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_report(args) -> int:
    """Generate report from saved results."""
    try:
        storage = ResultsStorage(output_dir=args.output_dir, run_name=args.run_name)
        results, summary = storage.load_results(args.run_name)

        # Load config if available
        config_path = Path(storage.run_dir) / "config.yaml"
        if config_path.exists():
            config = BenchmarkConfig.from_yaml(str(config_path))
        else:
            config = BenchmarkConfig()

        # Generate report
        reporter = Reporter()
        report_path = str(Path(storage.run_dir) / "report.md")
        reporter.generate_markdown_report(summary, config, results, report_path)

        print(f"✓ Report generated: {report_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args) -> int:
    """List available benchmark runs."""
    try:
        storage = ResultsStorage(output_dir=args.output_dir)
        runs = storage.list_runs()

        if not runs:
            print("No benchmark runs found.")
            return 0

        print(f"Found {len(runs)} benchmark runs:\n")
        for run in runs:
            run_dir = Path(args.output_dir) / run
            config_path = run_dir / "config.yaml"
            status = "✓" if config_path.exists() else "?"
            print(f"{status} {run}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_compare(args) -> int:
    """Compare multiple benchmark runs."""
    try:
        storage = ResultsStorage(output_dir=args.output_dir)

        runs = []
        for run_name in args.runs:
            try:
                results, summary = storage.load_results(run_name)
                runs.append((run_name, summary))
            except FileNotFoundError:
                print(f"Warning: Run '{run_name}' not found", file=sys.stderr)

        if not runs:
            print("No valid runs to compare.", file=sys.stderr)
            return 1

        # Generate comparison report
        reporter = Reporter()
        output_path = args.output or "comparison.md"
        reporter.generate_comparison_report(runs, output_path)

        print(f"✓ Comparison report saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SciBench benchmarking system for evaluating problem-solving accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config (10 problems)
  python -m src.benchmarking.cli run

  # Run with specific batch size
  python -m src.benchmarking.cli run --batch-size 50 --seed 42

  # Run with config file
  python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml

  # Resume interrupted run
  python -m src.benchmarking.cli resume --run-name 2025-12-29_14-30-00

  # List all runs
  python -m src.benchmarking.cli list

  # Compare runs
  python -m src.benchmarking.cli compare run1 run2 run3
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument("--config", help="Config file (YAML or JSON)")
    run_parser.add_argument("--batch-size", type=int, help="Number of problems to evaluate")
    run_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    run_parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem (seconds)")
    run_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    run_parser.set_defaults(func=cmd_run)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume benchmark from checkpoint")
    resume_parser.add_argument("--run-name", required=True, help="Run name to resume")
    resume_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    resume_parser.set_defaults(func=cmd_resume)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from saved results")
    report_parser.add_argument("--run-name", required=True, help="Run name to generate report for")
    report_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    report_parser.set_defaults(func=cmd_report)

    # List command
    list_parser = subparsers.add_parser("list", help="List available runs")
    list_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    list_parser.set_defaults(func=cmd_list)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument("runs", nargs="+", help="Run names to compare")
    compare_parser.add_argument("--output", help="Output file for comparison report")
    compare_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    compare_parser.set_defaults(func=cmd_compare)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
