"""Results storage and persistence module."""

from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
import csv

from src.benchmarking.runner.problem_executor import ProblemResult
from src.benchmarking.metrics.aggregator import MetricsSummary
from src.benchmarking.config.benchmark_config import BenchmarkConfig


class ResultsStorage:
    """Manage storage and persistence of benchmark results."""

    def __init__(self, output_dir: str = "benchmark_results", run_name: Optional[str] = None):
        """Initialize storage manager.

        Args:
            output_dir: Base output directory
            run_name: Run name (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.output_dir / self.run_name

    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return str(self.run_dir)

    def get_checkpoint_path(self) -> str:
        """Get checkpoint file path."""
        return str(self.run_dir / "checkpoint.json")

    def save_results(
        self,
        results: List[ProblemResult],
        summary: MetricsSummary,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        """Save all results to disk.

        Args:
            results: List of problem results
            summary: Aggregated summary
            config: Optional configuration to save
        """
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        self._save_detailed_results(results)

        # Save summary
        self._save_summary(summary)

        # Save config if provided
        if config:
            self._save_config(config)

        # Save CSV for plotting
        self._save_csv(results)

    def _save_detailed_results(self, results: List[ProblemResult]) -> None:
        """Save detailed results as JSON.

        Args:
            results: List of problem results
        """
        results_data = [json.loads(r.json()) for r in results]

        results_path = self.run_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

    def _save_summary(self, summary: MetricsSummary) -> None:
        """Save summary statistics as JSON.

        Args:
            summary: Aggregated summary
        """
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(json.loads(summary.json()), f, indent=2)

    def _save_config(self, config: BenchmarkConfig) -> None:
        """Save configuration.

        Args:
            config: Benchmark configuration
        """
        config_path = self.run_dir / "config.yaml"
        config.save_yaml(str(config_path))

    def _save_csv(self, results: List[ProblemResult]) -> None:
        """Export results as CSV.

        Args:
            results: List of problem results
        """
        csv_path = self.run_dir / "results.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "problem_id",
                "verdict",
                "ground_truth_answer",
                "ground_truth_unit",
                "predicted_answer",
                "predicted_unit",
                "total_time",
                "total_cost",
                "error_message",
            ])

            # Rows
            for result in results:
                writer.writerow([
                    result.problem_id,
                    result.verdict,
                    result.ground_truth_answer,
                    result.ground_truth_unit,
                    result.predicted_answer or "",
                    result.predicted_unit or "",
                    f"{result.total_time:.2f}",
                    f"{result.total_cost:.4f}",
                    result.error_message or "",
                ])

    def list_runs(self) -> List[str]:
        """List all available benchmark runs.

        Returns:
            List of run names
        """
        if not self.output_dir.exists():
            return []

        runs = sorted([d.name for d in self.output_dir.iterdir() if d.is_dir()])
        return runs

    def load_results(self, run_name: str) -> Tuple[List[ProblemResult], MetricsSummary]:
        """Load previous benchmark results.

        Args:
            run_name: Name of run to load

        Returns:
            Tuple of (results, summary)

        Raises:
            FileNotFoundError: If run not found
        """
        run_dir = self.output_dir / run_name

        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load detailed results
        results_path = run_dir / "detailed_results.json"
        with open(results_path, 'r') as f:
            results_data = json.load(f)

        results = [ProblemResult(**r) for r in results_data]

        # Load summary
        summary_path = run_dir / "summary.json"
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

        summary = MetricsSummary(**summary_data)

        return results, summary
