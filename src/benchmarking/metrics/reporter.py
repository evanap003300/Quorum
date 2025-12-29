"""Report generation for benchmark results."""

from typing import List, Tuple
from pathlib import Path
from datetime import datetime

from src.benchmarking.metrics.aggregator import MetricsSummary, SourceMetrics
from src.benchmarking.runner.problem_executor import ProblemResult
from src.benchmarking.config.benchmark_config import BenchmarkConfig


class Reporter:
    """Generate human-readable reports from benchmark results."""

    def generate_markdown_report(
        self,
        summary: MetricsSummary,
        config: BenchmarkConfig,
        results: List[ProblemResult],
        output_path: str,
    ) -> None:
        """Generate comprehensive Markdown report.

        Args:
            summary: Aggregated metrics summary
            config: Benchmark configuration
            results: List of individual results
            output_path: Path to save report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# SciBench Benchmark Report\n",
            f"**Generated**: {timestamp}\n",
            f"**Run Name**: {config.run_name}\n",
            "\n",
        ]

        # Configuration summary
        lines.extend(self._format_config_section(config))

        # Overall results
        lines.extend(self._format_overall_section(summary))

        # Performance statistics
        lines.extend(self._format_performance_section(summary))

        # Cost statistics
        lines.extend(self._format_cost_section(summary))

        # Breakdown by source
        if summary.by_source:
            lines.extend(self._format_source_breakdown_section(summary.by_source))

        # Error analysis
        if summary.error_breakdown:
            lines.extend(self._format_error_analysis_section(summary))

        # Detailed results (top issues)
        lines.extend(self._format_detailed_results_section(results))

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(lines)

    def _format_config_section(self, config: BenchmarkConfig) -> List[str]:
        """Format configuration section.

        Args:
            config: Benchmark configuration

        Returns:
            List of formatted lines
        """
        return [
            "## Configuration\n",
            "\n",
            "| Parameter | Value |\n",
            "|-----------|-------|\n",
            f"| Batch Size | {config.batch_size} |\n",
            f"| Timeout | {config.timeout_per_problem}s |\n",
            f"| Numeric Tolerance | {config.numeric_tolerance*100:.1f}% |\n",
            f"| Skip Images | {config.skip_image_problems} |\n",
            "\n",
        ]

    def _format_overall_section(self, summary: MetricsSummary) -> List[str]:
        """Format overall results section.

        Args:
            summary: Metrics summary

        Returns:
            List of formatted lines
        """
        return [
            "## Overall Results\n",
            "\n",
            "| Metric | Count | Percentage |\n",
            "|--------|-------|------------|\n",
            f"| Total Problems | {summary.total_problems} | 100% |\n",
            f"| Correct | {summary.correct} | {summary.accuracy*100:.1f}% |\n",
            f"| Incorrect | {summary.incorrect} | {summary.incorrect/max(summary.total_problems, 1)*100:.1f}% |\n",
            f"| Errors | {summary.errors} | {summary.errors/max(summary.total_problems, 1)*100:.1f}% |\n",
            f"| Timeouts | {summary.timeouts} | {summary.timeouts/max(summary.total_problems, 1)*100:.1f}% |\n",
            "\n",
        ]

    def _format_performance_section(self, summary: MetricsSummary) -> List[str]:
        """Format performance statistics section.

        Args:
            summary: Metrics summary

        Returns:
            List of formatted lines
        """
        lines = [
            "## Performance Statistics\n",
            "\n",
            "| Metric | Value |\n",
            "|--------|-------|\n",
            f"| Total Time | {summary.total_time:.1f}s |\n",
            f"| Avg Time per Problem | {summary.avg_time_per_problem:.2f}s |\n",
            f"| Median Time | {summary.median_time_per_problem:.2f}s |\n",
        ]

        for percentile, value in sorted(summary.time_percentiles.items()):
            lines.append(f"| {percentile.upper()} | {value:.2f}s |\n")

        lines.append("\n")
        return lines

    def _format_cost_section(self, summary: MetricsSummary) -> List[str]:
        """Format cost statistics section.

        Args:
            summary: Metrics summary

        Returns:
            List of formatted lines
        """
        lines = [
            "## Cost Statistics\n",
            "\n",
            "| Metric | Value |\n",
            "|--------|-------|\n",
            f"| Total Cost | ${summary.total_cost:.2f} |\n",
            f"| Avg Cost per Problem | ${summary.avg_cost_per_problem:.4f} |\n",
            f"| Median Cost | ${summary.median_cost_per_problem:.4f} |\n",
        ]

        for percentile, value in sorted(summary.cost_percentiles.items()):
            lines.append(f"| {percentile.upper()} | ${value:.4f} |\n")

        lines.append("\n")
        return lines

    def _format_source_breakdown_section(self, by_source: dict) -> List[str]:
        """Format breakdown by source section.

        Args:
            by_source: Dictionary of SourceMetrics

        Returns:
            List of formatted lines
        """
        lines = [
            "## Breakdown by Source\n",
            "\n",
            "| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |\n",
            "|--------|-------|---------|----------|----------|----------|\n",
        ]

        for source_name, metrics in sorted(by_source.items()):
            lines.append(
                f"| {source_name} | {metrics.total} | {metrics.correct} | "
                f"{metrics.accuracy*100:.1f}% | {metrics.avg_time:.2f}s | "
                f"${metrics.avg_cost:.4f} |\n"
            )

        lines.append("\n")
        return lines

    def _format_error_analysis_section(self, summary: MetricsSummary) -> List[str]:
        """Format error analysis section.

        Args:
            summary: Metrics summary

        Returns:
            List of formatted lines
        """
        lines = [
            "## Error Analysis\n",
            "\n",
            "| Error Type | Count |\n",
            "|------------|-------|\n",
        ]

        for error_type, count in sorted(summary.error_breakdown.items()):
            lines.append(f"| {error_type} | {count} |\n")

        lines.append("\n")
        return lines

    def _format_detailed_results_section(self, results: List[ProblemResult]) -> List[str]:
        """Format detailed results section (failures only).

        Args:
            results: List of problem results

        Returns:
            List of formatted lines
        """
        failures = [r for r in results if r.verdict != "CORRECT"]

        if not failures:
            return [
                "## Results\n",
                "\n",
                "✓ All problems solved correctly!\n",
                "\n",
            ]

        lines = [
            f"## Failed Problems ({len(failures)} failures)\n",
            "\n",
        ]

        # Show first 10 failures
        for result in failures[:10]:
            lines.append(f"### {result.problem_id}\n")
            lines.append(f"**Verdict**: {result.verdict}\n")
            if result.error_message:
                lines.append(f"**Error**: {result.error_message}\n")
            lines.append(f"**Time**: {result.total_time:.2f}s | **Cost**: ${result.total_cost:.4f}\n")
            lines.append("\n")

        if len(failures) > 10:
            lines.append(f"... and {len(failures) - 10} more failures\n\n")

        return lines

    def generate_comparison_report(
        self,
        runs: List[Tuple[str, MetricsSummary]],
        output_path: str,
    ) -> None:
        """Generate comparison report for multiple runs.

        Args:
            runs: List of (run_name, summary) tuples
            output_path: Path to save report
        """
        lines = [
            "# SciBench Benchmark Comparison\n",
            "\n",
            "| Run | Accuracy | Avg Time | Total Cost | Problems |\n",
            "|-----|----------|----------|------------|----------|\n",
        ]

        for run_name, summary in runs:
            lines.append(
                f"| {run_name} | {summary.accuracy*100:.1f}% | "
                f"{summary.avg_time_per_problem:.2f}s | ${summary.total_cost:.2f} | "
                f"{summary.total_problems} |\n"
            )

        lines.append("\n")

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(lines)
