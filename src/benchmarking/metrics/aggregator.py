"""Metrics aggregation module."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np

from src.benchmarking.runner.problem_executor import ProblemResult


class SourceMetrics(BaseModel):
    """Metrics for a specific source/subject."""

    source_name: str
    total: int
    correct: int
    incorrect: int
    errors: int
    timeouts: int
    accuracy: float

    avg_cost: float
    median_cost: float

    avg_time: float
    median_time: float


class MetricsSummary(BaseModel):
    """Aggregated metrics for benchmark run."""

    # Overall metrics
    total_problems: int
    correct: int
    incorrect: int
    errors: int
    timeouts: int
    accuracy: float

    # Timing statistics
    total_time: float
    avg_time_per_problem: float
    median_time_per_problem: float
    time_percentiles: Dict[str, float] = Field(default_factory=dict)

    # Cost statistics
    total_cost: float
    avg_cost_per_problem: float
    median_cost_per_problem: float
    cost_percentiles: Dict[str, float] = Field(default_factory=dict)

    # Breakdown by source
    by_source: Dict[str, SourceMetrics] = Field(default_factory=dict)

    # Error analysis
    error_breakdown: Dict[str, int] = Field(default_factory=dict)


class MetricsAggregator:
    """Calculate metrics from benchmark results."""

    def aggregate(self, results: List[ProblemResult]) -> MetricsSummary:
        """Calculate all metrics from results.

        Args:
            results: List of problem results

        Returns:
            MetricsSummary with all aggregated metrics
        """
        if not results:
            return MetricsSummary(
                total_problems=0,
                correct=0,
                incorrect=0,
                errors=0,
                timeouts=0,
                accuracy=0.0,
                total_time=0.0,
                avg_time_per_problem=0.0,
                median_time_per_problem=0.0,
                total_cost=0.0,
                avg_cost_per_problem=0.0,
                median_cost_per_problem=0.0,
            )

        # Count verdicts
        correct = sum(1 for r in results if r.verdict == "CORRECT")
        incorrect = sum(1 for r in results if r.verdict == "INCORRECT")
        errors = sum(1 for r in results if r.verdict == "ERROR")
        timeouts = sum(1 for r in results if r.verdict == "TIMEOUT")

        # Accuracy
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        # Timing statistics
        times = [r.total_time for r in results if r.total_time]
        total_time = sum(times)
        avg_time = np.mean(times) if times else 0.0
        median_time = float(np.median(times)) if times else 0.0
        time_percentiles = self._calculate_percentiles(times) if times else {}

        # Cost statistics
        costs = [r.total_cost for r in results if r.total_cost]
        total_cost = sum(costs)
        avg_cost = np.mean(costs) if costs else 0.0
        median_cost = float(np.median(costs)) if costs else 0.0
        cost_percentiles = self._calculate_percentiles(costs) if costs else {}

        # Breakdown by source
        by_source = self._breakdown_by_source(results)

        # Error breakdown
        error_breakdown = self._breakdown_errors(results)

        return MetricsSummary(
            total_problems=total,
            correct=correct,
            incorrect=incorrect,
            errors=errors,
            timeouts=timeouts,
            accuracy=accuracy,
            total_time=total_time,
            avg_time_per_problem=avg_time,
            median_time_per_problem=median_time,
            time_percentiles=time_percentiles,
            total_cost=total_cost,
            avg_cost_per_problem=avg_cost,
            median_cost_per_problem=median_cost,
            cost_percentiles=cost_percentiles,
            by_source=by_source,
            error_breakdown=error_breakdown,
        )

    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles.

        Args:
            values: List of numeric values

        Returns:
            Dictionary of percentile values
        """
        if not values:
            return {}

        return {
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    def _breakdown_by_source(self, results: List[ProblemResult]) -> Dict[str, SourceMetrics]:
        """Group results by source and calculate metrics.

        Args:
            results: List of results to group

        Returns:
            Dictionary mapping source name to SourceMetrics
        """
        # Group by source
        sources: Dict[str, List[ProblemResult]] = {}
        for result in results:
            # Extract source from problem_id or use default
            source = result.ground_truth_unit or "unknown"
            # Better heuristic: try to get from problem_id
            if "_" in result.problem_id:
                parts = result.problem_id.split("_")
                source = parts[0]

            if source not in sources:
                sources[source] = []
            sources[source].append(result)

        # Calculate metrics per source
        by_source = {}
        for source, source_results in sorted(sources.items()):
            correct = sum(1 for r in source_results if r.verdict == "CORRECT")
            incorrect = sum(1 for r in source_results if r.verdict == "INCORRECT")
            errors = sum(1 for r in source_results if r.verdict == "ERROR")
            timeouts = sum(1 for r in source_results if r.verdict == "TIMEOUT")
            total = len(source_results)

            times = [r.total_time for r in source_results if r.total_time]
            costs = [r.total_cost for r in source_results if r.total_cost]

            by_source[source] = SourceMetrics(
                source_name=source,
                total=total,
                correct=correct,
                incorrect=incorrect,
                errors=errors,
                timeouts=timeouts,
                accuracy=correct / total if total > 0 else 0.0,
                avg_cost=np.mean(costs) if costs else 0.0,
                median_cost=float(np.median(costs)) if costs else 0.0,
                avg_time=np.mean(times) if times else 0.0,
                median_time=float(np.median(times)) if times else 0.0,
            )

        return by_source

    def _breakdown_errors(self, results: List[ProblemResult]) -> Dict[str, int]:
        """Count errors by type.

        Args:
            results: List of results

        Returns:
            Dictionary mapping error type to count
        """
        error_counts: Dict[str, int] = {}

        for result in results:
            if result.error_type:
                error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1

        return error_counts
