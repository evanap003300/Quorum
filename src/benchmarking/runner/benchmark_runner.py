"""Main benchmark runner orchestrator."""

from typing import List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from src.benchmarking.config.benchmark_config import BenchmarkConfig
from src.benchmarking.dataset.scibench_loader import SciBenchLoader, BenchmarkProblem
from src.benchmarking.runner.problem_executor import ProblemExecutor, ProblemResult
from src.benchmarking.evaluation.answer_comparator import AnswerComparator
from src.benchmarking.metrics.aggregator import MetricsAggregator, MetricsSummary
from src.benchmarking.storage.results_storage import ResultsStorage


class BenchmarkRunner:
    """Main orchestrator for running benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.config.validate()

        self.loader = SciBenchLoader()
        self.executor = ProblemExecutor(timeout_seconds=config.timeout_per_problem)
        self.comparator = AnswerComparator(
            tolerance=config.numeric_tolerance,
            allow_unit_conversion=config.allow_unit_conversion,
        )
        self.aggregator = MetricsAggregator()
        self.storage = ResultsStorage(
            output_dir=config.output_dir,
            run_name=config.run_name,
        )

    def run(self) -> Tuple[List[ProblemResult], MetricsSummary]:
        """Run complete benchmark.

        Returns:
            Tuple of (results, summary)
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("SCIBENCH BENCHMARK")
            print("=" * 60)

        # Load dataset
        if self.config.verbose:
            print("\nLoading SciBench dataset...")
        all_problems = self.loader.load_dataset()
        if self.config.verbose:
            print(f"✓ Loaded {len(all_problems)} problems")

        # Filter text-only
        if self.config.skip_image_problems:
            all_problems = self.loader.filter_text_only(all_problems)
            if self.config.verbose:
                print(f"✓ Filtered to {len(all_problems)} text-only problems")

        # Sample batch
        problems = self.loader.sample_problems(
            problems=all_problems,
            batch_size=self.config.batch_size,
            random_seed=self.config.random_seed,
            specific_ids=self.config.problem_ids,
        )
        if self.config.verbose:
            print(f"✓ Sampled {len(problems)} problems")

            # Print breakdown
            breakdown = self.loader.get_subject_breakdown(problems)
            print("\nBreakdown by source:")
            for source, count in sorted(breakdown.items()):
                print(f"  {source}: {count}")

        # Check for existing checkpoint
        checkpoint = self._load_checkpoint()
        completed_ids = set(checkpoint.get("completed_ids", [])) if checkpoint else set()

        if completed_ids and self.config.verbose:
            print(f"\n✓ Resuming from checkpoint ({len(completed_ids)} already completed)")

        # Execute problems
        if self.config.verbose:
            print(f"\nRunning benchmark...\n")

        results = checkpoint.get("results", []) if checkpoint else []
        result_objs = [ProblemResult(**r) for r in results]

        problems_to_run = [p for p in problems if p.problem_id not in completed_ids]

        for problem in tqdm(problems_to_run, desc="Benchmark", unit="problem"):
            # Execute problem
            result = self.executor.execute(problem)

            # Compare answer
            try:
                comparison = self.comparator.compare(
                    predicted=result.predicted_answer,
                    predicted_unit=result.predicted_unit or "",
                    ground_truth=result.ground_truth_answer,
                    ground_truth_unit=result.ground_truth_unit,
                )

                # Update result verdict and comparison details
                result.verdict = comparison.verdict
                result.comparison_reason = comparison.reason

            except Exception as e:
                # Catch comparison errors and log them
                result.verdict = "ERROR"
                if result.error_message:
                    result.error_message += f" [Comparison error: {str(e)[:100]}]"
                else:
                    result.error_message = f"Answer comparison failed: {str(e)[:100]}"
                if self.config.verbose:
                    tqdm.write(f"  WARNING: Comparison error for {problem.problem_id}: {str(e)[:80]}")

            result_objs.append(result)
            completed_ids.add(problem.problem_id)

            # Print progress with more detail
            correct_count = sum(1 for r in result_objs if r.verdict == "CORRECT")
            if self.config.verbose:
                # Format predicted and ground truth for display
                pred_display = f"{result.predicted_answer} {result.predicted_unit or ''}".strip()
                truth_display = f"{result.ground_truth_answer} {result.ground_truth_unit}".strip()

                status_msg = (
                    f"Problem {problem.problem_id}: {result.verdict} "
                    f"(Accuracy: {correct_count}/{len(result_objs)})\n"
                    f"  Predicted: {pred_display}\n"
                    f"  Expected:  {truth_display}"
                )

                # Add detailed comparison reason
                if result.comparison_reason:
                    status_msg += f"\n  Reason:    {result.comparison_reason[:120]}"

                # Add error details if applicable
                if result.verdict == "ERROR" and result.error_message:
                    status_msg += f"\n  Error: {result.error_message[:80]}"

                tqdm.write(status_msg)

            # Save checkpoint
            if len(result_objs) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(result_objs)

        # Final checkpoint
        self._save_checkpoint(result_objs)

        # Aggregate metrics
        summary = self.aggregator.aggregate(result_objs)

        # Save results
        if self.config.verbose:
            print("\nSaving results...")
        self.storage.save_results(result_objs, summary)

        if self.config.verbose:
            print(f"✓ Results saved to: {self.storage.get_run_dir()}")

            # Print summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total problems: {summary.total_problems}")
            print(f"Correct: {summary.correct} ({summary.accuracy*100:.1f}%)")
            print(f"Incorrect: {summary.incorrect}")
            print(f"Errors: {summary.errors}")
            print(f"Timeouts: {summary.timeouts}")

            # Print error breakdown if there are errors
            if summary.error_breakdown:
                print("\nError types:")
                for error_type, count in sorted(summary.error_breakdown.items()):
                    print(f"  {error_type}: {count}")
            print(f"\nTotal time: {summary.total_time:.1f}s ({summary.avg_time_per_problem:.1f}s per problem)")
            print(f"Total cost: ${summary.total_cost:.2f} (${summary.avg_cost_per_problem:.3f} per problem)")
            print("\n" + "=" * 60 + "\n")

        return result_objs, summary

    def _load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint if it exists.

        Returns:
            Checkpoint dictionary or None
        """
        checkpoint_path = self.storage.get_checkpoint_path()
        if Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Could not load checkpoint: {e}")
        return None

    def _save_checkpoint(self, results: List[ProblemResult]) -> None:
        """Save checkpoint.

        Args:
            results: Current results list
        """
        checkpoint = {
            "completed_ids": [r.problem_id for r in results],
            "results": [json.loads(r.json()) for r in results],
        }

        checkpoint_path = self.storage.get_checkpoint_path()
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
