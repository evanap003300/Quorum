"""Main benchmark runner orchestrator."""

from typing import List, Optional, Tuple
from pathlib import Path
import json
import asyncio
import time
from dataclasses import dataclass
from tqdm import tqdm

from src.benchmarking.config.benchmark_config import BenchmarkConfig
from src.benchmarking.dataset.scibench_loader import SciBenchLoader, BenchmarkProblem
from src.benchmarking.runner.problem_executor import ProblemExecutor, ProblemResult
from src.benchmarking.evaluation.answer_comparator import AnswerComparator
from src.benchmarking.metrics.aggregator import MetricsAggregator, MetricsSummary
from src.benchmarking.storage.results_storage import ResultsStorage


@dataclass
class LogMessage:
    """Message for logging queue."""
    level: str  # "progress", "status", "error"
    content: str
    problem_id: Optional[str] = None


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
        self.run_dir = self.storage.run_dir

    async def _run_concurrent(
        self,
        problems: List[BenchmarkProblem],
        log_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
    ) -> List[ProblemResult]:
        """Run problems concurrently with bounded parallelism.

        Args:
            problems: List of problems to execute
            log_queue: Queue for logging messages
            result_queue: Queue for completed results

        Returns:
            List of ProblemResult objects
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_problems)

        async def execute_one(problem: BenchmarkProblem) -> ProblemResult:
            """Execute single problem with concurrency control."""
            async with semaphore:
                try:
                    # Execute problem (async)
                    result = await self.executor.execute_async(problem)

                    # Compare answer
                    try:
                        comparison = self.comparator.compare(
                            predicted=result.predicted_answer,
                            predicted_unit=result.predicted_unit or "",
                            ground_truth=problem.ground_truth_answer,
                            ground_truth_unit=problem.ground_truth_unit,
                        )
                    except Exception as e:
                        # Catch comparison errors and log them
                        comparison = type('obj', (object,), {'verdict': 'ERROR', 'reason': f"Comparison error: {str(e)[:100]}"})()
                        result.error_message = f"Answer comparison failed: {str(e)[:100]}"
                    result.verdict = comparison.verdict
                    result.comparison_reason = comparison.reason

                    # Log completion
                    if self.config.verbose:
                        status_msg = self._format_status_message(result, comparison)
                        await log_queue.put(LogMessage(
                            level="status",
                            content=status_msg,
                            problem_id=problem.problem_id
                        ))

                    # Queue result
                    await result_queue.put(result)

                    return result

                except Exception as e:
                    # Log error
                    error_msg = f"❌ {problem.problem_id} - ERROR: {str(e)[:100]}"
                    await log_queue.put(LogMessage(
                        level="error",
                        content=error_msg,
                        problem_id=problem.problem_id
                    ))

                    # Return error result
                    result = ProblemResult(
                        problem_id=problem.problem_id,
                        success=False,
                        predicted_answer=None,
                        predicted_unit=None,
                        ground_truth_answer=problem.ground_truth_answer,
                        ground_truth_unit=problem.ground_truth_unit,
                        total_time=0.0,
                        total_cost=0.0,
                        verdict="ERROR",
                        error_message=str(e)[:200],
                    )
                    await result_queue.put(result)
                    return result

        # Execute all problems concurrently
        tasks = [execute_one(problem) for problem in problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions (they were already logged)
        valid_results = [r for r in results if isinstance(r, ProblemResult)]
        return valid_results

    def _format_status_message(self, result: ProblemResult, comparison) -> str:
        """Format status message for a completed problem."""
        verdict_emoji = {
            "CORRECT": "✅",
            "INCORRECT": "❌",
            "ERROR": "⚠️",
            "TIMEOUT": "⏱️",
        }
        emoji = verdict_emoji.get(result.verdict, "❓")

        msg = f"{emoji} {result.problem_id} - {result.verdict}"

        if result.verdict == "CORRECT":
            msg += f" | Predicted: {result.predicted_answer} {result.predicted_unit}"
        elif result.verdict == "INCORRECT":
            msg += f" | Predicted: {result.predicted_answer} {result.predicted_unit} | Expected: {result.ground_truth_answer} {result.ground_truth_unit}"
            if comparison.reason:
                msg += f" | Reason: {comparison.reason}"
        elif result.verdict == "ERROR":
            msg += f" | Error: {result.error_message[:50]}"

        return msg

    async def _logging_consumer(
        self,
        log_queue: asyncio.Queue,
        progress_bar: tqdm,
        stop_event: asyncio.Event,
    ):
        """Consume log messages and write atomically.

        Args:
            log_queue: Queue of log messages
            progress_bar: tqdm progress bar to update
            stop_event: Event to signal shutdown
        """
        while not stop_event.is_set() or not log_queue.empty():
            try:
                msg = await asyncio.wait_for(log_queue.get(), timeout=0.1)

                if msg.level == "status":
                    # Write status message
                    tqdm.write(msg.content)
                    progress_bar.update(1)

                elif msg.level == "error":
                    # Write error message
                    tqdm.write(msg.content)
                    progress_bar.update(1)

                log_queue.task_done()

            except asyncio.TimeoutError:
                continue  # No messages, check stop event again

    async def _checkpoint_saver(
        self,
        result_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        """Periodically save checkpoints from completed results.

        Args:
            result_queue: Queue of completed results
            stop_event: Event to signal shutdown
        """
        results_buffer = []
        last_save = time.time()

        while not stop_event.is_set() or not result_queue.empty():
            try:
                # Collect results from queue
                result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                results_buffer.append(result)
                result_queue.task_done()
            except asyncio.TimeoutError:
                pass  # No results, check if it's time to save

            # Save checkpoint periodically
            current_time = time.time()
            if current_time - last_save >= self.config.checkpoint_interval_seconds:
                if results_buffer:
                    await asyncio.to_thread(self._save_checkpoint, results_buffer.copy())
                    last_save = current_time

        # Final save before exit
        if results_buffer:
            await asyncio.to_thread(self._save_checkpoint, results_buffer)

    async def _run_concurrent_with_logging(
        self,
        problems: List[BenchmarkProblem],
    ) -> List[ProblemResult]:
        """Execute problems concurrently with logging coordination.

        Args:
            problems: List of problems to execute

        Returns:
            List of completed ProblemResult objects
        """
        # Create queues
        log_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        stop_event = asyncio.Event()

        # Create progress bar
        progress_bar = tqdm(total=len(problems), desc="Benchmark", unit="problem")

        # Start background tasks
        logging_task = asyncio.create_task(
            self._logging_consumer(log_queue, progress_bar, stop_event)
        )
        checkpoint_task = asyncio.create_task(
            self._checkpoint_saver(result_queue, stop_event)
        )

        try:
            # Run concurrent execution
            results = await self._run_concurrent(problems, log_queue, result_queue)

            # Wait for queues to drain
            await log_queue.join()
            await result_queue.join()

        finally:
            # Signal shutdown
            stop_event.set()

            # Wait for background tasks
            await logging_task
            await checkpoint_task

            # Close progress bar
            progress_bar.close()

        return results

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
            print(f"\nRunning benchmark on {len(problems) - len(completed_ids)} problems...")
            print(f"   Concurrent workers: {self.config.max_concurrent_problems}")
            print(f"   Checkpoint interval: {self.config.checkpoint_interval_seconds}s\n")

        results = checkpoint.get("results", []) if checkpoint else []
        result_objs = [ProblemResult(**r) for r in results]

        problems_to_run = [p for p in problems if p.problem_id not in completed_ids]

        # Run concurrent execution
        concurrent_results = asyncio.run(self._run_concurrent_with_logging(problems_to_run))
        result_objs.extend(concurrent_results)

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
