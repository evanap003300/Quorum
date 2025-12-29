"""SciBench dataset loader and manager."""

from pydantic import BaseModel
from typing import List, Dict, Optional
import random
from datasets import load_dataset


class BenchmarkProblem(BaseModel):
    """Represents a single benchmark problem from SciBench."""

    problem_id: str
    problem_text: str
    ground_truth_answer: str
    ground_truth_unit: str
    source: str
    has_image: bool = False
    image_path: Optional[str] = None


class SciBenchLoader:
    """Load and filter SciBench dataset from HuggingFace."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize loader.

        Args:
            cache_dir: Optional directory to cache dataset locally
        """
        self.cache_dir = cache_dir
        self._dataset = None

    def load_dataset(self) -> List[BenchmarkProblem]:
        """Load SciBench dataset from HuggingFace.

        Returns:
            List of BenchmarkProblem objects
        """
        if self._dataset is None:
            # Load from HuggingFace
            dataset = load_dataset("xw27/scibench", cache_dir=self.cache_dir)
            self._dataset = dataset["train"]

        problems = []
        for item in self._dataset:
            problem = BenchmarkProblem(
                problem_id=item.get("problemid", "unknown"),
                problem_text=item.get("problem_text", ""),
                ground_truth_answer=str(item.get("answer_number", "")),
                ground_truth_unit=item.get("unit", ""),
                source=item.get("source", "unknown"),
                has_image=False,  # SciBench text dataset doesn't have images embedded
            )
            problems.append(problem)

        return problems

    def filter_text_only(self, problems: List[BenchmarkProblem]) -> List[BenchmarkProblem]:
        """Filter to text-only problems (exclude image-based problems).

        Args:
            problems: List of problems to filter

        Returns:
            Filtered list excluding problems with images
        """
        return [p for p in problems if not p.has_image]

    def sample_problems(
        self,
        problems: List[BenchmarkProblem],
        batch_size: int,
        random_seed: Optional[int] = None,
        specific_ids: Optional[List[str]] = None,
    ) -> List[BenchmarkProblem]:
        """Sample problems from the dataset.

        Args:
            problems: List of problems to sample from
            batch_size: Number of problems to sample
            random_seed: Random seed for reproducibility
            specific_ids: Specific problem IDs to select (overrides batch_size)

        Returns:
            Sampled list of problems
        """
        if specific_ids:
            # Select specific problems by ID
            id_set = set(specific_ids)
            selected = [p for p in problems if p.problem_id in id_set]
            if len(selected) < len(specific_ids):
                missing = id_set - {p.problem_id for p in selected}
                print(f"Warning: {len(missing)} requested problems not found: {missing}")
            return selected

        # Random sampling
        if random_seed is not None:
            random.seed(random_seed)

        batch_size = min(batch_size, len(problems))
        return random.sample(problems, batch_size)

    def get_subject_breakdown(self, problems: List[BenchmarkProblem]) -> Dict[str, int]:
        """Count problems by source/subject.

        Args:
            problems: List of problems to analyze

        Returns:
            Dictionary mapping source name to count
        """
        breakdown = {}
        for problem in problems:
            breakdown[problem.source] = breakdown.get(problem.source, 0) + 1
        return breakdown

    def print_dataset_stats(self, problems: List[BenchmarkProblem]) -> None:
        """Print statistics about the dataset.

        Args:
            problems: List of problems to analyze
        """
        print(f"Total problems: {len(problems)}")
        print(f"Problems with images: {sum(1 for p in problems if p.has_image)}")

        breakdown = self.get_subject_breakdown(problems)
        print("\nBreakdown by source:")
        for source, count in sorted(breakdown.items()):
            print(f"  {source}: {count} problems")
