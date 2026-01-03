"""Benchmark configuration module with Pydantic models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from pathlib import Path
import yaml
import json


class BenchmarkConfig(BaseModel):
    """Configuration for running benchmarks against SciBench dataset."""

    # Dataset configuration
    batch_size: int = Field(default=10, description="Number of problems to evaluate")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    problem_ids: Optional[List[str]] = Field(default=None, description="Specific problem IDs to run (overrides batch_size)")
    subject_filter: Optional[List[str]] = Field(default=None, description="Filter by textbook source (e.g., ['atkins', 'thermo'])")
    skip_image_problems: bool = Field(default=True, description="Skip problems with images")

    # Execution configuration
    timeout_per_problem: int = Field(default=300, description="Timeout in seconds per problem (5 min default)")
    max_retries: int = Field(default=0, description="Maximum retries for failed problems")
    save_intermediate_results: bool = Field(default=True, description="Save results after each problem")
    checkpoint_frequency: int = Field(default=5, description="Save checkpoint every N problems")
    max_concurrent_problems: int = Field(default=5, description="Number of problems to run concurrently (1 = sequential)")
    checkpoint_interval_seconds: int = Field(default=30, description="Time between checkpoint saves in seconds")

    # Output configuration
    output_dir: str = Field(default="benchmark_results", description="Output directory for results")
    run_name: Optional[str] = Field(default=None, description="Run name (auto-generated if None)")
    verbose: bool = Field(default=True, description="Verbose logging")

    # Evaluation configuration
    numeric_tolerance: float = Field(default=0.015, description="Relative tolerance for numeric comparison (1.5% default)")
    allow_unit_conversion: bool = Field(default=True, description="Allow unit conversion when comparing answers")

    class Config:
        """Pydantic config."""
        extra = "forbid"
        validate_assignment = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

    def save_json(self, output_path: str) -> None:
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.timeout_per_problem <= 0:
            raise ValueError(f"timeout_per_problem must be positive, got {self.timeout_per_problem}")
        if not 0 <= self.numeric_tolerance <= 1:
            raise ValueError(f"numeric_tolerance must be between 0 and 1, got {self.numeric_tolerance}")
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}")
        if self.max_concurrent_problems <= 0:
            raise ValueError(f"max_concurrent_problems must be positive, got {self.max_concurrent_problems}")
        if self.checkpoint_interval_seconds <= 0:
            raise ValueError(f"checkpoint_interval_seconds must be positive, got {self.checkpoint_interval_seconds}")
