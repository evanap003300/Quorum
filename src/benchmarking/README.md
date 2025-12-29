# SciBench Benchmarking System

Automated benchmarking system for evaluating problem-solving accuracy against the SciBench dataset.

## Overview

The SciBench benchmarking system provides:

- **Automated benchmark execution**: Run 10-692 problems from SciBench with configurable batch sizes
- **Answer evaluation**: Compare predicted answers with ground truth using relative tolerance
- **Metrics tracking**: Accuracy, cost breakdown, performance timing, error analysis
- **Progress checkpointing**: Resume interrupted runs without re-running completed problems
- **Comprehensive reporting**: Markdown reports with detailed statistics and visualizations
- **CLI interface**: Simple command-line access to all functionality

## Quick Start

### Installation

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

### Run Quick Test (5 problems)

```bash
python -m src.benchmarking.cli run --config benchmark_configs/scibench_quick_test.yaml
```

### Run Default Benchmark (10 problems)

```bash
python -m src.benchmarking.cli run --batch-size 10 --seed 42
```

### Run with Config File

```bash
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
```

## Results

Results are saved to timestamped directories in `benchmark_results/`:

```
benchmark_results/
└── 2025-12-29_14-30-00/         # Timestamp: run date/time
    ├── config.yaml              # Configuration used
    ├── detailed_results.json    # Full results per problem
    ├── summary.json             # Aggregated metrics
    ├── report.md                # Human-readable report
    ├── results.csv              # Export for plotting
    └── checkpoint.json          # For resuming
```

## Configuration

### Command-Line Arguments

```bash
python -m src.benchmarking.cli run [OPTIONS]

OPTIONS:
  --batch-size N           Number of problems to evaluate (default: 10)
  --seed SEED              Random seed for reproducibility
  --config FILE            Load config from YAML or JSON file
  --timeout SECONDS        Timeout per problem in seconds (default: 300)
  --output-dir DIR         Output directory (default: benchmark_results)
```

### Configuration File

Create a YAML config file:

```yaml
batch_size: 10
random_seed: 42
skip_image_problems: true
timeout_per_problem: 300
numeric_tolerance: 0.01       # 1% relative tolerance
allow_unit_conversion: true
output_dir: benchmark_results
verbose: true
```

See example configs in `benchmark_configs/`:

- `scibench_quick_test.yaml`: 5 problems (quick iteration)
- `scibench_default.yaml`: 10 problems (standard)
- `scibench_full.yaml`: 692 problems (full benchmark)

## CLI Commands

### Run Benchmark

Execute a new benchmark:

```bash
python -m src.benchmarking.cli run --batch-size 10 --seed 42
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
```

### Resume from Checkpoint

Resume an interrupted benchmark:

```bash
python -m src.benchmarking.cli resume --run-name 2025-12-29_14-30-00
```

### Generate Report

Generate a report from saved results:

```bash
python -m src.benchmarking.cli report --run-name 2025-12-29_14-30-00
```

### List Runs

List all available benchmark runs:

```bash
python -m src.benchmarking.cli list
```

### Compare Runs

Compare multiple benchmark runs side-by-side:

```bash
python -m src.benchmarking.cli compare run1 run2 run3 --output comparison.md
```

## Understanding Results

### Overall Accuracy

The primary metric is **accuracy**: percentage of problems solved correctly.

```
Accuracy = Correct Problems / Total Problems
```

Example: 8/10 problems correct = 80% accuracy

### Answer Comparison

Answers are compared using **relative tolerance** (default: 1%):

```
Relative Error = |Predicted - GroundTruth| / |GroundTruth|
Correct if: Relative Error <= Tolerance
```

Example with 1% tolerance:
- Ground truth: 100.0
- Predicted: 100.5
- Error: 0.5%
- Result: ✓ CORRECT (error ≤ 1%)

### Unit Conversion

If predicted and ground truth have different units, automatic conversion is attempted using the `pint` library.

Example:
- Ground truth: 1000 m/s
- Predicted: 1 km/s
- Conversion: 1 km/s = 1000 m/s
- Result: ✓ CORRECT match

### Cost Metrics

Track API call costs from different stages:

- **Total Cost**: Sum of all component costs
- **Avg Cost per Problem**: Mean cost per problem
- **Cost Breakdown**: By source (textbook), helps identify expensive topics

### Performance Metrics

Track execution speed:

- **Total Time**: Sum of all problem solving times
- **Avg Time per Problem**: Mean time per problem
- **Percentiles (P50, P90, P95, P99)**: Distribution of problem solving times

## Report Output

Generated reports (`report.md`) include:

### Configuration Summary
Lists the benchmark parameters and settings

### Overall Results
- Total problems evaluated
- Correct/incorrect/error/timeout counts
- Overall accuracy percentage

### Performance Statistics
- Total time and average time per problem
- Median and percentile timing data (P50, P90, P95, P99)

### Cost Statistics
- Total cost and average cost per problem
- Median and percentile cost data

### Breakdown by Source
Metrics grouped by textbook source (atkins, thermo, calculus, etc.):
- Accuracy per subject
- Average time and cost per subject

### Error Analysis
- Count of error types (TIMEOUT, PLANNING_ERROR, EXECUTION_ERROR, etc.)
- Failed problems list (first 10 shown)

## Example Report Output

```markdown
# SciBench Benchmark Report

**Generated**: 2025-12-29 14:30:45
**Run Name**: 2025-12-29_14-30-00

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 10 |
| Timeout | 300s |
| Numeric Tolerance | 1.0% |

## Overall Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Problems | 10 | 100% |
| Correct | 8 | 80.0% |
| Incorrect | 1 | 10.0% |
| Errors | 1 | 10.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 135.2s |
| Avg Time per Problem | 13.52s |
| Median Time | 12.80s |
| P90 | 18.20s |
```

## Performance Tips

### Speed Up Benchmarking

1. **Use smaller batches**: Start with `--batch-size 5` to iterate quickly
2. **Set shorter timeouts**: `--timeout 120` for faster feedback
3. **Run specific problems**: Use checkpoint to resume from last failure

### Cost Optimization

1. **Use default tolerance**: 1% tolerance is reasonable for most problems
2. **Disable unit conversion**: If your problems use consistent units
3. **Skip expensive problems**: Edit the problem selection logic if needed

## Architecture

The benchmarking system is modular and decoupled from the solver:

```
Dataset Loader → Problem Executor → Answer Comparator → Metrics Aggregator
    (load problems)  (with timeout)   (compare answers)  (calculate stats)
                           ↓
                    solve_problem()  ← Uses existing solver (no modifications)
```

### Modules

- `config/`: Configuration models
- `dataset/`: SciBench dataset loading
- `runner/`: Problem execution with timeout handling
- `evaluation/`: Answer comparison and unit conversion
- `metrics/`: Statistics aggregation
- `storage/`: Results persistence
- `cli.py`: Command-line interface

## Troubleshooting

### Issue: "Problem execution exceeded timeout"

**Solution**: Increase `timeout_per_problem` in config

```yaml
timeout_per_problem: 600  # 10 minutes
```

### Issue: "Could not convert units"

**Solution**: Disable unit conversion or check unit strings

```yaml
allow_unit_conversion: false
```

### Issue: "Resumed run not updating results"

**Solution**: Manually delete the checkpoint file

```bash
rm benchmark_results/<run-name>/checkpoint.json
```

### Issue: Dataset not loading from HuggingFace

**Solution**: Set cache directory and check internet connection

```bash
export HF_HOME=/path/to/cache
python -m src.benchmarking.cli run --batch-size 5
```

## Future Enhancements

Potential additions to the benchmarking system:

1. **Image problem support**: Benchmark the full 94-problem multimodal subset
2. **Parallel execution**: Run multiple problems concurrently
3. **Comparison mode**: Automatically compare against GPT-4, Gemini baselines
4. **Error categorization**: Deep analysis of common failure patterns
5. **Visualization**: Plots of accuracy, cost, and timing distributions
6. **Integration testing**: Automated regression testing for system changes

## References

**SciBench Dataset**
- Paper: [SciBench: Evaluating College-Level Scientific Problem-Solving](https://arxiv.org/abs/2307.10635)
- GitHub: https://github.com/mandyyyyii/scibench
- HuggingFace: https://huggingface.co/datasets/xw27/scibench
- Website: https://scibench-ucla.github.io

**Dataset Details**
- 692 problems from college-level textbooks
- Subjects: Physics, Chemistry, Mathematics
- Multi-step reasoning required
- Published at ICML 2024
- Best reported baseline: 43.22% accuracy (GPT-4 with CoT + tools)
