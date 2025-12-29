# SciBench Benchmarking - Quick Start Guide

Your benchmarking system is now ready to use! Here's how to get started.

## Setup (One-time)

Your virtual environment has been fixed and all dependencies installed. To use in the future, activate it:

```bash
source .venv/bin/activate
```

## Run Your First Benchmark

### Option 1: Fastest Test (5 problems, ~3-5 minutes, skip reviews)

```bash
source .venv/bin/activate
SKIP_PHYSICS_REVIEW=true python -m src.benchmarking.cli run --config benchmark_configs/scibench_fast.yaml
```

**This is the recommended starting point - it skips the Physics Lawyer review which can timeout.**

### Option 2: Quick Test (5 problems - ~5-10 minutes with reviews)

```bash
source .venv/bin/activate
python -m src.benchmarking.cli run --config benchmark_configs/scibench_quick_test.yaml
```

### Option 3: Default Benchmark (10 problems - ~10-20 minutes)

```bash
source .venv/bin/activate
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
```

### Option 4: Custom Batch Size with Seed

```bash
source .venv/bin/activate
python -m src.benchmarking.cli run --batch-size 3 --seed 42
```

## Environment Variables

- `SKIP_PHYSICS_REVIEW=true` - Skip the Physics Lawyer review (faster, good for quick testing)
- `SKIP_PHYSICS_REVIEW=false` - Enable physics review (default, more thorough)

## View Results

After the benchmark completes, results are saved with a timestamp. For example:

```
benchmark_results/2025-12-29_14-30-00/
├── report.md              ← Human-readable report
├── detailed_results.json  ← Full problem details
├── summary.json           ← Metrics only
├── results.csv            ← Export for plotting
└── checkpoint.json        ← For resuming
```

View the report:

```bash
cat benchmark_results/2025-12-29_14-30-00/report.md
```

## Useful Commands

### List all your benchmark runs

```bash
source .venv/bin/activate
python -m src.benchmarking.cli list
```

### Resume an interrupted run

```bash
source .venv/bin/activate
python -m src.benchmarking.cli resume --run-name 2025-12-29_14-30-00
```

### Generate report from existing results

```bash
source .venv/bin/activate
python -m src.benchmarking.cli report --run-name 2025-12-29_14-30-00
```

### Compare multiple runs

```bash
source .venv/bin/activate
python -m src.benchmarking.cli compare run1 run2 run3 --output comparison.md
```

## Understanding the Results

### Key Metrics

- **Accuracy**: Percentage of problems solved correctly (your primary metric)
- **Avg Time**: Average seconds per problem
- **Avg Cost**: Average API cost per problem
- **Breakdown by Source**: Performance by textbook (atkins, thermo, etc.)

### Example Report Output

```
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
```

## Configuration Files

Pre-built config files are in `benchmark_configs/`:

- **scibench_quick_test.yaml** - 5 problems (quick iteration)
- **scibench_default.yaml** - 10 problems (standard)
- **scibench_full.yaml** - 692 problems (comprehensive - takes several hours)

Create your own config by copying and editing:

```bash
cp benchmark_configs/scibench_default.yaml my_benchmark.yaml
# Edit my_benchmark.yaml with your settings
python -m src.benchmarking.cli run --config my_benchmark.yaml
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

Make sure you activated the venv:

```bash
source .venv/bin/activate
```

### "Problem execution exceeded timeout"

Increase the timeout in your config or command:

```bash
python -m src.benchmarking.cli run --batch-size 5 --timeout 600  # 10 minutes
```

### "Could not convert units"

This is normal - some problems may have incompatible units. Disable conversion:

Edit config file and set:
```yaml
allow_unit_conversion: false
```

## What's Included

**17 Python modules** providing:

- Dataset loading from HuggingFace
- Problem execution with timeout handling
- Answer comparison with numeric tolerance
- Unit conversion (using pint)
- Metrics aggregation (accuracy, timing, cost)
- Markdown report generation
- JSON persistence and checkpoint system
- CLI with 5 commands (run, resume, report, list, compare)

**3 Example configs**:
- Quick test (5 problems)
- Default (10 problems)
- Full (all 692 problems)

## Documentation

Full documentation: `src/benchmarking/README.md`

Key sections:
- Architecture overview
- All CLI commands
- Configuration reference
- Understanding results
- Troubleshooting

## Next Steps

1. **Run a quick test**: `python -m src.benchmarking.cli run --batch-size 5 --seed 42`
2. **View results**: Open the generated `report.md`
3. **Iterate**: Make changes to your solver and re-run benchmarks
4. **Compare**: Use `compare` command to track improvements

## Example Workflow

```bash
# Day 1: Establish baseline
source .venv/bin/activate
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
# → Results in: benchmark_results/2025-12-29_14-30-00/

# Day 2: Make improvements to solver
# ... (edit your solver code) ...

# Day 3: Test improvements
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
# → Results in: benchmark_results/2025-12-30_09-15-00/

# Day 4: Compare baseline vs improvements
python -m src.benchmarking.cli compare \
  2025-12-29_14-30-00 \
  2025-12-30_09-15-00 \
  --output improvement_analysis.md
```

---

**Enjoy benchmarking!** 🚀
