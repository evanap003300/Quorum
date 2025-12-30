# SciBench Benchmark Report
**Generated**: 2025-12-29 18:58:05
**Run Name**: None

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 5 |
| Timeout | 120s |
| Numeric Tolerance | 1.0% |
| Skip Images | True |

## Overall Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Problems | 5 | 100% |
| Correct | 2 | 40.0% |
| Incorrect | 1 | 20.0% |
| Errors | 2 | 40.0% |
| Timeouts | 0 | 0.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 434.8s |
| Avg Time per Problem | 86.97s |
| Median Time | 85.63s |
| P50 | 85.63s |
| P75 | 103.87s |
| P90 | 115.15s |
| P95 | 118.91s |
| P99 | 121.92s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.13 |
| Avg Cost per Problem | $0.0332 |
| Median Cost | $0.0332 |
| P50 | $0.0332 |
| P75 | $0.0390 |
| P90 | $0.0393 |
| P95 | $0.0394 |
| P99 | $0.0395 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 85.63s | $0.0000 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 103.87s | $0.0388 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 122.67s | $0.0395 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 1 | 100.0% | 43.17s | $0.0275 |
| $^\circ$ | 1 | 1 | 100.0% | 79.50s | $0.0271 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 1 |

## Failed Problems (3 failures)

###  16.30
**Verdict**: ERROR
**Time**: 103.87s | **Cost**: $0.0388

### 12.1
**Verdict**: ERROR
**Error**: Solver error: Multi-output step should return dicts
**Time**: 85.63s | **Cost**: $0.0000

###  Problem 9.12
**Verdict**: INCORRECT
**Time**: 122.67s | **Cost**: $0.0395

