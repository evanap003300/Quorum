# SciBench Benchmark Report
**Generated**: 2025-12-29 18:28:53
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
| Correct | 1 | 20.0% |
| Incorrect | 1 | 20.0% |
| Errors | 3 | 60.0% |
| Timeouts | 0 | 0.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 429.3s |
| Avg Time per Problem | 85.85s |
| Median Time | 66.65s |
| P50 | 66.65s |
| P75 | 129.79s |
| P90 | 130.39s |
| P95 | 130.59s |
| P99 | 130.75s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.13 |
| Avg Cost per Problem | $0.0316 |
| Median Cost | $0.0296 |
| P50 | $0.0296 |
| P75 | $0.0333 |
| P90 | $0.0371 |
| P95 | $0.0384 |
| P99 | $0.0394 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 66.65s | $0.0312 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 130.79s | $0.0000 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 129.79s | $0.0397 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 1 | 100.0% | 35.77s | $0.0280 |
| $^\circ$ | 1 | 0 | 0.0% | 66.25s | $0.0275 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 2 |

## Failed Problems (4 failures)

###  16.30
**Verdict**: ERROR
**Error**: Solver error: Multi-output step should return dicts
**Time**: 130.79s | **Cost**: $0.0000

### 12.1
**Verdict**: ERROR
**Error**: Step 1 failed: Failed to extract: ['theta_deg', 'd_pm']. Output:
Error: unexpected indent (<string>, line 5)
**Time**: 66.65s | **Cost**: $0.0312

###  7.10
**Verdict**: ERROR
**Time**: 66.25s | **Cost**: $0.0275

###  Problem 9.12
**Verdict**: INCORRECT
**Time**: 129.79s | **Cost**: $0.0397

