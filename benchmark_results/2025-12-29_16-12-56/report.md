# SciBench Benchmark Report
**Generated**: 2025-12-29 16:18:03
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
| Incorrect | 0 | 0.0% |
| Errors | 3 | 60.0% |
| Timeouts | 0 | 0.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 305.9s |
| Avg Time per Problem | 61.17s |
| Median Time | 65.03s |
| P50 | 65.03s |
| P75 | 71.85s |
| P90 | 77.16s |
| P95 | 78.93s |
| P99 | 80.34s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.12 |
| Avg Cost per Problem | $0.0310 |
| Median Cost | $0.0310 |
| P50 | $0.0310 |
| P75 | $0.0371 |
| P90 | $0.0375 |
| P95 | $0.0377 |
| P99 | $0.0378 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 53.35s | $0.0000 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 80.69s | $0.0369 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 71.85s | $0.0378 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 1 | 100.0% | 34.94s | $0.0243 |
| $^\circ$ | 1 | 1 | 100.0% | 65.03s | $0.0250 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 2 |

## Failed Problems (3 failures)

###  16.30
**Verdict**: ERROR
**Time**: 80.69s | **Cost**: $0.0369

### 12.1
**Verdict**: ERROR
**Error**: Solver error: Multi-output step should return dicts
**Time**: 53.35s | **Cost**: $0.0000

###  Problem 9.12
**Verdict**: ERROR
**Error**: Step 2 failed: Failed to extract: ['m_tank_empty', 'm_astro', 'm_post_gas']. Output:
{'m_tank_empty': '8.0 kg', 'm_astro': '90.0 kg', 'm_post_gas': '98.0 kg'}

**Time**: 71.85s | **Cost**: $0.0378

