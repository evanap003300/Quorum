# SciBench Benchmark Report
**Generated**: 2025-12-29 15:58:33
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
| Incorrect | 0 | 0.0% |
| Errors | 4 | 80.0% |
| Timeouts | 0 | 0.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 287.9s |
| Avg Time per Problem | 57.57s |
| Median Time | 61.44s |
| P50 | 61.44s |
| P75 | 66.80s |
| P90 | 72.91s |
| P95 | 74.95s |
| P99 | 76.58s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.12 |
| Avg Cost per Problem | $0.0291 |
| Median Cost | $0.0289 |
| P50 | $0.0289 |
| P75 | $0.0348 |
| P90 | $0.0363 |
| P95 | $0.0368 |
| P99 | $0.0372 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 61.44s | $0.0000 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 66.80s | $0.0373 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 76.99s | $0.0340 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 1 | 100.0% | 26.01s | $0.0215 |
| $^\circ$ | 1 | 0 | 0.0% | 56.63s | $0.0238 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 2 |

## Failed Problems (4 failures)

###  16.30
**Verdict**: ERROR
**Time**: 66.80s | **Cost**: $0.0373

### 12.1
**Verdict**: ERROR
**Error**: Solver error: Multi-output step should return dicts
**Time**: 61.44s | **Cost**: $0.0000

###  7.10
**Verdict**: ERROR
**Time**: 56.63s | **Cost**: $0.0238

###  Problem 9.12
**Verdict**: ERROR
**Error**: Step 2 failed: Failed to extract: ['m_tank_empty', 'm_astro', 'm_after_gas']. Output:
{'m_tank_empty': '8.0 kg', 'm_astro': '90.0 kg', 'm_after_gas': '98.0 kg'}

**Time**: 76.99s | **Cost**: $0.0340

