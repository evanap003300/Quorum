# SciBench Benchmark Report
**Generated**: 2025-12-29 15:30:45
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
| Correct | 0 | 0.0% |
| Incorrect | 0 | 0.0% |
| Errors | 5 | 100.0% |
| Timeouts | 0 | 0.0% |

## Performance Statistics

| Metric | Value |
|--------|-------|
| Total Time | 262.1s |
| Avg Time per Problem | 52.42s |
| Median Time | 55.48s |
| P50 | 55.48s |
| P75 | 60.69s |
| P90 | 64.45s |
| P95 | 65.70s |
| P99 | 66.71s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.15 |
| Avg Cost per Problem | $0.0291 |
| Median Cost | $0.0285 |
| P50 | $0.0285 |
| P75 | $0.0320 |
| P90 | $0.0351 |
| P95 | $0.0362 |
| P99 | $0.0370 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 50.80s | $0.0285 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 60.69s | $0.0320 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 66.96s | $0.0372 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 0 | 0.0% | 28.19s | $0.0246 |
| $^\circ$ | 1 | 0 | 0.0% | 55.48s | $0.0233 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 2 |

## Failed Problems (5 failures)

###  16.30
**Verdict**: ERROR
**Time**: 60.69s | **Cost**: $0.0320

### 12.1
**Verdict**: ERROR
**Time**: 50.80s | **Cost**: $0.0285

###  e3.7(a)(b)
**Verdict**: ERROR
**Time**: 28.19s | **Cost**: $0.0246

###  7.10
**Verdict**: ERROR
**Error**: Step 1 failed: Failed to extract: ['m', 'a', 'g']. Output:
Error: unexpected indent (<string>, line 7)
**Time**: 55.48s | **Cost**: $0.0233

###  Problem 9.12
**Verdict**: ERROR
**Error**: Step 2 failed: Failed to extract: ['m_shell', 'm_astro', 'm_post_gas']. Output:
Error: unterminated string literal (detected at line 12) (<string>, line 12)
**Time**: 66.96s | **Cost**: $0.0372

