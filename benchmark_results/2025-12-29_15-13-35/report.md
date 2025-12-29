# SciBench Benchmark Report
**Generated**: 2025-12-29 15:17:58
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
| Total Time | 260.9s |
| Avg Time per Problem | 52.19s |
| Median Time | 52.14s |
| P50 | 52.14s |
| P75 | 63.42s |
| P90 | 65.14s |
| P95 | 65.71s |
| P99 | 66.17s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.11 |
| Avg Cost per Problem | $0.0286 |
| Median Cost | $0.0270 |
| P50 | $0.0270 |
| P75 | $0.0309 |
| P90 | $0.0341 |
| P95 | $0.0352 |
| P99 | $0.0361 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 52.14s | $0.0291 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 66.28s | $0.0000 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 63.42s | $0.0363 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 0 | 0.0% | 30.37s | $0.0249 |
| $^\circ$ | 1 | 0 | 0.0% | 48.72s | $0.0241 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 2 |

## Failed Problems (5 failures)

###  16.30
**Verdict**: ERROR
**Error**: Solver error: Multi-output step should return dicts
**Time**: 66.28s | **Cost**: $0.0000

### 12.1
**Verdict**: ERROR
**Time**: 52.14s | **Cost**: $0.0291

###  e3.7(a)(b)
**Verdict**: ERROR
**Time**: 30.37s | **Cost**: $0.0249

###  7.10
**Verdict**: ERROR
**Time**: 48.72s | **Cost**: $0.0241

###  Problem 9.12
**Verdict**: ERROR
**Error**: Step 2 failed: Failed to extract: ['m_empty_tank', 'm_after_gas']. Output:
8.0 kg
98.0 kg

**Time**: 63.42s | **Cost**: $0.0363

