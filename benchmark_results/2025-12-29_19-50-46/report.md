# SciBench Benchmark Report
**Generated**: 2025-12-29 19:56:20
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
| Total Time | 332.8s |
| Avg Time per Problem | 66.55s |
| Median Time | 81.52s |
| P50 | 81.52s |
| P75 | 89.56s |
| P90 | 104.65s |
| P95 | 109.68s |
| P99 | 113.70s |

## Cost Statistics

| Metric | Value |
|--------|-------|
| Total Cost | $0.24 |
| Avg Cost per Problem | $0.0482 |
| Median Cost | $0.0522 |
| P50 | $0.0522 |
| P75 | $0.0640 |
| P90 | $0.0747 |
| P95 | $0.0782 |
| P99 | $0.0811 |

## Breakdown by Source

| Source | Total | Correct | Accuracy | Avg Time | Avg Cost |
|--------|-------|---------|----------|----------|----------|
|  $10^{-47} \mathrm{~kg} \mathrm{~m}^2$ | 1 | 0 | 0.0% | 89.56s | $0.0818 |
|  $10^{34} \mathrm{~m}^{-3} \mathrm{~s}^{-1}$ | 1 | 0 | 0.0% | 81.52s | $0.0640 |
| $ \mathrm{~m} / \mathrm{s}$  | 1 | 0 | 0.0% | 114.71s | $0.0522 |
| $\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$ | 1 | 1 | 100.0% | 20.16s | $0.0183 |
| $^\circ$ | 1 | 0 | 0.0% | 26.81s | $0.0247 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| UNKNOWN_ERROR | 1 |

## Failed Problems (4 failures)

###  16.30
**Verdict**: ERROR
**Time**: 81.52s | **Cost**: $0.0640

### 12.1
**Verdict**: ERROR
**Time**: 89.56s | **Cost**: $0.0818

###  7.10
**Verdict**: ERROR
**Error**: Could not find ANSWER in response:
To determine the angle at which the particle leaves the hemisphere, we apply the principles of conservation of energy and Newton's second law.

### 1. Conservation of Energy

Let the particle start at the top of the hemisphere (angle $\theta = 0$) with zero velocity. Let the radius of the hemisphere be $a$ and the mass of the particle be $m$. We define the potential energy reference level at the center of the hemisphere.

*   **Initial State ($\theta = 0$):**
    *   Velocity $v_0 = 0$
    *   Height $h_0 = a$
    *   Total Energy $E_0 = \text{KE} + \text{PE} = 0 + mga = mga$

*   **At an arbitrary angle $\theta$:**
    *   Velocity is $v$
    *   Height $h = a \cos(\theta)$
    *   Total Energy $E = \frac{1}{2}mv^2 + mga \cos(\theta)$

Since the surface is smooth, energy is conserved ($E_0 = E$):
$$mga = \frac{1}{2}mv^2 + mga \cos(\theta)$$

Solving for $v^2$:
$$mg a (1 - \cos(\theta)) = \frac{1}{2} m v^2$$
$$v^2 = 2ga(1 - \cos(\theta))$$

### 2. Newton's Second Law (Radial Direction)

We analyze the forces acting on the particle in the radial direction (towards the center of the hemisphere).
*   **Gravity component:** $mg \cos(\theta)$ acting towards the center.
*   **Normal force ($N$):** Acting outward, away from the center.

The net radial force provides the centripetal acceleration ($a_c = v^2/a$):
$$F_{\text{net}} = mg \cos(\theta) - N = m \frac{v^2}{a}$$

### 3. Condition for Leaving the Surface

The particle leaves the hemisphere when it loses contact with the surface. This occurs when the normal force becomes zero ($N = 0$).

Substituting $N=0$ into the force equation:
$$mg \cos(\theta) = m \frac{v^2}{a}$$
$$v^2 = ga \cos(\theta)$$

### 4. Solving for the Angle

Now we have two expressions for $v^2$. We equate them to find $\theta$:

1.  $v^2 = 2ga(1 - \cos(\theta))$ (from Energy)
2.  $v^2 = ga \cos(\theta)$ (from Dynamics)

$$2ga(1 - \cos(\theta)) = ga \cos(\theta)$$

Divide by $ga$ (since $g, a \neq 0$):
$$2(1 - \cos(\theta)) = \cos(\theta)$$
$$2 - 2\cos(\theta) = \cos(\theta)$$
$$2 = 3\cos(\theta)$$

$$\cos(\theta) = \frac{2}{3}$$

Therefore, the angle is:
$$\theta = \arccos\left(\frac{2}{3}\right)$$

Numerically, this is approximately $48.19^\circ$.

ANSWER: \arccos(2/3)
**Time**: 26.81s | **Cost**: $0.0247

###  Problem 9.12
**Verdict**: INCORRECT
**Time**: 114.71s | **Cost**: $0.0522

