# N=512 Queen Lattice Gas — Large-Scale Simulation Plan

> Date: 2026-03-23
> Author: Zong-Yue Liu
> Purpose: Extract ground-state entropy s_0 for N=512 via thermodynamic integration

---

## 1. Physical Goal

Ground-state entropy per queen:

```
s_0^MC = S(inf)/N - integral_0^inf (Cv / NT) dT
```

where `S(inf)/N = (1/N) * ln C(N^2, N)` is exact. For N=512:

- `S(inf)/N = 7.229462`
- `ln(N) = 6.238325`
- Expected `s_0 ≈ 4.29`, `gamma_MC = ln(N) - s_0 ≈ 1.937` (vs true gamma = 1.942)

---

## 2. Why N=512 is Feasible

### 2.1 tau_int does NOT grow at the Cv peak

| N   | tau_int(T=0.225) | tau_int(T=0.075) |
|-----|-------------------|-------------------|
| 64  | 135               | 14390             |
| 100 | 134               | 16454             |
| 128 | 133               | 18123             |
| 512 (extrap.) | ~127      | ~31000            |

The Cv peak (T ~ 0.225) is the region that contributes most to the integral.
There, tau_int ~ 130 sweeps regardless of N — dynamics are not critical-slowed.

### 2.2 Cv/N has converged

For N >= 32, Cv/N collapses to a universal function. N=512 data should be
essentially identical to N=128, with smaller finite-size corrections.

### 2.3 Integral contribution by temperature region (N=128 data)

| Region       | T range    | Integral contribution | Fraction |
|--------------|------------|----------------------|----------|
| Low-T        | 0 ~ 0.1   | 0.076                | 2.6%     |
| **Peak**     | 0.1 ~ 0.5 | 2.032                | **70.3%** |
| Mid-T        | 0.5 ~ 2.0 | 0.672                | 23.3%    |
| High-T       | 2.0 ~ 400 | 0.112                | 3.9%     |
| **Total**    |            | **2.892**            | 100%     |

---

## 3. Benchmark (Server: single core)

| N   | 10^6 sweeps | 10^8 sweeps |
|-----|-------------|-------------|
| 128 | 3.4 s       | 5.6 min     |
| 256 | 6.7 s       | 11.2 min    |
| 512 | 15.6 s      | 26 min      |

---

## 4. Temperature Grid (140 points total)

### Region 1: Low-T (5 points)
- T = 0.050, 0.0625, 0.075, 0.0875, 0.100
- nmeas = 10^7, therm = 10^6
- Reason: Cv ~ 0 here; just need to verify ground-state freezing
- tau_int ~ 20000-30000, but irrelevant since Cv/T ~ 0
- CPU: 5 x 2.6 min = 13 min

### Region 2: Peak (80 points) — CRITICAL
- T = 0.105, 0.110, ..., 0.500 (step 0.005)
- nmeas = 10^8, therm = 2 x 10^6
- Reason: 70% of the integral; need dense grid + high statistics
- tau_int ~ 130 (at peak) to 16000 (at T=0.1)
- At T=0.225: 10^8 / 133 = 750,000 independent samples
- CPU: 80 x 26 min = 34.7 hours

### Region 3: Mid-T (40 points)
- T = 0.55, 0.60, 0.65, ..., 2.0 (step 0.05 up to 1.0)
- T = 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 (step 0.5)
- nmeas = 10^7, therm = 5 x 10^5
- Reason: 23% of integral; tau_int ~ 5-7, very fast decorrelation
- CPU: 40 x 2.6 min = 1.7 hours

### Region 4: High-T (15 points)
- T = 6, 7, 8, 9, 10, 15, 20, 30, 40, 60, 80, 100, 200, 300, 400
- nmeas = 10^6, therm = 10^5
- Reason: < 4% of integral; dominated by 1/T^3 tail
- CPU: 15 x 0.26 min = 4 min

---

## 5. Computational Cost

| Item             | CPU time    |
|------------------|-------------|
| Low-T (5 pts)    | 13 min      |
| Peak (80 pts)    | 34.7 hours  |
| Mid-T (40 pts)   | 1.7 hours   |
| High-T (15 pts)  | 4 min       |
| **Total CPU**    | **~37 hours** |

With 30 parallel cores on `home` partition:
- **Wall time ~ 37h / 30 = 1.2 hours**
- Using 2 nodes (60 cores): ~40 minutes

---

## 6. Expected Results

| N   | gamma_MC | Deviation from 1.942 |
|-----|----------|---------------------|
| 64  | 1.891    | 2.6%                |
| 100 | 1.901    | 2.1%                |
| 128 | 1.921    | 1.1%                |
| 512 | ~1.937   | ~0.2-0.3%           |

The monotonic convergence of gamma_MC toward 1.942 with increasing N
confirms the mutual consistency of the Monte Carlo thermodynamics,
the Stirling approximation for S(inf), and the Simkin combinatorial formula.

---

## 7. SLURM Submission

```bash
# Single node, 32 CPUs, home partition (no time limit)
#SBATCH --partition=home
#SBATCH --cpus-per-task=32
```

Script: `run_N512.sh` — compiles mc_canonical.c with -O3,
runs all 140 temperature points with max 30 parallel processes,
merges output into `data_N512.dat`.

---

## 8. Post-Processing

```python
# Entropy extraction:
S_inf = lgamma(512**2 + 1) - lgamma(513) - lgamma(512**2 - 511) ) / 512
integral = trapz(Cv_over_N / T, T) + tail_correction
s0_MC = S_inf - integral
gamma_MC = ln(512) - s0_MC
```

Tail correction: fit Cv/N ~ c/T^2 at T_max, then integral from T_max to inf = c/(2*T_max^2).
