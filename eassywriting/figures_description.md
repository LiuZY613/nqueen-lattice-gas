# PRE Figures Description — Queen Lattice Gas (M=N)

## Data

Six system sizes: N = 8, 16, 32, 64, 100, 128 (M = N queens on an N x N board).

Each size has 222 temperature points merged from three sources:
- **Precise**: 30 points, T = 0.05 ~ 10.0
- **Dense**: 185 points, T = 0.5 ~ 20.0 (step 0.1, excluding overlap with precise)
- **High-T**: 7 points, T = 25, 40, 60, 80, 100, 200, 400

All simulations: 10^8 measurement sweeps, 2 x 10^6 thermalization sweeps, 200 jackknife bins.

Data files: `data_N{8,16,32,64,100,128}.dat`

---

## Figure 1: Energy per queen (`fig1_energy.pdf`)

Two-panel figure (7 x 2.8 inches).

### Panel (a) — Full temperature range
- **X-axis**: T/J, **log scale**
- **Y-axis**: E/M, linear
- **Content**: 6 curves (N = 8, 16, 32, 64, 100, 128) with line + markers (markers evenly spaced in log T)
- **Additional**: black dashed horizontal line at E/M = 5/3 (exact high-T limit for N -> infinity)
- **Range**: T = 0.05 ~ 400

### Panel (b) — Low-temperature zoom
- **X-axis**: T/J, **linear**, range 0 ~ 2
- **Y-axis**: E/M, linear
- **Content**: 6 curves with error bars
- **No** log scale on either axis

---

## Figure 2: Specific heat per queen (`fig2_specific_heat.pdf`)

Two-panel figure (7 x 2.8 inches).

### Panel (a) — Peak region
- **X-axis**: T/J, **linear**, range 0 ~ 1
- **Y-axis**: C_v/M, linear
- **Content**: 6 MC curves with error bars
- **Additional**: red dashed line = mean-field prediction, Eq. (22):
  C_v/M = (1 + gamma) / (4T^2) * exp(-1/(2T)), where gamma ≈ 1.942
- Peak location: T* ≈ 0.225 J, peak height ≈ 1.62

### Panel (b) — Full temperature range
- **X-axis**: T/J, **log scale**
- **Y-axis**: C_v/M, linear
- **Content**: 6 MC curves with line + markers (markers evenly spaced in log T)
- **Additional**: same red dashed mean-field line over full T range
- Shows MF agreement with MC at high T and the Schottky-like peak shape
- **Range**: T = 0.05 ~ 400

---

## Figure 3: Entropy increment per queen (`fig3_entropy.pdf`)

Two-panel figure (7 x 2.8 inches).

Entropy computed via thermodynamic integration from lowest T upward:
Delta S/M = integral from T_min to T of (C_v / M) / T' dT' (trapezoidal rule).
Initial entropy set to 0 at T_min = 0.05.

### Panel (a) — Full temperature range
- **X-axis**: T/J, **log scale**
- **Y-axis**: Delta S/M, **linear** (not log)
- **Content**: 6 MC curves with line + markers (markers evenly spaced in log T)
- **Additional line 1**: red dashed = mean-field entropy, computed as:
  Delta S_MF/M = integral from 0 to T of (1+gamma)/(4T'^3) * exp(-1/(2T')) dT'
- **Additional line 2**: black dashed horizontal line at 1 + gamma ≈ 2.94 (theoretical total entropy change from T=0 to T=infinity)
- **Range**: T = 0.001 ~ 400

### Panel (b) — Low-temperature zoom
- **X-axis**: T/J, **linear**, range 0 ~ 1.5
- **Y-axis**: Delta S/M, **linear** (not log)
- **Content**: 6 MC curves with markers
- **Additional**: same red dashed MF entropy line and black dashed 2.94 line
- Shows entropy buildup through the Schottky crossover region

---

## Style

- PRE double-column format, serif font (Times), 9 pt base font
- Panel labels: **(a)**, **(b)** in bold, top-left corner
- Tick marks: inward, on all four sides, minor ticks visible
- Output: PDF (vector) + PNG (300 dpi)
- Color scheme: blue, orange, green, purple, brown, cyan (colorblind-friendly); red reserved for mean-field theory
