"""
plot_PRE_figures.py — PRE publication figures for queen lattice gas (M=N case)

Generates 3 double-panel figures:
  Fig 1: E/M vs T (left: log-T full range + 5/3 line; right: low-T linear)
  Fig 2: C_v/M vs T (left: T=0~1; right: all T log-T + mean-field theory)
  Fig 3: Delta S/M vs T (left: all T log-T + 2.94 line; right: T=0~1.5 linear)

All data: precise (30 pts) + dense (T=0.5~20) + high-T (T=25~400)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import lgamma
import os
import glob

# ============================================================
# PRE style settings
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'errorbar.capsize': 1.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# PRE double-column width: 7 inches
FIG_WIDTH = 7.0
FIG_HEIGHT = 2.8

BASE = os.path.dirname(os.path.abspath(__file__))
TASK2 = os.path.dirname(BASE)  # task2_N等于L folder
PRECISE_DIR = os.path.join(TASK2, '精确结果')
DENSE_DIR = os.path.join(TASK2, '密集结果')
HIGHT_DIR = os.path.join(TASK2, '高温结果')


def load(path):
    """Load data file, skip comment lines."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) >= 8:
                data.append([float(x) for x in vals[:8]])
    return np.array(data) if data else np.empty((0, 8))


# ============================================================
# Load and merge all data
# ============================================================
Ns = [8, 16, 32, 64, 100, 128]

# Color scheme: professional, colorblind-friendly (avoid red, reserved for MF line)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#17becf']
markers = ['o', 's', '^', 'D', 'v', 'h']

merged = {}

for N in Ns:
    parts = []

    # Precise results (30 temperature points, T=0.05~10.0)
    f_precise = os.path.join(PRECISE_DIR, f'data_L{N}.dat')
    if os.path.exists(f_precise):
        d = load(f_precise)
        if d.shape[0] > 0:
            parts.append(d)
            print(f"  N={N}: precise {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})")

    # Dense results (T=0.5~20.0, step 0.1)
    f_dense = os.path.join(DENSE_DIR, f'data_L{N}_dense.dat')
    if os.path.exists(f_dense):
        d = load(f_dense)
        if d.shape[0] > 0:
            parts.append(d)
            print(f"  N={N}: dense {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})")

    # High-T results (T=25,40,60,80,100,200,400)
    f_highT = os.path.join(HIGHT_DIR, f'data_L{N}_highT.dat')
    if os.path.exists(f_highT):
        d = load(f_highT)
        if d.shape[0] > 0:
            parts.append(d)
            print(f"  N={N}: high-T {d.shape[0]} pts (T={d[0,0]:.1f}~{d[-1,0]:.1f})")

    if parts:
        all_data = np.vstack(parts)
        # Sort by temperature, remove duplicates
        _, unique_idx = np.unique(all_data[:, 0], return_index=True)
        merged[N] = all_data[unique_idx]
        print(f"  N={N}: merged {merged[N].shape[0]} temperature points "
              f"(T={merged[N][0,0]:.3f}~{merged[N][-1,0]:.1f})")

print()

# Columns: T(0) E/N(1) err(2) Cv/N(3) err(4) acc(5) E_tot(6) tau(7)
# Note: In the code, E/N means E/M (per queen), Cv/N means Cv/M


# ============================================================
# Figure 1: E/M vs T
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# --- Left panel: E/M vs T (log scale), full range ---
# Select marker positions uniformly in log(T) space for even visual spacing
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    ax1.plot(d[:, 0], d[:, 1], '-', color=colors[i], linewidth=0.8, alpha=0.9)
    # Pick ~15 markers evenly in log space to cover full T range including high-T
    logT = np.log10(d[:, 0])
    target = np.linspace(logT[0], logT[-1], 15)
    mk_idx = [np.argmin(np.abs(logT - t)) for t in target]
    mk_idx = sorted(set(mk_idx))
    ax1.plot(d[mk_idx, 0], d[mk_idx, 1], markers[i], color=colors[i],
             markersize=3, markerfacecolor='none', markeredgewidth=0.6,
             label=f'$N={N}$')

# 5/3 dashed line
ax1.axhline(y=5.0/3.0, color='k', linestyle='--', linewidth=0.8,
            label=r'$E/M = 5/3$', zorder=0)

ax1.set_xscale('log')
ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$E/M$')
ax1.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Right panel: low-T E/M, linear axes ---
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    mask = d[:, 0] <= 2.0
    ax2.errorbar(d[mask, 0], d[mask, 1], yerr=d[mask, 2],
                 fmt=markers[i]+'-', color=colors[i], markersize=2.5,
                 capsize=1, linewidth=0.8, markerfacecolor='none',
                 markeredgewidth=0.5, label=f'$N={N}$')

ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$E/M$')
ax2.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(BASE, 'fig1_energy.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig1_energy.png'), dpi=300)
plt.close()
print("Saved fig1_energy.pdf/png")


# ============================================================
# Figure 2: Cv/M vs T
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# --- Left panel: Cv/M vs T, T=0 to 1 ---
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
    if np.any(mask):
        ax1.errorbar(d[mask, 0], d[mask, 3], yerr=d[mask, 4],
                     fmt=markers[i]+'-', color=colors[i], markersize=2.5,
                     capsize=1, linewidth=0.8, markerfacecolor='none',
                     markeredgewidth=0.5, label=f'$N={N}$')

# Mean-field theory: C_v/M = (1+gamma)/(4T^2) * exp(-1/(2T))  [Eq. 22]
gamma_euler = 1.942  # Euler-type constant from Simkin formula
T_mf = np.linspace(0.05, 1.0, 500)
Cv_mf = ((1.0 + gamma_euler) / (4.0 * T_mf**2)) * np.exp(-1.0 / (2.0 * T_mf))
ax1.plot(T_mf, Cv_mf, color='#e41a1c', linestyle='--', linewidth=1.5,
         label='Mean field', zorder=10)

ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$C_v/M$')
ax1.set_xlim(0, 1.0)
ax1.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Right panel: Cv/M vs T (all T, log scale) ---
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    mask = d[:, 3] > 1e-8
    if np.any(mask):
        T_cv = d[mask, 0]
        Cv_cv = d[mask, 3]
        ax2.plot(T_cv, Cv_cv, '-', color=colors[i], linewidth=0.8, alpha=0.9)
        # Markers evenly in log space
        logT = np.log10(T_cv)
        target = np.linspace(logT[0], logT[-1], 12)
        mk_idx = [np.argmin(np.abs(logT - t)) for t in target]
        mk_idx = sorted(set(mk_idx))
        ax2.plot(T_cv[mk_idx], Cv_cv[mk_idx], markers[i],
                 color=colors[i], markersize=3, markerfacecolor='none',
                 markeredgewidth=0.6, label=f'$N={N}$')

# Mean-field theory on full range
T_mf_full = np.logspace(np.log10(0.05), np.log10(500), 1000)
Cv_mf_full = ((1.0 + gamma_euler) / (4.0 * T_mf_full**2)) * np.exp(-1.0 / (2.0 * T_mf_full))
ax2.plot(T_mf_full, Cv_mf_full, color='#e41a1c', linestyle='--', linewidth=1.5,
         label='Mean field', zorder=10)

ax2.set_xscale('log')
ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$C_v/M$')
ax2.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(BASE, 'fig2_specific_heat.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig2_specific_heat.png'), dpi=300)
plt.close()
print("Saved fig2_specific_heat.pdf/png")


# ============================================================
# Figure 3: Delta S/M vs T (thermodynamic integration from low T)
# ============================================================

# Compute entropy increment: integrate Cv/(M*T) from T_min upward
# Set S(T_min) = 0, so Delta S/M = integral from T_min to T of Cv/(M*T') dT'
entropy_data = {}

for N in Ns:
    if N not in merged:
        continue
    d = merged[N]
    T = d[:, 0].copy()
    Cv_M = d[:, 3].copy()  # This is Cv/M (= Cv/N in code notation)

    # Sort by T (should already be sorted)
    idx = np.argsort(T)
    T = T[idx]
    Cv_M = Cv_M[idx]

    # Integrate from lowest T upward using trapezoidal rule
    # Delta S/M (T_i) = sum_{j=0}^{i-1} 0.5*(Cv_M[j]/T[j] + Cv_M[j+1]/T[j+1]) * (T[j+1]-T[j])
    dS_M = np.zeros(len(T))
    for j in range(1, len(T)):
        dT = T[j] - T[j-1]
        integrand = 0.5 * (Cv_M[j-1] / T[j-1] + Cv_M[j] / T[j])
        dS_M[j] = dS_M[j-1] + integrand * dT

    entropy_data[N] = {'T': T, 'dS_M': dS_M}
    print(f"  N={N}: Delta S/M at T_max={T[-1]:.1f} is {dS_M[-1]:.4f}")

print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# --- Left panel: Delta S/M vs T (full range, log-T x-axis) ---
for i, N in enumerate(Ns):
    if N not in entropy_data:
        continue
    r = entropy_data[N]
    mask = r['T'] > 0
    T_plot = r['T'][mask]
    S_plot = r['dS_M'][mask]
    ax1.plot(T_plot, S_plot, '-', color=colors[i], linewidth=0.8, alpha=0.9)
    # Markers evenly spaced in log(T)
    logT = np.log10(T_plot)
    target = np.linspace(logT[0], logT[-1], 12)
    mk_idx = [np.argmin(np.abs(logT - t)) for t in target]
    mk_idx = sorted(set(mk_idx))
    ax1.plot(T_plot[mk_idx], S_plot[mk_idx], markers[i],
             color=colors[i], markersize=3, markerfacecolor='none',
             markeredgewidth=0.6, label=f'$N={N}$')

# Mean-field entropy: Delta S_MF/M = integral_0^T (1+gamma)/(4T'^3) * exp(-1/(2T')) dT'
from scipy.integrate import cumulative_trapezoid
T_mf_s = np.linspace(0.001, 500, 50000)
integrand_mf = ((1.0 + gamma_euler) / (4.0 * T_mf_s**3)) * np.exp(-1.0 / (2.0 * T_mf_s))
dS_mf = np.zeros(len(T_mf_s))
dS_mf[1:] = cumulative_trapezoid(integrand_mf, T_mf_s)

ax1.plot(T_mf_s, dS_mf, color='#e41a1c', linestyle='--', linewidth=1.5,
         label='Mean field', zorder=10)

# 2.94 dashed line (= 1 + gamma, theoretical total entropy change)
ax1.axhline(y=2.94, color='k', linestyle='--', linewidth=0.8,
            label=r'$1+\gamma \approx 2.94$', zorder=0)

ax1.set_xscale('log')
ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$\Delta S/M$')
ax1.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Right panel: Delta S/M vs T, T=0~1.5, linear ---
for i, N in enumerate(Ns):
    if N not in entropy_data:
        continue
    r = entropy_data[N]
    mask = r['T'] <= 1.5
    ax2.plot(r['T'][mask], r['dS_M'][mask], markers[i]+'-', color=colors[i],
             markersize=2.5, linewidth=0.8, markerfacecolor='none',
             markeredgewidth=0.5, label=f'$N={N}$')

# Mean-field entropy on right panel
mask_mf = T_mf_s <= 1.5
ax2.plot(T_mf_s[mask_mf], dS_mf[mask_mf], color='#e41a1c', linestyle='--',
         linewidth=1.5, label='Mean field', zorder=10)

ax2.axhline(y=2.94, color='k', linestyle='--', linewidth=0.8,
            label=r'$1+\gamma \approx 2.94$', zorder=0)

ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$\Delta S/M$')
ax2.set_xlim(0, 1.5)
ax2.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(BASE, 'fig3_entropy.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig3_entropy.png'), dpi=300)
plt.close()
print("Saved fig3_entropy.pdf/png")

print("\nAll PRE figures generated successfully!")
