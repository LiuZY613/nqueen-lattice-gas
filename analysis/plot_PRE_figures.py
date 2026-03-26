"""
plot_PRE_figures.py — PRE publication figures for queen lattice gas (N-queens)

Generates 3 figures in vertical (top-bottom) layout at single-column width:
  Fig 3: E/N vs T  (a: log-T full range + 5/3 line; b: low-T linear)
  Fig 4: C_v/N vs T  (a: T=0~1; b: all T log-T + mean-field theory)
  Fig 5: Mean-field vs Monte Carlo  (a: E/N; b: C_v/N)

Single-column width (3.4"), 10pt fonts to match revtex4-2 body text.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import lgamma
import os
import glob

# ============================================================
# PRE style — 10pt to match revtex4-2 body text at column width
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
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
    'savefig.pad_inches': 0.01,
})

COL_WIDTH = 3.4   # PRE single-column width
PANEL_HEIGHT = 2.5  # height per panel

BASE = os.path.dirname(os.path.abspath(__file__))


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
Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#000000', '#e41a1c', '#984ea3']
markers = ['o', 's', '^', 'D', 'v', 'h', 'p', '*']

merged = {}

for N in Ns:
    fpath = os.path.join(BASE, f'data_N{N}.dat')
    if os.path.exists(fpath):
        d = load(fpath)
        if d.shape[0] > 0:
            merged[N] = d
            print(f"  N={N}: loaded {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})")
    else:
        print(f"  N={N}: file not found, skipping")

print()

# Columns: T(0) E/N(1) err(2) Cv/N(3) err(4) acc(5) E_tot(6) tau(7)


# ============================================================
# Fig 3: E/N vs T — vertical layout
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2 * PANEL_HEIGHT))

# --- Top panel (a): E/N vs T (log scale), full range ---
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    ax1.plot(d[:, 0], d[:, 1], '-', color=colors[i], linewidth=0.8, alpha=0.9)
    logT = np.log10(d[:, 0])
    target = np.linspace(logT[0], logT[-1], 12)
    mk_idx = sorted(set([np.argmin(np.abs(logT - t)) for t in target]))
    ax1.plot(d[mk_idx, 0], d[mk_idx, 1], markers[i], color=colors[i],
             markersize=3, markerfacecolor='none', markeredgewidth=0.6,
             label=f'$N={N}$')

ax1.axhline(y=5.0/3.0, color='k', linestyle='--', linewidth=0.8,
            label=r'$E/N = 5/3$', zorder=0)
ax1.set_xscale('log')
ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$E/N$')
ax1.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Bottom panel (b): low-T E/N, linear axes ---
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
ax2.set_ylabel(r'$E/N$')
ax2.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(h_pad=0.5)
fig.savefig(os.path.join(BASE, 'fig3_energy.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig3_energy.png'), dpi=300)
plt.close()
print("Saved fig3_energy.pdf/png")


# ============================================================
# Fig 4: Cv/N vs T — vertical layout
# ============================================================
gamma_val = 1.942

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2 * PANEL_HEIGHT))

# --- Top panel (a): Cv/N vs T, T=0 to 1 ---
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

ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$C_v/N$')
ax1.set_xlim(0, 1.0)
ax1.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Inset: zoom into peak region for large N ---
axins = ax1.inset_axes([0.42, 0.08, 0.45, 0.45])
for i, N in enumerate(Ns):
    if N not in merged or N < 64:
        continue
    d = merged[N]
    mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
    if np.any(mask):
        axins.errorbar(d[mask, 0], d[mask, 3], yerr=d[mask, 4],
                       fmt=markers[i]+'-', color=colors[i], markersize=2.0,
                       capsize=0.8, linewidth=0.7, markerfacecolor='none',
                       markeredgewidth=0.4)
axins.set_xlim(0.19, 0.23)
axins.set_ylim(1.56, 1.63)
axins.tick_params(labelsize=6, width=0.4, length=2)
axins.set_xticks([0.19, 0.21, 0.23])
axins.set_yticks([1.57, 1.60, 1.63])
ax1.indicate_inset_zoom(axins, edgecolor='0.5', linewidth=0.6, alpha=0.8)

# --- Bottom panel (b): Cv/N vs T (all T, log scale) ---
for i, N in enumerate(Ns):
    if N not in merged:
        continue
    d = merged[N]
    mask = d[:, 3] > 1e-8
    if np.any(mask):
        T_cv = d[mask, 0]
        Cv_cv = d[mask, 3]
        ax2.plot(T_cv, Cv_cv, '-', color=colors[i], linewidth=0.8, alpha=0.9)
        logT = np.log10(T_cv)
        target = np.linspace(logT[0], logT[-1], 12)
        mk_idx = sorted(set([np.argmin(np.abs(logT - t)) for t in target]))
        ax2.plot(T_cv[mk_idx], Cv_cv[mk_idx], markers[i],
                 color=colors[i], markersize=3, markerfacecolor='none',
                 markeredgewidth=0.6, label=f'$N={N}$')

ax2.set_xscale('log')
ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$C_v/N$')
ax2.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(h_pad=0.5)
fig.savefig(os.path.join(BASE, 'fig4_cv.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig4_cv.png'), dpi=300)
plt.close()
print("Saved fig4_cv.pdf/png")


# ============================================================
# Fig 5: Mean-field vs Monte Carlo — vertical layout
# ============================================================

# Modified Poisson mean-field calculation
from scipy.optimize import brentq
from scipy.special import gammaln  # log(k!) = gammaln(k+1)

def modified_poisson_energy(mu, beta, kmax=30):
    """Compute expected collision energy f(mu, beta) for one attack line.
    P(k) = lambda^k / k! * exp(-beta*k*(k-1)/2) / g
    Find lambda s.t. <n> = mu, then return <k(k-1)/2>.
    """
    if beta > 50 and mu <= 1.0:
        return 0.0
    if mu < 1e-10:
        return 0.0

    def compute_probs(lam):
        """Compute log-weights, then normalize to probabilities."""
        log_w = np.zeros(kmax + 1)
        for k in range(kmax + 1):
            log_w[k] = k * np.log(max(lam, 1e-300)) - gammaln(k + 1) \
                        - beta * k * (k - 1) / 2.0
        log_w -= np.max(log_w)  # subtract max for numerical stability
        w = np.exp(log_w)
        return w / np.sum(w)

    def mean_n(lam):
        p = compute_probs(lam)
        return np.sum(np.arange(kmax + 1) * p)

    def energy_from_lam(lam):
        p = compute_probs(lam)
        ks = np.arange(kmax + 1, dtype=float)
        return np.sum(ks * (ks - 1) / 2.0 * p)

    try:
        lam_lo = 1e-10
        lam_hi = mu * np.exp(beta / 2) * 10 + 10
        if mean_n(lam_hi) < mu:
            lam_hi *= 100
        if mean_n(lam_lo) > mu:
            return 0.0
        lam = brentq(lambda l: mean_n(l) - mu, lam_lo, lam_hi, xtol=1e-12)
        return energy_from_lam(lam)
    except (ValueError, RuntimeError):
        return 0.0


def compute_MF_energy(T_arr, N_diag_pts=50):
    """Compute modified Poisson MF energy per queen E/N(T)."""
    E_MF = np.zeros(len(T_arr))
    mu_diag = np.linspace(0.02, 1.0, N_diag_pts)

    for it, T in enumerate(T_arr):
        beta = 1.0 / T
        # Rows + columns: mu = 1, 2 families
        f_rc = modified_poisson_energy(1.0, beta)
        e_rc = 2.0 * f_rc

        # Diagonals: integrate over mu from 0 to 1, 4 families
        f_diag = np.array([modified_poisson_energy(m, beta) for m in mu_diag])
        e_diag = 4.0 * np.trapz(f_diag, mu_diag)

        E_MF[it] = e_rc + e_diag
        if it % 20 == 0:
            print(f"  T={T:.3f}: E_rc={e_rc:.4f}, E_diag={e_diag:.4f}, E/N={E_MF[it]:.4f}")

    return E_MF


# Temperature grid for mean-field
T_mf = np.linspace(0.1, 2.0, 100)
print("Computing modified Poisson mean-field...")
E_MF = compute_MF_energy(T_mf)
# Numerical derivative for Cv
dT = T_mf[1] - T_mf[0]
Cv_MF = np.gradient(E_MF, dT)
print("Mean-field computation done.")

# Simple MF for comparison
T_simple = np.linspace(0.05, 2.0, 500)
E_simple = (1.0 + gamma_val) / 2.0 * np.exp(-1.0 / (2.0 * T_simple))
Cv_simple = ((1.0 + gamma_val) / (4.0 * T_simple**2)) * np.exp(-1.0 / (2.0 * T_simple))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2 * PANEL_HEIGHT))

# --- Top panel (a): E/N comparison ---
# MC data (only N=128, 256 for clarity)
for i, N in enumerate(Ns):
    if N not in merged or N < 128:
        continue
    d = merged[N]
    mask = d[:, 0] <= 2.0
    ax1.errorbar(d[mask, 0], d[mask, 1], yerr=d[mask, 2],
                 fmt=markers[i]+'-', color=colors[i], markersize=2.5,
                 capsize=1, linewidth=0.8, markerfacecolor='none',
                 markeredgewidth=0.5, label=f'MC $N={N}$')

# Modified Poisson MF
ax1.plot(T_mf, E_MF, '-', color='#377eb8', linewidth=1.5,
         label='Modified Poisson MF', zorder=10)
# Simple MF
ax1.plot(T_simple, E_simple, '--', color='#377eb8', linewidth=1.0,
         label='Simple MF', zorder=9)

ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel(r'$E/N$')
ax1.set_xlim(0, 2.0)
ax1.legend(loc='lower right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, fontsize=7.5,
           handletextpad=0.3)
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# --- Bottom panel (b): Cv/N comparison ---
for i, N in enumerate(Ns):
    if N not in merged or N < 128:
        continue
    d = merged[N]
    mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
    if np.any(mask):
        ax2.errorbar(d[mask, 0], d[mask, 3], yerr=d[mask, 4],
                     fmt=markers[i]+'-', color=colors[i], markersize=2.5,
                     capsize=1, linewidth=0.8, markerfacecolor='none',
                     markeredgewidth=0.5, label=f'MC $N={N}$')

# Modified Poisson MF Cv
mask_cv = (T_mf >= 0.1) & (T_mf <= 1.0)
ax2.plot(T_mf[mask_cv], Cv_MF[mask_cv], '-', color='#377eb8', linewidth=1.5,
         label='Modified Poisson MF', zorder=10)
# Simple MF Cv
mask_s = (T_simple >= 0.05) & (T_simple <= 1.0)
ax2.plot(T_simple[mask_s], Cv_simple[mask_s], '--', color='#377eb8', linewidth=1.0,
         label='Simple MF', zorder=9)

ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$C_v/N$')
ax2.set_xlim(0, 1.0)
ax2.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, fontsize=7.5,
           handletextpad=0.3)
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

plt.tight_layout(h_pad=0.5)
fig.savefig(os.path.join(BASE, 'fig5_meanfield.pdf'), dpi=300)
fig.savefig(os.path.join(BASE, 'fig5_meanfield.png'), dpi=300)
plt.close()
print("Saved fig5_meanfield.pdf/png")

print("\nAll PRE figures generated successfully!")
