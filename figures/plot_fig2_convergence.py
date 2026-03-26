"""
plot_fig2_convergence.py — PRE Fig.2: Convergence diagnostics (vertical layout)
  (a) Acceptance rate vs T/J  (top panel)
  (b) Integrated autocorrelation time tau_int vs T/J  (bottom panel)

Single-column width (3.4"), 10pt fonts to match revtex4-2 body text.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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

COL_WIDTH = 3.4  # PRE single-column width in inches
PANEL_HEIGHT = 2.5  # height per panel

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Load data
# ============================================================
Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#000000', '#e41a1c', '#984ea3']
markers = ['o', 's', '^', 'D', 'v', 'h', 'p', '*']

data = {}
for N in Ns:
    fpath = os.path.join(BASE, f'data_N{N}.dat')
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} not found, skipping N={N}")
        continue
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) >= 8:
                rows.append([float(x) for x in vals[:8]])
    if rows:
        data[N] = np.array(rows)
        print(f"N={N}: loaded {len(rows)} points")

# ============================================================
# Figure 2 — vertical layout (top-bottom)
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2 * PANEL_HEIGHT))

# --- Panel (a): Acceptance rate (top) ---
for i, N in enumerate(Ns):
    if N not in data:
        continue
    d = data[N]
    mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
    if not np.any(mask):
        continue
    T = d[mask, 0]
    acc = d[mask, 5]
    T_targets = np.linspace(T[0], T[-1], 12)
    mk_idx = sorted(set(np.argmin(np.abs(T[:, None] - T_targets[None, :]), axis=0)))
    ax1.plot(T, acc, '-', color=colors[i], linewidth=0.8, alpha=0.9)
    ax1.plot(T[mk_idx], acc[mk_idx], markers[i], color=colors[i],
             markersize=3, markerfacecolor='none', markeredgewidth=0.6,
             label=f'$N={N}$')

ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel('Acceptance rate')
ax1.set_xlim(0, 1.0)
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-6, top=1.0)
ax1.legend(loc='center right', frameon=True, fancybox=False,
           edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
           handletextpad=0.3)
ax1.text(0.97, 0.05, r'$\bf{(a)}$', transform=ax1.transAxes,
         fontsize=10, va='bottom', ha='right')

# --- Panel (b): Autocorrelation time (bottom) ---
for i, N in enumerate(Ns):
    if N not in data:
        continue
    d = data[N]
    mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
    if not np.any(mask):
        continue
    T = d[mask, 0]
    tau = d[mask, 7]
    T_targets = np.linspace(T[0], T[-1], 12)
    mk_idx = sorted(set(np.argmin(np.abs(T[:, None] - T_targets[None, :]), axis=0)))
    ax2.plot(T, tau, '-', color=colors[i], linewidth=0.8, alpha=0.9)
    ax2.plot(T[mk_idx], tau[mk_idx], markers[i], color=colors[i],
             markersize=3, markerfacecolor='none', markeredgewidth=0.6,
             label=f'$N={N}$')

ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel(r'$\tau_{\rm int}$ (sweeps)')
ax2.set_xlim(0, 1.0)
ax2.set_yscale('log')
ax2.set_ylim(bottom=0.3)
ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.75), frameon=True,
           fancybox=False, edgecolor='0.7', framealpha=0.9, ncol=2,
           columnspacing=0.8, handletextpad=0.3)
ax2.text(0.97, 0.05, r'$\bf{(b)}$', transform=ax2.transAxes,
         fontsize=10, va='bottom', ha='right')

plt.tight_layout(h_pad=0.5)
try:
    fig.savefig(os.path.join(BASE, 'fig2_convergence.pdf'), dpi=300)
except PermissionError:
    print("Warning: PDF locked, skipping PDF output")
fig.savefig(os.path.join(BASE, 'fig2_convergence.png'), dpi=300)
plt.close()
print("Saved fig2_convergence.pdf/png")
