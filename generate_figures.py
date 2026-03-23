"""
Generate all figures for the PRE manuscript.
Figures are saved as PDF for LaTeX inclusion.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import os

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
    'text.usetex': False,
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

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, 'figures')
TASK3 = os.path.join(os.path.dirname(os.path.dirname(BASE)),
                      'nqueen_simulation', 'task3_N等于L2除2')

# gamma constant from Simkin formula
GAMMA = 1.942

os.makedirs(FIGDIR, exist_ok=True)

def thin_data(T, n_target=60):
    """Select ~n_target evenly spaced indices in T, thinning dense regions."""
    if len(T) <= n_target:
        return np.arange(len(T))
    target = np.linspace(T[0], T[-1], n_target)
    idx = sorted(set([np.argmin(np.abs(T - t)) for t in target]))
    return np.array(idx)

def load(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data) if data else np.empty((0, 0))


# ==================================================================
# Load data
# ==================================================================
Ls_task2 = [8, 16, 32, 64, 100, 128]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#17becf']
markers = ['o', 's', '^', 'D', 'v', 'h']

# PRE double-column width
FIG_WIDTH = 7.0
FIG_HEIGHT = 2.8

data2 = {}
for L in Ls_task2:
    # Load from local figures directory (318 temperature points)
    fpath = os.path.join(FIGDIR, f'data_N{L}.dat')
    if os.path.exists(fpath):
        d = load(fpath)
        if d.shape[0] > 0:
            data2[L] = d
            print(f"  Loaded N={L}: {d.shape[0]} points (T={d[0,0]:.3f}~{d[-1,0]:.1f})")

# Task3 data
Ls_task3 = [4, 8, 16, 32, 64]
Ns_task3 = {4: 8, 8: 32, 16: 128, 32: 512, 64: 2048}
data3 = {}
for L in Ls_task3:
    N = Ns_task3[L]
    fpath = os.path.join(FIGDIR, f'data_L{L}_N{N}.dat')
    if os.path.exists(fpath):
        d = load(fpath)
        if d.shape[0] > 0:
            data3[L] = d


# ==================================================================
# Figure 1: Schematic (created with matplotlib)
# ==================================================================
def fig1_schematic():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    # Panel (a): board with queen and attack lines (white background)
    N = 8
    ax = ax1

    # Place a queen at (3, 4) (row 3, col 4, 0-indexed)
    qr, qc = 3, 4

    # Determine which squares are attacked
    attacked = {}  # (i,j) -> color
    for j in range(N):
        if j != qc:
            attacked[(qr, j)] = 'red'
    for i in range(N):
        if i != qr:
            attacked[(i, qc)] = 'blue'
    for i in range(N):
        j = qc + (i - qr)
        if 0 <= j < N and i != qr:
            if (i, j) not in attacked:
                attacked[(i, j)] = 'orange'
    for i in range(N):
        j = qc - (i - qr)
        if 0 <= j < N and i != qr:
            if (i, j) not in attacked:
                attacked[(i, j)] = 'purple'

    # Draw board: white for non-attacked, colored for attacked
    for i in range(N):
        for j in range(N):
            if (i, j) in attacked:
                ax.add_patch(Rectangle((j, N - 1 - i), 1, 1,
                             facecolor=attacked[(i, j)], alpha=0.25, edgecolor='gray', lw=0.3, zorder=2))
            else:
                ax.add_patch(Rectangle((j, N - 1 - i), 1, 1,
                             facecolor='white', edgecolor='gray', lw=0.3))

    ax.plot(qc + 0.5, N - 1 - qr + 0.5, 'ko', markersize=14, zorder=5)
    ax.plot(qc + 0.5, N - 1 - qr + 0.5, 'wo', markersize=10, zorder=6)
    ax.text(qc + 0.5, N - 1 - qr + 0.5, 'Q', ha='center', va='center',
            fontsize=9, fontweight='bold', zorder=7)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(a) Queen attack lines', fontsize=10)

    # Annotations
    ax.annotate('row', xy=(0.5, N - 1 - qr + 0.5), fontsize=7, color='red', fontweight='bold')
    ax.annotate('col', xy=(qc + 0.55, N - 0.3), fontsize=7, color='blue', fontweight='bold')

    # Panel (b): L and lambda illustration
    ax = ax2
    L = 8
    lam = 1  # lattice spacing = 1

    # Draw simplified board
    for i in range(L):
        for j in range(L):
            color = '#f0f0f0' if (i + j) % 2 == 0 else '#d0d0d0'
            ax.add_patch(Rectangle((j, L - 1 - i), 1, 1, facecolor=color, edgecolor='gray', lw=0.3))

    # Show L dimension
    ax.annotate('', xy=(L, -0.5), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(L / 2, -0.9, r'$L$', ha='center', va='top', fontsize=13, fontweight='bold')

    # Show lambda (one cell)
    ax.annotate('', xy=(1, -1.5), xytext=(0, -1.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(0.5, -1.9, r'$\lambda$', ha='center', va='top', fontsize=13, color='red', fontweight='bold')

    # Show N = L/lambda queens on the board (one per row, non-attacking)
    queen_cols = [1, 5, 0, 6, 3, 7, 2, 4]  # a valid 8-queens solution
    for i in range(L):
        j = queen_cols[i]
        ax.plot(j + 0.5, L - 1 - i + 0.5, 'ko', markersize=10)
        ax.plot(j + 0.5, L - 1 - i + 0.5, 'wo', markersize=7)

    ax.text(L + 0.3, L / 2, r'$N = L/\lambda$' + '\nqueens', fontsize=9,
            ha='left', va='center', style='italic')

    ax.set_xlim(-0.5, L + 2.5)
    ax.set_ylim(-2.5, L + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(b) Length scales: $L$ and $\\lambda$', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig1_schematic.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig1_schematic.pdf")


# ==================================================================
# Figure 2: E/M vs T
# ==================================================================
def fig2_energy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # --- Panel (a): log-T full range ---
    for i, L in enumerate(Ls_task2):
        if L not in data2:
            continue
        d = data2[L]
        ax1.plot(d[:, 0], d[:, 1], '-', color=colors[i], linewidth=0.8, alpha=0.9)
        # Markers evenly spaced in log(T)
        logT = np.log10(d[:, 0])
        target = np.linspace(logT[0], logT[-1], 15)
        mk_idx = sorted(set([np.argmin(np.abs(logT - t)) for t in target]))
        ax1.plot(d[mk_idx, 0], d[mk_idx, 1], markers[i], color=colors[i],
                 markersize=3, markerfacecolor='none', markeredgewidth=0.6,
                 label=f'$N={L}$')

    ax1.axhline(y=5.0/3.0, color='k', linestyle='--', linewidth=0.8,
                label=r'$E/M = 5/3$', zorder=0)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$T/J$')
    ax1.set_ylabel(r'$E/M$')
    ax1.legend(loc='lower right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
               handletextpad=0.3)
    ax1.text(0.03, 0.95, r'$\bf{(a)}$', transform=ax1.transAxes,
             fontsize=10, va='top')

    # --- Panel (b): linear T<=2 with error bars ---
    for i, L in enumerate(Ls_task2):
        if L not in data2:
            continue
        d = data2[L]
        mask = d[:, 0] <= 2.0
        dm = d[mask]
        idx = thin_data(dm[:, 0], n_target=50)
        ax2.plot(dm[:, 0], dm[:, 1], '-', color=colors[i], linewidth=0.8,
                 alpha=0.9)
        ax2.errorbar(dm[idx, 0], dm[idx, 1], yerr=dm[idx, 2],
                     fmt=markers[i], color=colors[i], markersize=2.5,
                     capsize=1, linewidth=0, markerfacecolor='none',
                     markeredgewidth=0.5, label=f'$N={L}$')

    ax2.set_xlabel(r'$T/J$')
    ax2.set_ylabel(r'$E/M$')
    ax2.legend(loc='lower right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
               handletextpad=0.3)
    ax2.text(0.03, 0.95, r'$\bf{(b)}$', transform=ax2.transAxes,
             fontsize=10, va='top')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(os.path.join(FIGDIR, 'fig2_energy.pdf'))
    plt.close()
    print("Saved fig2_energy.pdf")


# ==================================================================
# Figure 3: Cv/M vs T (no mean-field)
# ==================================================================
def fig3_cv():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # --- Panel (a): T=0~1, linear, with error bars ---
    for i, L in enumerate(Ls_task2):
        if L not in data2:
            continue
        d = data2[L]
        mask = (d[:, 0] >= 0.01) & (d[:, 0] <= 1.0)
        if np.any(mask):
            dm = d[mask]
            idx = thin_data(dm[:, 0], n_target=40)
            ax1.plot(dm[:, 0], dm[:, 3], '-', color=colors[i], linewidth=0.8,
                     alpha=0.9)
            ax1.errorbar(dm[idx, 0], dm[idx, 3], yerr=dm[idx, 4],
                         fmt=markers[i], color=colors[i], markersize=2.5,
                         capsize=1, linewidth=0, markerfacecolor='none',
                         markeredgewidth=0.5, label=f'$N={L}$')

    ax1.set_xlabel(r'$T/J$')
    ax1.set_ylabel(r'$C_v/M$')
    ax1.set_xlim(0, 1.0)
    ax1.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
               handletextpad=0.3)
    ax1.text(0.03, 0.95, r'$\bf{(a)}$', transform=ax1.transAxes,
             fontsize=10, va='top')

    # --- Panel (b): full T range, log scale ---
    for i, L in enumerate(Ls_task2):
        if L not in data2:
            continue
        d = data2[L]
        mask = d[:, 3] > 1e-8
        if np.any(mask):
            T_cv = d[mask, 0]
            Cv_cv = d[mask, 3]
            ax2.plot(T_cv, Cv_cv, '-', color=colors[i], linewidth=0.8, alpha=0.9)
            # Markers evenly in log space
            logT = np.log10(T_cv)
            target = np.linspace(logT[0], logT[-1], 12)
            mk_idx = sorted(set([np.argmin(np.abs(logT - t)) for t in target]))
            ax2.plot(T_cv[mk_idx], Cv_cv[mk_idx], markers[i],
                     color=colors[i], markersize=3, markerfacecolor='none',
                     markeredgewidth=0.6, label=f'$N={L}$')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$T/J$')
    ax2.set_ylabel(r'$C_v/M$')
    ax2.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, ncol=2, columnspacing=0.8,
               handletextpad=0.3)
    ax2.text(0.03, 0.95, r'$\bf{(b)}$', transform=ax2.transAxes,
             fontsize=10, va='top')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(os.path.join(FIGDIR, 'fig3_cv.pdf'))
    plt.close()
    print("Saved fig3_cv.pdf")


# ==================================================================
# Figure 4: Mean-field comparison — N=100, N=128 + simple MF + mod. Poisson MF
# ==================================================================
def fig4_meanfield():
    """Mean-field theory vs MC for N=100,128: E/M and Cv/M comparison"""
    from scipy.optimize import brentq

    def log_factorial(k):
        if k <= 1:
            return 0.0
        return sum(np.log(i) for i in range(1, k + 1))

    LOG_FACT = [log_factorial(k) for k in range(80)]

    def compute_log_terms(log_lam, beta, kmax=50):
        lt = np.full(kmax + 1, -np.inf)
        for k in range(kmax + 1):
            lt[k] = k * log_lam - LOG_FACT[k] - beta * k * (k - 1) / 2.0
        return lt

    def logsumexp(lv):
        m = np.max(lv)
        if m == -np.inf:
            return -np.inf
        return m + np.log(np.sum(np.exp(lv - m)))

    def compute_mean_nn1(log_lam, beta, kmax=50):
        lt = compute_log_terms(log_lam, beta, kmax)
        log_Z = logsumexp(lt)
        ln_terms = np.full(kmax + 1, -np.inf)
        for k in range(1, kmax + 1):
            ln_terms[k] = np.log(k) + lt[k]
        mean_n = np.exp(logsumexp(ln_terms) - log_Z)
        lnn1 = np.full(kmax + 1, -np.inf)
        for k in range(2, kmax + 1):
            lnn1[k] = np.log(k * (k - 1)) + lt[k]
        if np.max(lnn1) > -np.inf:
            mean_nn1 = np.exp(logsumexp(lnn1) - log_Z)
        else:
            mean_nn1 = 0.0
        return mean_n, mean_nn1

    def find_log_lam(mu, beta, kmax=50):
        if mu < 1e-15:
            return -100.0
        def eq(ll):
            mn, _ = compute_mean_nn1(ll, beta, kmax)
            return mn - mu
        try:
            return brentq(eq, -40, 80, xtol=1e-12, maxiter=200)
        except:
            return np.log(max(mu, 1e-300))

    def f_en(mu, beta, kmax=50):
        if mu < 1e-15:
            return 0.0
        ll = find_log_lam(mu, beta, kmax)
        _, mnn1 = compute_mean_nn1(ll, beta, kmax)
        return mnn1 / 2.0

    def E_MF(T, nq=150):
        beta = 1.0 / T
        e_rc = 2.0 * f_en(1.0, beta)
        mu_pts = np.linspace(0.005, 1.0, nq)
        f_vals = np.array([f_en(mu, beta) for mu in mu_pts])
        mu_full = np.concatenate([[0.0], mu_pts])
        f_full = np.concatenate([[0.0], f_vals])
        e_dg = 4.0 * np.trapz(f_full, mu_full)
        return e_rc + e_dg

    print("Computing modified Poisson MF curve...")
    T_arr = np.concatenate([
        np.linspace(0.05, 0.55, 80),
        np.linspace(0.56, 1.0, 20),
        np.linspace(1.1, 3.0, 15),
    ])
    E_arr = np.array([E_MF(T) for T in T_arr])
    # Cv by numerical differentiation
    Cv_arr = np.zeros_like(T_arr)
    for i, T in enumerate(T_arr):
        dT = max(T * 1e-4, 1e-5)
        Cv_arr[i] = (E_MF(T + dT) - E_MF(T - dT)) / (2 * dT)

    # Simple MF: E = (1+gamma)/2 * e^{-1/(2T)}, Cv = (1+gamma)/(4T^2) * e^{-1/(2T)}
    T_s = np.linspace(0.05, 3.0, 300)
    E_simple = (1 + GAMMA) / 2.0 * np.exp(-0.5 / T_s)
    Cv_simple = (1 + GAMMA) / (4.0 * T_s**2) * np.exp(-0.5 / T_s)

    # Only plot N=100 and N=128
    mc_sizes = [100, 128]
    mc_colors = ['#1f77b4', '#ff7f0e']
    mc_markers = ['o', 's']

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Panel (a): E/M, T=0~2
    ax = axes[0]
    for j, L in enumerate(mc_sizes):
        if L not in data2:
            continue
        d = data2[L]
        mask = d[:, 0] <= 2.0
        dm = d[mask]
        idx = thin_data(dm[:, 0], n_target=40)
        ax.plot(dm[:, 0], dm[:, 1], '-', color=mc_colors[j], linewidth=0.8, alpha=0.9)
        ax.plot(dm[idx, 0], dm[idx, 1], mc_markers[j], color=mc_colors[j],
                markersize=2.5, markerfacecolor='none',
                markeredgewidth=0.5, label=f'$N={L}$')

    mask_mf = T_arr <= 2.0
    ax.plot(T_arr[mask_mf], E_arr[mask_mf], 'k-', linewidth=1.5, label='Mod. Poisson MF')
    mask_s = T_s <= 2.0
    ax.plot(T_s[mask_s], E_simple[mask_s], 'k--', linewidth=1.2, label='Simple MF', alpha=0.7)
    ax.set_xlabel(r'$T/J$')
    ax.set_ylabel(r'$E/M$')
    ax.set_xlim(0, 2.0)
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='0.7', framealpha=0.9, handletextpad=0.3)
    ax.text(0.03, 0.95, r'$\bf{(a)}$', transform=ax.transAxes,
            fontsize=10, va='top')

    # Panel (b): Cv/M, T=0~1
    ax = axes[1]
    for j, L in enumerate(mc_sizes):
        if L not in data2:
            continue
        d = data2[L]
        mask = (d[:, 0] <= 1.0) & (d[:, 3] > 1e-8)
        dm = d[mask]
        idx = thin_data(dm[:, 0], n_target=40)
        ax.plot(dm[:, 0], dm[:, 3], '-', color=mc_colors[j], linewidth=0.8, alpha=0.9)
        ax.plot(dm[idx, 0], dm[idx, 3], mc_markers[j], color=mc_colors[j],
                markersize=2.5, markerfacecolor='none',
                markeredgewidth=0.5, label=f'$N={L}$')

    mask_mf = T_arr <= 1.0
    ax.plot(T_arr[mask_mf], Cv_arr[mask_mf], 'k-', linewidth=1.5, label='Mod. Poisson MF')
    mask_s = T_s <= 1.0
    ax.plot(T_s[mask_s], Cv_simple[mask_s], 'k--', linewidth=1.2, label='Simple MF', alpha=0.7)
    ax.set_xlabel(r'$T/J$')
    ax.set_ylabel(r'$C_v/M$')
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='0.7', framealpha=0.9, handletextpad=0.3)
    ax.text(0.03, 0.95, r'$\bf{(b)}$', transform=ax.transAxes,
            fontsize=10, va='top')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(os.path.join(FIGDIR, 'fig4_meanfield.pdf'))
    plt.close()
    print("Saved fig4_meanfield.pdf")


# ==================================================================
# Figure 6: Half-filling M = N^2/2
# ==================================================================
def fig5_halffilling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    hf_colors = ['#e74c3c', '#e67e22', '#2ca02c', '#2980b9', '#8e44ad']
    hf_markers = ['o', 's', '^', 'D', 'v']

    for i, L in enumerate(Ls_task3):
        if L not in data3:
            continue
        d = data3[L]
        N = Ns_task3[L]
        mask = d[:, 1] > 0
        dm = d[mask]
        idx = thin_data(dm[:, 0], n_target=35)
        ax1.plot(dm[:, 0], dm[:, 1], '-', color=hf_colors[i], linewidth=0.8, alpha=0.9)
        ax1.plot(dm[idx, 0], dm[idx, 1], hf_markers[i], color=hf_colors[i],
                 markersize=3, markerfacecolor='none', markeredgewidth=0.5,
                 label=f'$N={L}$, $M={N}$')

    ax1.set_xlabel(r'$T/J$')
    ax1.set_ylabel(r'$E/M$')
    ax1.set_xlim(0, 3)
    ax1.legend(loc='lower right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, fontsize=6.5,
               handletextpad=0.3)
    ax1.text(0.03, 0.95, r'$\bf{(a)}$', transform=ax1.transAxes,
             fontsize=10, va='top')

    for i, L in enumerate(Ls_task3):
        if L not in data3:
            continue
        d = data3[L]
        mask = d[:, 3] > 0
        dm = d[mask]
        idx = thin_data(dm[:, 0], n_target=35)
        ax2.plot(dm[:, 0], dm[:, 3], '-', color=hf_colors[i], linewidth=0.8, alpha=0.9)
        ax2.plot(dm[idx, 0], dm[idx, 3], hf_markers[i], color=hf_colors[i],
                 markersize=3, markerfacecolor='none', markeredgewidth=0.5,
                 label=f'$N={L}$')

    ax2.set_xlabel(r'$T/J$')
    ax2.set_ylabel(r'$C_v/M$')
    ax2.set_xlim(0, 2)
    ax2.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor='0.7', framealpha=0.9, fontsize=6.5,
               handletextpad=0.3)
    ax2.text(0.03, 0.95, r'$\bf{(b)}$', transform=ax2.transAxes,
             fontsize=10, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig5_halffilling.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig5_halffilling.pdf")


# ==================================================================
# Generate all
# ==================================================================
if __name__ == '__main__':
    print("Generating figures for PRE manuscript...")
    fig1_schematic()
    fig2_energy()
    fig3_cv()
    fig4_meanfield()
    fig5_halffilling()
    print("\nAll figures generated!")
