"""
plot_fig1_schematic.py — PRE-style Fig 1 for queen lattice gas

Panel (a): Single queen on 8x8 board, attacked squares gray, unattacked white
Panel (b): One of the 92 non-attacking 8-queens solutions, checkerboard pattern
Both panels: same size, no outer border, PRE font sizes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PRE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.pad_inches': 0.01,
})

N = 8
# One of the 92 solutions for 8-queens (0-indexed rows for each column)
solution = [0, 4, 7, 5, 2, 6, 1, 3]

def get_attacked_squares(queen_positions):
    """Return set of (row, col) attacked by queens at given positions."""
    if isinstance(queen_positions, tuple):
        queen_positions = [queen_positions]
    attacked = set()
    for qr, qc in queen_positions:
        for i in range(N):
            attacked.add((qr, i))  # row
            attacked.add((i, qc))  # column
            for d in range(-N, N):
                if 0 <= qr + d < N and 0 <= qc + d < N:
                    attacked.add((qr + d, qc + d))
                if 0 <= qr + d < N and 0 <= qc - d < N:
                    attacked.add((qr + d, qc - d))
    return attacked

def draw_board_attack(ax, queen_list, title_label):
    """Panel (a): gray = attacked, white = unattacked."""
    attacked = set()
    for q in queen_list:
        attacked.update(get_attacked_squares(q))

    for r in range(N):
        for c in range(N):
            if (r, c) in attacked and (r, c) not in queen_list:
                facecolor = '#D0D0D0'
            else:
                facecolor = '#FFFFFF'
            rect = plt.Rectangle((c, N - 1 - r), 1, 1,
                                  facecolor=facecolor,
                                  edgecolor='#888888', linewidth=0.5)
            ax.add_patch(rect)

    for qr, qc in queen_list:
        circle = plt.Circle((qc + 0.5, N - 1 - qr + 0.5), 0.3,
                             facecolor='black', edgecolor='none', zorder=5)
        ax.add_patch(circle)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.3, N + 0.3, title_label, fontsize=10, fontweight='bold',
            va='bottom', ha='left')

def draw_board_checkerboard(ax, queen_list, title_label):
    """Panel (b): alternating gray-white checkerboard like a real chess board."""
    color_light = '#FFFFFF'
    color_dark = '#D0D0D0'

    for r in range(N):
        for c in range(N):
            if (r + c) % 2 == 0:
                facecolor = color_light
            else:
                facecolor = color_dark
            rect = plt.Rectangle((c, N - 1 - r), 1, 1,
                                  facecolor=facecolor,
                                  edgecolor='#888888', linewidth=0.5)
            ax.add_patch(rect)

    for qr, qc in queen_list:
        circle = plt.Circle((qc + 0.5, N - 1 - qr + 0.5), 0.3,
                             facecolor='black', edgecolor='none', zorder=5)
        ax.add_patch(circle)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.3, N + 0.3, title_label, fontsize=10, fontweight='bold',
            va='bottom', ha='left')

# Create figure - two panels side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.8))
fig.subplots_adjust(wspace=0.3)

# Panel (a): single queen at center, show attacked squares
draw_board_attack(ax1, [(3, 3)], '(a)')

# Panel (b): 8-queens solution with checkerboard
queens_solution = [(solution[c], c) for c in range(N)]
draw_board_checkerboard(ax2, queens_solution, '(b)')

# Both panels same ylim (no annotations on panel b)
both_ylim = (-0.3, N + 0.6)
ax1.set_ylim(both_ylim)
ax2.set_ylim(both_ylim)

plt.savefig('C:/Users/刘宗岳/Desktop/eassy/figures/fig1_schematic.pdf', format='pdf')
plt.savefig('C:/Users/刘宗岳/Desktop/eassy/figures/fig1_schematic.png', format='png', dpi=300)
plt.close()
print("Fig 1 saved.")
