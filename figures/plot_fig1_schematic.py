"""
plot_fig1_schematic.py — PRE-style Fig 1 for n-queens lattice gas

Single panel: One of the 92 non-attacking 8-queens solutions on a
standard chessboard with queen symbols.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# PRE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.pad_inches': 0.02,
})

N = 8
# One of the 92 solutions for 8-queens (0-indexed rows for each column)
solution = [0, 4, 7, 5, 2, 6, 1, 3]

# Classic chessboard colors
COLOR_LIGHT = '#F0D9B5'
COLOR_DARK  = '#B58863'

fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

# Draw board squares
for r in range(N):
    for c in range(N):
        if (r + c) % 2 == 0:
            facecolor = COLOR_LIGHT
        else:
            facecolor = COLOR_DARK
        rect = plt.Rectangle((c, N - 1 - r), 1, 1,
                              facecolor=facecolor, edgecolor='none')
        ax.add_patch(rect)

# Thin outer border
border = plt.Rectangle((0, 0), N, N, facecolor='none',
                        edgecolor='#4a3728', linewidth=1.2)
ax.add_patch(border)

# Draw queens as chess symbols with stroke for visibility on both colors
queen_char = '\u265B'  # ♛ solid black chess queen
for c in range(N):
    r = solution[c]
    # White queen with black outline — visible on both light and dark squares
    ax.text(c + 0.5, N - 1 - r + 0.5, queen_char,
            fontsize=26, ha='center', va='center',
            fontfamily='Segoe UI Symbol',
            color='#1a1a1a', zorder=5,
            path_effects=[pe.withStroke(linewidth=0.8, foreground='white')])

ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.3)
plt.savefig('C:/Users/刘宗岳/Desktop/eassy/figures/fig1_schematic.pdf', format='pdf')
plt.savefig('C:/Users/刘宗岳/Desktop/eassy/figures/fig1_schematic.png', format='png', dpi=300)
plt.close()
print("Fig 1 saved.")
