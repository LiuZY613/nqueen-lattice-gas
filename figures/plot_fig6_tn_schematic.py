"""
Fig. 6: Schematic tensor network for exact N-queens counting (4×4).
Four bond families with directional arrows, all solid lines.
"""
import numpy as np
import matplotlib.pyplot as plt

FS = 14
FS_sm = 11

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': FS,
    'mathtext.fontset': 'cm',
})

N = 4
sp = 3.5
r_node = 0.34

C_node  = '#2B5B8A'
C_row   = '#CC2222'
C_col   = '#2222CC'
C_diag  = '#228B22'
C_adiag = '#8B228B'

fig, ax = plt.subplots(1, 1, figsize=(8.5, 8.5))
ax.set_aspect('equal')
ax.axis('off')


def xy(i, j):
    return (j * sp, (N - 1 - i) * sp)


# ── arrow drawing helpers ────────────────────────────────────────────
def draw_arrow(x0, y0, x1, y1, color, lw, ms=14, zorder=1):
    """Draw a solid arrow from (x0,y0) to (x1,y1)."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0),
                zorder=zorder)


def node_bond(i0, j0, i1, j1, color, lw, zorder=1):
    """Bond with two arrowheads both in propagation direction (v0→v1)."""
    x0, y0 = xy(i0, j0)
    x1, y1 = xy(i1, j1)
    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    g = r_node + 0.04
    sx, sy = x0 + ux*g, y0 + uy*g   # source node edge
    ex, ey = x1 - ux*g, y1 - uy*g   # dest node edge
    # 1) Plain line for the full bond
    ax.plot([sx, ex], [sy, ey], color=color, lw=lw,
            solid_capstyle='butt', zorder=zorder)
    # 2) Two arrowheads, both pointing source → dest
    a = 0.45  # arrow segment length
    # Arrowhead near source node (pointing forward)
    draw_arrow(sx, sy, sx + ux*a, sy + uy*a,
               color, lw, 14, zorder + 1)
    # Arrowhead near dest node (pointing forward)
    draw_arrow(ex - ux*a, ey - uy*a, ex, ey,
               color, lw, 14, zorder + 1)


# ── bonds (all solid, all arrowed) ──────────────────────────────────
lw_rc = 2.0
lw_d  = 1.5

# Row bonds (→)
for i in range(N):
    for j in range(N - 1):
        node_bond(i, j, i, j + 1, C_row, lw_rc, 1)

# Column bonds (↓)
for i in range(N - 1):
    for j in range(N):
        node_bond(i, j, i + 1, j, C_col, lw_rc, 1)

# SE-diagonal bonds (↘) — solid
for i in range(N - 1):
    for j in range(N - 1):
        node_bond(i, j, i + 1, j + 1, C_diag, lw_d, 0)

# NE-diagonal bonds (↗) — solid
for i in range(N - 1):
    for j in range(N - 1):
        node_bond(i + 1, j, i, j + 1, C_adiag, lw_d, 0)

# ── nodes (drawn on top of bonds) ───────────────────────────────────
for i in range(N):
    for j in range(N):
        x, y = xy(i, j)
        c = plt.Circle((x, y), r_node, fc='white', ec=C_node,
                        lw=2.0, zorder=5)
        ax.add_patch(c)
        ax.text(x - 0.03, y - 0.03, r'$C$', ha='center', va='center',
                fontsize=FS + 1, color=C_node, zorder=6)

# ── boundary stubs (arrowed) ────────────────────────────────────────
stub = 0.55
rr = r_node * 0.707   # diagonal offset from node centre
ds = 0.45             # diagonal stub length component
ms_s = 11             # mutation scale for stub arrowheads
lw_stub = 1.8
lw_dstub = 1.3

# Row stubs — all arrows point →
for i in range(N):
    xl, yl = xy(i, 0)
    draw_arrow(xl - r_node - stub, yl, xl - r_node - 0.02, yl,
               C_row, lw_stub, ms_s, 2)
    xr, yr = xy(i, N - 1)
    draw_arrow(xr + r_node + 0.02, yr, xr + r_node + stub, yr,
               C_row, lw_stub, ms_s, 2)

# Column stubs — all arrows point ↓
for j in range(N):
    xt, yt = xy(0, j)
    draw_arrow(xt, yt + r_node + stub, xt, yt + r_node + 0.02,
               C_col, lw_stub, ms_s, 2)
    xb, yb = xy(N - 1, j)
    draw_arrow(xb, yb - r_node - 0.02, xb, yb - r_node - stub,
               C_col, lw_stub, ms_s, 2)

# SE-diagonal stubs (↘): entry from upper-left, exit to lower-right
# Top edge entry (row 0, cols 0..N-2)
for j in range(N - 1):
    x, y = xy(0, j)
    draw_arrow(x - rr - ds, y + rr + ds, x - rr - 0.01, y + rr + 0.01,
               C_diag, lw_dstub, ms_s, 2)
# Left edge entry (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    draw_arrow(x - rr - ds, y + rr + ds, x - rr - 0.01, y + rr + 0.01,
               C_diag, lw_dstub, ms_s, 2)
# Bottom edge exit (row N-1, cols 1..N-1)
for j in range(1, N):
    x, y = xy(N - 1, j)
    draw_arrow(x + rr + 0.01, y - rr - 0.01, x + rr + ds, y - rr - ds,
               C_diag, lw_dstub, ms_s, 2)
# Right edge exit (rows 1..N-2, col N-1)
for i in range(1, N - 1):
    x, y = xy(i, N - 1)
    draw_arrow(x + rr + 0.01, y - rr - 0.01, x + rr + ds, y - rr - ds,
               C_diag, lw_dstub, ms_s, 2)
# Corner (N-1,N-1) exit
x, y = xy(N - 1, N - 1)
draw_arrow(x + rr + 0.01, y - rr - 0.01, x + rr + ds, y - rr - ds,
           C_diag, lw_dstub, ms_s, 2)

# NE-diagonal stubs (↗): entry from lower-left, exit to upper-right
# Bottom edge entry (row N-1, cols 0..N-2)
for j in range(N - 1):
    x, y = xy(N - 1, j)
    draw_arrow(x - rr - ds, y - rr - ds, x - rr - 0.01, y - rr - 0.01,
               C_adiag, lw_dstub, ms_s, 2)
# Left edge entry (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    draw_arrow(x - rr - ds, y - rr - ds, x - rr - 0.01, y - rr - 0.01,
               C_adiag, lw_dstub, ms_s, 2)
# Top edge exit (row 0, cols 1..N-1)
for j in range(1, N):
    x, y = xy(0, j)
    draw_arrow(x + rr + 0.01, y + rr + 0.01, x + rr + ds, y + rr + ds,
               C_adiag, lw_dstub, ms_s, 2)
# Right edge exit (rows 1..N-2, col N-1)
for i in range(1, N - 1):
    x, y = xy(i, N - 1)
    draw_arrow(x + rr + 0.01, y + rr + 0.01, x + rr + ds, y + rr + ds,
               C_adiag, lw_dstub, ms_s, 2)
# Corner (0,N-1) exit
x, y = xy(0, N - 1)
draw_arrow(x + rr + 0.01, y + rr + 0.01, x + rr + ds, y + rr + ds,
           C_adiag, lw_dstub, ms_s, 2)

# ── single-node diagonal stubs (corners missing one diagonal type) ──
# (0,0): NE-diag entry ↙ + exit ↗
x, y = xy(0, 0)
draw_arrow(x - rr - ds, y - rr - ds, x - rr - 0.01, y - rr - 0.01,
           C_adiag, lw_dstub, ms_s, 2)
draw_arrow(x + rr + 0.01, y + rr + 0.01, x + rr + ds, y + rr + ds,
           C_adiag, lw_dstub, ms_s, 2)
# (0,N-1): SE-diag entry ↖ + exit ↘
x, y = xy(0, N - 1)
draw_arrow(x - rr - ds, y + rr + ds, x - rr - 0.01, y + rr + 0.01,
           C_diag, lw_dstub, ms_s, 2)
draw_arrow(x + rr + 0.01, y - rr - 0.01, x + rr + ds, y - rr - ds,
           C_diag, lw_dstub, ms_s, 2)
# (N-1,0): SE-diag entry ↖ + exit ↘
x, y = xy(N - 1, 0)
draw_arrow(x - rr - ds, y + rr + ds, x - rr - 0.01, y + rr + 0.01,
           C_diag, lw_dstub, ms_s, 2)
draw_arrow(x + rr + 0.01, y - rr - 0.01, x + rr + ds, y - rr - ds,
           C_diag, lw_dstub, ms_s, 2)
# (N-1,N-1): NE-diag entry ↙ + exit ↗
x, y = xy(N - 1, N - 1)
draw_arrow(x - rr - ds, y - rr - ds, x - rr - 0.01, y - rr - 0.01,
           C_adiag, lw_dstub, ms_s, 2)
draw_arrow(x + rr + 0.01, y + rr + 0.01, x + rr + ds, y + rr + ds,
           C_adiag, lw_dstub, ms_s, 2)

# ── boundary labels (every stub labeled: 36 total) ──────────────────
FS_lab = 15

# Offsets: push labels just past the stub tips, different per type
off_rc  = 0.12   # row / column label gap from stub tip
off_d   = 0.10   # diagonal label gap from stub tip (along 45°)

# --- Row entry v0 (left edge) ---
for i in range(N):
    x, y = xy(i, 0)
    ax.text(x - r_node - stub - off_rc, y, r'$\mathbf{v}_0$',
            fontsize=FS_lab, color=C_row, ha='right', va='center')

# --- Row exit v1 (right edge) ---
for i in range(N):
    x, y = xy(i, N - 1)
    ax.text(x + r_node + stub + off_rc, y, r'$\mathbf{v}_1$',
            fontsize=FS_lab, color=C_row, ha='left', va='center')

# --- Column entry v0 (top edge) ---
for j in range(N):
    x, y = xy(0, j)
    ax.text(x, y + r_node + stub + off_rc, r'$\mathbf{v}_0$',
            fontsize=FS_lab, color=C_col, ha='center', va='bottom')

# --- Column exit v1 (bottom edge) ---
for j in range(N):
    x, y = xy(N - 1, j)
    ax.text(x, y - r_node - stub - off_rc, r'$\mathbf{v}_1$',
            fontsize=FS_lab, color=C_col, ha='center', va='top')

# --- SE-diag entry v0 (↘ incoming from upper-left) ---
# top edge (row 0, cols 0..N-2)
for j in range(N - 1):
    x, y = xy(0, j)
    tx = x - rr - ds - off_d
    ty = y + rr + ds + off_d
    ax.text(tx, ty, r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_diag,
            ha='right', va='bottom')
# left edge (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    tx = x - rr - ds - off_d
    ty = y + rr + ds + off_d
    ax.text(tx, ty, r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_diag,
            ha='right', va='bottom')

# --- SE-diag exit v2 (↘ outgoing to lower-right) ---
_se_exit = set()
for j in range(1, N):
    _se_exit.add((N - 1, j))
for i in range(1, N - 1):
    _se_exit.add((i, N - 1))
_se_exit.add((N - 1, N - 1))
for (i, j) in _se_exit:
    x, y = xy(i, j)
    tx = x + rr + ds + off_d
    ty = y - rr - ds - off_d
    ax.text(tx, ty, r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_diag,
            ha='left', va='top')

# --- NE-diag entry v0 (↗ incoming from lower-left) ---
# bottom edge (row N-1, cols 0..N-2)
for j in range(N - 1):
    x, y = xy(N - 1, j)
    tx = x - rr - ds - off_d
    ty = y - rr - ds - off_d
    ax.text(tx, ty, r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_adiag,
            ha='right', va='top')
# left edge (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    tx = x - rr - ds - off_d
    ty = y - rr - ds - off_d
    ax.text(tx, ty, r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_adiag,
            ha='right', va='top')

# --- NE-diag exit v2 (↗ outgoing to upper-right) ---
_ne_exit = set()
for j in range(1, N):
    _ne_exit.add((0, j))
for i in range(1, N - 1):
    _ne_exit.add((i, N - 1))
_ne_exit.add((0, N - 1))
for (i, j) in _ne_exit:
    x, y = xy(i, j)
    tx = x + rr + ds + off_d
    ty = y + rr + ds + off_d
    ax.text(tx, ty, r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_adiag,
            ha='left', va='bottom')

# --- Single-node diagonal corner labels ---
# (0,0): NE-diag v0 entry ↙ + v2 exit ↗
x, y = xy(0, 0)
ax.text(x - rr - ds - off_d, y - rr - ds - off_d,
        r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_adiag,
        ha='right', va='top')
ax.text(x + rr + ds + off_d, y + rr + ds + off_d,
        r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_adiag,
        ha='left', va='bottom')
# (0,N-1): SE-diag v0 entry ↖ + v2 exit ↘
x, y = xy(0, N - 1)
ax.text(x - rr - ds - off_d, y + rr + ds + off_d,
        r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_diag,
        ha='right', va='bottom')
ax.text(x + rr + ds + off_d, y - rr - ds - off_d,
        r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_diag,
        ha='left', va='top')
# (N-1,0): SE-diag v0 entry ↖ + v2 exit ↘
x, y = xy(N - 1, 0)
ax.text(x - rr - ds - off_d, y + rr + ds + off_d,
        r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_diag,
        ha='right', va='bottom')
ax.text(x + rr + ds + off_d, y - rr - ds - off_d,
        r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_diag,
        ha='left', va='top')
# (N-1,N-1): NE-diag v0 entry ↙ + v2 exit ↗
x, y = xy(N - 1, N - 1)
ax.text(x - rr - ds - off_d, y - rr - ds - off_d,
        r'$\mathbf{v}_0$', fontsize=FS_lab, color=C_adiag,
        ha='right', va='top')
ax.text(x + rr + ds + off_d, y + rr + ds + off_d,
        r'$\mathbf{v}_2$', fontsize=FS_lab, color=C_adiag,
        ha='left', va='bottom')

# ── contraction arrow ────────────────────────────────────────────────
# ── limits ───────────────────────────────────────────────────────────
ax.set_xlim(-2.5, (N - 1) * sp + 3.2)
ax.set_ylim(-2.8, (N - 1) * sp + 2.8)

plt.savefig('fig6_tn_schematic.png', dpi=400, bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
plt.savefig('fig6_tn_schematic.pdf', bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
print("Done")
plt.close()
