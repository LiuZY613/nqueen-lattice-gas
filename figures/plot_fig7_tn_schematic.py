"""
Fig. 7: Schematic tensor network for exact N-queens counting (4×4).
Four bond families with directional boundary stubs.
"""
import numpy as np
import matplotlib.pyplot as plt

FS = 14        # main labels (~10 pt after scaling to column width)
FS_sm = 11

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': FS,
    'mathtext.fontset': 'cm',
})

N = 4
sp = 2.4
r_node = 0.34

C_node  = '#2B5B8A'
C_row   = '#CC2222'
C_col   = '#2222CC'
C_diag  = '#228B22'
C_adiag = '#8B228B'

fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
ax.set_aspect('equal')
ax.axis('off')


def xy(i, j):
    return (j * sp, (N - 1 - i) * sp)


# ── bonds ────────────────────────────────────────────────────────────
lw_rc = 2.0
lw_d  = 1.3

for i in range(N):
    for j in range(N - 1):
        x0, y0 = xy(i, j); x1, y1 = xy(i, j + 1)
        ax.plot([x0, x1], [y0, y1], color=C_row, lw=lw_rc, zorder=1)

for i in range(N - 1):
    for j in range(N):
        x0, y0 = xy(i, j); x1, y1 = xy(i + 1, j)
        ax.plot([x0, x1], [y0, y1], color=C_col, lw=lw_rc, zorder=1)

for i in range(N - 1):
    for j in range(N - 1):
        x0, y0 = xy(i, j); x1, y1 = xy(i + 1, j + 1)
        ax.plot([x0, x1], [y0, y1], color=C_diag, lw=lw_d,
                ls='--', dashes=(4, 2.5), zorder=0)

for i in range(N - 1):
    for j in range(N - 1):
        x0, y0 = xy(i + 1, j); x1, y1 = xy(i, j + 1)
        ax.plot([x0, x1], [y0, y1], color=C_adiag, lw=lw_d,
                ls=':', dashes=(1.5, 2), zorder=0)

# ── nodes ────────────────────────────────────────────────────────────
for i in range(N):
    for j in range(N):
        x, y = xy(i, j)
        c = plt.Circle((x, y), r_node, fc='white', ec=C_node,
                        lw=2.0, zorder=5)
        ax.add_patch(c)
        ax.text(x, y, r'$T$', ha='center', va='center',
                fontsize=FS + 1, color=C_node, zorder=6,
                fontstyle='italic')

# ── physical index p ─────────────────────────────────────────────────
for i in range(N):
    for j in range(N):
        x, y = xy(i, j)
        ax.plot([x, x], [y + r_node + 0.02, y + r_node + 0.22],
                color='#999999', lw=1.0, zorder=3)
xp, yp = xy(0, 0)
ax.text(xp + 0.2, yp + r_node + 0.28, r'$p$', fontsize=FS_sm,
        color='#666666', ha='left', va='bottom')

# ── boundary stubs ───────────────────────────────────────────────────
stub = 0.40
cap  = 0.10
diag_stub = 0.32
sq2 = np.sqrt(2)
rr = r_node * 0.707   # offset along diagonal from node centre


def hv_stub(x, y, dx, dy, color, lw=1.8):
    """Horizontal or vertical boundary stub with a cap."""
    ax.plot([x, x + dx], [y, y + dy], color=color, lw=lw, zorder=2,
            solid_capstyle='round')
    norm = np.hypot(dx, dy)
    px, py = -dy / norm * cap, dx / norm * cap
    xe, ye = x + dx, y + dy
    ax.plot([xe - px, xe + px], [ye - py, ye + py],
            color=color, lw=lw, solid_capstyle='round', zorder=2)


def diag_stub(x, y, dx, dy, color, ls_kw=None):
    """Diagonal boundary stub (thinner)."""
    kw = dict(color=color, lw=1.3, zorder=2, solid_capstyle='round')
    if ls_kw:
        kw.update(ls_kw)
    ax.plot([x, x + dx], [y, y + dy], **kw)
    norm = np.hypot(dx, dy)
    px, py = -dy / norm * cap * 0.8, dx / norm * cap * 0.8
    xe, ye = x + dx, y + dy
    ax.plot([xe - px, xe + px], [ye - py, ye + py],
            color=color, lw=1.3, solid_capstyle='round', zorder=2)


ds = 0.32  # diagonal stub length component
se_kw = dict(ls='--', dashes=(3, 2))
ne_kw = dict(ls=':', dashes=(1.2, 1.8))

# Row: left v0, right v1
for i in range(N):
    xl, yl = xy(i, 0)
    hv_stub(xl - r_node, yl, -stub, 0, C_row)
    xr, yr = xy(i, N - 1)
    hv_stub(xr + r_node, yr, +stub, 0, C_row)

# Column: top v0, bottom v1
for j in range(N):
    xt, yt = xy(0, j)
    hv_stub(xt, yt + r_node, 0, +stub, C_col)
    xb, yb = xy(N - 1, j)
    hv_stub(xb, yb - r_node, 0, -stub, C_col)

# SE-diagonal (↘): starts on top/left edges, ends on bottom/right edges
# Only draw at non-corner edge nodes to reduce clutter
# Top edge (row 0, cols 1..N-2)
for j in range(1, N - 1):
    x, y = xy(0, j)
    diag_stub(x - rr, y + rr, -ds, +ds, C_diag, se_kw)
# Left edge (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    diag_stub(x - rr, y + rr, -ds, +ds, C_diag, se_kw)
# Bottom edge (row N-1, cols 1..N-2)
for j in range(1, N - 1):
    x, y = xy(N - 1, j)
    diag_stub(x + rr, y - rr, +ds, -ds, C_diag, se_kw)
# Right edge (rows 1..N-2, col N-1)
for i in range(1, N - 1):
    x, y = xy(i, N - 1)
    diag_stub(x + rr, y - rr, +ds, -ds, C_diag, se_kw)
# Corner stubs: only (0,0) start and (N-1,N-1) end
x, y = xy(0, 0)
diag_stub(x - rr, y + rr, -ds, +ds, C_diag, se_kw)
x, y = xy(N - 1, N - 1)
diag_stub(x + rr, y - rr, +ds, -ds, C_diag, se_kw)

# NE-diagonal (↗): starts on bottom/left edges, ends on top/right edges
# Bottom edge (row N-1, cols 1..N-2)
for j in range(1, N - 1):
    x, y = xy(N - 1, j)
    diag_stub(x - rr, y - rr, -ds, -ds, C_adiag, ne_kw)
# Left edge (rows 1..N-2, col 0)
for i in range(1, N - 1):
    x, y = xy(i, 0)
    diag_stub(x - rr, y - rr, -ds, -ds, C_adiag, ne_kw)
# Top edge (row 0, cols 1..N-2)
for j in range(1, N - 1):
    x, y = xy(0, j)
    diag_stub(x + rr, y + rr, +ds, +ds, C_adiag, ne_kw)
# Right edge (rows 1..N-2, col N-1)
for i in range(1, N - 1):
    x, y = xy(i, N - 1)
    diag_stub(x + rr, y + rr, +ds, +ds, C_adiag, ne_kw)
# Corner stubs: only (N-1,0) start and (0,N-1) end
x, y = xy(N - 1, 0)
diag_stub(x - rr, y - rr, -ds, -ds, C_adiag, ne_kw)
x, y = xy(0, N - 1)
diag_stub(x + rr, y + rr, +ds, +ds, C_adiag, ne_kw)

# ── boundary labels ──────────────────────────────────────────────────
# Row v0/v1: label at row 2 (middle-bottom, well away from diagonal labels)
xl, yl = xy(2, 0)
ax.text(xl - r_node - stub - 0.25, yl, r'$\mathbf{v}_0$',
        fontsize=FS, color=C_row, ha='right', va='center')
xr, yr = xy(2, N - 1)
ax.text(xr + r_node + stub + 0.25, yr, r'$\mathbf{v}_1$',
        fontsize=FS, color=C_row, ha='left', va='center')

# Column v0/v1: label at col 2
xt, yt = xy(0, 2)
ax.text(xt, yt + r_node + stub + 0.22, r'$\mathbf{v}_0$',
        fontsize=FS, color=C_col, ha='center', va='bottom')
xb, yb = xy(N - 1, 2)
ax.text(xb, yb - r_node - stub - 0.22, r'$\mathbf{v}_1$',
        fontsize=FS, color=C_col, ha='center', va='top')

# SE-diag v0 label: near (0,0) upper-left stub
x, y = xy(0, 0)
ax.text(x - rr - ds - 0.22, y + rr + ds + 0.18,
        r'$\mathbf{v}_0$', fontsize=FS, color=C_diag, ha='center', va='center')
# SE-diag v2 label: near (N-1,N-1) lower-right stub
x, y = xy(N - 1, N - 1)
ax.text(x + rr + ds + 0.22, y - rr - ds - 0.18,
        r'$\mathbf{v}_2$', fontsize=FS, color=C_diag, ha='center', va='center')

# NE-diag v0 label: near (N-1,0) lower-left stub
x, y = xy(N - 1, 0)
ax.text(x - rr - ds - 0.22, y - rr - ds - 0.18,
        r'$\mathbf{v}_0$', fontsize=FS, color=C_adiag, ha='center', va='center')
# NE-diag v2 label: near (0,N-1) upper-right stub
x, y = xy(0, N - 1)
ax.text(x + rr + ds + 0.22, y + rr + ds + 0.18,
        r'$\mathbf{v}_2$', fontsize=FS, color=C_adiag, ha='center', va='center')

# ── contraction arrow ────────────────────────────────────────────────
arr_x = -2.0
y_top = xy(0, 0)[1]; y_bot = xy(N - 1, 0)[1]
ax.annotate('', xy=(arr_x, y_bot - 0.2), xytext=(arr_x, y_top + 0.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.6))
ax.text(arr_x - 0.12, (y_top + y_bot) / 2,
        'contraction', fontsize=FS_sm, ha='right', va='center',
        rotation=90, style='italic')

# ── limits ───────────────────────────────────────────────────────────
ax.set_xlim(-3.0, (N - 1) * sp + 2.0)
ax.set_ylim(-2.0, (N - 1) * sp + 2.0)

plt.savefig('fig7_tn_schematic.png', dpi=400, bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
plt.savefig('fig7_tn_schematic.pdf', bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
print("Done")
plt.close()
