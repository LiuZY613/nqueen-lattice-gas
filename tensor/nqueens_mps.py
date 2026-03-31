#!/usr/bin/env python3
"""
N-Queens Problem — MPS/MPO Tensor Network Solver
=================================================

Represents the boundary state between rows as a Matrix Product State (MPS)
over N column-sites (local dimension d=8), and applies the row transfer as
two Matrix Product Operators (MPOs):

  Phase 1: Queen placement + column update + SE diagonal shift  (MPO bond dim 4)
  Phase 2: NE diagonal left-shift                                (MPO bond dim 2)

After each MPO application, the MPS is compressed via SVD.

State encoding per column-site:
    s = 4*c + 2*dse + dne   (s ∈ {0..7})
    c:   column occupied (0/1)
    dse: ↘ diagonal attack signal arriving at this column (0/1)
    dne: ↗ diagonal attack signal arriving at this column (0/1)

Queen MPO (left→right, bond dim 4):
    Bond = (row_placed, delta_SE):
      row_placed: has a queen been placed in this row? (0→1 transition)
      delta_SE:   SE signal being shifted rightward from the left neighbor

NE-shift MPO (left→right, bond dim 2):
    Bond carries the NE signal flowing right→left across sites.
    At each site, deposits the NE signal from the right neighbor and
    sends the current NE signal to the left.

This is mathematically equivalent to the bitmask transfer matrix solver,
but uses a factored MPS representation instead of an explicit state dictionary.

Usage:
    python nqueens_mps.py                     # demo for N=1..14
    python nqueens_mps.py 8                   # solve N=8
    python nqueens_mps.py 8 --max-chi 32      # approximate with bond dim cap
    python nqueens_mps.py 8 --verbose          # show per-row bond dimensions
"""

import numpy as np
import time
import sys


# =============================================================================
# State encoding: 3 bits per column-site → local dimension 8
# =============================================================================

D = 8  # local Hilbert space dimension per site


def _enc(c, dse, dne):
    """Encode (col_used, diag_SE, diag_NE) → integer 0..7."""
    return (c << 2) | (dse << 1) | dne


def _dec(s):
    """Decode integer 0..7 → (col_used, diag_SE, diag_NE)."""
    return (s >> 2) & 1, (s >> 1) & 1, s & 1


# =============================================================================
# MPS / MPO core operations
# =============================================================================

def apply_mpo(mps, mpo):
    """
    Apply MPO to MPS by contracting physical indices and merging bonds.

    MPS tensor M[j]:  (χ_L, d_in, χ_R)
    MPO tensor W[j]:  (b_L, d_out, d_in, b_R)
    Result:           (χ_L * b_L, d_out, χ_R * b_R)
    """
    new_mps = []
    for M, W in zip(mps, mpo):
        # einsum: contract over d_in (index 's')
        # M[i,s,k] W[a,t,s,b] → T[i,a,t,k,b]
        T = np.einsum('isk,atsb->iatkb', M, W)
        χL, bL, d_out, χR, bR = T.shape
        new_mps.append(T.reshape(χL * bL, d_out, χR * bR))
    return new_mps


def compress(mps, max_chi=None, cutoff=1e-14):
    """
    Left→right SVD sweep to compress MPS bond dimensions.

    For exact computation: set max_chi=None (keeps all non-zero singular values).
    For approximate computation: set max_chi to cap bond dimensions.
    """
    N = len(mps)
    for j in range(N - 1):
        χL, d, χR = mps[j].shape
        mat = mps[j].reshape(χL * d, χR)

        # Scale for numerical stability (SVD can fail on large-valued matrices)
        scale = np.max(np.abs(mat))
        if scale == 0:
            # Zero tensor: compress to bond dim 1
            mps[j] = np.zeros((χL, d, 1))
            mps[j + 1] = np.zeros((1, mps[j + 1].shape[1], mps[j + 1].shape[2]))
            continue
        mat_scaled = mat / scale

        try:
            U, S, Vh = np.linalg.svd(mat_scaled, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: use scipy with more robust driver
            from scipy.linalg import svd as scipy_svd
            U, S, Vh = scipy_svd(mat_scaled, full_matrices=False, lapack_driver='gesvd')

        S *= scale  # restore scale

        # Determine how many singular values to keep
        k = len(S)
        if cutoff > 0 and S[0] > 0:
            k = max(1, int(np.sum(S > cutoff * S[0])))
        if max_chi is not None:
            k = min(k, max_chi)

        U = U[:, :k]
        SV = np.diag(S[:k]) @ Vh[:k, :]          # (k, χR_old)

        mps[j] = U.reshape(χL, d, k)
        mps[j + 1] = np.einsum('ij,jsk->isk', SV, mps[j + 1])  # absorb into next

    return mps


def mps_contract(mps, v):
    """
    Contract MPS with a local vector v at each site → scalar.

    Computes: Σ_{s0,...,s_{N-1}} ψ(s0,...,s_{N-1}) · v[s0] · ... · v[s_{N-1}]
    """
    R = np.einsum('isj,s->ij', mps[0], v)
    for j in range(1, len(mps)):
        R = R @ np.einsum('isj,s->ij', mps[j], v)
    return R.item()


def mps_bond_dims(mps):
    """Return list of bond dimensions [χ_0, χ_1, ..., χ_N]."""
    return [mps[0].shape[0]] + [t.shape[2] for t in mps]


def mps_memory(mps):
    """Total number of float64 elements stored in all MPS tensors."""
    return sum(t.size for t in mps)


# =============================================================================
# MPO construction
# =============================================================================

def _queen_bulk():
    """
    Bulk MPO tensor for queen placement + column update + SE shift.

    Shape: (4, 8, 8, 4)
    Bond index b = 2*r + δ:
      r ∈ {0,1}: row constraint (queen placed?)
      δ ∈ {0,1}: SE signal shifted from left neighbor

    At each column j, two options:
    (a) No queen:  pass through column & NE, deposit SE from left, carry SE right
    (b) Queen:     requires r=0, c=0, dse=0, dne=0;
                   sets c=1, emits SE (δ→right) and NE (local)
    """
    G = np.zeros((4, D, D, 4))
    for bL in range(4):
        rL, δL = bL >> 1, bL & 1
        for si in range(D):
            c, dse, dne = _dec(si)

            # (a) No queen at this column
            so = _enc(c, δL, dne)      # col unchanged, deposit SE from left, NE raw
            bR = (rL << 1) | dse       # carry current SE rightward, row state unchanged
            G[bL, so, si, bR] += 1.0

            # (b) Queen at this column
            if rL == 0 and c == 0 and dse == 0 and dne == 0:
                so = _enc(1, δL, 1)    # col=1, deposit SE from left, emit NE=1
                bR = (1 << 1) | 1      # row_placed=1, emit SE=1 (shift to j+1)
                G[bL, so, si, bR] += 1.0

    return G


def queen_mpo(N):
    """Build queen-placement MPO for N columns, with boundary conditions."""
    G = _queen_bulk()
    mpo = []
    for j in range(N):
        if N == 1:
            # Both boundaries: left=b0(r=0,δ=0), right=sum over r=1
            W = G[0:1, :, :, 2:3] + G[0:1, :, :, 3:4]  # (1,8,8,1)
        elif j == 0:
            # Left boundary: start with (r=0, δ=0)
            W = G[0:1, :, :, :].copy()                    # (1,8,8,4)
        elif j == N - 1:
            # Right boundary: require r=1 (queen placed), sum over δ
            W = G[:, :, :, 2:3] + G[:, :, :, 3:4]        # (4,8,8,1)
        else:
            W = G.copy()
        mpo.append(W)
    return mpo


def _ne_bulk():
    """
    Bulk MPO tensor for NE diagonal left-shift.

    Shape: (2, 8, 8, 2)
    Bond index = NE signal flowing right→left.

    At each column j:
      - Receive NE signal from right (b_R) → deposit as new dne
      - Send current dne to the left (b_L)
    """
    G = np.zeros((2, D, D, 2))
    for bR in range(2):
        for si in range(D):
            c, dse, dne = _dec(si)
            so = _enc(c, dse, bR)    # deposit NE from right neighbor
            bL = dne                  # send current NE leftward
            G[bL, so, si, bR] += 1.0
    return G


def ne_mpo(N):
    """Build NE-shift MPO for N columns, with boundary conditions."""
    G = _ne_bulk()
    mpo = []
    for j in range(N):
        if N == 1:
            # NE signal goes nowhere; just clear it (bR=0, discard bL)
            W = np.zeros((1, D, D, 1))
            for si in range(D):
                c, dse, dne = _dec(si)
                W[0, _enc(c, dse, 0), si, 0] += 1.0
            mpo.append(W)
        elif j == 0:
            # Left boundary: discard NE going to position -1 (sum over bL)
            W = G.sum(axis=0, keepdims=True)   # (1,8,8,2)
            mpo.append(W)
        elif j == N - 1:
            # Right boundary: no NE from outside (bR=0)
            W = G[:, :, :, 0:1].copy()         # (2,8,8,1)
            mpo.append(W)
        else:
            mpo.append(G.copy())
    return mpo


# =============================================================================
# MPS Solver
# =============================================================================

def solve_mps(N, max_chi=None, verbose=False):
    """
    Count N-queens solutions using MPS/MPO tensor network contraction.

    Returns (count, bond_history) where bond_history[row] = max bond dim after that row.
    """
    if N <= 0:
        return (1, []) if N == 0 else (0, [])
    if N == 1:
        return (1, [1])

    # Initial MPS: product state |000...0⟩ (all free, no signals)
    mps = [np.zeros((1, D, 1)) for _ in range(N)]
    for j in range(N):
        mps[j][0, 0, 0] = 1.0

    # Build MPOs (reused every row since row transfer is row-independent)
    Wq = queen_mpo(N)
    Wn = ne_mpo(N)

    bond_hist = []

    for row in range(N):
        # Phase 1: Queen placement + column update + SE diagonal shift
        mps = apply_mpo(mps, Wq)
        mps = compress(mps, max_chi)

        # Phase 2: NE diagonal left-shift
        mps = apply_mpo(mps, Wn)
        mps = compress(mps, max_chi)

        bonds = mps_bond_dims(mps)
        max_bond = max(bonds[1:-1]) if N > 2 else bonds[1]
        bond_hist.append(max_bond)

        if verbose:
            mem = mps_memory(mps)
            print(f"  row {row}: max bond = {max_bond:6d},  "
                  f"bonds = {bonds},  "
                  f"tensors = {mem:,d} floats ({mem*8/1024:.1f} KB)")

    # Extract count: project each site onto c=1, sum over dse and dne
    v_final = np.array([1.0 if _dec(s)[0] == 1 else 0.0 for s in range(D)])
    count = int(round(mps_contract(mps, v_final)))

    return count, bond_hist


# =============================================================================
# Bitmask solver (reference, from nqueens_transfer_matrix.py)
# =============================================================================

def solve_bitmask(N):
    """
    Row-by-row bitmask transfer matrix solver.
    Returns (count, state_counts) where state_counts[row] = #states after that row.
    """
    if N <= 0:
        return (1, []) if N == 0 else (0, [])
    if N == 1:
        return (1, [1])

    mask = (1 << N) - 1
    states = {(0, 0, 0): 1}
    scounts = []

    for row in range(N):
        new = {}
        for (c, dse, dne), cnt in states.items():
            avail = mask & ~c & ~dse & ~dne
            b = avail
            while b:
                q = b & (-b)
                key = (c | q, ((dse | q) << 1) & mask, (dne | q) >> 1)
                new[key] = new.get(key, 0) + cnt
                b &= b - 1
        states = new
        scounts.append(len(states))

    return sum(states.values()), scounts


# =============================================================================
# Known values (OEIS A000170)
# =============================================================================

KNOWN = {
    0: 1, 1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40,
    8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712,
    14: 365596, 15: 2279184, 16: 14772512, 17: 95815104,
    18: 666090624, 19: 4968057848, 20: 39029188884,
}


# =============================================================================
# Main
# =============================================================================

def main():
    max_chi = None
    verbose = False
    args = list(sys.argv[1:])

    if '--verbose' in args:
        verbose = True
        args.remove('--verbose')
    if '-v' in args:
        verbose = True
        args.remove('-v')
    if '--max-chi' in args:
        i = args.index('--max-chi')
        max_chi = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    Ns = [int(args[0])] if args else list(range(1, 12))

    print("=" * 78)
    print("  N-Queens: MPS/MPO  vs  Bitmask Transfer Matrix")
    print(f"  Bond dim cap: {max_chi or 'none (exact)'}")
    print("=" * 78)
    print()
    print(f"{'N':>3}  {'Z_N':>12}  {'ok':>2}  "
          f"{'max_bond':>9}  {'#states':>9}  "
          f"{'MPS_mem':>10}  "
          f"{'t_mps':>8}  {'t_bitmask':>10}")
    print("-" * 78)

    for N in Ns:
        # MPS solve
        t0 = time.time()
        z_mps, bh = solve_mps(N, max_chi, verbose=verbose)
        t1 = time.time()

        # Bitmask solve
        z_bm, sc = solve_bitmask(N)
        t2 = time.time()

        ok = 'OK' if z_mps == KNOWN.get(N, z_bm) else 'FAIL'
        mb = max(bh) if bh else 0
        ms = max(sc) if sc else 0

        # MPS memory estimate (sum of tensor sizes × 8 bytes)
        # Rough: ~ N * max_bond^2 * d * 8 bytes
        mem_mps_floats = N * mb * mb * D if mb > 0 else 0
        mem_str = f"{mem_mps_floats * 8 / 1024:.0f} KB" if mem_mps_floats < 1e8 else f"{mem_mps_floats * 8 / 1048576:.0f} MB"

        print(f"{N:3d}  {z_mps:12d}  {ok:>2}  "
              f"{mb:9d}  {ms:9d}  "
              f"{mem_str:>10}  "
              f"{t1 - t0:7.3f}s  {t2 - t1:9.3f}s")

        if verbose:
            print()

        if t1 - t0 > 120:
            print("  (MPS solver too slow, stopping)")
            break

    print()

    # Summary explanation
    print("Column explanation:")
    print("  max_bond  -- largest MPS bond dimension (controls MPS memory)")
    print("  #states   -- number of distinct bitmask states (controls bitmask memory)")
    print("  MPS_mem   -- approximate MPS tensor storage")
    print()


if __name__ == '__main__':
    main()
