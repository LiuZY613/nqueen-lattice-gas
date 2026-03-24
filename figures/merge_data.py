"""
merge_data.py — 合并精确+密集+高温+峰区数据，输出到eassywriting/data_N{N}.dat
"""
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
TASK2 = os.path.dirname(BASE)
PRECISE_DIR = os.path.join(TASK2, '精确结果')
DENSE_DIR = os.path.join(TASK2, '密集结果')
HIGHT_DIR = os.path.join(TASK2, '高温结果')
PEAK_DIR = os.path.join(TASK2, '峰区结果')

Ns = [8, 16, 32, 64, 100, 128]

def load(path):
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

for N in Ns:
    parts = []
    sources = []

    f_precise = os.path.join(PRECISE_DIR, f'data_L{N}.dat')
    if os.path.exists(f_precise):
        d = load(f_precise)
        if d.shape[0] > 0:
            parts.append(d)
            sources.append(f'precise {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})')

    f_dense = os.path.join(DENSE_DIR, f'data_L{N}_dense.dat')
    if os.path.exists(f_dense):
        d = load(f_dense)
        if d.shape[0] > 0:
            parts.append(d)
            sources.append(f'dense {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})')

    f_highT = os.path.join(HIGHT_DIR, f'data_L{N}_highT.dat')
    if os.path.exists(f_highT):
        d = load(f_highT)
        if d.shape[0] > 0:
            parts.append(d)
            sources.append(f'high-T {d.shape[0]} pts (T={d[0,0]:.1f}~{d[-1,0]:.1f})')

    f_peak = os.path.join(PEAK_DIR, f'data_L{N}_peak.dat')
    if os.path.exists(f_peak):
        d = load(f_peak)
        if d.shape[0] > 0:
            parts.append(d)
            sources.append(f'peak {d.shape[0]} pts (T={d[0,0]:.3f}~{d[-1,0]:.3f})')

    if not parts:
        continue

    all_data = np.vstack(parts)
    # Sort by T, deduplicate (keep first occurrence = precise data preferred)
    _, unique_idx = np.unique(all_data[:, 0], return_index=True)
    merged = all_data[unique_idx]

    outfile = os.path.join(BASE, f'data_N{N}.dat')
    with open(outfile, 'w') as f:
        f.write(f'# N={N} M={N}  Queen lattice gas, canonical ensemble M=N\n')
        f.write(f'# Merged from: {"; ".join(sources)}\n')
        f.write(f'# Total: {merged.shape[0]} temperature points (T={merged[0,0]:.3f}~{merged[-1,0]:.1f})\n')
        f.write(f'# nmeas=1e8 sweeps, therm=2e6, nbin=200\n')
        f.write(f'# T  E/M  err_E/M  Cv/M  err_Cv/M  accept_rate  E_total  tau_int\n')
        for row in merged:
            f.write(f'{row[0]:.6f}  {row[1]:.8f}  {row[2]:.8f}  '
                    f'{row[3]:.8f}  {row[4]:.8f}  {row[5]:.6f}  {row[6]:.4f}  {row[7]:.1f}\n')

    print(f'N={N}: {merged.shape[0]} pts -> {outfile}')

print('\nDone.')
