#!/usr/bin/env python3
"""Merge outputs from run_peak_grid.sh into one file per system size."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    for n in (8, 16, 32, 64, 100, 128):
        source = args.run_root / f"N{n}"
        files = sorted(source.glob(f"result_N{n}_T*.dat"))
        if len(files) != 101:
            raise SystemExit(f"N={n}: expected 101 files, found {len(files)}")
        rows = sorted(
            (path.read_text(encoding="utf-8").strip() for path in files),
            key=lambda row: float(row.split()[0]),
        )
        output = args.run_root / f"data_N{n}_peak_101pt.dat"
        header = (
            f"# N={n} peak grid T=0.200--0.300 step 0.001\n"
            "# therm=2000000 nmeas=100000000 nbin=200\n"
            "# T E/N err_E/N Cv/N err_Cv/N accept_rate E_total tau_int\n"
        )
        output.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
