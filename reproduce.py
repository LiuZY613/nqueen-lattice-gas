#!/usr/bin/env python3
"""One-command reproduction of numerical tables and publication figures."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    ROOT / "analysis" / "reproduce_results.py",
    ROOT / "analysis" / "plot_fig2_convergence.py",
    ROOT / "analysis" / "plot_PRE_figures.py",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"\n==> {script.relative_to(ROOT)}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nReproduction complete. Outputs are in results/.")


if __name__ == "__main__":
    main()
