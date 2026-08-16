#!/usr/bin/env python3
"""Reproduce the numerical tables and validation checks in the PRE manuscript.

This script performs no Monte Carlo sampling.  It starts from the archived
production data, checks their grids and metadata, repeats the thermodynamic
integration, propagates the tabulated jackknife errors, reconstructs the
finite-size fit of gamma, and repeats the dense peak analysis with a
deterministic bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PEAK = DATA / "peak"
DEFAULT_OUT = ROOT / "results"
MAIN_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)
GAMMA_SIZES = (32, 64, 128, 256, 512, 1024)
PEAK_SIZES = (8, 16, 32, 64, 100, 128)
BASELINE_PEAK_T = np.array([0.200, 0.225, 0.250, 0.275, 0.300])
REPORTED_PEAK = {
    8: (0.281, 1.6915, 0.0002),
    16: (0.227, 1.6783, 0.0006),
    32: (0.231, 1.6288, 0.0005),
    64: (0.237, 1.6216, 0.0004),
    100: (0.237, 1.6229, 0.0004),
    128: (0.237, 1.6225, 0.0004),
}


def load_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path, comments="#")
    if table.ndim == 1:
        table = table[None, :]
    if table.shape[1] < 8:
        raise ValueError(f"{path} has {table.shape[1]} columns; expected at least 8")
    return table[:, :8]


def expected_temperature_grid() -> np.ndarray:
    values = []
    values.extend(np.arange(50, 101, 5) / 1000)
    values.extend(np.arange(1025, 5001, 25) / 10000)
    values.extend(np.arange(525, 1001, 25) / 1000)
    values.extend(np.arange(105, 201, 5) / 100)
    values.extend(np.arange(210, 301, 10) / 100)
    values.extend(np.arange(310, 501, 10) / 100)
    values.extend(np.arange(55, 101, 5) / 10)
    values.extend(np.arange(11, 21))
    values.extend([25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
    values.extend([120, 150, 200, 250, 300, 350, 400, 450, 500])
    return np.asarray(values, dtype=float)


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(x)
    weights[0] = (x[1] - x[0]) / 2
    weights[-1] = (x[-1] - x[-2]) / 2
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    return weights


def entropy_and_gamma(n: int, table: np.ndarray) -> dict[str, float]:
    # The production grid contains 280 points through 500 J.  The reported
    # integral uses the first 278 points through 400 J; 450 and 500 J are
    # independent high-temperature checks.
    selected = table[:, 0] <= 400.0
    t = table[selected, 0]
    cv_per_n = table[selected, 3]
    err_cv_per_n = table[selected, 4]
    if len(t) != 278 or not np.isclose(t[-1], 400.0):
        raise ValueError(f"N={n}: expected 278 quadrature points through 400 J")

    integrand = cv_per_n / t
    integral_grid = float(np.trapezoid(integrand, t))
    # If Cv/N = a/T^2 at high T, the tail from Tmax to infinity is
    # a/(2 Tmax^2) = [Cv(Tmax)/N]/2.
    high_t_tail = float(cv_per_n[-1] / 2)
    integral = integral_grid + high_t_tail

    weights = trapezoid_weights(t) / t
    sigma_integral = float(
        np.sqrt(np.sum((weights * err_cv_per_n) ** 2) + (err_cv_per_n[-1] / 2) ** 2)
    )
    s_inf = (
        math.lgamma(n * n + 1)
        - math.lgamma(n + 1)
        - math.lgamma(n * n - n + 1)
    ) / n
    s0 = s_inf - integral
    gamma_eff = math.log(n) - s0
    return {
        "N": n,
        "quadrature_points": len(t),
        "integral_grid": integral_grid,
        "high_T_tail": high_t_tail,
        "integral_total": integral,
        "S_inf_per_N": s_inf,
        "s0_MC": s0,
        "sigma": sigma_integral,
        "gamma_MC": gamma_eff,
    }


def stationary_maximum(coefficients: np.ndarray) -> tuple[float, float]:
    derivative_roots = np.roots(np.polyder(coefficients))
    candidates = [
        root.real
        for root in derivative_roots
        if abs(root.imag) < 1e-8 and 0.2 <= root.real <= 0.3
    ]
    candidates.extend([0.2, 0.3])
    values = np.asarray([np.polyval(coefficients, value) for value in candidates])
    index = int(np.argmax(values))
    return float(candidates[index]), float(values[index])


def peak_fit(n: int, bootstrap_samples: int, rng: np.random.Generator) -> dict[str, float]:
    baseline = load_table(PEAK / f"data_N{n}_baseline.dat")
    additional = load_table(PEAK / f"data_N{n}_additional.dat")
    keep = np.any(
        np.isclose(baseline[:, 0, None], BASELINE_PEAK_T[None, :], atol=5e-7),
        axis=1,
    )
    combined = np.vstack([baseline[keep], additional])
    combined = combined[np.argsort(combined[:, 0])]
    expected = np.arange(200, 301) / 1000
    if len(combined) != 101 or not np.allclose(combined[:, 0], expected):
        raise ValueError(f"N={n}: dense peak scan is not the expected 101-point grid")

    t = combined[:, 0]
    cv = combined[:, 3]
    sigma = combined[:, 4]
    coefficients = np.polyfit(t, cv, deg=4, w=1 / sigma)
    fit_t, fit_cv = stationary_maximum(coefficients)

    boot_t = np.empty(bootstrap_samples)
    boot_cv = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        sample = rng.normal(cv, sigma)
        sample_coefficients = np.polyfit(t, sample, deg=4, w=1 / sigma)
        boot_t[index], boot_cv[index] = stationary_maximum(sample_coefficients)

    reported_t, reported_cv, reported_sigma = REPORTED_PEAK[n]
    return {
        "N": n,
        "points": len(combined),
        "fit_T_peak": fit_t,
        "fit_Cv_max_per_N": fit_cv,
        "bootstrap_sigma_T": float(np.std(boot_t, ddof=1)),
        "bootstrap_sigma_Cv": float(np.std(boot_cv, ddof=1)),
        "reported_T_peak": reported_t,
        "reported_Cv_max_per_N": reported_cv,
        "reported_error": reported_sigma,
        "difference_in_bootstrap_sigma": float(
            abs(fit_cv - reported_cv) / np.std(boot_cv, ddof=1)
        ),
    }


def power_law(n: np.ndarray, gamma_inf: float, amplitude: float, exponent: float) -> np.ndarray:
    return gamma_inf + amplitude * n ** (-exponent)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260326)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected_grid = expected_temperature_grid()
    if len(expected_grid) != 280:
        raise RuntimeError("internal temperature-grid construction is inconsistent")

    entropy_rows = []
    main_hashes = []
    for n in MAIN_SIZES:
        path = DATA / f"data_N{n}.dat"
        table = load_table(path)
        if len(table) != 280 or not np.allclose(table[:, 0], expected_grid):
            raise ValueError(f"N={n}: production temperature grid failed validation")
        entropy_rows.append(entropy_and_gamma(n, table))
        main_hashes.append((path.relative_to(ROOT).as_posix(), sha256(path)))

    rng = np.random.default_rng(args.bootstrap_seed)
    peak_rows = [peak_fit(n, args.bootstrap_samples, rng) for n in PEAK_SIZES]

    gamma_rows = [row for row in entropy_rows if row["N"] in GAMMA_SIZES]
    n_values = np.asarray([row["N"] for row in gamma_rows], dtype=float)
    # Figure 4(b) and the quoted fit use the table values at their published
    # precision.  The full-precision integration values remain in the CSV.
    gamma_values = np.asarray([1.833, 1.885, 1.909, 1.925, 1.931, 1.946])
    gamma_errors = np.asarray([0.001, 0.002, 0.002, 0.002, 0.003, 0.003])
    fit, covariance = curve_fit(
        power_law,
        n_values,
        gamma_values,
        sigma=gamma_errors,
        absolute_sigma=True,
        p0=(1.944, -1.0, 0.8),
        maxfev=20000,
    )
    fit_error = np.sqrt(np.diag(covariance))

    write_csv(args.output_dir / "entropy_gamma.csv", entropy_rows)
    write_csv(args.output_dir / "peak_scaling.csv", peak_rows)

    lines = [
        "# Reproduced numerical results",
        "",
        "Generated by `python reproduce.py` from the archived data files.",
        "",
        "## Thermodynamic integration",
        "",
        "The data contain 280 temperatures through 500 J. The integral uses 278 points through 400 J; the 450 J and 500 J points are high-temperature checks.",
        "",
        "| N | s0_MC | gamma_MC | propagated sigma |",
        "|---:|---:|---:|---:|",
    ]
    for row in entropy_rows:
        gamma_text = "--" if row["N"] < 32 else f'{row["gamma_MC"]:.6f}'
        lines.append(
            f'| {row["N"]} | {row["s0_MC"]:.6f} | {gamma_text} | {row["sigma"]:.6f} |'
        )
    lines.extend(
        [
            "",
            "## Finite-size correction fit",
            "",
            f"Weighted fit to the published Table III values, gamma(N) = gamma_inf + a N^(-alpha): gamma_inf = {fit[0]:.6f} +/- {fit_error[0]:.6f}, alpha = {fit[2]:.6f} +/- {fit_error[2]:.6f}.",
            "",
            "## Dense peak analysis",
            "",
            "Each fit combines five baseline points with 96 additional points to form the complete 0.200--0.300 J grid at spacing 0.001 J. The fit is weighted quartic regression; uncertainties use 5000 deterministic Gaussian bootstrap replicas.",
            "",
            "| N | fit T_peak | fit Cv_max/N | bootstrap sigma | manuscript Cv_max/N | difference in sigma |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in peak_rows:
        lines.append(
            f'| {row["N"]} | {row["fit_T_peak"]:.6f} | {row["fit_Cv_max_per_N"]:.6f} | '
            f'{row["bootstrap_sigma_Cv"]:.6f} | {row["reported_Cv_max_per_N"]:.4f} | '
            f'{row["difference_in_bootstrap_sigma"]:.2f} |'
        )
    lines.extend(["", "## Production-data SHA-256", ""])
    for name, digest in main_hashes:
        lines.append(f"- `{name}`: `{digest}`")
    (args.output_dir / "reproduction_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"Validated {len(MAIN_SIZES)} production data sets (280 points each).")
    print(f"Reproduced thermodynamic integration and {len(PEAK_SIZES)} dense peak fits.")
    print(f"gamma_inf = {fit[0]:.6f} +/- {fit_error[0]:.6f}")
    print(f"alpha     = {fit[2]:.6f} +/- {fit_error[2]:.6f}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
