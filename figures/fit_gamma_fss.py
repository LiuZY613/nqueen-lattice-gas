"""
Finite-size scaling extrapolation of gamma_MC(N) -> gamma_inf.

Data from Table III of the paper.
Try several fitting forms and compare extrapolated gamma with
the Nobel et al. precise value gamma = 1.94400(1).
"""

import numpy as np
from scipy.optimize import curve_fit

# Data from Table III (N >= 32)
N_data = np.array([32, 64, 128, 256, 512, 1024], dtype=float)
gamma_data = np.array([1.833, 1.885, 1.909, 1.925, 1.931, 1.946])
gamma_err = np.array([0.001, 0.002, 0.002, 0.002, 0.003, 0.003])

gamma_exact = 1.94400  # Nobel et al.

print("=" * 60)
print("Finite-size scaling extrapolation of gamma_MC")
print("=" * 60)
print(f"\nNobel et al. exact value: gamma = {gamma_exact}")
print(f"\nData points:")
for n, g, e in zip(N_data, gamma_data, gamma_err):
    print(f"  N={int(n):4d}: gamma_MC = {g:.3f} +/- {e:.3f}")

# ---------------------------------------------------------------
# Fit 1: gamma(N) = gamma_inf + a * N^{-alpha}  (3 parameters)
# ---------------------------------------------------------------
def model_power(N, gamma_inf, a, alpha):
    return gamma_inf + a * N**(-alpha)

try:
    popt1, pcov1 = curve_fit(model_power, N_data, gamma_data,
                              sigma=gamma_err, absolute_sigma=True,
                              p0=[1.944, -1.0, 0.5],
                              maxfev=10000)
    perr1 = np.sqrt(np.diag(pcov1))
    residuals1 = gamma_data - model_power(N_data, *popt1)
    chi2_1 = np.sum((residuals1 / gamma_err)**2)
    ndof1 = len(N_data) - 3
    print(f"\n--- Fit 1: gamma + a * N^(-alpha) ---")
    print(f"  gamma_inf = {popt1[0]:.5f} +/- {perr1[0]:.5f}")
    print(f"  a         = {popt1[1]:.4f} +/- {perr1[1]:.4f}")
    print(f"  alpha     = {popt1[2]:.4f} +/- {perr1[2]:.4f}")
    print(f"  chi2/ndof = {chi2_1:.2f}/{ndof1}")
    print(f"  Deviation from exact: {abs(popt1[0] - gamma_exact):.5f} ({abs(popt1[0] - gamma_exact)/gamma_exact*100:.3f}%)")
except Exception as e:
    print(f"\nFit 1 failed: {e}")
    popt1 = None

# ---------------------------------------------------------------
# Fit 2: gamma(N) = gamma_inf + a / N  (2 parameters, alpha=1)
# ---------------------------------------------------------------
def model_invN(N, gamma_inf, a):
    return gamma_inf + a / N

popt2, pcov2 = curve_fit(model_invN, N_data, gamma_data,
                          sigma=gamma_err, absolute_sigma=True,
                          p0=[1.944, -3.0])
perr2 = np.sqrt(np.diag(pcov2))
residuals2 = gamma_data - model_invN(N_data, *popt2)
chi2_2 = np.sum((residuals2 / gamma_err)**2)
ndof2 = len(N_data) - 2
print(f"\n--- Fit 2: gamma + a / N ---")
print(f"  gamma_inf = {popt2[0]:.5f} +/- {perr2[0]:.5f}")
print(f"  a         = {popt2[1]:.4f} +/- {perr2[1]:.4f}")
print(f"  chi2/ndof = {chi2_2:.2f}/{ndof2}")
print(f"  Deviation from exact: {abs(popt2[0] - gamma_exact):.5f} ({abs(popt2[0] - gamma_exact)/gamma_exact*100:.3f}%)")

# ---------------------------------------------------------------
# Fit 3: gamma(N) = gamma_inf + a / sqrt(N)  (2 parameters)
# ---------------------------------------------------------------
def model_invsqrtN(N, gamma_inf, a):
    return gamma_inf + a / np.sqrt(N)

popt3, pcov3 = curve_fit(model_invsqrtN, N_data, gamma_data,
                          sigma=gamma_err, absolute_sigma=True,
                          p0=[1.944, -1.0])
perr3 = np.sqrt(np.diag(pcov3))
residuals3 = gamma_data - model_invsqrtN(N_data, *popt3)
chi2_3 = np.sum((residuals3 / gamma_err)**2)
ndof3 = len(N_data) - 2
print(f"\n--- Fit 3: gamma + a / sqrt(N) ---")
print(f"  gamma_inf = {popt3[0]:.5f} +/- {perr3[0]:.5f}")
print(f"  a         = {popt3[1]:.4f} +/- {perr3[1]:.4f}")
print(f"  chi2/ndof = {chi2_3:.2f}/{ndof3}")
print(f"  Deviation from exact: {abs(popt3[0] - gamma_exact):.5f} ({abs(popt3[0] - gamma_exact)/gamma_exact*100:.3f}%)")

# ---------------------------------------------------------------
# Fit 4: gamma(N) = gamma_inf + a / ln(N)  (2 parameters)
# ---------------------------------------------------------------
def model_invlnN(N, gamma_inf, a):
    return gamma_inf + a / np.log(N)

popt4, pcov4 = curve_fit(model_invlnN, N_data, gamma_data,
                          sigma=gamma_err, absolute_sigma=True,
                          p0=[1.944, -1.0])
perr4 = np.sqrt(np.diag(pcov4))
residuals4 = gamma_data - model_invlnN(N_data, *popt4)
chi2_4 = np.sum((residuals4 / gamma_err)**2)
ndof4 = len(N_data) - 2
print(f"\n--- Fit 4: gamma + a / ln(N) ---")
print(f"  gamma_inf = {popt4[0]:.5f} +/- {perr4[0]:.5f}")
print(f"  a         = {popt4[1]:.4f} +/- {perr4[1]:.4f}")
print(f"  chi2/ndof = {chi2_4:.2f}/{ndof4}")
print(f"  Deviation from exact: {abs(popt4[0] - gamma_exact):.5f} ({abs(popt4[0] - gamma_exact)/gamma_exact*100:.3f}%)")

# ---------------------------------------------------------------
# Fit 5: only use large N (N >= 128) with power law
# ---------------------------------------------------------------
mask = N_data >= 128
try:
    popt5, pcov5 = curve_fit(model_power, N_data[mask], gamma_data[mask],
                              sigma=gamma_err[mask], absolute_sigma=True,
                              p0=[1.944, -1.0, 0.5],
                              maxfev=10000)
    perr5 = np.sqrt(np.diag(pcov5))
    residuals5 = gamma_data[mask] - model_power(N_data[mask], *popt5)
    chi2_5 = np.sum((residuals5 / gamma_err[mask])**2)
    ndof5 = np.sum(mask) - 3
    print(f"\n--- Fit 5: gamma + a * N^(-alpha), N >= 128 only ---")
    print(f"  gamma_inf = {popt5[0]:.5f} +/- {perr5[0]:.5f}")
    print(f"  a         = {popt5[1]:.4f} +/- {perr5[1]:.4f}")
    print(f"  alpha     = {popt5[2]:.4f} +/- {perr5[2]:.4f}")
    print(f"  chi2/ndof = {chi2_5:.2f}/{ndof5}")
    print(f"  Deviation from exact: {abs(popt5[0] - gamma_exact):.5f} ({abs(popt5[0] - gamma_exact)/gamma_exact*100:.3f}%)")
except Exception as e:
    print(f"\nFit 5 failed: {e}")
    popt5 = None

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Fit':30s} {'gamma_inf':>12s} {'deviation':>10s}")
print("-" * 60)
if popt1 is not None:
    print(f"{'a*N^(-alpha), all N':30s} {popt1[0]:12.5f} {abs(popt1[0]-gamma_exact)/gamma_exact*100:9.3f}%")
print(f"{'a/N, all N':30s} {popt2[0]:12.5f} {abs(popt2[0]-gamma_exact)/gamma_exact*100:9.3f}%")
print(f"{'a/sqrt(N), all N':30s} {popt3[0]:12.5f} {abs(popt3[0]-gamma_exact)/gamma_exact*100:9.3f}%")
print(f"{'a/ln(N), all N':30s} {popt4[0]:12.5f} {abs(popt4[0]-gamma_exact)/gamma_exact*100:9.3f}%")
if popt5 is not None:
    print(f"{'a*N^(-alpha), N>=128':30s} {popt5[0]:12.5f} {abs(popt5[0]-gamma_exact)/gamma_exact*100:9.3f}%")
print("-" * 60)
print(f"{'Nobel exact':30s} {gamma_exact:12.5f}")
