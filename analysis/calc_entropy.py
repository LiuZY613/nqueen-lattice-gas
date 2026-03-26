import numpy as np
import math

# gamma constant from Simkin formula
gamma = 1.942

data_dir = "/c/Users/刘宗岳/Desktop/nqueen_simulation/task2_N等于L"
sizes = [4, 8, 16, 32, 64, 128]

# Known Q(N) values for small N
Q_exact = {4: 2, 8: 92, 16: 14772512}

print(f"{'M':>5s}  {'S_inf/M (theory)':>16s}  {'int Cv/(MT) dT':>16s}  {'S_inf/M (exact)':>16s}  {'Delta_S theory':>14s}  {'Delta_S MC':>14s}  {'diff%':>8s}")
print("-" * 105)

for N in sizes:
    fname = f"{data_dir}/data_L{N}.dat"
    data = np.loadtxt(fname)
    T = data[:, 0]
    CvN = data[:, 3]  # Cv/N
    err_CvN = data[:, 4]  # err of Cv/N
    
    # Integrate Cv/(N*T) dT = (Cv/N)/T dT using trapezoidal rule
    integrand = CvN / T
    
    # Main integral (trapezoidal)
    integral = np.trapz(integrand, T)
    
    # High-T tail correction: Cv/N ~ c/T^2, so Cv/(NT) ~ c/T^3
    # Estimate c from last data point
    T_last = T[-1]
    c = CvN[-1] * T_last**2
    tail_correction = c / (2.0 * T_last**2)
    
    total_integral = integral + tail_correction
    
    # Theoretical S(inf)/M = ln(C(N^2, N)) / N
    # Using Stirling or exact
    # S(inf)/M = [ln(N^2!) - ln(N!) - ln(N^2-N)!] / N
    # For practical computation, use lgamma
    S_inf_exact = (math.lgamma(N*N + 1) - math.lgamma(N + 1) - math.lgamma(N*N - N + 1)) / N
    
    # Theoretical S(inf)/M - S(0)/M = 1 + gamma (asymptotic)
    delta_S_theory = 1 + gamma  # asymptotic, same for all N
    
    # More precise: use exact S(inf) and exact/asymptotic S(0)
    if N in Q_exact:
        S0_exact = math.log(Q_exact[N]) / N
    else:
        # Simkin asymptotic: Q(N) ~ (N/e^gamma)^N => S(0)/N = ln(N) - gamma
        S0_exact = math.log(N) - gamma
    
    delta_S_exact = S_inf_exact - S0_exact
    
    # MC result
    delta_S_MC = total_integral
    
    diff_pct = abs(delta_S_exact - delta_S_MC) / delta_S_exact * 100
    
    print(f"{N:>5d}  {S_inf_exact:>16.4f}  {total_integral:>16.4f}  {S_inf_exact:>16.4f}  {delta_S_exact:>14.4f}  {delta_S_MC:>14.4f}  {diff_pct:>7.1f}%")

print()
print("=" * 60)
print("Table for manuscript (S_inf - S_0 comparison):")
print("=" * 60)
print()

for N in sizes:
    fname = f"{data_dir}/data_L{N}.dat"
    data = np.loadtxt(fname)
    T = data[:, 0]
    CvN = data[:, 3]
    
    integrand = CvN / T
    integral = np.trapz(integrand, T)
    T_last = T[-1]
    c = CvN[-1] * T_last**2
    tail_correction = c / (2.0 * T_last**2)
    total_integral = integral + tail_correction
    
    S_inf_exact = (math.lgamma(N*N + 1) - math.lgamma(N + 1) - math.lgamma(N*N - N + 1)) / N
    
    if N in Q_exact:
        S0_exact = math.log(Q_exact[N]) / N
    else:
        S0_exact = math.log(N) - gamma
    
    delta_S_exact = S_inf_exact - S0_exact
    delta_S_MC = total_integral
    diff_pct = abs(delta_S_exact - delta_S_MC) / delta_S_exact * 100
    
    print(f"  M={N}: theory={delta_S_exact:.3f}, MC={delta_S_MC:.3f}, diff={diff_pct:.1f}%")

