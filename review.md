# Referee Report: "Statistical mechanics of the n-queens lattice gas: Monte Carlo simulations and Poisson mean-field theory"

## Summary

The authors formulate the N-queens problem as a lattice gas with pairwise repulsive interactions along shared attack lines and study it via canonical Monte Carlo simulations (N = 8–1024, 10⁸ sweeps, 280 temperature points). The central result is a thermodynamic integration of Cv/T that recovers the ground-state entropy and hence the Simkin constant γ ≈ 1.944, reaching 0.11% accuracy at N = 1024. A Poisson mean-field theory is also developed.

The paper is clearly written and presents a clean physical picture. However, I have a number of concerns regarding the novelty claims, the rigor of certain arguments, and several technical issues that should be addressed before publication.

---

## Major Issues

**1. Circularity of the "first physical determination" claim.**
The abstract and conclusion prominently claim a "first physical determination" of γ. However, the thermodynamic integration (Eq. 10) requires the *exact* combinatorial high-temperature entropy S(∞)/N = (1/N) ln C(N²,N) as input. The result is then compared against the combinatorial value γ = 1.944001 from Nobel et al. This is a self-consistency check, not an independent determination. The authors partially acknowledge this ("nontrivial self-consistency check" appears in the conclusions), but the abstract and introduction still frame the result as a "first physical determination." This language should be tempered throughout, or the authors should clearly articulate what would constitute a truly independent extraction (e.g., finite-size extrapolation of γ_MC(N) → γ without reference to the Nobel et al. value).

**2. No finite-size scaling analysis of γ_MC(N).**
The convergence of γ_MC from 1.83 (N=32) to 1.946 (N=1024) is slow and monotonic, suggesting well-behaved finite-size corrections. A finite-size scaling fit (e.g., γ_MC(N) = γ + a/N^α or γ + a/ln N) could (a) dramatically improve the infinite-N estimate beyond what raw N=1024 gives, and (b) would constitute a genuinely *predictive* physical determination of γ without relying on the Nobel et al. value. The absence of such an extrapolation is a significant missed opportunity and weakens the paper's central claim.

**3. Phase transition ruling is too strong for the evidence presented.**
The claim that Cv/N saturation "rules out a thermodynamic phase transition" rests on Table II, which only goes up to N = 128 (despite data existing to N = 1024). A logarithmic divergence ∝ ln N would produce increments of ~0.06 per doubling of N—comparable to the observed variation. The authors should:
- Include N = 256, 512, 1024 in Table II and explain why these were omitted.
- Attempt a quantitative finite-size scaling fit (e.g., Cv_max/N vs 1/N or ln N) to distinguish saturation from weak divergence.
- Soften the language from "ruling out" to "consistent with the absence of."

**4. Error propagation through thermodynamic integration.**
The "Deviation" column in Table III has no error bars. Jackknife errors are computed for individual (T, Cv) data points, but these are never propagated through the trapezoidal integration to yield an uncertainty on s₀^MC or γ_MC. At N = 1024, the claimed 0.11% deviation could be partly or wholly statistical. Propagated error bars are essential to evaluate the significance of the agreement.

**5. Simple mean-field: ad hoc construction.**
The "entropy-weighted effective number of interaction directions" νeff = 1 + γ (Eq. 15) lacks physical justification. Why should the *entropy cost* of a constraint (a ground-state property) determine the *effective interaction strength* at finite temperature? The prefactor happens to work because it's calibrated to reproduce the correct T → 0 limit, but this is tautological—it uses γ as input to "predict" γ. The authors should more clearly distinguish this as a phenomenological fit rather than a derivation.

---

## Minor Issues

**6. Inconsistency: 260 cores vs 280 temperature points.**
Section II states "260 CPU cores in parallel" for "280 temperature points." If each core handles one temperature, this should be 280. If 260 is correct, explain how the remaining 20 temperatures were handled.

**7. Table II: missing system sizes.**
The caption states "dedicated high-resolution simulations at N = 8–128" — but the main simulations go to N = 1024. If the peak data for larger N is available from the 280-point grid, it should be included. If the grid resolution is insufficient to resolve the peak for large N, this should be stated.

**8. Missing connection to CSP literature.**
The statistical mechanics of constraint satisfaction problems (CSPs) is a rich field (Mézard, Parisi, Zdeborová, and others). The N-queens problem is a paradigmatic CSP, and the lattice gas formulation connects naturally to this body of work. The paper would benefit from discussing how its findings relate to the general CSP phase-transition phenomenology (e.g., satisfiability thresholds, clustering transitions).

**9. Modified Poisson MF: independence assumption.**
The passage from single-line partition functions (Eq. 19–22) to the total energy (Eq. 26) assumes statistical independence of occupation numbers across different attack lines. Since each queen sits on exactly four attack lines, the correlations are significant and structured. A brief discussion of *why* the approximation works to within 2–7% (perhaps the attack-line graph is locally tree-like?) would strengthen this section.

**10. Data and code availability.**
There is no mention of making the simulation code or data publicly available. PRE encourages this; at minimum the authors should state where the data can be obtained.

**11. High-temperature limit: not truly new.**
The exact high-T energy calculation via linearity of expectation (Sec. III B) is attributed to Polson & Sokolov (2024). The authors should more clearly state what, if anything, is new in their derivation versus what is reproduced from the literature.

**12. Ground-state entropy "hierarchy" novelty.**
The decomposition s₀ = ln N − γ into column (1 unit) and diagonal (γ−1 units) costs is a restatement of Stirling's approximation combined with Simkin's result. While pedagogically useful, this should not be presented as a new result without qualification.

---

## Recommendation

The paper addresses an interesting interface between combinatorics and statistical mechanics, and the computational effort is substantial. However, the central claim ("first physical determination of γ") is overstated given the circularity with combinatorial inputs, and the absence of finite-size extrapolation, error propagation, and quantitative phase-transition analysis leaves significant gaps. I recommend **major revision** addressing the points above, after which the paper could be suitable for Physical Review E.
