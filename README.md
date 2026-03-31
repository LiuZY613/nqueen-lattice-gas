# n-Queens Lattice Gas: Monte Carlo Simulations

Simulation code and data accompanying the paper:

> Z.-Y. Liu and L. Wang, "Statistical mechanics of the *n*-queens lattice gas: Monte Carlo simulations and thermodynamic integration," submitted to Physical Review E.

## Model

We study the *n*-queens problem as a lattice gas on an N x N chessboard. Each site carries an occupation variable n_ij in {0, 1}, and the Hamiltonian counts mutually attacking queen pairs:

H = J * sum_{attacking pairs} n_ij * n_i'j'

where the sum runs over all pairs sharing a row, column, main diagonal, or anti-diagonal. We work in the canonical ensemble with fixed particle number N and coupling J = 1.

The Monte Carlo dynamics uses Kawasaki (queen-vacancy exchange) moves with the Metropolis acceptance criterion. One sweep consists of N attempted exchanges.

## Repository Structure

```
src/                    Monte Carlo simulation source code
  mc_canonical.c        Main simulation code (C, single-file)
  Makefile              Build instructions

scripts/                SLURM job submission scripts
  run_8to512.sh         Run N = 8, 16, 32, 64, 128, 256, 512 (280 temperatures each)
  run_1024.sh           Run N = 1024 (280 temperatures)
  worker.sh             Single (N, T) simulation task (called by srun)

data/                   Simulation results (280 temperature points per system size)
  data_N8.dat           N = 8
  data_N16.dat          N = 16
  data_N32.dat          N = 32
  data_N64.dat          N = 64
  data_N128.dat         N = 128
  data_N256.dat         N = 256
  data_N512.dat         N = 512
  data_N1024.dat        N = 1024

analysis/               Python scripts for data analysis and figure generation
  plot_PRE_figures.py   Generate publication figures (energy, specific heat)
  plot_fig2_convergence.py  Convergence diagnostics figure
  plot_fig1_schematic.py    Schematic diagram
  merge_data.py         Merge raw simulation outputs into data files
  compute_gamma.py      Compute Simkin constant via thermodynamic integration
  calc_entropy.py       Calculate entropy from Cv data

tensor/                 Tensor network / transfer matrix exact solvers
  nqueens_transfer_matrix.py  Row-by-row bitmask transfer matrix solver
  nqueens_mps.py              MPS/MPO solver with SVD compression
  nqueens_mps_vs_bitmask.md   Comparison analysis: MPS vs bitmask approach
```

## Building

```bash
cd src
make
```

Requires only a C compiler (gcc recommended) and the standard math library. No external dependencies.

## Running a Single Simulation

```bash
./mc_canonical -L 64 -N 64 -T 0.235 -therm 2000000 -nmeas 100000000 -nbin 200 -seed 42
```

**Parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `-L` | Board size (L x L) | 8 |
| `-N` | Number of queens | 8 |
| `-T` | Temperature (units of J) | 1.0 |
| `-therm` | Thermalization sweeps | 100000 |
| `-nmeas` | Measurement sweeps | 1000000 |
| `-nbin` | Jackknife bins | 200 |
| `-seed` | RNG seed | 12345 |
| `-max_lag` | Max lag for autocorrelation | 2000 |
| `-acf_interval` | ACF sampling interval (sweeps) | 10 |

**Output format** (single line, space-separated):
```
T  E/N  err_E/N  Cv/N  err_Cv/N  accept_rate  E_total  tau_int
```

## Reproducing the Paper Data

The paper uses 280 temperature points per system size with 10^8 measurement sweeps and 2 x 10^6 thermalization sweeps. On a SLURM cluster:

```bash
# Compile
cd src && make && cd ..

# Submit jobs (requires 280 CPU cores)
sbatch scripts/run_8to512.sh    # N = 8 to 512
sbatch scripts/run_1024.sh      # N = 1024
```

Each job runs 280 temperatures in parallel via `srun`. Total wall-clock time is approximately 9 hours on 280 cores.

## Data Format

Each `data_N{N}.dat` file contains 280 rows (one per temperature point) with 8 columns:

| Column | Quantity | Description |
|--------|----------|-------------|
| 1 | T | Temperature (units of J) |
| 2 | E/N | Energy per queen |
| 3 | err_E/N | Jackknife error on E/N |
| 4 | Cv/N | Specific heat per queen |
| 5 | err_Cv/N | Jackknife error on Cv/N |
| 6 | accept_rate | Metropolis acceptance rate |
| 7 | E_total | Total energy |
| 8 | tau_int | Integrated autocorrelation time (sweeps) |

Temperature range: T = 0.05 to 500 J, with dense sampling (step 0.0025 J) in the specific heat peak region T = 0.1025 to 0.5 J.

## Generating Figures

```bash
cd analysis
python plot_PRE_figures.py       # Fig. 3 (energy) and Fig. 4 (specific heat)
python plot_fig2_convergence.py  # Fig. 2 (convergence diagnostics)
python plot_fig1_schematic.py    # Fig. 1 (schematic)
```

Requires Python 3 with NumPy and Matplotlib.

## Tensor Network Solvers

The `tensor/` directory contains exact solvers based on transfer matrix methods:

```bash
cd tensor
python nqueens_transfer_matrix.py   # Bitmask transfer matrix (exact, efficient)
python nqueens_mps.py               # MPS/MPO solver (exact with full bond dim)
```

The bitmask solver enumerates valid constraint states (column, SE/NE diagonals) row by row and is efficient up to N ~ 25. The MPS solver represents the boundary state as a Matrix Product State and applies MPO operators per row, with optional SVD compression. See `nqueens_mps_vs_bitmask.md` for a detailed comparison showing that bitmask is more efficient due to the high entanglement structure of diagonal constraints.

## Computing the Simkin Constant

```bash
cd analysis
python compute_gamma.py
```

This performs the thermodynamic integration of Cv/T to extract the ground-state entropy and the Simkin constant gamma_MC for each system size.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite:

```bibtex
@article{liu2026nqueens,
  title={Statistical mechanics of the $n$-queens lattice gas:
         Monte Carlo simulations and thermodynamic integration},
  author={Liu, Zong-Yue and Wang, Lei},
  journal={Physical Review E},
  year={2026},
  note={submitted}
}
```
