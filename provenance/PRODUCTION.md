# Production provenance

This record separates directly archived facts from metadata that was not
retained by the historical scheduler.

## Main 280-point campaign

The source file in this repository has the same SHA-256 digest as the source
left in the production directory:

`a2facd9cd0ca12150ced2f3e38826df885315b4132213f6a3198a1a613a35c25`

It was compiled with GCC 8.5.0 using `-O3 -Wall` and linked only against
`libm`. Each state point used 2,000,000 thermalization sweeps, 100,000,000
measurement sweeps, 200 jackknife bins, xorshift128+, and the seed formula
documented in the main README.

Successful job 11961 ran N = 8 through 512 on 280 tasks from
2026-03-24 20:40:14 CST to 23:18:53 CST. Successful job 11964 ran N = 1024
on 280 tasks from 2026-03-24 20:43:44 CST to 2026-03-25 01:51:38 CST.
Their stdout is preserved verbatim in `provenance/logs/`.

The move-throughput values in the README count both thermalization and
measurement attempts and use the recorded elapsed times. They therefore
include the cost of incremental energy tracking, accumulation of E and E
squared, jackknife-bin accumulation, and autocorrelation-series collection and
processing.

## Dedicated peak campaign

The final dense peak data were produced on 2026-03-22 in six overlapping
batches. Each batch used one process per state point and the same C source and
compiler flags. The recorded allocations and elapsed times were:

- g1: 96 CPUs for 7 min 30 s.
- g2: 78 CPUs for 11 min 58 s.
- g3: 78 CPUs for 11 min 49 s.
- g4: 108 CPUs for 8 min 31 s.
- g5: 108 CPUs for 7 min 11 s.
- g6: 108 CPUs for 7 min 20 s.

The summed allocation is approximately 84.4 core-hours. Each state point again
used 2,000,000 thermalization and 100,000,000 measurement sweeps.

## Hardware metadata boundary

Slurm accounting storage is disabled on the production cluster, and the
successful jobs did not print `/proc/cpuinfo` or `lscpu` into stdout. The
historical CPU marketing-model strings and exact node list therefore cannot be
recovered reliably. We do not infer or fabricate them. The retained records
establish the x86-64 Linux environment, partition, task counts, source digest,
compiler, exact run parameters, elapsed times, outputs, and measured
throughput. None of the numerical results depends on processor-specific math
libraries.
