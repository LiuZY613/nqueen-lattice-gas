/*
 * mc_canonical.c — 皇后格点气体：正则系综 Monte Carlo (Task 2 版)
 *
 * 在 Task 1 基础上增加了内置自相关时间计算。
 * 输出：T  E/N  err_E/N  Cv/N  err_Cv/N  accept_rate  E_total  tau_int
 *
 * tau_int: 积分自相关时间（单位：sweeps）
 *
 * 用法：./mc_canonical -L 8 -N 8 -T 1.0 -therm 100000 -nmeas 1000000
 *        [-nbin 200] [-seed 42] [-tsfile ts.dat] [-ts_interval 100]
 *        [-max_lag 2000] [-acf_interval 10]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ===================== RNG: xorshift128+ ===================== */
static uint64_t rng_s[2];

void rng_seed(uint64_t seed) {
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0x6a09e667f3bcc908ULL;
    for (int i = 0; i < 20; i++) {
        uint64_t s1 = rng_s[0], s0 = rng_s[1];
        rng_s[0] = s0;
        s1 ^= s1 << 23;
        rng_s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    }
}

static inline uint64_t rng_next(void) {
    uint64_t s1 = rng_s[0], s0 = rng_s[1];
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return rng_s[1] + s0;
}

static inline double rng_double(void) {
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline int rng_int(int n) {
    return (int)(rng_double() * n);
}

/* ===================== Global variables ===================== */
int L, N, L2;
int *board;
int *sites;
int *site_idx;
int *row_count, *col_count, *diag_p, *diag_m;
int total_energy;

/* ===================== Energy ===================== */
int compute_energy_from_counts(void) {
    int E = 0;
    for (int i = 0; i < L; i++)
        E += row_count[i] * (row_count[i] - 1) / 2;
    for (int i = 0; i < L; i++)
        E += col_count[i] * (col_count[i] - 1) / 2;
    for (int i = 0; i < 2 * L - 1; i++)
        E += diag_p[i] * (diag_p[i] - 1) / 2;
    for (int i = 0; i < 2 * L - 1; i++)
        E += diag_m[i] * (diag_m[i] - 1) / 2;
    return E;
}

/* ===================== Init ===================== */
void alloc_arrays(void) {
    L2 = L * L;
    board    = (int *)calloc(L2, sizeof(int));
    sites    = (int *)malloc(L2 * sizeof(int));
    site_idx = (int *)malloc(L2 * sizeof(int));
    row_count = (int *)calloc(L, sizeof(int));
    col_count = (int *)calloc(L, sizeof(int));
    diag_p    = (int *)calloc(2 * L - 1, sizeof(int));
    diag_m    = (int *)calloc(2 * L - 1, sizeof(int));
}

void init_random(void) {
    for (int i = 0; i < L2; i++) sites[i] = i;
    for (int i = 0; i < N; i++) {
        int j = i + rng_int(L2 - i);
        int tmp = sites[i]; sites[i] = sites[j]; sites[j] = tmp;
    }
    memset(board, 0, L2 * sizeof(int));
    memset(row_count, 0, L * sizeof(int));
    memset(col_count, 0, L * sizeof(int));
    memset(diag_p, 0, (2 * L - 1) * sizeof(int));
    memset(diag_m, 0, (2 * L - 1) * sizeof(int));
    for (int i = 0; i < L2; i++) site_idx[sites[i]] = i;
    for (int i = 0; i < N; i++) {
        int pos = sites[i];
        int r = pos / L, c = pos % L;
        board[pos] = 1;
        row_count[r]++; col_count[c]++;
        diag_p[r + c]++; diag_m[r - c + L - 1]++;
    }
    total_energy = compute_energy_from_counts();
}

/* ===================== MC step ===================== */
int mc_step(double beta) {
    int qi = rng_int(N);
    int pos_old = sites[qi];
    int r_old = pos_old / L, c_old = pos_old % L;
    int ei = N + rng_int(L2 - N);
    int pos_new = sites[ei];
    int r_new = pos_new / L, c_new = pos_new % L;

    int conf_old = (row_count[r_old] - 1) + (col_count[c_old] - 1)
                 + (diag_p[r_old + c_old] - 1) + (diag_m[r_old - c_old + L - 1] - 1);
    row_count[r_old]--; col_count[c_old]--;
    diag_p[r_old + c_old]--; diag_m[r_old - c_old + L - 1]--;
    int conf_new = row_count[r_new] + col_count[c_new]
                 + diag_p[r_new + c_new] + diag_m[r_new - c_new + L - 1];
    int dE = conf_new - conf_old;

    if (dE <= 0 || rng_double() < exp(-beta * dE)) {
        row_count[r_new]++; col_count[c_new]++;
        diag_p[r_new + c_new]++; diag_m[r_new - c_new + L - 1]++;
        board[pos_old] = 0; board[pos_new] = 1;
        sites[qi] = pos_new; sites[ei] = pos_old;
        site_idx[pos_new] = qi; site_idx[pos_old] = ei;
        total_energy += dE;
        return 1;
    } else {
        row_count[r_old]++; col_count[c_old]++;
        diag_p[r_old + c_old]++; diag_m[r_old - c_old + L - 1]++;
        return 0;
    }
}

void mc_sweep(double beta, int *accepted) {
    int acc = 0;
    for (int i = 0; i < N; i++) acc += mc_step(beta);
    *accepted = acc;
}

/* ===================== Main ===================== */
int main(int argc, char **argv) {
    L = 8; N = 8;
    double T = 1.0;
    int therm = 100000, nmeas = 1000000;
    int n_bin = 200;
    uint64_t seed = 12345;
    char tsfile[512] = "";
    int ts_interval = 0;
    int max_lag = 2000;
    int acf_interval = 10;  /* store E every acf_interval sweeps for ACF */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-L") == 0) L = atoi(argv[++i]);
        else if (strcmp(argv[i], "-N") == 0) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0) T = atof(argv[++i]);
        else if (strcmp(argv[i], "-therm") == 0) therm = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nmeas") == 0) nmeas = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nbin") == 0) n_bin = atoi(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0) seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-tsfile") == 0) strncpy(tsfile, argv[++i], 511);
        else if (strcmp(argv[i], "-ts_interval") == 0) ts_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "-max_lag") == 0) max_lag = atoi(argv[++i]);
        else if (strcmp(argv[i], "-acf_interval") == 0) acf_interval = atoi(argv[++i]);
    }

    if (N > L * L) { fprintf(stderr, "Error: N=%d > L^2=%d\n", N, L*L); return 1; }
    if (n_bin < 2) n_bin = 2;
    if (acf_interval < 1) acf_interval = 1;

    double beta = 1.0 / T;
    rng_seed(seed);
    alloc_arrays();
    init_random();

    FILE *ts_fp = NULL;
    if (tsfile[0]) {
        ts_fp = fopen(tsfile, "w");
        if (ts_fp) fprintf(ts_fp, "# sweep  E  phase(0=therm,1=meas)\n");
    }

    /* Thermalization */
    for (int s = 0; s < therm; s++) {
        int acc;
        mc_sweep(beta, &acc);
        if (ts_fp && ts_interval > 0 && (s + 1) % ts_interval == 0)
            fprintf(ts_fp, "%d  %d  0\n", s + 1, total_energy);
    }

    /* Allocate ACF buffer */
    int acf_buf_cap = nmeas / acf_interval + 1;
    double *acf_buf = (double *)malloc(acf_buf_cap * sizeof(double));
    int acf_count = 0;

    /* Measurement with binning */
    int bin_size = nmeas / n_bin;
    if (bin_size < 1) bin_size = 1;
    int actual_bins = nmeas / bin_size;

    double *bin_E  = (double *)calloc(actual_bins, sizeof(double));
    double *bin_E2 = (double *)calloc(actual_bins, sizeof(double));

    long total_accepted = 0, total_attempted = 0;
    int cur_bin = 0, bin_count = 0;
    double bE_acc = 0.0, bE2_acc = 0.0;

    for (int s = 0; s < nmeas; s++) {
        int acc;
        mc_sweep(beta, &acc);
        total_accepted += acc;
        total_attempted += N;

        double E = (double)total_energy;
        bE_acc += E;
        bE2_acc += E * E;
        bin_count++;

        if (bin_count == bin_size && cur_bin < actual_bins) {
            bin_E[cur_bin]  = bE_acc / bin_count;
            bin_E2[cur_bin] = bE2_acc / bin_count;
            cur_bin++;
            bE_acc = 0.0; bE2_acc = 0.0; bin_count = 0;
        }

        /* Store for ACF */
        if ((s + 1) % acf_interval == 0 && acf_count < acf_buf_cap) {
            acf_buf[acf_count++] = E;
        }

        if (ts_fp && ts_interval > 0 && (s + 1) % ts_interval == 0)
            fprintf(ts_fp, "%d  %d  1\n", therm + s + 1, total_energy);
    }
    if (ts_fp) fclose(ts_fp);

    int nb = cur_bin;

    /* Full averages from bins */
    double E_sum = 0.0, E2_sum = 0.0;
    for (int b = 0; b < nb; b++) {
        E_sum  += bin_E[b];
        E2_sum += bin_E2[b];
    }
    double E_avg  = E_sum / nb;
    double E2_avg = E2_sum / nb;
    double Cv = (E2_avg - E_avg * E_avg) / (T * T);

    /* Jackknife error estimation */
    double jk_E_var = 0.0, jk_Cv_var = 0.0;
    for (int b = 0; b < nb; b++) {
        double jk_E  = (E_sum  - bin_E[b])  / (nb - 1);
        double jk_E2 = (E2_sum - bin_E2[b]) / (nb - 1);
        double jk_Cv = (jk_E2 - jk_E * jk_E) / (T * T);
        jk_E_var  += (jk_E - E_avg) * (jk_E - E_avg);
        jk_Cv_var += (jk_Cv - Cv)   * (jk_Cv - Cv);
    }
    double err_E  = sqrt((nb - 1.0) / nb * jk_E_var);
    double err_Cv = sqrt((nb - 1.0) / nb * jk_Cv_var);

    double accept_rate = (double)total_accepted / total_attempted;

    /* ============ Compute autocorrelation time ============ */
    double tau_int = 0.5;  /* in units of sweeps */

    if (acf_count > max_lag + 10) {
        /* Compute mean */
        double acf_mean = 0.0;
        for (int i = 0; i < acf_count; i++) acf_mean += acf_buf[i];
        acf_mean /= acf_count;

        /* Compute C(0) = variance */
        double C0 = 0.0;
        for (int i = 0; i < acf_count; i++) {
            double d = acf_buf[i] - acf_mean;
            C0 += d * d;
        }
        C0 /= acf_count;

        if (C0 > 1e-15) {
            tau_int = 0.5;
            for (int t = 1; t < max_lag && t < acf_count / 4; t++) {
                double Ct = 0.0;
                int nt = acf_count - t;
                for (int i = 0; i < nt; i++) {
                    Ct += (acf_buf[i] - acf_mean) * (acf_buf[i + t] - acf_mean);
                }
                Ct /= nt;
                double rho = Ct / C0;
                if (rho < 0.0) break;
                tau_int += rho;
            }
            /* Convert from acf_interval units to sweep units */
            tau_int *= acf_interval;
        }
    }

    /* Output: T  E/N  err_E/N  Cv/N  err_Cv/N  accept_rate  E_total  tau_int */
    printf("%.6f  %.8f  %.8f  %.8f  %.8f  %.6f  %.4f  %.1f\n",
           T, E_avg/N, err_E/N, Cv/N, err_Cv/N, accept_rate, E_avg, tau_int);

    /* Energy consistency check */
    int E_final = compute_energy_from_counts();
    if (E_final != total_energy)
        fprintf(stderr, "WARNING: energy drift! tracked=%d computed=%d\n", total_energy, E_final);

    free(acf_buf);
    free(bin_E); free(bin_E2);
    free(board); free(sites); free(site_idx);
    free(row_count); free(col_count); free(diag_p); free(diag_m);
    return 0;
}
