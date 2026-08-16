// Eq. 9 relocation self-check (MCMC paper, Kheradmand et al.).
//
//   clang++ -O2 -std=c++17 tests/test_relocation.cpp -o /tmp/test_relocation && /tmp/test_relocation
//
// Mirrors Model::computeRelocation's corrected branch (core/src/model.cpp).
// Two invariants, both of which the historical sqrt(1/N) scale rule violated:
//   1. centre alpha of the N-fold composite reproduces o_old exactly
//   2. the integral of composite alpha equals the original Gaussian's
// plus N == 1 being an exact no-op (the case that hid the bug while the
// ensemble size was off by one).
#include <cmath>
#include <cstdio>
#include <initializer_list>

static void reloc(float o_old, float s_old, int N, float &o_new, float &s_new) {
    if (N < 1) N = 1;
    if (N > 51) N = 51;
    o_new = 1.0f - std::pow(1.0f - o_old, 1.0f / (float)N);
    const double on = (double)o_new;
    double denom = 0.0, binom = 1.0;
    for (int k = 1; k <= N; k++) {
        binom = binom * (double)(N - k + 1) / (double)k;
        const double term = binom * std::pow(on, (double)k) / std::sqrt((double)k);
        denom += (k % 2 == 1) ? term : -term;
    }
    double coeff = (denom > 1e-12) ? ((double)o_old / denom) : 1.0;
    if (!std::isfinite(coeff) || coeff <= 0.0) coeff = 1.0;
    s_new = (float)((double)s_old * coeff);
}

int main() {
    int fails = 0;
    for (float o : {0.05f, 0.1f, 0.5f, 0.9f, 0.998f}) {
        for (int N : {1, 2, 3, 5, 12, 51}) {
            float on, sn;
            reloc(o, 1.0f, N, on, sn);

            // (1) centre alpha of the composite
            const double centre = 1.0 - std::pow(1.0 - (double)on, (double)N);
            if (std::fabs(centre - (double)o) > 1e-5) {
                printf("FAIL centre o=%.3f N=%d: %.6f != %.6f\n", o, N, centre, o);
                fails++;
            }

            // (2) integral of composite alpha vs the original
            double I = 0.0;
            const double dx = 0.0005;
            for (double x = -20; x < 20; x += dx) {
                const double a = (double)on * std::exp(-x * x / (2.0 * sn * sn));
                I += (1.0 - std::pow(1.0 - a, (double)N)) * dx;
            }
            const double want = (double)o * std::sqrt(2 * M_PI);
            if (std::fabs(I / want - 1.0) > 2e-3) {
                printf("FAIL integral o=%.3f N=%d: %.6f vs %.6f (ratio %.5f)\n",
                       o, N, I, want, I / want);
                fails++;
            }

            // (3) N == 1 must be an exact no-op
            if (N == 1 && (std::fabs(sn - 1.0f) > 1e-6f || std::fabs(on - o) > 1e-6f)) {
                printf("FAIL N=1 no-op o=%.3f: o_new=%.6f s=%.6f\n", o, on, sn);
                fails++;
            }

            // (4) the historical rule over-shrank: correct coeff must exceed
            // sqrt(1/N) for every N > 1 (this is the -1.2 dB regression)
            if (N > 1 && !(sn > std::sqrt(1.0f / (float)N))) {
                printf("FAIL over-shrink guard o=%.3f N=%d: %.5f <= %.5f\n",
                       o, N, sn, std::sqrt(1.0f / (float)N));
                fails++;
            }
        }
    }
    printf(fails ? "%d FAILURES\n" : "all OK\n", fails);
    return fails ? 1 : 0;
}
