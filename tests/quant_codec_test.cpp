// Q codec CPU checks (design note s6.1). Build: xcrun clang++ -O2 -std=c++17 -Icore/include tests/quant_codec_test.cpp
// Exact properties are asserted; the R1 (fake step) study prints numbers for the plan report.
#include "quant_codec.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace qcodec;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void exactProperties() {
    std::mt19937 g(1);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    const float bounds[] = {1e-6f, 3e-3f, 0.37f, 1.0f, 2.5f, 1e3f};
    for (int M : {127, 32767}) {
        CHECK(encSym(0.0f, 1.0f, M) == 0 && decSym(0, 1.0f, M) == 0.0f, "zero M=%d", M);
        CHECK(encSym(0.3f, 0.0f, M) == 0, "zero bound M=%d", M);
        for (float a : bounds) {
            double q = a / M, worst = 0;
            for (int i = 0; i < 200000; i++) {   // round trip <= q/2 (+ float rounding of the two ops)
                float x = (2.f * u01(g) - 1.f) * a;
                double e = std::fabs((double)decSym(encSym(x, a, M), a, M) - x);
                worst = std::max(worst, e);
            }
            CHECK(worst <= q / 2 * (1 + 1e-5) + 4e-7 * a, "round trip M=%d a=%g worst=%g q/2=%g", M, a, worst, q / 2);
            int bad = 0;                           // fixed bound: decode -> encode is the identity on codes
            for (int c = -M; c <= M; c++) bad += encSym(decSym(c, a, M), a, M) != c;
            CHECK(bad == 0, "idempotence M=%d a=%g bad=%d", M, a, bad);
            float a2 = a * 1.7f; double worst2 = 0; // widening: re-encode at the new bound, error <= q_new/2
            for (int c = -M; c <= M; c += (M > 1000 ? 7 : 1)) {
                float x = decSym(c, a, M);
                worst2 = std::max(worst2, std::fabs((double)decSym(encSym(x, a2, M), a2, M) - x));
            }
            CHECK(worst2 <= a2 / M / 2 * (1 + 1e-5) + 4e-7 * a2, "widen M=%d a=%g worst=%g", M, a, worst2);
        }
    }
    for (int M : {255, 65535}) {
        CHECK(encUns(0.0f, 5.0f, M) == 0 && decUns(0, 5.0f, M) == 0.0f, "uns zero M=%d", M);
        int bad = 0;
        for (int c = 0; c <= M; c++) bad += encUns(decUns(c, 31.f, M), 31.f, M) != c;
        CHECK(bad == 0, "uns idempotence M=%d bad=%d", M, bad);
    }
    float m, v;
    adamDecode(0.0f, 0.0f, m, v);
    CHECK(m == 0.0f && v == 0.0f, "adam (0,0) -> m = v = 0");
    CHECK(adamU(0.0f, 0.0f) == 0.0f && adamLogS(0.0f) == 0.0f, "adam m = v = 0 -> (0,0)");

    // attempt 2 (v8 s1.1): q_log / q_exp exact at the zero points, accuracy as measured (results/q2_2026-11-02,
    // max abs 4.010e-6 / max rel 8.048e-7 on the same 2e6 inputs), monotone log; Adam round trip through them
    CHECK(q_log(1.0f) == 0.0f && q_exp(0.0f) == 1.0f, "q_log(1) = 0, q_exp(0) = 1");
    {
        std::mt19937_64 g64(5); std::uniform_real_distribution<double> U(0, 1);
        double wl = 0, we = 0;
        for (int i = 0; i < 2000000; i++) {
            float z = 1.0f + ((float)std::exp(U(g64) * 64.0) - 1.0f);
            wl = std::max(wl, std::fabs((double)q_log(z) - std::log((double)z)));
            float L = (float)(U(g64) * 64.0);
            we = std::max(we, std::fabs((double)q_exp(L) / std::exp((double)L) - 1.0));
        }
        // monotonicity, exhaustive (Codex C1 #3): every float in [1, 2^94) for q_log, [0, 64] for q_exp. q_log reverses
        // by one ulp at some sqrt(2) 2^k branch points (measured: 6, bits 0x?(3|b)5 04f4); any other reversal fails.
        int monoL = 0, offBranch = 0, monoE = 0; uint32_t worst = 0;
        float prev = q_log(1.0f);
        for (uint32_t x = q_bits(1.0f) + 1; x < q_bits(1.0f) + (94u << 23); x++) {
            const float l = q_log(q_float(x));
            if (l < prev) { monoL++; worst = std::max(worst, q_bits(prev) - q_bits(l)); offBranch += (x & 0x007FFFFFu) != 0x003504F4u; }
            prev = l;
        }
        prev = q_exp(0.0f);
        for (uint32_t x = 1; x <= q_bits(64.0f); x++) { const float e = q_exp(q_float(x)); monoE += e < prev; prev = e; }
        CHECK(wl <= 4.02e-6 && we <= 8.06e-7 && worst <= 1 && offBranch == 0 && monoE == 0,
              "q_log max abs %.4g, q_exp max rel %.4g, q_log reversals %d (max %u ulp, off branch point %d), q_exp reversals %d",
              wl, we, monoL, worst, offBranch, monoE);
        printf("q_log max abs err %.4g, q_exp max rel err %.4g; exhaustive: q_log reversals %d (max %u ulp, all at sqrt(2) branch "
               "points: %s), q_exp reversals %d\n", wl, we, monoL, worst, offBranch ? "no" : "yes", monoE);
    }
    {   // fixed non-SH bounds: decode(encode) of in-range moments stays within one u / log_s quantum
        std::mt19937 g(9); std::uniform_real_distribution<float> uu(-NS_AU, NS_AU), ll(0.f, NS_H);
        int bad = 0;
        for (int i = 0; i < 200000; i++) {
            float L = ll(g), u = uu(g), sv = ADAM_EPS * (q_exp(L) - 1.0f), vv = sv * sv, mm = u * (sv + ADAM_EPS), m2, v2;
            if (!(vv > 1e-36f)) continue;
            int cu = encSym(adamU(mm, vv), NS_AU, 32767), cl = encUns(adamLogS(vv), NS_H, 65535);
            adamDecode(decSym(cu, NS_AU, 32767), decUns(cl, NS_H, 65535), m2, v2);
            bad += std::fabs(decSym(cu, NS_AU, 32767) - adamU(mm, vv)) > NS_AU / 32767 || std::fabs(decUns(cl, NS_H, 65535) - adamLogS(vv)) > NS_H / 65535;
        }
        CHECK(bad == 0, "fixed-bound Adam round trip off by more than one quantum in %d cells", bad);
    }
    // SH value bound rule: 0 stays 0, floor SH_A_MIN, else 1.25 max
    CHECK(shBound(0.0f) == 0.0f && shBound(1e-5f) == SH_A_MIN && shBound(7.9e-4f) == SH_A_MIN && shBound(1.0f) == 1.25f,
          "shBound: 0 -> 0, small -> 1e-3, 1 -> 1.25");
}

// R1 study: one Adam element with random gradients for `warm` steps, then g = 0 for `quiet` steps.
// fp32 reference vs (u, log_s) codes with fixed block bounds (set at 1.25 x the warm-phase maxima,
// grow-only). Rounding: 0 = nearest (rint), 1 = toward zero, 2 = deterministic hash stochastic.
static uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
static float roundMode(float t, int mode, uint32_t key) {
    if (mode == 0) return std::rint(t);
    if (mode == 1) return std::trunc(t);
    float r = (hash32(key) >> 8) * (1.0f / 16777216.0f);
    return std::floor(t + r);
}
struct R1 { double driftQuiet, driftRef, activeErr; int uCode, lsCode; };
static R1 r1run(int UM, int LM, int mode, int warm, int quiet, uint32_t seed) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, lr = 1.6e-4f;
    std::mt19937 g(seed); std::normal_distribution<float> n(0.f, 1e-3f);
    std::vector<float> grads(warm); for (auto &x : grads) x = n(g);
    // pass 1: fp32 reference, and the warm-phase maxima for the bounds
    float m = 0, v = 0, p = 0, maxU = 0, maxL = 0;
    double pWarm = 0;
    for (int t = 1; t <= warm + quiet; t++) {
        float gr = t <= warm ? grads[t - 1] : 0.f;
        m = std::fma(b1, m, (1 - b1) * gr); v = std::fma(b2, v, (1 - b2) * gr * gr);
        float bc1 = 1 - std::pow(b1, (float)t), bc2s = std::sqrt(1 - std::pow(b2, (float)t));
        p -= (lr / bc1) * m / (std::sqrt(v) / bc2s + eps);
        if (t <= warm) { maxU = std::max(maxU, std::fabs(adamU(m, v))); maxL = std::max(maxL, adamLogS(v)); }
        if (t == warm) pWarm = p;
    }
    const double pRefEnd = p;
    const float aU = GROW * maxU, hL = GROW * maxL;
    // pass 2: quantized state
    int cu = 0, cl = 0; float q = 0; double qWarm = 0;
    for (int t = 1; t <= warm + quiet; t++) {
        float gr = t <= warm ? grads[t - 1] : 0.f;
        float u = decSym(cu, aU, UM), ls = decUns(cl, hL, LM), mm, vv;
        adamDecode(u, ls, mm, vv);
        mm = std::fma(b1, mm, (1 - b1) * gr); vv = std::fma(b2, vv, (1 - b2) * gr * gr);
        float bc1 = 1 - std::pow(b1, (float)t), bc2s = std::sqrt(1 - std::pow(b2, (float)t));
        q -= (lr / bc1) * mm / (std::sqrt(vv) / bc2s + eps);
        float tu = (adamU(mm, vv) * (float)UM) / aU, tl = (adamLogS(vv) * (float)LM) / hL;
        cu = (int)std::clamp(roundMode(tu, mode, (uint32_t)t * 2654435761u), (float)-UM, (float)UM);
        cl = (int)std::clamp(roundMode(tl, mode, (uint32_t)t * 2246822519u + 1u), 0.f, (float)LM);
        if (t == warm) qWarm = q;
    }
    return {q - qWarm, pRefEnd - pWarm, std::fabs(qWarm - pWarm), cu, cl};
}

int main() {
    exactProperties();
    printf("exact properties: %s (%d failures)\n", fails ? "FAIL" : "ok", fails);
    const char *modes[] = {"nearest", "toward-zero", "hash-stochastic"};
    printf("\nR1 study (lr 1.6e-4, warm 2000 steps g~N(0,1e-3), then 25000 steps g=0; mean over 50 seeds):\n");
    for (auto [UM, LM, name] : {std::tuple{127, 255, "SH Adam int8"}, std::tuple{32767, 65535, "non-SH Adam int16"}}) {
        for (int mode = 0; mode < 3; mode++) {
            double dq = 0, dr = 0, ae = 0; int stuck = 0;
            for (uint32_t s = 0; s < 50; s++) {
                R1 r = r1run(UM, LM, mode, 2000, 25000, s + 1);
                dq += std::fabs(r.driftQuiet); dr += std::fabs(r.driftRef); ae += r.activeErr; stuck += r.uCode != 0;
            }
            printf("  %-17s %-15s |quiet drift| q=%.3e fp32=%.3e  (ratio %.1f)  active-phase |err|=%.3e  u code still != 0 after quiet: %d/50\n",
                   name, modes[mode], dq / 50, dr / 50, dq / dr, ae / 50, stuck);
        }
    }
    return fails ? 1 : 0;
}
