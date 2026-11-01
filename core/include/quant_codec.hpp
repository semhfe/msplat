// Q1-Q3 (--quant-level 1) CPU reference codecs. Design: docs Q_KODEK_TASARIM_NOTU.md s2.
// Every codec maps 0 to code 0 and code 0 back to exactly 0. The Metal kernels implement the
// same formulas in the same operation order (multiply, then divide), so the SH value codec is
// bit-identical between CPU and GPU (checked by the GPU parity test).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace qcodec {

constexpr int SH_M = 32767;            // Q1 SH values: int16 symmetric
constexpr float GROW = 1.25f;          // a bound that is exceeded grows to GROW * max|x|
// Attempt 2 (runs/2026-11-02-hiz-q2-onkayit.md s1): a non-zero SH value bound that grows or shrinks is at least
// SH_A_MIN; non-SH and DC Adam use the fixed bounds (NS_AU, NS_H) in every block, values beyond are clamped (usat, qsat).
constexpr float SH_A_MIN = 1e-3f;
constexpr float NS_AU = 8.0f, NS_H = 64.0f;
// New SH value bound for a non-zero block max mx (growth: mx > old bound; shrink: mx < old bound / 2).
inline float shBound(float mx) { return mx == 0.0f ? 0.0f : std::max(GROW * mx, SH_A_MIN); }

// Symmetric linear codec with bound a >= 0: x in [-a, a] -> integer code in [-M, M].
inline int encSym(float x, float a, int M) {
    if (!(a > 0.0f)) return 0;
    float c = std::rint((x * (float)M) / a);
    return (int)std::clamp(c, (float)-M, (float)M);
}
inline float decSym(int c, float a, int M) { return c == 0 ? 0.0f : ((float)c * a) / (float)M; }

// Unsigned linear codec with bound h >= 0: x in [0, h] -> [0, M].
inline int encUns(float x, float h, int M) {
    if (!(h > 0.0f) || !(x > 0.0f)) return 0;
    float c = std::rint((x * (float)M) / h);
    return (int)std::min(c, (float)M);
}
inline float decUns(int c, float h, int M) { return c == 0 ? 0.0f : ((float)c * h) / (float)M; }

// Deterministic log / exp (attempt 2), copied from runs/hiz-checks/results/q2_2026-11-02/qdet.h; the GPU copies in
// msplat_q.metal give the same bits (pre-registration v8 s1.1).
static inline uint32_t q_bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static inline float q_float(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
// natural log of z >= 1 (z finite): z = m * 2^e, m in [sqrt(1/2), sqrt(2)); log m = 2 atanh(t), t = (m - 1) / (m + 1)
static inline float q_log(float z) {
#pragma clang fp contract(off)
    uint32_t b = q_bits(z);
    int e = (int)((b >> 23) & 0xFF) - 127;
    float m = q_float((b & 0x007FFFFFu) | 0x3F800000u);          // [1, 2)
    if (m > 1.41421356f) { m = m * 0.5f; e = e + 1; }              // exact
    float t = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float p = 1.0f / 9.0f;
    p = p * t2 + 1.0f / 7.0f;
    p = p * t2 + 1.0f / 5.0f;
    p = p * t2 + 1.0f / 3.0f;
    p = p * t2 + 1.0f;
    float lm = 2.0f * t * p;
    return (float)e * 0.693147180559945f + lm;
}
// e^L for L in [0, 88]: L = k ln2 + r, r in [0, ln2); e^r by a fixed degree-7 polynomial; scale by 2^k via the exponent
static inline float q_exp(float L) {
#pragma clang fp contract(off)
    float kf = L * 1.44269504088896f;
    int k = (int)kf;                                                 // trunc = floor for L >= 0
    float r = L - (float)k * 0.693145751953125f;                     // ln2 hi (exact product for k < 2^11)
    r = r - (float)k * 1.42860682030941723212e-6f;                   // ln2 lo
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return q_float(q_bits(p) + ((uint32_t)k << 23));
}

// Adam moments (m, v) as (u, log_s): u = m / (sqrt(v) + EPS), log_s = log(1 + sqrt(v) / EPS).
// m = v = 0 -> u = log_s = 0 -> codes (0, 0), and codes (0, 0) decode to m = v = 0 (not conversely: small values also code to 0).
constexpr float ADAM_EPS = 1e-15f;
// log(1 + x) and exp(x) - 1 through q_log / q_exp on both sides (attempt 2; attempt 1 used std::log / std::exp here and
// precise::log / precise::exp on the GPU, which differ by an ulp).
inline float adamU(float m, float v) { return m / (std::sqrt(v) + ADAM_EPS); }
inline float adamLogS(float v) { return q_log(1.0f + std::sqrt(v) / ADAM_EPS); }
inline void adamDecode(float u, float logS, float &m, float &v) {
    float sv = logS == 0.0f ? 0.0f : ADAM_EPS * (q_exp(logS) - 1.0f);
    v = sv * sv;
    m = u * (sv + ADAM_EPS);
    if (u == 0.0f) m = 0.0f;
}

}  // namespace qcodec
