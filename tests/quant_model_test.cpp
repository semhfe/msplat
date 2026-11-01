// Q1-Q3 Model CPU checks (P1; runs/2026-11-01-hiz-q-yerlesim.md s5). No GPU kernel is dispatched:
// the Model allocates shared Metal buffers and every operation tested here is CPU code on them.
// Build (after `cmake --build build-q`; default.metallib must sit next to the binary):
//   xcrun clang++ -std=c++17 -O2 -Icore/include -Icore/metal tests/quant_model_test.cpp build-q/libmsplat_core.a \
//     -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders \
//     -framework ImageIO -framework CoreGraphics -o build-q/quant_model_test && build-q/quant_model_test
#include <fstream>
#include <iterator>
#include "bindings.h"
#define private public
#include "model.hpp"
#undef private
#include "quant_codec.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>

using namespace qcodec;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

// |error| bound of one encode at bound a, with the float slack of quant_codec_test.cpp.
static double qtol(float a) { return a / SH_M / 2.0 * (1 + 1e-5) + 4e-7 * a; }
static int degOf(int64_t j) { int64_t k = j / 3; return k < 3 ? 0 : k < 8 ? 1 : 2; }

// Seeds in a 2-unit cube, cap 4x seeds, quant-level 1; SH codes and bounds random, fp32 params as the
// Package B check fills them, Adam codes random non-zero.
static Model *makeModel(int seeds, int cap, int quant, int shd = 3) {
    InputData d;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    d.points.count = seeds;
    for (int i = 0; i < seeds * 3; i++) { d.points.xyz.push_back(u(rng)); d.points.rgb.push_back((uint8_t)(rng() & 255)); }
    Model *m = new Model(d, 1, shd, 1000, cap, 0.f, 0.01f, 0.01f, 30000, false, nullptr, 30000, false, 0.5f, 0.1f, quant);
    m->cull_radius = 3.0f;
    std::normal_distribution<float> n(0.f, 1.f);
    MTensor *bufs[] = {&m->means_buf, &m->scales_buf, &m->quats_buf, &m->featuresDc_buf, &m->opacities_buf};
    for (int k = 0; k < 5; k++) {
        float *p = bufs[k]->data<float>(); int64_t st = bufs[k]->stride0();
        for (int64_t i = 0; i < (int64_t)m->num_active * st; i++) {
            float v = n(rng);
            if (k == 0) v *= 0.5f;
            if (k == 1) v = -4.f + 0.5f * v;
            if (k == 4) v = 1.0f + 2.f * v;
            p[i] = v;
        }
    }
    if (!quant) return m;
    int16_t *c = m->fr_codes.data<int16_t>();
    float *a = m->fr_bounds.data<float>();
    for (int64_t b = 0; b < m->fr_bounds.size(0); b++)
        for (int dd = 0; dd < 3; dd++) a[b * 3 + dd] = 0.05f + std::fabs(n(rng));
    for (int64_t i = 0; i < m->fr_codes.numel(); i++) c[i] = (int16_t)((int)(rng() % 65535) - 32767);
    uint16_t *s = m->shq_state.data<uint16_t>();
    for (int64_t i = 0; i < m->shq_state.numel(); i++) s[i] = (uint16_t)(rng() | 1);
    for (int g = 0; g < Model::N_NSQ; g++) {
        uint32_t *q = m->nsq_state[g].data<uint32_t>();
        for (int64_t i = 0; i < m->nsq_state[g].numel(); i++) q[i] = rng() | 0x10001u;
        float *qb = m->nsq_bounds[g].data<float>();
        for (int64_t i = 0; i < m->nsq_bounds[g].numel(); i++) qb[i] = 1.0f + i;
    }
    return m;
}

static float dec(const Model &m, int row, int64_t j) {
    const int64_t st = m.num_rest_coeffs * 3;
    return decSym(m.fr_codes.data<int16_t>()[row * st + j], m.fr_bounds.data<float>()[(row / Model::QB) * 3 + degOf(j)], SH_M);
}

static void layout() {
    Model *m = makeModel(3000, 12000, 1);
    const int64_t nblk = (12000 + 255) / 256;
    CHECK(m->fr_codes.dtype() == DType::Int16 && m->fr_codes.size(0) == nblk * 256 && m->fr_codes.size(1) == 15
          && m->fr_codes.size(2) == 3, "fr_codes shape");
    CHECK(m->fr_bounds.size(0) == nblk && m->shq_bounds.numel() == nblk * 6 && m->shq_state.dtype() == DType::UInt16,
          "SH bounds / state shape");
    const int dims[5] = {3, 3, 4, 3, 1};
    for (int g = 0; g < 5; g++)
        CHECK(m->nsq_state[g].size(1) == dims[g] && m->nsq_state[g].dtype() == DType::UInt32
              && m->nsq_bounds[g].numel() == nblk * 2, "nsq group %d shape", g);
    CHECK(!m->featuresRest_buf.defined() && m->featuresRest.stride0() == 0 && m->numRestCoeffs() == 15, "no fp32 SH store");
    for (int g = 0; g < Model::N_ADAM_GROUPS; g++)
        CHECK(!m->adam_exp_avg_buf[g].defined() && !m->adam_exp_avg_sq_buf[g].defined(), "no fp32 Adam store %d", g);
    // (P1 checked here that render threw in Q mode; P2 implements that path, and this test dispatches no GPU work.)
    delete m;
    // fresh seeds: fr_rest = 0 encodes to code 0, bound 0
    InputData d; d.points.count = 300;
    for (int i = 0; i < 900; i++) { d.points.xyz.push_back((i % 7) * 0.1f); d.points.rgb.push_back(9); }
    Model z(d, 1, 3, 1000, 10000, 0.f, 0.01f, 0.01f, 30000, false, nullptr, 30000, false, 0.5f, 0.1f, 1);
    bool zero = true;
    for (int64_t i = 0; i < z.fr_codes.numel(); i++) zero &= z.fr_codes.data<int16_t>()[i] == 0;
    for (int64_t i = 0; i < z.fr_bounds.numel(); i++) zero &= z.fr_bounds.data<float>()[i] == 0.0f;
    CHECK(zero, "seed encode: codes 0, bounds 0");
    bool fixed = true;  // attempt 2: non-SH / DC Adam bounds fixed at (8, 64) in every block
    for (int g = 0; g < Model::N_NSQ; g++)
        for (int64_t b = 0; b < z.nsq_bounds[g].size(0); b++)
            fixed &= z.nsq_bounds[g].data<float>()[b * 2] == NS_AU && z.nsq_bounds[g].data<float>()[b * 2 + 1] == NS_H;
    CHECK(fixed, "nsq_bounds fixed at (8, 64)");
}

// Attempt 2 SH value bound floor: a widening copy of small values and a shrink below 1e-3 give the bound SH_A_MIN; a
// block already at the floor is not shrunk again; an all-zero block shrinks to 0.
static void boundFloor() {
    Model *m = makeModel(3000, 12000, 1);
    const int64_t st = 45;
    int16_t *c = m->fr_codes.data<int16_t>();
    float *a = m->fr_bounds.data<float>();
    // widen: source block 2 bounds 1e-4 (|x| <= 1e-4), destination block 40 bounds 0 -> new bound 1e-3
    for (int dd = 0; dd < 3; dd++) { a[2 * 3 + dd] = 1e-4f; a[40 * 3 + dd] = 0.0f; }
    for (int r = 40 * 256; r < 41 * 256; r++) for (int64_t j = 0; j < st; j++) c[r * st + j] = 0;
    const int src = 2 * 256 + 17, dst = 40 * 256 + 5, w0 = m->q_widen;
    m->copyGaussianRow(dst, src);
    bool ok = m->q_widen == w0 + 3;
    for (int dd = 0; dd < 3; dd++) ok &= a[40 * 3 + dd] == SH_A_MIN;
    for (int64_t j = 0; j < st; j++) ok &= std::fabs((double)dec(*m, dst, j) - dec(*m, src, j)) <= qtol(SH_A_MIN);
    CHECK(ok, "widen of values below 1e-3 sets the bound to SH_A_MIN");
    // shrink: block 5 degree 0 (bound 0.05) with |x| <= 0.05/1000 -> SH_A_MIN; block 6 degree 0 at the floor with small
    // values -> unchanged and not counted; block 7 degree 0 all zero -> 0
    a[5 * 3] = 0.05f; a[6 * 3] = SH_A_MIN; a[7 * 3] = 0.3f;
    for (int r = 5 * 256; r < 8 * 256; r++)
        for (int64_t j = 0; j < 9; j++) c[r * st + j] = r < 7 * 256 ? (int16_t)((int)c[r * st + j] / 1000) : 0;
    std::vector<float> b5(256 * 9), b6(256 * 9);
    for (int r = 0; r < 256; r++) for (int64_t j = 0; j < 9; j++) { b5[r * 9 + j] = dec(*m, 5 * 256 + r, j); b6[r * 9 + j] = dec(*m, 6 * 256 + r, j); }
    // every other block keeps random codes near M, so it never shrinks; only blocks 5 and 7 (degree 0) do
    const int s0 = m->q_shrink;
    m->shrinkQuant();
    ok = a[5 * 3] == SH_A_MIN && a[6 * 3] == SH_A_MIN && a[7 * 3] == 0.0f && m->q_shrink == s0 + 2;
    for (int r = 0; r < 256; r++)
        for (int64_t j = 0; j < 9; j++) {
            ok &= std::fabs((double)dec(*m, 5 * 256 + r, j) - b5[r * 9 + j]) <= qtol(SH_A_MIN);
            ok &= dec(*m, 6 * 256 + r, j) == b6[r * 9 + j];
        }
    CHECK(ok, "shrink: floor SH_A_MIN, at-floor block untouched, zero block -> 0 (shrinks %d)", m->q_shrink - s0);
    delete m;
}

static void rowCopy() {
    Model *m = makeModel(3000, 12000, 1);
    const int64_t st = 45;
    float *a = m->fr_bounds.data<float>();
    // same block: exact decode, bounds untouched
    int samePairs = 0;
    for (int k = 0; k < 200; k++) {
        int b = k % 11, src = b * 256 + (k * 37) % 256, dst = b * 256 + (k * 91 + 5) % 256;
        if (src == dst) continue;
        float a0[3] = {a[b*3], a[b*3+1], a[b*3+2]};
        int w0 = m->q_widen;
        m->copyGaussianRow(dst, src);
        bool ok = m->q_widen == w0 && !memcmp(a0, a + b * 3, sizeof(a0));
        for (int64_t j = 0; j < st; j++) ok &= dec(*m, dst, j) == dec(*m, src, j);
        CHECK(ok, "same-block copy %d -> %d", src, dst);
        samePairs++;
    }
    // equal bounds in different blocks: exact
    for (int dd = 0; dd < 3; dd++) a[20 * 3 + dd] = a[3 * 3 + dd];
    m->copyGaussianRow(20 * 256 + 4, 3 * 256 + 9);
    bool eq = true;
    for (int64_t j = 0; j < st; j++) eq &= dec(*m, 20 * 256 + 4, j) == dec(*m, 3 * 256 + 9, j);
    CHECK(eq, "equal-bound cross-block copy exact");
    // cross block, destination bound larger: no widening, |dst - src| <= q_dst/2
    for (int dd = 0; dd < 3; dd++) a[30 * 3 + dd] = 10.0f * a[5 * 3 + dd];
    int w0 = m->q_widen;
    m->copyGaussianRow(30 * 256 + 1, 5 * 256 + 2);
    bool nw = m->q_widen == w0;
    for (int64_t j = 0; j < st; j++)
        nw &= std::fabs((double)dec(*m, 30 * 256 + 1, j) - dec(*m, 5 * 256 + 2, j)) <= qtol(a[30 * 3 + degOf(j)]);
    CHECK(nw, "cross-block copy without widening");
    // cross block, destination bound smaller: widens all three degrees
    int widenCases = 0;
    for (int k = 0; k < 20; k++) {
        int bs = 2 + k, bd = 45 - k, src = bs * 256 + (k * 13) % 256, dst = bd * 256 + (k * 29 + 7) % 256;
        for (int dd = 0; dd < 3; dd++) a[bd * 3 + dd] = 0.01f * a[bs * 3 + dd];
        std::vector<float> before(256 * st);
        for (int r = 0; r < 256; r++) for (int64_t j = 0; j < st; j++) before[r * st + j] = dec(*m, bd * 256 + r, j);
        int wbefore = m->q_widen;
        m->copyGaussianRow(dst, src);
        bool ok = m->q_widen == wbefore + 3;
        double worstCopy = 0, worstOther = 0;
        for (int64_t j = 0; j < st; j++) {
            const double qn = qtol(a[bd * 3 + degOf(j)]);
            double e = std::fabs((double)dec(*m, dst, j) - dec(*m, src, j));
            ok &= e <= qn; worstCopy = std::max(worstCopy, e / qn);
            for (int r = 0; r < 256; r++) {
                if (bd * 256 + r == dst) continue;
                double eo = std::fabs((double)dec(*m, bd * 256 + r, j) - before[r * st + j]);
                ok &= eo <= qn; worstOther = std::max(worstOther, eo / qn);
            }
        }
        CHECK(ok, "widening copy %d -> %d (worst/tol copy %.3f other %.3f)", src, dst, worstCopy, worstOther);
        widenCases++;
    }
    // Adam state untouched by the copy, zeroed by zeroAdamAt; bounds untouched
    const int dst = 7 * 256 + 3;
    std::vector<float> nb0(m->nsq_bounds[2].data<float>(), m->nsq_bounds[2].data<float>() + m->nsq_bounds[2].numel());
    uint32_t keep = m->nsq_state[0].data<uint32_t>()[(dst + 1) * 3];
    m->zeroAdamAt(dst);
    bool z = true;
    for (int64_t i = 0; i < 45; i++) z &= m->shq_state.data<uint16_t>()[dst * 45 + i] == 0;
    const int dims[5] = {3, 3, 4, 3, 1};
    for (int g = 0; g < 5; g++) for (int i = 0; i < dims[g]; i++) z &= m->nsq_state[g].data<uint32_t>()[dst * dims[g] + i] == 0;
    z &= m->nsq_state[0].data<uint32_t>()[(dst + 1) * 3] == keep && m->shq_state.data<uint16_t>()[(dst + 1) * 45] != 0;
    z &= !memcmp(nb0.data(), m->nsq_bounds[2].data<float>(), nb0.size() * sizeof(float));
    CHECK(z, "zeroAdamAt zeroes the row codes only");
    printf("{\"check\": \"rowcopy_q\", \"same_block_pairs\": %d, \"widen_cases\": %d, \"widens\": %d}\n",
           samePairs, widenCases, m->q_widen);
    delete m;
}

static void shrink() {
    Model *m = makeModel(3000, 12000, 1);
    const int64_t st = 45;
    int16_t *c = m->fr_codes.data<int16_t>();
    float *a = m->fr_bounds.data<float>();
    // block 4, degree 1: every |code| <= M/4 -> shrinks; block 6: one code at M -> stays; block 8 degree 2: NaN bound
    for (int r = 4 * 256; r < 5 * 256; r++) for (int64_t j = 9; j < 24; j++) c[r * st + j] /= 4;
    c[6 * 256 * st + 9] = SH_M;
    for (int r = 6 * 256; r < 7 * 256; r++) for (int64_t j = 9; j < 24; j++) c[r * st + j] /= (r == 6 * 256 ? 1 : 4);
    a[8 * 3 + 2] = NAN;
    std::vector<float> before(256 * st), before6(256 * st);
    for (int r = 0; r < 256; r++) for (int64_t j = 0; j < st; j++) {
        before[r * st + j] = dec(*m, 4 * 256 + r, j);
        before6[r * st + j] = dec(*m, 6 * 256 + r, j);
    }
    const float a41 = a[4 * 3 + 1], a61 = a[6 * 3 + 1];
    int mc = 0;
    for (int r = 4 * 256; r < 5 * 256; r++) for (int64_t j = 9; j < 24; j++) mc = std::max(mc, std::abs((int)c[r * st + j]));
    // block 11 shrinks only if every block's maxima are below a/2 (random codes reach ~M): count shrinks exactly
    const int s0 = m->q_shrink;
    m->shrinkQuant();
    bool ok = a[4 * 3 + 1] == GROW * decSym(mc, a41, SH_M) && a[4 * 3 + 1] < a41 && a[6 * 3 + 1] == a61;
    for (int r = 0; r < 256; r++) for (int64_t j = 0; j < st; j++) {
        ok &= std::fabs((double)dec(*m, 4 * 256 + r, j) - before[r * st + j]) <= (degOf(j) == 1 ? qtol(a[4 * 3 + 1]) : 0.0);
        ok &= dec(*m, 6 * 256 + r, j) == before6[r * st + j];
    }
    bool nanReset = a[8 * 3 + 2] == 0.0f;
    for (int r = 8 * 256; r < 9 * 256; r++) for (int64_t j = 24; j < 45; j++) nanReset &= c[r * st + j] == 0;
    CHECK(ok, "shrink: block 4 degree 1 shrunk within q_new/2, block 6 untouched");
    CHECK(nanReset, "shrink: non-finite bound reset to 0 with its codes");
    CHECK(m->q_shrink == s0 + 1, "shrink count %d", m->q_shrink - s0);
    // mcmcAfterTrain: a NaN bound makes its block's Gaussians nonfinite; after the pass every bound is finite
    a[9 * 3 + 0] = INFINITY;
    m->mcmcAfterTrain(1000);
    bool fin = true;
    for (int64_t b = 0; b < (m->num_active + 255) / 256; b++)
        for (int dd = 0; dd < 3; dd++) fin &= std::isfinite(a[b * 3 + dd]);
    MTensor fr = m->decodeFeaturesRest();
    for (int64_t i = 0; i < fr.numel(); i++) fin &= std::isfinite(fr.data<float>()[i]);
    CHECK(fin, "mcmcAfterTrain leaves finite bounds and decoded SH");
    delete m;
}

static void checkpoint() {
    char dir[] = "/tmp/quant_model_test_XXXXXX";
    if (!mkdtemp(dir)) { CHECK(false, "mkdtemp"); return; }
    const std::string p2 = std::string(dir) + "/q.ckpt", p1 = std::string(dir) + "/f.ckpt", ply = std::string(dir) + "/q.ply";
    Model *m = makeModel(3000, 12000, 1);
    m->adam_step_count = 1234;
    m->num_active = 2900;  // partial last block
    m->refreshViews();
    m->saveCheckpoint(p2, 777);
    Model *r = makeModel(500, 12000, 1);
    int step = r->loadCheckpoint(p2);
    const int64_t nba = (2900 + 255) / 256;
    bool ok = step == 777 && r->num_active == 2900 && r->adam_step_count == 1234 && r->num_rest_coeffs == 15
           && r->fr_codes.size(0) == m->fr_codes.size(0);
    auto same = [&](const MTensor &x, const MTensor &y, size_t bytes) { return !memcmp(x.data_ptr(), y.data_ptr(), bytes); };
    ok &= same(m->means, r->means, m->means.nbytes()) && same(m->scales, r->scales, m->scales.nbytes())
       && same(m->quats, r->quats, m->quats.nbytes()) && same(m->featuresDc, r->featuresDc, m->featuresDc.nbytes())
       && same(m->opacities, r->opacities, m->opacities.nbytes());
    ok &= same(m->fr_codes, r->fr_codes, nba * 256 * 45 * 2) && same(m->fr_bounds, r->fr_bounds, nba * 3 * 4)
       && same(m->shq_state, r->shq_state, nba * 256 * 45 * 2) && same(m->shq_bounds, r->shq_bounds, nba * 6 * 4);
    for (int g = 0; g < 5; g++)
        ok &= same(m->nsq_state[g], r->nsq_state[g], m->nsq_state[g].stride0() * nba * 256 * 4)
           && same(m->nsq_bounds[g], r->nsq_bounds[g], nba * 2 * 4);
    bool tail = true;  // rows past the saved blocks are code 0
    for (int64_t i = nba * 256 * 45; i < r->fr_codes.numel(); i++) tail &= r->fr_codes.data<int16_t>()[i] == 0;
    CHECK(ok && tail, "checkpoint v2 round trip");
    // version mismatch refused both ways
    Model *f = makeModel(500, 12000, 0);
    f->saveCheckpoint(p1, 5);
    bool t1 = false, t2 = false;
    try { r->loadCheckpoint(p1); } catch (const std::runtime_error &) { t1 = true; }
    try { f->loadCheckpoint(p2); } catch (const std::runtime_error &) { t2 = true; }
    CHECK(t1 && t2, "checkpoint version / quant-level mismatch refused (v1->Q %d, v2->fp32 %d)", t1, t2);
    // Codex C4 (attempt 2): an attempt-1 version-2 file and a truncated file are refused
    {
        std::ifstream in(p2, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string pv = std::string(dir) + "/v2.ckpt", pt = std::string(dir) + "/t.ckpt";
        std::string v2 = bytes; const uint32_t two = 2; memcpy(&v2[4], &two, 4);
        std::ofstream(pv, std::ios::binary).write(v2.data(), v2.size());
        std::ofstream(pt, std::ios::binary).write(bytes.data(), bytes.size() - 1000);
        bool tv = false, tt = false;
        Model *q = makeModel(500, 12000, 1);
        try { q->loadCheckpoint(pv); } catch (const std::runtime_error &) { tv = true; }
        try { q->loadCheckpoint(pt); } catch (const std::runtime_error &) { tt = true; }
        CHECK(tv && tt, "version-2 checkpoint refused %d, truncated checkpoint refused %d", tv, tt);
        // Codex C4 follow-up: an absurd dimension (means rows 2^62) is refused before any allocation
        const std::string ph = std::string(dir) + "/h.ckpt";
        std::string hb = bytes;
        char pat[20]; const uint32_t two2 = 2; const int64_t np = 2900, three = 3, huge = int64_t(1) << 62;
        memcpy(pat, &two2, 4); memcpy(pat + 4, &np, 8); memcpy(pat + 12, &three, 8);
        const size_t at = hb.find(std::string(pat, 20));
        bool th = false;
        if (at != std::string::npos) {
            memcpy(&hb[at + 4], &huge, 8);
            std::ofstream(ph, std::ios::binary).write(hb.data(), hb.size());
            try { q->loadCheckpoint(ph); } catch (const std::runtime_error &) { th = true; }
        }
        CHECK(at != std::string::npos && th, "means header with 2^62 rows refused (found %d, thrown %d)", at != std::string::npos, th);
        // compound overflow: [2^20, 2^19, 2^25] passes each dimension limit and wraps to 0 elements unless checked
        // before multiplying; written as a 3-D first tensor with a 0 byte count
        bool tc = false;
        if (at != std::string::npos) {
            std::string cb = bytes.substr(0, at);
            const uint32_t three3 = 3; const int64_t dims[3] = {int64_t(1) << 20, int64_t(1) << 19, int64_t(1) << 25}; const uint64_t zero = 0;
            cb.append(reinterpret_cast<const char*>(&three3), 4); cb.append(reinterpret_cast<const char*>(dims), 24);
            cb.append(reinterpret_cast<const char*>(&zero), 8); cb.append(bytes.substr(at + 20 + 8));
            std::ofstream(ph, std::ios::binary).write(cb.data(), cb.size());
            try { q->loadCheckpoint(ph); } catch (const std::runtime_error &) { tc = true; }
        }
        CHECK(tc, "header [2^20, 2^19, 2^25] with 0 bytes refused (thrown %d)", tc);
        delete q; std::remove(pv.c_str()); std::remove(pt.c_str()); std::remove(ph.c_str());
    }
    // Codex C4 follow-up: degree 0 ([rows, 0, 3] SH stores) round-trips; a degree mismatch is refused
    {
        const std::string p0 = std::string(dir) + "/d0.ckpt";
        Model *m0 = makeModel(1000, 4000, 1, 0), *r0 = makeModel(300, 4000, 1, 0);
        m0->saveCheckpoint(p0, 9);
        bool ok0 = false, tm = false;
        try { ok0 = r0->loadCheckpoint(p0) == 9 && r0->num_active == m0->num_active && r0->num_rest_coeffs == 0
                    && !memcmp(r0->means.data_ptr(), m0->means.data_ptr(), m0->means.nbytes()); } catch (const std::runtime_error &) {}
        try { r->loadCheckpoint(p0); } catch (const std::runtime_error &) { tm = true; }
        CHECK(ok0 && tm, "degree-0 checkpoint round trip %d, degree-0 file into a degree-3 model refused %d", ok0, tm);
        delete m0; delete r0; std::remove(p0.c_str());
    }
    // PLY: decoded SH written, read back bit-exact into a fp32 model
    m->savePly(ply, 777);
    f->loadPly(ply);
    MTensor fr = m->decodeFeaturesRest();
    CHECK(f->num_active == 2900 && f->featuresRest.nbytes() == fr.nbytes() && same(f->featuresRest, fr, fr.nbytes()),
          "savePly writes the decoded SH");
    delete m; delete r; delete f;
    std::remove(p1.c_str()); std::remove(p2.c_str()); std::remove(ply.c_str()); rmdir(dir);
}

// Codex C4 (attempt 2): data with more than 15 rest coefficients (degree-4 PLY) is refused, not truncated
static void degreeGuard() {
    Model *f = makeModel(500, 2000, 0);
    f->featuresRest_buf = gpu_zeros({2000, 24, 3}, DType::Float32);
    f->refreshViews();
    bool thrown = false;
    try { f->setupQuant(f->num_active, {}); } catch (const std::runtime_error &) { thrown = true; }
    CHECK(thrown && f->featuresRest.size(-2) == 24, "setupQuant refuses 24 rest coefficients (thrown %d)", thrown);
    delete f;
}

int main() {
    layout();
    degreeGuard();
    rowCopy();
    shrink();
    boundFloor();
    checkpoint();
    printf(fails ? "%d FAILURES\n" : "all OK\n", fails);
    return fails ? 1 : 0;
}
