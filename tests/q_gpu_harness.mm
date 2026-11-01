// Q codec GPU harness (P2; runs/2026-11-01-hiz-q-kodek-onkayit.md s3, s4). Standalone Metal program, no libmsplat.
// Build: xcrun clang++ -std=c++17 -O2 -ffp-contract=off -fobjc-arc -Icore/include tests/q_gpu_harness.mm \
//          -framework Foundation -framework Metal -o build-q/q_gpu_harness
// Run:   tests/q_harness_run.sh <reference|mutant-check> <new|old>   (wraps the call below with the expected manifest sha)
//        build-q/q_gpu_harness --mode reference|mutant-check --fixture DIR --expect-manifest <sha256 of manifest.sha256>
//          [--metallib build-q/default.metallib]
// Fixture identity: the sha256 of manifest.sha256 must equal --expect-manifest, the manifest and the directory must hold
//   exactly the 36 expected files, and every file hash must match; otherwise exit 1.
//
// --mode reference (P2-M1): the s4.1 fixture-verification run. Per step t the Q state is decoded on the CPU, the
//   UNCHANGED fp32 kernels run (project_and_sh_backward_kernel: projection grads + SH/DC Adam; fused_adam_kernel:
//   means, scales, quats, opacity), and the fp32 results are encoded on the CPU with the Q algorithm (s0): a bound
//   exceeded by a newly computed value grows to 1.25*max; newly computed values follow --quant-round; a cell whose
//   value is byte-identical to its decoded old value keeps its code (bound not grown) or is re-encoded nearest
//   (bound grown). All three rounding modes run. It verifies the s4.2 coverage events, prints an inventory and
//   exits 1 if a required event is missing or an invariant fails. Not a gate result; no Q kernel runs.
// --mode mutant-check: runs the reference trajectory and, after every step, a candidate = the reference outputs with a
//   deliberate bug (every inactive row's SH Adam and DC Adam codes set to 0, bounds kept), fed to compareInactiveRows,
//   the s4.2 inactive-row comparator. PASS (exit 0) = the comparator flags the mutant in every run.
// --mode q (P2-M2): s4.2, the Q kernels of core/metal/msplat_q.metal against the unchanged fp32 kernels + CPU Q encoding;
//   --mode parity (P2-M2): s3 codec parity. See the P2-M2 block above main.
// Radii: a [steps][N] file (time-varying mask, v7) or an [N] file (static mask, fixtures before v7).
// Attempt 2 (runs/2026-11-02-hiz-q2-onkayit.md, v8): codec log/exp = q_log / q_exp (quant_codec.hpp); a grown SH value
//   bound is max(1.25 max, 1e-3); non-SH and DC Adam use fixed bounds (8, 64) with clamps counted in usat / qsat, and
//   their coverage events are usat = 0, qsat = 0 and non-zero codes > 0 instead of growth events.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "quant_codec.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace qcodec;

static const int QB = 256, ST = 45;
static const int SHA_MU = 127, SHA_ML = 255, NSA_MU = 32767, NSA_ML = 65535;
static const float H_MAX = 64.0f;
static const int NG = 5, GDIM[NG] = {3, 3, 4, 3, 1}, GADAM[NG] = {0, 1, 2, 3, 5};  // means scales quats dc opac
static const float LR[6] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f};
static const char *MODE_NAME[3] = {"nearest", "stochastic", "stochastic-adam"};

// component ids: 0-2 SH value d0-d2 | 3-5 SH Adam u d0-d2 | 6-8 SH Adam log_s d0-d2 | 9 + 2g, 10 + 2g non-SH group g (u, log_s)
static const int NCOMP = 19;
static const char *CN[NCOMP] = {"shv_d0", "shv_d1", "shv_d2", "shA_u_d0", "shA_u_d1", "shA_u_d2", "shA_ls_d0", "shA_ls_d1",
                                "shA_ls_d2", "means_u", "means_ls", "scales_u", "scales_ls", "quats_u", "quats_ls", "dc_u",
                                "dc_ls", "opac_u", "opac_ls"};

static int fails = 0;
#define INV(c, ...) do { if (!(c)) { if (++fails <= 30) { printf("INVARIANT FAIL: " __VA_ARGS__); printf("\n"); } } } while (0)

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static float hashR(uint32_t step, uint32_t cell, uint32_t comp) {
    return (float)(hash32(step * 2654435761u + cell * 2246822519u + comp * 3266489917u) >> 8) * 0x1p-24f;
}
static int encS(float x, float a, int M, bool st, uint32_t step, uint32_t cell, uint32_t comp) {
    if (!st) return encSym(x, a, M);
    if (!(a > 0.0f)) return 0;
    float c = std::floor((x * (float)M) / a + hashR(step, cell, comp));
    return (int)std::clamp(c, (float)-M, (float)M);
}
static int encU(float x, float h, int M, bool st, uint32_t step, uint32_t cell, uint32_t comp) {
    if (!st) return encUns(x, h, M);
    if (!(h > 0.0f) || !(x > 0.0f)) return 0;
    float c = std::floor((x * (float)M) / h + hashR(step, cell, comp));
    return (int)std::clamp(c, 0.0f, (float)M);
}
static double tolN(float a, int M) { return (a / (double)M / 2.0) * (1 + 1e-5) + 4e-7 * a + 2.0 * FLT_MIN; }
static bool same(float a, float b) { return !memcmp(&a, &b, 4); }
static int degOf(int j) { int k = j / 3; return k < 3 ? 0 : k < 8 ? 1 : 2; }
static bool isZ(int b) { return b <= 3; }
static bool isN(int b) { return b >= 100 && b <= 109; }

struct PackSH {  // uint16: bits 0-7 int8 u, bits 8-15 uint8 log_s
    using T = uint16_t;
    static int u(T c) { return (int8_t)(c & 0xFF); }
    static int l(T c) { return c >> 8; }
    static T pack(int u, int l) { return (T)((uint8_t)(int8_t)u | (l << 8)); }
};
struct PackNS {  // uint32: bits 0-15 int16 u, bits 16-31 uint16 log_s
    using T = uint32_t;
    static int u(T c) { return (int16_t)(c & 0xFFFF); }
    static int l(T c) { return (int)(c >> 16); }
    static T pack(int u, int l) { return (uint32_t)(uint16_t)(int16_t)u | ((uint32_t)l << 16); }
};

// Q state of one fixture: codes and bounds (s yerlesim s3).
struct QState {
    std::vector<int16_t> fr;  std::vector<float> frB;    // [rows,45] SH value codes, [nblk,3] bounds
    std::vector<uint16_t> shq; std::vector<float> shqB;  // [rows,45] SH Adam (u|log_s), [nblk,3,2] bounds
    std::vector<uint32_t> nsq[NG]; std::vector<float> nsqB[NG];  // [rows,dim_g] non-SH/DC Adam, [nblk,2] bounds
};

// s4.2 inactive-row comparator (SH value, SH Adam, DC Adam), shared by --mode reference, --mode mutant-check and,
// later, --mode q: compareInactiveRows(state entering step t, candidate state after step t, radii of step t, N).
// Per (block, component): bound bitwise unchanged => the code is byte-identical; bound changed (grown) => the decoded
// change is <= tol_n(new bound, M) for that component (SH value; u and log_s separately).
struct InactCmp { long cells = 0, flagged = 0; };
static InactCmp compareInactiveRows(const QState &prev, const QState &cand, const int32_t *radii, int N) {
    InactCmp r;
    auto chkSym = [&](int co, int cn, float ao, float an, int M) {
        r.cells++;
        bool ok = same(ao, an) ? co == cn : std::fabs((double)decSym(cn, an, M) - decSym(co, ao, M)) <= tolN(an, M);
        r.flagged += !ok;
    };
    auto chkUns = [&](int co, int cn, float ho, float hn, int M) {
        r.cells++;
        bool ok = same(ho, hn) ? co == cn : std::fabs((double)decUns(cn, hn, M) - decUns(co, ho, M)) <= tolN(hn, M);
        r.flagged += !ok;
    };
    for (int row = 0; row < N; row++) {
        if (radii[row] != 0) continue;
        const int b = row / QB;
        for (int j = 0; j < ST; j++) {
            const size_t c = (size_t)row * ST + j;
            const int d = degOf(j);
            chkSym(prev.fr[c], cand.fr[c], prev.frB[b * 3 + d], cand.frB[b * 3 + d], SH_M);
            chkSym(PackSH::u(prev.shq[c]), PackSH::u(cand.shq[c]), prev.shqB[b * 6 + d * 2], cand.shqB[b * 6 + d * 2], SHA_MU);
            chkUns(PackSH::l(prev.shq[c]), PackSH::l(cand.shq[c]), prev.shqB[b * 6 + d * 2 + 1], cand.shqB[b * 6 + d * 2 + 1], SHA_ML);
        }
        for (int k = 0; k < 3; k++) {  // DC Adam (group 3)
            const size_t c = (size_t)row * 3 + k;
            chkSym(PackNS::u(prev.nsq[3][c]), PackNS::u(cand.nsq[3][c]), prev.nsqB[3][b * 2], cand.nsqB[3][b * 2], NSA_MU);
            chkUns(PackNS::l(prev.nsq[3][c]), PackNS::l(cand.nsq[3][c]), prev.nsqB[3][b * 2 + 1], cand.nsqB[3][b * 2 + 1], NSA_ML);
        }
    }
    return r;
}

// ---------------- files ----------------
static std::string hexd(const unsigned char *d) {
    char b[65];
    for (int i = 0; i < 32; i++) snprintf(b + 2 * i, 3, "%02x", d[i]);
    return std::string(b, 64);
}
static std::map<std::string, std::vector<char>> g_files;
static const char *EXPECTED_FILES[] = {
    "camera.f32", "f1_means.f32", "f1_scales.f32", "f1_quats.f32", "f1_dc.f32", "f1_opac.f32", "f1_conics.f32",
    "f1_radii.i32", "f1_fr_codes.i16", "f1_fr_bounds.f32", "f1_degrees.u32", "f1_grads.f32", "f2_means.f32",
    "f2_scales.f32", "f2_quats.f32", "f2_dc.f32", "f2_opac.f32", "f2_conics.f32", "f2_radii.i32", "f2_fr_codes.i16",
    "f2_fr_bounds.f32", "f2_degrees.u32", "f2_grads.f32", "s3_sh_bounds.f32", "s3_sh_a.f32", "s3_sh_b.f32", "s3_sh_c.f32",
    "s3_sh_d.f32", "s3_adam_pairs.f32", "s3_adam_a.f32", "s3_adam_d.f32", "s3_adam_b_sh.i32", "s3_adam_c_sh.i32",
    "s3_adam_b_nsh.i32", "s3_adam_c_nsh.i32", "manifest_layout.txt"};
static const int N_EXPECTED = sizeof(EXPECTED_FILES) / sizeof(EXPECTED_FILES[0]);  // 36

// Fixture identity: manifest sha256 == expect, manifest and directory hold exactly the expected files, all hashes match.
static bool loadFixture(const std::string &dir, const std::string &expect) {
    std::ifstream mf(dir + "/manifest.sha256", std::ios::binary);
    if (!mf) { printf("FAIL: no manifest in %s\n", dir.c_str()); return false; }
    std::string mtext((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    unsigned char md[32];
    CC_SHA256(mtext.data(), (CC_LONG)mtext.size(), md);
    bool ok = hexd(md) == expect;
    printf("manifest.sha256 sha256 %s, expected %s: %s\n", hexd(md).c_str(), expect.c_str(), ok ? "match" : "MISMATCH");
    std::set<std::string> want(EXPECTED_FILES, EXPECTED_FILES + N_EXPECTED), listed, onDisk;
    NSArray<NSString *> *entries = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:@(dir.c_str()) error:nil];
    for (NSString *e in entries) if (![e isEqualToString:@"manifest.sha256"]) onDisk.insert(e.UTF8String);
    std::istringstream m(mtext);
    std::string line; int n = 0;
    printf("inventory:\n");
    while (std::getline(m, line)) {
        if (line.size() < 67) { printf("FAIL: malformed manifest line '%s'\n", line.c_str()); ok = false; continue; }
        std::string want_h = line.substr(0, 64), name = line.substr(66);
        if (!want.count(name) || listed.count(name)) { printf("FAIL: unexpected or duplicate manifest entry %s\n", name.c_str()); ok = false; }
        listed.insert(name);
        std::ifstream f(dir + "/" + name, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        unsigned char d[32];
        CC_SHA256(buf.data(), (CC_LONG)buf.size(), d);
        const bool hok = hexd(d) == want_h;
        printf("  %-22s %10zu B  %s\n", name.c_str(), buf.size(), hok ? "ok" : "HASH MISMATCH");
        ok &= hok;
        g_files[name] = std::move(buf);
        n++;
    }
    for (auto &w : want) if (!listed.count(w)) { printf("FAIL: manifest lacks %s\n", w.c_str()); ok = false; }
    for (auto &f : onDisk) if (!want.count(f)) { printf("FAIL: unexpected file in fixture dir: %s\n", f.c_str()); ok = false; }
    for (auto &w : want) if (!onDisk.count(w)) { printf("FAIL: missing file in fixture dir: %s\n", w.c_str()); ok = false; }
    printf("fixture: %s, %d files (expected %d), identity %s\n", dir.c_str(), n, N_EXPECTED, ok ? "verified" : "FAILED");
    return ok && n == N_EXPECTED;
}
template <class T> static const T *F(const std::string &name, size_t count) {
    auto it = g_files.find(name);
    if (it == g_files.end() || it->second.size() != count * sizeof(T)) {
        printf("FAIL: fixture file %s missing or wrong size\n", name.c_str());
        exit(1);
    }
    return reinterpret_cast<const T *>(it->second.data());
}

// ---------------- Metal ----------------
static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> psoBwd, psoAdam;
static id<MTLBuffer> buf(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    memset(b.contents, 0, b.length);
    return b;
}
template <class T> static T *P(id<MTLBuffer> b) { return (T *)b.contents; }

struct Ev { int t, blk, comp; bool fromZero; };
struct Run {
    std::string fx; int mode, N, nblk, steps;
    std::vector<Ev> ev;
    std::vector<std::array<long, 3>> shchg;  // changed SH value codes per degree per step
    std::vector<uint32_t> dtu;
    long qsat = 0, nCodeChanges = 0;
    float maxA = 0, maxAu = 0, maxH = 0, maxLs = 0;
    std::vector<int> inactPerRole;  // base-inactive rows in role blocks
    long inactCells = 0, switchedMax = 0;
    // (a) late growth: growth events at steps where the block has switched-inactive rows; covered = such a row holds a
    // non-zero code of that component entering the step. Groups: 0 blocks 11-13, 1 block 7, 2 Z blocks, 3 other.
    std::array<long, NCOMP> lateGrow{}, lateCov{};
    long lateCovGrp[NCOMP][4] = {};
    // (b) switched-inactive rows with a non-zero code of a component whose block bound did not grow that step
    std::array<long, NCOMP> stableNZ{};
    long mutFlagged = 0, mutCells = 0;
    std::vector<int> mutSteps;
    // attempt 2, non-SH / DC Adam (components 9-18): clamped values (|u| > 8 or log_s > 64) and the largest number of
    // cells with a non-zero code in any step
    std::array<long, NCOMP> sat{}, nzMax{};
};

// Attempt 2: non-SH / DC Adam with the fixed bounds (NS_AU, NS_H), no growth. A changed cell takes its new codes (the
// step's rounding; |u| > NS_AU and log_s > NS_H clamp and count in sat[compU] / sat[compL], log_s also in qsat); an
// unchanged cell keeps its codes.
static void encAdamFixed(Run &R, int t, int blk, int compU, int compL, int dim, const float *mo, const float *vo,
                         const float *mn, const float *vn, uint32_t *codes, bool stoch) {
    const int r0 = blk * QB, r1 = std::min(R.N, (blk + 1) * QB);
    for (int r = r0; r < r1; r++)
        for (int k = 0; k < dim; k++) {
            const size_t c = (size_t)r * dim + k;
            if (same(mn[c], mo[c]) && same(vn[c], vo[c])) continue;
            INV(std::isfinite(mn[c]) && std::isfinite(vn[c]), "%s t=%d nonfinite Adam moment cell %zu", R.fx.c_str(), t, c);
            const float u = adamU(mn[c], vn[c]), l = adamLogS(vn[c]);
            if (std::fabs(u) > NS_AU) R.sat[compU]++;
            if (l > NS_H) { R.sat[compL]++; R.qsat++; }
            codes[c] = PackNS::pack(encS(u, NS_AU, NSA_MU, stoch, t, (uint32_t)c, 0), encU(l, NS_H, NSA_ML, stoch, t, (uint32_t)c, 1));
        }
}
// Non-zero code cells per non-SH / DC component (9-18) of a state, rows < N; keeps the maximum over steps in R.nzMax.
static void nzTrack(Run &R, const QState &S, int N) {
    for (int g = 0; g < NG; g++) {
        long nu = 0, nl = 0;
        for (size_t c = 0; c < (size_t)N * GDIM[g]; c++) { nu += PackNS::u(S.nsq[g][c]) != 0; nl += PackNS::l(S.nsq[g][c]) != 0; }
        R.nzMax[9 + 2 * g] = std::max(R.nzMax[9 + 2 * g], nu);
        R.nzMax[10 + 2 * g] = std::max(R.nzMax[10 + 2 * g], nl);
    }
}

// Encodes one (block, Adam component pair) from the fp32 results. ks: sub-indices within a row (stride `st`).
template <class PK>
static void encAdam(Run &R, int t, int blk, int compU, int compL, const std::vector<int> &ks, int st, const float *mo,
                    const float *vo, const float *mn, const float *vn, typename PK::T *codes, float *bnd, int Mu, int Ml,
                    bool stoch) {
    const int r0 = blk * QB, r1 = std::min(R.N, (blk + 1) * QB);
    float mxu = 0, mxl = 0;
    bool any = false;
    for (int r = r0; r < r1; r++)
        for (int k : ks) {
            size_t c = (size_t)r * st + k;
            if (same(mn[c], mo[c]) && same(vn[c], vo[c])) continue;
            INV(std::isfinite(mn[c]) && std::isfinite(vn[c]), "%s t=%d nonfinite Adam moment cell %zu", R.fx.c_str(), t, c);
            any = true;
            mxu = std::max(mxu, std::fabs(adamU(mn[c], vn[c])));
            mxl = std::max(mxl, adamLogS(vn[c]));
        }
    if (!any) return;
    const float au = bnd[0], h = bnd[1];
    const bool gu = mxu > au, gl = mxl > h;
    float au2 = au, h2 = h;
    if (gu) { au2 = GROW * mxu; R.ev.push_back({t, blk, compU, au == 0.0f}); }
    if (gl) {
        h2 = GROW * mxl;
        if (h2 > H_MAX) { h2 = H_MAX; R.qsat++; }
        R.ev.push_back({t, blk, compL, h == 0.0f});
    }
    R.maxAu = std::max(R.maxAu, au2); R.maxH = std::max(R.maxH, GROW * mxl); R.maxLs = std::max(R.maxLs, mxl);
    for (int r = r0; r < r1; r++)
        for (int k : ks) {
            size_t c = (size_t)r * st + k;
            typename PK::T old = codes[c];
            int cu = PK::u(old), cl = PK::l(old);
            if (!(same(mn[c], mo[c]) && same(vn[c], vo[c]))) {
                cu = encS(adamU(mn[c], vn[c]), au2, Mu, stoch, t, (uint32_t)c, 0);
                cl = encU(adamLogS(vn[c]), h2, Ml, stoch, t, (uint32_t)c, 1);
            } else {
                if (gu) cu = encSym(decSym(cu, au, Mu), au2, Mu);
                if (gl) cl = encUns(decUns(cl, h, Ml), h2, Ml);
            }
            codes[c] = PK::pack(cu, cl);
        }
    bnd[0] = au2; bnd[1] = h2;
}

static Run runFixture(const std::string &fx, int N, int steps, int mode, bool mutant) {
    Run R; R.fx = fx; R.mode = mode; R.N = N; R.steps = steps;
    const int nblk = (N + QB - 1) / QB, rows = nblk * QB;
    R.nblk = nblk;
    const bool stSH = mode == 1, stAdam = mode >= 1;
    const float *cam = F<float>("camera.f32", 42);
    const bool perStep = g_files[fx + "_radii.i32"].size() == (size_t)steps * N * 4;
    const int32_t *radiiAll = F<int32_t>(fx + "_radii.i32", perStep ? (size_t)steps * N : (size_t)N), *radii1 = radiiAll;
    const uint32_t *dtuF = F<uint32_t>(fx + "_degrees.u32", steps);
    const float *grads = F<float>(fx + "_grads.f32", (size_t)steps * N * 10);
    R.dtu.assign(dtuF, dtuF + steps);

    // fp32 params live in GPU buffers (updated in place by the kernels)
    id<MTLBuffer> means = buf(N * 12), scales = buf(N * 12), quats = buf(N * 16), dc = buf(N * 12), opac = buf(N * 4);
    memcpy(means.contents, F<float>(fx + "_means.f32", N * 3), N * 12);
    memcpy(scales.contents, F<float>(fx + "_scales.f32", N * 3), N * 12);
    memcpy(quats.contents, F<float>(fx + "_quats.f32", N * 4), N * 16);
    memcpy(dc.contents, F<float>(fx + "_dc.f32", N * 3), N * 12);
    memcpy(opac.contents, F<float>(fx + "_opac.f32", N), N * 4);
    id<MTLBuffer> radiiB = buf(N * 4), conicsB = buf(N * 12), viewB = buf(64), projB = buf(64);
    memcpy(conicsB.contents, F<float>(fx + "_conics.f32", N * 3), N * 12);
    memcpy(viewB.contents, cam, 64); memcpy(projB.contents, cam + 16, 64);
    id<MTLBuffer> vxy = buf(N * 8), vdep = buf(N * 4), vcon = buf(N * 12), vcol = buf(N * 12), vop = buf(N * 4);
    id<MTLBuffer> vmean = buf(N * 12), vscale = buf(N * 12), vquat = buf(N * 16), nanc = buf(64 * 4);
    id<MTLBuffer> rest = buf((size_t)N * ST * 4), restM = buf((size_t)N * ST * 4), restV = buf((size_t)N * ST * 4);
    id<MTLBuffer> gm[NG], gv[NG];
    for (int g = 0; g < NG; g++) { gm[g] = buf((size_t)N * GDIM[g] * 4); gv[g] = buf((size_t)N * GDIM[g] * 4); }

    // Q state: SH codes/bounds from the fixture; Adam codes and bounds 0
    const int16_t *fr0 = F<int16_t>(fx + "_fr_codes.i16", (size_t)rows * ST);
    const float *frB0 = F<float>(fx + "_fr_bounds.f32", nblk * 3);
    QState S;
    auto &fr = S.fr; auto &frB = S.frB; auto &shq = S.shq; auto &shqB = S.shqB; auto &nsq = S.nsq; auto &nsqB = S.nsqB;
    fr.assign(fr0, fr0 + (size_t)rows * ST);
    frB.assign(frB0, frB0 + nblk * 3);
    shq.assign((size_t)rows * ST, 0);
    shqB.assign((size_t)nblk * 6, 0.0f);
    for (int g = 0; g < NG; g++) {
        nsq[g].assign((size_t)rows * GDIM[g], 0);
        nsqB[g].assign((size_t)nblk * 2, 0.0f);
        for (int b = 0; b < nblk; b++) { nsqB[g][b * 2] = NS_AU; nsqB[g][b * 2 + 1] = NS_H; }  // attempt 2: fixed bounds
    }
    // N-block snapshot (codes and bounds must never change)
    auto nSnap = [&]() {
        std::vector<char> s;
        auto add = [&](const void *p, size_t b) { s.insert(s.end(), (const char *)p, (const char *)p + b); };
        for (int b = 100; b <= 109 && b < nblk; b++) {
            add(&fr[(size_t)b * QB * ST], QB * ST * 2); add(&frB[b * 3], 12);
            add(&shq[(size_t)b * QB * ST], QB * ST * 2); add(&shqB[b * 6], 24);
            for (int g = 0; g < NG; g++) { add(&nsq[g][(size_t)b * QB * GDIM[g]], QB * GDIM[g] * 4); add(&nsqB[g][b * 2], 8); }
        }
        return s;
    };
    const std::vector<char> n0 = nSnap();
    const int roles[] = {0, 1, 2, 3, 9, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    for (int b : roles) {
        int in = 0;
        for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++) in += radii1[r] == 0;
        R.inactPerRole.push_back(in);
    }

    std::vector<float> restOld((size_t)N * ST), restMO((size_t)N * ST), restVO((size_t)N * ST);
    std::vector<float> gmO[NG], gvO[NG];
    std::vector<int> ksDeg[3], ksAll[NG];
    for (int j = 0; j < ST; j++) ksDeg[degOf(j)].push_back(j);
    for (int g = 0; g < NG; g++) for (int k = 0; k < GDIM[g]; k++) ksAll[g].push_back(k);

    for (int t = 1; t <= steps; t++) {
        const uint32_t dtu = dtuF[t - 1], degree = 3;
        const int32_t *radii = perStep ? radiiAll + (size_t)(t - 1) * N : radiiAll;
        memcpy(radiiB.contents, radii, N * 4);
        // 1-2. decode the Q state into the fp32 buffers
        for (int r = 0; r < N; r++) {
            const int b = r / QB;
            for (int j = 0; j < ST; j++) {
                size_t c = (size_t)r * ST + j;
                int d = degOf(j);
                restOld[c] = decSym(fr[c], frB[b * 3 + d], SH_M);
                adamDecode(decSym(PackSH::u(shq[c]), shqB[b * 6 + d * 2], SHA_MU),
                           decUns(PackSH::l(shq[c]), shqB[b * 6 + d * 2 + 1], SHA_ML), restMO[c], restVO[c]);
            }
        }
        memcpy(rest.contents, restOld.data(), restOld.size() * 4);
        memcpy(restM.contents, restMO.data(), restMO.size() * 4);
        memcpy(restV.contents, restVO.data(), restVO.size() * 4);
        for (int g = 0; g < NG; g++) {
            gmO[g].resize((size_t)N * GDIM[g]); gvO[g].resize((size_t)N * GDIM[g]);
            for (size_t c = 0; c < gmO[g].size(); c++) {
                const int b = (int)(c / GDIM[g]) / QB;
                adamDecode(decSym(PackNS::u(nsq[g][c]), nsqB[g][b * 2], NSA_MU), decUns(PackNS::l(nsq[g][c]), nsqB[g][b * 2 + 1], NSA_ML),
                           gmO[g][c], gvO[g][c]);
            }
            memcpy(gm[g].contents, gmO[g].data(), gmO[g].size() * 4);
            memcpy(gv[g].contents, gvO[g].data(), gvO[g].size() * 4);
        }
        // gradient inputs of step t; projection-gradient outputs zeroed before the dispatch
        const float *gt = grads + (size_t)(t - 1) * N * 10;
        memcpy(vxy.contents, gt, N * 8); memcpy(vdep.contents, gt + 2 * (size_t)N, N * 4);
        memcpy(vcon.contents, gt + 3 * (size_t)N, N * 12); memcpy(vcol.contents, gt + 6 * (size_t)N, N * 12);
        memcpy(vop.contents, gt + 9 * (size_t)N, N * 4);
        memset(vmean.contents, 0, N * 12); memset(vscale.contents, 0, N * 12); memset(vquat.contents, 0, N * 16);
        memset(nanc.contents, 0, nanc.length);

        // 3. unchanged fp32 kernels, bindings as msplat_metal.mm encode_proj_sh_bwd_adam
        float bc1 = 1.0f - std::pow(0.9f, t), bc2 = 1.0f - std::pow(0.999f, t);
        struct { float dc_ss, dc_bc2, rest_ss, rest_bc2, b1, b2, eps; } hp = {
            LR[3] / bc1, std::sqrt(bc2), LR[4] / bc1, std::sqrt(bc2), 0.9f, 0.999f, 1e-8f};
        int np = N; float glob = 1.0f;
        float intr[4] = {cam[32], cam[33], cam[34], cam[35]}, campos[4] = {cam[36], cam[37], cam[38], 0.0f};
        uint32_t isz[2] = {(uint32_t)cam[40], (uint32_t)cam[41]};
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psoBwd];
        [e setBytes:&np length:4 atIndex:0]; [e setBuffer:means offset:0 atIndex:1]; [e setBuffer:scales offset:0 atIndex:2];
        [e setBytes:&glob length:4 atIndex:3]; [e setBuffer:quats offset:0 atIndex:4];
        [e setBuffer:viewB offset:0 atIndex:5]; [e setBuffer:projB offset:0 atIndex:6];
        [e setBytes:intr length:16 atIndex:7]; [e setBytes:isz length:8 atIndex:8];
        [e setBuffer:radiiB offset:0 atIndex:9]; [e setBuffer:conicsB offset:0 atIndex:10];
        [e setBuffer:vxy offset:0 atIndex:11]; [e setBuffer:vdep offset:0 atIndex:12]; [e setBuffer:vcon offset:0 atIndex:13];
        [e setBuffer:vmean offset:0 atIndex:14]; [e setBuffer:vscale offset:0 atIndex:15]; [e setBuffer:vquat offset:0 atIndex:16];
        [e setBytes:&degree length:4 atIndex:17]; [e setBytes:&dtu length:4 atIndex:18];
        [e setBytes:campos length:16 atIndex:19]; [e setBuffer:vcol offset:0 atIndex:20];
        [e setBuffer:dc offset:0 atIndex:21]; [e setBuffer:rest offset:0 atIndex:22];
        [e setBuffer:gm[3] offset:0 atIndex:23]; [e setBuffer:gv[3] offset:0 atIndex:24];
        [e setBuffer:restM offset:0 atIndex:25]; [e setBuffer:restV offset:0 atIndex:26];
        [e setBytes:&hp length:sizeof hp atIndex:27]; [e setBuffer:nanc offset:0 atIndex:28];
        [e dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(psoBwd.maxTotalThreadsPerThreadgroup, N), 1, 1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        id<MTLBuffer> params[NG] = {means, scales, quats, dc, opac}, gr[NG] = {vmean, vscale, vquat, nil, vop};
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, bc2s = std::sqrt(bc2);
        for (int g : {0, 1, 2, 4}) {
            uint32_t n = (uint32_t)(N * GDIM[g]), gi = (uint32_t)GADAM[g];
            float ss = LR[GADAM[g]] / bc1;
            [e setComputePipelineState:psoAdam];
            [e setBuffer:params[g] offset:0 atIndex:0]; [e setBuffer:gr[g] offset:0 atIndex:1];
            [e setBuffer:gm[g] offset:0 atIndex:2]; [e setBuffer:gv[g] offset:0 atIndex:3];
            [e setBytes:&ss length:4 atIndex:4]; [e setBytes:&b1 length:4 atIndex:5]; [e setBytes:&b2 length:4 atIndex:6];
            [e setBytes:&bc2s length:4 atIndex:7]; [e setBytes:&eps length:4 atIndex:8]; [e setBytes:&n length:4 atIndex:9];
            [e setBuffer:nanc offset:0 atIndex:10]; [e setBytes:&gi length:4 atIndex:11];
            [e dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(psoAdam.maxTotalThreadsPerThreadgroup, n), 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        INV(cb.status == MTLCommandBufferStatusCompleted, "%s t=%d command buffer status %ld", fx.c_str(), t, (long)cb.status);
        for (int i = 0; i < 64; i++) INV(P<uint32_t>(nanc)[i] == 0, "%s t=%d nan_counts[%d] = %u", fx.c_str(), t, i, P<uint32_t>(nanc)[i]);

        // outputs finite; projection grads of inactive rows stay 0
        for (int g = 0; g < NG; g++) {
            const float *p = P<float>(params[g]);
            for (int i = 0; i < N * GDIM[g]; i++) INV(std::isfinite(p[i]), "%s t=%d nonfinite param g%d[%d]", fx.c_str(), t, g, i);
        }
        long inactNZ = 0;
        for (int r = 0; r < N; r++) {
            if (radii[r]) continue;
            for (int k = 0; k < 3; k++) inactNZ += P<float>(vmean)[r * 3 + k] != 0.0f || P<float>(vscale)[r * 3 + k] != 0.0f;
            for (int k = 0; k < 4; k++) inactNZ += P<float>(vquat)[r * 4 + k] != 0.0f;
        }
        INV(inactNZ == 0, "%s t=%d %ld nonzero projection grads on inactive rows", fx.c_str(), t, inactNZ);

        // 4. CPU Q encoding of the fp32 results
        const float *restN = P<float>(rest), *restMN = P<float>(restM), *restVN = P<float>(restV);
        const QState prev = S;
        const auto &frPrev = prev.fr; const auto &shqPrev = prev.shq; const auto &frBPrev = prev.frB; const auto &shqBPrev = prev.shqB;
        const size_t nev0 = R.ev.size();
        for (int b = 0; b < nblk; b++) {
            const int r0 = b * QB, r1 = std::min(N, (b + 1) * QB);
            for (int d = 0; d < 3; d++) {  // SH value
                float mx = 0; bool any = false;
                for (int r = r0; r < r1; r++)
                    for (int j : ksDeg[d]) {
                        size_t c = (size_t)r * ST + j;
                        if (same(restN[c], restOld[c])) continue;
                        INV(std::isfinite(restN[c]), "%s t=%d nonfinite SH cell %zu", fx.c_str(), t, c);
                        any = true; mx = std::max(mx, std::fabs(restN[c]));
                    }
                if (!any) continue;
                const float a = frB[b * 3 + d];
                const bool grow = mx > a;
                const float a2 = grow ? shBound(mx) : a;  // attempt 2: max(1.25 max, SH_A_MIN)
                if (grow) R.ev.push_back({t, b, d, a == 0.0f});
                R.maxA = std::max(R.maxA, a2);
                for (int r = r0; r < r1; r++)
                    for (int j : ksDeg[d]) {
                        size_t c = (size_t)r * ST + j;
                        int old = fr[c], code = old;
                        if (!same(restN[c], restOld[c])) code = encS(restN[c], a2, SH_M, stSH, t, (uint32_t)c, 2);
                        else if (grow) code = encSym(restOld[c], a2, SH_M);
                        fr[c] = (int16_t)code;
                    }
                frB[b * 3 + d] = a2;
            }
            for (int d = 0; d < 3; d++)  // SH Adam
                encAdam<PackSH>(R, t, b, 3 + d, 6 + d, ksDeg[d], ST, restMO.data(), restVO.data(), restMN, restVN, shq.data(),
                                &shqB[b * 6 + d * 2], SHA_MU, SHA_ML, stAdam);
            for (int g = 0; g < NG; g++)  // non-SH and DC Adam: fixed bounds (attempt 2)
                encAdamFixed(R, t, b, 9 + 2 * g, 10 + 2 * g, GDIM[g], gmO[g].data(), gvO[g].data(), P<float>(gm[g]),
                             P<float>(gv[g]), nsq[g].data(), stAdam);
        }
        nzTrack(R, S, N);
        // changed SH codes per degree; unopened degrees byte-same
        std::array<long, 3> chg = {0, 0, 0};
        long unopened = 0;
        for (size_t c = 0; c < (size_t)N * ST; c++) {
            int d = degOf((int)(c % ST));
            if (fr[c] != frPrev[c]) chg[d]++;
            if ((uint32_t)d >= dtu) unopened += fr[c] != frPrev[c] || shq[c] != shqPrev[c];
        }
        for (int b = 0; b < nblk; b++)
            for (int d = (int)dtu; d < 3; d++)
                unopened += !same(frB[b * 3 + d], frBPrev[b * 3 + d]) || !same(shqB[b * 6 + d * 2], shqBPrev[b * 6 + d * 2]) ||
                            !same(shqB[b * 6 + d * 2 + 1], shqBPrev[b * 6 + d * 2 + 1]);
        INV(unopened == 0, "%s t=%d unopened degrees changed (%ld cells/bounds)", fx.c_str(), t, unopened);
        R.shchg.push_back(chg);
        if (nSnap() != n0) R.nCodeChanges++;

        // s4.2 inactive-row rule on the reference itself
        InactCmp ic = compareInactiveRows(prev, S, radii, N);
        R.inactCells += ic.cells;
        INV(ic.flagged == 0, "%s t=%d %ld inactive-row cells break the s4.2 rule", fx.c_str(), t, ic.flagged);

        // coverage (a)/(b): switched-inactive rows (inactive now, active at t = 1) holding non-zero codes
        auto nz = [&](int r, int comp) {
            if (comp <= 8) {
                for (int j : ksDeg[comp % 3]) {
                    const size_t c = (size_t)r * ST + j;
                    if (comp <= 2 ? prev.fr[c] != 0 : comp <= 5 ? PackSH::u(prev.shq[c]) != 0 : PackSH::l(prev.shq[c]) != 0) return true;
                }
                return false;
            }
            for (int k = 0; k < 3; k++) {  // 15 dc_u, 16 dc_ls
                const uint32_t q = prev.nsq[3][(size_t)r * 3 + k];
                if (comp == 15 ? PackNS::u(q) != 0 : PackNS::l(q) != 0) return true;
            }
            return false;
        };
        std::vector<char> grew((size_t)nblk * NCOMP, 0);
        for (size_t i = nev0; i < R.ev.size(); i++) grew[(size_t)R.ev[i].blk * NCOMP + R.ev[i].comp] = 1;
        const int covComps[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16};
        long nsw = 0;
        for (int b = 0; b < nblk; b++) {
            std::vector<int> sw;
            for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++) if (radii[r] == 0 && radii1[r] != 0) sw.push_back(r);
            nsw += (long)sw.size();
            if (sw.empty()) continue;
            const int grp = (b >= 11 && b <= 13) ? 0 : b == 7 ? 1 : isZ(b) ? 2 : 3;
            for (int comp : covComps) {
                if (grew[(size_t)b * NCOMP + comp]) {
                    R.lateGrow[comp]++;
                    bool cov = false;
                    for (int r : sw) if (nz(r, comp)) { cov = true; break; }
                    if (cov) { R.lateCov[comp]++; R.lateCovGrp[comp][grp]++; }
                } else if (comp >= 3) {
                    for (int r : sw) R.stableNZ[comp] += nz(r, comp);
                }
            }
        }
        R.switchedMax = std::max(R.switchedMax, nsw);

        // mutant: the reference outputs with every inactive row's SH Adam and DC Adam codes zeroed (bounds kept)
        if (mutant) {
            QState cand = S;
            for (int r = 0; r < N; r++) {
                if (radii[r] != 0) continue;
                for (int j = 0; j < ST; j++) cand.shq[(size_t)r * ST + j] = 0;
                for (int k = 0; k < 3; k++) cand.nsq[3][(size_t)r * 3 + k] = 0;
            }
            InactCmp mc = compareInactiveRows(prev, cand, radii, N);
            R.mutCells += mc.cells; R.mutFlagged += mc.flagged;
            if (mc.flagged) R.mutSteps.push_back(t);
        }
    }
    // F2 padding rows stay code 0
    long pad = 0;
    for (size_t c = (size_t)N * ST; c < fr.size(); c++) pad += fr[c] != 0 || shq[c] != 0;
    for (int g = 0; g < NG; g++) for (size_t c = (size_t)N * GDIM[g]; c < nsq[g].size(); c++) pad += nsq[g][c] != 0;
    INV(pad == 0, "%s padding rows hold %ld nonzero codes", fx.c_str(), pad);
    if (rows > N) printf("%s %s: padding rows %d, nonzero codes %ld\n", fx.c_str(), MODE_NAME[mode], rows - N, pad);
    return R;
}

static int countBlocks(const Run &R, int comp, int t, int fromZero, bool (*sel)(int)) {
    std::vector<char> seen(R.nblk, 0);
    int n = 0;
    for (auto &e : R.ev)
        if (e.comp == comp && (t < 0 || e.t == t) && (fromZero < 0 || e.fromZero == (bool)fromZero) && (!sel || sel(e.blk)) && !seen[e.blk]) {
            seen[e.blk] = 1; n++;
        }
    return n;
}
static int firstStep(const Run &R, int comp, int blk, int fromZero) {
    int s = 1 << 30;
    for (auto &e : R.ev) if (e.comp == comp && e.blk == blk && e.fromZero == (bool)fromZero) s = std::min(s, e.t);
    return s == (1 << 30) ? -1 : s;
}

// =====================================================================================================================
// P2-M2: --mode q (s4.2) and --mode parity (s3). Q kernels: core/metal/msplat_q.metal, linked into the same metallib.
// --mode q, per fixture (F1 70 steps, F2 10 steps) and rounding mode: the Q arm runs project_and_sh_backward_q +
//   fused_adam_q16 on its own Q state. Before every step the reference arm takes the Q arm's state (codes decoded on the
//   CPU, fp32 params copied), runs the unchanged fp32 kernels and encodes their results on the CPU (one-step consistency,
//   no accumulation). Both arms get the same gradient inputs; the Q non-SH Adam takes the reference arm's projection
//   gradients (s4.1 "ortak girdi"). After each step every s4.2 threshold is checked; the Q arm continues from its own
//   state. Forward equivalence on F1 at t = 0 and after t = 70, degrees_to_use 0-3. Junk set after every step.
// --mode parity: the codec parity kernels of msplat_q.metal on the s3 sample files. Stochastic keys: step = 1,
//   cell = element index (comp 2 SH value, 0 u, 1 log_s), on both sides.
// =====================================================================================================================
static id<MTLComputePipelineState> psoBwdQ, psoAdamQ, psoFwd, psoFwdQ, psoRegQ, psoParSh, psoParAe, psoParAd, psoParLE;
static int qSectFail = 0, qSectN = 0;
// Q_DEBUG_CELLS=<path> (attempt-2 diagnosis): cell dump side output of --mode q; no verdict depends on it.
static FILE *g_dbg = nullptr;
static int g_dbgLs = 0, g_dbgLsRepro = 0, g_dbgLsUlp = 0, g_dbgLsBoth = 0, g_dbgLsDblQ = 0;
template <class... A> static void sect(bool ok, const char *fmt, A... a) {
    printf("  [%s] ", ok ? "PASS" : "FAIL");
    printf(fmt, a...);
    printf("\n");
    qSectFail += !ok; qSectN++;
}
static float fz(float x) { return std::fabs(x) < FLT_MIN ? 0.0f : x; }  // s0: |x| < FLT_MIN counts as 0 on the CPU side

// Bound of component c in block b (component ids as CN).
static float qBound(const QState &S, int b, int c) {
    if (c <= 2) return S.frB[b * 3 + c];
    if (c <= 8) return S.shqB[b * 6 + ((c - 3) % 3) * 2 + (c >= 6)];
    return S.nsqB[(c - 9) / 2][b * 2 + (c - 9) % 2];
}

struct CodeStat { long den = 0, diff = 0, steps = 0; int maxAbs = 0; double worstRate = 0; int worstT = 0; bool fail = false; };
struct FwdStat { std::string when; int dtu; long intMis = 0, sentMis = 0, floatViol = 0, refNonfinite = 0; double floatMax = 0; };
struct QRes {
    std::string fx; int mode = 0, N = 0, steps = 0;
    Run RQ;  // Q-arm events and coverage bookkeeping (fields as in the reference run)
    CodeStat code[NCOMP];
    long boundViol = 0, boundNonfinite = 0; double boundMaxRel = 0;
    long evMis = 0, evQ = 0, evRef = 0;
    double paramMax = 0; long paramViol = 0, paramNonfinite = 0;
    double gradMax = 0; long gradViol = 0, gradInactNZ = 0, gradNonfinite = 0;
    long unopened = 0, inactCells = 0, inactFlagged = 0, refInactFlagged = 0, padNZ = 0;
    bool completed = true;
    uint32_t qsat = 0, usat = 0; long refQsat = 0, refUsat = 0, nanQ = 0, nanRef = 0, junkMis = 0, junkRows = 0;
    std::vector<FwdStat> fwd;
};

static QRes runQ(const std::string &fx, int N, int steps, int mode) {
    QRes Q; Q.fx = fx; Q.mode = mode; Q.N = N; Q.steps = steps;
    Run &RQ = Q.RQ; RQ.fx = fx; RQ.mode = mode; RQ.N = N; RQ.steps = steps;
    Run RR = RQ;  // reference-arm events (encAdam bookkeeping)
    const int nblk = (N + QB - 1) / QB, rows = nblk * QB;
    RQ.nblk = RR.nblk = nblk;
    const bool stSH = mode == 1, stAdam = mode >= 1;
    const float *cam = F<float>("camera.f32", 42);
    const bool perStep = g_files[fx + "_radii.i32"].size() == (size_t)steps * N * 4;
    const int32_t *radiiAll = F<int32_t>(fx + "_radii.i32", perStep ? (size_t)steps * N : (size_t)N), *radii1 = radiiAll;
    const uint32_t *dtuF = F<uint32_t>(fx + "_degrees.u32", steps);
    const float *grads = F<float>(fx + "_grads.f32", (size_t)steps * N * 10);
    RQ.dtu.assign(dtuF, dtuF + steps);

    // Q-arm fp32 params and reference-arm copies
    id<MTLBuffer> qp[NG], rp[NG];
    const char *pn[NG] = {"_means.f32", "_scales.f32", "_quats.f32", "_dc.f32", "_opac.f32"};
    for (int g = 0; g < NG; g++) {
        qp[g] = buf((size_t)N * GDIM[g] * 4); rp[g] = buf((size_t)N * GDIM[g] * 4);
        memcpy(qp[g].contents, F<float>(fx + pn[g], (size_t)N * GDIM[g]), (size_t)N * GDIM[g] * 4);
    }
    id<MTLBuffer> radiiB = buf(N * 4), conicsB = buf(N * 12), viewB = buf(64), projB = buf(64);
    memcpy(conicsB.contents, F<float>(fx + "_conics.f32", N * 3), N * 12);
    memcpy(viewB.contents, cam, 64); memcpy(projB.contents, cam + 16, 64);
    id<MTLBuffer> vxy = buf(N * 8), vdep = buf(N * 4), vcon = buf(N * 12), vcol = buf(N * 12), vop = buf(N * 4);
    id<MTLBuffer> vmean = buf(N * 12), vscale = buf(N * 12), vquat = buf(N * 16), nanc = buf(64 * 4);
    id<MTLBuffer> vmeanQ = buf(N * 12), vscaleQ = buf(N * 12), vquatQ = buf(N * 16), nancQ = buf(64 * 4), qsatB = buf(16);
    id<MTLBuffer> rest = buf((size_t)N * ST * 4), restM = buf((size_t)N * ST * 4), restV = buf((size_t)N * ST * 4);
    id<MTLBuffer> gm[NG], gv[NG];
    for (int g = 0; g < NG; g++) { gm[g] = buf((size_t)N * GDIM[g] * 4); gv[g] = buf((size_t)N * GDIM[g] * 4); }
    id<MTLBuffer> zeroOp = buf(N * 4), vopJ = buf(N * 4), vscJ = buf(N * 12);

    // Q-arm state on the GPU: SH codes/bounds from the fixture, Adam codes and bounds 0
    id<MTLBuffer> frQ = buf((size_t)rows * ST * 2), frBQ = buf((size_t)nblk * 12), shqQ = buf((size_t)rows * ST * 2),
                  shqBQ = buf((size_t)nblk * 24), nsqQ[NG], nsqBQ[NG];
    memcpy(frQ.contents, F<int16_t>(fx + "_fr_codes.i16", (size_t)rows * ST), (size_t)rows * ST * 2);
    memcpy(frBQ.contents, F<float>(fx + "_fr_bounds.f32", nblk * 3), (size_t)nblk * 12);
    for (int g = 0; g < NG; g++) {
        nsqQ[g] = buf((size_t)rows * GDIM[g] * 4); nsqBQ[g] = buf((size_t)nblk * 8);
        for (int b = 0; b < nblk; b++) { P<float>(nsqBQ[g])[b * 2] = NS_AU; P<float>(nsqBQ[g])[b * 2 + 1] = NS_H; }  // attempt 2
    }
    auto readQ = [&]() {
        QState S;
        S.fr.assign(P<int16_t>(frQ), P<int16_t>(frQ) + (size_t)rows * ST);
        S.frB.assign(P<float>(frBQ), P<float>(frBQ) + nblk * 3);
        S.shq.assign(P<uint16_t>(shqQ), P<uint16_t>(shqQ) + (size_t)rows * ST);
        S.shqB.assign(P<float>(shqBQ), P<float>(shqBQ) + nblk * 6);
        for (int g = 0; g < NG; g++) {
            S.nsq[g].assign(P<uint32_t>(nsqQ[g]), P<uint32_t>(nsqQ[g]) + (size_t)rows * GDIM[g]);
            S.nsqB[g].assign(P<float>(nsqBQ[g]), P<float>(nsqBQ[g]) + nblk * 2);
        }
        return S;
    };
    auto nSnap = [&](const QState &S) {
        std::vector<char> s;
        auto add = [&](const void *p, size_t b) { s.insert(s.end(), (const char *)p, (const char *)p + b); };
        for (int b = 100; b <= 109 && b < nblk; b++) {
            add(&S.fr[(size_t)b * QB * ST], QB * ST * 2); add(&S.frB[b * 3], 12);
            add(&S.shq[(size_t)b * QB * ST], QB * ST * 2); add(&S.shqB[b * 6], 24);
            for (int g = 0; g < NG; g++) { add(&S.nsq[g][(size_t)b * QB * GDIM[g]], QB * GDIM[g] * 4); add(&S.nsqB[g][b * 2], 8); }
        }
        return s;
    };
    const std::vector<char> n0 = nSnap(readQ());
    const int roles[] = {0, 1, 2, 3, 9, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    for (int b : roles) {
        int in = 0;
        for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++) in += radii1[r] == 0;
        RQ.inactPerRole.push_back(in);
    }
    std::vector<int> ksDeg[3], ksAll[NG];
    for (int j = 0; j < ST; j++) ksDeg[degOf(j)].push_back(j);
    for (int g = 0; g < NG; g++) for (int k = 0; k < GDIM[g]; k++) ksAll[g].push_back(k);
    float intr[4] = {cam[32], cam[33], cam[34], cam[35]}, campos[4] = {cam[36], cam[37], cam[38], 0.0f};
    uint32_t isz[2] = {(uint32_t)cam[40], (uint32_t)cam[41]};

    // ---- forward equivalence (s4.2 "Ileri gecis"): F1 at t = 0 and after t = 70, degrees_to_use 0-3 ----
    auto fwdCheck = [&](const char *when) {
        const QState S = readQ();
        std::vector<float> restF((size_t)N * ST);
        for (int r = 0; r < N; r++)
            for (int j = 0; j < ST; j++) restF[(size_t)r * ST + j] = decSym(S.fr[(size_t)r * ST + j], S.frB[(r / QB) * 3 + degOf(j)], SH_M);
        id<MTLBuffer> restB = buf((size_t)N * ST * 4);
        memcpy(restB.contents, restF.data(), restF.size() * 4);
        const float SENT = 12345.0f;
        const size_t fl[5] = {(size_t)N * 2, (size_t)N, (size_t)N * 3, (size_t)N * 3, (size_t)N * 2};  // xys depths conics colors aabb
        for (uint32_t dtu = 0; dtu <= 3; dtu++) {
            id<MTLBuffer> fo[2][5], io[2][2], nc[2];
            for (int a = 0; a < 2; a++) {
                for (int k = 0; k < 5; k++) { fo[a][k] = buf(fl[k] * 4); for (size_t i = 0; i < fl[k]; i++) P<float>(fo[a][k])[i] = SENT; }
                for (int k = 0; k < 2; k++) { io[a][k] = buf((size_t)N * 4); for (int i = 0; i < N; i++) P<int32_t>(io[a][k])[i] = -7; }
                nc[a] = buf(64 * 4);
            }
            int np = N; float glob = 1.0f, clip = 0.01f; uint32_t degree = 3;
            uint32_t tb[4] = {(isz[0] + 15) / 16, (isz[1] + 15) / 16, 1, 0xDEAD};
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (int a = 0; a < 2; a++) {
                [e setComputePipelineState:a ? psoFwdQ : psoFwd];
                [e setBytes:&np length:4 atIndex:0]; [e setBuffer:qp[0] offset:0 atIndex:1]; [e setBuffer:qp[1] offset:0 atIndex:2];
                [e setBytes:&glob length:4 atIndex:3]; [e setBuffer:qp[2] offset:0 atIndex:4];
                [e setBuffer:viewB offset:0 atIndex:5]; [e setBuffer:projB offset:0 atIndex:6];
                [e setBytes:intr length:16 atIndex:7]; [e setBytes:isz length:8 atIndex:8]; [e setBytes:tb length:16 atIndex:9];
                [e setBytes:&clip length:4 atIndex:10];
                [e setBuffer:fo[a][0] offset:0 atIndex:11]; [e setBuffer:fo[a][1] offset:0 atIndex:12];
                [e setBuffer:io[a][0] offset:0 atIndex:13]; [e setBuffer:fo[a][2] offset:0 atIndex:14];
                [e setBuffer:io[a][1] offset:0 atIndex:15];
                [e setBytes:&degree length:4 atIndex:16]; [e setBytes:&dtu length:4 atIndex:17]; [e setBytes:campos length:16 atIndex:18];
                [e setBuffer:qp[3] offset:0 atIndex:19]; [e setBuffer:(a ? frQ : restB) offset:0 atIndex:20];
                [e setBuffer:fo[a][3] offset:0 atIndex:21]; [e setBuffer:fo[a][4] offset:0 atIndex:22];
                [e setBuffer:nc[a] offset:0 atIndex:23];
                if (a) [e setBuffer:frBQ offset:0 atIndex:24];
                id<MTLComputePipelineState> p = a ? psoFwdQ : psoFwd;
                [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, N), 1, 1)];
            }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            FwdStat fs; fs.when = when; fs.dtu = (int)dtu;
            if (cb.status != MTLCommandBufferStatusCompleted) Q.completed = false;
            for (int k = 0; k < 2; k++)
                for (int i = 0; i < N; i++) fs.intMis += P<int32_t>(io[0][k])[i] != P<int32_t>(io[1][k])[i];
            for (int k = 0; k < 5; k++) {
                const float *r = P<float>(fo[0][k]), *q = P<float>(fo[1][k]);
                for (size_t i = 0; i < fl[k]; i++) {
                    if (!same(r[i], SENT) && !std::isfinite(r[i])) fs.refNonfinite++;
                    if (same(r[i], q[i])) continue;
                    if (same(r[i], SENT) || same(q[i], SENT)) { fs.sentMis++; continue; }
                    const double d = std::fabs((double)q[i] - r[i]) / std::max(1.0, std::fabs((double)r[i]));
                    if (!(d <= 1e-6)) fs.floatViol++;
                    if (d > fs.floatMax || d != d) fs.floatMax = d != d ? INFINITY : d;
                }
            }
            Q.fwd.push_back(fs);
        }
    };
    if (fx == "f1") fwdCheck("t=0");

    for (int t = 1; t <= steps; t++) {
        const uint32_t dtu = dtuF[t - 1], degree = 3;
        const int32_t *radii = perStep ? radiiAll + (size_t)(t - 1) * N : radiiAll;
        memcpy(radiiB.contents, radii, N * 4);
        const QState prev = readQ();
        // reference arm input: the Q arm's state entering step t
        for (int g = 0; g < NG; g++) memcpy(rp[g].contents, qp[g].contents, (size_t)N * GDIM[g] * 4);
        std::vector<float> restOld((size_t)N * ST), restMO((size_t)N * ST), restVO((size_t)N * ST), gmO[NG], gvO[NG];
        for (int r = 0; r < N; r++) {
            const int b = r / QB;
            for (int j = 0; j < ST; j++) {
                size_t c = (size_t)r * ST + j;
                int d = degOf(j);
                restOld[c] = decSym(prev.fr[c], prev.frB[b * 3 + d], SH_M);
                adamDecode(decSym(PackSH::u(prev.shq[c]), prev.shqB[b * 6 + d * 2], SHA_MU),
                           decUns(PackSH::l(prev.shq[c]), prev.shqB[b * 6 + d * 2 + 1], SHA_ML), restMO[c], restVO[c]);
            }
        }
        memcpy(rest.contents, restOld.data(), restOld.size() * 4);
        memcpy(restM.contents, restMO.data(), restMO.size() * 4);
        memcpy(restV.contents, restVO.data(), restVO.size() * 4);
        for (int g = 0; g < NG; g++) {
            gmO[g].resize((size_t)N * GDIM[g]); gvO[g].resize((size_t)N * GDIM[g]);
            for (size_t c = 0; c < gmO[g].size(); c++) {
                const int b = (int)(c / GDIM[g]) / QB;
                adamDecode(decSym(PackNS::u(prev.nsq[g][c]), prev.nsqB[g][b * 2], NSA_MU),
                           decUns(PackNS::l(prev.nsq[g][c]), prev.nsqB[g][b * 2 + 1], NSA_ML), gmO[g][c], gvO[g][c]);
            }
            memcpy(gm[g].contents, gmO[g].data(), gmO[g].size() * 4);
            memcpy(gv[g].contents, gvO[g].data(), gvO[g].size() * 4);
        }
        const float *gt = grads + (size_t)(t - 1) * N * 10;
        memcpy(vxy.contents, gt, N * 8); memcpy(vdep.contents, gt + 2 * (size_t)N, N * 4);
        memcpy(vcon.contents, gt + 3 * (size_t)N, N * 12); memcpy(vcol.contents, gt + 6 * (size_t)N, N * 12);
        memcpy(vop.contents, gt + 9 * (size_t)N, N * 4);
        for (id<MTLBuffer> b : {vmean, vscale, vquat, vmeanQ, vscaleQ, vquatQ, nanc, nancQ, vopJ}) memset(b.contents, 0, b.length);

        float bc1 = 1.0f - std::pow(0.9f, t), bc2 = 1.0f - std::pow(0.999f, t);
        struct { float dc_ss, dc_bc2, rest_ss, rest_bc2, b1, b2, eps; } hp = {
            LR[3] / bc1, std::sqrt(bc2), LR[4] / bc1, std::sqrt(bc2), 0.9f, 0.999f, 1e-8f};
        struct { float dc_ss, dc_bc2, rest_ss, rest_bc2, b1, b2, eps; uint32_t step, qround; } hpQ = {
            LR[3] / bc1, std::sqrt(bc2), LR[4] / bc1, std::sqrt(bc2), 0.9f, 0.999f, 1e-8f, (uint32_t)t, (uint32_t)mode};
        int np = N; float glob = 1.0f;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, bc2s = std::sqrt(bc2);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        // reference arm: unchanged fp32 kernels, bindings as --mode reference
        [e setComputePipelineState:psoBwd];
        [e setBytes:&np length:4 atIndex:0]; [e setBuffer:rp[0] offset:0 atIndex:1]; [e setBuffer:rp[1] offset:0 atIndex:2];
        [e setBytes:&glob length:4 atIndex:3]; [e setBuffer:rp[2] offset:0 atIndex:4];
        [e setBuffer:viewB offset:0 atIndex:5]; [e setBuffer:projB offset:0 atIndex:6];
        [e setBytes:intr length:16 atIndex:7]; [e setBytes:isz length:8 atIndex:8];
        [e setBuffer:radiiB offset:0 atIndex:9]; [e setBuffer:conicsB offset:0 atIndex:10];
        [e setBuffer:vxy offset:0 atIndex:11]; [e setBuffer:vdep offset:0 atIndex:12]; [e setBuffer:vcon offset:0 atIndex:13];
        [e setBuffer:vmean offset:0 atIndex:14]; [e setBuffer:vscale offset:0 atIndex:15]; [e setBuffer:vquat offset:0 atIndex:16];
        [e setBytes:&degree length:4 atIndex:17]; [e setBytes:&dtu length:4 atIndex:18];
        [e setBytes:campos length:16 atIndex:19]; [e setBuffer:vcol offset:0 atIndex:20];
        [e setBuffer:rp[3] offset:0 atIndex:21]; [e setBuffer:rest offset:0 atIndex:22];
        [e setBuffer:gm[3] offset:0 atIndex:23]; [e setBuffer:gv[3] offset:0 atIndex:24];
        [e setBuffer:restM offset:0 atIndex:25]; [e setBuffer:restV offset:0 atIndex:26];
        [e setBytes:&hp length:sizeof hp atIndex:27]; [e setBuffer:nanc offset:0 atIndex:28];
        [e dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(psoBwd.maxTotalThreadsPerThreadgroup, N), 1, 1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        id<MTLBuffer> gr[NG] = {vmean, vscale, vquat, nil, vop};
        for (int g : {0, 1, 2, 4}) {
            uint32_t n = (uint32_t)(N * GDIM[g]), gi = (uint32_t)GADAM[g];
            float ss = LR[GADAM[g]] / bc1;
            [e setComputePipelineState:psoAdam];
            [e setBuffer:rp[g] offset:0 atIndex:0]; [e setBuffer:gr[g] offset:0 atIndex:1];
            [e setBuffer:gm[g] offset:0 atIndex:2]; [e setBuffer:gv[g] offset:0 atIndex:3];
            [e setBytes:&ss length:4 atIndex:4]; [e setBytes:&b1 length:4 atIndex:5]; [e setBytes:&b2 length:4 atIndex:6];
            [e setBytes:&bc2s length:4 atIndex:7]; [e setBytes:&eps length:4 atIndex:8]; [e setBytes:&n length:4 atIndex:9];
            [e setBuffer:nanc offset:0 atIndex:10]; [e setBytes:&gi length:4 atIndex:11];
            [e dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(psoAdam.maxTotalThreadsPerThreadgroup, n), 1, 1)];
        }
        // Q arm: project_and_sh_backward_q (bindings as msplat_metal.mm q_encode_bwd_adam), then fused_adam_q16
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:psoBwdQ];
        [e setBytes:&np length:4 atIndex:0]; [e setBuffer:qp[0] offset:0 atIndex:1]; [e setBuffer:qp[1] offset:0 atIndex:2];
        [e setBytes:&glob length:4 atIndex:3]; [e setBuffer:qp[2] offset:0 atIndex:4];
        [e setBuffer:viewB offset:0 atIndex:5]; [e setBuffer:projB offset:0 atIndex:6];
        [e setBytes:intr length:16 atIndex:7]; [e setBytes:isz length:8 atIndex:8];
        [e setBuffer:radiiB offset:0 atIndex:9]; [e setBuffer:conicsB offset:0 atIndex:10];
        [e setBuffer:vxy offset:0 atIndex:11]; [e setBuffer:vdep offset:0 atIndex:12]; [e setBuffer:vcon offset:0 atIndex:13];
        [e setBuffer:vmeanQ offset:0 atIndex:14]; [e setBuffer:vscaleQ offset:0 atIndex:15]; [e setBuffer:vquatQ offset:0 atIndex:16];
        [e setBytes:&degree length:4 atIndex:17]; [e setBytes:&dtu length:4 atIndex:18];
        [e setBytes:campos length:16 atIndex:19]; [e setBuffer:vcol offset:0 atIndex:20];
        [e setBuffer:qp[3] offset:0 atIndex:21];
        [e setBuffer:frQ offset:0 atIndex:22]; [e setBuffer:frBQ offset:0 atIndex:23];
        [e setBuffer:shqQ offset:0 atIndex:24]; [e setBuffer:shqBQ offset:0 atIndex:25];
        [e setBuffer:nsqQ[3] offset:0 atIndex:26]; [e setBuffer:nsqBQ[3] offset:0 atIndex:27];
        [e setBytes:&hpQ length:sizeof hpQ atIndex:28]; [e setBuffer:nancQ offset:0 atIndex:29]; [e setBuffer:qsatB offset:0 atIndex:30];
        [e dispatchThreadgroups:MTLSizeMake(nblk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        for (int g : {0, 1, 2, 4}) {
            uint32_t rowsN = (uint32_t)N, gi = (uint32_t)GADAM[g], dim = (uint32_t)GDIM[g], st = (uint32_t)t, rd = (uint32_t)mode;
            float ss = LR[GADAM[g]] / bc1;
            [e setComputePipelineState:psoAdamQ];
            [e setBuffer:qp[g] offset:0 atIndex:0]; [e setBuffer:gr[g] offset:0 atIndex:1];  // reference projection grads
            [e setBuffer:nsqQ[g] offset:0 atIndex:2]; [e setBuffer:nsqBQ[g] offset:0 atIndex:3];
            [e setBytes:&ss length:4 atIndex:4]; [e setBytes:&b1 length:4 atIndex:5]; [e setBytes:&b2 length:4 atIndex:6];
            [e setBytes:&bc2s length:4 atIndex:7]; [e setBytes:&eps length:4 atIndex:8]; [e setBytes:&rowsN length:4 atIndex:9];
            [e setBuffer:nancQ offset:0 atIndex:10]; [e setBytes:&gi length:4 atIndex:11]; [e setBytes:&dim length:4 atIndex:12];
            [e setBytes:&st length:4 atIndex:13]; [e setBytes:&rd length:4 atIndex:14]; [e setBuffer:qsatB offset:0 atIndex:15];
            [e dispatchThreadgroups:MTLSizeMake(nblk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        // junk discriminator on the Q arm's step-t state: opacity logits 0, op_reg 1, sc_reg 0, invis_scale 0
        // => v_opacity != 0 exactly on junk rows
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            float opr = 1.0f, scr = 0.0f, inv = 0.0f;
            [e setComputePipelineState:psoRegQ];
            [e setBuffer:zeroOp offset:0 atIndex:0]; [e setBuffer:qp[1] offset:0 atIndex:1];
            [e setBuffer:vopJ offset:0 atIndex:2]; [e setBuffer:vscJ offset:0 atIndex:3];
            [e setBytes:&np length:4 atIndex:4]; [e setBytes:&opr length:4 atIndex:5]; [e setBytes:&scr length:4 atIndex:6];
            [e setBuffer:radiiB offset:0 atIndex:7]; [e setBytes:&inv length:4 atIndex:8]; [e setBuffer:nsqQ[3] offset:0 atIndex:9];
            [e dispatchThreads:MTLSizeMake(N, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(psoRegQ.maxTotalThreadsPerThreadgroup, N), 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) Q.completed = false;
        for (int i = 0; i < 64; i++) { Q.nanRef += P<uint32_t>(nanc)[i]; Q.nanQ += P<uint32_t>(nancQ)[i]; }

        // reference arm: CPU Q encoding of the fp32 results, starting from the Q arm's state (as --mode reference)
        QState ref = prev;
        const size_t nevR = RR.ev.size();
        {
            const float *restN = P<float>(rest), *restMN = P<float>(restM), *restVN = P<float>(restV);
            for (int b = 0; b < nblk; b++) {
                const int r0 = b * QB, r1 = std::min(N, (b + 1) * QB);
                for (int d = 0; d < 3; d++) {  // SH value
                    float mx = 0; bool any = false;
                    for (int r = r0; r < r1; r++)
                        for (int j : ksDeg[d]) {
                            size_t c = (size_t)r * ST + j;
                            if (same(restN[c], restOld[c])) continue;
                            any = true; mx = std::max(mx, std::fabs(restN[c]));
                        }
                    if (!any) continue;
                    const float a = ref.frB[b * 3 + d];
                    const bool grow = mx > a;
                    const float a2 = grow ? shBound(mx) : a;  // attempt 2 floor
                    if (grow) RR.ev.push_back({t, b, d, a == 0.0f});
                    for (int r = r0; r < r1; r++)
                        for (int j : ksDeg[d]) {
                            size_t c = (size_t)r * ST + j;
                            int code = ref.fr[c];
                            if (!same(restN[c], restOld[c])) code = encS(restN[c], a2, SH_M, stSH, t, (uint32_t)c, 2);
                            else if (grow) code = encSym(restOld[c], a2, SH_M);
                            ref.fr[c] = (int16_t)code;
                        }
                    ref.frB[b * 3 + d] = a2;
                }
                for (int d = 0; d < 3; d++)
                    encAdam<PackSH>(RR, t, b, 3 + d, 6 + d, ksDeg[d], ST, restMO.data(), restVO.data(), restMN, restVN,
                                    ref.shq.data(), &ref.shqB[b * 6 + d * 2], SHA_MU, SHA_ML, stAdam);
                for (int g = 0; g < NG; g++)  // fixed bounds (attempt 2)
                    encAdamFixed(RR, t, b, 9 + 2 * g, 10 + 2 * g, GDIM[g], gmO[g].data(), gvO[g].data(), P<float>(gm[g]),
                                 P<float>(gv[g]), ref.nsq[g].data(), stAdam);
            }
        }
        const QState cur = readQ();
        nzTrack(RQ, cur, N);

        // codes (per component; denominators: SH value / SH Adam = active rows x open-degree cells, DC = active rows,
        // means/scales/quats/opacity = all rows)
        long den[NCOMP] = {}, dif[NCOMP] = {}; int mxa[NCOMP] = {};
        auto acc = [&](int comp, int a, int b) { den[comp]++; const int d = std::abs(a - b); dif[comp] += d != 0; mxa[comp] = std::max(mxa[comp], d); };
        for (int r = 0; r < N; r++) {
            const bool act = radii[r] != 0;
            if (act) {
                for (int j = 0; j < ST; j++) {
                    const int d = degOf(j);
                    if ((uint32_t)d >= dtu) continue;
                    const size_t c = (size_t)r * ST + j;
                    acc(d, cur.fr[c], ref.fr[c]);
                    acc(3 + d, PackSH::u(cur.shq[c]), PackSH::u(ref.shq[c]));
                    acc(6 + d, PackSH::l(cur.shq[c]), PackSH::l(ref.shq[c]));
                }
                for (int k = 0; k < 3; k++) {
                    const size_t c = (size_t)r * 3 + k;
                    acc(15, PackNS::u(cur.nsq[3][c]), PackNS::u(ref.nsq[3][c]));
                    acc(16, PackNS::l(cur.nsq[3][c]), PackNS::l(ref.nsq[3][c]));
                }
            }
            for (int g : {0, 1, 2, 4})
                for (int k = 0; k < GDIM[g]; k++) {
                    const size_t c = (size_t)r * GDIM[g] + k;
                    acc(9 + 2 * g, PackNS::u(cur.nsq[g][c]), PackNS::u(ref.nsq[g][c]));
                    acc(10 + 2 * g, PackNS::l(cur.nsq[g][c]), PackNS::l(ref.nsq[g][c]));
                }
        }
        for (int c = 0; c < NCOMP; c++) {
            CodeStat &s = Q.code[c];
            if (!den[c]) continue;
            s.steps++; s.den += den[c]; s.diff += dif[c];
            s.maxAbs = std::max(s.maxAbs, mxa[c]);
            const double rate = (double)dif[c] / den[c];
            if (!s.worstT || rate > s.worstRate) { s.worstRate = rate; s.worstT = t; }
            if (mxa[c] > 1 || rate > 1e-3) s.fail = true;
        }
        // Q_DEBUG_CELLS side output (attempt-2 diagnosis, 2 Nov): dumps cells; never changes a verdict.
        if (g_dbg) {
            const float *restN = P<float>(rest), *restMN = P<float>(restM), *restVN = P<float>(restV), *vcl = P<float>(vcol);
            // (A) SH value: blocks with an active open-degree |d| >= 2 this step; every |d| >= 1 cell of those blocks
            std::set<int> hot;
            for (int r = 0; r < N; r++) {
                if (!radii[r]) continue;
                for (int j = 0; j < ST; j++)
                    if ((uint32_t)degOf(j) < dtu && std::abs(cur.fr[(size_t)r * ST + j] - ref.fr[(size_t)r * ST + j]) >= 2) hot.insert(r / QB);
            }
            for (int b : hot)
                for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++)
                    for (int j = 0; j < ST; j++) {
                        const size_t c = (size_t)r * ST + j;
                        const int d = degOf(j), dd = std::abs(cur.fr[c] - ref.fr[c]);
                        if ((uint32_t)d >= dtu || dd < 1) continue;
                        const float aO = prev.frB[b * 3 + d], aQ = cur.frB[b * 3 + d], aR = ref.frB[b * 3 + d];
                        const float auO = prev.shqB[b * 6 + d * 2], hO = prev.shqB[b * 6 + d * 2 + 1];
                        fprintf(g_dbg, "SHV fx=%s mode=%s t=%d row=%d blk=%d deg=%d coef=%d ch=%d active=%d |d|=%d grewQ=%d grewR=%d "
                                "old_code=%d old_a=%.9g a_Q=%.9g a_R=%.9g code_Q=%d code_R=%d dec_Q=%.9g dec_R=%.9g x_old=%.9g "
                                "x_new_ref=%.9g refUnchanged=%d gpuKeptOld=%d hash_r=%.9g st_ref=%d st_gpu=%d t_ref=%.6f "
                                "cand_nearest_aQ=%d cand_stoch_aQ=%d cand_reenc_aQ=%d shA_u_old=%d shA_l_old=%d au_old=%.6g h_old=%.6g "
                                "m_old=%.6g v_old=%.6g m_new_ref=%.6g v_new_ref=%.6g vcol=%.6g\n",
                                fx.c_str(), MODE_NAME[mode], t, r, b, d, j / 3, j % 3, radii[r] != 0, dd, !same(aQ, aO), !same(aR, aO),
                                prev.fr[c], aO, aQ, aR, cur.fr[c], ref.fr[c], decSym(cur.fr[c], aQ, SH_M), decSym(ref.fr[c], aR, SH_M),
                                restOld[c], restN[c], same(restN[c], restOld[c]), cur.fr[c] == prev.fr[c], hashR(t, (uint32_t)c, 2), stSH,
                                mode == 1, (double)restN[c] * SH_M / aQ, encSym(restN[c], aQ, SH_M), encS(restN[c], aQ, SH_M, true, t, (uint32_t)c, 2),
                                encSym(restOld[c], aQ, SH_M), PackSH::u(prev.shq[c]), PackSH::l(prev.shq[c]), auO, hO, restMO[c], restVO[c],
                                restMN[c], restVN[c], vcl[(size_t)r * 3 + j % 3]);
                    }
            // (B) non-SH / DC log_s |d| = 1 cells, nearest mode, F1, up to 1000: the reference's new (m, v) re-encoded by the
            // GPU codec (q_parity_adam_enc_kernel, identical input and the reference's bounds) and the +-1/2 ulp log test
            if (mode == 0 && fx == "f1" && g_dbgLs < 1000) {
                struct Cell { int g, b; size_t c; float m, v, au, h; int lq, lr; };
                std::vector<Cell> cells;
                for (int g = 0; g < NG && g_dbgLs + (int)cells.size() < 1000; g++)
                    for (size_t c = 0; c < (size_t)N * GDIM[g] && g_dbgLs + (int)cells.size() < 1000; c++) {
                        const int lq = PackNS::l(cur.nsq[g][c]), lr = PackNS::l(ref.nsq[g][c]);
                        if (std::abs(lq - lr) != 1) continue;
                        const int b = (int)(c / GDIM[g]) / QB;
                        cells.push_back({g, b, c, P<float>(gm[g])[c], P<float>(gv[g])[c], ref.nsqB[g][b * 2], ref.nsqB[g][b * 2 + 1], lq, lr});
                    }
                if (!cells.empty()) {
                    const uint32_t n1 = 1, st0 = 0, s0 = 0;
                    const int Mu = NSA_MU, Ml = NSA_ML;
                    id<MTLBuffer> mvb = buf(cells.size() * 8), co = buf(cells.size() * 8);
                    for (size_t i = 0; i < cells.size(); i++) { P<float>(mvb)[2 * i] = cells[i].m; P<float>(mvb)[2 * i + 1] = cells[i].v; }
                    id<MTLCommandBuffer> dcb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> de = [dcb computeCommandEncoder];
                    [de setComputePipelineState:psoParAe];
                    for (size_t i = 0; i < cells.size(); i++) {
                        [de setBuffer:mvb offset:8 * i atIndex:0]; [de setBuffer:co offset:8 * i atIndex:1];
                        [de setBytes:&cells[i].au length:4 atIndex:2]; [de setBytes:&cells[i].h length:4 atIndex:3];
                        [de setBytes:&Mu length:4 atIndex:4]; [de setBytes:&Ml length:4 atIndex:5]; [de setBytes:&st0 length:4 atIndex:6];
                        [de setBytes:&s0 length:4 atIndex:7]; [de setBytes:&n1 length:4 atIndex:8];
                        [de dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    }
                    [de endEncoding]; [dcb commit]; [dcb waitUntilCompleted];
                    for (size_t i = 0; i < cells.size(); i++) {
                        const Cell &k = cells[i];
                        const float ls = adamLogS(k.v), sq = std::sqrt(k.v), ratio = sq / ADAM_EPS, onep = 1.0f + ratio;
                        const int lpar = P<int32_t>(co)[2 * i + 1], lcpu = encUns(ls, k.h, Ml);
                        const int lup = encUns(nextafterf(ls, INFINITY), k.h, Ml), ldn = encUns(nextafterf(ls, -INFINITY), k.h, Ml);
                        const double tl = (double)ls * Ml / k.h, frac = tl - std::floor(tl);
                        const double lsD = std::log1p((double)sq / 1e-15);  // double reference of the same formula
                        const bool repro = lpar == k.lq, ulp1 = (lup == k.lq || ldn == k.lq);
                        g_dbgLsRepro += repro; g_dbgLsUlp += ulp1; g_dbgLsBoth += repro && ulp1;
                        g_dbgLsDblQ += (encUns((float)lsD, k.h, Ml) == k.lq);
                        fprintf(g_dbg, "LS fx=%s t=%d group=%s blk=%d cell=%zu m=%.9g v=%.9g h_R=%.9g code_Q=%d code_R=%d code_cpu=%d "
                                "code_gpu_same_input=%d code_ls_up1ulp=%d code_ls_dn1ulp=%d ls_cpu=%.9g ls_double=%.12g t=%.6f frac=%.6f "
                                "onep=%.9g\n", fx.c_str(), t, CN[9 + 2 * k.g], k.b, k.c, k.m, k.v, k.h, k.lq, k.lr, lcpu, lpar, lup, ldn, ls,
                                lsD, tl, frac, onep);
                    }
                    g_dbgLs += (int)cells.size();
                }
            }
        }
        // bounds and growth events
        std::set<std::pair<int, int>> evq, evr;
        for (size_t i = nevR; i < RR.ev.size(); i++) evr.insert({RR.ev[i].blk, RR.ev[i].comp});
        const size_t nevQ = RQ.ev.size();
        for (int b = 0; b < nblk; b++)
            for (int c = 0; c < NCOMP; c++) {
                const float q = qBound(cur, b, c), r = qBound(ref, b, c), o = qBound(prev, b, c);
                if (!std::isfinite(q)) Q.boundNonfinite++;
                const double dd = std::fabs((double)q - r);
                if (!(dd <= 1e-6 * std::fabs((double)r) + 2.0 * FLT_MIN)) Q.boundViol++;
                if (r != 0.0f) Q.boundMaxRel = std::max(Q.boundMaxRel, dd / std::fabs((double)r));
                else if (q != 0.0f) Q.boundMaxRel = INFINITY;
                if (!same(q, o)) { evq.insert({b, c}); RQ.ev.push_back({t, b, c, o == 0.0f}); }
            }
        Q.evQ += (long)evq.size(); Q.evRef += (long)evr.size();
        for (auto &x : evq) Q.evMis += !evr.count(x);
        for (auto &x : evr) Q.evMis += !evq.count(x);
        // fp32 params and projection gradients
        for (int g = 0; g < NG; g++) {
            const float *q = P<float>(qp[g]), *r = P<float>(rp[g]);
            for (size_t i = 0; i < (size_t)N * GDIM[g]; i++) {
                if (!std::isfinite(q[i])) Q.paramNonfinite++;
                const double d = std::fabs((double)q[i] - r[i]) / std::max(1.0, std::fabs((double)r[i]));
                if (!(d <= 1e-6)) Q.paramViol++;
                if (d > Q.paramMax || d != d) Q.paramMax = d != d ? INFINITY : d;
            }
        }
        {
            id<MTLBuffer> gq[3] = {vmeanQ, vscaleQ, vquatQ}, grr[3] = {vmean, vscale, vquat};
            const int gd[3] = {3, 3, 4};
            for (int k = 0; k < 3; k++) {
                const float *q = P<float>(gq[k]), *r = P<float>(grr[k]);
                for (size_t i = 0; i < (size_t)N * gd[k]; i++) {
                    if (!std::isfinite(q[i])) Q.gradNonfinite++;
                    if (radii[i / gd[k]] == 0 && q[i] != 0.0f) Q.gradInactNZ++;
                    const double d = std::fabs((double)q[i] - r[i]) / std::max(1.0, std::fabs((double)r[i]));
                    if (!(d <= 1e-6)) Q.gradViol++;
                    if (d > Q.gradMax || d != d) Q.gradMax = d != d ? INFINITY : d;
                }
            }
        }
        // unopened degrees byte-identical (Q arm), inactive rows (compareInactiveRows), padding rows, N blocks
        std::array<long, 3> chg = {0, 0, 0};
        for (size_t c = 0; c < (size_t)N * ST; c++) {
            const int d = degOf((int)(c % ST));
            if (cur.fr[c] != prev.fr[c]) chg[d]++;
            if ((uint32_t)d >= dtu) Q.unopened += cur.fr[c] != prev.fr[c] || cur.shq[c] != prev.shq[c];
        }
        for (int b = 0; b < nblk; b++)
            for (int d = (int)dtu; d < 3; d++)
                Q.unopened += !same(cur.frB[b * 3 + d], prev.frB[b * 3 + d]) || !same(cur.shqB[b * 6 + d * 2], prev.shqB[b * 6 + d * 2]) ||
                              !same(cur.shqB[b * 6 + d * 2 + 1], prev.shqB[b * 6 + d * 2 + 1]);
        RQ.shchg.push_back(chg);
        {
            InactCmp ic = compareInactiveRows(prev, cur, radii, N);
            Q.inactCells += ic.cells; Q.inactFlagged += ic.flagged;
            RQ.inactCells += ic.cells;
            Q.refInactFlagged += compareInactiveRows(prev, ref, radii, N).flagged;
        }
        for (size_t c = (size_t)N * ST; c < cur.fr.size(); c++) Q.padNZ += cur.fr[c] != 0 || cur.shq[c] != 0;
        for (int g = 0; g < NG; g++) for (size_t c = (size_t)N * GDIM[g]; c < cur.nsq[g].size(); c++) Q.padNZ += cur.nsq[g][c] != 0;
        if (nSnap(cur) != n0) RQ.nCodeChanges++;
        // junk sets
        for (int r = 0; r < N; r++) {
            const uint32_t *q = &cur.nsq[3][(size_t)r * 3];
            const bool cpu = !(q[0] >> 16) && !(q[1] >> 16) && !(q[2] >> 16), gpu = P<float>(vopJ)[r] != 0.0f;
            Q.junkMis += cpu != gpu; Q.junkRows += cpu;
        }
        // coverage (a)/(b) on the Q arm, as --mode reference
        auto nz = [&](int r, int comp) {
            if (comp <= 8) {
                for (int j : ksDeg[comp % 3]) {
                    const size_t c = (size_t)r * ST + j;
                    if (comp <= 2 ? prev.fr[c] != 0 : comp <= 5 ? PackSH::u(prev.shq[c]) != 0 : PackSH::l(prev.shq[c]) != 0) return true;
                }
                return false;
            }
            for (int k = 0; k < 3; k++) {
                const uint32_t q = prev.nsq[3][(size_t)r * 3 + k];
                if (comp == 15 ? PackNS::u(q) != 0 : PackNS::l(q) != 0) return true;
            }
            return false;
        };
        std::vector<char> grew((size_t)nblk * NCOMP, 0);
        for (size_t i = nevQ; i < RQ.ev.size(); i++) grew[(size_t)RQ.ev[i].blk * NCOMP + RQ.ev[i].comp] = 1;
        const int covComps[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16};
        long nsw = 0;
        for (int b = 0; b < nblk; b++) {
            std::vector<int> sw;
            for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++) if (radii[r] == 0 && radii1[r] != 0) sw.push_back(r);
            nsw += (long)sw.size();
            if (sw.empty()) continue;
            const int grp = (b >= 11 && b <= 13) ? 0 : b == 7 ? 1 : isZ(b) ? 2 : 3;
            for (int comp : covComps) {
                if (grew[(size_t)b * NCOMP + comp]) {
                    RQ.lateGrow[comp]++;
                    bool cov = false;
                    for (int r : sw) if (nz(r, comp)) { cov = true; break; }
                    if (cov) { RQ.lateCov[comp]++; RQ.lateCovGrp[comp][grp]++; }
                } else if (comp >= 3) {
                    for (int r : sw) RQ.stableNZ[comp] += nz(r, comp);
                }
            }
        }
        RQ.switchedMax = std::max(RQ.switchedMax, nsw);
    }
    Q.qsat = P<uint32_t>(qsatB)[0];
    Q.usat = P<uint32_t>(qsatB)[1];
    RQ.qsat = Q.qsat;
    // Q-arm clamp counts exist only as GPU totals: every non-SH / DC u component gets usat, every log_s component qsat
    for (int c = 9; c < NCOMP; c++) RQ.sat[c] = c % 2 ? Q.usat : Q.qsat;
    Q.refQsat = RR.qsat;
    for (int c = 9; c < NCOMP; c += 2) Q.refUsat += RR.sat[c];
    if (fx == "f1") fwdCheck("t=70");
    return Q;
}

// Mandatory coverage (s4.2) on the Q-arm events, the list of the --mode reference inventory.
static int qCoverage(const Run &f1, const Run &f2) {
    int missing = 0;
    auto req = [&](bool ok, const char *fmt, auto... a) {
        printf("  [%s] coverage: ", ok ? "PASS" : "FAIL");
        printf(fmt, a...);
        printf("\n");
        missing += !ok; qSectFail += !ok; qSectN++;
    };
    const int open[3] = {11, 21, 31};
    for (int d = 0; d < 3; d++) {
        int n = 0, first = 1 << 30;
        for (int b = 0; b < 4; b++) { int s = firstStep(f1, d, b, 1); if (s > 0) { n++; first = std::min(first, s); } }
        req(n >= 1 && first == open[d], "Z shv_d%d from zero: %d/4 Z blocks, first t=%d (degree opens t=%d)", d, n, n ? first : -1, open[d]);
    }
    for (int c = 3; c <= 8; c++)  // SH Adam; non-SH / DC have fixed bounds (attempt 2), events below
        req(countBlocks(f1, c, -1, 1, isZ) >= 1, "Z %s from zero: F1 %d/4 Z blocks, F2 %d/4", CN[c],
            countBlocks(f1, c, -1, 1, isZ), countBlocks(f2, c, -1, 1, isZ));
    for (int d = 0; d < 3; d++) {
        int s = firstStep(f1, d, 11 + d, 0), n = 0;
        for (auto &e : f1.ev) n += e.comp == d && e.blk == 11 + d && !e.fromZero;
        req(s >= open[d], "P shv_d%d block %d from positive bound (after degree opens, t=%d): %d growth steps, first t=%d", d, 11 + d,
            open[d], n, s);
    }
    for (int c = 3; c <= 8; c++) {
        const int blk = 7;
        int n = 0, first = -1;
        for (auto &e : f1.ev)
            if (e.comp == c && e.blk == blk && e.t >= 41 && !e.fromZero) { n++; if (first < 0) first = e.t; }
        req(n >= 1, "P %s block %d from positive bound in t=41-70: %d growth steps, first t=%d", CN[c], blk, n, first);
    }
    for (const Run *R : {&f1, &f2}) {
        int g = 0;
        for (int c = 0; c < NCOMP; c++) g += countBlocks(*R, c, -1, -1, isN);
        req(g == 0 && R->nCodeChanges == 0, "N blocks 100-109 %s: %d growth (block, component) pairs, %ld steps with a code/bound change",
            R->fx.c_str(), g, R->nCodeChanges);
    }
    for (const Run *R : {&f1, &f2}) {
        long mn[3] = {-1, -1, -1};
        int miss = 0;
        for (int t = 0; t < R->steps; t++)
            for (int d = 0; d < (int)R->dtu[t]; d++) {
                if (mn[d] < 0 || R->shchg[t][d] < mn[d]) mn[d] = R->shchg[t][d];
                miss += R->shchg[t][d] == 0;
            }
        req(miss == 0, "%s changed SH codes > 0 in every open degree and step: min d0 %ld d1 %ld d2 %ld", R->fx.c_str(), mn[0], mn[1], mn[2]);
    }
    for (int c = 9; c < NCOMP; c++) {
        // Codex C1 #5: the Q-arm bound stays the constant from initialisation on (any change is a bound event)
        long bev1 = 0, bev2 = 0;
        for (auto &e : f1.ev) bev1 += e.comp == c;
        for (auto &e : f2.ev) bev2 += e.comp == c;
        req(f1.sat[c] == 0 && f2.sat[c] == 0 && f1.nzMax[c] > 0 && f2.nzMax[c] > 0 && bev1 == 0 && bev2 == 0,
            "%s fixed bounds: GPU %s F1 %ld F2 %ld; non-zero code cells (max over steps) F1 %ld F2 %ld; bound changes F1 %ld F2 %ld",
            CN[c], c % 2 ? "usat" : "qsat", f1.sat[c], f2.sat[c], f1.nzMax[c], f2.nzMax[c], bev1, bev2);
    }
    for (int c = 3; c <= 8; c++) {
        const int d = (c - 3) % 3, t1 = open[d];
        req(countBlocks(f1, c, t1, 1, nullptr) >= 1 && countBlocks(f2, c, 1, 1, nullptr) >= 1, "%s from zero: F1 t=%d %d blocks, F2 t=1 %d blocks",
            CN[c], t1, countBlocks(f1, c, t1, 1, nullptr), countBlocks(f2, c, 1, 1, nullptr));
    }
    for (int c : {0, 1, 2, 3, 4, 5, 6, 7, 8})  // DC: fixed bound, never grows; (a) does not apply
        req(f1.lateCov[c] >= 1, "(a) %s: late growth events %ld, covered by a switched-inactive non-zero row %ld (blocks 11-13 %ld, 7 %ld, Z %ld, other %ld)",
            CN[c], f1.lateGrow[c], f1.lateCov[c], f1.lateCovGrp[c][0], f1.lateCovGrp[c][1], f1.lateCovGrp[c][2], f1.lateCovGrp[c][3]);
    for (int c : {3, 4, 5, 6, 7, 8, 15, 16})
        req(f1.stableNZ[c] >= 1, "(b) %s: switched-inactive non-zero (row, step) pairs in non-growing blocks %ld", CN[c], f1.stableNZ[c]);
    for (const Run *R : {&f1, &f2}) {
        int mn = 1 << 30;
        for (int v : R->inactPerRole) mn = std::min(mn, v);
        req(mn >= 1, "%s inactive rows in every role block (0-3, 9, 100-109): min %d", R->fx.c_str(), mn);
    }
    return missing;
}

static void reportQ(const QRes &Q) {
    const char *fm = Q.fx.c_str(), *mn = MODE_NAME[Q.mode];
    printf("\n--- %s, mode %s (%d steps, N %d) ---\n", fm, mn, Q.steps, Q.N);
    for (int c = 0; c < NCOMP; c++) {
        const CodeStat &s = Q.code[c];
        sect(s.steps > 0 && !s.fail, "%s %s codes %-10s: max|d| %d, worst d!=0 rate %.3g (t=%d), cells %ld over %ld steps, d!=0 %ld",
             fm, mn, CN[c], s.maxAbs, s.worstRate, s.worstT, s.den, s.steps, s.diff);
    }
    sect(Q.boundViol == 0 && Q.boundNonfinite == 0, "%s %s bounds: violations of rel <= 1e-6 (+2 FLT_MIN) %ld, max rel diff %.3g, nonfinite %ld",
         fm, mn, Q.boundViol, Q.boundMaxRel, Q.boundNonfinite);
    sect(Q.evMis == 0 && Q.evQ > 0, "%s %s growth-event sets: Q %ld, reference %ld, mismatched (block, component, step) %ld", fm, mn, Q.evQ, Q.evRef,
         Q.evMis);
    sect(Q.paramViol == 0 && Q.paramNonfinite == 0, "%s %s fp32 params: max |d|/max(1,|p|) %.3g, over 1e-6 %ld, nonfinite %ld", fm, mn,
         Q.paramMax, Q.paramViol, Q.paramNonfinite);
    sect(Q.gradViol == 0 && Q.gradNonfinite == 0 && Q.gradInactNZ == 0,
         "%s %s projection grads (all rows): max |d|/max(1,|g|) %.3g, over 1e-6 %ld, nonfinite %ld, nonzero on inactive rows %ld", fm, mn,
         Q.gradMax, Q.gradViol, Q.gradNonfinite, Q.gradInactNZ);
    sect(Q.unopened == 0, "%s %s unopened degrees byte-identical: changed cells/bounds %ld", fm, mn, Q.unopened);
    sect(Q.inactFlagged == 0 && Q.inactCells > 0, "%s %s inactive rows (compareInactiveRows): cells %ld, flagged %ld (reference arm flagged %ld)",
         fm, mn, Q.inactCells, Q.inactFlagged, Q.refInactFlagged);
    if (Q.N % QB) sect(Q.padNZ == 0, "%s %s padding rows (%d) byte-identical 0: nonzero codes %ld", fm, mn, (Q.N + QB - 1) / QB * QB - Q.N, Q.padNZ);
    sect(Q.completed, "%s %s kernels completed: %s", fm, mn, Q.completed ? "yes" : "NO");
    sect(Q.qsat == 0 && Q.refQsat == 0 && Q.usat == 0 && Q.refUsat == 0, "%s %s qsat = 0, usat = 0: Q %u / %u, reference arm %ld / %ld",
         fm, mn, Q.qsat, Q.usat, Q.refQsat, Q.refUsat);
    sect(Q.nanQ == 0 && Q.nanRef == 0, "%s %s NaN-gate counters: Q %ld, reference arm %ld", fm, mn, Q.nanQ, Q.nanRef);
    sect(Q.junkMis == 0, "%s %s junk sets (mcmc_reg_grad_q vs CPU rule, every step): mismatched rows %ld (junk row-steps %ld)", fm, mn,
         Q.junkMis, Q.junkRows);
    for (const FwdStat &f : Q.fwd)
        sect(f.intMis == 0 && f.sentMis == 0 && f.floatViol == 0 && f.refNonfinite == 0,
             "%s %s forward %s dtu=%d: int mismatches %ld, sentinel mismatches %ld, float over 1e-6 %ld (max %.3g), reference nonfinite %ld",
             fm, mn, f.when.c_str(), f.dtu, f.intMis, f.sentMis, f.floatViol, f.floatMax, f.refNonfinite);
}

// s3 codec parity.
static void runParity() {
    auto run1 = [&](id<MTLComputePipelineState> p, std::initializer_list<id<MTLBuffer>> bufs, const void *bytes[], const int *lens, int nb,
                    uint32_t n) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        int i = 0;
        for (id<MTLBuffer> b : bufs) [e setBuffer:b offset:0 atIndex:i++];
        for (int k = 0; k < nb; k++) [e setBytes:bytes[k] length:lens[k] atIndex:i++];
        [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, n), 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted;
    };
    const uint32_t STEP = 1;
    const float *shA = F<float>("s3_sh_bounds.f32", 6), *sa = F<float>("s3_sh_a.f32", 6 * 65535), *sb = F<float>("s3_sh_b.f32", 600000),
                *sc = F<float>("s3_sh_c.f32", 6 * 28), *sd = F<float>("s3_sh_d.f32", 1000);
    printf("\n=== s3 SH value parity (codes and decodes byte-identical) ===\n");
    for (int m = 0; m < 3; m++) {
        const uint32_t st = m == 1;
        for (int ai = 0; ai < 6; ai++) {
            const float a = shA[ai];
            std::vector<float> x; std::vector<int32_t> cin; std::vector<int> part;
            for (int c = -SH_M; c <= SH_M; c++) { x.push_back(sa[ai * 65535 + c + SH_M]); cin.push_back(c); part.push_back(0); }
            for (int i = 0; i < 100000; i++) { x.push_back(sb[ai * 100000 + i]); part.push_back(1); }
            for (int i = 0; i < 28; i++) { x.push_back(sc[ai * 28 + i]); part.push_back(2); }
            for (int i = 0; i < 1000; i++) { x.push_back(sd[i]); part.push_back(3); }
            const uint32_t n = (uint32_t)x.size();
            std::vector<int32_t> ccpu(n);
            for (uint32_t i = 0; i < n; i++) ccpu[i] = encS(fz(x[i]), a, SH_M, st, STEP, i, 2);
            for (uint32_t i = 65535; i < n; i++) cin.push_back(ccpu[i]);
            id<MTLBuffer> xb = buf(n * 4), cb_in = buf(n * 4), co = buf(n * 4), dq = buf(n * 4);
            memcpy(xb.contents, x.data(), n * 4); memcpy(cb_in.contents, cin.data(), n * 4);
            const void *bytes[] = {&a, &st, &STEP, &n}; const int lens[] = {4, 4, 4, 4};
            const bool done = run1(psoParSh, {xb, cb_in, co, dq}, bytes, lens, 4, n);
            long encMis[4] = {}, decMis[4] = {};
            for (uint32_t i = 0; i < n; i++) {
                encMis[part[i]] += P<int32_t>(co)[i] != ccpu[i];
                decMis[part[i]] += !same(fz(P<float>(dq)[i]), fz(decSym(cin[i], a, SH_M)));
            }
            const long tot = encMis[0] + encMis[1] + encMis[2] + encMis[3] + decMis[0] + decMis[1] + decMis[2] + decMis[3];
            sect(done && tot == 0, "parity %s shv a=%g: code mismatches (a) %ld (b) %ld (c) %ld (d) %ld; decode mismatches (a) %ld (b) %ld (c) %ld (d) %ld",
                 MODE_NAME[m], a, encMis[0], encMis[1], encMis[2], encMis[3], decMis[0], decMis[1], decMis[2], decMis[3]);
        }
    }
    printf("\n=== s3 Adam (u, log_s) parity ===\n");
    const float *pairs = F<float>("s3_adam_pairs.f32", 32), *aa = F<float>("s3_adam_a.f32", (size_t)16 * 200000), *ad = F<float>("s3_adam_d.f32", 2000);
    for (int m = 0; m < 3; m++) {
        const uint32_t st = m >= 1;
        for (int sh = 1; sh >= 0; sh--) {
            const int Mu = sh ? SHA_MU : NSA_MU, Ml = sh ? SHA_ML : NSA_ML;
            const std::string cn = sh ? "sh" : "nsh";
            const size_t nb = sh ? 255 : 9363, nc = sh ? 256 : 9363;
            const int32_t *cb_ = F<int32_t>("s3_adam_b_" + cn + ".i32", nb * 2), *cc_ = F<int32_t>("s3_adam_c_" + cn + ".i32", nc * 2);
            // v7 s3 cases unchanged: 16 bound pairs, encoder + decoder each. Attempt 2 (v8 s3, Codex C1 #1) adds, for
            // non-SH only, the production bounds (8, 64): encoder on every sample set drawn inside them (a_u <= 8,
            // h <= 64: 12 of 16) with pre-clamp excursions required to be 0, and one decoder case.
            struct Case { float au, h; int p; bool enc, dec, prod; };
            std::vector<Case> cases;
            for (int p = 0; p < 16; p++) cases.push_back({pairs[p * 2], pairs[p * 2 + 1], p, true, true, false});
            if (!sh) {
                for (int p = 0; p < 16; p++)
                    if (pairs[p * 2] <= NS_AU && pairs[p * 2 + 1] <= NS_H) cases.push_back({NS_AU, NS_H, p, true, false, true});
                cases.push_back({NS_AU, NS_H, 0, false, true, true});
            }
            for (const Case &cs : cases) {
                const float au = cs.au, h = cs.h; const int p = cs.p;
                bool done = true;
                if (cs.enc) {
                // encoder: (a) 1e5 pairs + (d) 1e3 zeros
                const uint32_t n = 101000;
                std::vector<float> mv((size_t)n * 2);
                memcpy(mv.data(), aa + (size_t)p * 200000, 200000 * 4); memcpy(mv.data() + 200000, ad, 2000 * 4);
                id<MTLBuffer> mvb = buf((size_t)n * 8), co = buf((size_t)n * 8);
                memcpy(mvb.contents, mv.data(), (size_t)n * 8);
                const void *eb[] = {&au, &h, &Mu, &Ml, &st, &STEP, &n}; const int el[] = {4, 4, 4, 4, 4, 4, 4};
                done = run1(psoParAe, {mvb, co}, eb, el, 7, n);
                long nzIn = 0, du = 0, dl = 0, big = 0, exc = 0;
                for (uint32_t i = 0; i < n; i++) {
                    const float m_ = fz(mv[2 * i]), v_ = fz(mv[2 * i + 1]);
                    if (i < 100000 && (mv[2 * i] != 0.0f || mv[2 * i + 1] != 0.0f)) nzIn++;
                    exc += std::fabs(adamU(m_, v_)) > au || adamLogS(v_) > h;
                    const int cu = encS(adamU(m_, v_), au, Mu, st, STEP, i, 0), cl = encU(adamLogS(v_), h, Ml, st, STEP, i, 1);
                    const int gu = P<int32_t>(co)[2 * i], gl = P<int32_t>(co)[2 * i + 1];
                    du += gu != cu; dl += gl != cl;
                    big += std::abs(gu - cu) > 1 || std::abs(gl - cl) > 1;
                }
                const double ru = nzIn ? (double)du / nzIn : 1.0, rl = nzIn ? (double)dl / nzIn : 1.0;
                sect(done && big == 0 && ru <= 1e-3 && rl <= 1e-3 && (!cs.prod || exc == 0),
                     "parity %s adam-%s enc%s a_u=%g h=%g (samples of pair a_u=%g h=%g): cells |d|>1 %ld, d_u!=0 %ld (rate %.3g), d_ls!=0 %ld (rate %.3g), denominator %ld, pre-clamp excursions %ld",
                     MODE_NAME[m], cn.c_str(), cs.prod ? " production" : "", au, h, pairs[p * 2], pairs[p * 2 + 1], big, du, ru, dl, rl, nzIn, exc);
                }
                if (!cs.dec) continue;
                // decoder: (b) + (c) code pairs, plus (0, 0)
                std::vector<int32_t> codes(cb_, cb_ + nb * 2);
                codes.insert(codes.end(), cc_, cc_ + nc * 2);
                codes.push_back(0); codes.push_back(0);
                const uint32_t nd = (uint32_t)(codes.size() / 2);
                id<MTLBuffer> cbuf = buf((size_t)nd * 8), mo = buf((size_t)nd * 8);
                memcpy(cbuf.contents, codes.data(), (size_t)nd * 8);
                const void *db[] = {&au, &h, &Mu, &Ml, &nd}; const int dlen[] = {4, 4, 4, 4, 4};
                done = run1(psoParAd, {cbuf, mo}, db, dlen, 5, nd);
                long mBad = 0, vBad = 0, nonfin = 0, zeroBad = 0;
                double mWorst = 0, vWorst = 0;
                for (uint32_t i = 0; i < nd; i++) {
                    float mc, vc;
                    adamDecode(decSym(codes[2 * i], au, Mu), decUns(codes[2 * i + 1], h, Ml), mc, vc);
                    mc = fz(mc); vc = fz(vc);
                    const float mg = P<float>(mo)[2 * i], vg = P<float>(mo)[2 * i + 1];
                    if (!std::isfinite(mg) || !std::isfinite(vg)) nonfin++;
                    const double em = std::fabs((double)mg - mc), ev = std::fabs((double)vg - vc);
                    if (!(em <= 4e-6 * std::fabs((double)mc) + 2.0 * FLT_MIN)) mBad++;
                    if (!(ev <= 8e-6 * std::fabs((double)vc) + 2.0 * FLT_MIN)) vBad++;
                    if (mc != 0.0f) mWorst = std::max(mWorst, em / std::fabs((double)mc));
                    if (vc != 0.0f) vWorst = std::max(vWorst, ev / std::fabs((double)vc));
                    if (codes[2 * i] == 0 && codes[2 * i + 1] == 0) zeroBad += !(mg == 0.0f && vg == 0.0f);
                }
                sect(done && mBad == 0 && vBad == 0 && nonfin == 0 && zeroBad == 0,
                     "parity %s adam-%s dec%s a_u=%g h=%g: %u pairs, m over bound %ld (max rel %.3g), v over bound %ld (max rel %.3g), (0,0) not exact %ld, nonfinite %ld",
                     MODE_NAME[m], cn.c_str(), cs.prod ? " production" : "", au, h, nd, mBad, mWorst, vBad, vWorst, zeroBad, nonfin);
            }
        }
    }
    // attempt 2 (v8 s3 addition): q_log / q_exp GPU == CPU byte-identical on 2e6 inputs (the generator of
    // runs/hiz-checks/results/q2_2026-11-02/run.mm: mt19937_64(5); z = 1 + (exp(64 U) - 1), L = 64 U; first 1000 near 1 / 0)
    printf("\n=== q_log / q_exp parity ===\n");
    {
        const uint32_t nR = 2000000;
        std::vector<float> z(nR), L(nR);
        std::mt19937_64 g(5); std::uniform_real_distribution<double> U(0, 1);
        for (uint32_t i = 0; i < nR; i++) { z[i] = 1.0f + ((float)std::exp(U(g) * 64.0) - 1.0f); L[i] = (float)(U(g) * 64.0); }
        for (int i = 0; i < 1000; i++) { z[i] = 1.0f + (float)i * 1e-7f; L[i] = (float)i * 1e-4f; }
        // Codex C1 #3: directed inputs, +-64 ulp around every sqrt(2) 2^k branch point and 2^k exponent step of q_log
        // (k = 0..93) and every integer k / ln 2 range-reduction step of q_exp in [0, 64]
        for (int k = 0; k <= 93; k++)
            for (uint32_t c : {q_bits(1.41421356f) + ((uint32_t)k << 23), q_bits(1.0f) + ((uint32_t)k << 23)})
                for (int d = -64; d <= 64; d++) { const float x = q_float(c + (uint32_t)d); if (x >= 1.0f) { z.push_back(x); L.push_back(0.0f); } }
        for (int k = 1; (float)k * 0.693147180559945f <= 64.0f; k++)
            for (int d = -64; d <= 64; d++) { z.push_back(1.0f); L.push_back(q_float(q_bits((float)k * 0.693147180559945f) + (uint32_t)d)); }
        const uint32_t n = (uint32_t)z.size();
        id<MTLBuffer> zb = buf((size_t)n * 4), lb = buf((size_t)n * 4), ol = buf((size_t)n * 4), oe = buf((size_t)n * 4);
        memcpy(zb.contents, z.data(), (size_t)n * 4); memcpy(lb.contents, L.data(), (size_t)n * 4);
        const void *bytes[] = {&n}; const int lens[] = {4};
        const bool done = run1(psoParLE, {zb, lb, ol, oe}, bytes, lens, 1, n);
        long dl = 0, de = 0;
        for (uint32_t i = 0; i < n; i++) { dl += q_bits(P<float>(ol)[i]) != q_bits(q_log(z[i])); de += q_bits(P<float>(oe)[i]) != q_bits(q_exp(L[i])); }
        sect(done && dl == 0 && de == 0, "parity q_log/q_exp GPU==CPU byte-identical on %u inputs: q_log differences %ld, q_exp differences %ld", n, dl, de);
    }
}

// --mode q / --mode parity entry: fixture identity, pipelines, run, inventory, exit code.
static int runQModes(const std::string &mode, const std::string &fixture, const std::string &expect, const std::string &metallib) {
    printf("q_gpu_harness --mode %s (P2-M2; %s)\n", mode.c_str(),
           mode == "q" ? "s4.2 Q kernels vs unchanged fp32 kernels + CPU Q encoding" : "s3 codec parity");
    if (!loadFixture(fixture, expect)) { printf("FAIL: fixture identity\n"); return 1; }
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("FAIL: no Metal device\n"); return 1; }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib.c_str())] error:&err];
    if (!lib) { printf("FAIL: cannot load %s\n", metallib.c_str()); return 1; }
    auto pso = [&](NSString *name) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!p) { printf("FAIL: pipeline %s\n", name.UTF8String); exit(1); }
        return p;
    };
    queue = [dev newCommandQueue];
    printf("device: %s; metallib: %s\n", dev.name.UTF8String, metallib.c_str());
    if (mode == "parity") {
        psoParSh = pso(@"q_parity_sh_kernel"); psoParAe = pso(@"q_parity_adam_enc_kernel"); psoParAd = pso(@"q_parity_adam_dec_kernel");
        psoParLE = pso(@"q_parity_logexp_kernel");
        runParity();
        // SH (mode x bound) + SH Adam (mode x pair x enc/dec) + non-SH Adam (mode x (16 enc + 16 dec + 12 production enc
        // + 1 production dec)) + q_log/q_exp
        const int expectN = 3 * 6 + 3 * 16 * 2 + 3 * (16 * 2 + 12 + 1) + 1;
        printf("\ninventory: %d sections evaluated (expected %d)\n", qSectN, expectN);
        const bool ok = qSectFail == 0 && qSectN == expectN;
        printf("summary --mode parity: %d FAIL sections -> %s\n", qSectFail + (qSectN != expectN), ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    psoBwd = pso(@"project_and_sh_backward_kernel"); psoAdam = pso(@"fused_adam_kernel");
    psoFwd = pso(@"project_and_sh_forward_kernel");
    psoBwdQ = pso(@"project_and_sh_backward_q"); psoAdamQ = pso(@"fused_adam_q16");
    if (const char *dp = std::getenv("Q_DEBUG_CELLS")) {
        g_dbg = fopen(dp, "w");
        if (!g_dbg) { printf("FAIL: cannot open Q_DEBUG_CELLS file %s\n", dp); return 1; }
        psoParAe = pso(@"q_parity_adam_enc_kernel");
        printf("Q_DEBUG_CELLS: cell dump to %s (side output)\n", dp);
    }
    psoFwdQ = pso(@"project_and_sh_forward_q"); psoRegQ = pso(@"mcmc_reg_grad_q");
    printf("Q PSO maxThreads: bwd_q %lu adam_q16 %lu fwd_q %lu reg_q %lu\n", (unsigned long)psoBwdQ.maxTotalThreadsPerThreadgroup,
           (unsigned long)psoAdamQ.maxTotalThreadsPerThreadgroup, (unsigned long)psoFwdQ.maxTotalThreadsPerThreadgroup,
           (unsigned long)psoRegQ.maxTotalThreadsPerThreadgroup);
    int expectN = 0;
    for (int m = 0; m < 3; m++) {
        printf("\n=== mode %s ===\n", MODE_NAME[m]);
        const QRes f1 = runQ("f1", 65536, 70, m), f2 = runQ("f2", 65436, 10, m);
        reportQ(f1); reportQ(f2);
        printf("\n--- coverage (s4.2, Q-arm growth events), mode %s ---\n", MODE_NAME[m]);
        qCoverage(f1.RQ, f2.RQ);
        // per mode: F1 19 codes + 10 other + 8 forward; F2 19 codes + 11 other (with padding); coverage 57
        expectN += (NCOMP + 10 + 8) + (NCOMP + 11) + (3 + 6 + 3 + 6 + 2 + 2 + 10 + 6 + 9 + 8 + 2);
    }
    if (g_dbg) {
        fprintf(g_dbg, "LS_SUMMARY cells %d: GPU codec on the reference's (m, v) reproduces code_Q %d; +-1 ulp of the CPU log_s "
                "reproduces code_Q %d; both %d; double-precision log_s gives code_Q %d\n", g_dbgLs, g_dbgLsRepro, g_dbgLsUlp, g_dbgLsBoth,
                g_dbgLsDblQ);
        fclose(g_dbg);
    }
    printf("\ninventory: %d sections evaluated (expected %d)\n", qSectN, expectN);
    const bool ok = qSectFail == 0 && qSectN == expectN && fails == 0;
    printf("summary --mode q: %d FAIL sections, %d invariant failures (reference-arm encAdam) -> %s\n", qSectFail + (qSectN != expectN), fails,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        std::string mode, fixture, expect, metallib = "build-q/default.metallib";
        for (int i = 1; i + 1 < argc; i += 2) {
            std::string k = argv[i];
            if (k == "--mode") mode = argv[i + 1];
            else if (k == "--fixture") fixture = argv[i + 1];
            else if (k == "--metallib") metallib = argv[i + 1];
            else if (k == "--expect-manifest") expect = argv[i + 1];
        }
        if ((mode == "q" || mode == "parity") && !fixture.empty() && expect.size() == 64)
            return runQModes(mode, fixture, expect, metallib);
        if ((mode != "reference" && mode != "mutant-check") || fixture.empty() || expect.size() != 64) {
            fprintf(stderr, "usage: q_gpu_harness --mode reference|mutant-check|q --fixture DIR --expect-manifest SHA256 [--metallib PATH]\n");
            return 2;
        }
        const bool mutant = mode == "mutant-check";
        printf("q_gpu_harness --mode %s (P2-M1; unchanged fp32 kernels + CPU Q encoding; not a gate)\n", mode.c_str());
        if (!loadFixture(fixture, expect)) { printf("FAIL: fixture identity\n"); return 1; }
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("FAIL: no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib.c_str())] error:&err];
        if (!lib) { printf("FAIL: cannot load %s\n", metallib.c_str()); return 1; }
        auto pso = [&](NSString *name) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (!p) { printf("FAIL: pipeline %s\n", name.UTF8String); exit(1); }
            return p;
        };
        psoBwd = pso(@"project_and_sh_backward_kernel");
        psoAdam = pso(@"fused_adam_kernel");
        queue = [dev newCommandQueue];
        printf("device: %s; metallib: %s\n", dev.name.UTF8String, metallib.c_str());

        std::vector<Run> runs;
        for (int m = 0; m < 3; m++) {
            runs.push_back(runFixture("f1", 65536, 70, m, mutant));
            runs.push_back(runFixture("f2", 65436, 10, m, mutant));
        }

        if (mutant) {
            bool all = true;
            printf("\nmutation check (candidate = reference with inactive rows' SH Adam + DC Adam codes zeroed; comparator = "
                   "compareInactiveRows, the s4.2 inactive-row rule):\n");
            for (auto &R : runs) {
                std::string st;
                for (int t : R.mutSteps) st += " " + std::to_string(t);
                printf("  %s %-15s flagged cells %ld of %ld inactive-row cells checked; steps with flags %zu:%s\n", R.fx.c_str(),
                       MODE_NAME[R.mode], R.mutFlagged, R.mutCells, R.mutSteps.size(), st.c_str());
                all &= R.mutFlagged > 0;
            }
            printf("\nreference invariant failures: %d\n", fails);
            printf("mutation check: %s\n", all ? "PASS (the comparator flags the mutant in every run)"
                                               : "NOT FLAGGED in at least one run (mutation check FAIL)");
            return all && !fails ? 0 : 1;
        }

        // ---------- inventory ----------
        int missing = 0;
        auto req = [&](bool ok, const char *fmt, auto... a) {
            printf("  %s ", ok ? "[ok]  " : "[MISS]");
            printf(fmt, a...);
            printf("\n");
            missing += !ok;
        };
        for (int m = 0; m < 3; m++) {
            const Run &f1 = runs[m * 2], &f2 = runs[m * 2 + 1];
            printf("\n=== mode %s ===\n", MODE_NAME[m]);
            for (const Run *R : {&f1, &f2}) {
                printf("%s growth events per component (blocks; from-zero / from-positive):\n", R->fx.c_str());
                for (int c = 0; c < NCOMP; c++)
                    printf("  %-10s %3d / %3d   Z %d/4  N %d\n", CN[c], countBlocks(*R, c, -1, 1, nullptr),
                           countBlocks(*R, c, -1, 0, nullptr), countBlocks(*R, c, -1, -1, isZ), countBlocks(*R, c, -1, -1, isN));
                printf("%s changed SH value codes per step [d0 d1 d2] (dtu):", R->fx.c_str());
                for (int t = 0; t < R->steps; t++)
                    printf("%s t%d(%u) %ld %ld %ld", t % 5 ? "" : "\n   ", t + 1, R->dtu[t], R->shchg[t][0], R->shchg[t][1], R->shchg[t][2]);
                printf("\n%s qsat %ld; max bounds: SH a %.4g, Adam a_u %.4g, required h %.4g\n", R->fx.c_str(), R->qsat, R->maxA,
                       R->maxAu, R->maxH);
            }
            printf("coverage (s4.2), mode %s:\n", MODE_NAME[m]);
            // Z: growth from zero in Z blocks, every component; SH value at its degree's opening step (F1)
            const int open[3] = {11, 21, 31};
            for (int d = 0; d < 3; d++) {
                int n = 0, first = 1 << 30;
                for (int b = 0; b < 4; b++) { int s = firstStep(f1, d, b, 1); if (s > 0) { n++; first = std::min(first, s); } }
                req(n >= 1 && first == open[d], "Z shv_d%d from zero: %d/4 Z blocks, first t=%d (degree opens t=%d)", d, n, n ? first : -1, open[d]);
            }
            for (int c = 3; c <= 8; c++)  // SH Adam (non-SH / DC: fixed bounds, events below)
                req(countBlocks(f1, c, -1, 1, isZ) >= 1, "Z %s from zero: F1 %d/4 Z blocks, F2 %d/4", CN[c],
                    countBlocks(f1, c, -1, 1, isZ), countBlocks(f2, c, -1, 1, isZ));
            // P: SH value blocks 11-13 from positive; SH Adam block 7 in t = 41-70
            for (int d = 0; d < 3; d++) {
                int s = firstStep(f1, d, 11 + d, 0), n = 0;
                for (auto &e : f1.ev) n += e.comp == d && e.blk == 11 + d && !e.fromZero;
                req(s >= open[d], "P shv_d%d block %d from positive bound (after degree opens, t=%d): %d growth steps, first t=%d",
                    d, 11 + d, open[d], n, s);
            }
            for (int c = 3; c <= 8; c++) {
                const int blk = 7;
                std::string steps;
                int n = 0;
                for (auto &e : f1.ev)
                    if (e.comp == c && e.blk == blk && e.t >= 41 && !e.fromZero) { n++; steps += " " + std::to_string(e.t); }
                req(n >= 1, "P %s block %d from positive bound in t=41-70: %d growth steps (first listed), at t =%s", CN[c], blk, n, steps.c_str());
            }
            // N: never grow, codes constant (F1 and F2)
            for (const Run *R : {&f1, &f2}) {
                int g = 0;
                for (int c = 0; c < NCOMP; c++) g += countBlocks(*R, c, -1, -1, isN);
                req(g == 0 && R->nCodeChanges == 0, "N blocks 100-109 %s: %d growth (block, component) pairs, %ld steps with a code/bound change",
                    R->fx.c_str(), g, R->nCodeChanges);
            }
            // every open degree, every step: changed SH codes > 0
            for (const Run *R : {&f1, &f2}) {
                long mn[3] = {-1, -1, -1};
                int miss = 0;
                for (int t = 0; t < R->steps; t++)
                    for (int d = 0; d < (int)R->dtu[t]; d++) {
                        if (mn[d] < 0 || R->shchg[t][d] < mn[d]) mn[d] = R->shchg[t][d];
                        miss += R->shchg[t][d] == 0;
                    }
                req(miss == 0, "%s changed SH codes > 0 in every open degree and step: min d0 %ld d1 %ld d2 %ld", R->fx.c_str(), mn[0], mn[1], mn[2]);
            }
            // non-SH and DC Adam (fixed bounds, attempt 2): no clamped value, and non-zero codes exist (F1, F2)
            for (int c = 9; c < NCOMP; c++)
                req(f1.sat[c] == 0 && f2.sat[c] == 0 && f1.nzMax[c] > 0 && f2.nzMax[c] > 0,
                    "%s fixed bounds: clamped (%s) F1 %ld F2 %ld; non-zero code cells (max over steps) F1 %ld F2 %ld", CN[c],
                    c % 2 ? "|u| > 8" : "log_s > 64", f1.sat[c], f2.sat[c], f1.nzMax[c], f2.nzMax[c]);
            // SH Adam from a zero bound: F1 at t = 11/21/31, F2 at t = 1
            for (int c = 3; c <= 8; c++) {
                const int d = (c - 3) % 3, t1 = open[d];
                req(countBlocks(f1, c, t1, 1, nullptr) >= 1 && countBlocks(f2, c, 1, 1, nullptr) >= 1,
                    "%s from zero: F1 t=%d %d blocks, F2 t=1 %d blocks", CN[c], t1, countBlocks(f1, c, t1, 1, nullptr),
                    countBlocks(f2, c, 1, 1, nullptr));
            }
            // (a) late growth with a switched-inactive row holding a non-zero code (F1), per component
            for (int c : {0, 1, 2, 3, 4, 5, 6, 7, 8})  // DC: fixed bound, never grows, so (a) does not apply (attempt 2)
                req(f1.lateCov[c] >= 1, "(a) %s: late growth events %ld, covered by a switched-inactive non-zero row %ld "
                    "(blocks 11-13 %ld, 7 %ld, Z %ld, other %ld)", CN[c], f1.lateGrow[c], f1.lateCov[c], f1.lateCovGrp[c][0],
                    f1.lateCovGrp[c][1], f1.lateCovGrp[c][2], f1.lateCovGrp[c][3]);
            // (b) switched-inactive rows with non-zero SH/DC Adam codes in a non-growing block (byte-identity checked)
            for (int c : {3, 4, 5, 6, 7, 8, 15, 16})
                req(f1.stableNZ[c] >= 1, "(b) %s: switched-inactive non-zero (row, step) pairs in non-growing blocks %ld", CN[c], f1.stableNZ[c]);
            for (const Run *R : {&f1, &f2})
                printf("  %s inactive-row rule cells checked %ld; max switched-inactive rows in a step %ld\n", R->fx.c_str(),
                       R->inactCells, R->switchedMax);
            // inactive rows in every role block; qsat = 0
            for (const Run *R : {&f1, &f2}) {
                int mn = 1 << 30;
                for (int v : R->inactPerRole) mn = std::min(mn, v);
                req(mn >= 1, "%s inactive rows in every role block (0-3, 9, 100-109; blocks 7, 11-13 all active by decision): min %d", R->fx.c_str(), mn);
                req(R->qsat == 0, "%s qsat = %ld", R->fx.c_str(), R->qsat);
            }
        }
        printf("\nmax over runs: log_s %.4g, required h %.4g, a_u %.4g, SH a %.4g\n", [&] { float x = 0; for (auto &r : runs) x = std::max(x, r.maxLs); return x; }(), [&] { float x = 0; for (auto &r : runs) x = std::max(x, r.maxH); return x; }(),
               [&] { float x = 0; for (auto &r : runs) x = std::max(x, r.maxAu); return x; }(), [&] { float x = 0; for (auto &r : runs) x = std::max(x, r.maxA); return x; }());
        printf("\nsummary: %d required coverage events missing, %d invariant failures -> %s\n", missing, fails,
               missing || fails ? "FAIL" : "all events present");
        return missing || fails ? 1 : 0;
    }
}
