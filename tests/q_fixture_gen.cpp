// Q codec fixture generator (P2-M1; runs/2026-11-01-hiz-q-kodek-onkayit.md s3 + s4.1). CPU only.
// Build: xcrun clang++ -std=c++17 -O2 -ffp-contract=off -Icore/include tests/q_fixture_gen.cpp -o build-q/q_fixture_gen
// Run:   build-q/q_fixture_gen [out_dir]   (default /Volumes/fe-ssd-2/hiz_q_fixture_2026-11-01-v7)
//
// PRNG: splitmix64, state seed = 0x51A5E5D1 + stream index. Uniform u = (x >> 11) * 2^-53 (double).
// Normal: Box-Muller in double on consecutive uniforms (u1, u2): r = sqrt(-2 ln(1 - u1)), z0 = r cos(2 pi u2),
// z1 = r sin(2 pi u2); z0 then z1 are consumed in order. value = (float)(sigma * z') with z' the role-adjusted draw.
// No std:: distributions.
//
// Stream-index assignment:
//   stream t, t = 1..70 : F1 gradient inputs of step t (N = 65536).
//   stream t, t = 1..10 : F2 gradient inputs of step t (N = 65436; drawn with F2's own N, so F2 is not a prefix of F1).
//     Draw order inside a step (normals): v_xy [N,2], v_depth [N], v_conic [N,3], v_colors [N,3], v_opacity [N].
//     sigma 1e-3, 1e-4, 1e-3, 1e-2, 1e-2. Every row is drawn (inactive rows too; the backward kernel skips them
//     except the dense opacity Adam), then N-block rows (blocks 100-109) are set to 0.
//     P rows (block 7: v_colors; block 9: v_xy, v_depth, v_conic, v_opacity), decided 1 Nov before freeze:
//     t = 1: +|n_1|; t = 2-40: 0, except block 7 v_colors = +|n_t| at t = 11, 21, 31 (degree openings);
//     t = 41-70: +|n_t| * 1e4. F2: t = 1 +|n_1|, t = 2-10 zero. (Replaces the alternating-sign plan, 1 Nov.)
//   Block 7 rows are all active (exempt from the hash mask and the z test); rows with camera z <= 0.2 are reported.
//   Block 7 SH bounds use the normal 1.25*max rule (v7).
//   Time-varying mask (v7): radii file is [steps][N]. Base mask (inactive at every step): hash32(i ^ 0xA5A5A5A5) % 5 == 0
//     or camera z <= 0.2, blocks 7 and 11-13 exempt. Switched subset: rows not base-inactive with
//     hash32(i ^ 0x5EED5EED) % 5 == 0, in every block (r* excepted), are active before and inactive from the switch step:
//     F1 t >= 32 in blocks 11-13, t >= 35 elsewhere; F2 t >= 6.
//   SH-value P (decided 1 Nov): blocks 11, 12, 13 for degree d = 0, 1, 2, all rows active. In block 11 + d the first
//     coefficient k* of degree d (k = 0, 3, 8), channel 0, of the first row r* whose initial forward colour (degree 3,
//     sh_coeffs_to_color of msplat_metal.metal, channel 0, with x* already set) satisfies raw + 0.5 >= 0.1 is set to
//     sign(x)*max|x| over degree d of the block (sign of 0 taken as +); degree d's bound is tight (a = max|x|).
//     r*'s v_colors channel 0 = |n_t| * s at every step, s = -sign(x*) * sign(basis_k*(dir_r*)), so the Adam step moves
//     x* outward. Other rows and channels of these blocks use the normal stream.
//   stream 7  : s3 SH value inputs (b): 1e5 uniforms u, x = (float)(a * (2u - 1)); the same 1e5 uniforms for every a.
//   stream 11 : s3 Adam inputs (a): 1e5 samples, uniforms in the order L, u per sample; the same sequence for every
//     (a_u, h) pair. L = h*u1, u = a_u*(2u2 - 1), sv = 1e-15*(exp(L) - 1), v = sv^2, m = u*(sv + 1e-15) in double.
//   Streams 7 and 11 are also the gradient streams of steps 7 and 11 (the spec names both; kept as written).
//
// Files (little endian, raw arrays; shapes in manifest_layout.txt), manifest.sha256 in `shasum -a 256` format.
#include "quant_codec.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace qcodec;

static const char *PLY_SHA = "3aacd0cfbf00dfa7e7bf2784b342942006fe2b0b1324350e45f6cb1f23df8f94";
static const int QB = 256, NREST = 15, ST = NREST * 3;

struct SplitMix {
    uint64_t s;
    explicit SplitMix(uint64_t stream) : s(0x51A5E5D1ull + stream) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (double)(next() >> 11) * 0x1.0p-53; }
    bool have = false; double spare = 0;
    double normal() {
        if (have) { have = false; return spare; }
        double u1 = uni(), u2 = uni();
        double r = std::sqrt(-2.0 * std::log(1.0 - u1)), th = 2.0 * M_PI * u2;
        spare = r * std::sin(th); have = true;
        return r * std::cos(th);
    }
};

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static std::string hex(const unsigned char *d) {
    char b[65];
    for (int i = 0; i < 32; i++) snprintf(b + 2 * i, 3, "%02x", d[i]);
    return std::string(b, 64);
}

static std::string outDir;
static std::string manifest, layout;

static void put(const std::string &name, const void *p, size_t bytes, const std::string &shape) {
    std::string path = outDir + "/" + name;
    FILE *f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, 1, bytes, f) != bytes || fclose(f) != 0) { fprintf(stderr, "write failed: %s\n", path.c_str()); exit(1); }
    unsigned char d[32];
    CC_SHA256(p, (CC_LONG)bytes, d);
    manifest += hex(d) + "  " + name + "\n";
    layout += name + "  " + shape + "  " + std::to_string(bytes) + " B\n";
}
template <class T> static void put(const std::string &name, const std::vector<T> &v, const std::string &shape) {
    put(name, v.data(), v.size() * sizeof(T), shape);
}

struct Ply { std::vector<float> means, normals, dc, rest, opac, scales, quats; };

// Reads the first n vertices; the file sha256 is checked over the whole file.
static Ply readPly(const std::string &path, int n) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    CC_SHA256_CTX ctx; CC_SHA256_Init(&ctx);
    std::vector<unsigned char> buf(1 << 22);
    size_t r;
    while ((r = fread(buf.data(), 1, buf.size(), f)) > 0) CC_SHA256_Update(&ctx, buf.data(), (CC_LONG)r);
    unsigned char d[32]; CC_SHA256_Final(d, &ctx);
    if (hex(d) != PLY_SHA) { fprintf(stderr, "PLY sha256 mismatch: %s (expected %s)\n", hex(d).c_str(), PLY_SHA); exit(1); }
    printf("ply sha256 ok %s\n", PLY_SHA);
    rewind(f);
    // header: expect x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3, all float
    char line[256]; int props = 0; long nv = 0; bool bin = false;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "format binary_little_endian", 27)) bin = true;
        if (!strncmp(line, "element vertex", 14)) nv = atol(line + 15);
        if (!strncmp(line, "property float", 14)) props++;
        else if (!strncmp(line, "property", 8)) { fprintf(stderr, "unexpected property %s", line); exit(1); }
        if (!strncmp(line, "end_header", 10)) break;
    }
    if (!bin || props != 62 || nv < n) { fprintf(stderr, "unexpected PLY header (bin %d props %d nv %ld)\n", bin, props, nv); exit(1); }
    Ply p;
    p.means.resize(n * 3); p.normals.resize(n * 3); p.dc.resize(n * 3); p.rest.resize((size_t)n * ST);
    p.opac.resize(n); p.scales.resize(n * 3); p.quats.resize(n * 4);
    float row[62];
    for (int i = 0; i < n; i++) {
        if (fread(row, 4, 62, f) != 62) { fprintf(stderr, "short PLY\n"); exit(1); }
        memcpy(&p.means[i * 3], row, 12);
        memcpy(&p.dc[i * 3], row + 6, 12);
        for (int ch = 0; ch < 3; ch++)           // PLY [3, 15] -> internal [15, 3] (save_gaussians.cpp loader)
            for (int b = 0; b < NREST; b++) p.rest[(size_t)i * ST + b * 3 + ch] = row[9 + ch * NREST + b];
        p.opac[i] = row[54];
        memcpy(&p.scales[i * 3], row + 55, 12);
        memcpy(&p.quats[i * 4], row + 58, 16);
    }
    fclose(f);
    return p;
}

static int degOf(int j) { int k = j / 3; return k < 3 ? 0 : k < 8 ? 1 : 2; }
static bool isN(int b) { return b >= 100 && b <= 109; }
static bool allActive(int b) { return b == 7 || (b >= 11 && b <= 13); }

// msplat_metal.metal constants and sh_coeffs_to_color (channel c, degree 3), same operation order, in float
static const float SH_C0 = 0.28209479177387814f, SH_C1 = 0.4886025119029199f;
static const float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
static const float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                               -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};
static void viewdir(const float *p, const float *cam, float &x, float &y, float &z) {
    float dx = p[0] - cam[36], dy = p[1] - cam[37], dz = p[2] - cam[38];
    float inv = 1.0f / std::sqrt(dx * dx + dy * dy + dz * dz);
    x = dx * inv; y = dy * inv; z = dz * inv;
}
static float shColor(float dc, const float *rc, int c, float x, float y, float z) {
    const int C = 3;
    float xx = x * x, xy = x * y, xz = x * z, yy = y * y, yz = y * z, zz = z * z;
    float col = SH_C0 * dc;
    col += SH_C1 * (-y * rc[0 * C + c] + z * rc[1 * C + c] - x * rc[2 * C + c]);
    col += (SH_C2[0] * xy * rc[3 * C + c] + SH_C2[1] * yz * rc[4 * C + c] + SH_C2[2] * (2.f * zz - xx - yy) * rc[5 * C + c] +
            SH_C2[3] * xz * rc[6 * C + c] + SH_C2[4] * (xx - yy) * rc[7 * C + c]);
    col += (SH_C3[0] * y * (3.f * xx - yy) * rc[8 * C + c] + SH_C3[1] * xy * z * rc[9 * C + c] +
            SH_C3[2] * y * (4.f * zz - xx - yy) * rc[10 * C + c] + SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) * rc[11 * C + c] +
            SH_C3[4] * x * (4.f * zz - xx - yy) * rc[12 * C + c] + SH_C3[5] * z * (xx - yy) * rc[13 * C + c] +
            SH_C3[6] * x * (xx - 3.f * yy) * rc[14 * C + c]);
    return col;
}
// basis value of the first coefficient of degree d (k = 0, 3, 8), as in project_and_sh_backward_kernel
static float basisFirst(int d, float x, float y, float z) {
    return d == 0 ? -SH_C1 * y : d == 1 ? SH_C2[0] * (x * y) : SH_C3[0] * y * (3.f * (x * x) - (y * y));
}

static void fixture(const Ply &ply, const char *tag, int N, int steps, const float *cam) {
    const int nblk = (N + QB - 1) / QB, rows = nblk * QB;
    auto sl = [&](const std::vector<float> &v, int d) { return std::vector<float>(v.begin(), v.begin() + (size_t)N * d); };
    std::string t(tag), sN = std::to_string(N);
    put(t + "_means.f32", sl(ply.means, 3), "float [" + sN + ",3]");
    put(t + "_scales.f32", sl(ply.scales, 3), "float [" + sN + ",3]");
    put(t + "_quats.f32", sl(ply.quats, 4), "float [" + sN + ",4]");
    put(t + "_dc.f32", sl(ply.dc, 3), "float [" + sN + ",3]");
    put(t + "_opac.f32", sl(ply.opac, 1), "float [" + sN + ",1]");

    // base activity mask: inactive <=> hash32(i ^ 0xA5A5A5A5) % 5 == 0 or camera z <= 0.2 (viewmat [I | (0,0,4)])
    std::vector<int32_t> radii(N);
    std::vector<float> conics((size_t)N * 3);
    int inH = 0, inZ = 0;
    for (int i = 0; i < N; i++) {
        const float *v = cam;  // row-major viewmat
        const float *p = &ply.means[i * 3];
        float zc = v[8] * p[0] + v[9] * p[1] + v[10] * p[2] + v[11];
        bool h = hash32((uint32_t)i ^ 0xA5A5A5A5u) % 5 == 0, z = zc <= 0.2f;
        inH += h; inZ += z;
        radii[i] = (h || z) && !allActive(i / QB) ? 0 : 3;  // blocks 7 (Adam P) and 11-13 (SH value P) always active
        if (allActive(i / QB) && z) printf("%s: block %d row %d has camera z %.4f <= 0.2 (kept active)\n", tag, i / QB, i, zc);
        conics[i * 3] = 0.1f; conics[i * 3 + 1] = 0.0f; conics[i * 3 + 2] = 0.1f;
    }
    // every role block needs inactive rows (block 7 excepted: all active) and active rows, for growth
    int roleBlocks[] = {0, 1, 2, 3, 7, 9, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    int minIn = 1 << 30, minAct = 1 << 30;
    for (int b : roleBlocks) {
        int in = 0, act = 0;
        for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++) (radii[r] ? act : in)++;
        if (b != 7) minIn = std::min(minIn, in);
        minAct = std::min(minAct, act);
    }
    int act = 0;
    for (int i = 0; i < N; i++) act += radii[i] != 0;
    printf("%s: N=%d blocks=%d active=%d inactive_hash=%d inactive_z=%d role_blocks_min_inactive=%d min_active=%d\n",
           tag, N, nblk, act, inH, inZ, minIn, minAct);
    if (minIn < 1 || minAct < 1) { fprintf(stderr, "%s: a role block lacks inactive or active rows\n", tag); exit(1); }
    put(t + "_conics.f32", conics, "float [" + sN + ",3]");

    // initial SH codes: Z blocks 0; bound 1.25*block max per degree (blocks 11-13 tight in degree b - 11); nearest encode
    std::vector<float> rest(ply.rest.begin(), ply.rest.begin() + (size_t)N * ST);
    for (int r = 0; r < std::min(N, 4 * QB); r++) for (int j = 0; j < ST; j++) rest[(size_t)r * ST + j] = 0.0f;
    // SH-value P cells
    int pRow[3] = {-1, -1, -1};
    float pSign[3] = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
        const int b = 11 + d, k = d == 0 ? 0 : d == 1 ? 3 : 8, j = k * 3;
        float mx = 0;
        for (int r = b * QB; r < (b + 1) * QB; r++)
            for (int jj = 0; jj < ST; jj++) if (degOf(jj) == d) mx = std::max(mx, std::fabs(rest[(size_t)r * ST + jj]));
        for (int r = b * QB; r < (b + 1) * QB && pRow[d] < 0; r++) {
            float rc[ST];
            memcpy(rc, &rest[(size_t)r * ST], sizeof rc);
            const float x0 = rc[j], xs = std::copysign(mx, x0 == 0.0f ? 1.0f : x0);
            rc[j] = xs;
            float vx, vy, vz;
            viewdir(&ply.means[r * 3], cam, vx, vy, vz);
            const float col = shColor(ply.dc[r * 3], rc, 0, vx, vy, vz);
            if (!(col + 0.5f >= 0.1f)) continue;
            const float bas = basisFirst(d, vx, vy, vz);
            if (bas == 0.0f) continue;
            pRow[d] = r;
            pSign[d] = -(xs > 0 ? 1.0f : -1.0f) * (bas > 0 ? 1.0f : -1.0f);
            rest[(size_t)r * ST + j] = xs;
            printf("%s: SH-value P block %d degree %d: r* = %d, k* = %d (channel 0, cell %d), x* %.6g -> %.6g (block max), "
                   "colour raw+0.5 = %.4f, basis %.6g (sign %+d), s = %+d\n", tag, b, d, r, k, j, x0, xs, col + 0.5f, bas,
                   bas > 0 ? 1 : -1, (int)pSign[d]);
        }
        if (pRow[d] < 0) { fprintf(stderr, "%s: no qualifying row in block %d\n", tag, b); exit(1); }
    }
    // per-step radii: base mask plus the switched subset (r* never switched)
    {
        std::vector<int32_t> rs((size_t)steps * N);
        std::vector<int> swPerBlock(nblk, 0);
        int nsw = 0;
        for (int i = 0; i < N; i++) {
            const int b = i / QB;
            const bool sw = radii[i] != 0 && hash32((uint32_t)i ^ 0x5EED5EEDu) % 5 == 0 && i != pRow[0] && i != pRow[1] && i != pRow[2];
            const int t0 = steps == 70 ? (b >= 11 && b <= 13 ? 32 : 35) : 6;
            nsw += sw; swPerBlock[b] += sw;
            for (int s = 1; s <= steps; s++) rs[(size_t)(s - 1) * N + i] = (sw && s >= t0) ? 0 : radii[i];
        }
        int mnsw = 1 << 30;
        for (int b = 0; b < nblk; b++) mnsw = std::min(mnsw, swPerBlock[b]);
        printf("%s: switched rows %d, min per block %d (blocks 11-13: %d %d %d; block 7: %d)\n", tag, nsw, mnsw, swPerBlock[11],
               swPerBlock[12], swPerBlock[13], swPerBlock[7]);
        if (mnsw < 1) { fprintf(stderr, "%s: a block has no switched rows\n", tag); exit(1); }
        put(t + "_radii.i32", rs, "int32 [" + std::to_string(steps) + "][" + sN + "] radii per step (0 inactive, 3 active)");
    }
    std::vector<int16_t> codes((size_t)rows * ST, 0);
    std::vector<float> bounds((size_t)nblk * 3, 0.0f);
    for (int b = 0; b < nblk; b++) {
        float mx[3] = {0, 0, 0};
        for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++)
            for (int j = 0; j < ST; j++) mx[degOf(j)] = std::max(mx[degOf(j)], std::fabs(rest[(size_t)r * ST + j]));
        for (int d = 0; d < 3; d++) bounds[b * 3 + d] = (b == 11 + d) ? mx[d] : GROW * mx[d];
        for (int r = b * QB; r < std::min(N, (b + 1) * QB); r++)
            for (int j = 0; j < ST; j++)
                codes[(size_t)r * ST + j] = (int16_t)encSym(rest[(size_t)r * ST + j], bounds[b * 3 + degOf(j)], SH_M);
    }
    put(t + "_fr_codes.i16", codes, "int16 [" + std::to_string(rows) + ",15,3]");
    put(t + "_fr_bounds.f32", bounds, "float [" + std::to_string(nblk) + ",3]");

    // degree schedule and gradient streams
    std::vector<uint32_t> dtu(steps);
    for (int s = 1; s <= steps; s++) dtu[s - 1] = steps == 70 ? (s <= 10 ? 0 : s <= 20 ? 1 : s <= 30 ? 2 : 3) : 3;
    put(t + "_degrees.u32", dtu, "uint32 [" + std::to_string(steps) + "] degrees_to_use for t = 1..");
    const size_t per = (size_t)N * 10;
    std::vector<float> g(per * steps);
    for (int s = 1; s <= steps; s++) {
        SplitMix rng((uint64_t)s);
        float *o = &g[per * (s - 1)];
        float *vxy = o, *vd = o + 2 * (size_t)N, *vc = o + 3 * (size_t)N, *vcol = o + 6 * (size_t)N, *vop = o + 9 * (size_t)N;
        auto fill = [&](float *dst, size_t n, double sigma, int dim, bool pColors) {
            for (size_t k = 0; k < n; k++) {
                int row = (int)(k / dim), b = row / QB;
                double z = rng.normal();
                if (isN(b)) z = 0.0;
                else if (pColors ? b == 7 : b == 9) {
                    const bool on = s == 1 || s >= 41 || (pColors && (s == 11 || s == 21 || s == 31));
                    z = on ? std::fabs(z) * (s >= 41 ? 1e4 : 1.0) : 0.0;
                }
                if (pColors && k % 3 == 0)
                    for (int d = 0; d < 3; d++) if (row == pRow[d]) z = std::fabs(z) * pSign[d];
                dst[k] = (float)(sigma * z);
            }
        };
        fill(vxy, 2 * (size_t)N, 1e-3, 2, false);
        fill(vd, (size_t)N, 1e-4, 1, false);
        fill(vc, 3 * (size_t)N, 1e-3, 3, false);
        fill(vcol, 3 * (size_t)N, 1e-2, 3, true);
        fill(vop, (size_t)N, 1e-2, 1, false);
    }
    put(t + "_grads.f32", g, "float [" + std::to_string(steps) + "][v_xy N*2 | v_depth N | v_conic N*3 | v_colors N*3 | v_opacity N], N=" + sN);
}

int main(int argc, char **argv) {
    outDir = argc > 1 ? argv[1] : "/Volumes/fe-ssd-2/hiz_q_fixture_2026-11-01-v7";
    mkdir(outDir.c_str(), 0755);
    const char *home = getenv("HOME");
    std::string plyPath = std::string(home ? home : "") + "/Movies/pilot_d2_2026-10-04/truck_d2_cap1000000_r1/truck_d2_cap1000000_r1.ply";
    Ply ply = readPly(plyPath, 65536);

    // camera: viewmat [I | (0,0,4)], fx = fy = 1000, cx 640, cy 360, 1280x720; projmat = Model::prepareCam formula
    float cam[42] = {};
    float vm[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 4, 0, 0, 0, 1};
    const float fx = 1000.f, fy = 1000.f, W = 1280.f, H = 720.f;
    float fovX = 2.0f * std::atan(W / (2.0f * fx)), fovY = 2.0f * std::atan(H / (2.0f * fy));
    float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
    float pm[16] = {0.001f / r_p, 0, 0, 0, 0, 0.001f / t_p, 0, 0, 0, 0, (1000.0f + 0.001f) / (1000.0f - 0.001f),
                    -1000.0f * 0.001f / (1000.0f - 0.001f), 0, 0, 1, 0};
    float pvm[16] = {};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) pvm[i * 4 + j] += pm[i * 4 + k] * vm[k * 4 + j];
    memcpy(cam, vm, 64); memcpy(cam + 16, pvm, 64);
    cam[32] = fx; cam[33] = fy; cam[34] = 640.f; cam[35] = 360.f;
    cam[36] = 0.f; cam[37] = 0.f; cam[38] = -4.f; cam[39] = 0.f;
    cam[40] = W; cam[41] = H;
    put("camera.f32", cam, sizeof cam, "float [42]: viewmat[16] row-major | projmat[16] | fx fy cx cy | cam_pos xyz 0 | width height");

    fixture(ply, "f1", 65536, 70, cam);
    fixture(ply, "f2", 65436, 10, cam);

    // s3 SH value inputs
    const float shA[6] = {1e-6f, 3e-3f, 0.37f, 1.0f, 2.5f, 1e3f};
    put("s3_sh_bounds.f32", shA, sizeof shA, "float [6]");
    std::vector<float> sa, sb, sc, sd(1000, 0.0f);
    {
        SplitMix r7(7);
        std::vector<double> u(100000);
        for (auto &x : u) x = r7.uni();
        const int cs[7] = {0, 1, -1, 2, -2, SH_M - 1, -(SH_M - 1)};
        for (float a : shA) {
            for (int c = -SH_M; c <= SH_M; c++) sa.push_back(decSym(c, a, SH_M));
            for (double x : u) sb.push_back((float)((double)a * (2.0 * x - 1.0)));
            for (int c : cs) for (double h : {-0.5, 0.5}) {
                float f = (float)((c + h) * (double)a / SH_M);
                sc.push_back(nextafterf(f, -INFINITY)); sc.push_back(nextafterf(f, INFINITY));
            }
        }
    }
    put("s3_sh_a.f32", sa, "float [6][65535] decSym(c, a) for c = -32767..32767");
    put("s3_sh_b.f32", sb, "float [6][100000] a*(2u-1), stream 7");
    put("s3_sh_c.f32", sc, "float [6][7 c][2 half -/+][2 ulp -/+], c = 0,1,-1,2,-2,32766,-32766");
    put("s3_sh_d.f32", sd, "float [1000] zeros");

    // s3 Adam inputs
    const float au[4] = {1e-3f, 0.1f, 1.0f, 10.0f}, hh[4] = {1.0f, 16.0f, 46.0f, 64.0f};
    std::vector<float> pairs, aa, ad(2000, 0.0f);
    for (float a : au) for (float h : hh) { pairs.push_back(a); pairs.push_back(h); }
    put("s3_adam_pairs.f32", pairs, "float [16][a_u, h], a_u-major");
    {
        SplitMix r11(11);
        std::vector<double> u(200000);
        for (auto &x : u) x = r11.uni();
        for (int p = 0; p < 16; p++) {
            double a = pairs[p * 2], h = pairs[p * 2 + 1];
            for (int i = 0; i < 100000; i++) {
                double L = h * u[2 * i], uu = a * (2.0 * u[2 * i + 1] - 1.0);
                double sv = 1e-15 * (std::exp(L) - 1.0);
                aa.push_back((float)(uu * (sv + 1e-15))); aa.push_back((float)(sv * sv));
            }
        }
    }
    put("s3_adam_a.f32", aa, "float [16 pairs][100000][m, v], stream 11");
    put("s3_adam_d.f32", ad, "float [1000][m, v] zeros");
    for (int sh = 1; sh >= 0; sh--) {
        const int Mu = sh ? 127 : 32767, Ml = sh ? 255 : 65535, stp = sh ? 1 : 7;
        std::vector<int32_t> b, c;
        for (int k = -Mu; k <= Mu; k += stp) { b.push_back(k); b.push_back(Ml / 2); }
        for (int k = 0; k <= Ml; k += stp) { c.push_back(Mu / 2); c.push_back(k); }
        const std::string s = sh ? "sh" : "nsh";
        put("s3_adam_b_" + s + ".i32", b, "int32 [" + std::to_string(b.size() / 2) + "][u code, log_s code]");
        put("s3_adam_c_" + s + ".i32", c, "int32 [" + std::to_string(c.size() / 2) + "][u code, log_s code]");
    }

    put("manifest_layout.txt", layout.data(), layout.size(), "text");
    FILE *f = fopen((outDir + "/manifest.sha256").c_str(), "w");
    if (!f) { fprintf(stderr, "manifest write failed\n"); return 1; }
    fputs(manifest.c_str(), f);
    fclose(f);
    printf("wrote %s/manifest.sha256\n", outDir.c_str());
    return 0;
}
