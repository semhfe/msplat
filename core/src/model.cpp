#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include "model.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double mse = 0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - g[i]; mse += d * d; }
    mse /= n;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::abs(r[i] - g[i]);
    return (float)(sum / n);
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int capMax, float noiseLr, float opacityReg, float scaleReg, float anisoReg,
    int maxSteps, bool keepCrs,
    const float* bgColor)
    : numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), shDegreeInterval(shDegreeInterval),
      maxSteps(maxSteps), keepCrs(keepCrs),
      cap_max(capMax), noise_lr(noiseLr),
      opacity_reg(opacityReg), scale_reg(scaleReg), aniso_reg(anisoReg) {

    int64_t numPoints = inputData.points.count;
    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), inputData.points.xyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KD-tree nearest neighbor distances, log'd, repeated 3x
    {
        PointsTensor pt(inputData.points.xyz.data(), numPoints);
        auto sc = pt.scales();  // vector<float> of length numPoints
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float v = std::log(sc[i]);
            sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
        }
    }

    // Random quaternions
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float u = dist(rng), v = dist(rng), w = dist(rng);
            qp[i*4+0] = std::sqrt(1-u) * std::sin(2*M_PI*v);
            qp[i*4+1] = std::sqrt(1-u) * std::cos(2*M_PI*v);
            qp[i*4+2] = std::sqrt(u) * std::sin(2*M_PI*w);
            qp[i*4+3] = std::sqrt(u) * std::cos(2*M_PI*w);
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        const uint8_t *rgb = inputData.points.rgb.data();
        for (int64_t i = 0; i < numPoints; i++) {
            for (int c = 0; c < 3; c++)
                dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // Opacities: logit(0.5) = 0.0 — MCMC initialization
    {
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < numPoints; i++) op[i] = 0.0f;
    }

    // Background color — default is magenta (high-contrast against typical scenes,
    // makes under-reconstructed regions obvious during training)
    backgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.6130f, 0.0101f, 0.3984f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();


    num_active = means.size(0);
    buf_capacity = cap_max;

    // Bug fix: when COLMAP produces more 3D points than cap_max, clamp so the
    // memcpy in allocBuf never writes past the cap_max-sized buffer.
    int64_t rows_to_copy = std::min((int64_t)num_active, (int64_t)cap_max);
    if (num_active > cap_max) {
        std::cerr << "[mcmc] init points (" << num_active
                  << ") exceed cap_max=" << cap_max
                  << "; truncating to cap_max\n";
        num_active = cap_max;
    }

    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        // Copy only rows_to_copy rows (param may have more rows than cap_max).
        size_t row_bytes = param.nbytes() / (size_t)param.size(0);
        memcpy(buf.data_ptr(), param.data_ptr(), (size_t)rows_to_copy * row_bytes);
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    static constexpr float lr_init[] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f};
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    adam_step_count = 0;
    means_lr_init = 0.00016f;
    means_lr_final = 0.0000016f;

    // MCMC: preallocated noise buffer for SGLD perturbation (reused every step)
    sgld_noise_buf = gpu_empty({(int64_t)cap_max * 3}, DType::Float32);

    // Pre-allocate relocation scratch to avoid repeated heap allocs at step %100
    scratch_probs_.reserve(cap_max);
    scratch_count_.reserve(cap_max);

    refreshViews();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    sgld_noise_buf.reset();
}

void Model::schedulersStep(int step){
    float t = std::clamp((float)step / (float)maxSteps, 0.f, 1.f);
    adam_lr[0] = std::exp(std::log(means_lr_init) * (1.f - t) + std::log(means_lr_final) * t);
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

// ── MCMC helpers ────────────────────────────────────────────────────────────

// Equation 9 of the MCMC paper: when N samples are drawn from a single Gaussian,
// reduce each clone's opacity/scale so that the ensemble preserves total weight.
void Model::computeRelocation(float opacity_old_sig, float* scale_old_exp, int N,
                              float& opacity_new_sig, float* scale_new_exp) {
    opacity_new_sig = 1.0f - std::pow(1.0f - opacity_old_sig, 1.0f / (float)N);
    float s = std::sqrt(1.0f / (float)N);
    for (int i = 0; i < 3; i++) scale_new_exp[i] = scale_old_exp[i] * s;
}

// Move dead Gaussians onto alive ones (weighted by opacity) and redistribute
// opacity/scale so total weight is preserved.
void Model::relocate(const std::vector<int>& dead, const std::vector<int>& alive) {
    if (dead.empty() || alive.empty()) return;

    float *mbuf = means_buf.data<float>();
    float *sbuf = scales_buf.data<float>();
    float *qbuf = quats_buf.data<float>();
    float *fdcbuf = featuresDc_buf.data<float>();
    float *frbuf = featuresRest_buf.data<float>();
    float *obuf = opacities_buf.data<float>();
    int64_t fr_stride = featuresRest_buf.stride0();

    auto zeroAdamAt = [&](int idx) {
        for (int g = 0; g < N_ADAM_GROUPS; g++) {
            int64_t st = adam_exp_avg_buf[g].stride0();
            memset(adam_exp_avg_buf[g].data<float>() + idx * st, 0, st * sizeof(float));
            memset(adam_exp_avg_sq_buf[g].data<float>() + idx * st, 0, st * sizeof(float));
        }
    };

    // Weighted sample of source alive-indices, one per dead slot.
    scratch_probs_.resize(alive.size());
    for (size_t i = 0; i < alive.size(); i++)
        scratch_probs_[i] = 1.0f / (1.0f + std::exp(-obuf[alive[i]]));

    std::mt19937 rng((unsigned)(0x9E3779B9u ^ (unsigned)(++mcmc_relocation_count)));
    std::discrete_distribution<int> dist(scratch_probs_.begin(), scratch_probs_.end());

    scratch_count_.assign(alive.size(), 0);
    std::vector<int> chosen_alive(dead.size());
    for (size_t i = 0; i < dead.size(); i++) {
        int a = dist(rng);
        chosen_alive[i] = a;
        scratch_count_[a]++;
    }

    // Apply Eq. 9 to each source with count > 0 (writes updated opacity/scale
    // back to source so dead slots can copy the post-relocation values).
    for (size_t a = 0; a < alive.size(); a++) {
        if (scratch_count_[a] == 0) continue;
        int src = alive[a];
        int N = scratch_count_[a];

        float op_old_sig = 1.0f / (1.0f + std::exp(-obuf[src]));
        float s_old_exp[3] = {
            std::exp(sbuf[src*3]), std::exp(sbuf[src*3+1]), std::exp(sbuf[src*3+2])
        };
        float op_new_sig;
        float s_new_exp[3];
        computeRelocation(op_old_sig, s_old_exp, N, op_new_sig, s_new_exp);

        // Clamp to avoid log(0)/division-by-zero at extremes.
        op_new_sig = std::clamp(op_new_sig, 1e-6f, 1.0f - 1e-6f);
        obuf[src] = std::log(op_new_sig / (1.0f - op_new_sig));
        for (int j = 0; j < 3; j++) sbuf[src*3+j] = std::log(std::max(s_new_exp[j], 1e-10f));
        zeroAdamAt(src);
    }

    // Copy source params into each dead slot.
    for (size_t i = 0; i < dead.size(); i++) {
        int dst = dead[i];
        int src = alive[chosen_alive[i]];
        memcpy(mbuf + dst*3,   mbuf + src*3,   3 * sizeof(float));
        memcpy(qbuf + dst*4,   qbuf + src*4,   4 * sizeof(float));
        memcpy(fdcbuf + dst*3, fdcbuf + src*3, 3 * sizeof(float));
        memcpy(frbuf + dst*fr_stride, frbuf + src*fr_stride, fr_stride * sizeof(float));
        obuf[dst] = obuf[src];
        memcpy(sbuf + dst*3, sbuf + src*3, 3 * sizeof(float));
        zeroAdamAt(dst);
    }
}

// Grow num_active toward cap_max by ~5% per call.
int Model::addNewGaussians() {
    int target = std::min(cap_max, (int)(1.05f * (float)num_active));
    int num_new = target - num_active;
    if (num_new <= 0) return 0;

    float *mbuf = means_buf.data<float>();
    float *sbuf = scales_buf.data<float>();
    float *qbuf = quats_buf.data<float>();
    float *fdcbuf = featuresDc_buf.data<float>();
    float *frbuf = featuresRest_buf.data<float>();
    float *obuf = opacities_buf.data<float>();
    int64_t fr_stride = featuresRest_buf.stride0();

    auto zeroAdamAt = [&](int idx) {
        for (int g = 0; g < N_ADAM_GROUPS; g++) {
            int64_t st = adam_exp_avg_buf[g].stride0();
            memset(adam_exp_avg_buf[g].data<float>() + idx * st, 0, st * sizeof(float));
            memset(adam_exp_avg_sq_buf[g].data<float>() + idx * st, 0, st * sizeof(float));
        }
    };

    scratch_probs_.resize(num_active);
    for (int i = 0; i < num_active; i++)
        scratch_probs_[i] = 1.0f / (1.0f + std::exp(-obuf[i]));

    std::mt19937 rng((unsigned)(0xA24BAED4u ^ (unsigned)(++mcmc_relocation_count)));
    std::discrete_distribution<int> dist(scratch_probs_.begin(), scratch_probs_.end());

    scratch_count_.assign(num_active, 0);
    std::vector<int> chosen_src(num_new);
    for (int i = 0; i < num_new; i++) {
        int s = dist(rng);
        chosen_src[i] = s;
        scratch_count_[s]++;
    }

    // Apply Eq. 9 to each source with count > 0 (source's N counts itself via
    // each sample drawn — the new slots then copy its post-relocation values).
    for (int src = 0; src < num_active; src++) {
        if (scratch_count_[src] == 0) continue;
        int N = scratch_count_[src];

        float op_old_sig = 1.0f / (1.0f + std::exp(-obuf[src]));
        float s_old_exp[3] = {
            std::exp(sbuf[src*3]), std::exp(sbuf[src*3+1]), std::exp(sbuf[src*3+2])
        };
        float op_new_sig;
        float s_new_exp[3];
        computeRelocation(op_old_sig, s_old_exp, N, op_new_sig, s_new_exp);

        op_new_sig = std::clamp(op_new_sig, 1e-6f, 1.0f - 1e-6f);
        obuf[src] = std::log(op_new_sig / (1.0f - op_new_sig));
        for (int j = 0; j < 3; j++) sbuf[src*3+j] = std::log(std::max(s_new_exp[j], 1e-10f));
        zeroAdamAt(src);
    }

    for (int i = 0; i < num_new; i++) {
        int dst = num_active + i;
        int src = chosen_src[i];
        memcpy(mbuf + dst*3,   mbuf + src*3,   3 * sizeof(float));
        memcpy(qbuf + dst*4,   qbuf + src*4,   4 * sizeof(float));
        memcpy(fdcbuf + dst*3, fdcbuf + src*3, 3 * sizeof(float));
        memcpy(frbuf + dst*fr_stride, frbuf + src*fr_stride, fr_stride * sizeof(float));
        obuf[dst] = obuf[src];
        memcpy(sbuf + dst*3, sbuf + src*3, 3 * sizeof(float));
        zeroAdamAt(dst);
    }

    num_active += num_new;
    return num_new;
}

void Model::mcmcAfterTrain(int step) {
    if (step < 500 || step % 100 != 0) return;
    msplat_gpu_sync();

    const float *op = opacities.data<float>();
    const float *mn = means.data<float>();
    const float r2 = cull_radius > 0.0f ? cull_radius * cull_radius : -1.0f;
    std::vector<int> dead, alive;
    dead.reserve(num_active / 8);
    alive.reserve(num_active);
    int culled = 0;
    for (int i = 0; i < num_active; i++) {
        float sig = 1.0f / (1.0f + std::exp(-op[i]));
        bool is_dead = (sig <= 0.005f);
        if (!is_dead && r2 > 0.0f) {
            float x = mn[i*3 + 0], y = mn[i*3 + 1], z = mn[i*3 + 2];
            if (x*x + y*y + z*z > r2) { is_dead = true; culled++; }
        }
        if (is_dead) dead.push_back(i);
        else         alive.push_back(i);
    }

    int dead_count = (int)dead.size();
    if (!dead.empty() && !alive.empty()) relocate(dead, alive);
    int grown = addNewGaussians();

    if (dead_count > 0 || grown > 0) refreshViews();

    // Throttle stdout to every 1000 steps — matches the step-timer output cadence.
    if (step % 1000 == 0) {
        std::cout << "MCMC step=" << step
                  << " active=" << num_active
                  << " dead=" << dead_count
                  << " culled=" << culled
                  << " grown=" << grown << std::endl;
    }
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step);
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSplat(filename, p);
}

int Model::loadPly(const std::string &filename){
    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
static constexpr uint32_t CKPT_VERSION = 1;

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static MTensor readTensor(std::ifstream &f) {
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++)
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    uint64_t bytes;
    f.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    MTensor t = gpu_empty(shape, DType::Float32);
    f.read(reinterpret_cast<char*>(t.data_ptr()), bytes);
    return t;
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg_sq[g]);

    f.close();
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians, "
              << fs::file_size(filename) / (1024*1024) << " MB)" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file: " + filename);

    // Header
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != CKPT_MAGIC) throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version != CKPT_VERSION) throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));

    // Scalar state
    uint32_t step, numPts, shDeg, adamSteps;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&numPts), sizeof(numPts));
    f.read(reinterpret_cast<char*>(&shDeg), sizeof(shDeg));
    f.read(reinterpret_cast<char*>(&adamSteps), sizeof(adamSteps));

    f.read(reinterpret_cast<char*>(adam_lr), sizeof(adam_lr));
    f.read(reinterpret_cast<char*>(&means_lr_init), sizeof(means_lr_init));
    f.read(reinterpret_cast<char*>(&means_lr_final), sizeof(means_lr_final));
    adam_step_count = (int)adamSteps;

    // Gaussian parameters — read into fresh tensors
    means = readTensor(f);
    scales = readTensor(f);
    quats = readTensor(f);
    featuresDc = readTensor(f);
    featuresRest = readTensor(f);
    opacities = readTensor(f);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg[g] = readTensor(f);
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg_sq[g] = readTensor(f);

    f.close();

    // Rebuild backing buffers with loaded data (don't call setupOptimizers —
    // it would zero the optimizer state we just loaded)
    num_active = (int)numPts;
    buf_capacity = cap_max;

    // Copy gaussian params into oversized backing buffers
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    // Copy optimizer state into oversized backing buffers
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = adam_exp_avg[g].shape();
        shape[0] = buf_capacity;
        MTensor avg_buf = gpu_zeros(shape, DType::Float32);
        MTensor sq_buf = gpu_zeros(shape, DType::Float32);
        memcpy(avg_buf.data_ptr(), adam_exp_avg[g].data_ptr(), adam_exp_avg[g].nbytes());
        memcpy(sq_buf.data_ptr(), adam_exp_avg_sq[g].data_ptr(), adam_exp_avg_sq[g].nbytes());
        adam_exp_avg_buf[g] = avg_buf;
        adam_exp_avg_sq_buf[g] = sq_buf;
    }

    // MCMC: preallocated noise buffer for SGLD perturbation
    sgld_noise_buf = gpu_empty({(int64_t)cap_max * 3}, DType::Float32);

    refreshViews();

    std::cout << "Checkpoint loaded: " << filename << " (step " << step
              << ", " << num_active << " gaussians)" << std::endl;

    return (int)step;
}

Model::CamSetup Model::prepareCam(Camera& cam, int step) {
    const float sf = getDownscaleFactor(step);
    CamSetup s;
    s.fx = cam.fx / sf; s.fy = cam.fy / sf;
    s.cx = cam.cx / sf; s.cy = cam.cy / sf;
    s.height = static_cast<int>(cam.height / sf);
    s.width = static_cast<int>(cam.width / sf);

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    if (!cam.cachedViewMat.defined() || cam.cachedFovX != fovX || cam.cachedFovY != fovY) {
        const float *d = cam.camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] = d[i*4+0]; R[i][1] = -d[i*4+1]; R[i][2] = -d[i*4+2]; T[i] = d[i*4+3];
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++) Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);
        float vm[16] = { Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1 };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        cam.cachedViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0] = T[0]; cam.cachedCamPos[1] = T[1]; cam.cachedCamPos[2] = T[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
    }

    s.degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

MTensor Model::render(Camera& cam, int step){
    auto s = prepareCam(cam, step);
    return msplat_render(
        means.size(0), means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor);
}

void Model::fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight){
    auto s = prepareCam(cam, step);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    // Initialize SSIM window (once)
    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
    MTensor adam_p[N_ADAM_GROUPS];
    MTensor adam_ea[N_ADAM_GROUPS], adam_eas[N_ADAM_GROUPS];
    float adam_ss[N_ADAM_GROUPS], adam_bc2s[N_ADAM_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    float lossInvN = 1.0f / (float)(s.height * s.width * 3);

    auto [r, loss] = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor, gt, window2d, ssimWeight,
        lossInvN, (int)featuresRest.size(-2),
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps);

    radii = r;

    // MCMC: SGLD noise perturbation, then post-Adam opacity/scale regularization.
    // Noise is generated on GPU (PCG+Box-Muller, seeded by step) — previously this
    // was a CPU mt19937 loop costing ~10–50 ms/iter at 1M splats.
    msplat_sgld_noise_gen(num_active, sgld_noise_buf, (uint32_t)step);
    msplat_sgld_noise(num_active, means, scales, quats, opacities,
                      sgld_noise_buf, noise_lr, adam_lr[0]);
    msplat_mcmc_regularization(num_active, opacities, scales,
                               adam_lr[0], opacity_reg, scale_reg, aniso_reg);
}
