// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

namespace msplat {

// ── Dataset::Impl ───────────────────────────────────────────────────────────

struct Dataset::Impl {
    InputData data;
    std::vector<Camera> trainCams;
    std::vector<Camera> testCams;
};

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    impl->data = inputDataFromX(path);

    for (auto& cam : impl->data.cameras)
        cam.loadImage(downscaleFactor);

    if (evalMode) {
        auto split = impl->data.splitTrainTest(testEvery);
        impl->trainCams = std::get<0>(split);
        impl->testCams = std::get<1>(split);
    } else {
        auto t = impl->data.getCameras(false);
        impl->trainCams = std::get<0>(t);
    }
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainCams.size(); }
int Dataset::numTest() const { return (int)impl->testCams.size(); }
void Dataset::cameraPose(int index, float camToWorld[16]) const {
    if (index >= 0 && index < (int)impl->trainCams.size())
        memcpy(camToWorld, impl->trainCams[index].camToWorld, 16 * sizeof(float));
}
void* Dataset::_handle() const { return impl.get(); }

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;

    // Camera iteration
    std::vector<size_t> camIndices;
    size_t camIterPos = 0;
    std::mt19937 rng{42};

    void shuffleCameras() {
        std::shuffle(camIndices.begin(), camIndices.end(), rng);
        camIterPos = 0;
    }

    size_t nextCamera() {
        if (camIterPos >= camIndices.size()) shuffleCameras();
        return camIndices[camIterPos++];
    }
};

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
    impl->config = config;
    impl->ds = static_cast<Dataset::Impl*>(dataset._handle());

    impl->model = std::make_unique<Model>(
        impl->ds->data,
        (int)impl->ds->trainCams.size(),
        config.shDegree, config.shDegreeInterval,
        config.capMax, config.noiseLr, config.opacityReg, config.scaleReg,
        config.iterations, config.keepCrs,
        config.bgColor
    );
    impl->model->cull_radius = config.cullRadius;

    impl->camIndices.resize(impl->ds->trainCams.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
}

Trainer::~Trainer() = default;

Stats Trainer::step() {
    impl->currentStep++;
    size_t camIdx = impl->nextCamera();
    Camera& cam = impl->ds->trainCams[camIdx];

    int ds = impl->model->getDownscaleFactor(impl->currentStep);
    MTensor& gt = cam.getGPUImage();

    auto t0 = std::chrono::high_resolution_clock::now();

    impl->model->fullIteration(cam, impl->currentStep, gt, impl->config.ssimWeight);
    impl->model->schedulersStep(impl->currentStep);
    impl->model->mcmcAfterTrain(impl->currentStep);
    msplat_commit();

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = ms;
    return s;
}

void Trainer::train(int callbackEvery) {
    while (impl->currentStep < impl->config.iterations) {
        step();
        // Note: callbacks handled at the Swift level via polling iteration()
        // to keep the C++ API free of function pointer complexity
    }
}

EvalMetrics Trainer::evaluate() {
    auto& testCams = impl->ds->testCams;
    if (testCams.empty())
        return {};

    double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
    int n = (int)testCams.size();

    for (int i = 0; i < n; i++) {
        Camera& cam = testCams[i];
        MTensor rgb = impl->model->render(cam, impl->config.iterations);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        int dsf = impl->model->getDownscaleFactor(impl->config.iterations);
        MTensor gtCpu = dequantize_gt(cam.getGPUImage());

        sumPsnr += psnr(rgbCpu, gtCpu);
        sumSsim += ssim_eval(rgbCpu, gtCpu);
        sumL1 += l1_loss(rgbCpu, gtCpu);
    }

    EvalMetrics m;
    m.psnr = (float)(sumPsnr / n);
    m.ssim = (float)(sumSsim / n);
    m.l1 = (float)(sumL1 / n);
    m.numTest = n;
    m.numGaussians = (int)impl->model->means.size(0);
    return m;
}

PixelBuffer Trainer::render(int cameraIndex, bool useTest) {
    auto& cams = useTest ? impl->ds->testCams : impl->ds->trainCams;
    if (cameraIndex < 0 || cameraIndex >= (int)cams.size())
        return {};

    Camera& cam = cams[cameraIndex];
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    // Use malloc so callers can free() — PixelBuffer destructor handles both
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size())
        return {};

    Camera cam = cams[refCameraIndex];  // copy intrinsics
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    // Invalidate cached matrices so prepareCam recomputes from the new pose
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, int* outWidth, int* outHeight) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size()) {
        *outWidth = 0; *outHeight = 0; return;
    }

    Camera cam = cams[refCameraIndex];
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    *outWidth = w;
    *outHeight = h;
    if (!outRGBA) return;

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportSplat(const std::string& path) {
    impl->model->saveSplat(path);
}

void Trainer::saveCheckpoint(const std::string& path) {
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    impl->currentStep = impl->model->loadCheckpoint(path);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    return impl->currentStep;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() { msplat_gpu_sync(); }
void cleanup() { cleanup_msplat_metal(); }

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"
#include <memory>

// Error convention: no C++ exception crosses the C boundary. A failed call returns its
// neutral value (NULL handle, zeroed struct, -1 for int results, nothing for void) and
// msplat_last_error() returns the message until the next C API call on the same thread.
// Fixed buffer, no allocation: set_error runs inside catch handlers and must not throw
// (a std::string assignment could throw bad_alloc out of the handler). Long messages are cut.
static thread_local char g_last_error[1024];
static thread_local bool g_has_error = false;

static void set_error(const char *prefix, const char *what) noexcept {
    snprintf(g_last_error, sizeof(g_last_error), "%s%s", prefix, what ? what : "unknown error");
    g_has_error = true;
    fprintf(stderr, "%s%s\n", strncmp(g_last_error, "msplat", 6) == 0 ? "" : "msplat: ", g_last_error);
}

template <typename R, typename F>
static R guarded(R fail, F &&f) {
    g_has_error = false;
    try { return f(); }
    catch (const std::exception &e) { set_error("", e.what()); }
    catch (...) { set_error("", "unknown C++ exception"); }
    return fail;
}

template <typename F>
static void guarded_void(F &&f) {
    guarded(0, [&] { f(); return 0; });
}

const char* msplat_last_error(void) { return g_has_error ? g_last_error : nullptr; }

static msplat::Config configFromC(MsplatConfig c) {
    msplat::Config cfg;
    cfg.iterations = c.iterations;
    cfg.shDegree = c.shDegree;
    cfg.shDegreeInterval = c.shDegreeInterval;
    cfg.ssimWeight = c.ssimWeight;
    cfg.capMax = c.capMax;
    cfg.noiseLr = c.noiseLr;
    cfg.opacityReg = c.opacityReg;
    cfg.scaleReg = c.scaleReg;
    cfg.cullRadius = c.cullRadius;
    cfg.keepCrs = c.keepCrs;
    cfg.downscaleFactor = c.downscaleFactor;
    memcpy(cfg.bgColor, c.bgColor, sizeof(cfg.bgColor));
    return cfg;
}

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    return guarded<MsplatDataset>(nullptr, [&] {
        return static_cast<MsplatDataset>(new msplat::Dataset(std::string(path), downscaleFactor, evalMode, testEvery));
    });
}

void msplat_dataset_destroy(MsplatDataset ds) {
    guarded_void([&] { delete static_cast<msplat::Dataset*>(ds); });
}

int msplat_dataset_num_train(MsplatDataset ds) {
    return guarded(-1, [&] { return static_cast<msplat::Dataset*>(ds)->numTrain(); });
}

int msplat_dataset_num_test(MsplatDataset ds) {
    return guarded(-1, [&] { return static_cast<msplat::Dataset*>(ds)->numTest(); });
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    guarded_void([&] { static_cast<msplat::Dataset*>(ds)->cameraPose(cameraIndex, camToWorld); });
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    return guarded<MsplatTrainer>(nullptr, [&] {
        auto* dataset = static_cast<msplat::Dataset*>(ds);
        return static_cast<MsplatTrainer>(new msplat::Trainer(*dataset, configFromC(config)));
    });
}

void msplat_trainer_destroy(MsplatTrainer t) {
    guarded_void([&] { delete static_cast<msplat::Trainer*>(t); });
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    return guarded(MsplatStats{}, [&] {
        auto stats = static_cast<msplat::Trainer*>(t)->step();
        return MsplatStats{stats.iteration, stats.splatCount, stats.msPerStep};
    });
}

void msplat_trainer_train(MsplatTrainer t) {
    guarded_void([&] { static_cast<msplat::Trainer*>(t)->train(0); });
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    return guarded(MsplatEvalMetrics{}, [&] {
        auto m = static_cast<msplat::Trainer*>(t)->evaluate();
        return MsplatEvalMetrics{m.psnr, m.ssim, m.l1, m.numTest, m.numGaussians};
    });
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    return guarded(MsplatPixelBuffer{}, [&] {
        auto buf = static_cast<msplat::Trainer*>(t)->render(cameraIndex, useTest);
        MsplatPixelBuffer result{buf.data, buf.width, buf.height};
        buf.data = nullptr; // Transfer ownership to caller
        return result;
    });
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    return guarded(MsplatPixelBuffer{}, [&] {
        auto buf = static_cast<msplat::Trainer*>(t)->renderFromPose(camToWorld, refCameraIndex);
        MsplatPixelBuffer result{buf.data, buf.width, buf.height};
        buf.data = nullptr;
        return result;
    });
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    guarded_void([&] {
        static_cast<msplat::Trainer*>(t)->renderFromPoseToBuffer(
            camToWorld, refCameraIndex, outRGBA, outWidth, outHeight);
    });
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    guarded_void([&] { static_cast<msplat::Trainer*>(t)->exportPly(std::string(path)); });
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    guarded_void([&] { static_cast<msplat::Trainer*>(t)->exportSplat(std::string(path)); });
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    guarded_void([&] { static_cast<msplat::Trainer*>(t)->saveCheckpoint(std::string(path)); });
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    return guarded(-1, [&] { return static_cast<msplat::Trainer*>(t)->loadCheckpoint(std::string(path)); });
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    return guarded(-1, [&] { return static_cast<msplat::Trainer*>(t)->splatCount(); });
}

int msplat_trainer_iteration(MsplatTrainer t) {
    return guarded(-1, [&] { return static_cast<msplat::Trainer*>(t)->iteration(); });
}

void msplat_sync(void) { guarded_void([] { msplat::sync(); }); }
void msplat_cleanup(void) { guarded_void([] { msplat::cleanup(); }); }

// ── Viewer (render-only, no training) ───────────────────────────────────────
//
// Bypasses Model/Trainer entirely: loads Gaussian parameters from a PLY
// directly into GPU tensors, builds synthetic camera matrices from FOV +
// render resolution, and calls msplat_render. No optimizer state, no dataset
// loading, no image I/O — just the GPU sort+rasterize pipeline.

#include "msplat_c_api.h"
#include "loaders.hpp"

struct ViewerImpl {
    // Gaussian parameters — loaded once from PLY, never mutated.
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    MTensor backgroundColor;

    // Cached GPU matrices (reallocated on resize/FOV change).
    MTensor viewMat, projMat;

    int width = 0, height = 0;
    float fovY = 0;
    int numPoints = 0;

    // SH degree derived from PLY data.
    int shDegree = 0;
    int shBases = 0;

    // PLY normalization (for keepCrs=false, these are identity).
    float scale = 1.0f;
    float translation[3] = {};

    // Cached bounding box (computed once at load time).
    float boundsMin[3] = {0, 0, 0};
    float boundsMax[3] = {0, 0, 0};

    // Highest-density region center (computed once at load time).
    float densityCenterPos[3] = {0, 0, 0};

    void updateIntrinsics() {
        // Allocate or reuse the 4x4 matrix GPU buffers.
        if (!viewMat.defined())
            viewMat = gpu_empty({4, 4}, DType::Float32);
        if (!projMat.defined())
            projMat = gpu_empty({4, 4}, DType::Float32);
    }

    void computeBounds() {
        if (numPoints == 0) return;
        // means is [N, 3] Float32 on shared memory — read directly.
        const float* p = (const float*)means.data_ptr();
        float mn[3] = {p[0], p[1], p[2]};
        float mx[3] = {p[0], p[1], p[2]};
        for (int i = 1; i < numPoints; i++) {
            for (int j = 0; j < 3; j++) {
                float v = p[i * 3 + j];
                if (v < mn[j]) mn[j] = v;
                if (v > mx[j]) mx[j] = v;
            }
        }
        memcpy(boundsMin, mn, sizeof(mn));
        memcpy(boundsMax, mx, sizeof(mx));
    }

    /// Find the center of the highest-density region using an 8×8×8 grid bucketing.
    /// This tells the camera where the actual detail is, not just the geometric center.
    void computeDensityCenter() {
        if (numPoints == 0) return;
        const float* p = (const float*)means.data_ptr();
        constexpr int G = 8;  // grid resolution per axis
        int grid[G * G * G] = {};
        float ext[3] = {
            boundsMax[0] - boundsMin[0],
            boundsMax[1] - boundsMin[1],
            boundsMax[2] - boundsMin[2]
        };
        // Avoid division by zero for degenerate axes.
        for (int j = 0; j < 3; j++)
            if (ext[j] < 1e-6f) ext[j] = 1e-6f;

        for (int i = 0; i < numPoints; i++) {
            int ix = std::min((int)((p[i*3]   - boundsMin[0]) / ext[0] * G), G - 1);
            int iy = std::min((int)((p[i*3+1] - boundsMin[1]) / ext[1] * G), G - 1);
            int iz = std::min((int)((p[i*3+2] - boundsMin[2]) / ext[2] * G), G - 1);
            grid[ix * G * G + iy * G + iz]++;
        }
        // Find the densest cell.
        int maxCount = 0, maxIdx = 0;
        for (int i = 0; i < G * G * G; i++) {
            if (grid[i] > maxCount) { maxCount = grid[i]; maxIdx = i; }
        }
        // Convert back to world coordinates (center of the cell).
        int ix = maxIdx / (G * G);
        int iy = (maxIdx / G) % G;
        int iz = maxIdx % G;
        densityCenterPos[0] = boundsMin[0] + (ix + 0.5f) / G * ext[0];
        densityCenterPos[1] = boundsMin[1] + (iy + 0.5f) / G * ext[1];
        densityCenterPos[2] = boundsMin[2] + (iz + 0.5f) / G * ext[2];
    }

    // Shared matrix setup — returns camera position in world space.
    void preparePose(const float camToWorld[16], float cam_pos[3],
                     float* fx_out, float* fy_out, float* cx_out, float* cy_out,
                     int* degree_out, int* degreesToUse_out) {
        // --- Build view matrix from camToWorld (replicates Model::prepareCam) ---
        // camToWorld is 4x4 row-major, OpenGL convention (Y-up, -Z forward).
        // msplat's internal convention: negate Y and Z columns of the rotation.
        const float* d = camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] =  d[i*4+0];
            R[i][1] = -d[i*4+1];
            R[i][2] = -d[i*4+2];
            T[i]    =  d[i*4+3];
        }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++)
            Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);

        float vm[16] = {
            Rinv[0][0], Rinv[0][1], Rinv[0][2], Tinv[0],
            Rinv[1][0], Rinv[1][1], Rinv[1][2], Tinv[1],
            Rinv[2][0], Rinv[2][1], Rinv[2][2], Tinv[2],
            0, 0, 0, 1
        };

        // Intrinsics from FOV + render size.
        float fy = (float)height / (2.0f * tanf(fovY * 0.5f));
        float aspect = (float)width / (float)height;
        float fovX = 2.0f * atanf(tanf(fovY * 0.5f) * aspect);
        float fx = (float)width / (2.0f * tanf(fovX * 0.5f));

        *fx_out = fx; *fy_out = fy;
        *cx_out = (float)width * 0.5f;
        *cy_out = (float)height * 0.5f;

        // Projection-view matrix (OpenGL style, matches prepareCam).
        float t_p = 0.001f * tanf(0.5f * fovY);
        float r_p = 0.001f * tanf(0.5f * fovX);
        float pm[16] = {
            0.001f/r_p, 0, 0, 0,
            0, 0.001f/t_p, 0, 0,
            0, 0, (1000.0f+0.001f)/(1000.0f-0.001f), -1000.0f*0.001f/(1000.0f-0.001f),
            0, 0, 1, 0
        };
        float pvm[16] = {};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        memcpy(viewMat.data_ptr(), vm, sizeof(vm));
        memcpy(projMat.data_ptr(), pvm, sizeof(pvm));

        cam_pos[0] = T[0]; cam_pos[1] = T[1]; cam_pos[2] = T[2];

        // SH degrees to use — for a fully trained model, use all available.
        int frBases = (int)featuresRest.size(-2) + 1;
        *degree_out = (frBases <= 1) ? 0 : (frBases <= 4) ? 1 : (frBases <= 9) ? 2 : (frBases <= 16) ? 3 : 4;
        *degreesToUse_out = *degree_out;
    }

    // Legacy render path: GPU compute → gpu_sync → CPU float→uint8 readback.
    void render(const float camToWorld[16], uint8_t* outRGBA) {
        float cam_pos[3], fx, fy, cx, cy;
        int degree, degreesToUse;
        preparePose(camToWorld, cam_pos, &fx, &fy, &cx, &cy, &degree, &degreesToUse);

        auto tileBounds = std::make_tuple(
            (width  + BLOCK_X - 1) / BLOCK_X,
            (height + BLOCK_Y - 1) / BLOCK_Y, 1);

        MTensor rgb = msplat_render(
            numPoints, means, scales, 1.0f,
            quats, viewMat, projMat, fx, fy, cx, cy,
            (unsigned)height, (unsigned)width, tileBounds, 0.01f,
            (unsigned)degree, (unsigned)degreesToUse, cam_pos,
            featuresDc, featuresRest, opacities, backgroundColor);
        msplat_gpu_sync();

        // Convert RGB float32 → RGBA uint8 (unified memory — no copy needed).
        const float* src = (const float*)rgb.data_ptr();
        int n = width * height;
        for (int i = 0; i < n; i++) {
            outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
            outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
            outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
            outRGBA[i * 4 + 3] = 255;
        }
    }

    // Zero-copy render path: GPU compute → commit (non-blocking) → return MTLBuffer.
    // The returned buffer (g_tcache.out_img) is [H, W, 3] Float32 linear RGB.
    // Caller is responsible for GPU-side sRGB conversion and blit.
    id<MTLBuffer> renderBuffer(const float camToWorld[16]) {
        float cam_pos[3], fx, fy, cx, cy;
        int degree, degreesToUse;
        preparePose(camToWorld, cam_pos, &fx, &fy, &cx, &cy, &degree, &degreesToUse);

        auto tileBounds = std::make_tuple(
            (width  + BLOCK_X - 1) / BLOCK_X,
            (height + BLOCK_Y - 1) / BLOCK_Y, 1);

        MTensor rgb = msplat_render(
            numPoints, means, scales, 1.0f,
            quats, viewMat, projMat, fx, fy, cx, cy,
            (unsigned)height, (unsigned)width, tileBounds, 0.01f,
            (unsigned)degree, (unsigned)degreesToUse, cam_pos,
            featuresDc, featuresRest, opacities, backgroundColor);
        // Non-blocking commit — GPU work is submitted but we don't wait.
        // The viewer runs at VSync cadence with 1 frame in flight, so no
        // unbounded queue buildup risk. The Swift side's own command buffer
        // on the same serial queue provides implicit ordering.
        msplat_commit_nosync();

        return rgb.buffer();
    }
};

MsplatViewer msplat_viewer_create(const char* plyPath,
                                   int width, int height, float fovY) {
    g_has_error = false;
    try {
        // Owned until fully built: a throw below destroys it (and the tensors it already holds).
        auto v = std::make_unique<ViewerImpl>();
        v->width = width;
        v->height = height;
        v->fovY = fovY;

        // Load Gaussian parameters directly to GPU — no optimizer buffers.
        // scale=1, translation={0,0,0}, keepCrs=false: PLY is in normalized space.
        float zero[3] = {0, 0, 0};
        auto g = loadGaussianPly(std::string(plyPath), 1.0f, zero, false);
        v->means = std::move(g.means);
        v->scales = std::move(g.scales);
        v->quats = std::move(g.quats);
        v->featuresDc = std::move(g.featuresDc);
        v->featuresRest = std::move(g.featuresRest);
        v->opacities = std::move(g.opacities);
        v->numPoints = (int)v->means.size(0);

        // Default black background.
        v->backgroundColor = gpu_empty({3}, DType::Float32);
        float bg[3] = {0, 0, 0};
        memcpy(v->backgroundColor.data_ptr(), bg, sizeof(bg));

        v->updateIntrinsics();
        v->computeBounds();
        v->computeDensityCenter();

        fprintf(stderr, "[viewer] Loaded %d Gaussians from %s (%dx%d, fov=%.1f°, bounds=[%.2f,%.2f,%.2f]-[%.2f,%.2f,%.2f], density_center=[%.2f,%.2f,%.2f])\n",
                v->numPoints, plyPath, width, height, fovY * 180.0f / M_PI,
                v->boundsMin[0], v->boundsMin[1], v->boundsMin[2],
                v->boundsMax[0], v->boundsMax[1], v->boundsMax[2],
                v->densityCenterPos[0], v->densityCenterPos[1], v->densityCenterPos[2]);
        return static_cast<MsplatViewer>(v.release());
    } catch (const std::exception& e) {
        set_error("viewer: ", e.what());
        return nullptr;
    } catch (...) {
        set_error("viewer: ", "unknown C++ exception");
        return nullptr;
    }
}

void msplat_viewer_resize(MsplatViewer v, int width, int height) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (!viewer || (viewer->width == width && viewer->height == height)) return;
    viewer->width = width;
    viewer->height = height;
    // ensure_forward will reallocate render buffers on next render call.
}

void msplat_viewer_set_fov(MsplatViewer v, float fovY) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (viewer) viewer->fovY = fovY;
}

void msplat_viewer_set_background(MsplatViewer v, float r, float g, float b) {
    guarded_void([&] {
        auto* viewer = static_cast<ViewerImpl*>(v);
        if (!viewer) return;
        float bg[3] = {r, g, b};
        memcpy(viewer->backgroundColor.data_ptr(), bg, sizeof(bg));
    });
}

void msplat_viewer_render(MsplatViewer v, const float camToWorld[16],
                           uint8_t* outRGBA) {
    guarded_void([&] {
        auto* viewer = static_cast<ViewerImpl*>(v);
        if (!viewer || !outRGBA) return;
        viewer->render(camToWorld, outRGBA);
    });
}

int msplat_viewer_splat_count(MsplatViewer v) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    return viewer ? viewer->numPoints : 0;
}

void msplat_viewer_get_size(MsplatViewer v, int* outWidth, int* outHeight) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (viewer) {
        if (outWidth) *outWidth = viewer->width;
        if (outHeight) *outHeight = viewer->height;
    }
}

void* msplat_viewer_render_buffer(MsplatViewer v, const float camToWorld[16],
                                   int* outWidth, int* outHeight) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (!viewer) { *outWidth = 0; *outHeight = 0; return nullptr; }
    if (outWidth)  *outWidth  = viewer->width;
    if (outHeight) *outHeight = viewer->height;
    return guarded<void*>(nullptr, [&] { return (__bridge void*)viewer->renderBuffer(camToWorld); });
}

void msplat_viewer_bounds(MsplatViewer v, float boundsMin[3], float boundsMax[3]) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (!viewer) return;
    memcpy(boundsMin, viewer->boundsMin, 3 * sizeof(float));
    memcpy(boundsMax, viewer->boundsMax, 3 * sizeof(float));
}

void msplat_viewer_density_center(MsplatViewer v, float center[3]) {
    auto* viewer = static_cast<ViewerImpl*>(v);
    if (!viewer) return;
    memcpy(center, viewer->densityCenterPos, 3 * sizeof(float));
}

void* msplat_viewer_command_queue(MsplatViewer v) {
    (void)v;  // queue is global singleton, not per-viewer
    return guarded<void*>(nullptr, [] { return (__bridge void*)msplat_command_queue(); });
}

void msplat_viewer_destroy(MsplatViewer v) {
    guarded_void([&] { delete static_cast<ViewerImpl*>(v); });
}
