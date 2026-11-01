#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <sys/stat.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading (single-allocation UMA pipeline) ──────────────────────────
//
// On Apple Silicon, MTLResourceStorageModeShared buffers share the same physical
// memory between CPU and GPU. We decode the image into a temporary CPU vector,
// copy into a Metal buffer, and immediately discard the temporary. Only one
// allocation persists per camera: the gpuBaseImage MTensor.

void Camera::loadImage(float downscaleFactor, bool fusedDecode, MTensor into) {
    const bool fused = fusedDecode && downscaleFactor > 1.0f;
    int rawW, rawH;
    Image raw = fused ? imreadRGBArea(filePath, downscaleFactor, rawW, rawH) : imreadRGB(filePath);
    if (!fused) { rawW = raw.width; rawH = raw.height; }
    if (raw.empty()) return;

    // If actual image dimensions differ from metadata, rescale intrinsics
    if (width > 0 && height > 0 && (rawW != width || rawH != height)) {
        float sx = (float)rawW / (float)width;
        float sy = (float)rawH / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = rawW; height = rawH;
    } else if (width == 0 || height == 0) {
        width = rawW; height = rawH;
    }

    // Downscale (already done by imreadRGBArea when fused: width == rawW here, same newW/newH)
    if (downscaleFactor > 1.0f) {
        int newW = (int)(width / downscaleFactor);
        int newH = (int)(height / downscaleFactor);
        if (!fused) raw = resizeArea(raw, newW, newH);
        // Scale intrinsics by the true resize ratio: int(W/d) truncates, so
        // 1/d misregisters camera and image by up to 0.5–0.75 px at the far edge.
        float sx = (float)newW / (float)width;
        float sy = (float)newH / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = newW; height = newH;
    }

    // Undistort if needed
    if (hasDistortion()) {
        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        raw = std::move(result.image);
        fx = result.fx; fy = result.fy;
        cx = result.cx; cy = result.cy;
        width = result.width; height = result.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    // Quantize float→uint8 after all geometric transforms (resize, undistort)
    // have been applied in full float precision. Source images are 8-bit JPG/MP4
    // so the ±0.5/255 round-trip error is below PSNR-meaningful precision.
    // Cuts per-pixel GT storage from 12 bytes (Float32×3) to 3 bytes (uint8×3).
    // Metal loss/SSIM kernels cast back to float on-the-fly at read time.
    size_t N = (size_t)raw.width * (size_t)raw.height;
    // Only an explicit destination (--image-stream recycling) is reused, and only when its shape
    // matches; every byte is written below, so the result is the same as a fresh buffer. A plain
    // reload always allocates, so tensors still holding the previous image never see it change.
    if (into.defined() && into.dtype() == DType::UInt8 && into.nbytes() == N * 3 &&
        into.size(0) == raw.height && into.size(1) == raw.width)
        gpuBaseImage = std::move(into);
    else
        gpuBaseImage = gpu_empty({raw.height, raw.width, 3}, DType::UInt8);
    uint8_t* dst = static_cast<uint8_t*>(gpuBaseImage.data_ptr());
    const float* src = raw.ptr();
    for (size_t i = 0; i < N * 3; ++i)
        dst[i] = (uint8_t)std::clamp(std::round(src[i] * 255.0f), 0.0f, 255.0f);
    // raw goes out of scope here → std::vector<float> freed immediately
}

MTensor& Camera::getGPUImage() {
    return gpuBaseImage;
}

// ── GT image stream (--image-stream) ────────────────────────────────────────
//
// Replays loadImage on a copy of the pre-load camera: the same decode, resize, undistort
// and quantization as the resident path. Runs on worker threads, so it touches only its
// arguments (src is immutable once shared) and gpu_empty (MTLDevice allocation is
// thread-safe); it encodes no GPU work.
struct FileStamp { int64_t size = -1, mtimeNs = 0; uint64_t ino = 0; };
static FileStamp fileStamp(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) throw std::runtime_error("Failed to load image: " + path);
    return {(int64_t)st.st_size, (int64_t)st.st_mtimespec.tv_sec * 1000000000 + st.st_mtimespec.tv_nsec,
            (uint64_t)st.st_ino};
}

void Camera::beginStream() {
    FileStamp f = fileStamp(filePath);
    streamSize = f.size; streamMtimeNs = f.mtimeNs; streamIno = f.ino;
    streamSource = std::make_shared<const Camera>(*this);
}

// The bytes must be the ones the run started with: a file replaced or edited in place since
// startup (even at the same size) is an error, checked before and after the decode.
static void checkStamp(const Camera &src) {
    FileStamp f = fileStamp(src.filePath);
    if (f.size != src.streamSize || f.mtimeNs != src.streamMtimeNs || f.ino != src.streamIno)
        throw std::runtime_error("msplat: --image-stream: " + src.filePath +
                                 " changed since startup (size, modification time or inode differ)");
}

// MSPLAT_IMAGE_STREAM_FUSED=1: stream decodes use imreadRGBArea (same bytes, no full-size float image).
static bool streamFused() {
    static const bool v = [] { const char *e = std::getenv("MSPLAT_IMAGE_STREAM_FUSED"); return e && e[0] == '1'; }();
    return v;
}

static MTensor decodeStreamed(const Camera &src, int width, int height, float downscaleFactor,
                              MTensor into = MTensor()) {
    try {
        checkStamp(src);
        Camera c = src;
        c.loadImage(downscaleFactor, streamFused(), std::move(into));  // undefined into: loadImage allocates
        checkStamp(src);
        if (!c.gpuBaseImage.defined() || c.width != width || c.height != height)
            throw std::runtime_error("msplat: --image-stream: " + src.filePath + " decodes to " +
                                     std::to_string(c.width) + "x" + std::to_string(c.height) + ", expected " +
                                     std::to_string(width) + "x" + std::to_string(height) +
                                     " (file changed since startup?)");
        return std::move(c.gpuBaseImage);
    } catch (const std::exception &e) {
        throw ImageStreamError(e.what());  // same message ("Failed to load image: …", "out of memory", …)
    }
}

MTensor Camera::loadStreamedImage(float downscaleFactor) const {
    return decodeStreamed(*streamSource, width, height, downscaleFactor);
}

void ImageStream::push(size_t idx) {
    const Camera &cam = cams_[idx];
    try {
        queue_.emplace_back(idx, std::async(std::launch::async,
            [src = cam.streamSource, w = cam.width, h = cam.height, d = downscale_,
             into = takeFree((size_t)cam.width * (size_t)cam.height * 3)] {
                return decodeStreamed(*src, w, h, d, into);
            }));
    } catch (const std::exception &e) {  // thread start / allocation
        throw ImageStreamError(std::string("msplat: --image-stream: cannot start a decode: ") + e.what());
    }
}

MTensor ImageStream::pop() {
    std::future<MTensor> &f = queue_.front().second;
    if (f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        auto t0 = std::chrono::steady_clock::now();
        f.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (pops == 0) firstWaitMs = ms;  // the queue's first fill has no GPU step to hide behind
        else { stallMs += ms; stalls++; }
    }
    MTensor t = f.get();
    queue_.pop_front();
    pops++;
    if (recycle_) {
        // The previous image's step is fully encoded and committed by now; its buffer is free
        // once a full sync completes after this point (see takeFree).
        if (inUse_.defined()) free_.push_back({msplat_sync_epoch(), msplat_committed_serial(), std::move(inUse_)});
        if (free_.size() > 4) free_.pop_front();  // no sync since (none recyclable): let the oldest go
        inUse_ = t;
    }
    return t;
}

// A recycled buffer of exactly `nbytes` whose last step is complete on the GPU, or an undefined
// tensor (the decode then allocates). Main thread only, like push/pop.
MTensor ImageStream::takeFree(size_t nbytes) {
    const uint64_t epoch = msplat_sync_epoch();
    for (auto it = free_.begin(); it != free_.end(); ++it)
        if (it->epoch < epoch && it->t.nbytes() == nbytes) {
            // Lifetime invariant, checked independently of the epoch rule: the last commit of the
            // step that used this buffer has completed. If not, the buffer is not reused (counted).
            if (msplat_completed_serial() < it->needSerial) { recycleViolations++; return MTensor(); }
            MTensor t = std::move(it->t);
            free_.erase(it);
            recycled++;
            return t;
        }
    for (const auto &f : free_)
        if (f.epoch < epoch) { recycleSizeMiss++; break; }   // only other-size buffers were ready: fresh allocation
    return MTensor();
}

uint64_t ImageStream::heldBytes() {
    uint64_t bytes = 0;
    for (auto &entry : queue_) {
        MTensor t = entry.second.get();  // rethrows a failed decode: never count an absent buffer
        bytes += t.nbytes();
        std::promise<MTensor> ready;     // put the image back for its pop()
        ready.set_value(std::move(t));
        entry.second = ready.get_future();
    }
    return bytes;
}

// Load the per-image training mask (white = use, black = ignore). All-white
// masks are dropped on load so mask memory stays proportional to actual
// masked coverage. The mask is resized to the post-downscale image dims and
// binarized at 0.5 so interpolation never produces soft loss weights.
void Camera::loadMask(const std::string &maskDir) {
    if (maskDir.empty() || relativeName.empty() || width <= 0 || height <= 0) return;

    fs::path appended = fs::path(maskDir) / (relativeName + ".png");
    fs::path swapped = (fs::path(maskDir) / relativeName).replace_extension(".png");
    std::string path;
    if (fs::exists(appended)) path = appended.string();
    else if (fs::exists(swapped)) path = swapped.string();
    else return;

    Image m = imreadRGB(path);  // grayscale PNGs decode to replicated RGB
    if (m.empty()) return;
    if (m.width != width || m.height != height) {
        m = resizeArea(m, width, height);
    }

    size_t N = (size_t)width * (size_t)height;
    std::vector<uint8_t> keep(N);
    int64_t kept = 0;
    const float *src = m.ptr();
    for (size_t i = 0; i < N; ++i) {
        bool k = src[i * 3] > 0.5f;
        keep[i] = k ? 255 : 0;
        kept += k ? 1 : 0;
    }
    if (kept == (int64_t)N) return;  // all-white: nothing masked

    gpuMask = gpu_empty({height, width}, DType::UInt8);
    memcpy(gpuMask.data_ptr(), keep.data(), N);
    unmaskedPixels = kept;
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position.
    // CONTRACT: this camera-anchored convention is load-bearing. The fixed
    // position LR, the SGLD noise_lr default (5e5), and cull_radius default
    // (3.0) all assume cameras span ≈[-1, 1] after this transform. Anchoring
    // the scale to point-cloud extent instead (tried 2026-06-11) enlarges the
    // frame by λ and makes relative SGLD noise λ× stronger (noise ∝ s²),
    // which produced a 96%-NaN model on the park dataset. It also breaks the
    // downstream assumption that training space == partitioner-normalized
    // space (keepCrs=false PLY export feeds SeamlessMerger directly).
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    fs::path root(path);

    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json"))
        return loaders::loadNerfstudio(path);

    // COLMAP: cameras.bin (direct or in sparse/0/)
    if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin"))
        return loaders::loadColmap(path, colmapImagePath);

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json"))
        return loaders::loadPolycam(path);

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)");
}
