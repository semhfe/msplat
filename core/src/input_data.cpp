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

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading (single-allocation UMA pipeline) ──────────────────────────
//
// On Apple Silicon, MTLResourceStorageModeShared buffers share the same physical
// memory between CPU and GPU. We decode the image into a temporary CPU vector,
// copy into a Metal buffer, and immediately discard the temporary. Only one
// allocation persists per camera: the gpuBaseImage MTensor.

void Camera::loadImage(float downscaleFactor) {
    Image raw = imreadRGB(filePath);
    if (raw.empty()) return;

    // If actual image dimensions differ from metadata, rescale intrinsics
    if (width > 0 && height > 0 && (raw.width != width || raw.height != height)) {
        float sx = (float)raw.width / (float)width;
        float sy = (float)raw.height / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = raw.width; height = raw.height;
    } else if (width == 0 || height == 0) {
        width = raw.width; height = raw.height;
    }

    // Downscale
    if (downscaleFactor > 1.0f) {
        int newW = (int)(width / downscaleFactor);
        int newH = (int)(height / downscaleFactor);
        raw = resizeArea(raw, newW, newH);
        float s = 1.0f / downscaleFactor;
        fx *= s; fy *= s; cx *= s; cy *= s;
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
