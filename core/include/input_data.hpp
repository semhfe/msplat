#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;
    // Dataset-relative image name (e.g. "cam0/frame_000001.jpg") as stored in
    // COLMAP images.bin — used to resolve the matching training mask.
    std::string relativeName;

    // Single GPU image buffer (UMA shared storage — CPU & GPU read the same memory).
    // Replaces the previous dual-copy design (CPU std::vector + Metal MTensor).
    MTensor gpuBaseImage;
    // Optional training mask (H, W) uint8: 255 = pixel contributes to the loss,
    // 0 = ignored (COLMAP mask polarity). Undefined when the camera has no mask
    // or the mask is all-white. See Camera::loadMask.
    MTensor gpuMask;
    // Count of contributing pixels when gpuMask is defined; -1 otherwise.
    int64_t unmaskedPixels = -1;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    // Pose-opt (Task 2+): index into PoseOpt's per-image delta array (-1 if
    // pose-opt is off or this camera wasn't assigned one), and the rig-camera
    // slot parsed from relativeName's "cam<k>/" prefix (-1 for non-rig datasets).
    int poseIndex = -1;
    int rigCamId = -1;

    void loadImage(float downscaleFactor);
    // Load "<maskDir>/<relativeName>.png" (COLMAP convention; falls back to the
    // extension-swapped name) — call AFTER loadImage so dimensions are final.
    void loadMask(const std::string &maskDir);
    MTensor& getGPUImage();
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;

    std::tuple<std::vector<Camera>, Camera*> getCameras(bool validate, const std::string &valImage = "random");
    std::tuple<std::vector<Camera>, std::vector<Camera>> splitTrainTest(int testEvery);
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");

#endif
