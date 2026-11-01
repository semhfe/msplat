#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <deque>
#include <future>
#include <memory>
#include <stdexcept>
#include <cstdlib>
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

    void loadImage(float downscaleFactor, bool fusedDecode = false,  // fused: imreadRGBArea (same bytes)
                   MTensor into = MTensor());                         // into: recycled destination, used if shape matches
    // Load "<maskDir>/<relativeName>.png" (COLMAP convention; falls back to the
    // extension-swapped name) — call AFTER loadImage so dimensions are final.
    void loadMask(const std::string &maskDir);
    MTensor& getGPUImage();
    // --image-stream: the camera as it was before loadImage (metadata only). The pixels are
    // not kept resident; loadStreamedImage() replays loadImage on a copy of it, so the GT
    // bytes come from the same code on the same inputs as the resident gpuBaseImage.
    std::shared_ptr<const Camera> streamSource;
    // Size, mtime (ns) and inode of filePath when streaming began; a decode fails if they changed.
    int64_t streamSize = -1, streamMtimeNs = 0;
    uint64_t streamIno = 0;
    void beginStream();          // call BEFORE loadImage: stamps the file, saves streamSource
    MTensor loadStreamedImage(float downscaleFactor) const;
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

// Every --image-stream failure (missing/changed file, decode, allocation, thread start) is
// thrown as this type, so the CLI can save the run before it exits.
struct ImageStreamError : std::runtime_error { using std::runtime_error::runtime_error; };

// --image-stream: GT images decoded from disk ahead of use instead of kept resident.
// FIFO in consumption order: push(idx) starts an asynchronous decode of camera idx,
// pop() returns the oldest one (waiting for it if needed: a stall). With D pushes ahead,
// camera t+D decodes while the GPU runs steps t..t+D-1.
class ImageStream {
public:
    ImageStream(const std::vector<Camera> &cams, float downscaleFactor)
        : cams_(cams), downscale_(downscaleFactor) {}
    void push(size_t idx);
    size_t front() const { return queue_.front().first; }
    MTensor pop();               // rethrows a decode error
    uint64_t heldBytes();        // waits for the queued decodes; bytes they hold (rethrows a failure)
    uint64_t pops = 0, stalls = 0;  // stalls: a pop after the first that had to wait
    uint64_t recycled = 0;          // decodes that reused a GT buffer (MSPLAT_IMAGE_STREAM_RECYCLE=1)
    uint64_t recycleViolations = 0; // reuse refused: its last commit not yet complete (expected 0)
    uint64_t recycleSizeMiss = 0;   // fresh allocation although a completed buffer of another size was free
    double stallMs = 0, firstWaitMs = 0;
private:
    MTensor takeFree(size_t nbytes);
    const std::vector<Camera> &cams_;
    float downscale_;
    // MSPLAT_IMAGE_STREAM_RECYCLE=1: reuse GT buffers instead of a fresh allocation per step.
    const bool recycle_ = [] { const char *e = std::getenv("MSPLAT_IMAGE_STREAM_RECYCLE"); return e && e[0] == '1'; }();
    MTensor inUse_;                                   // the image handed out by the last pop
    struct FreeBuf { uint64_t epoch, needSerial; MTensor t; };  // sync epoch and commit serial when returned
    std::deque<FreeBuf> free_;
    std::deque<std::pair<size_t, std::future<MTensor>>> queue_;
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
