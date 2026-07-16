// C API for Swift interop. Thin wrapper around msplat C++ types.
// Opaque handles + free functions — works with any Swift version.

#ifndef MSPLAT_C_API_H
#define MSPLAT_C_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Config ──────────────────────────────────────────────────────────────────

typedef struct {
    int iterations;
    int shDegree;
    int shDegreeInterval;
    float ssimWeight;
    // MCMC parameters
    int capMax;
    float noiseLr;
    float opacityReg;
    float scaleReg;
    float cullRadius;
    bool keepCrs;
    float downscaleFactor;
    float bgColor[3];
} MsplatConfig;

static inline MsplatConfig msplat_default_config(void) {
    MsplatConfig c;
    c.iterations = 30000;
    c.shDegree = 3;
    c.shDegreeInterval = 1000;
    c.ssimWeight = 0.2f;
    c.capMax = 1000000;
    c.noiseLr = 5e5f;
    c.opacityReg = 0.01f;
    c.scaleReg = 0.01f;
    c.cullRadius = 3.0f;
    c.keepCrs = false;
    c.downscaleFactor = 1.0f;
    c.bgColor[0] = 0.6130f; c.bgColor[1] = 0.0101f; c.bgColor[2] = 0.3984f;
    return c;
}

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep;
} MsplatStats;

typedef struct {
    float psnr;
    float ssim;
    float l1;
    int numTest;
    int numGaussians;
} MsplatEvalMetrics;

// ── Pixel buffer ────────────────────────────────────────────────────────────

typedef struct {
    float* data;   // RGB float32, HWC layout. Caller must free() this.
    int width;
    int height;
} MsplatPixelBuffer;

// ── Dataset ─────────────────────────────────────────────────────────────────

typedef void* MsplatDataset;

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery);
void msplat_dataset_destroy(MsplatDataset ds);
int msplat_dataset_num_train(MsplatDataset ds);
int msplat_dataset_num_test(MsplatDataset ds);

// ── Trainer ─────────────────────────────────────────────────────────────────

typedef void* MsplatTrainer;

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config);
void msplat_trainer_destroy(MsplatTrainer t);

MsplatStats msplat_trainer_step(MsplatTrainer t);
void msplat_trainer_train(MsplatTrainer t);
MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t);
MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest);
MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex);

/// Render into a caller-provided RGBA uint8 buffer (no allocation, no float copy).
/// outRGBA must be at least width*height*4 bytes. Returns dimensions via outWidth/outHeight.
/// Call once with outRGBA=NULL to get dimensions, then allocate and call again.
void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight);
void msplat_trainer_export_ply(MsplatTrainer t, const char* path);
void msplat_trainer_export_splat(MsplatTrainer t, const char* path);
void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_splat_count(MsplatTrainer t);
int msplat_trainer_iteration(MsplatTrainer t);
void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]);

// ── Lifecycle ───────────────────────────────────────────────────────────────

void msplat_set_metallib_path(const char* path);
void msplat_sync(void);
void msplat_cleanup(void);

// ── Viewer (render-only, no training) ───────────────────────────────────────

typedef void* MsplatViewer;

/// Load a trained Gaussian PLY for view-only rendering.
/// width/height: initial render resolution. fovY: vertical FOV in radians.
/// Returns NULL on failure (bad path, parse error).
MsplatViewer msplat_viewer_create(const char* plyPath,
                                   int width, int height, float fovY);

/// Update render resolution (e.g. window resize).
void msplat_viewer_resize(MsplatViewer v, int width, int height);

/// Update vertical field of view (radians).
void msplat_viewer_set_fov(MsplatViewer v, float fovY);

/// Set background color (RGB, each 0–1).
void msplat_viewer_set_background(MsplatViewer v, float r, float g, float b);

/// Render from a camera-to-world pose (4x4 row-major, OpenGL convention)
/// into a caller-provided RGBA uint8 buffer.
/// outRGBA must be at least width*height*4 bytes.
void msplat_viewer_render(MsplatViewer v, const float camToWorld[16],
                           uint8_t* outRGBA);

/// Number of Gaussians in the loaded scene.
int msplat_viewer_splat_count(MsplatViewer v);

/// Render dimensions (after the most recent create/resize).
void msplat_viewer_get_size(MsplatViewer v, int* outWidth, int* outHeight);

/// Zero-copy render: executes GPU compute pipeline and returns the raw output
/// MTLBuffer (Float32 RGB, shape [H*W*3], StorageModeShared).
/// The buffer is valid until the next render call. Caller must NOT free it.
/// Calls msplat_commit_nosync() internally (non-blocking, no semaphore wait).
/// Returns NULL if viewer is invalid. outWidth/outHeight receive render dims.
void* msplat_viewer_render_buffer(MsplatViewer v, const float camToWorld[16],
                                   int* outWidth, int* outHeight);

/// Scene bounding box from loaded PLY gaussian means.
/// Sets boundsMin/boundsMax to the axis-aligned bounding box of all positions.
void msplat_viewer_bounds(MsplatViewer v, float boundsMin[3], float boundsMax[3]);

/// Center of the highest-density region (8×8×8 grid bucketing of gaussian means).
/// Useful for initial camera positioning — points the camera at where the detail is.
void msplat_viewer_density_center(MsplatViewer v, float center[3]);

/// Returns the engine's MTLCommandQueue (as void*) for use in Swift-side
/// blit/compute encoders. Ensures GPU work is serialized on the same queue.
void* msplat_viewer_command_queue(MsplatViewer v);

/// Destroy viewer and release all GPU resources.
void msplat_viewer_destroy(MsplatViewer v);

#ifdef __cplusplus
}
#endif

#endif // MSPLAT_C_API_H
