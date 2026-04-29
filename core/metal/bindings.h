#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
);

// Fused forward + backward + Adam in one encoder (no grad stats accumulation)
// Returns: (radii [N], loss_value float)
std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps
);

// MCMC: fill an N*3 noise buffer with N(0,1) samples on GPU (PCG+Box-Muller).
void msplat_sgld_noise_gen(int N, MTensor &noise, uint32_t seed);

// MCMC: Inject SGLD noise into Gaussian positions, weighted by covariance and opacity
void msplat_sgld_noise(
    int N, MTensor &means, MTensor &scales, MTensor &quats,
    MTensor &opacities, MTensor &noise,
    float noise_lr, float xyz_lr
);

// MCMC: Apply opacity, scale, and anisotropy regularization as post-Adam
// parameter nudges. aniso_reg is a hinge on the per-Gaussian scale ratio
// max(exp(s))/min(exp(s)) above 10:1 — set to 0 to disable.
void msplat_mcmc_regularization(
    int N, MTensor &opacities, MTensor &scales,
    float lr, float opacity_reg, float scale_reg, float aniso_reg
);

#endif
