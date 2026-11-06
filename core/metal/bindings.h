#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include <utility>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
id<MTLCommandQueue> msplat_command_queue();
#else
void* msplat_device();
void* msplat_command_queue();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (blocks on inflight semaphore for training safety)
void msplat_commit();

// Commit without blocking — for viewer render path only.
// Safe because the viewer has a single frame in flight at VSync cadence.
void msplat_commit_nosync();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();
// Count of full syncs: when it has advanced past a value read earlier, every GPU command
// encoded before that read has completed.
uint64_t msplat_sync_epoch();
// Commit serials: work committed through msplat_commit / _nosync up to serial k has completed
// once msplat_completed_serial() >= k (independent of the sync epoch; --image-stream recycling check).
uint64_t msplat_committed_serial();
uint64_t msplat_completed_serial();

// Diagnostic: last forward's per-tile intersection counts (after msplat_gpu_sync()).
extern "C" int msplat_last_tile_counts(const uint32_t **counts, int *num_points,
                                       int64_t *capacity, uint32_t *k_max);

// Fixed intersection buffers: key slots (8 B) and output slots (40 B), set before the first
// step; 0, 0 = legacy capacity 48·N. Views that do not fit run in tile groups (K = 1 only).
extern "C" void msplat_set_isect_budget(int64_t key_slots, int64_t out_slots);
extern "C" void msplat_isect_group_stats(uint64_t *steps, uint64_t *groups, uint32_t *max_groups);
// Diagnostic: whole-view lists, backward run in these groups without rebuilds (order-matched test).
extern "C" void msplat_set_isect_order_groups(int64_t key_slots, int64_t out_slots);
// Diagnostic: fixed-buffer cache state (after msplat_gpu_sync()): forward/backward N, key/output slots.
extern "C" void msplat_isect_cache_state(int64_t out[4]);
// Diagnostic: last step's image, transmittance and gradients as raw files (after msplat_gpu_sync()).
extern "C" void msplat_dump_step_buffers(const char *dir);

// Step profiler (runs/2026-11-06-hiz-profil-plan.md s2). MSPLAT_PROFILE_STEP=<file>: per-command-buffer GPU
// start/end, CPU leaf spans and step metadata as text lines; encoders unchanged (level 1).
// MSPLAT_PROFILE_SPLIT=count|nocount (level 2): the training encoders are split per stage, with (count) or
// without (nocount) stage-boundary timestamps. Without MSPLAT_PROFILE_STEP every call below is a no-op.
bool msplat_prof_on();
void msplat_prof_set_step(int step);
uint64_t msplat_prof_now();                                        // mach_absolute_time
void msplat_prof_span(const char *name, uint64_t t0, uint64_t t1); // leaf CPU span (never nested)
void msplat_prof_step_wall(uint64_t t0, uint64_t t1);              // whole step, start to end of the loop body
void msplat_prof_meta(const char *key, double value);
void msplat_prof_cb_kind(const char *kind);                        // label of the next committed command buffer
void msplat_prof_calibrate();                                      // paired CPU/GPU timestamp sample
void msplat_prof_flush();
struct MsplatProfSpan {
    const char *name; uint64_t t0;
    explicit MsplatProfSpan(const char *n) : name(n), t0(msplat_prof_on() ? msplat_prof_now() : 0) {}
    ~MsplatProfSpan() { if (t0) msplat_prof_span(name, t0, msplat_prof_now()); }
};

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
    MTensor &gt,
    // Training mask (H, W) uint8, COLMAP polarity (255 = use, 0 = ignore).
    // When has_mask == 0 the tensor is a dummy binding and is never read.
    MTensor &mask, int has_mask,
    MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    // MCMC reg-parity: reg gradients injected pre-Adam (reference 3dgs-mcmc loss
    // terms, torch-mean normalization). Pass 0,0 to disable. invis_scale = reg
    // multiplier for Gaussians invisible in the current view (visibility duty-cycle
    // correction for sparse-visibility captures; 1.0 = exact reference).
    float mcmc_op_reg, float mcmc_sc_reg, float mcmc_reg_invis_scale,
    // Package B1: when non-null, the rasterizer backward also writes, per gaussian, the max over this
    // view's pixels of its blending weight alpha*T into this [>= N] float buffer (atomic max on the
    // float bits; the caller zeroes it). nullptr = off (the buffer is not touched).
    MTensor *blend_wmax = nullptr
);

// MCMC: fill an N*3 noise buffer with N(0,1) samples on GPU (PCG+Box-Muller).
void msplat_sgld_noise_gen(int N, MTensor &noise, uint32_t seed);

// MCMC: Inject SGLD noise into Gaussian positions, weighted by covariance and opacity
void msplat_sgld_noise(
    int N, MTensor &means, MTensor &scales, MTensor &quats,
    MTensor &opacities, MTensor &noise,
    float noise_lr, float xyz_lr, float max_disp
);

// ── Q1-Q3 (--quant-level 1): code stores read and written by the Q kernels (core/metal/msplat_q.metal).
// Layout and contract: docs/superpowers/2026-10-28-confidence-densification/runs/2026-11-01-hiz-q-yerlesim.md.
// nsq index: 0 means, 1 scales, 2 quats, 3 featuresDc, 4 opacities.
struct MsplatQ {
    MTensor *fr_codes, *fr_bounds, *shq_state, *shq_bounds;
    MTensor *nsq_state[5], *nsq_bounds[5];
    uint32_t step;   // adam_step_count of this step (stochastic-rounding hash key)
    uint32_t round;  // Model::quant_round: 0 nearest, 1 stochastic, 2 stochastic Adam codes only
};
// Selects the Q stores for the msplat_render / msplat_train_step calls that follow; nullptr = the fp32 path.
// The first non-null call creates the Q pipelines. Callers use the entry points below.
void msplat_q_select(const MsplatQ *q);
struct MsplatQScope {
    explicit MsplatQScope(const MsplatQ &q) { msplat_q_select(&q); }
    ~MsplatQScope() { msplat_q_select(nullptr); }
};
// Q entry points: the arguments of msplat_render / msplat_train_step after q. The Q kernels replace
// project_and_sh_forward_kernel, project_and_sh_backward_kernel, mcmc_reg_grad_kernel and fused_adam_kernel;
// features_rest is the zero-width placeholder and adam_exp_avg / adam_exp_avg_sq are not read.
template <class... A> MTensor msplat_render_q(const MsplatQ &q, A &&...a) {
    MsplatQScope s(q);
    return msplat_render(std::forward<A>(a)...);
}
template <class... A> std::tuple<MTensor, float> msplat_train_step_q(const MsplatQ &q, A &&...a) {
    MsplatQScope s(q);
    return msplat_train_step(std::forward<A>(a)...);
}
// Cumulative Q saturation counts (0 before the first Q step; read after a sync). qsat: SH Adam log_s bounds held at
// h = 64 plus non-SH/DC log_s values clamped at 64; usat: non-SH/DC |u| values clamped at a_u = 8 (attempt 2).
uint32_t msplat_q_qsat();
uint32_t msplat_q_usat();

// MCMC: Apply opacity and scale regularization as post-Adam parameter nudge.
// Reg-parity: op_lr / sc_lr are the opacity and scale Adam group LRs.
void msplat_mcmc_regularization(
    int N, MTensor &opacities, MTensor &scales,
    float op_lr, float sc_lr, float opacity_reg, float scale_reg
);

#endif
