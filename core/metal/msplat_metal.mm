#import "bindings.h"
#define BLOCK_X 16
#define BLOCK_Y 16

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <chrono>
#import <dlfcn.h>
#import <unordered_map>
#import <functional>
#import <array>
#import <mutex>
#import <mach/mach_time.h>

// GPU profiling infrastructure.
// PROFILE_GPU=1: per-CB total GPU time via completion handlers.
// PROFILE_STAGES=1: per-stage GPU time via Metal timestamp counters + separate encoders.
static bool g_gpu_timing_enabled = false;
static bool g_gpu_timing_checked = false;
static std::mutex g_gpu_timing_mutex;
static std::vector<double> g_gpu_times_ms;

// Per-stage profiling
static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Stage names for training pipeline
static const char* g_train_stage_names[] = {
    "blit_zero", "proj_sh_fwd", "prefix_sort_pack", "rast_fwd",
    "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam"
};
static constexpr int N_TRAIN_STAGES = 7;

static std::mutex g_stage_timing_mutex;
// Per-stage accumulated times (ms), indexed by stage
static std::vector<double> g_stage_times[N_TRAIN_STAGES];
static int g_stage_report_count = 0;

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    // Command buffer lifecycle using MPSCommandBuffer for commitAndContinue support.
    MPSCommandBuffer* _currentCB = nil;

    // Bounds committed-but-incomplete command buffers (at most 3 in flight).
    // Unbounded commitAndContinue submission — the training loop commits every
    // step but only truly syncs every 100 steps (and only every 1000 after the
    // densify cutoff) — exhausts the queue's 64-buffer pool and lets driver
    // residency thrash under memory pressure: measured 26-47 ms/step collapsing
    // to 2000-3100 ms/step on a 24 GB machine. Waiting here restores the fast
    // path without the full pipeline drain that BENCHMARK's per-step sync pays.
    dispatch_semaphore_t inflight_sem;

    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_currentCB) {
            _currentCB = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [_currentCB retain];
        }
        return _currentCB;
    }
    void commitCB() {
        if (_currentCB) {
            if (g_gpu_timing_enabled) {
                [_currentCB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    if (gpu_ms > 0) {
                        std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
                        g_gpu_times_ms.push_back(gpu_ms);
                    }
                }];
            }
            dispatch_semaphore_t sem = inflight_sem;
            [_currentCB addCompletedHandler:^(id<MTLCommandBuffer>) {
                dispatch_semaphore_signal(sem);
            }];
            [_currentCB commitAndContinue];
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        }
    }
    void syncCB() {
        if (_currentCB) {
            [_currentCB commit];
            [_currentCB waitUntilCompleted];
            [_currentCB release];
            _currentCB = nil;
        }
    }

    // Per-stage GPU timestamp profiling (Metal counter sample buffer)
    id<MTLCounterSampleBuffer> counterSampleBuffer;
    bool counterSamplingAvailable = false;
    double ticksToMs = 0.0;  // conversion factor from GPU ticks to milliseconds

    void initCounterSampling() {
        // Need 2 samples per stage (start + end)
        NSUInteger sampleCount = N_TRAIN_STAGES * 2;

        // Find timestamp counter set
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestampSet = cs;
                break;
            }
        }
        if (!timestampSet) {
            fprintf(stderr, "PROFILE_STAGES: MTLCommonCounterSetTimestamp not available\n");
            return;
        }

        // Check if stage boundary sampling is supported (guaranteed on Apple Silicon)
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "PROFILE_STAGES: AtStageBoundary sampling not supported\n");
            return;
        }

        MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = timestampSet;
        desc.sampleCount = sampleCount;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"msplat stage profiling";

        NSError *error = nil;
        counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!counterSampleBuffer) {
            fprintf(stderr, "PROFILE_STAGES: Failed to create counter sample buffer: %s\n",
                    error.localizedDescription.UTF8String);
            return;
        }

        // Compute ticks-to-ms conversion (Apple Silicon: mach_absolute_time units)
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        ticksToMs = (double)tb.numer / (double)tb.denom / 1e6;

        counterSamplingAvailable = true;
        fprintf(stderr, "PROFILE_STAGES: GPU timestamp profiling enabled (%lu sample slots)\n",
                (unsigned long)sampleCount);
    }

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    // Hybrid per-tile sort pipeline (replaces global radix sort)
    id<MTLComputePipelineState> count_intersections_kernel_cpso;
    id<MTLComputePipelineState> prefix_sum_tiles_kernel_cpso;
    id<MTLComputePipelineState> scatter_intersections_kernel_cpso;
    id<MTLComputePipelineState> hybrid_sort_pack_kernel_cpso;
    // Legacy radix sort PSOs (kept for reference, no longer dispatched)
    id<MTLComputePipelineState> map_gaussian_to_intersects_kernel_cpso;
    id<MTLComputePipelineState> get_tile_bin_edges_kernel_cpso;
    id<MTLComputePipelineState> pack_sorted_gaussians_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_histogram_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_scan_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_scatter_kernel_cpso;
    // Prefix sum (for per-Gaussian prefix sum, now unused by sort but kept)
    id<MTLComputePipelineState> prefix_sum_kernel_cpso;
    id<MTLComputePipelineState> block_reduce_kernel_cpso;
    id<MTLComputePipelineState> block_scan_propagate_kernel_cpso;
    // Depth-chunked rasterization
    id<MTLComputePipelineState> rasterize_forward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_forward_merge_kernel_cpso;
    id<MTLComputePipelineState> compute_chunk_prefix_suffix_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso;
    // Separable SSIM loss kernels
    id<MTLComputePipelineState> ssim_h_fwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_fwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_fused_v_fwd_h_bwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_bwd_kernel_cpso;
    // Backward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_backward_kernel_cpso;
    id<MTLComputePipelineState> fused_adam_kernel_cpso;
    // MCMC kernels
    id<MTLComputePipelineState> sgld_noise_gen_kernel_cpso;
    id<MTLComputePipelineState> sgld_noise_kernel_cpso;
    id<MTLComputePipelineState> mcmc_regularization_kernel_cpso;
    id<MTLComputePipelineState> mcmc_reg_grad_kernel_cpso;
};

// Explicit metallib path (set by Swift/Python wrappers before first use)
static char* g_metallib_path = NULL;

extern "C" void msplat_set_metallib_path(const char* path) {
    free(g_metallib_path);
    g_metallib_path = path ? strdup(path) : NULL;
}

MetalContext* init_msplat_metal_context() {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = dispatch_queue_create("com.msplat.metal", DISPATCH_QUEUE_SERIAL);
    // malloc does not run member initializers — set the buffer/backpressure
    // state explicitly (a garbage _currentCB here would crash the first commit).
    ctx->_currentCB = nil;
    ctx->inflight_sem = dispatch_semaphore_create(2);  // wait-after-commit ⇒ ≤3 in flight

    // Find precompiled metallib: explicit path (XCFramework/Python) or auto-discover
    NSError *error = nil;
    id<MTLLibrary> metal_library = nil;

    if (g_metallib_path) {
        // Explicit path (set by XCFramework / Swift wrapper)
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:g_metallib_path]];
        metal_library = [device newLibraryWithURL:url error:&error];
    } else {
        // Auto-discover default.metallib next to this library or the executable
        NSFileManager *fm = [NSFileManager defaultManager];

        // 1. Next to this shared library (Python .so / linked .a)
        Dl_info dl_info;
        if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
            NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if ([fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
        // 2. Next to the main executable (CLI build)
        if (!metal_library) {
            NSString *dir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if (dir && [fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
    }

    if (!metal_library) {
        fprintf(stderr, "msplat: failed to load metallib: %s\n",
                error ? [[error description] UTF8String] : "default.metallib not found");
        free(ctx);
        return NULL;
    }

    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [metal_library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "msplat: kernel not found: %s\n", [name UTF8String]);
            return nil;
        }
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        [fn release];
        if (error) {
            fprintf(stderr, "msplat: failed to create pipeline for %s: %s\n",
                    [name UTF8String], [[error description] UTF8String]);
        }
        return pso;
    };

    // Forward pipeline
    ctx->project_and_sh_forward_kernel_cpso       = load(@"project_and_sh_forward_kernel");
    ctx->nd_rasterize_forward_kernel_cpso         = load(@"nd_rasterize_forward_kernel");
    // Hybrid per-tile sort pipeline
    ctx->count_intersections_kernel_cpso          = load(@"count_intersections_kernel");
    ctx->prefix_sum_tiles_kernel_cpso             = load(@"prefix_sum_tiles_kernel");
    ctx->scatter_intersections_kernel_cpso        = load(@"scatter_intersections_kernel");
    ctx->hybrid_sort_pack_kernel_cpso             = load(@"hybrid_sort_pack_kernel");
    // Runtime safety checks — always on (NSCAssert is compiled out in Release,
    // which is exactly the build TestFlight testers run). Both sort kernels are
    // dispatched with exactly 1024 threads/threadgroup; if this device's PSO
    // limit is lower (per-family register allocation), the dispatch is invalid
    // and silently corrupts sort offsets → gaussian_ids → NaN epidemic.
    // Fail fast with an actionable message instead of training garbage.
    {
        NSUInteger tg_mem_needed = 2048 * sizeof(uint64_t);  // BITONIC_TG_CAP * 8
        if (device.maxThreadgroupMemoryLength < tg_mem_needed) {
            fprintf(stderr,
                    "msplat: FATAL: device max threadgroup memory (%lu) is below the "
                    "configured BITONIC_TG_CAP (2048 x 8 = %lu bytes). Lower BITONIC_TG_CAP.\n",
                    (unsigned long)device.maxThreadgroupMemoryLength,
                    (unsigned long)tg_mem_needed);
            abort();
        }
        struct FixedTG { const char* name; id<MTLComputePipelineState> pso; };
        const FixedTG fixed_tg[] = {
            {"prefix_sum_tiles_kernel", ctx->prefix_sum_tiles_kernel_cpso},
            {"hybrid_sort_pack_kernel", ctx->hybrid_sort_pack_kernel_cpso},
        };
        for (const auto& k : fixed_tg) {
            if (!k.pso || k.pso.maxTotalThreadsPerThreadgroup < 1024) {
                fprintf(stderr,
                        "msplat: FATAL: %s supports only %lu threads/threadgroup on %s "
                        "but is dispatched with 1024. This device cannot run the hybrid "
                        "sort correctly — please report this GPU model.\n",
                        k.name,
                        k.pso ? (unsigned long)k.pso.maxTotalThreadsPerThreadgroup : 0ul,
                        [device.name UTF8String]);
                abort();
            }
        }
    }
    // Legacy radix sort pipeline (kept loadable, no longer dispatched)
    ctx->map_gaussian_to_intersects_kernel_cpso   = load(@"map_gaussian_to_intersects_kernel");
    ctx->get_tile_bin_edges_kernel_cpso           = load(@"get_tile_bin_edges_kernel");
    ctx->pack_sorted_gaussians_kernel_cpso        = load(@"pack_sorted_gaussians_kernel");
    ctx->radix_sort_histogram_kernel_cpso         = load(@"radix_sort_histogram_kernel");
    ctx->radix_sort_scan_kernel_cpso              = load(@"radix_sort_scan_kernel");
    ctx->radix_sort_scatter_kernel_cpso           = load(@"radix_sort_scatter_kernel");
    // Prefix sum (legacy, no longer dispatched by sort pipeline)
    ctx->prefix_sum_kernel_cpso                   = load(@"prefix_sum_kernel");
    ctx->block_reduce_kernel_cpso                 = load(@"block_reduce_kernel");
    ctx->block_scan_propagate_kernel_cpso         = load(@"block_scan_propagate_kernel");
    // Depth-chunked rasterization
    ctx->rasterize_forward_chunked_kernel_cpso    = load(@"rasterize_forward_chunked_kernel");
    ctx->rasterize_forward_merge_kernel_cpso      = load(@"rasterize_forward_merge_kernel");
    ctx->compute_chunk_prefix_suffix_kernel_cpso  = load(@"compute_chunk_prefix_suffix_kernel");
    ctx->rasterize_backward_chunked_kernel_cpso   = load(@"rasterize_backward_chunked_kernel");
    ctx->rasterize_backward_kernel_cpso           = load(@"rasterize_backward_kernel");
    // Separable SSIM loss
    ctx->ssim_h_fwd_kernel_cpso                   = load(@"ssim_h_fwd_kernel");
    ctx->ssim_v_fwd_kernel_cpso                   = load(@"ssim_v_fwd_kernel");
    ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso       = load(@"ssim_fused_v_fwd_h_bwd_kernel");
    ctx->ssim_v_bwd_kernel_cpso                   = load(@"ssim_v_bwd_kernel");
    // Backward pipeline
    ctx->project_and_sh_backward_kernel_cpso      = load(@"project_and_sh_backward_kernel");
    ctx->fused_adam_kernel_cpso                    = load(@"fused_adam_kernel");
    // MCMC kernels
    ctx->sgld_noise_gen_kernel_cpso               = load(@"sgld_noise_gen_kernel");
    ctx->sgld_noise_kernel_cpso                   = load(@"sgld_noise_kernel");
    ctx->mcmc_regularization_kernel_cpso          = load(@"mcmc_regularization_kernel");
    ctx->mcmc_reg_grad_kernel_cpso                = load(@"mcmc_reg_grad_kernel");

    // Device + engine provenance. Emitted once per run so every workspace log
    // records which GPU family and per-PSO thread limits produced the result
    // (limits are per-device compiler decisions, not properties of the metallib).
    {
        const char* fam = "unknown";
        if      ([device supportsFamily:MTLGPUFamilyApple9]) fam = "apple9";
        else if ([device supportsFamily:MTLGPUFamilyApple8]) fam = "apple8";
        else if ([device supportsFamily:MTLGPUFamilyApple7]) fam = "apple7";
        else if ([device supportsFamily:MTLGPUFamilyApple6]) fam = "apple6";
        fprintf(stderr, "[engine] msplat %s | GPU: %s | family: %s | maxTGmem: %lu\n",
                APP_VERSION, [device.name UTF8String], fam,
                (unsigned long)device.maxThreadgroupMemoryLength);

        struct PsoInfo { const char* name; id<MTLComputePipelineState> pso; };
        const PsoInfo psos[] = {
            {"project_fwd",      ctx->project_and_sh_forward_kernel_cpso},
            {"count_isect",      ctx->count_intersections_kernel_cpso},
            {"prefix_sum_tiles", ctx->prefix_sum_tiles_kernel_cpso},
            {"scatter_isect",    ctx->scatter_intersections_kernel_cpso},
            {"hybrid_sort_pack", ctx->hybrid_sort_pack_kernel_cpso},
            {"rast_fwd_chunked", ctx->rasterize_forward_chunked_kernel_cpso},
            {"rast_fwd_merge",   ctx->rasterize_forward_merge_kernel_cpso},
            {"chunk_prefix_suffix", ctx->compute_chunk_prefix_suffix_kernel_cpso},
            {"rast_bwd",         ctx->rasterize_backward_kernel_cpso},
            {"rast_bwd_chunked", ctx->rasterize_backward_chunked_kernel_cpso},
            {"ssim_h_fwd",       ctx->ssim_h_fwd_kernel_cpso},
            {"ssim_v_fwd",       ctx->ssim_v_fwd_kernel_cpso},
            {"ssim_fused_bwd",   ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso},
            {"ssim_v_bwd",       ctx->ssim_v_bwd_kernel_cpso},
            {"project_bwd",      ctx->project_and_sh_backward_kernel_cpso},
            {"fused_adam",       ctx->fused_adam_kernel_cpso},
            {"sgld_noise_gen",   ctx->sgld_noise_gen_kernel_cpso},
            {"sgld_noise",       ctx->sgld_noise_kernel_cpso},
            {"mcmc_reg",         ctx->mcmc_regularization_kernel_cpso},
        };
        fprintf(stderr, "[engine] PSO maxThreads:");
        for (const auto& p : psos) {
            fprintf(stderr, " %s=%lu", p.name,
                    p.pso ? (unsigned long)p.pso.maxTotalThreadsPerThreadgroup : 0ul);
        }
        fprintf(stderr, "\n");

        // Fail fast if any kernel's static threadgroup memory exceeds this
        // device's limit — dispatching such a PSO is undefined behavior that
        // Release builds execute silently. Skip the abort under Metal shader
        // validation: its instrumentation adds shadow threadgroup memory
        // (~2x static usage), so the SSIM kernels legitimately read over the
        // limit there while being legal in normal execution.
        const bool shader_validation = std::getenv("MTL_SHADER_VALIDATION") != nullptr;
        for (const auto& p : psos) {
            if (!p.pso) continue;
            NSUInteger tg_static = p.pso.staticThreadgroupMemoryLength;
            if (tg_static > device.maxThreadgroupMemoryLength) {
                fprintf(stderr,
                        "msplat: %s: %s uses %lu bytes of threadgroup memory but %s "
                        "allows only %lu.%s\n",
                        shader_validation ? "WARNING (shader validation inflates this)" : "FATAL",
                        p.name, (unsigned long)tg_static, [device.name UTF8String],
                        (unsigned long)device.maxThreadgroupMemoryLength,
                        shader_validation ? "" : " Dispatching it would corrupt training.");
                if (!shader_validation) abort();
            }
        }
    }

    [metal_library release];

    // Initialize counter sampling if PROFILE_STAGES is set
    ctx->counterSampleBuffer = nil;
    ctx->counterSamplingAvailable = false;
    ctx->ticksToMs = 0.0;
    if (std::getenv("PROFILE_STAGES")) {
        g_profile_stages = true;
        g_profile_stages_checked = true;
        ctx->initCounterSampling();
    }

    return ctx;
}

MetalContext* get_global_context() {
    static MetalContext* ctx = NULL;
    if (ctx == NULL) {
        ctx = init_msplat_metal_context();
    }
    return ctx;
}



#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_BUF(encoder, x, i) [encoder setBuffer:x.buffer() offset:0 atIndex:i]

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

id<MTLCommandQueue> msplat_command_queue() {
    return get_global_context()->queue;
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
}

void msplat_commit() {
    if (!g_gpu_timing_checked) {
        g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
        g_gpu_timing_checked = true;
    }
    get_global_context()->commitCB();
}

void msplat_commit_nosync() {
    MetalContext* ctx = get_global_context();
    if (ctx->_currentCB) {
        // Commit without semaphore wait — safe for the viewer because:
        // 1. Only 1 frame in flight at VSync cadence
        // 2. The Swift side creates its own CB on the same serial queue
        // 3. Metal guarantees submission order on a serial queue
        [ctx->_currentCB commitAndContinue];
    }
}

void msplat_gpu_sync() {
    get_global_context()->syncCB();
}

void msplat_enable_gpu_timing(bool enable) {
    g_gpu_timing_enabled = enable;
    g_gpu_timing_checked = true;
}

void msplat_drain_gpu_times(std::vector<double>& out) {
    std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
    out = std::move(g_gpu_times_ms);
    g_gpu_times_ms.clear();
}

void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names) {
    std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
    n_stages = std::min(max_stages, N_TRAIN_STAGES);
    for (int i = 0; i < n_stages; i++) {
        stage_times[i] = std::move(g_stage_times[i]);
        g_stage_times[i].clear();
        stage_names[i] = g_train_stage_names[i];
    }
}

#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    int fwd_num_points = 0, capacity = 0, img_height = 0, img_width = 0, num_tiles = 0;
    int bwd_num_points = 0, features_rest_bases = 0;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, num_tiles_hit, colors, aabb;
    MTensor gaussian_ids;
    MTensor packed_xy_opac, packed_conic, packed_rgb;
    MTensor out_img, final_Ts, final_idx;
    MTensor loss_intermediates;
    MTensor ssim_h_buf;
    MTensor tile_bins, loss_sum;

    // Hybrid per-tile sort buffers
    MTensor tile_counts;            // [num_tiles] uint32 — per-tile intersection counts
    MTensor tile_write_counters;    // [num_tiles] uint32 — atomic write positions per tile
    MTensor exact_offsets;          // [num_tiles] uint32 — exact start offset per tile
    MTensor sort_offsets;           // [num_tiles+1] uint32 — padded start offset per tile
    MTensor isect_keys_unsorted;    // [capacity] uint64 — unsorted per-tile sort keys

    // NOTE (2026-06-18): The hybrid per-tile bitonic sort indexes isect_keys_unsorted
    // by PADDED per-tile offsets (each tile rounded up to next_pow2(max(count,32))),
    // which can be up to ~2x the exact intersection total. The old radix sort indexed
    // by EXACT offsets, so 16x sufficed. With padding, dense scenes (e.g. cağlar at
    // 300K) overflow a 16x buffer → unguarded OOB scatter writes → corrupted
    // gaussian_ids → OOB param reads → NaN epidemic. 48x covers the padding headroom.
    int64_t capacity_multiplier = 48;

    // Fixed intersection buffers (msplat_set_isect_budget; 0 = legacy capacity 48·N). Key slots
    // (8 B) and output slots (gaussian_ids + packed attributes, 40 B) are sized once before
    // training and never grow; a view that does not fit is built and rastered in tile groups.
    int64_t key_slots = 0, out_slots = 0;
    int64_t order_key_slots = 0, order_out_slots = 0;  // diagnostic order-only groups (msplat_set_isect_order_groups)
    int key_capacity = 0;
    uint64_t isect_steps = 0, isect_groups = 0;
    uint32_t isect_max_groups = 0;

    // Depth-chunked rasterization buffers
    uint32_t current_K_max = 1;
    int chunk_K_max = 0;
    MTensor chunk_T, chunk_C, chunk_final_idx;
    MTensor prefix_T, after_C;

    // Backward gradient accumulators
    MTensor v_rendered;
    MTensor v_xy, v_conic, v_colors_rast, v_opacity, v_depth;
    MTensor v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest;

    // Cumulative NaN-gate counters (never zeroed; CPU logs deltas).
    // Slots: 0=v_mean3d 1=v_scale 2=v_quat 3=v_colors(SH) 4=spare
    //        5=v_opacity 6=v_rendered pixels 7=spare
    MTensor grad_nan_counts;

    void ensure_forward(int np, int64_t cap, int64_t key_cap, int ih, int iw, int nt,
                        id<MTLDevice> dev) {
        // MTensor has no destructor/move-assign: reassigning a cached buffer
        // orphans its retained MTLBuffer (leak). During MCMC growth these ensure_*
        // reallocs fire repeatedly, so the leaked intermediates accumulate in
        // proportion to the seed→cap growth path — this was the dominant,
        // un-enumerable "residual" in the RAM model. reset() the old buffer first.
        if (np != fwd_num_points) {
            fwd_num_points = np;
            xys.reset();            xys = mtensor_empty(dev, {np, 2}, DType::Float32);
            depths.reset();         depths = mtensor_empty(dev, {np}, DType::Float32);
            radii_out.reset();      radii_out = mtensor_empty(dev, {np}, DType::Int32);
            conics.reset();         conics = mtensor_empty(dev, {np, 3}, DType::Float32);
            num_tiles_hit.reset();  num_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            colors.reset();         colors = mtensor_empty(dev, {np, 3}, DType::Float32);
            aabb.reset();           aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
        }
        if (cap != capacity || key_cap != key_capacity) {
            capacity = cap;
            key_capacity = key_cap;
            gaussian_ids.reset();        gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            isect_keys_unsorted.reset(); isect_keys_unsorted = mtensor_empty(dev, {key_cap}, DType::Int64);
            packed_xy_opac.reset();      packed_xy_opac = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_conic.reset();        packed_conic = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_rgb.reset();          packed_rgb = mtensor_empty(dev, {cap, 3}, DType::Float32);
        }
        if (ih != img_height || iw != img_width) {
            img_height = ih; img_width = iw;
            out_img.reset();            out_img = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            final_Ts.reset();           final_Ts = mtensor_empty(dev, {ih, iw}, DType::Float32);
            final_idx.reset();          final_idx = mtensor_empty(dev, {ih, iw}, DType::Int32);
            loss_intermediates.reset(); loss_intermediates = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
            ssim_h_buf.reset();         ssim_h_buf = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
            v_rendered.reset();         v_rendered = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
        }
        if (nt != num_tiles) {
            num_tiles = nt;
            tile_bins.reset();           tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
            tile_counts.reset();         tile_counts = mtensor_empty(dev, {(int64_t)nt}, DType::Int32);
            tile_write_counters.reset(); tile_write_counters = mtensor_empty(dev, {(int64_t)nt}, DType::Int32);
            exact_offsets.reset();       exact_offsets = mtensor_empty(dev, {(int64_t)nt}, DType::Int32);
            sort_offsets.reset();        sort_offsets = mtensor_empty(dev, {(int64_t)(nt + 1)}, DType::Int32);
        }
        if (!loss_sum.defined()) {
            loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        }
        if (!grad_nan_counts.defined()) {
            grad_nan_counts = mtensor_zeros(dev, {16}, DType::Int32);
        }
    }

    void ensure_chunks(int K, int ih, int iw, id<MTLDevice> dev) {
        if (K <= chunk_K_max && ih == img_height && iw == img_width) return;
        chunk_K_max = K;
        // reset() before reassign — see the leak note in ensure_forward.
        chunk_T.reset();         chunk_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        chunk_C.reset();         chunk_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
        chunk_final_idx.reset(); chunk_final_idx = mtensor_empty(dev, {K, ih, iw}, DType::Int32);
        prefix_T.reset();        prefix_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        after_C.reset();         after_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
    }

    void ensure_backward(int np, int frb, id<MTLDevice> dev) {
        if (np != bwd_num_points || frb != features_rest_bases || !v_xy.defined()) {
            bwd_num_points = np;
            features_rest_bases = frb;
            // reset() before reassign — see the leak note in ensure_forward.
            v_xy.reset();            v_xy = mtensor_empty(dev, {np, 2}, DType::Float32);
            v_conic.reset();         v_conic = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_colors_rast.reset();   v_colors_rast = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_opacity.reset();       v_opacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            v_depth.reset();         v_depth = mtensor_empty(dev, {np}, DType::Float32);
            v_mean3d.reset();        v_mean3d = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_scale.reset();         v_scale = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_quat.reset();          v_quat = mtensor_empty(dev, {np, 4}, DType::Float32);
            v_features_dc.reset();   v_features_dc = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_features_rest.reset(); v_features_rest = mtensor_empty(dev, {(int64_t)np, (int64_t)frb, 3}, DType::Float32);
        }
    }
};
static FusedTensorCache g_tcache;

void cleanup_msplat_metal() {
    g_tcache = FusedTensorCache{};
}

// Cumulative NaN-gate counters for CPU-side delta logging (call after a GPU
// sync). Zeros until the first training step allocates the buffer.
extern "C" void msplat_grad_nan_counts(uint32_t out[16]) {
    if (!g_tcache.grad_nan_counts.defined()) {
        std::memset(out, 0, 16 * sizeof(uint32_t));
        return;
    }
    std::memcpy(out, g_tcache.grad_nan_counts.data<int32_t>(), 16 * sizeof(uint32_t));
}

// Diagnostic (CLI MSPLAT_ISECT_LOG): per-tile intersection counts of the last forward
// pass, read in place (shared storage; count kernel output, never clamped). Call only
// after msplat_gpu_sync(). Returns num_tiles (0 before the first forward).
extern "C" int msplat_last_tile_counts(const uint32_t **counts, int *num_points,
                                       int64_t *capacity, uint32_t *k_max) {
    if (!g_tcache.tile_counts.defined()) return 0;
    *counts = (const uint32_t *)g_tcache.tile_counts.data<int32_t>();
    *num_points = g_tcache.fwd_num_points;
    *capacity = g_tcache.capacity;
    *k_max = g_tcache.current_K_max;
    return g_tcache.num_tiles;
}

// Fixed intersection buffers (call before the first step; 0, 0 = legacy 48·N). The cache keeps
// capacities as int and the packed float3 outputs are addressed as int 3·idx+2, so the slot
// counts are bounded by those (Codex r3 A4).
extern "C" void msplat_set_isect_budget(int64_t key_slots, int64_t out_slots) {
    if (key_slots < 0 || out_slots < 0 || key_slots > (int64_t)INT32_MAX || out_slots > ((int64_t)INT32_MAX - 2) / 3) {
        fprintf(stderr, "msplat: intersection buffer of %lld key / %lld output slots is out of range\n",
                (long long)key_slots, (long long)out_slots);
        std::exit(3);
    }
    g_tcache.key_slots = key_slots;
    g_tcache.out_slots = out_slots;
}

// Diagnostic for the memory check (Codex r3 D1): the cache's actual sizing state, so the
// allocation model is evaluated at the state the probe stopped in, not at cap.
extern "C" void msplat_isect_cache_state(int64_t out[4]) {
    out[0] = g_tcache.fwd_num_points;
    out[1] = g_tcache.bwd_num_points;
    out[2] = g_tcache.key_capacity;
    out[3] = g_tcache.capacity;
}

// Diagnostic for the correctness test: with fixed buffers that hold the whole view, build its
// lists once but run the backward in the groups these slot counts give, without rebuilds.
extern "C" void msplat_set_isect_order_groups(int64_t key_slots, int64_t out_slots) {
    g_tcache.order_key_slots = key_slots;
    g_tcache.order_out_slots = out_slots;
}

// Tile groups per training step with fixed buffers: steps, total groups, max groups.
extern "C" void msplat_isect_group_stats(uint64_t *steps, uint64_t *groups, uint32_t *max_groups) {
    *steps = g_tcache.isect_steps;
    *groups = g_tcache.isect_groups;
    *max_groups = g_tcache.isect_max_groups;
}

// Contiguous tile groups whose padded keys and exact outputs each fit the fixed buffers
// (64-bit sums, Codex r2). Every tile is in exactly one group, so every pixel is written once
// and empty tiles still get the background. A single tile larger than a buffer is an error.
static std::vector<std::array<uint32_t, 2>> isect_tile_groups(const uint32_t *counts, int num_tiles,
                                                              int64_t key_slots, int64_t out_slots) {
    std::vector<std::array<uint32_t, 2>> groups;
    int64_t keys = 0, outs = 0;
    uint32_t lo = 0;
    for (int t = 0; t < num_tiles; t++) {
        const int64_t c = counts[t];
        if (c == 0) continue;
        int64_t p = 32;
        while (p < c) p <<= 1;  // pad_count
        if (p > key_slots || c > out_slots) {
            fprintf(stderr, "msplat: tile %d has %lld intersections, more than the fixed buffer holds "
                    "(%lld key / %lld output slots)\n", t, (long long)c, (long long)key_slots, (long long)out_slots);
            std::exit(3);
        }
        if (keys + p > key_slots || outs + c > out_slots) {
            groups.push_back({lo, (uint32_t)t});
            lo = (uint32_t)t;
            keys = outs = 0;
        }
        keys += p;
        outs += c;
    }
    groups.push_back({lo, (uint32_t)num_tiles});
    return groups;
}

// Diagnostic (CLI MSPLAT_STEP_DUMP): the last step's image, transmittance and gradients as raw
// little-endian files in dir. Call only after msplat_gpu_sync().
extern "C" void msplat_dump_step_buffers(const char *dir) {
    const std::pair<const char *, MTensor *> bufs[] = {
        {"out_img", &g_tcache.out_img}, {"final_Ts", &g_tcache.final_Ts}, {"v_rendered", &g_tcache.v_rendered},
        {"v_xy", &g_tcache.v_xy}, {"v_conic", &g_tcache.v_conic}, {"v_colors_rast", &g_tcache.v_colors_rast},
        {"v_opacity", &g_tcache.v_opacity}, {"v_mean3d", &g_tcache.v_mean3d}, {"v_scale", &g_tcache.v_scale},
        {"v_quat", &g_tcache.v_quat}, {"radii", &g_tcache.radii_out}, {"loss_sum", &g_tcache.loss_sum}};
    for (auto &b : bufs) {
        if (!b.second->defined()) continue;
        std::string path = std::string(dir) + "/" + b.first + ".bin";
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "msplat: cannot write %s\n", path.c_str()); continue; }
        fwrite(b.second->data_ptr(), 1, b.second->nbytes(), f);
        fclose(f);
    }
}

// Intersection list stages shared by render and training: count once per view, then build
// (prefix, scatter, sort/pack) the lists of tiles [lo, hi) — the whole image on the legacy
// path, one tile group with fixed buffers. Buffers and capacities come from g_tcache.
static void encode_count_isect(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                               uint32_t num_points, const std::array<uint32_t, 4> &tile_bounds) {
    NSUInteger tpg = MIN(ctx->count_intersections_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    [enc setComputePipelineState:ctx->count_intersections_kernel_cpso];
    ENC_BUF(enc, g_tcache.xys, 0); ENC_BUF(enc, g_tcache.radii_out, 1); ENC_BUF(enc, g_tcache.depths, 2);
    [enc setBytes:tile_bounds.data() length:sizeof(tile_bounds) atIndex:3];
    ENC_BUF(enc, g_tcache.tile_counts, 4);
    ENC_SCALAR(enc, num_points, 5); ENC_BUF(enc, g_tcache.aabb, 6);
    [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

static void encode_build_isect(MetalContext *ctx, id<MTLComputeCommandEncoder> enc, uint32_t num_points,
                               uint32_t num_tiles, const std::array<uint32_t, 4> &tile_bounds,
                               MTensor &opacities, uint32_t lo, uint32_t hi) {
    uint32_t capacity_u32 = (uint32_t)g_tcache.capacity;
    uint32_t key_capacity_u32 = (uint32_t)g_tcache.key_capacity;
    uint32_t tile_range[2] = {lo, hi};
    // prefix_sum_tiles — cooperative scan over tile_counts (tiles outside [lo, hi) count 0)
    [enc setComputePipelineState:ctx->prefix_sum_tiles_kernel_cpso];
    ENC_BUF(enc, g_tcache.tile_counts, 0);
    ENC_BUF(enc, g_tcache.exact_offsets, 1);
    ENC_BUF(enc, g_tcache.sort_offsets, 2);
    ENC_BUF(enc, g_tcache.tile_bins, 3);
    ENC_SCALAR(enc, num_tiles, 4);
    ENC_SCALAR(enc, capacity_u32, 5);
    [enc setBytes:tile_range length:sizeof(tile_range) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // scatter_intersections — write sort keys into per-tile buckets
    NSUInteger tpg = MIN(ctx->scatter_intersections_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
    [enc setComputePipelineState:ctx->scatter_intersections_kernel_cpso];
    ENC_BUF(enc, g_tcache.xys, 0); ENC_BUF(enc, g_tcache.depths, 1); ENC_BUF(enc, g_tcache.radii_out, 2);
    [enc setBytes:tile_bounds.data() length:sizeof(tile_bounds) atIndex:3];
    ENC_BUF(enc, g_tcache.sort_offsets, 4);
    ENC_BUF(enc, g_tcache.tile_write_counters, 5);
    ENC_BUF(enc, g_tcache.isect_keys_unsorted, 6);
    ENC_SCALAR(enc, num_points, 7); ENC_BUF(enc, g_tcache.aabb, 8);
    ENC_SCALAR(enc, key_capacity_u32, 9);
    [enc setBytes:tile_range length:sizeof(tile_range) atIndex:10];
    [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // hybrid_sort_pack — per-tile bitonic sort + pack into rasterizer buffers
    [enc setComputePipelineState:ctx->hybrid_sort_pack_kernel_cpso];
    ENC_BUF(enc, g_tcache.tile_counts, 0);
    ENC_BUF(enc, g_tcache.sort_offsets, 1);
    ENC_BUF(enc, g_tcache.exact_offsets, 2);
    ENC_BUF(enc, g_tcache.isect_keys_unsorted, 3);
    ENC_BUF(enc, g_tcache.xys, 4); ENC_BUF(enc, g_tcache.conics, 5);
    ENC_BUF(enc, opacities, 6); ENC_BUF(enc, g_tcache.colors, 7);
    ENC_BUF(enc, g_tcache.packed_xy_opac, 8); ENC_BUF(enc, g_tcache.packed_conic, 9);
    ENC_BUF(enc, g_tcache.packed_rgb, 10); ENC_BUF(enc, g_tcache.gaussian_ids, 11);
    ENC_SCALAR(enc, capacity_u32, 12); ENC_SCALAR(enc, num_points, 13);
    ENC_BUF(enc, g_tcache.grad_nan_counts, 14); // zombie-gate telemetry
    [enc setBytes:tile_range length:sizeof(tile_range) atIndex:15];
    ENC_SCALAR(enc, key_capacity_u32, 16);
    [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

// Internal forward pipeline — used by both msplat_render and msplat_train_step.
// When compute_loss=false, gt/window2d/ssim_weight are ignored.
static void forward_pipeline(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    bool compute_loss
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    const bool fixed_isect = g_tcache.key_slots > 0;
    int64_t capacity = fixed_isect ? g_tcache.out_slots : (int64_t)num_points * g_tcache.capacity_multiplier;
    int64_t key_capacity = fixed_isect ? g_tcache.key_slots : capacity;
    uint32_t channels = 3;

    // --- Cached buffer pool: only reallocate on dimension change (densification) ---
    g_tcache.ensure_forward(num_points, capacity, key_capacity, img_height, img_width, num_tiles, ctx->device);
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // --- Constants (heap-allocated for Obj-C block) ---
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

    // Periodic diagnostic: print key dimensions for roofline analysis
    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") && (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
            fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
            fprintf(stderr, "  num_points:     %d\n", num_points);
            fprintf(stderr, "  capacity:       %lld (= num_points * %lld)\n", (long long)capacity, (long long)g_tcache.capacity_multiplier);
            fprintf(stderr, "  img:            %u x %u = %u pixels\n", img_width, img_height, img_width * img_height);
            fprintf(stderr, "  tiles:          %d x %d = %d\n", tile_bounds_x, tile_bounds_y, num_tiles);
            fprintf(stderr, "  SH degree:      %u (bases: %u)\n", degree, (degree + 1) * (degree + 1));
            fprintf(stderr, "  features_rest:  [%lld x %lld x %lld]\n",
                (long long)features_rest.size(0), (long long)features_rest.size(1), (long long)features_rest.size(2));
            fprintf(stderr, "  sort:           hybrid per-tile bitonic (BITONIC_TG_CAP=2048)\n");
            fprintf(stderr, "  sort buffer:    %.1f MB (isect_keys_unsorted)\n", (double)capacity * 8.0 / 1e6);
            fprintf(stderr, "  opacities:      [%lld]\n", (long long)opacities.size(0));
            fprintf(stderr, "===========================\n\n");
    }

    // Helper lambdas to encode each stage onto a given encoder
    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, g_tcache.grad_nan_counts, 23); // zombie-gate telemetry

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_count = [&](id<MTLComputeCommandEncoder> enc) {
        encode_count_isect(ctx, enc, num_points_u32, *tile_bounds_arr);
    };
    auto encode_build = [&](id<MTLComputeCommandEncoder> enc, uint32_t lo, uint32_t hi) {
        encode_build_isect(ctx, enc, num_points_u32, (uint32_t)num_tiles, *tile_bounds_arr, opacities, lo, hi);
    };
    auto encode_prefix_sort_pack = [&](id<MTLComputeCommandEncoder> enc) {
        encode_count(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_build(enc, 0, (uint32_t)num_tiles);
    };

    // K_max for chunked rasterization — set after GPU readback
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;

    auto encode_rast_fwd_monolithic = [&](id<MTLComputeCommandEncoder> enc, uint32_t lo, uint32_t hi) {
        MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        uint32_t tile_range[2] = {lo, hi};
        [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
        ENC_BUF(enc, background, 10);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
        [enc setBytes:tile_range length:sizeof(tile_range) atIndex:12];
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    auto encode_rast_fwd_chunked = [&](id<MTLComputeCommandEncoder> enc) {
        // Phase 1: dispatch forward chunked kernel — grid (tile_x, tile_y, K_max)
        uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
        uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
        uint32_t num_pix = img_width * img_height;
        auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
        MTLSize chunked_tg = MTLSizeMake(tile_x, tile_y, K_max);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
        ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
        ENC_BUF(enc, g_tcache.grad_nan_counts, 13); // slot 15: nonfinite-sigma births
        [enc dispatchThreadgroups:chunked_tg threadsPerThreadgroup:tg_size];

        // Phase 2: merge kernel — one thread per pixel
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger merge_tpg = 256;
        [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
        ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
        ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
        ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
        ENC_BUF(enc, background, 8);
        [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
        [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            encode_rast_fwd_monolithic(enc, 0, (uint32_t)num_tiles);
        } else {
            encode_rast_fwd_chunked(enc);
        }
    };

    auto encode_loss_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        // Separable SSIM forward: H conv → barrier → V conv + SSIM + reduction
        MTLSize tg = MTLSizeMake(16, 16, 1);
        // Full 16x16 threadgroups (ceil), NOT dispatchThreads: these kernels
        // run cooperative threadgroup loops with a fixed 256-thread stride,
        // and the truncated edge groups produced by dispatchThreads leave
        // shared-memory slots uninitialized — garbage SSIM moments at every
        // right/bottom edge tile (the Apple7 epidemic's true seed; benign-
        // looking stale floats on Apple9). Out-of-image threads are already
        // guarded at every read/write.
        MTLSize ssim_tgs = MTLSizeMake((img_width + 15) / 16, (img_height + 15) / 16, 1);

        // Pass 1: horizontal convolution
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        [enc dispatchThreadgroups:ssim_tgs threadsPerThreadgroup:tg];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Pass 2: vertical convolution + SSIM/L1 + loss reduction
        [enc setComputePipelineState:ctx->ssim_v_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4);
        ENC_BUF(enc, loss_intermediates, 5); ENC_BUF(enc, loss_sum, 6);
        [enc dispatchThreadgroups:ssim_tgs threadsPerThreadgroup:tg];
    };

    // Zero cached buffers will be done inside the command encoder via blit.
    // (CPU memset would race with previous CB's GPU reads.)

    // Compute K_max conservatively from CPU-side data (no GPU sync needed).
    // avg_per_tile = capacity / num_tiles. Use 6x average to cover heavy-tailed
    // tile distributions. Overestimate is cheap: empty chunks early-exit immediately.
    // If the densest tile exceeds K_max * CHUNK_SIZE, those gaussians are silently
    // skipped — but 6x covers typical skew (measured max/avg ratio: ~1.5-2.5x).
    if (fixed_isect && num_tiles < 400) {
        fprintf(stderr, "msplat: fixed intersection buffers need >= 400 tiles per view (K = 1); this view has %d\n", num_tiles);
        std::exit(3);
    }
    if (num_tiles >= 400) {
        // High-res: enough tiles for good GPU occupancy, skip chunking
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;  // 6x average — covers heavy-tailed distributions
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    if (fixed_isect) {
        // Fixed buffers (K = 1): project + count, wait for the counts, then build and raster
        // the tile groups that fit. Forward only, so no list survives a group.
        id<MTLCommandBuffer> cb1 = ctx->getCommandBuffer();
        dispatch_sync(ctx->d_queue, ^(){
            id<MTLBlitCommandEncoder> blit = [cb1 blitCommandEncoder];
            [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_counts.buffer() range:NSMakeRange(0, g_tcache.tile_counts.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_write_counters.buffer() range:NSMakeRange(0, g_tcache.tile_write_counters.nbytes()) value:0];
            [blit endEncoding];
            id<MTLComputeCommandEncoder> encoder = [cb1 computeCommandEncoder];
            encode_proj_sh(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_count(encoder);
            [encoder endEncoding];
        });
        ctx->syncCB();
        const auto groups = isect_tile_groups((const uint32_t *)g_tcache.tile_counts.data<int32_t>(), num_tiles,
                                              key_capacity, capacity);
        id<MTLCommandBuffer> cb2 = ctx->getCommandBuffer();
        dispatch_sync(ctx->d_queue, ^(){
            id<MTLComputeCommandEncoder> encoder = [cb2 computeCommandEncoder];
            for (const auto &g : groups) {
                encode_build(encoder, g[0], g[1]);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_rast_fwd_monolithic(encoder, g[0], g[1]);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            if (compute_loss) encode_loss_fwd(encoder);
            [encoder endEncoding];
        });
        return;
    }

    {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            // Blit-zero buffers that accumulate across gaussians (must be GPU-side
            // to avoid racing with previous CB's reads on pipelined execution)
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            // tile_counts and tile_write_counters are atomically incremented by kernels 1 and 3
            [blit fillBuffer:g_tcache.tile_counts.buffer() range:NSMakeRange(0, g_tcache.tile_counts.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_write_counters.buffer() range:NSMakeRange(0, g_tcache.tile_write_counters.nbytes()) value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");

            encode_proj_sh(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_sort_pack(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(encoder);
            if (compute_loss) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_loss_fwd(encoder);
            }

            [encoder endEncoding];
        });
    }

    // All outputs are in g_tcache — no return needed
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    MTensor dummyGt, dummyWindow;
    forward_pipeline(num_points, means3d, scales, glob_scale,
        quats, viewmat, projmat, fx, fy, cx, cy,
        img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background, dummyGt, dummyWindow, 0.0f, false);
    return g_tcache.out_img;
}

std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &mask, int has_mask,
    MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    float mcmc_op_reg, float mcmc_sc_reg, float mcmc_reg_invis_scale
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    const bool fixed_isect = g_tcache.key_slots > 0;
    int64_t capacity = fixed_isect ? g_tcache.out_slots : (int64_t)num_points * g_tcache.capacity_multiplier;
    int64_t key_capacity = fixed_isect ? g_tcache.key_slots : capacity;
    uint32_t channels = 3;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, key_capacity, img_height, img_width, num_tiles, ctx->device);
    g_tcache.ensure_backward(num_points, features_rest_bases, ctx->device);

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    MTensor &v_rendered = g_tcache.v_rendered;
    MTensor &v_xy = g_tcache.v_xy;
    MTensor &v_conic = g_tcache.v_conic;
    MTensor &v_colors_rast = g_tcache.v_colors_rast;
    MTensor &v_opacity = g_tcache.v_opacity;
    MTensor &v_depth = g_tcache.v_depth;
    MTensor &v_mean3d = g_tcache.v_mean3d;
    MTensor &v_scale = g_tcache.v_scale;
    MTensor &v_quat = g_tcache.v_quat;
    MTensor &v_features_dc = g_tcache.v_features_dc;
    MTensor &v_features_rest = g_tcache.v_features_rest;

    // Wire backward outputs as Adam grads (MTensor references for gradient buffers)
    auto adam_grads = std::make_shared<std::array<MTensor, 6>>(
        std::array<MTensor, 6>{v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest, v_opacity});

    // --- Constants (heap-allocated for Obj-C block capture) ---
    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    auto rast_tb = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD});
    auto rast_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_bwd_intr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_bwd_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // --- K_max for chunked rasterization ---
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;
    if (fixed_isect && num_tiles < 400) {
        fprintf(stderr, "msplat: fixed intersection buffers need >= 400 tiles per view (K = 1); this view has %d\n", num_tiles);
        std::exit(3);
    }
    if (num_tiles >= 400) {
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;
    // A1 A/B switch: MSPLAT_LEGACY_BWD_CAP=1 restores the pre-A1 backward (skip capped gaussians).
    static const uint32_t legacy_bwd_cap = [] {
        const char *e = std::getenv("MSPLAT_LEGACY_BWD_CAP");
        const uint32_t v = (e && e[0] == '1') ? 1u : 0u;
        if (v) fprintf(stderr, "msplat: MSPLAT_LEGACY_BWD_CAP=1 (pre-A1 backward alpha-cap)\n");
        return v;
    }();

    // ========================== FORWARD ENCODE LAMBDAS ==========================

    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, g_tcache.grad_nan_counts, 23); // zombie-gate telemetry

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_count = [&](id<MTLComputeCommandEncoder> enc) {
        encode_count_isect(ctx, enc, num_points_u32, *tile_bounds_arr);
    };
    auto encode_build = [&](id<MTLComputeCommandEncoder> enc, uint32_t lo, uint32_t hi) {
        encode_build_isect(ctx, enc, num_points_u32, (uint32_t)num_tiles, *tile_bounds_arr, opacities, lo, hi);
    };
    auto encode_prefix_sort_pack = [&](id<MTLComputeCommandEncoder> enc) {
        encode_count(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_build(enc, 0, (uint32_t)num_tiles);
    };

    // [lo, hi): tile group (monolithic kernels only; the chunked K > 1 path is legacy-only)
    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc, uint32_t lo, uint32_t hi) {
        if (K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
            uint32_t tile_range[2] = {lo, hi};
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, final_Ts, 7); ENC_BUF(enc, final_idx, 8); ENC_BUF(enc, out_img, 9);
            ENC_BUF(enc, background, 10);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:11];
            [enc setBytes:tile_range length:sizeof(tile_range) atIndex:12];
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, g_tcache.chunk_T, 7); ENC_BUF(enc, g_tcache.chunk_C, 8); ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_SCALAR(enc, CHUNK_SIZE, 10); ENC_SCALAR(enc, K_max, 11);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
            ENC_BUF(enc, g_tcache.grad_nan_counts, 13); // slot 15: nonfinite-sigma births
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Merge
            [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
            ENC_BUF(enc, background, 8);
            [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
    };

    // Fused loss: ssim_h_fwd → fused_v_fwd_h_bwd → ssim_v_bwd
    // Eliminates loss_intermediates round-trip (130 MB/iter bandwidth saved).
    auto encode_loss_fwd_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize tg = MTLSizeMake(16, 16, 1);
        // Full 16x16 threadgroups (ceil), NOT dispatchThreads: these kernels
        // run cooperative threadgroup loops with a fixed 256-thread stride,
        // and the truncated edge groups produced by dispatchThreads leave
        // shared-memory slots uninitialized — garbage SSIM moments at every
        // right/bottom edge tile (the Apple7 epidemic's true seed; benign-
        // looking stale floats on Apple9). Out-of-image threads are already
        // guarded at every read/write.
        MTLSize ssim_tgs = MTLSizeMake((img_width + 15) / 16, (img_height + 15) / 16, 1);
        // Pass 1: H conv on images → ssim_h_buf
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        [enc dispatchThreadgroups:ssim_tgs threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 2: Fused V fwd + H bwd
        [enc setComputePipelineState:ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, loss_intermediates, 6); ENC_BUF(enc, loss_sum, 7);
        ENC_BUF(enc, g_tcache.grad_nan_counts, 8); // slot 4: nonfinite-moment windows
        ENC_BUF(enc, mask, 9); ENC_SCALAR(enc, has_mask, 10);
        [enc dispatchThreadgroups:ssim_tgs threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 3: V bwd
        [enc setComputePipelineState:ctx->ssim_v_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, loss_intermediates, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, v_rendered, 6);
        ENC_BUF(enc, g_tcache.grad_nan_counts, 7);
        ENC_BUF(enc, mask, 8); ENC_SCALAR(enc, has_mask, 9);
        [enc dispatchThreadgroups:ssim_tgs threadsPerThreadgroup:tg];
    };

    auto encode_rast_bwd = [&](id<MTLComputeCommandEncoder> enc, uint32_t lo, uint32_t hi) {
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            uint32_t tile_range[2] = {lo, hi};
            [enc setBytes:tile_range length:sizeof(tile_range) atIndex:15];
            [enc setComputePipelineState:ctx->rasterize_backward_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, v_rendered, 10);
            ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_conic, 12);
            ENC_BUF(enc, v_colors_rast, 13); ENC_BUF(enc, v_opacity, 14);
            ENC_SCALAR(enc, legacy_bwd_cap, 16);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked backward
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto bwd_img_sz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            // Phase 1: prefix_T and after_C
            [enc setComputePipelineState:ctx->compute_chunk_prefix_suffix_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, bwd_K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, g_tcache.prefix_T, 5); ENC_BUF(enc, g_tcache.after_C, 6);
            [enc setBytes:bwd_img_sz->data() length:sizeof(*bwd_img_sz) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Phase 2: backward chunked
            [enc setComputePipelineState:ctx->rasterize_backward_chunked_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, background, 7); ENC_BUF(enc, final_Ts, 8);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 9);
            ENC_BUF(enc, g_tcache.prefix_T, 10); ENC_BUF(enc, g_tcache.chunk_T, 11);
            ENC_BUF(enc, g_tcache.after_C, 12);
            ENC_BUF(enc, v_rendered, 13);
            ENC_BUF(enc, v_xy, 14); ENC_BUF(enc, v_conic, 15);
            ENC_BUF(enc, v_colors_rast, 16); ENC_BUF(enc, v_opacity, 17);
            ENC_SCALAR(enc, BWD_CHUNK_SIZE, 18); ENC_SCALAR(enc, bwd_K_max, 19);
            ENC_SCALAR(enc, legacy_bwd_cap, 20);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, bwd_K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        }
    };

    // Packed SH Adam hyperparameters (must match SHAdamParams in .metal)
    struct SHAdamParams {
        float dc_step_size, dc_bc2_sqrt;
        float rest_step_size, rest_bc2_sqrt;
        float beta1, beta2, eps;
    };
    auto sh_adam_hp = std::make_shared<SHAdamParams>();
    if (num_adam_groups >= 5) {
        sh_adam_hp->dc_step_size = adam_step_sizes[3];
        sh_adam_hp->dc_bc2_sqrt = adam_bc2_sqrts[3];
        sh_adam_hp->rest_step_size = adam_step_sizes[4];
        sh_adam_hp->rest_bc2_sqrt = adam_bc2_sqrts[4];
        sh_adam_hp->beta1 = adam_beta1;
        sh_adam_hp->beta2 = adam_beta2;
        sh_adam_hp->eps = adam_eps;
    }

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_backward_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0); ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr->data() length:sizeof(*proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz->data() length:sizeof(*proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_depth, 12); ENC_BUF(enc, v_conic, 13);
        ENC_BUF(enc, v_mean3d, 14); ENC_BUF(enc, v_scale, 15); ENC_BUF(enc, v_quat, 16);
        ENC_SCALAR(enc, degree, 17); ENC_SCALAR(enc, degrees_to_use, 18);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:19];
        ENC_BUF(enc, v_colors_rast, 20);
        // Fused SH backward + Adam: pass params + optimizer state instead of gradient buffers
        [enc setBuffer:adam_params[3].buffer() offset:0 atIndex:21];  // features_dc params
        [enc setBuffer:adam_params[4].buffer() offset:0 atIndex:22];  // features_rest params
        [enc setBuffer:adam_exp_avg[3].buffer() offset:0 atIndex:23]; // dc exp_avg
        [enc setBuffer:adam_exp_avg_sq[3].buffer() offset:0 atIndex:24]; // dc exp_avg_sq
        [enc setBuffer:adam_exp_avg[4].buffer() offset:0 atIndex:25]; // rest exp_avg
        [enc setBuffer:adam_exp_avg_sq[4].buffer() offset:0 atIndex:26]; // rest exp_avg_sq
        [enc setBytes:sh_adam_hp.get() length:sizeof(SHAdamParams) atIndex:27];
        ENC_BUF(enc, g_tcache.grad_nan_counts, 28);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        // MCMC reg-parity: add regularizer gradients to v_opacity / v_scale BEFORE the
        // Adam updates below (reference 3dgs-mcmc semantics — the reg term is part of
        // the loss, so Adam normalizes photometric+reg together). Never-rendered
        // Gaussians (photometric grad == 0) thus receive near-full-LR downward steps.
        if (mcmc_op_reg > 0.0f || mcmc_sc_reg > 0.0f) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            NSUInteger rtpg = MIN(ctx->mcmc_reg_grad_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->mcmc_reg_grad_kernel_cpso];
            [enc setBuffer:adam_params[5].buffer() offset:0 atIndex:0];  // opacity params
            [enc setBuffer:adam_params[1].buffer() offset:0 atIndex:1];  // scale params
            ENC_BUF(enc, v_opacity, 2);
            ENC_BUF(enc, v_scale, 3);
            ENC_SCALAR(enc, num_points, 4);
            ENC_SCALAR(enc, mcmc_op_reg, 5);
            ENC_SCALAR(enc, mcmc_sc_reg, 6);
            ENC_BUF(enc, radii_out, 7);
            ENC_SCALAR(enc, mcmc_reg_invis_scale, 8);
            [enc setBuffer:adam_exp_avg_sq[3].buffer() offset:0 atIndex:9];  // features_dc v: sum==0 ⟺ never effectively rendered (reg can't poison; nonzero for any alpha-contributing Gaussian, isotropic or not)
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(rtpg, 1, 1)];
        }
        // Adam for remaining groups (skip 3=featuresDc, 4=featuresRest — fused above)
        if (num_adam_groups > 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for (int g = 0; g < num_adam_groups; ++g) {
                if (g == 3 || g == 4) continue;  // fused into backward kernel
                uint32_t n = adam_params[g].numel();
                if (n == 0) continue;
                NSUInteger atpg = MIN(ctx->fused_adam_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
                [enc setComputePipelineState:ctx->fused_adam_kernel_cpso];
                [enc setBuffer:adam_params[g].buffer() offset:0 atIndex:0];
                [enc setBuffer:(*adam_grads)[g].buffer() offset:0 atIndex:1];
                [enc setBuffer:adam_exp_avg[g].buffer() offset:0 atIndex:2];
                [enc setBuffer:adam_exp_avg_sq[g].buffer() offset:0 atIndex:3];
                ENC_SCALAR(enc, adam_step_sizes[g], 4);
                ENC_SCALAR(enc, adam_beta1, 5);
                ENC_SCALAR(enc, adam_beta2, 6);
                ENC_SCALAR(enc, adam_bc2_sqrts[g], 7);
                ENC_SCALAR(enc, adam_eps, 8);
                ENC_SCALAR(enc, n, 9);
                ENC_BUF(enc, g_tcache.grad_nan_counts, 10);
                uint32_t adam_group_idx = (uint32_t)g;
                ENC_SCALAR(enc, adam_group_idx, 11);
                [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            }
        }
    };

    // ========================== DISPATCH ==========================

    // (MCMC: grad stats accumulation removed — no longer needed for densification)

    // Blit-zero helper (shared by both paths)
    auto do_blit_zero = [&](id<MTLCommandBuffer> cb, bool zero_loss) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (zero_loss) [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        // tile_counts and tile_write_counters are atomically incremented by kernels 1 and 3
        [blit fillBuffer:g_tcache.tile_counts.buffer() range:NSMakeRange(0, g_tcache.tile_counts.nbytes()) value:0];
        [blit fillBuffer:g_tcache.tile_write_counters.buffer() range:NSMakeRange(0, g_tcache.tile_write_counters.nbytes()) value:0];
        [blit fillBuffer:v_xy.buffer() range:NSMakeRange(0, v_xy.nbytes()) value:0];
        [blit fillBuffer:v_conic.buffer() range:NSMakeRange(0, v_conic.nbytes()) value:0];
        [blit fillBuffer:v_colors_rast.buffer() range:NSMakeRange(0, v_colors_rast.nbytes()) value:0];
        [blit fillBuffer:v_opacity.buffer() range:NSMakeRange(0, v_opacity.nbytes()) value:0];
        [blit fillBuffer:v_depth.buffer() range:NSMakeRange(0, v_depth.nbytes()) value:0];
        [blit fillBuffer:v_mean3d.buffer() range:NSMakeRange(0, v_mean3d.nbytes()) value:0];
        [blit fillBuffer:v_scale.buffer() range:NSMakeRange(0, v_scale.nbytes()) value:0];
        [blit fillBuffer:v_quat.buffer() range:NSMakeRange(0, v_quat.nbytes()) value:0];
        // v_features_dc and v_features_rest no longer needed — SH grads fused into Adam
        [blit endEncoding];
    };

    if (!g_profile_stages_checked) {
        g_profile_stages = std::getenv("PROFILE_STAGES") != nullptr;
        g_profile_stages_checked = true;
    }

    float fixed_prev_loss = 0.0f;
    if (fixed_isect) {
        // Fixed intersection buffers (C', K = 1). Group boundaries need the per-tile counts on
        // the CPU, so: project + count, wait, then encode the groups. With one group this is the
        // legacy step order; with several, each group's lists are built and rastered forward,
        // the loss runs on the whole image, then each group's lists are rebuilt (same sorted
        // order: keys depth|id are unique) for its backward. Parameters stay fixed until the
        // single projection backward + Adam at the end; gradients accumulate across groups.
        // The loss accumulator is zeroed in CB2, so after the wait it still holds the previous
        // (completed) step's loss — what this function returns on this path.
        id<MTLCommandBuffer> cb1 = ctx->getCommandBuffer();
        dispatch_sync(ctx->d_queue, ^(){
            do_blit_zero(cb1, false);
            id<MTLComputeCommandEncoder> enc = [cb1 computeCommandEncoder];
            encode_proj_sh(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_count(enc);
            [enc endEncoding];
        });
        ctx->syncCB();
        fixed_prev_loss = *loss_sum.data<float>();
        const uint32_t *counts = (const uint32_t *)g_tcache.tile_counts.data<int32_t>();
        // Diagnostic order-only mode: one list build for the whole view (it must fit the buffers),
        // but the backward runs group by group with the order-slot groups and no rebuild — the
        // atomic summation order of a multi-group step without its list rebuilds.
        const bool order_only = g_tcache.order_key_slots > 0;
        if (order_only && isect_tile_groups(counts, num_tiles, key_capacity, capacity).size() != 1) {
            fprintf(stderr, "msplat: order-only diagnostic needs buffers that hold the whole view\n");
            std::exit(3);
        }
        const auto groups = order_only
            ? isect_tile_groups(counts, num_tiles, g_tcache.order_key_slots, g_tcache.order_out_slots)
            : isect_tile_groups(counts, num_tiles, key_capacity, capacity);
        g_tcache.isect_steps++;
        g_tcache.isect_groups += groups.size();
        g_tcache.isect_max_groups = std::max(g_tcache.isect_max_groups, (uint32_t)groups.size());
        const bool rebuild = groups.size() > 1 && !order_only;

        id<MTLCommandBuffer> cb2 = ctx->getCommandBuffer();
        dispatch_sync(ctx->d_queue, ^(){
            id<MTLBlitCommandEncoder> zero = [cb2 blitCommandEncoder];
            [zero fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            [zero endEncoding];
            id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
            if (order_only) {
                encode_build(enc, 0, (uint32_t)num_tiles);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_rast_fwd(enc, 0, (uint32_t)num_tiles);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else for (const auto &g : groups) {
                encode_build(enc, g[0], g[1]);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_rast_fwd(enc, g[0], g[1]);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            encode_loss_fwd_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (rebuild) {
                // every tile is in one group, so one re-zero serves all the rebuilds
                [enc endEncoding];
                id<MTLBlitCommandEncoder> blit = [cb2 blitCommandEncoder];
                [blit fillBuffer:g_tcache.tile_write_counters.buffer()
                           range:NSMakeRange(0, g_tcache.tile_write_counters.nbytes()) value:0];
                [blit endEncoding];
                enc = [cb2 computeCommandEncoder];
            }
            for (const auto &g : groups) {
                if (rebuild) {
                    encode_build(enc, g[0], g[1]);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                encode_rast_bwd(enc, g[0], g[1]);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            encode_proj_sh_bwd_adam(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        });
    } else if (g_profile_stages && ctx->counterSamplingAvailable) {
        // Per-stage profiling: separate encoders on the SAME command buffer,
        // each with start/end timestamp sampling via the pass descriptor.
        // Metal handles inter-encoder resource hazard tracking automatically.
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        id<MTLCounterSampleBuffer> csb = ctx->counterSampleBuffer;
        double ticksToMs = ctx->ticksToMs;

        // Encode each stage in its own encoder with timestamp bookends
        typedef void (^encode_fn_t)(id<MTLComputeCommandEncoder>);
        struct StageInfo {
            const char* name;
            bool isBlit;  // true = blit encoder, false = compute encoder
        };

        dispatch_sync(ctx->d_queue, ^(){
            // Stage 0: blit_zero (use blit encoder, no timestamp — blit pass descriptors
            // don't support counter sampling the same way, so we just wrap it)
            // Use a compute pass with start/end timestamps for the blit stage
            // Actually, blit must use blit encoder. We'll measure it via a dummy compute
            // encoder with timestamps before and after.

            // -- Blit zero (no direct timestamp, included in first compute stage overhead) --
            do_blit_zero(command_buffer, true);

            // Stage encoders with timestamps
            // We have 8 compute stages (indices 0-7 in counter sample buffer)
            // Each stage uses sample indices: start = stage*2, end = stage*2+1
            auto make_profiled_encoder = [&](int stage_idx) -> id<MTLComputeCommandEncoder> {
                MTLComputePassDescriptor *passDesc = [MTLComputePassDescriptor computePassDescriptor];
                passDesc.sampleBufferAttachments[0].sampleBuffer = csb;
                passDesc.sampleBufferAttachments[0].startOfEncoderSampleIndex = stage_idx * 2;
                passDesc.sampleBufferAttachments[0].endOfEncoderSampleIndex = stage_idx * 2 + 1;
                return [command_buffer computeCommandEncoderWithDescriptor:passDesc];
            };

            id<MTLComputeCommandEncoder> enc;

            // Stage 1: proj_sh_fwd
            enc = make_profiled_encoder(0);
            encode_proj_sh(enc);
            [enc endEncoding];

            // Stage 2: prefix_sort_pack
            enc = make_profiled_encoder(1);
            encode_prefix_sort_pack(enc);
            [enc endEncoding];

            // Stage 3: rast_fwd
            enc = make_profiled_encoder(2);
            encode_rast_fwd(enc, 0, (uint32_t)num_tiles);
            [enc endEncoding];

            // Stage 4+5: loss_fwd_bwd (fused)
            enc = make_profiled_encoder(3);
            encode_loss_fwd_bwd(enc);
            [enc endEncoding];

            // Stage 5: rast_bwd
            enc = make_profiled_encoder(4);
            encode_rast_bwd(enc, 0, (uint32_t)num_tiles);
            [enc endEncoding];

            // Stage 6: proj_sh_bwd + Adam
            enc = make_profiled_encoder(5);
            encode_proj_sh_bwd_adam(enc);
            [enc endEncoding];

            // (MCMC: grad_stats stage removed)
        });

        // Add completion handler to read timestamps after GPU finishes
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            @autoreleasepool {
                NSData *data = [csb resolveCounterRange:NSMakeRange(0, (N_TRAIN_STAGES - 1) * 2)];
                if (!data) return;
                const MTLCounterResultTimestamp *samples =
                    (const MTLCounterResultTimestamp *)[data bytes];

                std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
                for (int i = 0; i < N_TRAIN_STAGES - 1; i++) {
                    uint64_t start = samples[i * 2].timestamp;
                    uint64_t end = samples[i * 2 + 1].timestamp;
                    if (start == MTLCounterErrorValue || end == MTLCounterErrorValue) continue;
                    // stage_idx 0-7 maps to g_train_stage_names[1-8] (skip blit_zero)
                    g_stage_times[i + 1].push_back((double)(end - start) * ticksToMs);
                }
                g_stage_report_count++;

                if (g_stage_report_count % 500 == 0) {
                    fprintf(stderr, "\n  === GPU Stage Profile (n=%d) ===\n", g_stage_report_count);
                    double total_median = 0;
                    for (int i = 1; i < N_TRAIN_STAGES; i++) {
                        auto &v = g_stage_times[i];
                        if (v.empty()) continue;
                        auto sorted = v;
                        std::sort(sorted.begin(), sorted.end());
                        double med = sorted[sorted.size() / 2];
                        double sum = 0;
                        for (auto x : sorted) sum += x;
                        total_median += med;
                        fprintf(stderr, "  %-20s median=%.3fms  mean=%.3fms\n",
                                g_train_stage_names[i], med, sum / sorted.size());
                    }
                    fprintf(stderr, "  %-20s %.3fms\n", "TOTAL (sum medians)", total_median);
                }
            }
        }];

    } else {
        // Production: single encoder for everything
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            do_blit_zero(command_buffer, true);

            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            assert(enc && "Failed to create compute command encoder");

            // --- Forward: proj_sh → prefix_map → rast_fwd → loss_fwd ---
            encode_proj_sh(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_sort_pack(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(enc, 0, (uint32_t)num_tiles);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // --- Fused loss forward + backward ---
            encode_loss_fwd_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_bwd(enc, 0, (uint32_t)num_tiles);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_proj_sh_bwd_adam(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc endEncoding];
        });
    }

    float loss_val = (fixed_isect ? fixed_prev_loss : *loss_sum.data<float>()) / (float)(img_height * img_width);
    return std::make_tuple(radii_out, loss_val);
}

// ============================================================================
// MCMC: fill noise buffer with N(0,1) samples on GPU via PCG+Box-Muller.
// `seed` should vary per iteration — typically the training step.
// ============================================================================
void msplat_sgld_noise_gen(int N, MTensor &noise, uint32_t seed) {
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        NSUInteger tpg = MIN(ctx->sgld_noise_gen_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
        [enc setComputePipelineState:ctx->sgld_noise_gen_kernel_cpso];
        ENC_BUF(enc, noise, 0);
        ENC_SCALAR(enc, N, 1);
        ENC_SCALAR(enc, seed, 2);
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
    });
}

// ============================================================================
// MCMC: SGLD noise injection
// ============================================================================
void msplat_sgld_noise(
    int N, MTensor &means, MTensor &scales, MTensor &quats,
    MTensor &opacities, MTensor &noise,
    float noise_lr, float xyz_lr, float max_disp
) {
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        NSUInteger tpg = MIN(ctx->sgld_noise_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
        [enc setComputePipelineState:ctx->sgld_noise_kernel_cpso];
        ENC_BUF(enc, means, 0);
        ENC_BUF(enc, scales, 1);
        ENC_BUF(enc, quats, 2);
        ENC_BUF(enc, opacities, 3);
        ENC_BUF(enc, noise, 4);
        ENC_SCALAR(enc, N, 5);
        ENC_SCALAR(enc, noise_lr, 6);
        ENC_SCALAR(enc, xyz_lr, 7);
        ENC_SCALAR(enc, max_disp, 8);
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
    });
}

// ============================================================================
// MCMC: Opacity + scale regularization (post-Adam nudge)
// ============================================================================
void msplat_mcmc_regularization(
    int N, MTensor &opacities, MTensor &scales,
    float op_lr, float sc_lr, float opacity_reg, float scale_reg
) {
    MetalContext* ctx = get_global_context();
    dispatch_sync(ctx->d_queue, ^(){
        id<MTLCommandBuffer> cb = ctx->getCommandBuffer();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        NSUInteger tpg = MIN(ctx->mcmc_regularization_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
        [enc setComputePipelineState:ctx->mcmc_regularization_kernel_cpso];
        ENC_BUF(enc, opacities, 0);
        ENC_BUF(enc, scales, 1);
        ENC_SCALAR(enc, N, 2);
        ENC_SCALAR(enc, op_lr, 3);
        ENC_SCALAR(enc, opacity_reg, 4);
        ENC_SCALAR(enc, scale_reg, 5);
        ENC_SCALAR(enc, sc_lr, 6);
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
    });
}

