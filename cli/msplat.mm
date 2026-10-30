#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "random_iter.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "bindings.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    CLI::App app{"msplat — 3D Gaussian Splatting for Apple Silicon"};
    app.set_version_flag("--version", APP_VERSION);

    // Required
    std::string projectRoot;
    app.add_option("input", projectRoot, "Path to dataset (COLMAP, Nerfstudio, Polycam)")
        ->required()
        ->check(CLI::ExistingDirectory);

    // Output
    std::string outputScene = "splat.ply";
    app.add_option("-o,--output", outputScene, "Output scene path");
    int saveEvery = -1;
    app.add_option("-s,--save-every", saveEvery, "Save every N steps (-1 to disable)");

    // Resume
    std::string resume;
    app.add_option("--resume", resume, "Resume training from PLY file")
        ->check(CLI::ExistingFile);

    // Validation
    bool validate = false;
    app.add_flag("--val", validate, "Withhold a camera for validation");
    std::string valImage = "random";
    app.add_option("--val-image", valImage, "Validation image filename");
    std::string valRender;
    app.add_option("--val-render", valRender, "Directory to render validation images");
    // Output only: with --resume <trained.ply> and -n at its step no training step runs,
    // so this re-renders a finished model's eval views for external metrics (LPIPS).
    std::string evalRender;
    app.add_option("--eval-render", evalRender, "Directory to write each eval view's render and ground truth (PNG)");

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(1, 100));   // 1 = every view is an eval view (all-frame re-render with --resume)

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    int shDegree = 3;
    app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4));
    int shDegreeInterval = 1000;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps");
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    // MCMC parameters (3DGS-MCMC, NeurIPS 2024): fixed Gaussian budget + relocation.
    int capMax = 1000000;
    app.add_option("--cap-max", capMax, "Maximum Gaussian budget (fixed)")
        ->check(CLI::Range(10000, 10000000));
    float noiseLr = 5e5f;
    app.add_option("--noise-lr", noiseLr, "SGLD noise learning rate");
    float opacityReg = 0.01f;
    app.add_option("--opacity-reg", opacityReg, "Opacity regularization weight");
    float scaleReg = 0.01f;
    app.add_option("--scale-reg", scaleReg, "Scale regularization weight");
    float cullRadius = 3.0f;
    app.add_option("--cull-radius", cullRadius, "Cull splats with ||mean|| > r in normalized space (0 = off)")
        ->check(CLI::Range(0.0f, 100.0f));
    float regVisibilityDuty = 1.0f;
    app.add_option("--reg-visibility-duty", regVisibilityDuty,
                   "Reg multiplier for Gaussians invisible in the current view that HAVE "
                   "photometric training history (never-trained junk always gets full reg). "
                   "1.0 = reference 3dgs-mcmc; 0.0 = visibility-gated reg for "
                   "sparse-visibility captures (360 rigs)")
        ->check(CLI::Range(0.0f, 1.0f));
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    bool noSeedDecimation = false;
    app.add_flag("--no-seed-decimation", noSeedDecimation,
                 "Disable the automatic seed subsample (seeds > 75% of cap-max → keep "
                 "cap/2). For fixed-seed experiments where app-side decimation owns the "
                 "seed count.");
    std::vector<float> bgColor = {0.6130f, 0.0101f, 0.3984f};
    app.add_option("--bg-color", bgColor, "Background RGB (0-1), default magenta")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");
    std::string maskDir;
    app.add_option("--mask-dir", maskDir,
                   "Directory of per-image training masks (white=use, black=ignore); "
                   "resolves <mask-dir>/<image-name>.png. Masked pixels contribute zero loss.");
    int positionLrMaxSteps = 30000;
    app.add_option("--position-lr-max-steps", positionLrMaxSteps, "Step where position LR reaches its final value")
        ->check(CLI::Range(1, 1000000));
    // Quality-gated early stop. Measured iteration curves: outdoor/sparse scenes
    // saturate by ~30k (30k->45k = +0.0-0.13 dB) while dense indoor scenes keep
    // gaining — so a fixed ceiling both wastes and starves. Requires --eval.
    float earlyStopDelta = 0.0f;
    app.add_option("--early-stop", earlyStopDelta,
                   "Stop when test-subset PSNR improves less than this (dB) over 5000 steps (requires --eval; 0 = off)")
        ->check(CLI::Range(0.0f, 10.0f));

    // Seed initialization (reference 3dgs-mcmc: opacity 0.5, scale 0.1 x kNN spacing).
    // --init-scale 1.0 reproduces the pre-2026-09-12 engine (seed-fog collapse on indoor
    // captures with isolated near-camera SfM points, e.g. tnt_meetingroom).
    float initOpacity = 0.5f;
    app.add_option("--init-opacity", initOpacity, "Seed opacity at init")
        ->check(CLI::Range(0.001f, 0.999f));
    float initScaleMult = 0.1f;
    app.add_option("--init-scale", initScaleMult, "Seed scale multiplier on the kNN spacing")
        ->check(CLI::Range(0.001f, 10.0f));

    // Memory characterization: grow to cap, allocate every real buffer, print the
    // exact Metal allocation, and exit — used to fit the app's RAM estimator to the
    // engine's actual allocator (regenerated per build). No training/eval.
    bool probeMemory = false;
    app.add_flag("--probe-memory", probeMemory,
                 "Allocate all training buffers at cap, print [[PROBE]] memory line, and exit");
    bool imageStream = false;
    app.add_flag("--image-stream", imageStream,
                 "Decode ground-truth images from disk ahead of each step instead of keeping all "
                 "of them in RAM (same bytes; MSPLAT_IMAGE_STREAM_DEPTH images ahead, default 2)");

    CLI11_PARSE(app, argc, argv);

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        for (auto &cam : inputData.cameras) {
            if (imageStream) cam.beginStream();
            cam.loadImage(downScaleFactor);
            if (imageStream) cam.gpuBaseImage.reset();  // dims/intrinsics are final; pixels stream
        }

        // Load per-image training masks (after loadImage so dims are final).
        // All-white / missing masks are dropped inside loadMask, so mask GPU
        // memory stays proportional to actual masked coverage.
        if (!maskDir.empty()) {
            int64_t maskedCams = 0;
            for (auto &cam : inputData.cameras) {
                cam.loadMask(maskDir);
                if (cam.gpuMask.defined()) ++maskedCams;
            }
            std::cout << "Masks: " << maskedCams << "/" << inputData.cameras.size()
                      << " images have masked regions (from " << maskDir << ")" << std::endl;
        }

        extern void log_memory(const char* label);
        log_memory("images loaded");

        std::vector<Camera> cams;
        std::vector<Camera> testCams;
        Camera *valCam = nullptr;

        if (evalMode) {
            auto [train, test] = inputData.splitTrainTest(testEvery);
            cams = train; testCams = test;
            std::cout << "Eval mode: " << cams.size() << " train, " << testCams.size() << " test" << std::endl;
        } else {
            auto [train, val] = inputData.getCameras(validate, valImage);
            cams = train; valCam = val;
        }

        Model model(inputData, cams.size(),
                     shDegree, shDegreeInterval,
                     capMax, noiseLr, opacityReg, scaleReg,
                     numIters, keepCrs,
                     bgColor.data(),
                     positionLrMaxSteps,
                     !noSeedDecimation,
                     initOpacity, initScaleMult);
        model.cull_radius = cullRadius;
        model.reg_visibility_duty = regVisibilityDuty;
        if (regVisibilityDuty < 1.0f) {
            std::cout << "[mcmc] reg visibility duty = " << regVisibilityDuty
                      << " (invisible-Gaussian reg scaled; 1.0 = reference)" << std::endl;
        }

        log_memory("model init");

        // Fixed intersection buffers, sized once before training: default 1000 B per Gaussian of
        // --cap-max (2.0 GB at 2M) instead of the legacy 48 slots x 48 B = 2304 B/G grown with N. A view
        // that does not fit is rastered in tile groups (extra passes, same image). MSPLAT_ISECT_BUDGET=
        // <B/G> overrides, 0 = legacy. Split 1.5 key slots : 1 output slot, floored at the single-tile
        // guarantee (pad(N) keys, N outputs, N = max(cap, seeds) so --no-seed-decimation stays safe);
        // MSPLAT_ISECT_SLOTS=<keys>,<outputs> sets them exactly (tests). Views under 400 tiles need the
        // chunked K > 1 rasterizer, which only the legacy path has, so such datasets stay legacy.
        int64_t isectKeys = 0, isectOuts = 0;
        double isectBudget = 1000;
        if (const char *b = std::getenv("MSPLAT_ISECT_BUDGET")) isectBudget = std::atof(b);
        int minTiles = INT_MAX;
        for (const auto &c : inputData.cameras)
            minTiles = std::min(minTiles, ((c.width + BLOCK_X - 1) / BLOCK_X) * ((c.height + BLOCK_Y - 1) / BLOCK_Y));
        if (isectBudget > 0 && minTiles < 400) {
            std::cout << "[isect] legacy buffers: smallest view has " << minTiles << " tiles (< 400)" << std::endl;
        } else if (isectBudget > 0) {
            const int64_t n = std::max<int64_t>(capMax, model.means.size(0));
            int64_t padN = 32;
            while (padN < n) padN <<= 1;
            double outs = isectBudget * capMax / (8 * 1.5 + 40);
            isectKeys = std::max<int64_t>(padN, (int64_t)(1.5 * outs));
            isectOuts = std::max<int64_t>(n, (int64_t)outs);
        }
        if (const char *s = std::getenv("MSPLAT_ISECT_SLOTS")) {
            long long k = 0, o = 0;
            if (sscanf(s, "%lld,%lld", &k, &o) == 2 && k > 0 && o > 0) { isectKeys = k; isectOuts = o; }
        }
        if (const char *s = std::getenv("MSPLAT_ISECT_ORDER_SLOTS")) {  // correctness test only
            long long k = 0, o = 0;
            if (sscanf(s, "%lld,%lld", &k, &o) == 2 && k > 0 && o > 0) msplat_set_isect_order_groups(k, o);
        }
        if (isectKeys > 0) {
            msplat_set_isect_budget(isectKeys, isectOuts);
            std::cout << "[isect] fixed buffers: " << isectKeys << " key + " << isectOuts << " output slots = "
                      << std::fixed << std::setprecision(1) << (8.0 * isectKeys + 40.0 * isectOuts) / 1e6 << " MB"
                      << std::defaultfloat << std::setprecision(6) << std::endl;
        }

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);

        // --image-stream: decode `depth` training images ahead in the iterator's order (its RNG is
        // its own, so drawing indices early leaves the sequence unchanged).
        std::unique_ptr<ImageStream> stream;
        int streamDepth = 2;
        if (const char *s = std::getenv("MSPLAT_IMAGE_STREAM_DEPTH")) streamDepth = std::max(1, atoi(s));
        // Every stream operation runs through streamGuard: on an ImageStreamError (file gone or
        // changed, decode/allocation/thread failure) the last completed step is saved as a PLY
        // sidecar, which --resume reads whatever -o is, then the error goes on to the exit path.
        size_t lastDone = 0, resumedAt = 0;  // last fully trained step; the step training started after
        bool finalSaved = false;             // outputScene already holds the finished model
        auto emergencySave = [&]() {
            if (finalSaved) {
                std::cerr << "msplat: image stream failed after training; the model is saved in " << outputScene << std::endl;
                return;
            }
            if (lastDone <= resumedAt) return;  // nothing trained in this run
            try {
                fs::path p(outputScene);
                p.replace_filename(p.stem().string() + "_" + std::to_string(lastDone) + ".ply");
                model.save(p.string(), (int)lastDone);  // syncs the GPU; throws if the write fails
                std::cerr << "msplat: image stream failed after step " << lastDone << "; saved " << p.string()
                          << " (continue with --resume " << p.string() << ")" << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "msplat: image stream failed and the emergency save failed too: " << e.what() << std::endl;
            }
        };
        auto streamGuard = [&](auto &&op) -> decltype(op()) {
            try {
                return op();
            } catch (const ImageStreamError &) {
                emergencySave();
                throw;
            }
        };
        if (imageStream && !cams.empty()) {
            stream = std::make_unique<ImageStream>(cams, downScaleFactor);
            for (int i = 0; i < streamDepth; i++) streamGuard([&] { stream->push(camsIter.next()); });
        }
        // Evaluation ground truth (dequantized); with --image-stream decoded on demand.
        // MSPLAT_GT_HASH=<file> logs an FNV-1a hash of every training and eval ground truth (G7).
        FILE *gtHashLog = std::getenv("MSPLAT_GT_HASH") ? fopen(std::getenv("MSPLAT_GT_HASH"), "w") : nullptr;
        auto gtHash = [](const MTensor &t) {
            uint64_t h = 1469598103934665603ull;
            const uint8_t *p = static_cast<const uint8_t *>(t.data_ptr());
            for (size_t i = 0; i < t.nbytes(); i++) h = (h ^ p[i]) * 1099511628211ull;
            return (unsigned long long)h;
        };
        auto gtCpu = [&](Camera &c) {
            MTensor g = imageStream ? streamGuard([&] { return dequantize_gt(c.loadStreamedImage(downScaleFactor)); })
                                    : dequantize_gt(c.getGPUImage());
            if (gtHashLog) fprintf(gtHashLog, "eval %s %016llx %zu\n", c.filePath.c_str(), gtHash(g), g.nbytes());
            return g;
        };

        // ── Memory probe: reach peak allocation (num_active = cap), then report ──
        if (probeMemory && cams.empty()) {
            std::cerr << "No training views (--test-every 1): --probe-memory needs training views" << std::endl;
            return 1;
        }
        if (probeMemory) {
            // MCMC grows ~5%/100 steps toward cap; run until we reach it (or a safety
            // ceiling), touching the full forward+backward+sort path so every buffer
            // is allocated at its peak size. Low-cap probes hit cap at init already.
            int64_t target = (int64_t)(0.99 * capMax);
            int safetyMax = 12000;
            for (int step = 1; step <= safetyMax && model.num_active < target; step++) {
                if (stream) streamGuard([&] { stream->push(camsIter.next()); });
                Camera &cam = cams[stream ? stream->front() : camsIter.next()];
                MTensor gt = stream ? streamGuard([&] { return stream->pop(); }) : cam.getGPUImage();
                model.fullIteration(cam, step, gt, ssimWeight);
                model.schedulersStep(step);
                model.mcmcAfterTrain(step);
                msplat_commit();
            }
            msplat_gpu_sync();

            // Exact image-buffer bytes (uint8 RGB, one resident copy per camera).
            uint64_t imageBytes = 0;
            int64_t pixels = 0;
            for (auto &cam : inputData.cameras) {
                uint64_t p = (uint64_t)cam.width * (uint64_t)cam.height;
                imageBytes += p * 3;
                if ((int64_t)p > pixels) pixels = (int64_t)p;  // per-image training resolution
            }
            if (stream) {
                // Streamed: the image buffers alive now are the queued decodes (waited for), so
                // device_allocated - images_bytes stays the image-independent working set.
                imageBytes = streamGuard([&] { return stream->heldBytes(); });
                // bound from the code (S1_TASARIM_NOTU §2): D+1 queued after a push, plus released GTs
                // awaiting their command buffer (<= 3 on the legacy path, 1 on the fixed-buffer path)
                std::cout << "[image-stream] depth=" << streamDepth << " held_bytes=" << imageBytes
                          << " peak_bound_bytes=" << (uint64_t)(streamDepth + 4) * (uint64_t)pixels * 3 << std::endl;
            }
            uint64_t deviceAllocated = (uint64_t)[msplat_device() currentAllocatedSize];

            struct task_vm_info vmInfo;
            mach_msg_type_number_t vmCount = TASK_VM_INFO_COUNT;
            uint64_t phys = 0;
            if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vmInfo, &vmCount) == KERN_SUCCESS)
                phys = vmInfo.phys_footprint;

            std::cout << "[[PROBE]]"
                      << " cap=" << capMax
                      << " active=" << model.num_active
                      << " images=" << inputData.cameras.size()
                      << " pixels=" << pixels
                      << " images_bytes=" << imageBytes
                      << " device_allocated=" << deviceAllocated
                      << " phys=" << phys
                      << std::endl;
            // separate line (the app parses [[PROBE]]): the cache state the allocation model must use
            int64_t cs[4];
            msplat_isect_cache_state(cs);
            std::cout << "[isect-probe] fwd_n=" << cs[0] << " bwd_n=" << cs[1] << " key_slots=" << cs[2]
                      << " out_slots=" << cs[3] << std::endl;
            cleanup_msplat_metal();
            msplat_gpu_sync();
            return 0;
        }

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;
        lastDone = resumedAt = step - 1;
        if (cams.empty() && step <= (size_t)numIters) {
            // --test-every 1 puts every view in the test set: only valid for a no-training re-render (--resume at its step)
            std::cerr << "No training views (--test-every 1) but " << (numIters - step + 1) << " training steps requested" << std::endl;
            return 1;
        }

        double lastGatePsnr = -1e9;   // early-stop: previous 5k-step subset PSNR
        int gateMisses = 0;           // early-stop: consecutive sub-threshold gates
        size_t stoppedAt = 0;         // early-stop: step we stopped at (0 = ran to numIters)

        bool benchmarking = std::getenv("BENCHMARK") != nullptr;
        int bench_warmup = 50;
        std::vector<double> bench_iter_ms, bench_cpu_ms, bench_drain_ms;
        if (benchmarking) {
            bench_iter_ms.reserve(numIters);
            bench_cpu_ms.reserve(numIters);
            bench_drain_ms.reserve(numIters);
        }
        auto cpu_now = []() { return std::chrono::high_resolution_clock::now(); };

        // Diagnostic: MSPLAT_ISECT_LOG=<file> writes, every MSPLAT_ISECT_EVERY (default 10) steps, the
        // training forward's per-tile intersection counts: int32 {step, train-cam index, forward N, W, H,
        // tiles, K_max, capacity} then uint32[tiles]. GPU-synced, so it costs time; results are unchanged.
        FILE *isectLog = std::getenv("MSPLAT_ISECT_LOG") ? fopen(std::getenv("MSPLAT_ISECT_LOG"), "wb") : nullptr;
        const size_t isectEvery = std::getenv("MSPLAT_ISECT_EVERY") ? std::max(1, atoi(std::getenv("MSPLAT_ISECT_EVERY"))) : 10;

        auto bench_start = cpu_now();
        for (; step <= (size_t)numIters; step++) {
            @autoreleasepool {
            if (stream) streamGuard([&] { stream->push(camsIter.next()); });
            Camera &cam = cams[stream ? stream->front() : camsIter.next()];

            auto iter_start = cpu_now();
            MTensor gt = stream ? streamGuard([&] { return stream->pop(); }) : cam.getGPUImage();
            if (gtHashLog) fprintf(gtHashLog, "train %zu %d %016llx %zu\n", step, (int)(&cam - cams.data()), gtHash(gt), gt.nbytes());
            model.fullIteration(cam, step, gt, ssimWeight);
            model.schedulersStep(step);
            model.mcmcAfterTrain(step);
            msplat_commit();
            lastDone = step;

            if (isectLog && step % isectEvery == 0) {
                msplat_gpu_sync();
                const uint32_t *counts; int np; int64_t capacity; uint32_t kMax;
                int nt = msplat_last_tile_counts(&counts, &np, &capacity, &kMax);
                int32_t hdr[8] = {(int32_t)step, (int32_t)(&cam - cams.data()), np, model.lastWidth, model.lastHeight,
                                  nt, (int32_t)kMax, (int32_t)capacity};
                fwrite(hdr, sizeof(hdr), 1, isectLog);
                fwrite(counts, sizeof(uint32_t), nt, isectLog);
                if (step % 1000 == 0) fflush(isectLog);
            }

            // Progress output for GUI integration (every 1000 steps)
            if (step % 5000 == 0) {
                char buf[64]; snprintf(buf, sizeof(buf), "step %zu", step);
                log_memory(buf);
            }
            if (step % 1000 == 0 || step == 1) {
                msplat_gpu_sync();          // make this one print-step a true single-iteration timing
                auto iter_end_time = cpu_now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    iter_end_time - iter_start).count() / 1000.0;
                std::cout << "step=" << step
                          << " splats=" << model.means.size(0)
                          << " " << std::fixed << std::setprecision(1)
                          << ms << "ms/step (single-iter, gpu-synced)"
                          << std::defaultfloat << std::setprecision(6) << std::endl;
                if (isectKeys > 0) {
                    uint64_t gSteps, gTotal; uint32_t gMax;
                    msplat_isect_group_stats(&gSteps, &gTotal, &gMax);
                    std::cout << "[isect] tile groups: steps=" << gSteps << " total=" << gTotal
                              << " max=" << gMax << std::endl;
                }
            }

            // Diagnostic: MSPLAT_STEP_DUMP=<dir> writes the image, gradients, post-step parameters and Adam
            // moments of step MSPLAT_STEP_DUMP_AT (default: the first step run) for the fixed-buffer
            // correctness test; a comma list of steps writes each to <dir>/step<N>.
            if (const char *dumpRoot = std::getenv("MSPLAT_STEP_DUMP")) {
                static const std::vector<size_t> dumpSteps = [&] {
                    std::vector<size_t> v;
                    char *p = std::getenv("MSPLAT_STEP_DUMP_AT");
                    while (p && *p) {
                        v.push_back((size_t)strtoull(p, &p, 10));
                        if (*p != ',') break;
                        p++;
                    }
                    if (v.empty()) v.push_back(step);
                    return v;
                }();
                if (std::find(dumpSteps.begin(), dumpSteps.end(), step) != dumpSteps.end()) {
                    const std::string dir = dumpSteps.size() > 1 ? std::string(dumpRoot) + "/step" + std::to_string(step)
                                                                 : std::string(dumpRoot);
                    fs::create_directories(dir);
                    const char *dumpDir = dir.c_str();
                    msplat_gpu_sync();
                    msplat_dump_step_buffers(dumpDir);
                    const std::pair<const char *, MTensor *> params[] = {
                        {"means", &model.means}, {"scales", &model.scales}, {"quats", &model.quats},
                        {"featuresDc", &model.featuresDc}, {"featuresRest", &model.featuresRest},
                        {"opacities", &model.opacities}};
                    for (auto &p : params) {
                        FILE *f = fopen((std::string(dumpDir) + "/" + p.first + ".bin").c_str(), "wb");
                        if (f) { fwrite(p.second->data_ptr(), 1, p.second->nbytes(), f); fclose(f); }
                    }
                    // Adam moments (groups as in fullIteration: means, scales, quats, featuresDc, featuresRest,
                    // opacities); the SH groups are the only record of the fused SH gradients
                    for (int g = 0; g < Model::N_ADAM_GROUPS; g++) {
                        for (int sq = 0; sq < 2; sq++) {
                            MTensor &m = sq ? model.adam_exp_avg_sq[g] : model.adam_exp_avg[g];
                            FILE *f = fopen((std::string(dumpDir) + "/adam" + (sq ? "_sq" : "_m") + std::to_string(g) + ".bin").c_str(), "wb");
                            if (f) { fwrite(m.data_ptr(), 1, m.nbytes(), f); fclose(f); }
                        }
                    }
                    uint64_t gSteps = 0, gTotal = 0; uint32_t gMax = 0;
                    msplat_isect_group_stats(&gSteps, &gTotal, &gMax);
                    FILE *f = fopen((std::string(dumpDir) + "/meta.txt").c_str(), "w");
                    if (f) {
                        fprintf(f, "step %zu cam %d n %lld w %d h %d group_steps %llu groups %llu max_groups %u\n",
                                step, (int)(&cam - cams.data()), (long long)model.means.size(0), model.lastWidth,
                                model.lastHeight, (unsigned long long)gSteps, (unsigned long long)gTotal, gMax);
                        fclose(f);
                    }
                }
            }

            if (benchmarking && step > (size_t)bench_warmup) {
                auto pre_sync = cpu_now();
                msplat_gpu_sync();
                auto iter_end = cpu_now();
                double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(pre_sync - iter_start).count() / 1000.0;
                double drain_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - pre_sync).count() / 1000.0;
                bench_iter_ms.push_back(iter_ms);
                bench_cpu_ms.push_back(cpu_ms);
                bench_drain_ms.push_back(drain_ms);
            }

            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            // Quality gate: cheap test-subset eval every 5000 steps. Stops the
            // fine-tuning tail once it pays less than earlyStopDelta dB per 5k
            // (outdoor scenes saturate ~30k; dense indoor keeps earning).
            // Two consecutive sub-threshold gates are required: a 10-view probe
            // proved too aliased on orbit captures (read 0.0 dB across a span
            // where the full 90-view eval gained +0.24), so use 30 views + hysteresis.
            if (earlyStopDelta > 0.0f && evalMode && !testCams.empty()
                && step >= 10000 && step % 5000 == 0 && step < (size_t)numIters) {
                size_t nSub = std::min((size_t)30, testCams.size());
                size_t stride = testCams.size() / nSub;
                double sum = 0;
                for (size_t i = 0; i < nSub; i++) {
                    Camera &tc = testCams[i * stride];
                    MTensor rgb = model.render(tc, step);
                    msplat_gpu_sync();
                    MTensor rgb_cpu = rgb.cpu();
                    MTensor gt_cpu = gtCpu(tc);
                    sum += psnr(rgb_cpu, gt_cpu);
                }
                double gatePsnr = sum / nSub;
                std::cout << "gate step=" << step << std::fixed << std::setprecision(3)
                          << " subsetPSNR=" << gatePsnr;
                if (lastGatePsnr > -1e8) std::cout << " delta=" << (gatePsnr - lastGatePsnr);
                std::cout << std::endl;
                if (lastGatePsnr > -1e8 && gatePsnr - lastGatePsnr < earlyStopDelta) {
                    if (++gateMisses >= 2) {
                        std::cout << "early-stop: two gates below " << earlyStopDelta
                                  << " dB per 5000 steps; stopping at step " << step << std::endl;
                        stoppedAt = step;
                    }
                } else {
                    gateMisses = 0;
                }
                lastGatePsnr = gatePsnr;
            }
            if (stoppedAt) break;

            if (!valRender.empty() && step % 10 == 0) {
                MTensor rgb = model.render(*valCam, step);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                Image valImg;
                valImg.width = (int)rgb_cpu.size(1);
                valImg.height = (int)rgb_cpu.size(0);
                valImg.data.resize(valImg.width * valImg.height * 3);
                memcpy(valImg.ptr(), rgb_cpu.data_ptr(), valImg.data.size() * sizeof(float));
                imwriteRGB((fs::path(valRender) / (std::to_string(step) + ".png")).string(), valImg);
            }
            } // @autoreleasepool
        }

        if (stream) {
            std::cout << "[image-stream] depth=" << streamDepth << " pops=" << stream->pops << " stalls=" << stream->stalls
                      << " stall_ms=" << std::fixed << std::setprecision(1) << stream->stallMs
                      << " first_wait_ms=" << stream->firstWaitMs
                      << std::defaultfloat << std::setprecision(6) << std::endl;
            stream.reset();  // waits for the decodes still in flight and frees their buffers
        }

        if (benchmarking && !bench_iter_ms.empty()) {
            auto bench_end = cpu_now();
            double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start).count() / 1000.0;
            size_t n = bench_iter_ms.size();
            std::vector<double> sorted = bench_iter_ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            double mean = sum / n;
            double median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
            double sq_sum = 0;
            for (double v : sorted) sq_sum += (v - mean) * (v - mean);
            double stddev = std::sqrt(sq_sum / n);

            std::cout << "\n=== Benchmark (" << n << " iters, " << bench_warmup << " warmup, " << total_s << "s total) ===\n";
            std::cout << "  mean:   " << mean   << " ms/iter\n";
            std::cout << "  median: " << median  << " ms/iter\n";
            std::cout << "  stddev: " << stddev  << " ms/iter\n";
            std::cout << "  p5:     " << sorted[(size_t)(n * 0.05)] << " ms/iter\n";
            std::cout << "  p95:    " << sorted[(size_t)(n * 0.95)] << " ms/iter\n";
            std::cout << "  min:    " << sorted.front() << " ms/iter\n";
            std::cout << "  max:    " << sorted.back()  << " ms/iter\n";
            std::cout << "  wall:   " << total_s << "s for " << numIters << " iters\n";

            auto stats = [](std::vector<double> &v) {
                std::vector<double> s = v;
                std::sort(s.begin(), s.end());
                size_t n = s.size();
                double sum = std::accumulate(s.begin(), s.end(), 0.0);
                double med = (n % 2 == 0) ? (s[n/2-1] + s[n/2]) / 2.0 : s[n/2];
                return std::make_pair(sum / n, med);
            };
            auto [cpu_mean, cpu_med] = stats(bench_cpu_ms);
            auto [drain_mean, drain_med] = stats(bench_drain_ms);
            std::cout << "\n  --- CPU dispatch vs GPU drain ---\n";
            std::cout << "  cpu dispatch:  mean=" << cpu_mean << "  median=" << cpu_med << " ms\n";
            std::cout << "  gpu drain:     mean=" << drain_mean << "  median=" << drain_med << " ms\n";
            std::cout << "  gpu fraction:  " << (drain_med / median * 100) << "%\n";

            // GPU timing from completion handlers (PROFILE_GPU=1)
            std::vector<double> gpu_times;
            msplat_drain_gpu_times(gpu_times);
            if (!gpu_times.empty()) {
                auto [gpu_mean, gpu_med] = stats(gpu_times);
                std::vector<double> gs = gpu_times;
                std::sort(gs.begin(), gs.end());
                std::cout << "\n  --- GPU kernel time (from CB completion handlers) ---\n";
                std::cout << "  gpu exec:   mean=" << gpu_mean << "  median=" << gpu_med << " ms\n";
                std::cout << "  gpu p5:     " << gs[(size_t)(gs.size() * 0.05)] << " ms\n";
                std::cout << "  gpu p95:    " << gs[(size_t)(gs.size() * 0.95)] << " ms\n";
                std::cout << "  gpu min:    " << gs.front() << " ms\n";
                std::cout << "  gpu max:    " << gs.back() << " ms\n";
                std::cout << "  n_cbs:      " << gs.size() << "\n";
            }

            // Per-stage GPU timing (PROFILE_STAGES=1)
            constexpr int MAX_STAGES = 16;
            std::vector<double> stage_times[MAX_STAGES];
            const char* stage_names[MAX_STAGES] = {};
            int n_stages = 0;
            msplat_drain_stage_times(stage_times, MAX_STAGES, n_stages, stage_names);
            bool has_stage_data = false;
            for (int i = 0; i < n_stages; i++) if (!stage_times[i].empty()) { has_stage_data = true; break; }
            if (has_stage_data) {
                std::cout << "\n  --- Per-stage GPU time (Metal timestamp counters) ---\n";
                double total_med = 0;
                for (int i = 0; i < n_stages; i++) {
                    if (stage_times[i].empty()) continue;
                    auto [s_mean, s_med] = stats(stage_times[i]);
                    total_med += s_med;
                    std::cout << "  " << std::left << std::setw(22) << stage_names[i]
                              << "median=" << std::fixed << std::setprecision(3) << s_med
                              << "ms  mean=" << s_mean << "ms  (" << stage_times[i].size() << " samples)\n";
                }
                std::cout << "  " << std::left << std::setw(22) << "TOTAL (sum medians)"
                          << std::fixed << std::setprecision(3) << total_med << "ms\n";
            }
            std::cout << "\n";
        }

        if (isectKeys > 0) {
            uint64_t gSteps, gTotal; uint32_t gMax;
            msplat_isect_group_stats(&gSteps, &gTotal, &gMax);
            std::cout << "[isect] training: " << gSteps << " steps, " << gTotal - gSteps
                      << " extra tile-group passes, max " << gMax << " groups in one step" << std::endl;
        }

        // Stamp the step actually reached (early stop may end before numIters).
        int finalStep = stoppedAt ? (int)stoppedAt : numIters;

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, finalStep);
        finalSaved = true;

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], finalStep);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = gtCpu(testCams[i]);

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                if (!evalRender.empty()) {
                    fs::create_directories(evalRender);
                    auto stem = fs::path(evalRender) / fs::path(testCams[i].filePath).stem();
                    for (auto [t, suffix] : {std::pair{&rgb_cpu, "_render.png"}, std::pair{&gt_cpu, "_gt.png"}}) {
                        Image img;
                        img.width = (int)t->size(1);
                        img.height = (int)t->size(0);
                        img.data.resize(img.width * img.height * 3);
                        memcpy(img.ptr(), t->data_ptr(), img.data.size() * sizeof(float));
                        imwriteRGB(stem.string() + suffix, img);
                    }
                }

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, finalStep);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = gtCpu(*valCam);

            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }

        if (gtHashLog) fclose(gtHashLog);
        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
