#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
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

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(2, 100));

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
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    std::vector<float> bgColor = {0.6130f, 0.0101f, 0.3984f};
    app.add_option("--bg-color", bgColor, "Background RGB (0-1), default magenta")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");
    std::string maskDir;
    app.add_option("--mask-dir", maskDir,
                   "Directory of per-image training masks (white=use, black=ignore); "
                   "resolves <mask-dir>/<image-name>.png. Masked pixels contribute zero loss.");
    bool poseOpt = false;
    app.add_flag("--pose-opt", poseOpt,
                 "Per-image SE(3) pose refinement during training (360 rigs)");
    float poseLr = 5e-4f;
    app.add_option("--pose-lr", poseLr, "Pose refinement learning rate")
        ->check(CLI::Range(0.0f, 1.0f));
    int poseOptStart = 2000;
    app.add_option("--pose-opt-start", poseOptStart, "Step to enable pose refinement");
    int poseTta = 0;
    app.add_option("--pose-tta", poseTta,
                   "Test-time pose-alignment steps for the val view (requires --pose-opt)");
    std::string seamWeightDir;
    app.add_option("--seam-weight-dir", seamWeightDir,
                   "Directory of per-crop-cam loss-weight PNGs (cam0.png..): gray 0-255 scales the loss");
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

    // Memory characterization: grow to cap, allocate every real buffer, print the
    // exact Metal allocation, and exit — used to fit the app's RAM estimator to the
    // engine's actual allocator (regenerated per build). No training/eval.
    bool probeMemory = false;
    app.add_flag("--probe-memory", probeMemory,
                 "Allocate all training buffers at cap, print [[PROBE]] memory line, and exit");

    CLI11_PARSE(app, argc, argv);

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        for (auto &cam : inputData.cameras)
            cam.loadImage(downScaleFactor);

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

        auto rigCamOf = [](const std::string& rel) -> int {
            // "cam<k>/frame_....jpg" -> k; -1 for non-rig datasets
            if (rel.rfind("cam", 0) != 0) return -1;
            size_t slash = rel.find('/');
            if (slash == std::string::npos || slash <= 3) return -1;
            try { return std::stoi(rel.substr(3, slash - 3)); } catch (...) { return -1; }
        };
        int nRigCams = 0;
        for (size_t i = 0; i < cams.size(); i++) {
            cams[i].poseIndex = (int)i;
            cams[i].rigCamId = rigCamOf(cams[i].relativeName);
            nRigCams = std::max(nRigCams, cams[i].rigCamId + 1);
        }
        if (valCam) { valCam->poseIndex = (int)cams.size(); valCam->rigCamId = rigCamOf(valCam->relativeName); }

        Model model(inputData, cams.size(),
                     shDegree, shDegreeInterval,
                     capMax, noiseLr, opacityReg, scaleReg,
                     numIters, keepCrs,
                     bgColor.data(),
                     positionLrMaxSteps);
        model.cull_radius = cullRadius;

        if (poseOpt) {
            int nSlots = (int)cams.size() + (valCam ? 1 : 0);
            model.poseOpt = std::make_unique<PoseOpt>(nSlots, nRigCams, poseLr, poseOptStart);
            std::cout << "Pose-opt: ON  lr=" << poseLr << " start=" << poseOptStart
                      << " cams=" << nRigCams << " images=" << nSlots << std::endl;
        }

        log_memory("model init");

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);

        // ── Memory probe: reach peak allocation (num_active = cap), then report ──
        if (probeMemory) {
            // MCMC grows ~5%/100 steps toward cap; run until we reach it (or a safety
            // ceiling), touching the full forward+backward+sort path so every buffer
            // is allocated at its peak size. Low-cap probes hit cap at init already.
            int64_t target = (int64_t)(0.99 * capMax);
            int safetyMax = 12000;
            for (int step = 1; step <= safetyMax && model.num_active < target; step++) {
                Camera &cam = cams[camsIter.next()];
                MTensor gt = cam.getGPUImage();
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
            cleanup_msplat_metal();
            msplat_gpu_sync();
            return 0;
        }

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;

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

        auto bench_start = cpu_now();
        for (; step <= (size_t)numIters; step++) {
            @autoreleasepool {
            Camera &cam = cams[camsIter.next()];

            auto iter_start = cpu_now();
            MTensor gt = cam.getGPUImage();

            // ── FD-check hook (Task 4): verify the analytic pose gradient
            // against finite differences. Active only under
            // MSPLAT_POSE_FD_CHECK=1 with --pose-opt, at step 100, for images
            // with a pose slot. The analytic gradient is evaluated by the fused
            // backward at the PRE-update params, so we snapshot those params here
            // and restore them before the ± perturbation renders, otherwise
            // Adam+SGLD would move the operating point out from under the test.
            static const bool kFdCheck = std::getenv("MSPLAT_POSE_FD_CHECK") != nullptr;
            const bool doFd = kFdCheck && poseOpt && step == 100 && cam.poseIndex >= 0;
            MTensor* fdParams[6] = { &model.means, &model.scales, &model.quats,
                                     &model.featuresDc, &model.featuresRest, &model.opacities };
            std::vector<std::vector<float>> fdSnap;
            if (doFd) {
                msplat_gpu_sync();                 // drain step 99; params stable at P0
                fdSnap.resize(6);
                for (int i = 0; i < 6; ++i) {
                    size_t nb = fdParams[i]->nbytes();
                    fdSnap[i].resize(nb / sizeof(float));
                    memcpy(fdSnap[i].data(), fdParams[i]->data_ptr(), nb);
                }
            }

            model.fullIteration(cam, step, gt, ssimWeight);

            if (doFd) {
                msplat_gpu_sync();                 // drain step 100 (Adam+SGLD+MCMC)
                float analytic[6];
                for (int k = 0; k < 6; ++k) analytic[k] = model.lastPoseGrad[k];
                // Restore P0 so the FD renders share the analytic grad's operating point.
                for (int i = 0; i < 6; ++i)
                    memcpy(fdParams[i]->data_ptr(), fdSnap[i].data(), fdParams[i]->nbytes());
                auto fdLoss = [&]() -> float {
                    MTensor rgb = model.render(cam, (int)step);
                    msplat_gpu_sync();
                    MTensor rgbCpu = rgb.cpu();
                    MTensor gtCpu = dequantize_gt(cam.getGPUImage());
                    float ss = ssim_eval(rgbCpu, gtCpu);
                    float l1 = l1_loss(rgbCpu, gtCpu);
                    return ssimWeight * (1.0f - ss) + (1.0f - ssimWeight) * l1;
                };
                const char* nm[6] = {"rho_x","rho_y","rho_z","omega_x","omega_y","omega_z"};
                float L0 = fdLoss();
                fprintf(stderr, "FD pose-gradient check (image %d, step %zu, splats %lld) L0=%.8e\n",
                        cam.poseIndex, (size_t)step, (long long)model.means.size(0), L0);
                for (int k = 0; k < 6; ++k) {
                    float eps = (k < 3) ? 1e-4f : 1e-5f;
                    float& xi = model.poseOpt->xiImageRef(cam.poseIndex, k);
                    float x0 = xi;
                    xi = x0 + eps; float Lp = fdLoss();
                    xi = x0 - eps; float Lm = fdLoss();
                    xi = x0;
                    float num = (Lp - Lm) / (2.0f * eps);
                    float denom = std::max(std::fabs(analytic[k]), std::fabs(num));
                    float rel = denom > 0.0f ? std::fabs(analytic[k] - num) / denom : 0.0f;
                    bool signOk = (analytic[k] < 0.0f) == (num < 0.0f);
                    fprintf(stderr,
                        "FD  k=%d %-8s analytic=% .6e  numeric=% .6e  rel=%.3f  %s  Lp=%.8e Lm=%.8e dL=%.3e\n",
                        k, nm[k], analytic[k], num, rel, signOk ? "sign_ok" : "SIGN_MISMATCH",
                        Lp, Lm, Lp - Lm);
                }
                // Params intentionally left at P0 (the test run's output is discarded).
            }

            model.schedulersStep(step);
            model.mcmcAfterTrain(step);
            msplat_commit();

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
                          << ms << "ms/step (single-iter, gpu-synced)" << std::endl;
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
                    MTensor gt_cpu = dequantize_gt(tc.getGPUImage());
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

        // Stamp the step actually reached (early stop may end before numIters).
        int finalStep = stoppedAt ? (int)stoppedAt : numIters;

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, finalStep);

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], finalStep);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = dequantize_gt(testCams[i].getGPUImage());

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, finalStep);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = dequantize_gt(valCam->getGPUImage());

            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
