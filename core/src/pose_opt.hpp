// pose_opt.hpp
//
// Host-side SE(3) delta math + hierarchical (per-camera + per-image) lazy-Adam
// optimizer state for per-image camera-pose refinement ("pose-opt").
//
// Pure C++17. No Metal, no MTensor, no engine dependencies — this file must
// compile standalone with `clang++ -std=c++17 pose_opt.cpp`.
//
// Parameterization (spec §4 P1a): decoupled rotation+translation
//   E(xi) = [ Rodrigues(omega), rho ; 0 0 0 1 ]
// where xi = [rho_xyz, omega_xyz] (rho = translation, omega = rotation, in
// that order). This is NOT the coupled SE(3) exponential map, but it is
// identical to it to first order at xi = 0, which is where all gradients in
// this system are derived, so the distinction does not affect correctness
// of the optimization.
#pragma once

#include <array>
#include <string>
#include <vector>

// Hierarchy constants.
// xi_i (per-image) uses kImageLrScale * lr as its learning rate; xi_c
// (per-camera) uses the full lr. Weight decay pulls each delta back toward
// zero (amendment 2 structure) and is expressed as a fraction of lr_t.
constexpr float kImageLrScale = 0.5f;
constexpr float kCamWeightDecay = 1e-3f;
constexpr float kImageWeightDecay = 1e-2f;

// Gauge / over-correction control (added after the Task-4 validation showed the
// poses drifting to ~5x the physical stitch correction). Adam is scale-
// invariant, so it keeps taking ~lr-sized steps even when the true pose
// gradient is tiny near the optimum, drifting the poses until the (weak) decay
// balances it — and that drift equilibrium is independent of lr. Two hard
// controls tame it: (1) clamp each delta near the physical correction scale
// (~1.3px / 859px focal = 1.5e-3 rad), and (2) zero-mean the per-cam deltas so
// their unobservable common mode (a free global rig rotation/translation)
// cannot accumulate.
constexpr float kMaxDeltaRot = 3.0e-3f;    // radians (2x physical, generous)
constexpr float kMaxDeltaTrans = 3.0e-3f;  // normalized scene units

// Compose the SE(3) delta matrix for a 6-vector xi = [rho_xyz, omega_xyz]:
//   out4x4 = [ Rodrigues(omega), rho ; 0 0 0 1 ]  (row-major 4x4)
// Guards theta = |omega| < 1e-12 by returning the identity rotation (avoids
// division by zero in the Rodrigues formula; correct in the limit).
void se3_delta_matrix(const float xi[6], float out4x4[16]);

// Row-major 4x4 matrix multiply: out = A * B. Safe if out aliases A and/or B.
void matmul4x4(const float* A, const float* B, float* out);

// Per-image + per-camera (rig) SE(3) delta state with hierarchical lazy-Adam
// optimization and JSON-ish persistence.
struct PoseOpt {
    // nImages = training image count; nCams = distinct rig cams (16; 0 if none).
    PoseOpt(int nImages, int nCams, float lr, int startStep);

    // Compose delta for image i (rigCam c, -1 if none) onto base row-major 4x4
    // world->cam viewmat; writes composed 4x4 into out. Returns false (and
    // leaves out untouched) when the total delta is exactly zero (fast path).
    bool composeViewmat(int imageIdx, int rigCam, const float* baseVm, float* outVm) const;

    // Accumulate one step's gradient (6 floats: [rho_xyz, omega_xyz]) and run
    // one lazy-Adam update on xi_i(imageIdx) and xi_c(rigCam) with weight decay.
    void step(int imageIdx, int rigCam, const float grad6[6], int trainStep, int maxSteps);

    // TTA: update ONLY xi_i(imageIdx) (used for held-out eval alignment).
    void stepImageOnly(int imageIdx, const float grad6[6]);

    // FD-check hook only (Task 4): mutable access to a single per-image delta
    // component so the finite-difference test can perturb xi_i(imageIdx)[k] by
    // +/- eps, re-render, and compare against the analytic gradient. Not used
    // on any production path.
    float& xiImageRef(int imageIdx, int k) { return xiImage_[imageIdx][k]; }

    bool save(const std::string& jsonPath, const std::vector<std::string>& names) const;
    bool load(const std::string& jsonPath, const std::vector<std::string>& names);

    float lr;
    int startStep;

private:
    int nImages_;
    int nCams_;

    // Delta state.
    std::vector<std::array<float, 6>> xiImage_;
    std::vector<std::array<float, 6>> xiCam_;

    // Adam moments (per-parameter), separate for image and cam hierarchy.
    std::vector<std::array<float, 6>> mImage_, vImage_;
    std::vector<std::array<float, 6>> mCam_, vCam_;

    // Per-slot visit counters for lazy-Adam bias correction (only the slots
    // actually touched by step()/stepImageOnly() advance their own counter).
    std::vector<int> tImage_;
    std::vector<int> tCam_;
};
