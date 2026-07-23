#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "pose_opt.hpp"

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);

// Dequantize a uint8 MTensor to float [0,1]. Used at metric evaluation sites
// where psnr/ssim_eval/l1_loss expect float data but GT is stored as uint8.
// Returns the input unchanged if it's already Float32.
MTensor dequantize_gt(const MTensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int shDegree, int shDegreeInterval,
        int capMax, float noiseLr, float opacityReg, float scaleReg,
        int maxSteps, bool keepCrs,
        const float* bgColor = nullptr,
        int positionLrMaxSteps = 30000);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers(bool decimateSeeds = true);
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void mcmcAfterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
    // Composed (or, in the fast path, cached-unchanged) view/proj-view
    // matrices for this call. Default aliases cam.cachedViewMat/cachedProjViewMat
    // (same tensor handle, no data copy); pose-opt overrides them in-place
    // when the per-image delta is non-zero.
    MTensor viewMat, projViewMat;
  };
  CamSetup prepareCam(Camera& cam, int step);
  // freezeGaussians (Task 5): when true, zeroes every Gaussian Adam step size
  // (means/scales/quats/featuresDc/featuresRest/opacities all share the same
  // adam_ss[] array, so this covers the SH path too) and skips the SGLD noise
  // + MCMC regularization calls, leaving every Gaussian parameter unchanged.
  // Used by TTA (val-view pose alignment) and later by Task 6's eval path.
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight, bool freezeGaussians = false);
  MTensor render(Camera& cam, int step);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();

  // MCMC: SGLD noise buffer (preallocated at cap_max * 3 floats)
  MTensor sgld_noise_buf;
  // MCMC: monotonic counter used to diversify RNG seeds across relocation calls
  int mcmc_relocation_count = 0;
  // Pre-allocated scratch space for relocate() / addNewGaussians() to avoid
  // 4–16 MB alloc/free every 100 training steps at cap_max=1M.
  std::vector<float> scratch_probs_;
  std::vector<int>   scratch_count_;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int shDegree;
  int shDegreeInterval;
  int maxSteps;
  bool keepCrs;

  // MCMC parameters
  int cap_max;
  float noise_lr;
  float opacity_reg;
  float scale_reg;
  // MCMC: cull Gaussians whose normalized-space position ||mean|| > cull_radius.
  // 0 (or negative) disables the cull. Default 3.0 = 3× the scene's unit-sphere.
  float cull_radius = 0.0f;
  // Position LR schedule endpoint: LR decays from lr_init to lr_final over this many steps.
  // Default 30000 (reference MCMC). Partitioned training sets this to the cell's iteration count.
  int position_lr_max_steps_ = 30000;

  float scale;
  float translation[3] = {};

  // Pose-opt (Task 2+): per-image SE(3) delta optimizer. Null = off (default);
  // constructed by the CLI only when --pose-opt is passed.
  std::unique_ptr<PoseOpt> poseOpt;
  // Pose-opt (Task 3): lazily-allocated scratch tensors for the composed
  // viewmat/projViewMat, reused across prepareCam calls (avoids an alloc per
  // call on the path that lasts for the entire training run once composing
  // is active). Untouched (left undefined) on the flag-off / zero-delta path.
  MTensor poseScratchViewMat, poseScratchProjViewMat;

  // Pose-opt (Task 4): analytic per-image pose gradient dL/dxi for the last
  // fullIteration where pose-opt was enabled. Layout [rho_xyz, omega_xyz].
  // Computed as a byproduct of the projection backward (GPU per-threadgroup
  // reduction into a partials buffer, CPU-summed here). Consumed by Task 5's
  // optimizer step. Written ONLY on iterations where pose-opt is active for the
  // image; retains its prior value otherwise (init 0), so Task 5 must gate its
  // reads on the same poseEnabled condition rather than assume a fresh zero.
  float lastPoseGrad[6] = {};
  // Non-owning handle to the GPU per-threadgroup pose-gradient partials buffer
  // (owned by the engine's tensor cache); filled by msplat_train_step when
  // pose-opt is enabled so fullIteration can CPU-sum it. Untouched when off.
  MTensor poseGradPartials;

private:
  void relocate(const std::vector<int>& dead_indices, const std::vector<int>& alive_indices);
  int addNewGaussians();
  void computeRelocation(float opacity_old_sig, float* scale_old_exp, int N,
                         float& opacity_new_sig, float* scale_new_exp);
};

#endif
