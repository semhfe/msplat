#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

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
        int positionLrMaxSteps = 30000,
        // false = disable the 75%→50% seed auto-subsample (fixed-seed experiments;
        // app-side decimation then owns the seed count entirely).
        bool decimateSeeds = true,
        // Seed initialization, reference 3dgs-mcmc create_from_pcd: opacity 0.5,
        // scale 0.1 x the kNN spacing (see the seed-fog note in the constructor).
        float initOpacity = 0.5f,
        float initScaleMult = 0.1f);

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
  };
  CamSetup prepareCam(Camera& cam, int step);
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight);
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
  // Reg visibility duty-cycle: reg-gradient multiplier for Gaussians NOT visible in
  // the current training view. 1.0 = exact reference 3dgs-mcmc (assumes dense per-step
  // visibility); for sparse-visibility captures (360 crop rigs) set to the scene's
  // measured duty f = meanObservationMultiplicity / nImages (tools/metrics/cap_formula.py)
  // or live Gaussians starve (~1/f reg-down steps per photometric step).
  float reg_visibility_duty = 1.0f;
  // MCMC: cull Gaussians whose normalized-space position ||mean|| > cull_radius.
  // 0 (or negative) disables the cull. Default 3.0 = 3× the scene's unit-sphere.
  float cull_radius = 0.0f;
  // (Eq. 9 relocation is unconditional — see Model::computeRelocation. The
  // 2026-08-15 audit found two coupled defects there: the ensemble size was
  // off by one, and the scale rule was a naive sqrt(1/N). Both are fixed;
  // A/B at 30k on garden -d4 cap 2M measured neutral-to-better.)
  // Position LR schedule endpoint: LR decays from lr_init to lr_final over this many steps.
  // Default 30000 (reference MCMC). Partitioned training sets this to the cell's iteration count.
  int position_lr_max_steps_ = 30000;

  float scale;
  float translation[3] = {};

private:
  void relocate(const std::vector<int>& dead_indices, const std::vector<int>& alive_indices);
  int addNewGaussians();
  void computeRelocation(float opacity_old_sig, float* scale_old_exp, int N,
                         float& opacity_new_sig, float* scale_new_exp);
};

#endif
