#include "pose_opt.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kAdamBeta1 = 0.9f;
constexpr float kAdamBeta2 = 0.999f;
constexpr float kAdamEps = 1e-8f;

// One lazy-Adam update (with decoupled weight decay) applied in-place to a
// single 6-vector slot. `t` is the slot's own visit counter (lazy Adam: only
// slots that are actually touched advance their bias-correction counter).
void adamUpdate(std::array<float, 6>& xi, std::array<float, 6>& m, std::array<float, 6>& v,
                int& t, const float grad6[6], float lrT, float wd) {
    t += 1;
    const float bc1 = 1.0f - std::pow(kAdamBeta1, static_cast<float>(t));
    const float bc2 = 1.0f - std::pow(kAdamBeta2, static_cast<float>(t));
    for (int k = 0; k < 6; k++) {
        const float g = grad6[k];
        m[k] = kAdamBeta1 * m[k] + (1.0f - kAdamBeta1) * g;
        v[k] = kAdamBeta2 * v[k] + (1.0f - kAdamBeta2) * g * g;
        const float mHat = m[k] / bc1;
        const float vHat = v[k] / bc2;
        xi[k] -= lrT * mHat / (std::sqrt(vHat) + kAdamEps);
        xi[k] -= wd * lrT * xi[k];
    }
}

// ---- Minimal hand-rolled JSON-ish (de)serialization -----------------------
// This file has no include paths and must not depend on nlohmann or any
// other library; the persisted format is simple enough (fixed shape, no
// nested objects beyond one level) that a tiny hand-rolled scanner suffices.

void skipWs(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
}

bool expectChar(const std::string& s, size_t& pos, char c) {
    skipWs(s, pos);
    if (pos < s.size() && s[pos] == c) {
        pos++;
        return true;
    }
    return false;
}

bool isNumberChar(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' ||
           c == 'e' || c == 'E';
}

bool parseFloatArray6(const std::string& s, size_t& pos, float out[6]) {
    if (!expectChar(s, pos, '[')) return false;
    for (int k = 0; k < 6; k++) {
        skipWs(s, pos);
        size_t start = pos;
        while (pos < s.size() && isNumberChar(s[pos])) pos++;
        if (pos == start) return false;
        out[k] = std::strtof(s.substr(start, pos - start).c_str(), nullptr);
        if (k < 5 && !expectChar(s, pos, ',')) return false;
    }
    return expectChar(s, pos, ']');
}

bool parseString(const std::string& s, size_t& pos, std::string& out) {
    if (!expectChar(s, pos, '"')) return false;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) pos++;
        out.push_back(s[pos]);
        pos++;
    }
    if (pos >= s.size()) return false;
    pos++;  // consume closing quote
    return true;
}

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void writeFloatArray6(std::ostream& os, const std::array<float, 6>& a) {
    os << "[";
    for (int k = 0; k < 6; k++) {
        os << std::setprecision(9) << a[k];
        if (k < 5) os << ",";
    }
    os << "]";
}

}  // namespace

void se3_delta_matrix(const float xi[6], float out4x4[16]) {
    const float rho[3] = {xi[0], xi[1], xi[2]};
    const float wx = xi[3], wy = xi[4], wz = xi[5];
    const float theta2 = wx * wx + wy * wy + wz * wz;
    const float theta = std::sqrt(theta2);

    float R[3][3];
    if (theta < 1e-12f) {
        R[0][0] = 1.f; R[0][1] = 0.f; R[0][2] = 0.f;
        R[1][0] = 0.f; R[1][1] = 1.f; R[1][2] = 0.f;
        R[2][0] = 0.f; R[2][1] = 0.f; R[2][2] = 1.f;
    } else {
        const float s = std::sin(theta) / theta;
        const float c = (1.0f - std::cos(theta)) / theta2;
        // Skew-symmetric K = [omega]_x
        const float K[3][3] = {
            {0.f, -wz, wy},
            {wz, 0.f, -wx},
            {-wy, wx, 0.f},
        };
        float K2[3][3];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                K2[a][b] = K[a][0] * K[0][b] + K[a][1] * K[1][b] + K[a][2] * K[2][b];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                R[a][b] = (a == b ? 1.f : 0.f) + s * K[a][b] + c * K2[a][b];
    }

    out4x4[0] = R[0][0]; out4x4[1] = R[0][1]; out4x4[2] = R[0][2]; out4x4[3] = rho[0];
    out4x4[4] = R[1][0]; out4x4[5] = R[1][1]; out4x4[6] = R[1][2]; out4x4[7] = rho[1];
    out4x4[8] = R[2][0]; out4x4[9] = R[2][1]; out4x4[10] = R[2][2]; out4x4[11] = rho[2];
    out4x4[12] = 0.f; out4x4[13] = 0.f; out4x4[14] = 0.f; out4x4[15] = 1.f;
}

void matmul4x4(const float* A, const float* B, float* out) {
    float tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.f;
            for (int k = 0; k < 4; k++) sum += A[i * 4 + k] * B[k * 4 + j];
            tmp[i * 4 + j] = sum;
        }
    }
    for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

PoseOpt::PoseOpt(int nImages, int nCams, float lr, int startStep)
    : lr(lr),
      startStep(startStep),
      nImages_(nImages),
      nCams_(nCams),
      xiImage_(nImages, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      xiCam_(nCams, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      mImage_(nImages, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      vImage_(nImages, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      mCam_(nCams, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      vCam_(nCams, std::array<float, 6>{0, 0, 0, 0, 0, 0}),
      tImage_(nImages, 0),
      tCam_(nCams, 0) {}

bool PoseOpt::composeViewmat(int imageIdx, int rigCam, const float* baseVm, float* outVm) const {
    float xiTotal[6];
    const auto& xiI = xiImage_[imageIdx];
    if (rigCam >= 0) {
        const auto& xiC = xiCam_[rigCam];
        for (int k = 0; k < 6; k++) xiTotal[k] = xiC[k] + xiI[k];
    } else {
        for (int k = 0; k < 6; k++) xiTotal[k] = xiI[k];
    }

    bool allZero = true;
    for (int k = 0; k < 6; k++) {
        if (xiTotal[k] != 0.f) {
            allZero = false;
            break;
        }
    }
    if (allZero) return false;

    float E[16];
    se3_delta_matrix(xiTotal, E);
    matmul4x4(E, baseVm, outVm);
    return true;
}

void PoseOpt::step(int imageIdx, int rigCam, const float grad6[6], int trainStep, int maxSteps) {
    float lrT;
    if (maxSteps > startStep) {
        const float frac = static_cast<float>(trainStep - startStep) /
                            static_cast<float>(maxSteps - startStep);
        lrT = lr * 0.5f * (1.0f + std::cos(kPi * frac));
    } else {
        lrT = lr;
    }

    adamUpdate(xiImage_[imageIdx], mImage_[imageIdx], vImage_[imageIdx], tImage_[imageIdx],
               grad6, kImageLrScale * lrT, kImageWeightDecay);

    if (rigCam >= 0) {
        adamUpdate(xiCam_[rigCam], mCam_[rigCam], vCam_[rigCam], tCam_[rigCam], grad6, lrT,
                   kCamWeightDecay);
    }
}

void PoseOpt::stepImageOnly(int imageIdx, const float grad6[6]) {
    // No trainStep/maxSteps is available at TTA time, so there is no cosine
    // schedule to evaluate; use the base per-image rate (kImageLrScale * lr)
    // directly, continuing this slot's own Adam visit count.
    adamUpdate(xiImage_[imageIdx], mImage_[imageIdx], vImage_[imageIdx], tImage_[imageIdx],
               grad6, kImageLrScale * lr, kImageWeightDecay);
}

bool PoseOpt::save(const std::string& jsonPath, const std::vector<std::string>& names) const {
    std::ofstream f(jsonPath);
    if (!f.is_open()) return false;

    f << "{\"version\":1,\"cams\":[";
    for (int c = 0; c < nCams_; c++) {
        writeFloatArray6(f, xiCam_[c]);
        if (c < nCams_ - 1) f << ",";
    }
    f << "],\"images\":{";
    const int n = std::min(static_cast<int>(names.size()), nImages_);
    for (int i = 0; i < n; i++) {
        f << "\"" << escapeJson(names[i]) << "\":";
        writeFloatArray6(f, xiImage_[i]);
        if (i < n - 1) f << ",";
    }
    f << "}}";

    return f.good();
}

bool PoseOpt::load(const std::string& jsonPath, const std::vector<std::string>& names) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    size_t pos = 0;
    if (!expectChar(s, pos, '{')) return false;

    const size_t versionKey = s.find("\"version\"", pos);
    if (versionKey == std::string::npos) return false;

    const size_t camsKey = s.find("\"cams\"", versionKey);
    if (camsKey == std::string::npos) return false;
    pos = camsKey + 6;  // strlen("\"cams\"")
    if (!expectChar(s, pos, ':')) return false;
    if (!expectChar(s, pos, '[')) return false;

    std::vector<std::array<float, 6>> cams;
    skipWs(s, pos);
    if (pos < s.size() && s[pos] == ']') {
        pos++;
    } else {
        while (true) {
            float arr[6];
            if (!parseFloatArray6(s, pos, arr)) return false;
            std::array<float, 6> a;
            for (int k = 0; k < 6; k++) a[k] = arr[k];
            cams.push_back(a);
            skipWs(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                pos++;
                continue;
            }
            break;
        }
        if (!expectChar(s, pos, ']')) return false;
    }

    const size_t imagesKey = s.find("\"images\"", pos);
    if (imagesKey == std::string::npos) return false;
    pos = imagesKey + 8;  // strlen("\"images\"")
    if (!expectChar(s, pos, ':')) return false;
    if (!expectChar(s, pos, '{')) return false;

    std::unordered_map<std::string, std::array<float, 6>> images;
    skipWs(s, pos);
    if (pos < s.size() && s[pos] == '}') {
        pos++;
    } else {
        while (true) {
            std::string key;
            if (!parseString(s, pos, key)) return false;
            if (!expectChar(s, pos, ':')) return false;
            float arr[6];
            if (!parseFloatArray6(s, pos, arr)) return false;
            std::array<float, 6> a;
            for (int k = 0; k < 6; k++) a[k] = arr[k];
            images[key] = a;
            skipWs(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                pos++;
                continue;
            }
            break;
        }
        if (!expectChar(s, pos, '}')) return false;
    }

    for (int c = 0; c < nCams_ && c < static_cast<int>(cams.size()); c++) xiCam_[c] = cams[c];
    for (size_t i = 0; i < names.size() && static_cast<int>(i) < nImages_; i++) {
        auto it = images.find(names[i]);
        if (it != images.end()) xiImage_[i] = it->second;
    }

    return true;
}
