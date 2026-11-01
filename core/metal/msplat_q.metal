// Q1-Q3 (--quant-level 1) kernels: SH values and Adam moments held as block-scaled integer codes.
// Contract: docs/superpowers/2026-10-28-confidence-densification/runs/2026-11-01-hiz-q-yerlesim.md (layout),
// 2026-11-01-hiz-q-kodek-onkayit.md s0 (codecs, rounding, hash, domain). CPU reference: core/include/quant_codec.hpp.
//
// Compiled to its own .air and linked into the same default.metallib as msplat_metal.metal, which is not changed.
// The projection and SH helpers below are verbatim text copies of msplat_metal.metal inside namespace mq, compiled
// with the same default numeric settings (tests/q_copy_check.py checks the copies). Only the codec functions (encode,
// decode, Adam transform, hash) switch to safe math without contraction, function by function.
//
// Block = 256 Gaussians = one threadgroup in the SH backward. SH value and SH Adam bounds grow only (to 1.25 x max of
// the newly computed values; a grown SH value bound is at least 1e-3); when one grows the block's other rows are
// re-encoded (nearest); when it does not, rows whose values did not change are neither read nor written. Non-SH and DC
// Adam use the fixed bounds a_u = 8, h = 64 in every block (attempt 2, runs/2026-11-02-hiz-q2-onkayit.md s1); values
// beyond are clamped and counted (usat, qsat). A newly computed value byte-identical to the decoded old value keeps
// its code. log(1 + x) and exp(L) - 1 go through the deterministic q_log / q_exp (bit-identical to the CPU copies).
#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define CHANNELS 3

namespace mq {

// ---- verbatim copies from msplat_metal.metal (each block after "@copy" is byte-identical to source text;
// "@derived" blocks have a new signature and a body byte-identical to the named source function) ----

// @copy SH constants
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

// @copy
inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

// @copy
inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

// @copy
inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // Clamp axis-aligned bounding box to valid range [0, img_size).
    // Returns inclusive min, exclusive max.
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

// @copy
inline void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // Convert pixel-space center/radius to tile coordinates and compute AABB.
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// @copy
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// @copy
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

// @copy
inline bool finite_bits(float x) {
    return (as_type<uint>(x) & 0x7F800000u) != 0x7F800000u;
}

// @copy
inline bool finite_bits3(float3 v) {
    return finite_bits(v.x) && finite_bits(v.y) && finite_bits(v.z);
}

// @copy
inline bool finite_bits4(float4 v) {
    return finite_bits(v.x) && finite_bits(v.y) && finite_bits(v.z) && finite_bits(v.w);
}

// @copy
inline float3 exp_scale_clamped(const float3 log_scale) {
    return exp(clamp(log_scale, -15.f, 15.f));
}

// @copy
inline float3x3 quat_to_rotmat(const float4 quat) {
    // Epsilon floor: rsqrt(0) = Inf for a degenerate/zeroed quaternion.
    float s = rsqrt(max(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z,
        1e-8f
    ));
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    return float3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// @copy
inline bool clip_near_plane(
    const float3 p, 
    constant float *viewmat, 
    thread float3 &p_view, 
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

// @copy
inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// @copy
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, thread float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// @copy
float3 project_cov3d_ewa(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;

    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

// @copy
inline bool compute_cov2d_bounds(
    const float3 cov2d, 
    thread float3 &conic, 
    thread float &radius
) {
    // Invert 2x2 covariance (upper triangle in cov2d.xyz) to get the conic,
    // and compute the gaussian's screen-space radius from eigenvalues (3-sigma).
    // A valid (blurred) covariance has det ≥ ~0.09; reject near-zero, negative,
    // and NaN dets (the `!(>)` form catches NaN) — 1/tiny → Inf conic under fast-math.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (!(det > 1e-6f))
        return false;
    float inv_det = 1.f / det;

    // Conic = inverse of 2x2 covariance
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = sqrt(max(0.1f, b * b - det));
    // 3-sigma radius from the larger eigenvalue
    radius = ceil(3.f * sqrt(b + disc));
    return true;
}

// @copy
inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

// @copy
inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

// @copy
inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

// @copy
inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

// @copy
inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

// @copy
inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

// @copy
inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

// @copy
inline float3 project_pix_vjp(
    constant float *mat, const float3 p, const uint2 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y, 0.0f};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

// @copy
inline void cov2d_to_conic_vjp(
    float3 conic,
    float3 v_conic,
    thread float* v_cov2d
) {
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// @copy
void project_cov3d_ewa_vjp(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    thread float* v_cov3d,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

// @copy
inline float4 quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R) {
    // Same epsilon floor as quat_to_rotmat: rsqrt(0) = Inf.
    float s = rsqrt(max(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z,
        1e-8f
    ));
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

// @copy
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const thread float* v_cov3d,
    device float* v_scale,
    device float* v_quat
) {
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

// @derived sh_coeffs_to_color (thread-address-space rest coefficients; body identical)
inline void sh_coeffs_to_color_q(
    const uint degree,
    const float3 viewdir,
    constant float *dc_coeffs,
    thread const float *rest_coeffs,
    device float *colors
) {
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * dc_coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller (normalize() in project_and_sh_forward_kernel etc.)
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * rest_coeffs[0 * CHANNELS + c] +
                              z * rest_coeffs[1 * CHANNELS + c] -
                              x * rest_coeffs[2 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * rest_coeffs[3 * CHANNELS + c] +
             SH_C2[1] * yz * rest_coeffs[4 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * rest_coeffs[5 * CHANNELS + c] +
             SH_C2[3] * xz * rest_coeffs[6 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * rest_coeffs[7 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * rest_coeffs[8 * CHANNELS + c] +
             SH_C3[1] * xy * z * rest_coeffs[9 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * rest_coeffs[10 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 rest_coeffs[11 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * rest_coeffs[12 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * rest_coeffs[13 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * rest_coeffs[14 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * rest_coeffs[15 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * rest_coeffs[16 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * rest_coeffs[17 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * rest_coeffs[18 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 rest_coeffs[19 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * rest_coeffs[20 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 rest_coeffs[21 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * rest_coeffs[22 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 rest_coeffs[23 * CHANNELS + c]);
    }
}

// @derived adam_update_element (thread references; body identical)
inline void adam_update_q(
    thread float& param, thread float& ea, thread float& eas,
    float grad, float step_size, float beta1, float beta2, float bc2_sqrt, float eps
) {
    float m = fma(beta1, ea, (1.0f - beta1) * grad);
    float v = fma(beta2, eas, (1.0f - beta2) * grad * grad);
    param -= step_size * m / (sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

// ---- end of copies ----

// ---- Q codec. Same formulas and operation order as quant_codec.hpp and the harness reference (encS/encU/hashR):
// multiply, then divide; rint for nearest, floor(t + r) for stochastic; clamp to the code range. ----

#define QB 256
constant int Q_SH_M = 32767;                       // SH value: int16 symmetric
constant int Q_SHA_MU = 127, Q_SHA_ML = 255;       // SH Adam: int8 u, uint8 log_s
constant int Q_NSA_MU = 32767, Q_NSA_ML = 65535;   // non-SH Adam: int16 u, uint16 log_s
constant float Q_GROW = 1.25f;
constant float Q_H_MAX = 64.0f;                    // log_s bounds are capped here (qsat counts each cap)
constant float Q_ADAM_EPS = 1e-15f;
constant float Q_SH_A_MIN = 1e-3f;                 // floor of a grown non-zero SH value bound
constant float Q_NS_AU = 8.0f, Q_NS_H = 64.0f;     // fixed non-SH / DC Adam bounds
constant uint Q_KLO[3] = {0, 3, 8}, Q_KHI[3] = {3, 8, 15};  // SH degree d covers coefficients [Q_KLO[d], Q_KHI[d])

inline uint q_hash32(uint x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// Hash contract (s0): r = (hash32(step*2654435761 + cell*2246822519 + comp*3266489917) >> 8) * 2^-24.
// comp 0 = u, 1 = log_s, 2 = SH value; cell = flat cell index of the store; step = adam_step_count.
inline float q_hash_r(uint step, uint cell, uint comp) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return (float)(q_hash32(step * 2654435761u + cell * 2246822519u + comp * 3266489917u) >> 8) * 5.9604644775390625e-8f;
}
inline int q_enc_sym(float x, float a, int M, bool st, uint step, uint cell, uint comp) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (!(a > 0.0f)) return 0;
    const float t = precise::divide(x * (float)M, a);
    const float c = st ? floor(t + q_hash_r(step, cell, comp)) : rint(t);
    return (int)clamp(c, (float)-M, (float)M);
}
inline float q_dec_sym(int c, float a, int M) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return c == 0 ? 0.0f : precise::divide((float)c * a, (float)M);
}
inline int q_enc_uns(float x, float h, int M, bool st, uint step, uint cell, uint comp) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (!(h > 0.0f) || !(x > 0.0f)) return 0;
    const float t = precise::divide(x * (float)M, h);
    if (st) return (int)clamp(floor(t + q_hash_r(step, cell, comp)), 0.0f, (float)M);
    return (int)min(rint(t), (float)M);
}
inline float q_dec_uns(int c, float h, int M) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return c == 0 ? 0.0f : precise::divide((float)c * h, (float)M);
}
// Deterministic log / exp, copied from runs/hiz-checks/results/q2_2026-11-02/k.metal (bit-identical to quant_codec.hpp).
inline float q_log(float z) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    uint b = as_type<uint>(z);
    int e = (int)((b >> 23) & 0xFFu) - 127;
    float m = as_type<float>((b & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421356f) { m = m * 0.5f; e = e + 1; }
    float t = precise::divide(m - 1.0f, m + 1.0f);
    float t2 = t * t;
    float p = precise::divide(1.0f, 9.0f);
    p = p * t2 + precise::divide(1.0f, 7.0f);
    p = p * t2 + precise::divide(1.0f, 5.0f);
    p = p * t2 + precise::divide(1.0f, 3.0f);
    p = p * t2 + 1.0f;
    float lm = 2.0f * t * p;
    return (float)e * 0.693147180559945f + lm;
}
inline float q_exp(float L) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    float kf = L * 1.44269504088896f;
    int k = (int)kf;
    float r = L - (float)k * 0.693145751953125f;
    r = r - (float)k * 1.42860682030941723212e-6f;
    float p = precise::divide(1.0f, 5040.0f);
    p = p * r + precise::divide(1.0f, 720.0f);
    p = p * r + precise::divide(1.0f, 120.0f);
    p = p * r + precise::divide(1.0f, 24.0f);
    p = p * r + precise::divide(1.0f, 6.0f);
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return as_type<float>(as_type<uint>(p) + ((uint)k << 23));
}
// Adam moments as (u, log_s): u = m / (sqrt(v) + eps), log_s = q_log(1 + sqrt(v) / eps); decode sqrt(v) = eps*(q_exp(log_s) - 1).
inline float q_adam_u(float m, float v) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return precise::divide(m, precise::sqrt(v) + Q_ADAM_EPS);
}
inline float q_adam_logs(float v) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return q_log(1.0f + precise::divide(precise::sqrt(v), Q_ADAM_EPS));
}
inline void q_adam_decode(float u, float ls, thread float &m, thread float &v) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const float sv = ls == 0.0f ? 0.0f : Q_ADAM_EPS * (q_exp(ls) - 1.0f);
    v = sv * sv;
    m = u * (sv + Q_ADAM_EPS);
    if (u == 0.0f) m = 0.0f;
}
inline void q_dec_adam(int cu, int cl, float au, float h, int Mu, int Ml, thread float &m, thread float &v) {
    q_adam_decode(q_dec_sym(cu, au, Mu), q_dec_uns(cl, h, Ml), m, v);
}
// Grow-only bound: max > b grows b to max(1.25 * max, floor). cap > 0: the grown bound is held at cap and sat set (qsat).
inline float q_grow(float mx, float b, float cap, float floor_, thread bool &grown, thread bool &sat) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    grown = mx > b;
    sat = false;
    if (!grown) return b;
    float n = max(Q_GROW * mx, floor_);
    if (cap > 0.0f && n > cap) { n = cap; sat = true; }
    return n;
}

// Code packing: shq_state uint16 = int8 u (bits 0-7) | uint8 log_s (bits 8-15); nsq_state uint32 = int16 u | uint16 log_s.
inline int q_sh_u(ushort s) { int u = (int)(s & 0xFFu); return u > 127 ? u - 256 : u; }
inline int q_sh_l(ushort s) { return (int)(s >> 8); }
inline ushort q_sh_pack(int u, int l) { return (ushort)(((uint)u & 0xFFu) | ((uint)l << 8)); }
inline int q_ns_u(uint w) { int u = (int)(w & 0xFFFFu); return u > 32767 ? u - 65536 : u; }
inline int q_ns_l(uint w) { return (int)(w >> 16); }
inline uint q_ns_pack(int u, int l) { return ((uint)u & 0xFFFFu) | ((uint)l << 16); }

inline bool q_same(float a, float b) { return as_type<uint>(a) == as_type<uint>(b); }
// Reduction inputs: |x| and |u| count only when finite (a bound is never written non-finite); log_s counts
// unless NaN, so +inf demands h > 64 and is capped (qsat), as in the CPU reference.
inline float q_mag(float x) { return finite_bits(x) ? fabs(x) : 0.0f; }
inline float q_nn(float x) { return (as_type<uint>(x) & 0x7FFFFFFFu) > 0x7F800000u ? 0.0f : x; }

// Max over the block's threads of n <= 8 non-negative values, returned to every thread. red: [simdgroups][8].
// Every thread of the threadgroup must call it (two barriers).
inline void q_block_max(thread float *v, int n, threadgroup float *red, uint lane, uint sg, uint nsg) {
    for (int i = 0; i < n; i++) {
        const float s = simd_max(v[i]);
        if (lane == 0) red[sg * 8 + i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < n; i++) {
        float r = 0.0f;
        for (uint g = 0; g < nsg; g++) r = max(r, red[g * 8 + i]);
        v[i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One Adam cell of a block after the reduction: a changed cell takes its new (u, log_s) at the new bounds with the
// step's rounding; an unchanged cell is re-encoded nearest, component by component, only where its bound grew.
inline void q_adam_reencode(int cu, int cl, bool changed, float u, float ls, float au, float au2, bool gu, float h,
                            float h2, bool gl, int Mu, int Ml, bool st, uint step, uint cell, thread int &ou, thread int &ol) {
    if (changed) {
        cu = q_enc_sym(u, au2, Mu, st, step, cell, 0);
        cl = q_enc_uns(ls, h2, Ml, st, step, cell, 1);
    } else {
        if (gu) cu = q_enc_sym(q_dec_sym(cu, au, Mu), au2, Mu, false, 0u, 0u, 0u);
        if (gl) cl = q_enc_uns(q_dec_uns(cl, h, Ml), h2, Ml, false, 0u, 0u, 0u);
    }
    ou = cu; ol = cl;
}

}  // namespace mq

using namespace mq;

// ===== Forward: project_and_sh_forward_kernel with the SH coefficients decoded from fr_codes =====
// Bindings 0-23 as project_and_sh_forward_kernel (20 = fr_codes instead of features_rest); 24 = fr_bounds.
kernel void project_and_sh_forward_q(
    constant int& num_points [[buffer(0)]],
    constant float* means3d [[buffer(1)]],
    constant float* scales [[buffer(2)]],
    constant float& glob_scale [[buffer(3)]],
    constant float* quats [[buffer(4)]],
    constant float* viewmat [[buffer(5)]],
    constant float* projmat [[buffer(6)]],
    constant float4& intrins [[buffer(7)]],
    constant uint2& img_size [[buffer(8)]],
    constant uint3& tile_bounds [[buffer(9)]],
    constant float& clip_thresh [[buffer(10)]],
    device float* xys [[buffer(11)]],
    device float* depths [[buffer(12)]],
    device int* radii [[buffer(13)]],
    device float* conics [[buffer(14)]],
    device int32_t* num_tiles_hit [[buffer(15)]],
    constant uint& degree [[buffer(16)]],
    constant uint& degrees_to_use [[buffer(17)]],
    constant float3& cam_pos [[buffer(18)]],
    constant float* features_dc [[buffer(19)]],
    constant short* fr_codes [[buffer(20)]],
    device float* colors [[buffer(21)]],
    device float* aabb [[buffer(22)]],
    device atomic_uint* nan_counts [[buffer(23)]],
    constant float* fr_bounds [[buffer(24)]],
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    float3 scale = exp_scale_clamped(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    float local_cov3d[6];
    scale_rot_to_cov3d(scale, glob_scale, quat, local_cov3d);

    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        local_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return;
    }
    write_packed_float3(conics, idx, conic);

    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});

    float aabb_x = ceil(3.0f * sqrt(cov2d.x));
    float aabb_y = ceil(3.0f * sqrt(cov2d.z));
    uint2 tile_min, tile_max;
    get_tile_bbox(center, float2(aabb_x, aabb_y), (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);

    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
    aabb[idx * 2] = aabb_x;
    aabb[idx * 2 + 1] = aabb_y;

    float3 viewdir = normalize(p_world - cam_pos);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;
    // Q: decode the coefficients of the degrees in use (x = code * a_d / 32767, a_d = this block's degree-d bound)
    float rest[45];
    const uint nd = min(min(degrees_to_use, 3u), num_bases >= 16 ? 3u : num_bases >= 9 ? 2u : num_bases >= 4 ? 1u : 0u);
    for (uint d = 0; d < nd; d++) {
        const float a = fr_bounds[(idx / QB) * 3 + d];
        for (uint j = 3 * Q_KLO[d]; j < 3 * Q_KHI[d]; j++) rest[j] = q_dec_sym(fr_codes[rest_idx + j], a, Q_SH_M);
    }
    sh_coeffs_to_color_q(degrees_to_use, viewdir, &(features_dc[dc_idx]), rest, &(colors[idx_col]));

    float3 out_rgb = read_packed_float3(colors, idx);
    bool ok_center = finite_bits(center.x) && finite_bits(center.y);
    bool ok_depth  = finite_bits(p_view.z);
    bool ok_aabb   = finite_bits(aabb_x) && finite_bits(aabb_y);
    bool ok_conic  = finite_bits3(conic);
    bool ok_rgb    = finite_bits3(out_rgb);
    if (!(ok_center && ok_depth && ok_aabb && ok_conic && ok_rgb)) {
        radii[idx] = 0;
        num_tiles_hit[idx] = 0;
        atomic_fetch_add_explicit(nan_counts + 7, 1u, memory_order_relaxed);
        if (!ok_center) atomic_fetch_add_explicit(nan_counts + 8,  1u, memory_order_relaxed);
        if (!ok_depth)  atomic_fetch_add_explicit(nan_counts + 9,  1u, memory_order_relaxed);
        if (!ok_aabb)   atomic_fetch_add_explicit(nan_counts + 10, 1u, memory_order_relaxed);
        if (!ok_conic)  atomic_fetch_add_explicit(nan_counts + 11, 1u, memory_order_relaxed);
        if (!ok_rgb)    atomic_fetch_add_explicit(nan_counts + 12, 1u, memory_order_relaxed);
    }
}

// Packed hyperparameters of the Q backward (SHAdamParams + the Q rounding inputs).
struct QSHAdamParams {
    float dc_step_size;
    float dc_bc2_sqrt;
    float rest_step_size;
    float rest_bc2_sqrt;
    float beta1;
    float beta2;
    float eps;
    uint step;     // adam_step_count (hash key)
    uint qround;   // 0 nearest, 1 stochastic (SH values and Adam), 2 stochastic Adam only
};

// ===== Backward: project_and_sh_backward_kernel on the Q stores =====
// Threadgroup = 256 Gaussians = one block (dispatch ceil(N / 256) threadgroups of 256). Rows without radii > 0
// (or past num_points) take part in every barrier as inactive rows. Per row: projection gradients, then DC and SH
// Adam on the decoded state; per block: one round per open SH degree (round 0 also DC) = decode + Adam,
// max reduction of the changed cells, grow-only bounds, encode.
// Bindings 0-21 as project_and_sh_backward_kernel; 22-27 Q stores; 28 hyperparameters; 29 NaN counters; 30 qsat.
kernel void project_and_sh_backward_q(
    constant int& num_points [[buffer(0)]],
    constant float* means3d [[buffer(1)]],
    constant float* scales [[buffer(2)]],
    constant float& glob_scale [[buffer(3)]],
    constant float* quats [[buffer(4)]],
    constant float* viewmat [[buffer(5)]],
    constant float* projmat [[buffer(6)]],
    constant float4& intrins [[buffer(7)]],
    constant uint2& img_size [[buffer(8)]],
    constant int* radii [[buffer(9)]],
    constant float* conics [[buffer(10)]],
    constant float* v_xy [[buffer(11)]],
    constant float* v_depth [[buffer(12)]],
    constant float* v_conic [[buffer(13)]],
    device float* v_mean3d [[buffer(14)]],
    device float* v_scale [[buffer(15)]],
    device float* v_quat [[buffer(16)]],
    constant uint& degree [[buffer(17)]],
    constant uint& degrees_to_use [[buffer(18)]],
    constant float3& cam_pos [[buffer(19)]],
    constant float* v_colors [[buffer(20)]],
    device float* features_dc [[buffer(21)]],
    device short* fr_codes [[buffer(22)]],        // [rows, R, 3]
    device float* fr_bounds [[buffer(23)]],       // [nblk, 3]
    device ushort* shq_state [[buffer(24)]],      // [rows, R, 3]
    device float* shq_bounds [[buffer(25)]],      // [nblk, 3, 2] (a_u, h)
    device uint* dc_state [[buffer(26)]],         // nsq_state[featuresDc] [rows, 3]
    device float* dc_bounds [[buffer(27)]],       // nsq_bounds[featuresDc] [nblk, 2] (fixed; not read)
    constant QSHAdamParams& adam_hp [[buffer(28)]],
    device atomic_uint* nan_counts [[buffer(29)]],  // slot 3: nonfinite v_colors (SH grads skipped)
    device atomic_uint* qsat [[buffer(30)]],        // [0] qsat: log_s capped at 64; [1] usat: non-SH/DC |u| clamped at 8
    uint tid [[thread_index_in_threadgroup]],
    uint blk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    threadgroup float red[32 * 8];
    const uint idx = blk * QB + tid;
    const bool valid = idx < (uint)num_points;
    const bool active = valid && radii[idx] > 0;

    float3 p_world = float3(0.0f);
    if (active) {
    p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;

    // Projection backward: v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx,
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // z gradient contribution to mean3d
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx,
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // v_cov2d from v_conic (thread-local, no device memory round-trip)
    float local_v_cov2d[3];
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx),
        read_packed_float3(v_conic, idx),
        local_v_cov2d
    );

    // Recompute cov3d from scales+quats (avoids saving/reading 3.6MB tensor)
    float3 exp_scale = exp_scale_clamped(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    float local_cov3d[6];
    scale_rot_to_cov3d(exp_scale, glob_scale, quat, local_cov3d);

    // v_cov3d (thread-local) and v_mean3d contribution
    float tan_fovx = 0.5f * (float)img_size.x / fx;
    float tan_fovy = 0.5f * (float)img_size.y / fy;
    float3 p_view = transform_4x3(viewmat, p_world);
    float local_v_cov3d[6];
    project_cov3d_ewa_vjp(
        local_cov3d,
        viewmat,
        fx,
        fy,
        tan_fovx,
        tan_fovy,
        float3(local_v_cov2d[0], local_v_cov2d[1], local_v_cov2d[2]),
        &(v_mean3d[3*idx]),
        local_v_cov3d,
        p_view
    );

    // v_scale and v_quat (reads v_cov3d from thread-local)
    scale_rot_to_cov3d_vjp(
        exp_scale,
        glob_scale,
        quat,
        local_v_cov3d,
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
    // Chain rule: dL/d(log_s) = dL/d(exp_s) * exp(log_s)
    v_scale[3*idx + 0] *= exp_scale.x;
    v_scale[3*idx + 1] *= exp_scale.y;
    v_scale[3*idx + 2] *= exp_scale.z;
    }

    // ---- SH + DC Adam on the decoded state ----
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    const uint R = num_bases - 1;                  // SH coefficients per channel minus DC (row stride / 3)
    uint dc_idx = num_channels * idx;
    uint rest_idx = R * num_channels * idx;
    const bool st_sh = adam_hp.qround == 1u, st_ad = adam_hp.qround >= 1u;

    float vc[3] = {0.0f, 0.0f, 0.0f};
    float basis[15];
    bool sh_ok = false;
    if (active) {
        vc[0] = v_colors[dc_idx]; vc[1] = v_colors[dc_idx + 1]; vc[2] = v_colors[dc_idx + 2];
        // Adam boundary gate (as the fp32 kernel): a nonfinite v_colors skips every SH update of this row
        if (!(finite_bits(vc[0]) && finite_bits(vc[1]) && finite_bits(vc[2]))) {
            atomic_fetch_add_explicit(nan_counts + 3, 1u, memory_order_relaxed);
        } else {
            sh_ok = true;
        }
        float3 viewdir = normalize(p_world - cam_pos);
        float x = viewdir.x, y = viewdir.y, z = viewdir.z;
        float xx = x*x, xy = x*y, xz = x*z, yy = y*y, yz = y*z, zz = z*z;
        // SH degree 1 (3 bases)
        basis[0] = -SH_C1 * y; basis[1] = SH_C1 * z; basis[2] = -SH_C1 * x;
        // SH degree 2 (5 bases)
        basis[3] = SH_C2[0] * xy;
        basis[4] = SH_C2[1] * yz;
        basis[5] = SH_C2[2] * (2.f * zz - xx - yy);
        basis[6] = SH_C2[3] * xz;
        basis[7] = SH_C2[4] * (xx - yy);
        // SH degree 3 (7 bases)
        basis[8] = SH_C3[0] * y * (3.f * xx - yy);
        basis[9] = SH_C3[1] * xy * z;
        basis[10] = SH_C3[2] * y * (4.f * zz - xx - yy);
        basis[11] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        basis[12] = SH_C3[4] * x * (4.f * zz - xx - yy);
        basis[13] = SH_C3[5] * z * (xx - yy);
        basis[14] = SH_C3[6] * x * (xx - 3.f * yy);
    }

    const uint ndeg = R >= 15 ? 3u : R >= 8 ? 2u : R >= 3 ? 1u : 0u;
    const uint nd = min(min(degrees_to_use, 3u), ndeg);   // open degrees
    const uint rounds = max(nd, 1u);                       // round 0 also carries DC
    const uint step = adam_hp.step;

    for (uint rd = 0; rd < rounds; rd++) {
        const bool open = rd < nd;
        const uint j0 = 3 * Q_KLO[rd], j1 = open ? 3 * min(Q_KHI[rd], R) : j0;
        // old bounds, read by every thread before the first barrier of q_block_max; thread 0 writes after it
        const float a  = open ? fr_bounds[blk * 3 + rd] : 0.0f;
        const float au = open ? shq_bounds[blk * 6 + rd * 2] : 0.0f;
        const float h  = open ? shq_bounds[blk * 6 + rd * 2 + 1] : 0.0f;
        const float dau = Q_NS_AU, dh = Q_NS_H;   // DC Adam: fixed bounds (dc_bounds hold the same constants)

        // [0] SH value |x|, [1] SH Adam |u|, [2] SH Adam log_s (changed cells only)
        float mx[3] = {0.0f, 0.0f, 0.0f};
        float xs[21], us[21], ls[21], dcu[3], dcl[3];
        uint vmask = 0, amask = 0, dmask = 0, nusat = 0, nqsat = 0;

        if (sh_ok && rd == 0) {
            // DC: grad = SH_C0 * v_colors[c]
            for (uint c = 0; c < 3; c++) {
                const uint i = dc_idx + c;
                const uint w = dc_state[i];
                float mo, vo;
                q_dec_adam(q_ns_u(w), q_ns_l(w), dau, dh, Q_NSA_MU, Q_NSA_ML, mo, vo);
                float p = features_dc[i], m = mo, v = vo;
                float g = SH_C0 * vc[c];
                adam_update_q(p, m, v, g, adam_hp.dc_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.dc_bc2_sqrt, adam_hp.eps);
                features_dc[i] = p;
                if (!(q_same(m, mo) && q_same(v, vo))) {
                    dmask |= 1u << c;
                    dcu[c] = q_adam_u(m, v);
                    dcl[c] = q_adam_logs(v);
                    nusat += fabs(dcu[c]) > Q_NS_AU;
                    nqsat += dcl[c] > Q_NS_H;
                }
            }
        }
        if (sh_ok && open) {
            for (uint k = 0; k < 21; k++) {
                const uint j = j0 + k;
                if (j >= j1) break;
                const uint i = rest_idx + j;
                const float xo = q_dec_sym(fr_codes[i], a, Q_SH_M);
                const ushort s = shq_state[i];
                float mo, vo;
                q_dec_adam(q_sh_u(s), q_sh_l(s), au, h, Q_SHA_MU, Q_SHA_ML, mo, vo);
                float p = xo, m = mo, v = vo;
                float g = basis[j / 3] * vc[j % 3];
                adam_update_q(p, m, v, g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
                if (!q_same(p, xo)) {
                    vmask |= 1u << k;
                    xs[k] = p;
                    mx[0] = max(mx[0], q_mag(p));
                }
                if (!(q_same(m, mo) && q_same(v, vo))) {
                    amask |= 1u << k;
                    us[k] = q_adam_u(m, v);
                    ls[k] = q_adam_logs(v);
                    mx[1] = max(mx[1], q_mag(us[k]));
                    mx[2] = max(mx[2], q_nn(ls[k]));
                }
            }
        }

        q_block_max(mx, 3, red, lane, sg, nsg);

        bool gv, gu, gl, s0, s1, s2;
        const float a2 = q_grow(mx[0], a, 0.0f, Q_SH_A_MIN, gv, s0);
        const float au2 = q_grow(mx[1], au, 0.0f, 0.0f, gu, s1);
        const float h2 = q_grow(mx[2], h, Q_H_MAX, 0.0f, gl, s2);
        if (tid == 0) {
            if (gv) fr_bounds[blk * 3 + rd] = a2;
            if (gu) shq_bounds[blk * 6 + rd * 2] = au2;
            if (gl) shq_bounds[blk * 6 + rd * 2 + 1] = h2;
            if (s2) atomic_fetch_add_explicit(qsat, 1u, memory_order_relaxed);
        }
        if (nqsat) atomic_fetch_add_explicit(qsat, nqsat, memory_order_relaxed);
        if (nusat) atomic_fetch_add_explicit(qsat + 1, nusat, memory_order_relaxed);

        if (!valid) continue;
        if (open) {
            for (uint k = 0; k < 21; k++) {
                const uint j = j0 + k;
                if (j >= j1) break;
                const uint i = rest_idx + j;
                if ((vmask >> k) & 1u) fr_codes[i] = (short)q_enc_sym(xs[k], a2, Q_SH_M, st_sh, step, i, 2u);
                else if (gv) fr_codes[i] = (short)q_enc_sym(q_dec_sym(fr_codes[i], a, Q_SH_M), a2, Q_SH_M, false, 0u, 0u, 0u);
                const bool ac = (amask >> k) & 1u;
                if (ac || gu || gl) {
                    const ushort s = shq_state[i];
                    int cu, cl;
                    q_adam_reencode(q_sh_u(s), q_sh_l(s), ac, ac ? us[k] : 0.0f, ac ? ls[k] : 0.0f, au, au2, gu, h, h2, gl,
                                    Q_SHA_MU, Q_SHA_ML, st_ad, step, i, cu, cl);
                    shq_state[i] = q_sh_pack(cu, cl);
                }
            }
        }
        if (rd == 0) {
            for (uint c = 0; c < 3; c++) {
                const uint i = dc_idx + c;
                if ((dmask >> c) & 1u)   // fixed bounds: only changed cells are written
                    dc_state[i] = q_ns_pack(q_enc_sym(dcu[c], dau, Q_NSA_MU, st_ad, step, i, 0u),
                                            q_enc_uns(dcl[c], dh, Q_NSA_ML, st_ad, step, i, 1u));
            }
        }
    }
}

// ===== Non-SH Adam (means, scales, quats, opacity) on the Q stores: fused_adam_kernel with fixed bounds =====
// Dense like fused_adam_kernel: every row < num_rows is updated. Bounds fixed (a_u = 8, h = 64; bounds buffer holds the
// same constants and is not read), so there is no reduction; dispatched as ceil(N / 256) threadgroups of 256. Same NaN
// gate and per-group counter as fused_adam_kernel (a nonfinite gradient leaves the cell unchanged).
kernel void fused_adam_q16(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device uint* state [[buffer(2)]],            // nsq_state[g] [rows, dim]
    device float* bounds [[buffer(3)]],          // nsq_bounds[g] [nblk, 2] (fixed; not read)
    constant float& step_size [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& bc2_sqrt [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant uint& num_rows [[buffer(9)]],
    device atomic_uint* nan_counts [[buffer(10)]],
    constant uint& group_idx [[buffer(11)]],
    constant uint& dim [[buffer(12)]],           // cells per row (<= 4)
    constant uint& step [[buffer(13)]],          // adam_step_count (hash key)
    constant uint& qround [[buffer(14)]],
    device atomic_uint* qsat [[buffer(15)]],     // [0] qsat (log_s > 64), [1] usat (|u| > 8)
    uint row [[thread_position_in_grid]]
) {
    if (row >= num_rows) return;
    const bool st = qround >= 1u;
    uint nusat = 0, nqsat = 0;
    for (uint k = 0; k < 4; k++) {
        if (k >= dim) break;
        const uint i = row * dim + k;
        float g = grads[i];
        if (!finite_bits(g)) {
            atomic_fetch_add_explicit(nan_counts + group_idx, 1u, memory_order_relaxed);
            continue;
        }
        const uint w = state[i];
        float mo, vo;
        q_dec_adam(q_ns_u(w), q_ns_l(w), Q_NS_AU, Q_NS_H, Q_NSA_MU, Q_NSA_ML, mo, vo);
        float p = params[i], m = mo, v = vo;
        adam_update_q(p, m, v, g, step_size, beta1, beta2, bc2_sqrt, eps);
        params[i] = p;
        if (!(q_same(m, mo) && q_same(v, vo))) {
            const float u = q_adam_u(m, v), l = q_adam_logs(v);
            nusat += fabs(u) > Q_NS_AU;
            nqsat += l > Q_NS_H;
            state[i] = q_ns_pack(q_enc_sym(u, Q_NS_AU, Q_NSA_MU, st, step, i, 0u), q_enc_uns(l, Q_NS_H, Q_NSA_ML, st, step, i, 1u));
        }
    }
    if (nqsat) atomic_fetch_add_explicit(qsat, nqsat, memory_order_relaxed);
    if (nusat) atomic_fetch_add_explicit(qsat + 1, nusat, memory_order_relaxed);
}

// ===== mcmc_reg_grad_kernel with the Q junk test: junk <=> the three featuresDc log_s codes are 0 =====
// Bindings as mcmc_reg_grad_kernel; buffer 9 = nsq_state[featuresDc] (uint32 [rows, 3]) instead of exp_avg_sq.
kernel void mcmc_reg_grad_q(
    device const float* opacities [[buffer(0)]],
    device const float* scales    [[buffer(1)]],
    device float* v_opacity       [[buffer(2)]],
    device float* v_scale         [[buffer(3)]],
    constant int& N               [[buffer(4)]],
    constant float& op_reg        [[buffer(5)]],
    constant float& sc_reg        [[buffer(6)]],
    device const int* radii       [[buffer(7)]],
    constant float& invis_scale   [[buffer(8)]],
    device const uint* dc_state   [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N) return;
    const bool junk = ((dc_state[idx * 3] >> 16) | (dc_state[idx * 3 + 1] >> 16) | (dc_state[idx * 3 + 2] >> 16)) == 0u;
    float duty = junk ? 1.0f : invis_scale;
    (void)radii;
    float invN = duty / (float)N;
    float sig = 1.0f / (1.0f + exp(-opacities[idx]));
    v_opacity[idx] += op_reg * sig * (1.0f - sig) * invN;
    float inv3N = invN * (1.0f / 3.0f);
    for (int i = 0; i < 3; i++) {
        float s = scales[idx*3 + i];
        // Clamp the exp input (an overflowed exp would poison the Adam state).
        v_scale[idx*3 + i] += sc_reg * exp(clamp(s, -15.0f, 15.0f)) * inv3N;
    }
}

// ===== Codec parity kernels (tests/q_gpu_harness.mm --mode parity, pre-registration s3). Not used in training. =====
// Keys of the stochastic paths: cell = element index, step as given.
kernel void q_parity_sh_kernel(
    device const float* x [[buffer(0)]],
    device const int* codes_in [[buffer(1)]],
    device int* codes_out [[buffer(2)]],          // encode(x)
    device float* dec_out [[buffer(3)]],          // decode(codes_in)
    constant float& a [[buffer(4)]],
    constant uint& st [[buffer(5)]],
    constant uint& step [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    codes_out[i] = q_enc_sym(x[i], a, Q_SH_M, st != 0u, step, i, 2u);
    dec_out[i] = q_dec_sym(codes_in[i], a, Q_SH_M);
}
kernel void q_parity_adam_enc_kernel(
    device const float* mv [[buffer(0)]],         // [n][m, v]
    device int* codes_out [[buffer(1)]],          // [n][u code, log_s code]
    constant float& au [[buffer(2)]],
    constant float& h [[buffer(3)]],
    constant int& Mu [[buffer(4)]],
    constant int& Ml [[buffer(5)]],
    constant uint& st [[buffer(6)]],
    constant uint& step [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const float m = mv[2 * i], v = mv[2 * i + 1];
    codes_out[2 * i] = q_enc_sym(q_adam_u(m, v), au, Mu, st != 0u, step, i, 0u);
    codes_out[2 * i + 1] = q_enc_uns(q_adam_logs(v), h, Ml, st != 0u, step, i, 1u);
}
kernel void q_parity_adam_dec_kernel(
    device const int* codes_in [[buffer(0)]],     // [n][u code, log_s code]
    device float* mv [[buffer(1)]],               // [n][m, v]
    constant float& au [[buffer(2)]],
    constant float& h [[buffer(3)]],
    constant int& Mu [[buffer(4)]],
    constant int& Ml [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    float m, v;
    q_dec_adam(codes_in[2 * i], codes_in[2 * i + 1], au, h, Mu, Ml, m, v);
    mv[2 * i] = m;
    mv[2 * i + 1] = v;
}
// q_log / q_exp parity (attempt 2, s3 addition): the k.metal test kernel on the codec functions above.
kernel void q_parity_logexp_kernel(
    device const float* z [[buffer(0)]],
    device const float* L [[buffer(1)]],
    device float* ol [[buffer(2)]],
    device float* oe [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    ol[i] = q_log(z[i]);
    oe[i] = q_exp(L[i]);
}
