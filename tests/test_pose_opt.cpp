#include "../core/src/pose_opt.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

static bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

int main() {
    // 1. exp(0) is identity.
    float xi0[6] = {0,0,0,0,0,0}, E[16];
    se3_delta_matrix(xi0, E);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        assert(near(E[i*4+j], i == j ? 1.f : 0.f));

    // 2. Small rotation about z by 0.01 rad matches Rodrigues to 1e-6.
    float xiz[6] = {0,0,0, 0,0,0.01f};
    se3_delta_matrix(xiz, E);
    assert(near(E[0], std::cos(0.01f), 1e-6f) && near(E[1], -std::sin(0.01f), 1e-6f));
    assert(near(E[4], std::sin(0.01f), 1e-6f) && near(E[5],  std::cos(0.01f), 1e-6f));

    // 3. Translation lands in the last column.
    float xit[6] = {0.5f, -0.25f, 0.125f, 0,0,0};
    se3_delta_matrix(xit, E);
    assert(near(E[3], 0.5f) && near(E[7], -0.25f) && near(E[11], 0.125f));

    // 4. composeViewmat fast-path: zero delta returns false.
    PoseOpt po(4, 2, 5e-4f, 0);
    float base[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}, out[16];
    assert(!po.composeViewmat(0, 0, base, out));

    // 5. Adam descends a toy quadratic L = 0.5*|xi|^2 (grad = xi_target drive):
    //    feed constant grad, expect xi moves opposite the gradient.
    float g[6] = {1,0,0,0,0,0};
    for (int s = 1; s <= 50; s++) po.step(1, 1, g, s, 100000);
    float composed[16];
    assert(po.composeViewmat(1, 1, base, composed));      // delta is now nonzero
    assert(composed[3] < 0.0f);                            // moved along -rho_x

    // 6. Save/load round-trip.
    std::vector<std::string> names = {"cam0/a.jpg","cam0/b.jpg","cam1/a.jpg","cam1/b.jpg"};
    assert(po.save("/tmp/pose_opt_test.json", names));
    PoseOpt po2(4, 2, 5e-4f, 0);
    assert(po2.load("/tmp/pose_opt_test.json", names));
    float out2[16];
    assert(po2.composeViewmat(1, 1, base, out2));
    for (int i = 0; i < 16; i++) assert(near(out2[i], composed[i], 1e-6f));

    printf("test_pose_opt: ALL PASS\n");
    return 0;
}
