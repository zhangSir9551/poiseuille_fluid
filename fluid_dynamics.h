#pragma once

#include "solver_state.h"

namespace ablation
{
    // D3Q19 离散速度方向
    static constexpr int cx19[19] = { 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 };
    static constexpr int cy19[19] = { 0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 };
    static constexpr int cz19[19] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 };

    class FluidDynamics
    {
    public:
        static void stepFluid(LBMSolverState& state);
        static void applyPressureBoundary(LBMSolverState& state);
        static void applyWallBoundaryNEE(LBMSolverState& state);

        static void initFluid(LBMSolverState& state)
        {
            for (int idx = 0; idx < state.num_nodes; ++idx)
            {
                // 无差别初始化所有节点，消除奇异初始值
                double feq[19];
                computeMrtFeq19(1.0, 0.0, 0.0, 0.0, feq);
                for (int i = 0; i < Q19; ++i)
                {
                    state.f[static_cast<size_t>(idx) * 19 + i] = feq[i];
                    state.f_post[static_cast<size_t>(idx) * 19 + i] = feq[i];
                }
            }
        }

        static inline void computeMrtFeq19(double rho, double ux, double uy, double uz, double* f_eq)
        {
            double jx = ux;
            double jy = uy;
            double jz = uz;
            double j_sq = jx * jx + jy * jy + jz * jz;

			double delta_rho = rho - 1.0;

            // 平衡态矩 (不可压缩近似)
            double m[19];
            m[0] = rho;
            m[1] = -11.0 - 11.0 * delta_rho + 19.0 * j_sq;
            m[2] = W_EPS + W_EPS * delta_rho + W_EPS_J * j_sq;
            m[3] = jx;
            m[4] = -2.0 / 3.0 * jx;
            m[5] = jy;
            m[6] = -2.0 / 3.0 * jy;
            m[7] = jz;
            m[8] = -2.0 / 3.0 * jz;
            m[9] = 2.0 * jx * jx - (jy * jy + jz * jz);
            m[10] = W_XX * m[9];
            m[11] = jy * jy - jz * jz;
            m[12] = W_XX * m[11];
            m[13] = jx * jy;
            m[14] = jy * jz;
            m[15] = jx * jz;
            m[16] = 0.0;
            m[17] = 0.0;
            m[18] = 0.0;

            // 逆矩阵 M^-1 投影回离散速度空间
            double s_m[19];
            s_m[0] = m[0] * (1.0 / 19.0);
            s_m[1] = m[1] * (1.0 / 2394.0);
            s_m[2] = m[2] * (1.0 / 252.0);
            s_m[3] = m[3] * (1.0 / 10.0);
            s_m[4] = m[4] * (1.0 / 40.0);
            s_m[5] = m[5] * (1.0 / 10.0);
            s_m[6] = m[6] * (1.0 / 40.0);
            s_m[7] = m[7] * (1.0 / 10.0);
            s_m[8] = m[8] * (1.0 / 40.0);
            s_m[9] = m[9] * (1.0 / 36.0);
            s_m[10] = m[10] * (1.0 / 72.0);
            s_m[11] = m[11] * (1.0 / 12.0);
            s_m[12] = m[12] * (1.0 / 24.0);
            s_m[13] = m[13] * (1.0 / 4.0);
            s_m[14] = m[14] * (1.0 / 4.0);
            s_m[15] = m[15] * (1.0 / 4.0);
            s_m[16] = m[16] * (1.0 / 8.0);
            s_m[17] = m[17] * (1.0 / 8.0);
            s_m[18] = m[18] * (1.0 / 8.0);

            f_eq[0] = s_m[0] - 30.0 * s_m[1] + 12.0 * s_m[2];
            f_eq[1] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[3] - 4.0 * s_m[4] + 2.0 * s_m[9] - 4.0 * s_m[10];
            f_eq[2] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[3] + 4.0 * s_m[4] + 2.0 * s_m[9] - 4.0 * s_m[10];
            f_eq[3] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[5] - 4.0 * s_m[6] - s_m[9] + 2.0 * s_m[10] + s_m[11] - 2.0 * s_m[12];
            f_eq[4] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[5] + 4.0 * s_m[6] - s_m[9] + 2.0 * s_m[10] + s_m[11] - 2.0 * s_m[12];
            f_eq[5] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[7] - 4.0 * s_m[8] - s_m[9] + 2.0 * s_m[10] - s_m[11] + 2.0 * s_m[12];
            f_eq[6] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[7] + 4.0 * s_m[8] - s_m[9] + 2.0 * s_m[10] - s_m[11] + 2.0 * s_m[12];
            f_eq[7] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] + s_m[5] + s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] + s_m[13] + s_m[16] - s_m[17];
            f_eq[8] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] + s_m[5] + s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] - s_m[13] - s_m[16] - s_m[17];
            f_eq[9] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] - s_m[5] - s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] - s_m[13] + s_m[16] + s_m[17];
            f_eq[10] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] - s_m[5] - s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] + s_m[13] - s_m[16] + s_m[17];
            f_eq[11] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] + s_m[7] + s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] + s_m[15] - s_m[16] + s_m[18];
            f_eq[12] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] + s_m[7] + s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] - s_m[15] + s_m[16] + s_m[18];
            f_eq[13] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] - s_m[7] - s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] - s_m[15] - s_m[16] - s_m[18];
            f_eq[14] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] - s_m[7] - s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] + s_m[15] + s_m[16] - s_m[18];
            f_eq[15] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[5] + s_m[6] + s_m[7] + s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] + s_m[14] + s_m[17] - s_m[18];
            f_eq[16] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[5] - s_m[6] + s_m[7] + s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] - s_m[14] - s_m[17] - s_m[18];
            f_eq[17] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[5] + s_m[6] - s_m[7] - s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] - s_m[14] + s_m[17] + s_m[18];
            f_eq[18] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[5] - s_m[6] - s_m[7] - s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] + s_m[14] - s_m[17] + s_m[18];
        }
    };

} // namespace ablation
