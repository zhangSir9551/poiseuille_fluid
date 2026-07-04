#include "fluid_dynamics.h"

namespace ablation
{
    static int inv_dir19[19] = { -1 };
    static bool inv_initialized19 = false;

    static void initInverseDirections19()
    {
        if (inv_initialized19) return;
        for (int i = 0; i < Q19; ++i) 
        {
            for (int j = 0; j < Q19; ++j) 
            {
                if (cx19[i] + cx19[j] == 0 && cy19[i] + cy19[j] == 0 && cz19[i] + cz19[j] == 0) 
                {
                    inv_dir19[i] = j;
                    break;
                }
            }
        }
        inv_initialized19 = true;
    }

    void FluidDynamics::stepFluid(LBMSolverState& state)
    {
        initInverseDirections19();

        // 松弛参数
        double omega = 1.0 / TAU;
        double s1 = 1.19;
        double s2 = 1.4;
        double s4 = 1.2;
        double s9 = omega;
        double s10 = 1.4;
        double s13 = omega;
        double s16 = 1.98;

        #pragma omp parallel for collapse(3) schedule(static)
        for (int z = 1; z < state.nz - 1; ++z)
        {
            for (int y = 1; y < state.ny - 1; ++y)
            {
                for (int x = 1; x < state.nx - 1; ++x)
                {
                    int idx = state.getIndex(x, y, z);

                    if (state.node_type[idx] == FLUID || state.node_type[idx] == INTERPHASE_NODE)
                    {
                        double f[19];
                        double rho = 0.0;
                        double jx = 0.0, jy = 0.0, jz = 0.0;

                        // Pull Stream
                        for (int i = 0; i < Q19; ++i)
                        {
                            int nx = x - cx19[i];
                            int ny = y - cy19[i];
                            int nz = z - cz19[i];
                            int n_idx = state.getIndex(nx, ny, nz);

                            bool is_solid = (state.node_type[n_idx] == SOLID_BOUNDARY);
                            int pull_i = is_solid ? inv_dir19[i] : i;
                            int pull_idx = is_solid ? idx : n_idx;

                            double val_f = state.f[static_cast<size_t>(pull_i) * state.num_nodes + pull_idx];

                            f[i] = val_f;
                            rho += val_f;
                            jx += val_f * cx19[i];
                            jy += val_f * cy19[i];
                            jz += val_f * cz19[i];
                        }

                        if (rho < 1e-6) 
                        {
                            rho = 1.0;
                            jx = 0;
                            jy = 0;
							jz = 0;
                        }

                        state.rho[idx] = rho;
                        state.ux[idx] = jx;
                        state.uy[idx] = jy;
                        state.uz[idx] = jz;

                        double j_sq = jx * jx + jy * jy + jz * jz;

                        // 投影至矩空间
                        double m[19];
                        m[0] = rho;
                        m[1] = -30 * f[0] - 11 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6]) + 8 * (f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] + f[14] + f[15] + f[16] + f[17] + f[18]);
                        m[2] = 12 * f[0] - 4 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6]) + 1 * (f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] + f[14] + f[15] + f[16] + f[17] + f[18]);
                        m[3] = jx;
                        m[4] = -4 * (f[1] - f[2]) + 1 * (f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]);
                        m[5] = jy;
                        m[6] = -4 * (f[3] - f[4]) + 1 * (f[7] + f[8] - f[9] - f[10] + f[15] - f[16] + f[17] - f[18]);
                        m[7] = jz;
                        m[8] = -4 * (f[5] - f[6]) + 1 * (f[11] + f[12] - f[13] - f[14] + f[15] + f[16] - f[17] - f[18]);
                        m[9] = 2 * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6]) + 1 * (f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] + f[14]) - 2 * (f[15] + f[16] + f[17] + f[18]);
                        m[10] = -4 * (f[1] + f[2]) + 2 * (f[3] + f[4] + f[5] + f[6]) + 1 * (f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] + f[14]) - 2 * (f[15] + f[16] + f[17] + f[18]);
                        m[11] = (f[3] + f[4]) - (f[5] + f[6]) + 1 * (f[7] + f[8] + f[9] + f[10]) - 1 * (f[11] + f[12] + f[13] + f[14]);
                        m[12] = -2 * (f[3] + f[4]) + 2 * (f[5] + f[6]) + 1 * (f[7] + f[8] + f[9] + f[10]) - 1 * (f[11] + f[12] + f[13] + f[14]);
                        m[13] = f[7] - f[8] - f[9] + f[10];
                        m[14] = f[15] - f[16] - f[17] + f[18];
                        m[15] = f[11] - f[12] - f[13] + f[14];
                        m[16] = f[7] - f[8] + f[9] - f[10] - f[11] + f[12] - f[13] + f[14];
                        m[17] = -f[7] - f[8] + f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
                        m[18] = f[11] + f[12] - f[13] - f[14] - f[15] - f[16] + f[17] + f[18];

                        // 提取密度扰动量以保留微小梯度精度
						double delta_rho = rho - 1.0;

                        // 矩空间碰撞弛豫
                        m[1] -= s1 * (m[1] - (-11.0 - 11.0 * delta_rho + 19.0 * j_sq));
                        m[2] -= s2 * (m[2] - (W_EPS + W_EPS * delta_rho + W_EPS_J * j_sq));
                        m[4] -= s4 * (m[4] - (-2.0 / 3.0 * jx));
                        m[6] -= s4 * (m[6] - (-2.0 / 3.0 * jy));
                        m[8] -= s4 * (m[8] - (-2.0 / 3.0 * jz));

                        double meq9 = 2.0 * jx * jx - (jy * jy + jz * jz);
                        m[9] -= s9 * (m[9] - meq9);
                        m[10] -= s10 * (m[10] - (W_XX * meq9));

                        double meq11 = jy * jy - jz * jz;
                        m[11] -= s9 * (m[11] - meq11);
                        m[12] -= s10 * (m[12] - (W_XX * meq11));

                        m[13] -= s13 * (m[13] - (jx * jy));
                        m[14] -= s13 * (m[14] - (jy * jz));
                        m[15] -= s13 * (m[15] - (jx * jz));
                        m[16] -= s16 * (m[16] - 0.0);
                        m[17] -= s16 * (m[17] - 0.0);
                        m[18] -= s16 * (m[18] - 0.0);

                        // 还原到物理空间
                        double s_m[19];
                        s_m[0] = m[0] / 19.0;
                        s_m[1] = m[1] / 2394.0;
                        s_m[2] = m[2] / 252.0;
                        s_m[3] = m[3] / 10.0;
                        s_m[4] = m[4] / 40.0;
                        s_m[5] = m[5] / 10.0;
                        s_m[6] = m[6] / 40.0;
                        s_m[7] = m[7] / 10.0;
                        s_m[8] = m[8] / 40.0;
                        s_m[9] = m[9] / 36.0;
                        s_m[10] = m[10] / 72.0;
                        s_m[11] = m[11] / 12.0;
                        s_m[12] = m[12] / 24.0;
                        s_m[13] = m[13] / 4.0;
                        s_m[14] = m[14] / 4.0;
                        s_m[15] = m[15] / 4.0;
                        s_m[16] = m[16] / 8.0;
                        s_m[17] = m[17] / 8.0;
                        s_m[18] = m[18] / 8.0;

                        double f_post[19];
                        f_post[0] = s_m[0] - 30.0 * s_m[1] + 12.0 * s_m[2];
                        f_post[1] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[3] - 4.0 * s_m[4] + 2.0 * s_m[9] - 4.0 * s_m[10];
                        f_post[2] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[3] + 4.0 * s_m[4] + 2.0 * s_m[9] - 4.0 * s_m[10];
                        f_post[3] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[5] - 4.0 * s_m[6] - s_m[9] + 2.0 * s_m[10] + s_m[11] - 2.0 * s_m[12];
                        f_post[4] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[5] + 4.0 * s_m[6] - s_m[9] + 2.0 * s_m[10] + s_m[11] - 2.0 * s_m[12];
                        f_post[5] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] + s_m[7] - 4.0 * s_m[8] - s_m[9] + 2.0 * s_m[10] - s_m[11] + 2.0 * s_m[12];
                        f_post[6] = s_m[0] - 11.0 * s_m[1] - 4.0 * s_m[2] - s_m[7] + 4.0 * s_m[8] - s_m[9] + 2.0 * s_m[10] - s_m[11] + 2.0 * s_m[12];

                        f_post[7] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] + s_m[5] + s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] + s_m[13] + s_m[16] - s_m[17];
                        f_post[8] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] + s_m[5] + s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] - s_m[13] - s_m[16] - s_m[17];
                        f_post[9] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] - s_m[5] - s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] - s_m[13] + s_m[16] + s_m[17];
                        f_post[10] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] - s_m[5] - s_m[6] + s_m[9] + s_m[10] + s_m[11] + s_m[12] + s_m[13] - s_m[16] + s_m[17];

                        f_post[11] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] + s_m[7] + s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] + s_m[15] - s_m[16] + s_m[18];
                        f_post[12] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] + s_m[7] + s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] - s_m[15] + s_m[16] + s_m[18];
                        f_post[13] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[3] + s_m[4] - s_m[7] - s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] - s_m[15] - s_m[16] - s_m[18];
                        f_post[14] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[3] - s_m[4] - s_m[7] - s_m[8] + s_m[9] + s_m[10] - s_m[11] - s_m[12] + s_m[15] + s_m[16] - s_m[18];

                        f_post[15] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[5] + s_m[6] + s_m[7] + s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] + s_m[14] + s_m[17] - s_m[18];
                        f_post[16] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[5] - s_m[6] + s_m[7] + s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] - s_m[14] - s_m[17] - s_m[18];
                        f_post[17] = s_m[0] + 8.0 * s_m[1] + s_m[2] + s_m[5] + s_m[6] - s_m[7] - s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] - s_m[14] + s_m[17] + s_m[18];
                        f_post[18] = s_m[0] + 8.0 * s_m[1] + s_m[2] - s_m[5] - s_m[6] - s_m[7] - s_m[8] - 2.0 * s_m[9] - 2.0 * s_m[10] + s_m[14] - s_m[17] + s_m[18];

                        // 写入更新后的状态
                        for (int i = 0; i < Q19; ++i)
                        {
                            state.f_post[static_cast<size_t>(i) * state.num_nodes + idx] = f_post[i];
                        }
                    }
                }
            }
        }

        applyPressureBoundary(state);
    }

    void FluidDynamics::applyPressureBoundary(LBMSolverState& state)
    {
        double rho_in = 1.005;
        double rho_out = 0.999;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < state.nz; ++z)
        {
            for (int y = 0; y < state.ny; ++y)
            {
                if (state.node_type[state.getIndex(0, y, z)] == SOLID_BOUNDARY) continue;

                // --- 入口端 (x = 0) ---
                int idx_in = state.getIndex(0, y, z);
                int idx_next = state.getIndex(1, y, z);

                double jx_in = state.ux[idx_next];
                double jy_in = state.uy[idx_next];
                double jz_in = state.uz[idx_next];
                double rho_next = state.rho[idx_next];

                state.rho[idx_in] = rho_in;
                state.ux[idx_in] = jx_in;
                state.uy[idx_in] = jy_in;
                state.uz[idx_in] = jz_in;

                // 使用 MRT 一致的平衡态分布进行边界非平衡推断
                double feq_in[19], feq_next[19];
                computeMrtFeq19(rho_in, jx_in, jy_in, jz_in, feq_in);
                computeMrtFeq19(rho_next, jx_in, jy_in, jz_in, feq_next);

                for (int i = 0; i < Q19; ++i)
                {
                    size_t mem_in = static_cast<size_t>(i) * state.num_nodes + idx_in;
                    size_t mem_next = static_cast<size_t>(i) * state.num_nodes + idx_next;

                    state.f[mem_in] = feq_in[i] + (state.f_post[mem_next] - feq_next[i]);
                    state.f_post[mem_in] = state.f[mem_in];
                }

                // --- 出口端 (x = Nx - 1) ---
                int idx_out = state.getIndex(state.nx - 1, y, z);
                int idx_prev = state.getIndex(state.nx - 2, y, z);

                double jx_out = state.ux[idx_prev];
                double jy_out = state.uy[idx_prev];
                double jz_out = state.uz[idx_prev];
                double rho_prev = state.rho[idx_prev];

                state.rho[idx_out] = rho_out;
                state.ux[idx_out] = jx_out;
                state.uy[idx_out] = jy_out;
                state.uz[idx_out] = jz_out;

                double feq_out[19], feq_prev[19];
                computeMrtFeq19(rho_out, jx_out, jy_out, jz_out, feq_out);
                computeMrtFeq19(rho_prev, jx_out, jy_out, jz_out, feq_prev);

                for (int i = 0; i < Q19; ++i)
                {
                    size_t mem_out = static_cast<size_t>(i) * state.num_nodes + idx_out;
                    size_t mem_prev = static_cast<size_t>(i) * state.num_nodes + idx_prev;

                    state.f[mem_out] = feq_out[i] + (state.f_post[mem_prev] - feq_prev[i]);
                    state.f_post[mem_out] = state.f[mem_out];
                }
            }
        }
    }

} // namespace ablation