#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>

#include "solver_state.h"
#include "fluid_dynamics.h"

using namespace ablation;

double analyticalPoiseuilleVelocity(double y, double z, double a, double b, double nu, double dp_dx)
{
    double u_x = 0.0;
    const double PI = 3.14159265358979323846;

    // 前置常数项系数
    double coeff = (16.0 * a * a) / (nu * PI * PI * PI) * dp_dx;

    // 级数求和
    for (int i = 1; i <= 99; i += 2)
    {
        double term1 = std::pow(-1.0, (i - 1) / 2);

        double cosh_bottom = std::cosh(i * PI * b / (2.0 * a));
        double term2 = 1.0 - (std::cosh(i * PI * z / (2.0 * a)) / cosh_bottom);

        double term3 = std::cos(i * PI * y / (2.0 * a)) / (i * i * i);

        u_x += term1 * term2 * term3;
    }

    return coeff * u_x;
}

// 导出中心轴截面速度至 CSV，方便后续画图
void exportVerificationCSV(const LBMSolverState& state, const std::string& filename)
{
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }

    out << std::fixed << std::setprecision(8);
    out << "y_lattice,ux_sim,ux_analytical\n";

    int x = state.nx / 2;
    int z = state.nz / 2; 

    double a = 10.0;
    double b = 10.0;

    double nu = KINEMATIC_VISCOSITY;   
    double dp_dx = (0.335 - 0.333) / (state.nx - 1.0);

    double center_y = (state.ny - 1.0) / 2.0;
    double center_z = (state.nz - 1.0) / 2.0;

    for (int y = 0; y < state.ny; ++y)
    {
        int idx = state.getIndex(x, y, z);
        double y_coord = y - center_y;
        double z_coord = z - center_z;

        double u_out = 0.0;
        if (state.node_type[idx] != SOLID_BOUNDARY)
        {
            u_out = state.ux[idx] / state.rho[idx]; // 动量除以密度得到真实速度
        }
        double u_analytical = analyticalPoiseuilleVelocity(y_coord, z_coord, a, b, nu, dp_dx);

        if (state.node_type[idx] == SOLID_BOUNDARY)
        {
            u_analytical = 0.0;
        }
        out << y_coord << "," << u_out << "," << u_analytical << "\n";
    }

    out.close();
    std::cout << "Successfully exported verification data to: " << filename << std::endl;
}

int main()
{
    // 1. 初始化验证网格
    LBMSolverState state(40, 21, 21);

    std::cout << "Initializing Poiseuille flow verification" << std::endl;

    // 2. 设置四周固壁 (y=0, y=20, z=0, z=20 为固体边界)
    for (int z = 0; z < state.nz; ++z)
    {
        for (int y = 0; y < state.ny; ++y)
        {
            for (int x = 0; x < state.nx; ++x)
            {
                int idx = state.getIndex(x, y, z);

                if (y == 0 || y == state.ny - 1 || z == 0 || z == state.nz - 1)
                {
                    state.node_type[idx] = SOLID_BOUNDARY;
                }
                else
                {
                    state.node_type[idx] = FLUID;
                }
            }
        }
    }

    // 3. 初始化流场平衡态
    FluidDynamics::initFluid(state);

    int max_steps = 10000;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 4. 开启演化循环
    for (int step = 1; step <= max_steps; ++step)
    {
        // 依次执行：压力边界 -> 固壁 NEE 外推边界 -> 核心碰撞迁移
        FluidDynamics::applyPressureBoundary(state);
        FluidDynamics::applyWallBoundaryNEE(state);
        FluidDynamics::stepFluid(state);

        if (step % 1000 == 0)
        {
            std::cout << "Step: " << step << " completed." << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "Simulation Finished in " << diff.count() << " seconds!\n";
    std::cout << "Average speed: " << (max_steps * (40 * 21 * 21) / diff.count() / 1e6) << " MLUPS (Million Lattice Updates Per Second)\n";

    exportVerificationCSV(state, "poiseuille_verification.csv");

    return 0;
}