# pragma once

# include <vector>
# include <cstddef>
# include <cmath>
# include <algorithm>

namespace ablation 
{
constexpr int Q19 = 19;
constexpr int Q7 = 7;

constexpr double KINEMATIC_VISCOSITY = 0.5;
constexpr double TAU = 3.0 * KINEMATIC_VISCOSITY + 0.5;
constexpr double W_EPS = 0.0;
constexpr double W_EPS_J = -475.0 / 63.0;
constexpr double W_XX = 0.0;

constexpr double TAU_C = 0.52; 
constexpr double TAU_T = 0.8;
constexpr double R_GAS = 8.314;       
constexpr double RHO_CARBON = 1900.0; 
constexpr double M_C = 0.012;

enum NodeType 
{
    FLUID = 0, SOLID_BOUNDARY = 1, INTERPHASE_NODE = 2
};

enum MaterialType
{
    MAT_CARBON_AEROGEL = 0
};

struct LBMSolverState 
{
    int nx, ny, nz, num_nodes;

    // 1. 流场 (D3Q19)
    std::vector<double> f, f_post;
    std::vector<double> rho, ux, uy, uz;

    // 2. 浓度场 (D3Q7)
    std::vector<double> h_c, h_c_post;
    std::vector<double> c_O2;

    // 3. 温度场 (D3Q7)
    std::vector<double> g, g_post;
    std::vector<double> temp;

    std::vector<double> S_c; // 氧气消耗源项
    std::vector<double> S_t; // 化学反应热源项

    // 4. 几何与材料
    std::vector<int> node_type;
    std::vector<int> original_material;
    std::vector<double> solid_fraction;

    LBMSolverState(int _nx, int _ny, int _nz) 
        : nx(_nx), ny(_ny), nz(_nz), num_nodes(_nx * _ny * _nz)
    {
        f.resize(Q19 * num_nodes, 0.0);
        f_post.resize(Q19 * num_nodes, 0.0);
        rho.resize(num_nodes, 1.0);
        ux.resize(num_nodes, 0.0); 
        uy.resize(num_nodes, 0.0); 
        uz.resize(num_nodes, 0.0);

        h_c.resize(Q7 * num_nodes, 0.0);
        h_c_post.resize(Q7 * num_nodes, 0.0);
        c_O2.resize(num_nodes, 0.0); 

        g.resize(Q7 * num_nodes, 0.0);
        g_post.resize(Q7 * num_nodes, 0.0);
        temp.resize(num_nodes, 298.15); 

        S_c.resize(num_nodes, 0.0);
        S_t.resize(num_nodes, 0.0);

        node_type.resize(num_nodes, FLUID);
        original_material.resize(num_nodes, MAT_CARBON_AEROGEL);
        solid_fraction.resize(num_nodes, 0.0);
    }

    inline int getIndex(int x, int y, int z) const 
    {
        return z * nx * ny + y * nx + x;
    }
};

} // namespace ablation
