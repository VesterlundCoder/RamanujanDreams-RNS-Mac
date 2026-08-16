#pragma once
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t MAX_CHECKPOINTS_HOST = 4;
constexpr uint32_t MAX_MATRIX_ELEMS_HOST = 100;

struct InstrHost {
    uint16_t op, dst, a, b;
};

struct PrimeMetaHost { uint32_t p; };

struct KernelConfigHost {
    uint32_t nTraj;
    uint32_t dim;
    uint32_t m;
    uint32_t nSteps;
    uint32_t nInstr;
    uint32_t nConst;
    uint32_t K;
    uint32_t nCheckpoints;
    uint32_t checkpointSteps[MAX_CHECKPOINTS_HOST];
};

struct Trajectory {
    std::vector<int32_t> shift;
    std::vector<int32_t> dir;
};

struct ProgramHost {
    uint32_t m = 10;
    uint32_t dim = 0;
    std::vector<InstrHost> instr;
    std::vector<uint32_t> constTable; // [K][nConst]
    std::vector<uint16_t> outReg;     // [m*m]
};

static_assert(sizeof(KernelConfigHost) == 48, "KernelConfigHost must match MSL KernelConfig layout");
