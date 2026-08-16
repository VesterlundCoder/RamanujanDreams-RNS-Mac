#include <metal_stdlib>
using namespace metal;

constant uint MAX_M = 10;
constant uint MAX_E = 100;
constant uint MAX_DIM = 16;
constant uint MAX_REG = 256;
constant uint MAX_CHECKPOINTS = 4;

enum Op : ushort {
    OP_NOP = 0,
    OP_LOAD_X,
    OP_LOAD_C,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_NEG,
    OP_POW2,
    OP_POW3,
    OP_INV,
    OP_MULINV,
    OP_COPY
};

struct Instr {
    ushort op;
    ushort dst;
    ushort a;
    ushort b;
};

struct PrimeMeta { uint p; };

struct KernelConfig {
    uint nTraj;
    uint dim;
    uint m;
    uint nSteps;
    uint nInstr;
    uint nConst;
    uint K;
    uint nCheckpoints;
    uint checkpointSteps[MAX_CHECKPOINTS];
};

inline uint add_mod(uint a, uint b, uint p) {
    ulong s = (ulong)a + (ulong)b;
    return (uint)(s % (ulong)p);
}

inline uint sub_mod(uint a, uint b, uint p) {
    return (a >= b) ? (a - b) : (uint)((ulong)a + p - b);
}

inline uint mul_mod(uint a, uint b, uint p) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)p);
}

inline uint pow_mod(uint a, uint e, uint p) {
    uint r = 1;
    uint x = a;
    while (e) {
        if (e & 1u) r = mul_mod(r, x, p);
        e >>= 1u;
        if (e) x = mul_mod(x, x, p);
    }
    return r;
}

inline uint inv_mod(uint a, uint p) {
    return pow_mod(a, p - 2u, p);
}

inline uint signed_to_mod(int v, uint p) {
    long x = (long)v;
    long pp = (long)p;
    long r = x % pp;
    if (r < 0) r += pp;
    return (uint)r;
}

kernel void encode_trajectories_rns(
    device const int *shifts [[buffer(0)]],
    device const int *dirs [[buffer(1)]],
    device const PrimeMeta *primes [[buffer(2)]],
    constant KernelConfig &cfg [[buffer(3)]],
    device uint *shiftRNS [[buffer(4)]],          // [K][B][dim]
    device uint *dirRNS [[buffer(5)]],            // [K][B][dim]
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)cfg.K * cfg.nTraj * cfg.dim;
    if ((ulong)gid >= total) return;

    uint d = gid % cfg.dim;
    uint q = gid / cfg.dim;
    uint b = q % cfg.nTraj;
    uint k = q / cfg.nTraj;
    uint p = primes[k].p;

    ulong src = (ulong)b * cfg.dim + d;
    shiftRNS[gid] = signed_to_mod(shifts[src], p);
    dirRNS[gid] = signed_to_mod(dirs[src], p);
}

inline bool eval_matrix(
    device const Instr *instr,
    device const uint *constTable,               // [K][nConst]
    device const ushort *outReg,                 // [m*m]
    thread const uint *x,
    uint k,
    uint p,
    constant KernelConfig &cfg,
    thread uint *M)
{
    uint regs[MAX_REG];
    for (uint r = 0; r < MAX_REG; ++r) regs[r] = 0;

    for (uint i = 0; i < cfg.nInstr; ++i) {
        Instr ins = instr[i];
        uint va = regs[ins.a];
        uint vb = regs[ins.b];
        uint result = 0;
        switch ((Op)ins.op) {
            case OP_NOP: result = 0; break;
            case OP_LOAD_X: result = x[ins.a]; break;
            case OP_LOAD_C: result = constTable[(ulong)k * cfg.nConst + ins.a]; break;
            case OP_ADD: result = add_mod(va, vb, p); break;
            case OP_SUB: result = sub_mod(va, vb, p); break;
            case OP_MUL: result = mul_mod(va, vb, p); break;
            case OP_NEG: result = va ? p - va : 0; break;
            case OP_POW2: result = mul_mod(va, va, p); break;
            case OP_POW3: result = mul_mod(mul_mod(va, va, p), va, p); break;
            case OP_INV:
                if (va == 0) return false;
                result = inv_mod(va, p);
                break;
            case OP_MULINV:
                if (vb == 0) return false;
                result = mul_mod(va, inv_mod(vb, p), p);
                break;
            case OP_COPY: result = va; break;
        }
        regs[ins.dst] = result;
    }

    uint E = cfg.m * cfg.m;
    for (uint e = 0; e < E; ++e) M[e] = regs[outReg[e]];
    return true;
}

inline void store_snapshot(
    device uint *snapshots,                     // [C][K][B][E]
    thread const uint *P,
    uint checkpoint,
    uint k,
    uint b,
    constant KernelConfig &cfg)
{
    uint E = cfg.m * cfg.m;
    ulong base = ((((ulong)checkpoint * cfg.K + k) * cfg.nTraj + b) * E);
    for (uint e = 0; e < E; ++e) snapshots[base + e] = P[e];
}

// Correctness-first kernel: one Metal thread owns one (trajectory, prime) pair.
// It preserves the complete matrix at every requested checkpoint.
kernel void walk_6f5_rns(
    device const uint *shiftRNS [[buffer(0)]],     // [K][B][dim]
    device const uint *dirRNS [[buffer(1)]],       // [K][B][dim]
    device const Instr *instr [[buffer(2)]],
    device const uint *constTable [[buffer(3)]],   // [K][nConst]
    device const ushort *outReg [[buffer(4)]],
    device const PrimeMeta *primes [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    device uint *snapshots [[buffer(7)]],          // [C][K][B][E]
    device uchar *alive [[buffer(8)]],             // [K][B]
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)cfg.K * cfg.nTraj;
    if ((ulong)gid >= total) return;

    uint b = gid % cfg.nTraj;
    uint k = gid / cfg.nTraj;
    uint p = primes[k].p;
    uint E = cfg.m * cfg.m;

    uint P[MAX_E];
    uint M[MAX_E];
    uint C[MAX_E];
    uint x[MAX_DIM];

    for (uint i = 0; i < cfg.m; ++i)
        for (uint j = 0; j < cfg.m; ++j)
            P[i * cfg.m + j] = (i == j) ? 1u : 0u;

    bool ok = true;
    ulong trajectoryBase = ((ulong)k * cfg.nTraj + b) * cfg.dim;
    uint nextCheckpoint = 0;

    for (uint t = 0; t < cfg.nSteps && ok; ++t) {
        for (uint d = 0; d < cfg.dim; ++d) {
            uint s = shiftRNS[trajectoryBase + d];
            uint v = dirRNS[trajectoryBase + d];
            x[d] = add_mod(s, mul_mod((uint)(t % p), v, p), p);
        }

        ok = eval_matrix(instr, constTable, outReg, x, k, p, cfg, M);
        if (!ok) break;

        for (uint i = 0; i < cfg.m; ++i) {
            for (uint j = 0; j < cfg.m; ++j) {
                uint acc = 0;
                for (uint z = 0; z < cfg.m; ++z) {
                    acc = add_mod(acc, mul_mod(P[i * cfg.m + z], M[z * cfg.m + j], p), p);
                }
                C[i * cfg.m + j] = acc;
            }
        }
        for (uint e = 0; e < E; ++e) P[e] = C[e];

        uint completedSteps = t + 1u;
        while (nextCheckpoint < cfg.nCheckpoints &&
               cfg.checkpointSteps[nextCheckpoint] == completedSteps) {
            store_snapshot(snapshots, P, nextCheckpoint, k, b, cfg);
            ++nextCheckpoint;
        }
    }

    // If a lane dies, checkpoints not reached remain undefined and alive=0.
    alive[(ulong)k * cfg.nTraj + b] = ok ? 1 : 0;
}
