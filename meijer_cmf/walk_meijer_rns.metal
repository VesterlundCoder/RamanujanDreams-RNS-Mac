#include <metal_stdlib>
using namespace metal;

// MeijerG(4,2,4,4,1) CMF trajectory walk in RNS (residue number system).
//
// Unlike the 6F5 bytecode kernel (rns_6f5.metal), the trajectory matrix of a
// FIXED (initial, direction) pair is a polynomial matrix in the step index n,
// so each matrix entry is described by a Horner coefficient table instead of
// a bytecode program.  Coefficients are pre-reduced per prime on the host
// (they can exceed 32 bits as exact integers).
//
// One thread = one (candidate, prime) pair.
// P <- P * M(n), n = 0..nSteps-1   (matches ramanujantools walk indexing).
// Full modular matrices are snapshotted at every requested checkpoint for
// CPU-side Garner CRT reconstruction, exactly like the 6F5 full-matrix mode:
//   snapshots[checkpoint][prime][candidate][m*m]

constant uint MAX_M = 8;            // matrix dimension bound
constant uint MAX_E = 64;           // MAX_M * MAX_M
constant uint MAX_CHECKPOINTS = 8;

struct PrimeMeta { uint p; };

struct KernelConfig {
    uint nTraj;                     // candidates (B)
    uint m;                         // matrix dimension
    uint deg1;                      // polynomial degree + 1 (Horner length)
    uint nSteps;                    // walk depth N
    uint K;                         // number of primes
    uint nCheckpoints;
    uint checkpointSteps[MAX_CHECKPOINTS];
};

inline uint add_mod(uint a, uint b, uint p) {
    ulong s = (ulong)a + (ulong)b;
    return (uint)(s % (ulong)p);
}

inline uint mul_mod(uint a, uint b, uint p) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)p);
}

// Horner evaluation of one entry at n = t (mod p).
// coeffs are stored high-degree first, length cfg.deg1.
inline uint horner(device const uint *c, uint deg1, uint tmod, uint p) {
    uint v = c[0];
    for (uint j = 1; j < deg1; ++j)
        v = add_mod(mul_mod(v, tmod, p), c[j], p);
    return v;
}

inline void store_snapshot(
    device uint *snapshots,          // [C][K][B][E]
    thread const uint *P,
    uint checkpoint, uint k, uint b,
    constant KernelConfig &cfg)
{
    uint E = cfg.m * cfg.m;
    ulong base = ((((ulong)checkpoint * cfg.K + k) * cfg.nTraj + b) * E);
    for (uint e = 0; e < E; ++e) snapshots[base + e] = P[e];
}

kernel void walk_meijer_rns(
    device const uint *coeffRNS [[buffer(0)]],   // [K][B][E][deg1]
    device const PrimeMeta *primes [[buffer(1)]],
    constant KernelConfig &cfg [[buffer(2)]],
    device uint *snapshots [[buffer(3)]],        // [C][K][B][E]
    device uchar *alive [[buffer(4)]],           // [K][B]
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)cfg.K * cfg.nTraj;
    if ((ulong)gid >= total) return;

    uint b = gid % cfg.nTraj;
    uint k = gid / cfg.nTraj;
    uint p = primes[k].p;
    uint m = cfg.m;
    uint E = m * m;

    uint P[MAX_E];
    uint M[MAX_E];
    uint Cn[MAX_E];

    for (uint i = 0; i < m; ++i)
        for (uint j = 0; j < m; ++j)
            P[i * m + j] = (i == j) ? 1u : 0u;

    ulong cbase = (((ulong)k * cfg.nTraj + b) * E) * cfg.deg1;
    uint nextCheckpoint = 0;

    for (uint t = 0; t < cfg.nSteps; ++t) {
        uint tmod = t % p;

        for (uint e = 0; e < E; ++e)
            M[e] = horner(coeffRNS + cbase + (ulong)e * cfg.deg1,
                          cfg.deg1, tmod, p);

        for (uint i = 0; i < m; ++i) {
            for (uint j = 0; j < m; ++j) {
                uint acc = 0;
                for (uint z = 0; z < m; ++z)
                    acc = add_mod(acc, mul_mod(P[i * m + z], M[z * m + j], p), p);
                Cn[i * m + j] = acc;
            }
        }
        for (uint e = 0; e < E; ++e) P[e] = Cn[e];

        uint completedSteps = t + 1u;
        while (nextCheckpoint < cfg.nCheckpoints &&
               cfg.checkpointSteps[nextCheckpoint] == completedSteps) {
            store_snapshot(snapshots, P, nextCheckpoint, k, b, cfg);
            ++nextCheckpoint;
        }
    }

    // Polynomial evaluation cannot divide by zero: lanes never die, but the
    // flag is kept for host-side compatibility with the 6F5 pipeline.
    alive[(ulong)k * cfg.nTraj + b] = 1;
}
