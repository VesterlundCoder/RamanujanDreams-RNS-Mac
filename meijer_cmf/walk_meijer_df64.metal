#include <metal_stdlib>
using namespace metal;

// MeijerG(4,2,4,4,1) CMF trajectory walk in df64 (double-single float32)
// with software exponents -- the no-primes screening stage, analog of
// walk_6f5_df64.metal.
//
// The trajectory matrix M(n) of a FIXED (initial, direction) pair is a
// polynomial matrix in n (3x3, degree ~67).  Exact integer coefficients
// overflow every float format, so the host supplies each coefficient as
//   mantissa (float2, |mant| in [0.5, 1)) + int exponent
// scaled by one global power of two per candidate (projectively exact).
// Horner accumulators and the product matrix carry their own software
// exponents and are renormalized every step, so nothing ever leaves
// float32 range regardless of depth.
//
// One thread = one candidate.  P <- P * M(n), n = 0..nSteps-1.
// Outputs at each checkpoint: normalized product matrix (df64 mantissas),
// per-matrix exponent (log2 growth up to the global coefficient scale).
// Limits: c1 = P[1,0]/P[0,0], c2 = P[2,0]/P[0,0]  (~14 digits).
//
// Validated CPU mirror: rns/df64_walk.py (float64; agrees with the exact
// sympy walk to ~1e-15 at N=1000).

constant uint MAX_M = 4;
constant uint MAX_E = 16;
constant uint MAX_DEG1 = 80;
constant uint MAX_CHECKPOINTS = 8;

struct Cfg {
    uint nTraj;
    uint m;
    uint deg1;
    uint nSteps;
    uint nCheckpoints;
    uint checkpointSteps[MAX_CHECKPOINTS];
};

// ---- df64 primitives (require correct fma) ----
inline float2 df(float x) { return float2(x, 0.0f); }

inline float2 df_renorm(float s, float e) {
    float hi = s + e;
    float lo = e - (hi - s);
    return float2(hi, lo);
}

inline float2 df_add(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float err = (a.x - (s - v)) + (b.x - v);
    err += a.y + b.y;
    return df_renorm(s, err);
}

inline float2 df_mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, e);
    e = fma(a.y, b.x, e);
    return df_renorm(p, e);
}

// ---- extended value: df64 mantissa + software exponent ----
struct Ext { float2 m; int e; };

inline Ext ext_norm(Ext a) {
    if (a.m.x == 0.0f) { a.e = 0; return a; }
    int ex;
    frexp(a.m.x, ex);                 // |m.x| in [0.5, 1) after scaling
    float s = ldexp(1.0f, -ex);
    a.m.x *= s;                       // exact
    a.m.y *= s;                       // exact
    a.e += ex;
    return a;
}

inline Ext ext_mul(Ext a, Ext b) {
    Ext r; r.m = df_mul(a.m, b.m); r.e = a.e + b.e;
    return ext_norm(r);
}

inline Ext ext_add(Ext a, Ext b) {
    if (a.m.x == 0.0f) return ext_norm(b);
    if (b.m.x == 0.0f) return a;
    int d = a.e - b.e;
    if (d > 60) return a;             // b negligible at df64 precision
    if (d < -60) return b;
    Ext r;
    if (d >= 0) {
        float s = ldexp(1.0f, -d);    // exact
        r.m = df_add(a.m, float2(b.m.x * s, b.m.y * s));
        r.e = a.e;
    } else {
        float s = ldexp(1.0f, d);
        r.m = df_add(float2(a.m.x * s, a.m.y * s), b.m);
        r.e = b.e;
    }
    return ext_norm(r);
}

kernel void walk_meijer_df64(
    device const float2 *coeffMant [[buffer(0)]],  // [B][E][deg1]
    device const int *coeffExp [[buffer(1)]],      // [B][E][deg1]
    constant Cfg &cfg [[buffer(2)]],
    device float2 *outMat [[buffer(3)]],           // [C][B][E] df64 mantissas
    device int *outExp [[buffer(4)]],              // [C][B][E] relative exponents
    device int *outGrowth [[buffer(5)]],           // [C][B] accumulated log2 growth
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.nTraj) return;

    uint m = cfg.m;
    uint E = m * m;
    ulong cbase = (ulong)gid * E * cfg.deg1;

    Ext P[MAX_E], M[MAX_E], T[MAX_E];
    for (uint i = 0; i < m; ++i)
        for (uint j = 0; j < m; ++j) {
            P[i * m + j].m = df(i == j ? 1.0f : 0.0f);
            P[i * m + j].e = 0;
        }

    uint nextCheckpoint = 0;
    int expSum = 0;

    for (uint t = 0; t < cfg.nSteps; ++t) {
        Ext tv; tv.m = df((float)t); tv.e = 0; tv = ext_norm(tv);

        // Horner per entry
        for (uint e = 0; e < E; ++e) {
            ulong off = cbase + (ulong)e * cfg.deg1;
            Ext v; v.m = coeffMant[off]; v.e = coeffExp[off];
            for (uint d = 1; d < cfg.deg1; ++d) {
                v = ext_mul(v, tv);
                Ext c; c.m = coeffMant[off + d]; c.e = coeffExp[off + d];
                v = ext_add(v, c);
            }
            M[e] = v;
        }

        // P <- P * M
        for (uint i = 0; i < m; ++i)
            for (uint j = 0; j < m; ++j) {
                Ext acc; acc.m = df(0.0f); acc.e = 0;
                for (uint z = 0; z < m; ++z)
                    acc = ext_add(acc, ext_mul(P[i * m + z], M[z * m + j]));
                T[i * m + j] = acc;
            }
        // common renormalization: shift all exponents so max is 0,
        // accumulate the shift as the log2 growth of the product
        int emax = T[0].e;
        for (uint e = 1; e < E; ++e) emax = max(emax, T[e].e);
        for (uint e = 0; e < E; ++e) { T[e].e -= emax; P[e] = T[e]; }
        expSum += emax;

        uint completedSteps = t + 1u;
        while (nextCheckpoint < cfg.nCheckpoints &&
               cfg.checkpointSteps[nextCheckpoint] == completedSteps) {
            ulong base = ((ulong)nextCheckpoint * cfg.nTraj + gid) * E;
            for (uint e = 0; e < E; ++e) {
                outMat[base + e] = P[e].m;
                outExp[base + e] = P[e].e;   // relative to expSum
            }
            outGrowth[(ulong)nextCheckpoint * cfg.nTraj + gid] = expSum;
            ++nextCheckpoint;
        }
    }
}
