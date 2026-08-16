#include <metal_stdlib>
using namespace metal;

// 6F5 companion walk in double-single (df64) precision.
// One thread = one trajectory. P <- P * M(n), n = 1..nSteps.
// Every step the matrix is rescaled by an exact power of two
// (no rounding error) and the exponent is accumulated.

constant uint DIM = 6;      // companion matrix size
constant uint NSHIFT = 11;  // 2*DIM-1 trajectory components
constant uint E = 36;       // DIM*DIM

struct Cfg {
    uint nTraj;
    uint nSteps;
    float zHi;
    float zLo;
};

// ---- df64 primitives (double-single arithmetic, requires correct fma) ----
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

inline float2 df_neg(float2 a) { return float2(-a.x, -a.y); }

kernel void walk_6f5_df64(
    device const int *shifts [[buffer(0)]],   // [B][NSHIFT]
    device const int *dirs [[buffer(1)]],     // [B][NSHIFT]
    constant Cfg &cfg [[buffer(2)]],
    device float2 *outMat [[buffer(3)]],      // [B][E] normalized product matrix
    device int *outExp [[buffer(4)]],         // [B] accumulated power-of-two exponent
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.nTraj) return;

    int sh[NSHIFT], dv[NSHIFT];
    for (uint d = 0; d < NSHIFT; ++d) {
        sh[d] = shifts[(ulong)gid * NSHIFT + d];
        dv[d] = dirs[(ulong)gid * NSHIFT + d];
    }

    const float2 z = float2(cfg.zHi, cfg.zLo);

    float2 P[E];
    for (uint i = 0; i < DIM; ++i)
        for (uint j = 0; j < DIM; ++j)
            P[i * DIM + j] = df(i == j ? 1.0f : 0.0f);

    int expSum = 0;

    for (uint n = 1; n <= cfg.nSteps; ++n) {
        // Roots (exact in float32: |value| < 2^24 for census parameter ranges)
        float2 f[DIM], g[DIM - 1];
        for (uint i = 0; i < DIM; ++i)
            f[i] = df((float)(sh[i] + (int)n * dv[i] + 1));
        for (uint j = 0; j < DIM - 1; ++j)
            g[j] = df((float)(sh[DIM + j] + (int)n * dv[DIM + j] + 2));

        // Elementary symmetric polynomials
        float2 ef[DIM + 1];
        ef[0] = df(1.0f);
        for (uint k = 1; k <= DIM; ++k) ef[k] = df(0.0f);
        for (uint i = 0; i < DIM; ++i)
            for (uint k = i + 1; k >= 1; --k)
                ef[k] = df_add(ef[k], df_mul(f[i], ef[k - 1]));

        float2 eg[DIM];
        eg[0] = df(1.0f);
        for (uint k = 1; k <= DIM - 1; ++k) eg[k] = df(0.0f);
        for (uint i = 0; i < DIM - 1; ++i)
            for (uint k = i + 1; k >= 1; --k)
                eg[k] = df_add(eg[k], df_mul(g[i], eg[k - 1]));

        // Companion last column: c[0] = -z*ef[6], c[k] = eg[6-k] - z*ef[6-k]
        float2 c[DIM];
        c[0] = df_neg(df_mul(z, ef[DIM]));
        for (uint k = 1; k < DIM; ++k)
            c[k] = df_add(eg[DIM - k], df_neg(df_mul(z, ef[DIM - k])));

        // P <- P * M. M is companion: subdiagonal ones + last column c.
        // (P*M)[i][j] = P[i][j+1] for j < DIM-1;  (P*M)[i][DIM-1] = sum_t P[i][t]*c[t]
        for (uint i = 0; i < DIM; ++i) {
            float2 last = df(0.0f);
            for (uint t = 0; t < DIM; ++t)
                last = df_add(last, df_mul(P[i * DIM + t], c[t]));
            for (uint j = 0; j + 1 < DIM; ++j)
                P[i * DIM + j] = P[i * DIM + j + 1];
            P[i * DIM + DIM - 1] = last;
        }

        // Exact power-of-two normalization
        float mx = 0.0f;
        for (uint e = 0; e < E; ++e) mx = max(mx, fabs(P[e].x));
        if (mx > 0.0f) {
            int ex;
            frexp(mx, ex);
            float scale = ldexp(1.0f, -ex);
            for (uint e = 0; e < E; ++e) {
                P[e].x *= scale;   // exact
                P[e].y *= scale;   // exact
            }
            expSum += ex;
        }
    }

    ulong base = (ulong)gid * E;
    for (uint e = 0; e < E; ++e) outMat[base + e] = P[e];
    outExp[gid] = expSum;
}
