#include <metal_stdlib>
using namespace metal;

// CORRECTED Ramanujan Dreams 6F5 CMF trajectory walk (df64 double-single).
//
// Semantics (mirrors lib/cmf_walk_corrected.py, the exact-rational oracle):
//   start:  x_i(0) = shift_i + 1,  y_j(0) = shift_{6+j} + 2
//   theta operator D(theta) = theta*prod(theta+y_j-1) - z*prod(theta+x_i)
//   monic companion column c_i = -d_i / d_rank  (rank 6 for z!=1, 5 for z=1)
//   axis operators  I + C(v)/a  (forward) and (I + C(v')/a)^{-1} (inverse):
//     x_i +1 : forward,  eval v,      a = x_i
//     x_i -1 : inverse,  eval v-e_i,  a = x_i - 1
//     y_j -1 : forward,  eval v,      a = y_j - 1
//     y_j +1 : inverse,  eval v+e_j,  a = y_j
//   direction decomposed into unit axis steps:
//     for level = max|t| .. 1: for axis = 10 .. 0: one +-1 step
// The matrix is renormalized by an exact power of two after EVERY unit
// axis step; the exponent accumulates in expSum.

constant uint MAX_DIM = 6;
constant uint NSHIFT = 11;
constant uint NX = 6;
constant uint E = 36;

// status codes
constant uint ST_OK = 0;
constant uint ST_ZERO_AXIS_DENOMINATOR = 1;
constant uint ST_THETA_LEAD_DEGENERATE = 2;
constant uint ST_INVERSE_SINGULAR = 3;
constant uint ST_NONFINITE = 4;
constant uint ST_NEEDS_REGULARIZATION = 5;

struct Cfg {
    uint nTraj;
    uint nSteps;
    uint rank;      // 5 (z==1) or 6
    float zHi;
    float zLo;
};

// ---- df64 primitives (double-single arithmetic, requires exact fma) ----
inline float2 df(float x) { return float2(x, 0.0f); }

inline float2 df_renorm(float s, float e) {
    float hi = s + e;
    float lo = e - (hi - s);
    return float2(hi, lo);
}

inline float2 df_add(float2 a, float2 b) {
    // accurate double-single addition (two two-sums + renormalizations)
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);   // exact error of hi sum

    float t = a.y + b.y;
    float w = t - a.y;
    float f = (a.y - (t - w)) + (b.y - w);   // exact error of lo sum

    e += t;
    float hi = s + e;
    float lo = e - (hi - s);
    lo += f;
    return df_renorm(hi, lo);
}

inline float2 df_mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, e);
    e = fma(a.y, b.x, e);
    e = fma(a.y, b.y, e);
    return df_renorm(p, e);
}

inline float2 df_neg(float2 a) { return float2(-a.x, -a.y); }

inline float2 df_sub(float2 a, float2 b) { return df_add(a, df_neg(b)); }

// double-single division: float32 quotient + two residual refinement passes
inline float2 df_div(float2 a, float2 b) {
    float q1 = a.x / b.x;
    float2 q = df(q1);
    float2 r = df_sub(a, df_mul(b, q));
    float q2 = (r.x + r.y) / b.x;
    q = df_add(q, df(q2));
    r = df_sub(a, df_mul(b, q));
    float q3 = (r.x + r.y) / b.x;
    return df_add(q, df(q3));
}

// ---- theta companion --------------------------------------------------
// col[i] = -d_i/d_rank of D(theta) at integer position pos[11].
// Returns false if the leading coefficient is degenerate.
inline bool theta_companion_column(thread const int *pos, float2 z,
                                   uint rank, thread float2 *col)
{
    // px = coeffs of prod(theta + x_i), degree 6
    float2 px[NX + 1];
    px[0] = df(1.0f);
    uint deg = 0;
    for (uint i = 0; i < NX; ++i) {
        float2 xi = df((float)pos[i]);
        px[deg + 1] = df(0.0f);
        for (uint k = deg + 1; k >= 1; --k)
            px[k] = df_add((k >= 1 ? px[k - 1] : df(0.0f)),
                           df_mul(xi, (k <= deg ? px[k] : df(0.0f))));
        px[0] = df_mul(xi, px[0]);
        ++deg;
    }

    // py = coeffs of prod(theta + y_j - 1), degree 5
    float2 py[NX];
    py[0] = df(1.0f);
    deg = 0;
    for (uint j = 0; j < NX - 1; ++j) {
        float2 yj = df((float)(pos[NX + j] - 1));
        py[deg + 1] = df(0.0f);
        for (uint k = deg + 1; k >= 1; --k)
            py[k] = df_add((k >= 1 ? py[k - 1] : df(0.0f)),
                           df_mul(yj, (k <= deg ? py[k] : df(0.0f))));
        py[0] = df_mul(yj, py[0]);
        ++deg;
    }

    // d[k] = -z*px[k]; d[k+1] += py[k]
    float2 d[NX + 1];
    for (uint k = 0; k <= NX; ++k) d[k] = df_neg(df_mul(z, px[k]));
    for (uint k = 0; k <= NX - 1; ++k) d[k + 1] = df_add(d[k + 1], py[k]);

    float2 lead = d[rank];
    if (!isfinite(lead.x) || fabs(lead.x) < 1e-30f) return false;

    for (uint i = 0; i < rank; ++i)
        col[i] = df_neg(df_div(d[i], lead));
    return true;
}

// ---- affine companion multiplications, row-wise O(r^2) -----------------
// W <- W * (I + C/a).  C: subdiagonal ones + last column col.
inline void right_mul_forward(thread float2 *W, thread const float2 *col,
                              float2 inva, uint rank)
{
    float2 cola[MAX_DIM];
    for (uint k = 0; k < rank; ++k) cola[k] = df_mul(col[k], inva);
    for (uint r = 0; r < rank; ++r) {
        thread float2 *row = W + r * MAX_DIM;
        float2 dot = df(0.0f);
        for (uint k = 0; k < rank; ++k)
            dot = df_add(dot, df_mul(row[k], cola[k]));
        float2 newLast = df_add(row[rank - 1], dot);
        for (uint j = 0; j + 1 < rank; ++j)
            row[j] = df_add(row[j], df_mul(row[j + 1], inva));
        row[rank - 1] = newLast;
    }
}

// W <- W * (I + C/a)^{-1} via per-row back-substitution alpha_j + beta_j*t.
// Returns a status: near-singular dens (|den| < 1e-4) lose too many digits
// at df64 precision and are flagged for exact CPU regularization.
inline uint right_mul_inverse(thread float2 *W, thread const float2 *col,
                              float2 inva, uint rank)
{
    float2 cola[MAX_DIM];
    for (uint k = 0; k < rank; ++k) cola[k] = df_mul(col[k], inva);
    for (uint r = 0; r < rank; ++r) {
        thread float2 *row = W + r * MAX_DIM;
        float2 alpha[MAX_DIM], beta[MAX_DIM];
        alpha[rank - 1] = df(0.0f);
        beta[rank - 1] = df(1.0f);
        for (int j = (int)rank - 2; j >= 0; --j) {
            alpha[j] = df_sub(row[j], df_mul(alpha[j + 1], inva));
            beta[j] = df_neg(df_mul(beta[j + 1], inva));
        }
        float2 Sa = df(0.0f), Sb = df(0.0f);
        for (uint k = 0; k < rank; ++k) {
            Sa = df_add(Sa, df_mul(alpha[k], cola[k]));
            Sb = df_add(Sb, df_mul(beta[k], cola[k]));
        }
        float2 den = df_add(df(1.0f), Sb);
        // NOTE: |den| ~ a^-rank is the NORMAL scale for inverse steps;
        // only a near-exact zero signals a true singular operator.
        if (!isfinite(den.x) || fabs(den.x) < 1e-14f) return ST_INVERSE_SINGULAR;
        float2 t = df_div(df_sub(row[rank - 1], Sa), den);
        row[rank - 1] = t;
        for (int j = (int)rank - 2; j >= 0; --j)
            row[j] = df_add(alpha[j], df_mul(beta[j], t));
    }
    return ST_OK;
}

// exact power-of-two normalization of the active rank x rank block
inline void normalize_pow2(thread float2 *W, uint rank, thread int &expSum)
{
    float mx = 0.0f;
    for (uint i = 0; i < rank; ++i)
        for (uint j = 0; j < rank; ++j)
            mx = max(mx, fabs(W[i * MAX_DIM + j].x));
    if (mx > 0.0f) {
        int ex;
        frexp(mx, ex);
        float scale = ldexp(1.0f, -ex);
        for (uint i = 0; i < rank; ++i)
            for (uint j = 0; j < rank; ++j) {
                W[i * MAX_DIM + j].x *= scale;  // exact
                W[i * MAX_DIM + j].y *= scale;  // exact
            }
        expSum += ex;
    }
}

// one +-1 unit axis step
inline uint apply_axis_step(thread float2 *W, thread int *pos,
                            int axis, int sign, float2 z, uint rank,
                            thread int &expSum)
{
    int evalPos[NSHIFT];
    for (uint d = 0; d < NSHIFT; ++d) evalPos[d] = pos[d];

    int aInt;
    bool inverse;
    if (axis < (int)NX) {               // numerator x_i
        if (sign > 0) { aInt = pos[axis];     inverse = false; }
        else          { evalPos[axis] -= 1;
                        aInt = pos[axis] - 1; inverse = true; }
    } else {                            // denominator y_j
        if (sign < 0) { aInt = pos[axis] - 1; inverse = false; }
        else          { evalPos[axis] += 1;
                        aInt = pos[axis];     inverse = true; }
    }
    if (aInt == 0) return ST_ZERO_AXIS_DENOMINATOR;

    float2 col[MAX_DIM];
    if (!theta_companion_column(evalPos, z, rank, col))
        return ST_THETA_LEAD_DEGENERATE;

    float2 inva = df_div(df(1.0f), df((float)aInt));
    if (inverse) {
        uint st = right_mul_inverse(W, col, inva, rank);
        if (st != ST_OK) return st;
    } else {
        right_mul_forward(W, col, inva, rank);
    }
    pos[axis] += sign;

    normalize_pow2(W, rank, expSum);
    if (!isfinite(W[0].x)) return ST_NONFINITE;
    return ST_OK;
}

kernel void walk_6f5_cmf_df64(
    device const int *shifts [[buffer(0)]],   // [B][NSHIFT]
    device const int *dirs [[buffer(1)]],     // [B][NSHIFT]
    constant Cfg &cfg [[buffer(2)]],
    device float2 *outMat [[buffer(3)]],      // [B][E]
    device int *outExp [[buffer(4)]],         // [B]
    device uint *outStatus [[buffer(5)]],     // [B]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.nTraj) return;

    int pos[NSHIFT], dv[NSHIFT];
    for (uint d = 0; d < NSHIFT; ++d) {
        int sh = shifts[(ulong)gid * NSHIFT + d];
        pos[d] = (d < NX) ? sh + 1 : sh + 2;   // actual Dreams start point
        dv[d] = dirs[(ulong)gid * NSHIFT + d];
    }

    const float2 z = float2(cfg.zHi, cfg.zLo);
    const uint rank = cfg.rank;

    float2 W[E];
    for (uint i = 0; i < MAX_DIM; ++i)
        for (uint j = 0; j < MAX_DIM; ++j)
            W[i * MAX_DIM + j] = df(i == j ? 1.0f : 0.0f);

    int expSum = 0;
    uint status = ST_OK;

    int maxAbs = 0;
    for (uint d = 0; d < NSHIFT; ++d) maxAbs = max(maxAbs, abs(dv[d]));

    for (uint step = 0; step < cfg.nSteps && status == ST_OK; ++step) {
        for (int level = maxAbs; level >= 1 && status == ST_OK; --level) {
            for (int axis = (int)NSHIFT - 1; axis >= 0; --axis) {
                if (abs(dv[axis]) < level) continue;
                int sign = dv[axis] > 0 ? 1 : -1;
                status = apply_axis_step(W, pos, axis, sign, z, rank, expSum);
                if (status != ST_OK) break;
            }
        }
    }

    ulong base = (ulong)gid * E;
    for (uint e = 0; e < E; ++e) outMat[base + e] = W[e];
    outExp[gid] = expSum;
    outStatus[gid] = status;
}
