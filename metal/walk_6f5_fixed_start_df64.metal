#include <metal_stdlib>
using namespace metal;

// Fixed-start 6F5 CMF trajectory walk (df64 double-single).
//
// IMPORTANT SEMANTICS:
//   start[] contains the ABSOLUTE CMF coordinates v0.
//   There is no +1/+2 translation and no pre-step by the trajectory.
//   A depth-N walk applies the trajectory operator at macro positions
//       v0, v0+t, ..., v0+(N-1)t.
//
// For a fixed ratio pair (row_num,row_den), only the corresponding two
// rows of the product matrix are propagated. Because every CMF update is
// a right multiplication W <- W*A, rows evolve independently. This is
// exactly equivalent to propagating the full matrix and discarding the
// other rows, while reducing row-update work by ~3x at rank 6.

constant uint MAX_DIM = 6;
constant uint NSHIFT = 11;
constant uint NX = 6;
constant uint NROWS = 2;
constant uint NCHECK = 2;

constant uint ST_OK = 0;
constant uint ST_ZERO_AXIS_DENOMINATOR = 1;
constant uint ST_THETA_LEAD_DEGENERATE = 2;
constant uint ST_INVERSE_SINGULAR = 3;
constant uint ST_NONFINITE = 4;
constant uint ST_NEEDS_REGULARIZATION = 5;

struct Cfg {
    uint nTraj;
    uint nSteps;
    uint rank;
    uint rowNum;
    uint rowDen;
    uint checkpointSteps[NCHECK];
    int start[NSHIFT];
    float zHi;
    float zLo;
};

// ---- df64 primitives --------------------------------------------------
inline float2 df(float x) { return float2(x, 0.0f); }

inline float2 df_renorm(float s, float e) {
    float hi = s + e;
    float lo = e - (hi - s);
    return float2(hi, lo);
}

inline float2 df_add(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);

    float t = a.y + b.y;
    float w = t - a.y;
    float f = (a.y - (t - w)) + (b.y - w);

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
inline bool theta_companion_column(thread const int *pos, float2 z,
                                   uint rank, thread float2 *col)
{
    float2 px[NX + 1];
    px[0] = df(1.0f);
    uint deg = 0;
    for (uint i = 0; i < NX; ++i) {
        float2 xi = df((float)pos[i]);
        px[deg + 1] = df(0.0f);
        for (uint k = deg + 1; k >= 1; --k)
            px[k] = df_add(px[k - 1],
                           df_mul(xi, (k <= deg ? px[k] : df(0.0f))));
        px[0] = df_mul(xi, px[0]);
        ++deg;
    }

    float2 py[NX];
    py[0] = df(1.0f);
    deg = 0;
    for (uint j = 0; j < NX - 1; ++j) {
        float2 yj = df((float)(pos[NX + j] - 1));
        py[deg + 1] = df(0.0f);
        for (uint k = deg + 1; k >= 1; --k)
            py[k] = df_add(py[k - 1],
                           df_mul(yj, (k <= deg ? py[k] : df(0.0f))));
        py[0] = df_mul(yj, py[0]);
        ++deg;
    }

    float2 d[NX + 1];
    for (uint k = 0; k <= NX; ++k) d[k] = df_neg(df_mul(z, px[k]));
    for (uint k = 0; k <= NX - 1; ++k) d[k + 1] = df_add(d[k + 1], py[k]);

    float2 lead = d[rank];
    if (!isfinite(lead.x) || fabs(lead.x) < 1e-30f) return false;

    for (uint i = 0; i < rank; ++i)
        col[i] = df_neg(df_div(d[i], lead));
    return true;
}

inline void row_mul_forward(thread float2 *row, thread const float2 *col,
                            float2 inva, uint rank)
{
    float2 cola[MAX_DIM];
    for (uint k = 0; k < rank; ++k) cola[k] = df_mul(col[k], inva);

    float2 dot = df(0.0f);
    for (uint k = 0; k < rank; ++k)
        dot = df_add(dot, df_mul(row[k], cola[k]));
    float2 newLast = df_add(row[rank - 1], dot);
    for (uint j = 0; j + 1 < rank; ++j)
        row[j] = df_add(row[j], df_mul(row[j + 1], inva));
    row[rank - 1] = newLast;
}

inline uint row_mul_inverse(thread float2 *row, thread const float2 *col,
                            float2 inva, uint rank)
{
    float2 cola[MAX_DIM];
    for (uint k = 0; k < rank; ++k) cola[k] = df_mul(col[k], inva);

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
    if (!isfinite(den.x) || fabs(den.x) < 1e-14f) return ST_INVERSE_SINGULAR;

    float2 t = df_div(df_sub(row[rank - 1], Sa), den);
    row[rank - 1] = t;
    for (int j = (int)rank - 2; j >= 0; --j)
        row[j] = df_add(alpha[j], df_mul(beta[j], t));
    return ST_OK;
}

// Apply one common exact power-of-two normalization to BOTH projected rows.
// A common scale preserves their ratio.
inline void normalize_pow2(thread float2 R[NROWS][MAX_DIM], uint rank,
                           thread int &expSum)
{
    float mx = 0.0f;
    for (uint r = 0; r < NROWS; ++r)
        for (uint j = 0; j < rank; ++j)
            mx = max(mx, fabs(R[r][j].x));
    if (mx > 0.0f) {
        int ex;
        frexp(mx, ex);
        float scale = ldexp(1.0f, -ex);
        for (uint r = 0; r < NROWS; ++r)
            for (uint j = 0; j < rank; ++j) {
                R[r][j].x *= scale;
                R[r][j].y *= scale;
            }
        expSum += ex;
    }
}

inline uint apply_axis_step(thread float2 R[NROWS][MAX_DIM],
                            thread int *pos, int axis, int sign,
                            float2 z, uint rank, thread int &expSum)
{
    int evalPos[NSHIFT];
    for (uint d = 0; d < NSHIFT; ++d) evalPos[d] = pos[d];

    int aInt;
    bool inverse;
    if (axis < (int)NX) {
        if (sign > 0) {
            aInt = pos[axis];
            inverse = false;
        } else {
            evalPos[axis] -= 1;
            aInt = pos[axis] - 1;
            inverse = true;
        }
    } else {
        if (sign < 0) {
            aInt = pos[axis] - 1;
            inverse = false;
        } else {
            evalPos[axis] += 1;
            aInt = pos[axis];
            inverse = true;
        }
    }
    if (aInt == 0) return ST_ZERO_AXIS_DENOMINATOR;

    float2 col[MAX_DIM];
    if (!theta_companion_column(evalPos, z, rank, col))
        return ST_THETA_LEAD_DEGENERATE;

    float2 inva = df_div(df(1.0f), df((float)aInt));
    for (uint r = 0; r < NROWS; ++r) {
        uint st = inverse ? row_mul_inverse(R[r], col, inva, rank) : ST_OK;
        if (st != ST_OK) return st;
        if (!inverse) row_mul_forward(R[r], col, inva, rank);
    }

    pos[axis] += sign;
    normalize_pow2(R, rank, expSum);
    for (uint r = 0; r < NROWS; ++r)
        if (!isfinite(R[r][rank - 1].x)) return ST_NONFINITE;
    return ST_OK;
}

kernel void walk_6f5_fixed_start_df64(
    device const int *dirs [[buffer(0)]],          // [B][11]
    constant Cfg &cfg [[buffer(1)]],
    device float2 *outVals [[buffer(2)]],          // [B][2 checkpoints][2 rows]
    device int *outExp [[buffer(3)]],              // [B][2 checkpoints]
    device uint *outStatus [[buffer(4)]],           // [B]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.nTraj) return;

    int pos[NSHIFT], dv[NSHIFT];
    for (uint d = 0; d < NSHIFT; ++d) {
        pos[d] = cfg.start[d];                      // ABSOLUTE v0
        dv[d] = dirs[(ulong)gid * NSHIFT + d];
    }

    const float2 z = float2(cfg.zHi, cfg.zLo);
    const uint rank = cfg.rank;

    float2 R[NROWS][MAX_DIM];
    for (uint r = 0; r < NROWS; ++r)
        for (uint j = 0; j < MAX_DIM; ++j)
            R[r][j] = df(0.0f);
    R[0][cfg.rowNum] = df(1.0f);
    R[1][cfg.rowDen] = df(1.0f);

    int expSum = 0;
    uint status = ST_OK;
    uint nextCheckpoint = 0;

    // Defensive initialization. Invalid/early-exit lanes never leak stale values.
    for (uint c = 0; c < NCHECK; ++c) {
        ulong b = ((ulong)gid * NCHECK + c);
        outVals[b * NROWS + 0] = float2(NAN, NAN);
        outVals[b * NROWS + 1] = float2(NAN, NAN);
        outExp[b] = 0;
    }

    int maxAbs = 0;
    for (uint d = 0; d < NSHIFT; ++d) maxAbs = max(maxAbs, abs(dv[d]));
    if (maxAbs == 0) status = ST_NEEDS_REGULARIZATION; // zero vector is not a trajectory

    for (uint step = 0; step < cfg.nSteps && status == ST_OK; ++step) {
        for (int level = maxAbs; level >= 1 && status == ST_OK; --level) {
            for (int axis = (int)NSHIFT - 1; axis >= 0; --axis) {
                if (abs(dv[axis]) < level) continue;
                int sign = dv[axis] > 0 ? 1 : -1;
                status = apply_axis_step(R, pos, axis, sign, z, rank, expSum);
                if (status != ST_OK) break;
            }
        }

        uint completedSteps = step + 1u;
        while (status == ST_OK && nextCheckpoint < NCHECK &&
               cfg.checkpointSteps[nextCheckpoint] == completedSteps) {
            ulong b = ((ulong)gid * NCHECK + nextCheckpoint);
            outVals[b * NROWS + 0] = R[0][rank - 1];
            outVals[b * NROWS + 1] = R[1][rank - 1];
            outExp[b] = expSum;
            ++nextCheckpoint;
        }
    }

    outStatus[gid] = status;
}
