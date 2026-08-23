#include "cmf_cpu.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define NX 6u

static int theta_companion_column(
    const int32_t pos[DREAMS_NSHIFT],
    df64_t z,
    uint32_t rank,
    df64_t col[DREAMS_MAX_DIM])
{
    df64_t px[NX + 1];
    px[0] = df64_from_float(1.0f);
    uint32_t deg = 0;
    for (uint32_t i = 0; i < NX; ++i) {
        df64_t xi = df64_from_float((float)pos[i]);
        px[deg + 1] = df64_from_float(0.0f);
        for (int k = (int)deg + 1; k >= 1; --k) {
            df64_t old = ((uint32_t)k <= deg) ? px[k] : df64_from_float(0.0f);
            px[k] = df64_add(px[k - 1], df64_mul(xi, old));
        }
        px[0] = df64_mul(xi, px[0]);
        ++deg;
    }

    df64_t py[NX];
    py[0] = df64_from_float(1.0f);
    deg = 0;
    for (uint32_t j = 0; j < NX - 1; ++j) {
        df64_t yj = df64_from_float((float)(pos[NX + j] - 1));
        py[deg + 1] = df64_from_float(0.0f);
        for (int k = (int)deg + 1; k >= 1; --k) {
            df64_t old = ((uint32_t)k <= deg) ? py[k] : df64_from_float(0.0f);
            py[k] = df64_add(py[k - 1], df64_mul(yj, old));
        }
        py[0] = df64_mul(yj, py[0]);
        ++deg;
    }

    df64_t d[NX + 1];
    for (uint32_t k = 0; k <= NX; ++k) d[k] = df64_neg(df64_mul(z, px[k]));
    for (uint32_t k = 0; k <= NX - 1; ++k) d[k + 1] = df64_add(d[k + 1], py[k]);

    df64_t lead = d[rank];
    if (!isfinite(lead.hi) || fabsf(lead.hi) < 1e-30f) return 0;

    for (uint32_t i = 0; i < rank; ++i) {
        col[i] = df64_neg(df64_div(d[i], lead));
    }
    return 1;
}

static void row_mul_forward(
    df64_t row[DREAMS_MAX_DIM],
    const df64_t col[DREAMS_MAX_DIM],
    df64_t inva,
    uint32_t rank)
{
    df64_t cola[DREAMS_MAX_DIM];
    for (uint32_t k = 0; k < rank; ++k) cola[k] = df64_mul(col[k], inva);

    df64_t dot = df64_from_float(0.0f);
    for (uint32_t k = 0; k < rank; ++k) dot = df64_add(dot, df64_mul(row[k], cola[k]));
    df64_t new_last = df64_add(row[rank - 1], dot);

    for (uint32_t j = 0; j + 1 < rank; ++j) {
        row[j] = df64_add(row[j], df64_mul(row[j + 1], inva));
    }
    row[rank - 1] = new_last;
}

static uint32_t row_mul_inverse(
    df64_t row[DREAMS_MAX_DIM],
    const df64_t col[DREAMS_MAX_DIM],
    df64_t inva,
    uint32_t rank)
{
    df64_t cola[DREAMS_MAX_DIM];
    for (uint32_t k = 0; k < rank; ++k) cola[k] = df64_mul(col[k], inva);

    df64_t alpha[DREAMS_MAX_DIM];
    df64_t beta[DREAMS_MAX_DIM];
    alpha[rank - 1] = df64_from_float(0.0f);
    beta[rank - 1] = df64_from_float(1.0f);

    for (int j = (int)rank - 2; j >= 0; --j) {
        alpha[j] = df64_sub(row[j], df64_mul(alpha[j + 1], inva));
        beta[j] = df64_neg(df64_mul(beta[j + 1], inva));
    }

    df64_t sa = df64_from_float(0.0f);
    df64_t sb = df64_from_float(0.0f);
    for (uint32_t k = 0; k < rank; ++k) {
        sa = df64_add(sa, df64_mul(alpha[k], cola[k]));
        sb = df64_add(sb, df64_mul(beta[k], cola[k]));
    }

    df64_t den = df64_add(df64_from_float(1.0f), sb);
    if (!isfinite(den.hi) || fabsf(den.hi) < 1e-14f) return DREAMS_ST_INVERSE_SINGULAR;

    df64_t t = df64_div(df64_sub(row[rank - 1], sa), den);
    row[rank - 1] = t;
    for (int j = (int)rank - 2; j >= 0; --j) {
        row[j] = df64_add(alpha[j], df64_mul(beta[j], t));
    }
    return DREAMS_ST_OK;
}

static void normalize_pow2(
    df64_t r[DREAMS_NROWS][DREAMS_MAX_DIM],
    uint32_t rank,
    int32_t *exp_sum)
{
    float mx = 0.0f;
    for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
        for (uint32_t j = 0; j < rank; ++j) {
            float x = fabsf(r[row][j].hi);
            if (x > mx) mx = x;
        }
    }

    if (mx > 0.0f) {
        int ex = 0;
        (void)frexpf(mx, &ex);
        float scale = ldexpf(1.0f, -ex);
        for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
            for (uint32_t j = 0; j < rank; ++j) {
                r[row][j].hi *= scale;
                r[row][j].lo *= scale;
            }
        }
        *exp_sum += ex;
    }
}

static uint32_t apply_axis_step(
    df64_t r[DREAMS_NROWS][DREAMS_MAX_DIM],
    int32_t pos[DREAMS_NSHIFT],
    int axis,
    int sign,
    df64_t z,
    uint32_t rank,
    int32_t *exp_sum)
{
    int32_t eval_pos[DREAMS_NSHIFT];
    memcpy(eval_pos, pos, sizeof(eval_pos));

    int32_t a_int;
    int inverse;
    if (axis < (int)NX) {
        if (sign > 0) {
            a_int = pos[axis];
            inverse = 0;
        } else {
            eval_pos[axis] -= 1;
            a_int = pos[axis] - 1;
            inverse = 1;
        }
    } else {
        if (sign < 0) {
            a_int = pos[axis] - 1;
            inverse = 0;
        } else {
            eval_pos[axis] += 1;
            a_int = pos[axis];
            inverse = 1;
        }
    }

    if (a_int == 0) return DREAMS_ST_ZERO_AXIS_DENOMINATOR;

    df64_t col[DREAMS_MAX_DIM];
    if (!theta_companion_column(eval_pos, z, rank, col)) {
        return DREAMS_ST_THETA_LEAD_DEGENERATE;
    }

    df64_t inva = df64_div(df64_from_float(1.0f), df64_from_float((float)a_int));
    for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
        if (inverse) {
            uint32_t st = row_mul_inverse(r[row], col, inva, rank);
            if (st != DREAMS_ST_OK) return st;
        } else {
            row_mul_forward(r[row], col, inva, rank);
        }
    }

    pos[axis] += sign;
    normalize_pow2(r, rank, exp_sum);

    for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
        if (!isfinite(r[row][rank - 1].hi)) return DREAMS_ST_NONFINITE;
    }
    return DREAMS_ST_OK;
}

uint32_t dreams_run_trajectory(
    const int32_t dir[DREAMS_NSHIFT],
    const dreams_cmf_cfg_t *cfg,
    dreams_cmf_result_t *out)
{
    int32_t pos[DREAMS_NSHIFT];
    memcpy(pos, cfg->start, sizeof(pos));

    df64_t r[DREAMS_NROWS][DREAMS_MAX_DIM];
    for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
        for (uint32_t j = 0; j < DREAMS_MAX_DIM; ++j) {
            r[row][j] = df64_from_float(0.0f);
        }
    }
    r[0][cfg->row_num] = df64_from_float(1.0f);
    r[1][cfg->row_den] = df64_from_float(1.0f);

    for (uint32_t c = 0; c < DREAMS_NCHECK; ++c) {
        for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
            out->values[c][row].hi = NAN;
            out->values[c][row].lo = NAN;
        }
        out->exp[c] = 0;
    }

    int max_abs = 0;
    for (uint32_t d = 0; d < DREAMS_NSHIFT; ++d) {
        int v = dir[d] < 0 ? -dir[d] : dir[d];
        if (v > max_abs) max_abs = v;
    }

    uint32_t status = max_abs == 0 ? DREAMS_ST_NEEDS_REGULARIZATION : DREAMS_ST_OK;
    int32_t exp_sum = 0;
    uint32_t next_checkpoint = 0;

    for (uint32_t step = 0; step < cfg->n_steps && status == DREAMS_ST_OK; ++step) {
        for (int level = max_abs; level >= 1 && status == DREAMS_ST_OK; --level) {
            for (int axis = (int)DREAMS_NSHIFT - 1; axis >= 0; --axis) {
                int av = dir[axis] < 0 ? -dir[axis] : dir[axis];
                if (av < level) continue;
                int sign = dir[axis] > 0 ? 1 : -1;
                status = apply_axis_step(r, pos, axis, sign, cfg->z, cfg->rank, &exp_sum);
                if (status != DREAMS_ST_OK) break;
            }
        }

        uint32_t completed = step + 1u;
        while (status == DREAMS_ST_OK &&
               next_checkpoint < DREAMS_NCHECK &&
               cfg->checkpoint_steps[next_checkpoint] == completed) {
            out->values[next_checkpoint][0] = r[0][cfg->rank - 1];
            out->values[next_checkpoint][1] = r[1][cfg->rank - 1];
            out->exp[next_checkpoint] = exp_sum;
            ++next_checkpoint;
        }
    }

    out->status = status;
    return status;
}

typedef struct {
    const int32_t *dirs;
    size_t n_traj;
    const dreams_cmf_cfg_t *cfg;
    dreams_cmf_result_t *out;
    atomic_size_t next;
    size_t grain;
} batch_ctx_t;

static void *batch_worker(void *arg) {
    batch_ctx_t *ctx = (batch_ctx_t *)arg;
    for (;;) {
        size_t begin = atomic_fetch_add_explicit(&ctx->next, ctx->grain, memory_order_relaxed);
        if (begin >= ctx->n_traj) break;
        size_t end = begin + ctx->grain;
        if (end > ctx->n_traj) end = ctx->n_traj;

        for (size_t i = begin; i < end; ++i) {
            dreams_run_trajectory(
                &ctx->dirs[i * DREAMS_NSHIFT],
                ctx->cfg,
                &ctx->out[i]);
        }
    }
    return NULL;
}

void dreams_run_batch_pthreads(
    const int32_t *dirs,
    size_t n_traj,
    const dreams_cmf_cfg_t *cfg,
    dreams_cmf_result_t *out,
    unsigned n_threads,
    size_t grain)
{
    if (n_threads == 0) n_threads = 1;
    if (grain == 0) grain = 32;
    if ((size_t)n_threads > n_traj && n_traj > 0) n_threads = (unsigned)n_traj;
    if (n_threads <= 1 || n_traj <= 1) {
        for (size_t i = 0; i < n_traj; ++i) {
            dreams_run_trajectory(&dirs[i * DREAMS_NSHIFT], cfg, &out[i]);
        }
        return;
    }

    batch_ctx_t ctx;
    ctx.dirs = dirs;
    ctx.n_traj = n_traj;
    ctx.cfg = cfg;
    ctx.out = out;
    atomic_init(&ctx.next, 0);
    ctx.grain = grain;

    pthread_t *threads = (pthread_t *)calloc(n_threads, sizeof(*threads));
    if (!threads) abort();

    for (unsigned t = 0; t < n_threads; ++t) {
        if (pthread_create(&threads[t], NULL, batch_worker, &ctx) != 0) abort();
    }
    for (unsigned t = 0; t < n_threads; ++t) pthread_join(threads[t], NULL);
    free(threads);
}
