#ifndef DREAMS_CMF_CPU_H
#define DREAMS_CMF_CPU_H

#include <stddef.h>
#include <stdint.h>
#include "df64.h"

#define DREAMS_MAX_DIM 6u
#define DREAMS_NSHIFT 11u
#define DREAMS_NROWS 2u
#define DREAMS_NCHECK 2u

enum dreams_status {
    DREAMS_ST_OK = 0,
    DREAMS_ST_ZERO_AXIS_DENOMINATOR = 1,
    DREAMS_ST_THETA_LEAD_DEGENERATE = 2,
    DREAMS_ST_INVERSE_SINGULAR = 3,
    DREAMS_ST_NONFINITE = 4,
    DREAMS_ST_NEEDS_REGULARIZATION = 5
};

typedef struct {
    uint32_t n_steps;
    uint32_t rank;
    uint32_t row_num;
    uint32_t row_den;
    uint32_t checkpoint_steps[DREAMS_NCHECK];
    int32_t start[DREAMS_NSHIFT];
    df64_t z;
} dreams_cmf_cfg_t;

typedef struct {
    df64_t values[DREAMS_NCHECK][DREAMS_NROWS];
    int32_t exp[DREAMS_NCHECK];
    uint32_t status;
} dreams_cmf_result_t;

uint32_t dreams_run_trajectory(
    const int32_t dir[DREAMS_NSHIFT],
    const dreams_cmf_cfg_t *cfg,
    dreams_cmf_result_t *out);

void dreams_run_batch_pthreads(
    const int32_t *dirs,
    size_t n_traj,
    const dreams_cmf_cfg_t *cfg,
    dreams_cmf_result_t *out,
    unsigned n_threads,
    size_t grain);

#endif
