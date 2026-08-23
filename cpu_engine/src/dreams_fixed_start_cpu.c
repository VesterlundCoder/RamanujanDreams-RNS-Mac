#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include "cmf_cpu.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static int parse_start(const char *s, int32_t out[DREAMS_NSHIFT]) {
    char *copy = strdup(s);
    if (!copy) return 0;
    char *save = NULL;
    char *tok = strtok_r(copy, ",", &save);
    uint32_t i = 0;
    while (tok) {
        if (i >= DREAMS_NSHIFT) {
            free(copy);
            return 0;
        }
        char *end = NULL;
        long v = strtol(tok, &end, 10);
        if (!end || *end != '\0' || v < INT32_MIN || v > INT32_MAX) {
            free(copy);
            return 0;
        }
        out[i++] = (int32_t)v;
        tok = strtok_r(NULL, ",", &save);
    }
    free(copy);
    return i == DREAMS_NSHIFT;
}

static unsigned auto_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    return (unsigned)n;
}

static uint32_t parse_u32(const char *s, const char *name) {
    char *end = NULL;
    unsigned long x = strtoul(s, &end, 10);
    if (!end || *end != '\0' || x > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)x;
}

static int32_t parse_i32(const char *s, const char *name) {
    char *end = NULL;
    long x = strtol(s, &end, 10);
    if (!end || *end != '\0' || x < INT32_MIN || x > INT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (int32_t)x;
}

static size_t parse_size(const char *s, const char *name) {
    char *end = NULL;
    unsigned long long x = strtoull(s, &end, 10);
    if (!end || *end != '\0' || x > SIZE_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (size_t)x;
}

static void write_or_die(const void *ptr, size_t size, size_t count, FILE *f) {
    if (fwrite(ptr, size, count, f) != count) die("output write failed");
}

static void write_header(
    FILE *out,
    const dreams_cmf_cfg_t *cfg,
    int32_t z_num,
    int32_t z_den,
    uint64_t n_traj_placeholder,
    long *n_traj_offset)
{
    const char magic[8] = {'F','S','D','F','6','4','0','1'};
    write_or_die(magic, 1, 8, out);

    uint32_t hdr32[8] = {
        DREAMS_MAX_DIM,
        cfg->rank,
        DREAMS_NSHIFT,
        cfg->n_steps,
        cfg->checkpoint_steps[0],
        cfg->checkpoint_steps[1],
        cfg->row_num,
        cfg->row_den
    };
    int32_t hz[2] = {z_num, z_den};

    write_or_die(hdr32, sizeof(uint32_t), 8, out);
    write_or_die(hz, sizeof(int32_t), 2, out);
    write_or_die(cfg->start, sizeof(int32_t), DREAMS_NSHIFT, out);

    long p = ftell(out);
    if (p < 0) die("ftell failed");
    *n_traj_offset = p;
    write_or_die(&n_traj_placeholder, sizeof(uint64_t), 1, out);
}

static void write_result(FILE *out, const dreams_cmf_result_t *r) {
    for (uint32_t c = 0; c < DREAMS_NCHECK; ++c) {
        for (uint32_t row = 0; row < DREAMS_NROWS; ++row) {
            write_or_die(&r->values[c][row].hi, sizeof(float), 1, out);
            write_or_die(&r->values[c][row].lo, sizeof(float), 1, out);
        }
    }
    write_or_die(r->exp, sizeof(int32_t), DREAMS_NCHECK, out);
    write_or_die(&r->status, sizeof(uint32_t), 1, out);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
            "usage: dreams_fixed_start_cpu directions.bin output.bin "
            "absolute_start_csv row_num row_den "
            "[z_num=1] [z_den=2] [nSteps=1000] [chunk=262144] "
            "[threads=auto] [grain=32]\n\n"
            "Semantics intentionally match dreams_fixed_start_df64:\n"
            "  * absolute_start is v0\n"
            "  * first trajectory operator is evaluated at n=0\n"
            "  * output format is FSDF6401\n"
            "  * two projected rows and two checkpoints\n");
        return 2;
    }

    const char *dir_path = argv[1];
    const char *out_path = argv[2];
    int32_t start[DREAMS_NSHIFT];
    if (!parse_start(argv[3], start)) die("absolute_start must contain exactly 11 int32 values");

    uint32_t row_num = parse_u32(argv[4], "row_num");
    uint32_t row_den = parse_u32(argv[5], "row_den");
    int32_t z_num = argc > 6 ? parse_i32(argv[6], "z_num") : 1;
    int32_t z_den = argc > 7 ? parse_i32(argv[7], "z_den") : 2;
    uint32_t n_steps = argc > 8 ? parse_u32(argv[8], "nSteps") : 1000;
    size_t chunk = argc > 9 ? parse_size(argv[9], "chunk") : 262144;
    unsigned threads = argc > 10 ? (unsigned)parse_u32(argv[10], "threads") : auto_threads();
    size_t grain = argc > 11 ? parse_size(argv[11], "grain") : 32;

    if (z_den == 0) die("z_den must be nonzero");
    if (n_steps == 0) die("nSteps=0 means zero operators; use nSteps>=1 for N=0 start semantics");
    if (chunk == 0) die("chunk must be > 0");
    if (threads == 0) threads = 1;
    if (grain == 0) grain = 1;

    uint32_t rank = z_num == z_den ? 5u : 6u;
    if (row_num >= rank || row_den >= rank || row_num == row_den) {
        die("row_num and row_den must be distinct active rows < rank");
    }

    dreams_cmf_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_steps = n_steps;
    cfg.rank = rank;
    cfg.row_num = row_num;
    cfg.row_den = row_den;
    cfg.checkpoint_steps[0] = n_steps / 2u;
    if (cfg.checkpoint_steps[0] < 1u) cfg.checkpoint_steps[0] = 1u;
    cfg.checkpoint_steps[1] = n_steps;
    memcpy(cfg.start, start, sizeof(start));

    double z = (double)z_num / (double)z_den;
    cfg.z.hi = (float)z;
    cfg.z.lo = (float)(z - (double)cfg.z.hi);

    fprintf(stderr,
        "CPU fixed-start df64: threads=%u grain=%zu rank=%u pair=(%u,%u) "
        "steps=%u checkpoints=%u,%u chunk=%zu\n",
        threads, grain, rank, row_num, row_den, n_steps,
        cfg.checkpoint_steps[0], cfg.checkpoint_steps[1], chunk);

    FILE *in = fopen(dir_path, "rb");
    if (!in) die_errno("cannot open input", dir_path);
    FILE *out = fopen(out_path, "wb");
    if (!out) die_errno("cannot open output", out_path);

    long n_traj_offset = 0;
    write_header(out, &cfg, z_num, z_den, 0, &n_traj_offset);

    if (chunk > SIZE_MAX / DREAMS_NSHIFT / sizeof(int32_t)) die("chunk too large");
    int32_t *dirs = (int32_t *)malloc(chunk * DREAMS_NSHIFT * sizeof(int32_t));
    dreams_cmf_result_t *results = (dreams_cmf_result_t *)malloc(chunk * sizeof(*results));
    if (!dirs || !results) die("allocation failed; reduce chunk");

    uint64_t total = 0;
    uint64_t ok = 0;
    double t0 = now_sec();
    double compute_sec = 0.0;

    for (;;) {
        size_t got = 0;
        while (got < chunk) {
            size_t n = fread(&dirs[got * DREAMS_NSHIFT], sizeof(int32_t), DREAMS_NSHIFT, in);
            if (n == DREAMS_NSHIFT) {
                ++got;
                continue;
            }
            if (n != 0) die("truncated directions.bin record");
            if (ferror(in)) die("input read failed");
            break;
        }
        if (got == 0) break;

        double c0 = now_sec();
        dreams_run_batch_pthreads(dirs, got, &cfg, results, threads, grain);
        double c1 = now_sec();
        compute_sec += c1 - c0;

        for (size_t i = 0; i < got; ++i) {
            write_result(out, &results[i]);
            if (results[i].status == DREAMS_ST_OK) ++ok;
        }

        total += (uint64_t)got;
        double wall = now_sec() - t0;
        fprintf(stderr,
            "done %" PRIu64 " dirs (%" PRIu64 " OK), %.2f dir/s wall, %.2f dir/s compute\n",
            total, ok,
            wall > 0.0 ? (double)total / wall : 0.0,
            compute_sec > 0.0 ? (double)total / compute_sec : 0.0);
    }

    if (fseek(out, n_traj_offset, SEEK_SET) != 0) die("fseek failed");
    write_or_die(&total, sizeof(uint64_t), 1, out);
    fclose(out);
    fclose(in);
    free(results);
    free(dirs);

    double wall = now_sec() - t0;
    fprintf(stderr,
        "TOTAL %" PRIu64 " directions (%" PRIu64 " OK, %" PRIu64 " flagged) "
        "in %.6f s => %.2f dir/s wall; compute %.6f s => %.2f dir/s\n",
        total, ok, total - ok, wall,
        wall > 0.0 ? (double)total / wall : 0.0,
        compute_sec,
        compute_sec > 0.0 ? (double)total / compute_sec : 0.0);

    return 0;
}
