#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define DREAMS_HAVE_NEON 1
#else
#define DREAMS_HAVE_NEON 0
#endif

#define N6 6u
#define E6 36u
#define FLOPS_PER_MATMUL 432.0

typedef void (*matmul_fn)(const float *, const float *, float *);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static float rand_float(uint64_t *state) {
    uint32_t x = (uint32_t)(splitmix64(state) >> 40);
    return ((float)x / 16777216.0f) * 2.0f - 1.0f;
}

static void matmul_scalar(const float *a, const float *b, float *c) {
    for (uint32_t i = 0; i < N6; ++i) {
        for (uint32_t j = 0; j < N6; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < N6; ++k) s += a[i * N6 + k] * b[k * N6 + j];
            c[i * N6 + j] = s;
        }
    }
}

static void matmul_unrolled(const float *a, const float *b, float *c) {
    for (uint32_t i = 0; i < N6; ++i) {
        const float a0 = a[i * N6 + 0];
        const float a1 = a[i * N6 + 1];
        const float a2 = a[i * N6 + 2];
        const float a3 = a[i * N6 + 3];
        const float a4 = a[i * N6 + 4];
        const float a5 = a[i * N6 + 5];
        for (uint32_t j = 0; j < N6; ++j) {
            c[i * N6 + j] =
                a0 * b[0 * N6 + j] +
                a1 * b[1 * N6 + j] +
                a2 * b[2 * N6 + j] +
                a3 * b[3 * N6 + j] +
                a4 * b[4 * N6 + j] +
                a5 * b[5 * N6 + j];
        }
    }
}

typedef struct {
    const float *a;
    const float *b;
    float *c;
    size_t begin;
    size_t end;
    unsigned repeats;
    matmul_fn fn;
} aos_thread_ctx_t;

static void *aos_worker(void *p) {
    aos_thread_ctx_t *ctx = (aos_thread_ctx_t *)p;
    for (unsigned r = 0; r < ctx->repeats; ++r) {
        for (size_t t = ctx->begin; t < ctx->end; ++t) {
            ctx->fn(ctx->a + t * E6, ctx->b + t * E6, ctx->c + t * E6);
        }
    }
    return NULL;
}

static void run_aos_threads(
    const float *a, const float *b, float *c,
    size_t n, unsigned repeats, unsigned threads, matmul_fn fn)
{
    if (threads < 1) threads = 1;
    if ((size_t)threads > n) threads = (unsigned)n;
    if (threads <= 1) {
        aos_thread_ctx_t ctx = {a, b, c, 0, n, repeats, fn};
        aos_worker(&ctx);
        return;
    }

    pthread_t *th = (pthread_t *)calloc(threads, sizeof(*th));
    aos_thread_ctx_t *ctx = (aos_thread_ctx_t *)calloc(threads, sizeof(*ctx));
    if (!th || !ctx) abort();

    for (unsigned i = 0; i < threads; ++i) {
        size_t begin = n * i / threads;
        size_t end = n * (i + 1) / threads;
        ctx[i] = (aos_thread_ctx_t){a, b, c, begin, end, repeats, fn};
        if (pthread_create(&th[i], NULL, aos_worker, &ctx[i]) != 0) abort();
    }
    for (unsigned i = 0; i < threads; ++i) pthread_join(th[i], NULL);
    free(ctx);
    free(th);
}

static void pack_soa(const float *aos, float *soa, size_t n, size_t npad) {
    for (uint32_t e = 0; e < E6; ++e) {
        for (size_t t = 0; t < n; ++t) soa[(size_t)e * npad + t] = aos[t * E6 + e];
        for (size_t t = n; t < npad; ++t) soa[(size_t)e * npad + t] = 0.0f;
    }
}

static void soa_scalar_range(
    const float *a, const float *b, float *c,
    size_t npad, size_t begin, size_t end, unsigned repeats)
{
    for (unsigned rep = 0; rep < repeats; ++rep) {
        for (size_t t = begin; t < end; ++t) {
            for (uint32_t i = 0; i < N6; ++i) {
                for (uint32_t j = 0; j < N6; ++j) {
                    float s = 0.0f;
                    for (uint32_t k = 0; k < N6; ++k) {
                        s += a[((size_t)i * N6 + k) * npad + t] *
                             b[((size_t)k * N6 + j) * npad + t];
                    }
                    c[((size_t)i * N6 + j) * npad + t] = s;
                }
            }
        }
    }
}

#if DREAMS_HAVE_NEON
static void soa_neon_range(
    const float *a, const float *b, float *c,
    size_t npad, size_t begin, size_t end, unsigned repeats)
{
    size_t t = begin;
    for (unsigned rep = 0; rep < repeats; ++rep) {
        t = begin;
        for (; t + 4 <= end; t += 4) {
            for (uint32_t i = 0; i < N6; ++i) {
                for (uint32_t j = 0; j < N6; ++j) {
                    float32x4_t s = vdupq_n_f32(0.0f);
                    for (uint32_t k = 0; k < N6; ++k) {
                        float32x4_t av = vld1q_f32(a + ((size_t)i * N6 + k) * npad + t);
                        float32x4_t bv = vld1q_f32(b + ((size_t)k * N6 + j) * npad + t);
                        s = vfmaq_f32(s, av, bv);
                    }
                    vst1q_f32(c + ((size_t)i * N6 + j) * npad + t, s);
                }
            }
        }
        if (t < end) soa_scalar_range(a, b, c, npad, t, end, 1);
    }
}
#endif

typedef struct {
    const float *a;
    const float *b;
    float *c;
    size_t npad;
    size_t begin;
    size_t end;
    unsigned repeats;
    int use_neon;
} soa_thread_ctx_t;

static void *soa_worker(void *p) {
    soa_thread_ctx_t *ctx = (soa_thread_ctx_t *)p;
#if DREAMS_HAVE_NEON
    if (ctx->use_neon) {
        soa_neon_range(ctx->a, ctx->b, ctx->c, ctx->npad, ctx->begin, ctx->end, ctx->repeats);
        return NULL;
    }
#else
    (void)ctx->use_neon;
#endif
    soa_scalar_range(ctx->a, ctx->b, ctx->c, ctx->npad, ctx->begin, ctx->end, ctx->repeats);
    return NULL;
}

static void run_soa_threads(
    const float *a, const float *b, float *c,
    size_t n, size_t npad, unsigned repeats, unsigned threads, int use_neon)
{
    if (threads < 1) threads = 1;
    if ((size_t)threads > n) threads = (unsigned)n;
    size_t quantum = use_neon ? 4u : 1u;
    size_t blocks = (n + quantum - 1) / quantum;
    if ((size_t)threads > blocks) threads = (unsigned)blocks;
    if (threads < 1) threads = 1;

    pthread_t *th = (pthread_t *)calloc(threads, sizeof(*th));
    soa_thread_ctx_t *ctx = (soa_thread_ctx_t *)calloc(threads, sizeof(*ctx));
    if (!th || !ctx) abort();

    for (unsigned i = 0; i < threads; ++i) {
        size_t bb = blocks * i / threads;
        size_t be = blocks * (i + 1) / threads;
        size_t begin = bb * quantum;
        size_t end = be * quantum;
        if (end > n) end = n;
        ctx[i] = (soa_thread_ctx_t){a, b, c, npad, begin, end, repeats, use_neon};
        if (pthread_create(&th[i], NULL, soa_worker, &ctx[i]) != 0) abort();
    }
    for (unsigned i = 0; i < threads; ++i) pthread_join(th[i], NULL);
    free(ctx);
    free(th);
}

static double checksum_aos(const float *c, size_t n) {
    double s = 0.0;
    size_t stride = n > 10000 ? n / 10000 : 1;
    for (size_t t = 0; t < n; t += stride) {
        for (uint32_t e = 0; e < E6; e += 5) s += c[t * E6 + e];
    }
    return s;
}

static double checksum_soa(const float *c, size_t n, size_t npad) {
    double s = 0.0;
    size_t stride = n > 10000 ? n / 10000 : 1;
    for (size_t t = 0; t < n; t += stride) {
        for (uint32_t e = 0; e < E6; e += 5) s += c[(size_t)e * npad + t];
    }
    return s;
}

static void emit_result(FILE *csv, const char *backend, size_t n, unsigned repeats,
                        unsigned threads, double sec, double checksum) {
    double ops = (double)n * (double)repeats;
    double mps = ops / sec;
    double gflops = ops * FLOPS_PER_MATMUL / sec / 1e9;
    printf("%-24s %10.3f Mmat/s  %10.3f GFLOP/s  %8.4f s  checksum=%.9g\n",
           backend, mps / 1e6, gflops, sec, checksum);
    if (csv) fprintf(csv, "%s,%zu,%u,%u,%.9f,%.6f,%.6f,%.17g\n",
                     backend, n, repeats, threads, sec, mps, gflops, checksum);
}

static unsigned cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (unsigned)n : 1u;
}

int main(int argc, char **argv) {
    size_t n = 200000;
    unsigned repeats = 5;
    unsigned threads = cpu_count();
    uint64_t seed = 1;
    const char *csv_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else {
            fprintf(stderr, "usage: matmul6_bench [--n N] [--repeats R] [--threads T] [--seed S] [--csv FILE]\n");
            return 2;
        }
    }
    if (n == 0 || repeats == 0 || threads == 0) return 2;

    size_t npad = (n + 3u) & ~(size_t)3u;
    size_t aos_count = n * E6;
    size_t soa_count = npad * E6;
    float *a = NULL, *b = NULL, *c = NULL, *as = NULL, *bs = NULL, *cs = NULL;
    size_t aos_bytes = (aos_count * sizeof(float) + 63u) & ~(size_t)63u;
    size_t soa_bytes = (soa_count * sizeof(float) + 63u) & ~(size_t)63u;
    if (posix_memalign((void **)&a, 64, aos_bytes) ||
        posix_memalign((void **)&b, 64, aos_bytes) ||
        posix_memalign((void **)&c, 64, aos_bytes) ||
        posix_memalign((void **)&as, 64, soa_bytes) ||
        posix_memalign((void **)&bs, 64, soa_bytes) ||
        posix_memalign((void **)&cs, 64, soa_bytes)) {
        fprintf(stderr, "aligned allocation failed\n");
        return 1;
    }
    if (!a || !b || !c || !as || !bs || !cs) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < aos_count; ++i) {
        a[i] = rand_float(&seed);
        b[i] = rand_float(&seed);
    }
    pack_soa(a, as, n, npad);
    pack_soa(b, bs, n, npad);

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { fprintf(stderr, "cannot open %s: %s\n", csv_path, strerror(errno)); return 1; }
        fprintf(csv, "backend,n,repeats,threads,seconds,matmuls_per_sec,gflops,checksum\n");
    }

    printf("6x6 FP32 batched matmul benchmark: n=%zu repeats=%u threads=%u NEON=%s\n",
           n, repeats, threads, DREAMS_HAVE_NEON ? "yes" : "no");

    double t0, t1;

    memset(c, 0, aos_count * sizeof(float));
    t0 = now_sec();
    run_aos_threads(a, b, c, n, repeats, 1, matmul_scalar);
    t1 = now_sec();
    emit_result(csv, "aos_scalar_1t", n, repeats, 1, t1 - t0, checksum_aos(c, n));

    memset(c, 0, aos_count * sizeof(float));
    t0 = now_sec();
    run_aos_threads(a, b, c, n, repeats, 1, matmul_unrolled);
    t1 = now_sec();
    emit_result(csv, "aos_unrolled_1t", n, repeats, 1, t1 - t0, checksum_aos(c, n));

    memset(c, 0, aos_count * sizeof(float));
    t0 = now_sec();
    run_aos_threads(a, b, c, n, repeats, threads, matmul_unrolled);
    t1 = now_sec();
    emit_result(csv, "aos_unrolled_pthreads", n, repeats, threads, t1 - t0, checksum_aos(c, n));

    memset(cs, 0, soa_count * sizeof(float));
    t0 = now_sec();
    run_soa_threads(as, bs, cs, n, npad, repeats, threads, 0);
    t1 = now_sec();
    emit_result(csv, "soa_scalar_pthreads", n, repeats, threads, t1 - t0, checksum_soa(cs, n, npad));

#if DREAMS_HAVE_NEON
    memset(cs, 0, soa_count * sizeof(float));
    t0 = now_sec();
    run_soa_threads(as, bs, cs, n, npad, repeats, threads, 1);
    t1 = now_sec();
    emit_result(csv, "soa_neon4_pthreads", n, repeats, threads, t1 - t0, checksum_soa(cs, n, npad));
#endif

    if (csv) fclose(csv);
    free(cs); free(bs); free(as); free(c); free(b); free(a);
    return 0;
}
