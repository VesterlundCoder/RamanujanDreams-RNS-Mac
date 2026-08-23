#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define N6 6u
#define E6 36u
#define MULS_PER_MATMUL 216.0

typedef struct {
    uint32_t p;
    uint64_t mu;
} mod_ctx_t;

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

static inline uint32_t add_mod(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t s = (uint64_t)a + (uint64_t)b;
    if (s >= p) s -= p;
    return (uint32_t)s;
}

static inline uint32_t mul_mod_div(uint32_t a, uint32_t b, uint32_t p) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)p);
}

static inline uint32_t mul_mod_barrett(uint32_t a, uint32_t b, mod_ctx_t m) {
    uint64_t x = (uint64_t)a * (uint64_t)b;
    uint64_t q = (uint64_t)(((__uint128_t)x * (__uint128_t)m.mu) >> 64);
    uint64_t r = x - q * (uint64_t)m.p;
    while (r >= m.p) r -= m.p;
    return (uint32_t)r;
}

static inline mod_ctx_t make_mod(uint32_t p) {
    mod_ctx_t m;
    m.p = p;
    m.mu = (uint64_t)((((__uint128_t)1) << 64) / p);
    return m;
}

static void matmul_div(const uint32_t *a, const uint32_t *b, uint32_t *c, mod_ctx_t m) {
    for (uint32_t i = 0; i < N6; ++i) {
        for (uint32_t j = 0; j < N6; ++j) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < N6; ++k) {
                acc = add_mod(acc, mul_mod_div(a[i * N6 + k], b[k * N6 + j], m.p), m.p);
            }
            c[i * N6 + j] = acc;
        }
    }
}

static void matmul_barrett(const uint32_t *a, const uint32_t *b, uint32_t *c, mod_ctx_t m) {
    for (uint32_t i = 0; i < N6; ++i) {
        for (uint32_t j = 0; j < N6; ++j) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < N6; ++k) {
                acc = add_mod(acc, mul_mod_barrett(a[i * N6 + k], b[k * N6 + j], m), m.p);
            }
            c[i * N6 + j] = acc;
        }
    }
}

typedef void (*rns_fn)(const uint32_t *, const uint32_t *, uint32_t *, mod_ctx_t);

typedef struct {
    uint32_t *a;
    const uint32_t *b;
    uint32_t *tmp;
    const mod_ctx_t *mods;
    size_t begin;
    size_t end;
    unsigned steps;
    rns_fn fn;
} worker_ctx_t;

static void *worker(void *vp) {
    worker_ctx_t *w = (worker_ctx_t *)vp;
    for (size_t t = w->begin; t < w->end; ++t) {
        uint32_t *a = w->a + t * E6;
        const uint32_t *b = w->b + t * E6;
        uint32_t *tmp = w->tmp + t * E6;
        mod_ctx_t m = w->mods[t];
        for (unsigned s = 0; s < w->steps; ++s) {
            w->fn(a, b, tmp, m);
            memcpy(a, tmp, E6 * sizeof(uint32_t));
        }
    }
    return NULL;
}

static void run_threads(uint32_t *a, const uint32_t *b, uint32_t *tmp,
                        const mod_ctx_t *mods, size_t pairs, unsigned steps,
                        unsigned threads, rns_fn fn) {
    if (threads < 1) threads = 1;
    if ((size_t)threads > pairs) threads = (unsigned)pairs;
    pthread_t *th = calloc(threads, sizeof(*th));
    worker_ctx_t *ctx = calloc(threads, sizeof(*ctx));
    if (!th || !ctx) abort();
    for (unsigned i = 0; i < threads; ++i) {
        size_t begin = pairs * i / threads;
        size_t end = pairs * (i + 1) / threads;
        ctx[i] = (worker_ctx_t){a, b, tmp, mods, begin, end, steps, fn};
        if (pthread_create(&th[i], NULL, worker, &ctx[i]) != 0) abort();
    }
    for (unsigned i = 0; i < threads; ++i) pthread_join(th[i], NULL);
    free(ctx);
    free(th);
}

static uint64_t checksum(const uint32_t *a, size_t pairs) {
    uint64_t s = 0;
    size_t stride = pairs > 10000 ? pairs / 10000 : 1;
    for (size_t t = 0; t < pairs; t += stride) {
        s ^= ((uint64_t)a[t * E6 + 0] << 32) ^ a[t * E6 + 35];
        s = (s << 7) | (s >> 57);
    }
    return s;
}

static unsigned cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (unsigned)n : 1u;
}

static void report(FILE *csv, const char *name, size_t pairs, unsigned steps,
                   unsigned threads, double sec, uint64_t sum) {
    double mm = (double)pairs * steps;
    double mps = mm / sec;
    double gmuls = mm * MULS_PER_MATMUL / sec / 1e9;
    printf("%-24s %10.3f Mmat/s  %10.3f Gmodmul/s  %8.4f s checksum=%016" PRIx64 "\n",
           name, mps / 1e6, gmuls, sec, sum);
    if (csv) fprintf(csv, "%s,%zu,%u,%u,%.9f,%.6f,%.6f,%016" PRIx64 "\n",
                     name, pairs, steps, threads, sec, mps, gmuls, sum);
}

int main(int argc, char **argv) {
    size_t pairs = 20000;
    unsigned steps = 20;
    unsigned threads = cpu_count();
    uint64_t seed = 7;
    const char *csv_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else {
            fprintf(stderr, "usage: rns_matmul6_bench [--pairs N] [--steps S] [--threads T] [--seed N] [--csv FILE]\n");
            return 2;
        }
    }
    if (!pairs || !steps || !threads) return 2;

    static const uint32_t moduli[] = {
        2147483647u, 2147483629u, 2147483587u, 2147483579u,
        2147483563u, 2147483549u, 2147483543u, 2147483497u,
        2147483489u, 2147483477u, 2147483423u, 2147483399u,
        2147483353u, 2147483323u, 2147483269u, 2147483249u
    };
    const size_t nmod = sizeof(moduli) / sizeof(moduli[0]);

    size_t count = pairs * E6;
    uint32_t *a0 = NULL, *a = NULL, *b = NULL, *tmp = NULL;
    mod_ctx_t *mods = NULL;
    size_t mat_bytes = (count * sizeof(uint32_t) + 63u) & ~(size_t)63u;
    size_t mod_bytes = (pairs * sizeof(mod_ctx_t) + 63u) & ~(size_t)63u;
    if (posix_memalign((void **)&a0, 64, mat_bytes) ||
        posix_memalign((void **)&a, 64, mat_bytes) ||
        posix_memalign((void **)&b, 64, mat_bytes) ||
        posix_memalign((void **)&tmp, 64, mat_bytes) ||
        posix_memalign((void **)&mods, 64, mod_bytes)) {
        fprintf(stderr, "aligned allocation failed\n");
        return 1;
    }
    if (!a0 || !a || !b || !tmp || !mods) { fprintf(stderr, "allocation failed\n"); return 1; }

    for (size_t t = 0; t < pairs; ++t) {
        uint32_t p = moduli[t % nmod];
        mods[t] = make_mod(p);
        for (uint32_t e = 0; e < E6; ++e) {
            a0[t * E6 + e] = (uint32_t)(splitmix64(&seed) % p);
            b[t * E6 + e] = (uint32_t)(splitmix64(&seed) % p);
        }
    }

    // Validate Barrett exactly against hardware division on a sample.
    for (size_t t = 0; t < (pairs < 1024 ? pairs : 1024); ++t) {
        mod_ctx_t m = mods[t];
        for (uint32_t e = 0; e < E6; ++e) {
            uint32_t x = a0[t * E6 + e], y = b[t * E6 + e];
            uint32_t u = mul_mod_div(x, y, m.p);
            uint32_t v = mul_mod_barrett(x, y, m);
            if (u != v) {
                fprintf(stderr, "Barrett validation failed: p=%u x=%u y=%u div=%u barrett=%u\n",
                        m.p, x, y, u, v);
                return 1;
            }
        }
    }

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { fprintf(stderr, "cannot open %s: %s\n", csv_path, strerror(errno)); return 1; }
        fprintf(csv, "backend,pairs,steps,threads,seconds,matmuls_per_sec,gmodmul_per_sec,checksum\n");
    }

    printf("6x6 uint32 RNS matmul chain: pairs=%zu steps=%u threads=%u\n", pairs, steps, threads);

    double t0, t1;
    memcpy(a, a0, count * sizeof(uint32_t));
    t0 = now_sec();
    run_threads(a, b, tmp, mods, pairs, steps, 1, matmul_div);
    t1 = now_sec();
    report(csv, "rns_div_1t", pairs, steps, 1, t1 - t0, checksum(a, pairs));

    memcpy(a, a0, count * sizeof(uint32_t));
    t0 = now_sec();
    run_threads(a, b, tmp, mods, pairs, steps, threads, matmul_div);
    t1 = now_sec();
    uint64_t sum_div = checksum(a, pairs);
    report(csv, "rns_div_pthreads", pairs, steps, threads, t1 - t0, sum_div);

    memcpy(a, a0, count * sizeof(uint32_t));
    t0 = now_sec();
    run_threads(a, b, tmp, mods, pairs, steps, 1, matmul_barrett);
    t1 = now_sec();
    report(csv, "rns_barrett_1t", pairs, steps, 1, t1 - t0, checksum(a, pairs));

    memcpy(a, a0, count * sizeof(uint32_t));
    t0 = now_sec();
    run_threads(a, b, tmp, mods, pairs, steps, threads, matmul_barrett);
    t1 = now_sec();
    uint64_t sum_barrett = checksum(a, pairs);
    report(csv, "rns_barrett_pthreads", pairs, steps, threads, t1 - t0, sum_barrett);

    if (sum_div != sum_barrett) {
        fprintf(stderr, "ERROR: final checksum mismatch between div and Barrett backends\n");
        return 1;
    }

    if (csv) fclose(csv);
    free(mods); free(tmp); free(b); free(a); free(a0);
    return 0;
}
