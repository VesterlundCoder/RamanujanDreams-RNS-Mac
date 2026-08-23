#ifndef DREAMS_DF64_H
#define DREAMS_DF64_H

#include <math.h>

typedef struct {
    float hi;
    float lo;
} df64_t;

static inline df64_t df64_from_float(float x) {
    df64_t r = {x, 0.0f};
    return r;
}

static inline df64_t df64_renorm(float s, float e) {
    float hi = s + e;
    float lo = e - (hi - s);
    df64_t r = {hi, lo};
    return r;
}

static inline df64_t df64_add(df64_t a, df64_t b) {
    float s = a.hi + b.hi;
    float v = s - a.hi;
    float e = (a.hi - (s - v)) + (b.hi - v);

    float t = a.lo + b.lo;
    float w = t - a.lo;
    float f = (a.lo - (t - w)) + (b.lo - w);

    e += t;
    float hi = s + e;
    float lo = e - (hi - s);
    lo += f;
    return df64_renorm(hi, lo);
}

static inline df64_t df64_neg(df64_t a) {
    df64_t r = {-a.hi, -a.lo};
    return r;
}

static inline df64_t df64_sub(df64_t a, df64_t b) {
    return df64_add(a, df64_neg(b));
}

static inline df64_t df64_mul(df64_t a, df64_t b) {
    float p = a.hi * b.hi;
    float e = fmaf(a.hi, b.hi, -p);
    e = fmaf(a.hi, b.lo, e);
    e = fmaf(a.lo, b.hi, e);
    e = fmaf(a.lo, b.lo, e);
    return df64_renorm(p, e);
}

static inline df64_t df64_div(df64_t a, df64_t b) {
    float q1 = a.hi / b.hi;
    df64_t q = df64_from_float(q1);
    df64_t r = df64_sub(a, df64_mul(b, q));
    float q2 = (r.hi + r.lo) / b.hi;
    q = df64_add(q, df64_from_float(q2));
    r = df64_sub(a, df64_mul(b, q));
    float q3 = (r.hi + r.lo) / b.hi;
    return df64_add(q, df64_from_float(q3));
}

static inline double df64_to_double(df64_t a) {
    return (double)a.hi + (double)a.lo;
}

#endif
