#pragma once
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <vector>

using BigInt = boost::multiprecision::cpp_int;
using BigFloat = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<200>>;

inline uint32_t mod_u32(const BigInt &x, uint32_t p) {
    return (uint32_t)(x % p);
}

inline uint32_t inv_u32(uint32_t a, uint32_t p) {
    int64_t t = 0, nt = 1;
    int64_t r = p, nr = a;
    while (nr != 0) {
        int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) throw std::runtime_error("non-invertible CRT modulus");
    if (t < 0) t += p;
    return (uint32_t)t;
}

inline BigInt garner(const uint32_t *residues, const std::vector<uint32_t> &p) {
    BigInt x = residues[0];
    BigInt M = p[0];
    for (size_t i = 1; i < p.size(); ++i) {
        uint32_t pi = p[i];
        uint32_t xm = mod_u32(x, pi);
        uint32_t Mm = mod_u32(M, pi);
        uint32_t diff = residues[i] >= xm ? residues[i] - xm : (uint32_t)((uint64_t)residues[i] + pi - xm);
        uint32_t c = (uint32_t)(((uint64_t)diff * inv_u32(Mm, pi)) % pi);
        x += M * c;
        M *= pi;
    }
    return x;
}

inline BigInt centered(BigInt x, const std::vector<uint32_t> &p) {
    BigInt M = 1;
    for (auto q : p) M *= q;
    BigInt half = M >> 1;
    if (x > half) x -= M;
    return x;
}

inline BigFloat ratio_decimal(const BigInt &num, const BigInt &den) {
    if (den == 0) throw std::runtime_error("zero denominator after CRT");
    return BigFloat(num.convert_to<std::string>()) / BigFloat(den.convert_to<std::string>());
}
