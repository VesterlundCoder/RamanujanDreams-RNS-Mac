#pragma once
#include "crt.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <vector>

struct RatioScanConfig {
    bool ordered = true;          // true => i/j and j/i are both candidates
    BigFloat maxRelativeDelta = BigFloat("1e-30");
    bool emitOnlyConverged = true;
};

inline BigFloat abs_big(const BigFloat &x) { return x < 0 ? -x : x; }

inline BigFloat relative_delta(const BigFloat &a, const BigFloat &b) {
    BigFloat scale = std::max(BigFloat(1), abs_big(b));
    return abs_big(b - a) / scale;
}

// matrices[c][e] contains the CRT-reconstructed exact integer for checkpoint c,
// matrix element e = row*m + col.
inline void scan_all_matrix_ratios(
    uint64_t trajectoryId,
    uint32_t m,
    const std::vector<std::vector<BigInt>> &matrices,
    const std::vector<uint32_t> &checkpointSteps,
    const RatioScanConfig &cfg,
    std::ostream &out)
{
    if (matrices.size() < 2) return;
    const uint32_t E = m * m;
    const size_t C = matrices.size();

    for (uint32_t numIdx = 0; numIdx < E; ++numIdx) {
        for (uint32_t denIdx = 0; denIdx < E; ++denIdx) {
            if (numIdx == denIdx) continue;
            if (!cfg.ordered && denIdx <= numIdx) continue;

            bool valid = true;
            std::vector<BigFloat> limits(C);
            for (size_t c = 0; c < C; ++c) {
                const BigInt &D = matrices[c][denIdx];
                if (D == 0) { valid = false; break; }
                limits[c] = ratio_decimal(matrices[c][numIdx], D);
            }
            if (!valid) continue;

            BigFloat maxRel = 0;
            BigFloat lastAbs = 0;
            BigFloat lastRel = 0;
            for (size_t c = 1; c < C; ++c) {
                BigFloat da = abs_big(limits[c] - limits[c-1]);
                BigFloat dr = relative_delta(limits[c-1], limits[c]);
                if (dr > maxRel) maxRel = dr;
                if (c + 1 == C) { lastAbs = da; lastRel = dr; }
            }

            if (cfg.emitOnlyConverged && lastRel > cfg.maxRelativeDelta) continue;

            out << trajectoryId << '\t'
                << numIdx << '\t' << (numIdx / m) << '\t' << (numIdx % m) << '\t'
                << denIdx << '\t' << (denIdx / m) << '\t' << (denIdx % m);
            for (size_t c = 0; c < C; ++c) {
                out << '\t' << checkpointSteps[c]
                    << '\t' << std::setprecision(200) << limits[c];
            }
            out << '\t' << std::setprecision(80) << lastAbs
                << '\t' << lastRel
                << '\t' << maxRel << '\n';
        }
    }
}
