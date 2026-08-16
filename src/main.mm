#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/metal_rns.hpp"
#include "crt.hpp"
#include "ratio_scan.hpp"

static std::string readText(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) throw std::runtime_error("Missing Metal function");
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) throw std::runtime_error([[err localizedDescription] UTF8String]);
    return p;
}

static void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, std::max<NSUInteger>(w, 256));
    tg = (tg / w) * w;
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

// Binary trajectory format: dim int32 shift values followed by dim int32 direction values.
static size_t readChunk(std::ifstream &f, uint32_t dim, size_t maxB,
                        std::vector<int32_t> &shift, std::vector<int32_t> &dir) {
    shift.resize(maxB * dim);
    dir.resize(maxB * dim);
    size_t got = 0;
    std::vector<int32_t> rec(2 * dim);
    while (got < maxB && f.read(reinterpret_cast<char*>(rec.data()), rec.size() * sizeof(int32_t))) {
        std::copy(rec.begin(), rec.begin() + dim, shift.begin() + got * dim);
        std::copy(rec.begin() + dim, rec.end(), dir.begin() + got * dim);
        ++got;
    }
    shift.resize(got * dim);
    dir.resize(got * dim);
    return got;
}

static std::vector<uint32_t> parseCheckpoints(const std::string &s, uint32_t nSteps) {
    std::vector<uint32_t> cp;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        uint32_t v = (uint32_t)std::stoul(tok);
        if (v == 0 || v > nSteps) throw std::runtime_error("checkpoint outside 1..nSteps");
        cp.push_back(v);
    }
    std::sort(cp.begin(), cp.end());
    cp.erase(std::unique(cp.begin(), cp.end()), cp.end());
    if (cp.size() < 2 || cp.size() > MAX_CHECKPOINTS_HOST)
        throw std::runtime_error("need 2..4 unique checkpoints");
    if (cp.back() != nSteps)
        throw std::runtime_error("last checkpoint must equal nSteps so the full final matrix is retained");
    return cp;
}

// Integration hook. Populate this from the repository's existing 6F5 compiler.
static ProgramHost load6F5ProgramOrThrow(uint32_t dim, uint32_t K) {
    (void)K;
    ProgramHost prog;
    prog.m = 10;
    prog.dim = dim;
    throw std::runtime_error(
        "6F5 bytecode loader not wired yet. Populate ProgramHost from cmf_6f5.json / the existing compiler. "
        "This fail-fast guard prevents scientifically invalid output from a placeholder matrix."
    );
}

static std::vector<uint32_t> loadPrimesOrThrow(uint32_t K) {
    (void)K;
    throw std::runtime_error(
        "Prime loader not wired yet. Reuse the exact prime set used to build ProgramHost.constTable in the ROCm pipeline."
    );
}

static void writeHeader(std::ostream &out, const std::vector<uint32_t> &cp) {
    out << "trajectory_id\tnum_idx\tnum_row\tnum_col\tden_idx\tden_row\tden_col";
    for (size_t c = 0; c < cp.size(); ++c)
        out << "\tcheckpoint_" << c << "\tlimit_" << c << "_200d";
    out << "\tdelta_abs_last\tdelta_rel_last\tdelta_rel_max\n";
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 8) {
            std::cerr
                << "usage: dreams_rns_metal trajectories.bin dim K chunk_size output.tsv "
                << "delta_rel_max checkpoints_csv [ordered|unordered]\n"
                << "example checkpoints_csv: 900,950,1000\n";
            return 2;
        }

        const std::string trajPath = argv[1];
        const uint32_t dim = (uint32_t)std::stoul(argv[2]);
        const uint32_t K = (uint32_t)std::stoul(argv[3]);
        const size_t chunkSize = (size_t)std::stoull(argv[4]);
        const std::string outPath = argv[5];
        const BigFloat deltaThreshold(argv[6]);
        const uint32_t nSteps = 1000;
        const std::vector<uint32_t> checkpoints = parseCheckpoints(argv[7], nSteps);
        const bool ordered = (argc < 9) ? true : (std::string(argv[8]) != "unordered");

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Metal device");
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:readText("metal/rns_6f5.metal").c_str()];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) throw std::runtime_error([[err localizedDescription] UTF8String]);
        auto encPSO = pipeline(dev, lib, @"encode_trajectories_rns");
        auto walkPSO = pipeline(dev, lib, @"walk_6f5_rns");

        ProgramHost prog = load6F5ProgramOrThrow(dim, K);
        std::vector<uint32_t> primes = loadPrimesOrThrow(K);
        if (prog.m != 10 || prog.outReg.size() != 100)
            throw std::runtime_error("6F5 program must produce a complete 10x10 matrix");
        if (primes.size() != K) throw std::runtime_error("prime count mismatch");
        if (prog.constTable.size() % K != 0) throw std::runtime_error("constTable must be [K][nConst]");

        std::ifstream in(trajPath, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open trajectories file");
        std::ofstream out(outPath);
        if (!out) throw std::runtime_error("cannot open output file");
        writeHeader(out, checkpoints);

        RatioScanConfig scanCfg;
        scanCfg.ordered = ordered;
        scanCfg.maxRelativeDelta = deltaThreshold;
        scanCfg.emitOnlyConverged = true;

        uint64_t globalId = 0;
        std::vector<int32_t> shifts, dirs;
        while (size_t B = readChunk(in, dim, chunkSize, shifts, dirs)) {
            KernelConfigHost cfg{};
            cfg.nTraj = (uint32_t)B;
            cfg.dim = dim;
            cfg.m = prog.m;
            cfg.nSteps = nSteps;
            cfg.nInstr = (uint32_t)prog.instr.size();
            cfg.nConst = (uint32_t)(prog.constTable.size() / K);
            cfg.K = K;
            cfg.nCheckpoints = (uint32_t)checkpoints.size();
            for (size_t c = 0; c < checkpoints.size(); ++c) cfg.checkpointSteps[c] = checkpoints[c];

            const size_t E = (size_t)prog.m * prog.m;
            const size_t KD = (size_t)K * B * dim;
            const size_t KB = (size_t)K * B;
            const size_t CKB_E = checkpoints.size() * (size_t)K * B * E;

            id<MTLBuffer> bShift = [dev newBufferWithBytes:shifts.data() length:shifts.size()*sizeof(int32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bDir   = [dev newBufferWithBytes:dirs.data() length:dirs.size()*sizeof(int32_t) options:MTLResourceStorageModeShared];
            std::vector<PrimeMetaHost> pm(K); for (uint32_t k=0;k<K;++k) pm[k].p=primes[k];
            id<MTLBuffer> bPrime = [dev newBufferWithBytes:pm.data() length:pm.size()*sizeof(pm[0]) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCfg   = [dev newBufferWithBytes:&cfg length:sizeof(cfg) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSRNS  = [dev newBufferWithLength:KD*sizeof(uint32_t) options:MTLResourceStorageModePrivate];
            id<MTLBuffer> bDRNS  = [dev newBufferWithLength:KD*sizeof(uint32_t) options:MTLResourceStorageModePrivate];
            id<MTLBuffer> bInstr = [dev newBufferWithBytes:prog.instr.data() length:prog.instr.size()*sizeof(InstrHost) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bConst = [dev newBufferWithBytes:prog.constTable.data() length:prog.constTable.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bOutR  = [dev newBufferWithBytes:prog.outReg.data() length:prog.outReg.size()*sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSnap  = [dev newBufferWithLength:CKB_E*sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bAlive = [dev newBufferWithLength:KB*sizeof(uint8_t) options:MTLResourceStorageModeShared];
            if (!bSnap) throw std::runtime_error("snapshot buffer allocation failed; reduce chunk_size");

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:encPSO];
            [ce setBuffer:bShift offset:0 atIndex:0]; [ce setBuffer:bDir offset:0 atIndex:1];
            [ce setBuffer:bPrime offset:0 atIndex:2]; [ce setBuffer:bCfg offset:0 atIndex:3];
            [ce setBuffer:bSRNS offset:0 atIndex:4]; [ce setBuffer:bDRNS offset:0 atIndex:5];
            dispatch1D(ce, encPSO, KD); [ce endEncoding];

            ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:walkPSO];
            [ce setBuffer:bSRNS offset:0 atIndex:0]; [ce setBuffer:bDRNS offset:0 atIndex:1];
            [ce setBuffer:bInstr offset:0 atIndex:2]; [ce setBuffer:bConst offset:0 atIndex:3];
            [ce setBuffer:bOutR offset:0 atIndex:4]; [ce setBuffer:bPrime offset:0 atIndex:5];
            [ce setBuffer:bCfg offset:0 atIndex:6]; [ce setBuffer:bSnap offset:0 atIndex:7];
            [ce setBuffer:bAlive offset:0 atIndex:8];
            dispatch1D(ce, walkPSO, KB); [ce endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError)
                throw std::runtime_error([[cb.error localizedDescription] UTF8String]);

            auto *snap = reinterpret_cast<uint32_t*>(bSnap.contents);
            auto *alv = reinterpret_cast<uint8_t*>(bAlive.contents);
            std::vector<uint32_t> residues(K);

            for (size_t b = 0; b < B; ++b) {
                bool ok = true;
                for (uint32_t k=0;k<K;++k) {
                    if (!alv[(size_t)k*B+b]) { ok=false; break; }
                }
                if (!ok) continue;

                std::vector<std::vector<BigInt>> matrices(checkpoints.size(), std::vector<BigInt>(E));
                for (size_t c = 0; c < checkpoints.size(); ++c) {
                    for (size_t e = 0; e < E; ++e) {
                        for (uint32_t k=0;k<K;++k) {
                            size_t idx = ((((c * (size_t)K + k) * B + b) * E) + e);
                            residues[k] = snap[idx];
                        }
                        matrices[c][e] = centered(garner(residues.data(), primes), primes);
                    }
                }

                scan_all_matrix_ratios(globalId+b, prog.m, matrices, checkpoints, scanCfg, out);
            }
            globalId += B;
        }
    }
}
