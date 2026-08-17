#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// MeijerG(4,2,4,4,1) CMF df64 walk host.
//
// Reads an export directory produced by
//   stringtheory/meijer_search/rns/df64_walk.py --export DIR
// containing config.json, coeffMant.bin (float32[B][E][deg1][2]) and
// coeffExp.bin (int32[B][E][deg1]), dispatches walk_meijer_df64 and writes
// for every checkpoint c and candidate b:
//   float2 mat[E], int32 expRel[E], int32 growth
// (the layout verified by df64_walk.py --check-gpu DIR out.bin).
//
// usage: dreams_meijer_df64 EXPORT_DIR KERNEL.metal OUTPUT.bin

static constexpr uint32_t MAX_CHECKPOINTS = 8;

struct Cfg {
    uint32_t nTraj;
    uint32_t m;
    uint32_t deg1;
    uint32_t nSteps;
    uint32_t nCheckpoints;
    uint32_t checkpointSteps[MAX_CHECKPOINTS];
};

static std::string readText(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<uint8_t> readBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::vector<uint8_t> buf((size_t)f.tellg());
    f.seekg(0);
    f.read((char *)buf.data(), buf.size());
    return buf;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) {
            std::cerr << "usage: dreams_meijer_df64 EXPORT_DIR KERNEL.metal OUTPUT.bin\n";
            return 2;
        }
        const std::string dir = argv[1];
        const std::string kernelPath = argv[2];
        const std::string outPath = argv[3];

        // ---- config.json
        NSString *cfgPath = [NSString stringWithUTF8String:(dir + "/config.json").c_str()];
        NSData *cfgData = [NSData dataWithContentsOfFile:cfgPath];
        if (!cfgData) throw std::runtime_error("cannot read config.json");
        NSError *err = nil;
        NSDictionary *j = [NSJSONSerialization JSONObjectWithData:cfgData options:0 error:&err];
        if (!j) throw std::runtime_error([[err localizedDescription] UTF8String]);

        Cfg cfg{};
        cfg.nTraj = [j[@"nTraj"] unsignedIntValue];
        cfg.m = [j[@"m"] unsignedIntValue];
        cfg.deg1 = [j[@"deg1"] unsignedIntValue];
        cfg.nSteps = [j[@"nSteps"] unsignedIntValue];
        NSArray *cps = j[@"checkpointSteps"];
        cfg.nCheckpoints = (uint32_t)[cps count];
        if (cfg.nCheckpoints > MAX_CHECKPOINTS)
            throw std::runtime_error("too many checkpoints");
        for (uint32_t i = 0; i < cfg.nCheckpoints; ++i)
            cfg.checkpointSteps[i] = [cps[i] unsignedIntValue];

        const uint32_t B = cfg.nTraj;
        const uint32_t E = cfg.m * cfg.m;
        const uint32_t C = cfg.nCheckpoints;

        std::vector<uint8_t> mant = readBin(dir + "/coeffMant.bin");
        std::vector<uint8_t> exps = readBin(dir + "/coeffExp.bin");
        if (mant.size() != (size_t)B * E * cfg.deg1 * 2 * sizeof(float))
            throw std::runtime_error("coeffMant.bin size mismatch");
        if (exps.size() != (size_t)B * E * cfg.deg1 * sizeof(int32_t))
            throw std::runtime_error("coeffExp.bin size mismatch");

        // ---- Metal setup
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Metal device");
        std::cerr << "device: " << [[dev name] UTF8String] << "\n";
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        NSString *src = [NSString stringWithUTF8String:readText(kernelPath).c_str()];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;  // exact IEEE fma required for df64
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) throw std::runtime_error([[err localizedDescription] UTF8String]);
        id<MTLFunction> fn = [lib newFunctionWithName:@"walk_meijer_df64"];
        if (!fn) throw std::runtime_error("missing kernel walk_meijer_df64");
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) throw std::runtime_error([[err localizedDescription] UTF8String]);

        id<MTLBuffer> bMant = [dev newBufferWithBytes:mant.data()
            length:mant.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bExp = [dev newBufferWithBytes:exps.data()
            length:exps.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOutMat = [dev newBufferWithLength:(size_t)C * B * E * 2 * sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOutExp = [dev newBufferWithLength:(size_t)C * B * E * sizeof(int32_t)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bGrowth = [dev newBufferWithLength:(size_t)C * B * sizeof(int32_t)
            options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:bMant offset:0 atIndex:0];
        [ce setBuffer:bExp offset:0 atIndex:1];
        [ce setBytes:&cfg length:sizeof(cfg) atIndex:2];
        [ce setBuffer:bOutMat offset:0 atIndex:3];
        [ce setBuffer:bOutExp offset:0 atIndex:4];
        [ce setBuffer:bGrowth offset:0 atIndex:5];
        NSUInteger w = pso.threadExecutionWidth;
        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                             std::max<NSUInteger>(w, 256));
        tg = (tg / w) * w;
        [ce dispatchThreads:MTLSizeMake(B, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(tg, B), 1, 1)];
        [ce endEncoding];

        auto g0 = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        auto g1 = std::chrono::steady_clock::now();
        if (cb.status == MTLCommandBufferStatusError)
            throw std::runtime_error([[cb.error localizedDescription] UTF8String]);
        double gpuSeconds = std::chrono::duration<double>(g1 - g0).count();

        // ---- output: per (checkpoint, candidate): mat[E] f2, expRel[E] i32, growth i32
        std::ofstream out(outPath, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open output");
        const float *om = (const float *)bOutMat.contents;
        const int32_t *oe = (const int32_t *)bOutExp.contents;
        const int32_t *og = (const int32_t *)bGrowth.contents;
        for (uint32_t c = 0; c < C; ++c)
            for (uint32_t b = 0; b < B; ++b) {
                size_t base = ((size_t)c * B + b) * E;
                out.write((const char *)(om + base * 2), E * 2 * sizeof(float));
                out.write((const char *)(oe + base), E * sizeof(int32_t));
                out.write((const char *)(og + (size_t)c * B + b), sizeof(int32_t));
            }
        out.close();

        std::cerr << "walked " << B << " candidates x " << cfg.nSteps
                  << " steps (deg1=" << cfg.deg1 << ") in " << gpuSeconds
                  << " s gpu => " << (double)B / gpuSeconds << " cand/s\n";
    }
    return 0;
}
