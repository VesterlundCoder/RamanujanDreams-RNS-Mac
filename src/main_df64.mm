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

// 6F5 df64 walk host.
// Reads trajectories.bin (int32 shift[11] + int32 dir[11] per record),
// runs the double-single Metal kernel with per-step power-of-two
// normalization, and stores the normalized 6x6 product matrix at N=1000
// plus the accumulated exponent for every trajectory.
//
// Output binary layout:
//   header: char magic[8]="DF64MAT1", uint32 dim=6, uint32 nshift=11,
//           uint32 nSteps, int32 z_num, int32 z_den, uint64 nTraj
//   per trajectory: 36 x (float hi, float lo), int32 expSum
// Full matrix element value = (hi + lo) * 2^expSum.

static constexpr uint32_t DIM = 6;
static constexpr uint32_t NSHIFT = 11;
static constexpr uint32_t E = DIM * DIM;

struct Cfg {
    uint32_t nTraj;
    uint32_t nSteps;
    float zHi;
    float zLo;
};

static std::string readText(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            std::cerr << "usage: dreams_rns_df64 trajectories.bin output.bin "
                      << "[z_num=1] [z_den=1] [nSteps=1000] [chunk=262144]\n";
            return 2;
        }
        const std::string trajPath = argv[1];
        const std::string outPath = argv[2];
        const int32_t zNum = (argc > 3) ? (int32_t)std::stol(argv[3]) : 1;
        const int32_t zDen = (argc > 4) ? (int32_t)std::stol(argv[4]) : 1;
        const uint32_t nSteps = (argc > 5) ? (uint32_t)std::stoul(argv[5]) : 1000;
        const size_t chunk = (argc > 6) ? (size_t)std::stoull(argv[6]) : 262144;
        if (zDen == 0) throw std::runtime_error("z_den must be nonzero");

        const double z = (double)zNum / (double)zDen;
        const float zHi = (float)z;
        const float zLo = (float)(z - (double)zHi);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Metal device");
        std::cerr << "device: " << [[dev name] UTF8String] << "\n";
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:readText("metal/walk_6f5_df64.metal").c_str()];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; // exact IEEE fma required for df64
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) throw std::runtime_error([[err localizedDescription] UTF8String]);
        id<MTLFunction> fn = [lib newFunctionWithName:@"walk_6f5_df64"];
        if (!fn) throw std::runtime_error("missing kernel");
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) throw std::runtime_error([[err localizedDescription] UTF8String]);

        std::ifstream in(trajPath, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open trajectories file");
        std::ofstream out(outPath, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open output file");

        // header (nTraj patched at the end)
        const char magic[8] = {'D','F','6','4','M','A','T','1'};
        uint64_t nTrajTotal = 0;
        out.write(magic, 8);
        uint32_t hdr32[3] = {DIM, NSHIFT, nSteps};
        int32_t hz[2] = {zNum, zDen};
        out.write((const char*)hdr32, sizeof(hdr32));
        out.write((const char*)hz, sizeof(hz));
        out.write((const char*)&nTrajTotal, sizeof(nTrajTotal));

        std::vector<int32_t> shifts(chunk * NSHIFT), dirs(chunk * NSHIFT);
        std::vector<int32_t> rec(2 * NSHIFT);

        auto t0 = std::chrono::steady_clock::now();
        double gpuSeconds = 0.0;

        for (;;) {
            size_t got = 0;
            while (got < chunk &&
                   in.read((char*)rec.data(), rec.size() * sizeof(int32_t))) {
                std::memcpy(&shifts[got * NSHIFT], rec.data(), NSHIFT * sizeof(int32_t));
                std::memcpy(&dirs[got * NSHIFT], rec.data() + NSHIFT, NSHIFT * sizeof(int32_t));
                ++got;
            }
            if (got == 0) break;

            Cfg cfg{(uint32_t)got, nSteps, zHi, zLo};

            id<MTLBuffer> bShift = [dev newBufferWithBytes:shifts.data()
                length:got * NSHIFT * sizeof(int32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bDir = [dev newBufferWithBytes:dirs.data()
                length:got * NSHIFT * sizeof(int32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bMat = [dev newBufferWithLength:got * E * 2 * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bExp = [dev newBufferWithLength:got * sizeof(int32_t)
                options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pso];
            [ce setBuffer:bShift offset:0 atIndex:0];
            [ce setBuffer:bDir offset:0 atIndex:1];
            [ce setBytes:&cfg length:sizeof(cfg) atIndex:2];
            [ce setBuffer:bMat offset:0 atIndex:3];
            [ce setBuffer:bExp offset:0 atIndex:4];
            NSUInteger w = pso.threadExecutionWidth;
            NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                                 std::max<NSUInteger>(w, 256));
            tg = (tg / w) * w;
            [ce dispatchThreads:MTLSizeMake(got, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [ce endEncoding];
            auto g0 = std::chrono::steady_clock::now();
            [cb commit];
            [cb waitUntilCompleted];
            auto g1 = std::chrono::steady_clock::now();
            gpuSeconds += std::chrono::duration<double>(g1 - g0).count();
            if (cb.status == MTLCommandBufferStatusError)
                throw std::runtime_error([[cb.error localizedDescription] UTF8String]);

            const float *mat = (const float*)bMat.contents;
            const int32_t *ex = (const int32_t*)bExp.contents;
            for (size_t b = 0; b < got; ++b) {
                out.write((const char*)(mat + b * E * 2), E * 2 * sizeof(float));
                out.write((const char*)(ex + b), sizeof(int32_t));
            }
            nTrajTotal += got;
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            std::cerr << "done " << nTrajTotal << " traj, "
                      << (double)nTrajTotal / el << " traj/s (wall), "
                      << (double)nTrajTotal / gpuSeconds << " traj/s (gpu)\n";
        }

        // patch nTraj into header
        out.seekp(8 + sizeof(uint32_t) * 3 + sizeof(int32_t) * 2);
        out.write((const char*)&nTrajTotal, sizeof(nTrajTotal));
        out.close();

        auto t1 = std::chrono::steady_clock::now();
        double total = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "TOTAL " << nTrajTotal << " trajectories in " << total
                  << " s wall (" << gpuSeconds << " s gpu) => "
                  << (double)nTrajTotal / total << " traj/s\n";
    }
    return 0;
}
