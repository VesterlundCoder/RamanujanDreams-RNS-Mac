#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host for metal/walk_6f5_fixed_start_df64.metal.
//
// Input directions.bin: repeated int32 direction[11] records. The fixed
// absolute CMF start point is supplied ONCE on the command line.
//
// Output FSDF6401:
//   header:
//     char magic[8] = "FSDF6401"
//     u32 max_dim, rank, nshift, nSteps, checkpoint0, checkpoint1,
//         rowNum, rowDen
//     i32 z_num, z_den
//     i32 absolute_start[11]
//     u64 nTraj
//   record:
//     4 x (float hi, float lo) in order
//       [cp0 numerator, cp0 denominator, cp1 numerator, cp1 denominator]
//     i32 exp0, exp1
//     u32 status
//
// The two rows are exactly the selected rows of the full CMF product.
// Their last-active-column ratio is therefore the same ratio that the
// full-matrix census would produce.

static constexpr uint32_t MAX_DIM = 6;
static constexpr uint32_t NSHIFT = 11;
static constexpr uint32_t NCHECK = 2;
static constexpr uint32_t NROWS = 2;

struct Cfg {
    uint32_t nTraj;
    uint32_t nSteps;
    uint32_t rank;
    uint32_t rowNum;
    uint32_t rowDen;
    uint32_t checkpointSteps[NCHECK];
    int32_t start[NSHIFT];
    float zHi;
    float zLo;
};

static std::string readText(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

static std::array<int32_t, NSHIFT> parseStart(const std::string &s) {
    std::array<int32_t, NSHIFT> out{};
    std::stringstream ss(s);
    std::string tok;
    uint32_t i = 0;
    while (std::getline(ss, tok, ',')) {
        if (i >= NSHIFT) throw std::runtime_error("absolute_start must have exactly 11 integers");
        out[i++] = (int32_t)std::stol(tok);
    }
    if (i != NSHIFT) throw std::runtime_error("absolute_start must have exactly 11 integers");
    return out;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 6) {
            std::cerr
                << "usage: dreams_fixed_start_df64 directions.bin output.bin "
                << "absolute_start_csv row_num row_den "
                << "[z_num=1] [z_den=2] [nSteps=1000] [chunk=262144]\n\n"
                << "IMPORTANT: nSteps is the number of trajectory operators. "
                << "The first operator is evaluated at n=0, i.e. exactly at "
                << "absolute_start. Do NOT pass nSteps=0 to mean start at N=0.\n";
            return 2;
        }

        const std::string dirPath = argv[1];
        const std::string outPath = argv[2];
        const auto start = parseStart(argv[3]);
        const uint32_t rowNum = (uint32_t)std::stoul(argv[4]);
        const uint32_t rowDen = (uint32_t)std::stoul(argv[5]);
        const int32_t zNum = (argc > 6) ? (int32_t)std::stol(argv[6]) : 1;
        const int32_t zDen = (argc > 7) ? (int32_t)std::stol(argv[7]) : 2;
        const uint32_t nSteps = (argc > 8) ? (uint32_t)std::stoul(argv[8]) : 1000;
        const size_t chunk = (argc > 9) ? (size_t)std::stoull(argv[9]) : 262144;

        if (zDen == 0) throw std::runtime_error("z_den must be nonzero");
        if (nSteps == 0) {
            throw std::runtime_error(
                "nSteps=0 means zero matrix multiplications and yields identity. "
                "For N=0 start semantics, keep nSteps>=1: the first factor is already evaluated at n=0."
            );
        }

        const uint32_t rank = (zNum == zDen) ? 5u : 6u;
        if (rowNum >= rank || rowDen >= rank || rowNum == rowDen)
            throw std::runtime_error("row_num and row_den must be distinct active rows < rank");

        const uint32_t cp0 = std::max<uint32_t>(1u, nSteps / 2u);
        const uint32_t cp1 = nSteps;

        const double z = (double)zNum / (double)zDen;
        const float zHi = (float)z;
        const float zLo = (float)(z - (double)zHi);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Metal device");
        std::cerr << "device: " << [[dev name] UTF8String]
                  << "  fixed-start rank=" << rank
                  << "  pair=(" << rowNum << "," << rowDen << ")"
                  << "  checkpoints=" << cp0 << "," << cp1 << "\n";
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:
            readText("metal/walk_6f5_fixed_start_df64.metal").c_str()];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        // Backward-compatible precise mode. On newer SDKs this property is
        // deprecated in favor of mathMode=MTLMathModeSafe, but fastMathEnabled
        // is available on the Xcode 15 SDK as well. df64 requires precise FMA.
        opts.fastMathEnabled = NO;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) throw std::runtime_error([[err localizedDescription] UTF8String]);
        id<MTLFunction> fn = [lib newFunctionWithName:@"walk_6f5_fixed_start_df64"];
        if (!fn) throw std::runtime_error("missing fixed-start Metal kernel");
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) throw std::runtime_error([[err localizedDescription] UTF8String]);

        std::ifstream in(dirPath, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open directions file");
        std::ofstream out(outPath, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open output file");

        const char magic[8] = {'F','S','D','F','6','4','0','1'};
        uint64_t nTrajTotal = 0;
        out.write(magic, 8);
        uint32_t hdr32[8] = {MAX_DIM, rank, NSHIFT, nSteps,
                             cp0, cp1, rowNum, rowDen};
        int32_t hz[2] = {zNum, zDen};
        out.write((const char*)hdr32, sizeof(hdr32));
        out.write((const char*)hz, sizeof(hz));
        out.write((const char*)start.data(), sizeof(int32_t) * NSHIFT);
        const std::streampos nTrajPos = out.tellp();
        out.write((const char*)&nTrajTotal, sizeof(nTrajTotal));

        std::vector<int32_t> dirs(chunk * NSHIFT);
        auto t0 = std::chrono::steady_clock::now();
        double gpuSeconds = 0.0;
        uint64_t nOk = 0;

        for (;;) {
            size_t got = 0;
            while (got < chunk &&
                   in.read((char*)(&dirs[got * NSHIFT]), NSHIFT * sizeof(int32_t))) {
                ++got;
            }
            if (got == 0) break;

            Cfg cfg{};
            cfg.nTraj = (uint32_t)got;
            cfg.nSteps = nSteps;
            cfg.rank = rank;
            cfg.rowNum = rowNum;
            cfg.rowDen = rowDen;
            cfg.checkpointSteps[0] = cp0;
            cfg.checkpointSteps[1] = cp1;
            for (uint32_t d = 0; d < NSHIFT; ++d) cfg.start[d] = start[d];
            cfg.zHi = zHi;
            cfg.zLo = zLo;

            id<MTLBuffer> bDir = [dev newBufferWithBytes:dirs.data()
                length:got * NSHIFT * sizeof(int32_t)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bVals = [dev newBufferWithLength:
                got * NCHECK * NROWS * 2 * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bExp = [dev newBufferWithLength:
                got * NCHECK * sizeof(int32_t)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bStatus = [dev newBufferWithLength:
                got * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            if (!bDir || !bVals || !bExp || !bStatus)
                throw std::runtime_error("Metal buffer allocation failed; reduce chunk size");

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pso];
            [ce setBuffer:bDir offset:0 atIndex:0];
            [ce setBytes:&cfg length:sizeof(cfg) atIndex:1];
            [ce setBuffer:bVals offset:0 atIndex:2];
            [ce setBuffer:bExp offset:0 atIndex:3];
            [ce setBuffer:bStatus offset:0 atIndex:4];

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

            const float *vals = (const float*)bVals.contents;
            const int32_t *ex = (const int32_t*)bExp.contents;
            const uint32_t *st = (const uint32_t*)bStatus.contents;
            for (size_t b = 0; b < got; ++b) {
                // 4 float2 values = 8 float32 values.
                out.write((const char*)(vals + b * NCHECK * NROWS * 2),
                          NCHECK * NROWS * 2 * sizeof(float));
                out.write((const char*)(ex + b * NCHECK),
                          NCHECK * sizeof(int32_t));
                out.write((const char*)(st + b), sizeof(uint32_t));
                if (st[b] == 0) ++nOk;
            }

            nTrajTotal += got;
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            std::cerr << "done " << nTrajTotal << " dirs (" << nOk << " OK), "
                      << (double)nTrajTotal / el << " dir/s wall, "
                      << (double)nTrajTotal / gpuSeconds << " dir/s gpu\n";
        }

        out.seekp(nTrajPos);
        out.write((const char*)&nTrajTotal, sizeof(nTrajTotal));
        out.close();

        double total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cerr << "TOTAL " << nTrajTotal << " directions (" << nOk
                  << " OK, " << (nTrajTotal - nOk) << " flagged) in "
                  << total << " s => " << (double)nTrajTotal / total
                  << " dir/s\n";
    }
    return 0;
}
