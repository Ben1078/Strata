// src/kernels/cpu/kq512_parity.cpp - the AVX-512 Q5_K / Q8_0 multi-token rows (kq_avx512.cpp) BIT FOR BIT against
// ggml-cpu's vec_dot, every token of 1..8-token groups, on ggml-quantized random rows and activations of several
// shapes (normal, heavy-tailed, tiny, a zero block).  --bench: one thread's time, hot (L2) and streamed (DRAM).
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

std::vector<uint8_t> quant_rows(int type, int rows, int cols, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::student_t_distribution<float> td(3.f);
    std::vector<float> w((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) {
        const float scale = 0.02f * (0.25f + (float) (r % 11) / 5.0f);
        for (int c = 0; c < cols; ++c) w[(size_t) r * cols + c] = scale * (r % 3 == 0 ? td(rng) : nd(rng));
    }
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, cols) * rows);
    ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

// activation t of `cols` floats, quantized to the type's vec_dot_type
std::vector<uint8_t> quant_act(int type, int cols, int t, int seed) {
    std::mt19937 rng(seed * 131 + t);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::student_t_distribution<float> td(2.f);
    std::vector<float> x((size_t) cols);
    const int shape = t % 4;
    for (int c = 0; c < cols; ++c) {
        float v = shape == 0 ? nd(rng) : shape == 1 ? td(rng) * 3.f : shape == 2 ? 1e-3f * nd(rng) : nd(rng) * (c % 97 == 0 ? 40.f : 1.f);
        if (shape == 3 && c < 256) v = 0.f;   // a zero (super)block
        x[(size_t) c] = v;
    }
    const ggml_type at = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot_type;
    std::vector<uint8_t> q(ggml_row_size(at, cols) + 64);
    ggml_get_type_traits_cpu(at)->from_float(x.data(), q.data(), cols);
    return q;
}

int check(int type, int cols, int rows, int seed) {
    const auto w = quant_rows(type, rows, cols, seed);
    const size_t rb = ggml_row_size((ggml_type) type, cols);
    const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
    std::vector<std::vector<uint8_t>> acts;
    const void* ap[8];
    for (int t = 0; t < 8; ++t) acts.push_back(quant_act(type, cols, t, seed));
    for (int t = 0; t < 8; ++t) ap[t] = acts[(size_t) t].data();
    std::vector<float> ref((size_t) 8 * rows);
    for (int t = 0; t < 8; ++t)
        for (int r = 0; r < rows; ++r) tc->vec_dot(cols, &ref[(size_t) t * rows + r], 0, w.data() + r * rb, 0, ap[t], 0, 1);
    int bad = 0;
    for (int nt = 1; nt <= 8; ++nt)
        for (int t0 = 0; t0 + nt <= 8; t0 += nt) {
            std::vector<float> got((size_t) nt * rows);
            float* op[8];
            for (int t = 0; t < nt; ++t) op[t] = got.data() + (size_t) t * rows;
            cpu::kq512_rows(type, w.data(), rb, cols, ap + t0, nt, op, 0, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r)
                    if (std::memcmp(&got[(size_t) t * rows + r], &ref[(size_t) (t0 + t) * rows + r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  %s n %d nt %d token %d row %d: %.9g vs ggml %.9g\n", ggml_type_name((ggml_type) type),
                                        cols, nt, t0 + t, r, got[(size_t) t * rows + r], ref[(size_t) (t0 + t) * rows + r]);
                        ++bad;
                    }
        }
    // gate/up with SwiGLU, against native_gu_rows' ggml loop
    if (type == 13 || type == 8) {
        const auto u = quant_rows(type, rows, cols, seed + 7);
        std::vector<uint8_t> blob(w.size() * 2);
        std::memcpy(blob.data(), w.data(), w.size());
        std::memcpy(blob.data() + w.size(), u.data(), u.size());
        for (int nt = 1; nt <= 8; ++nt) {
            std::vector<float> got((size_t) nt * rows);
            float* fp[8];
            for (int t = 0; t < nt; ++t) fp[t] = got.data() + (size_t) t * rows;
            cpu::kq512_gu_rows(type, blob.data(), rb, w.size(), cols, ap, nt, fp, 0, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) {
                    float g = 0.f, uu = 0.f;
                    tc->vec_dot(cols, &g, 0, blob.data() + r * rb, 0, ap[t], 0, 1);
                    tc->vec_dot(cols, &uu, 0, blob.data() + w.size() + r * rb, 0, ap[t], 0, 1);
                    const float e = (g / (1.f + std::exp(-g))) * uu;
                    if (std::memcmp(&e, &got[(size_t) t * rows + r], 4) != 0) {
                        if (bad < 5) std::printf("  gu %s nt %d token %d row %d differs\n", ggml_type_name((ggml_type) type), nt, t, r);
                        ++bad;
                    }
                }
        }
    }
    std::printf("kq512 %s n %d: %d rows x 8 tokens x every group size: %s (%d differ)\n", ggml_type_name((ggml_type) type),
                cols, rows, bad ? "DIFFERS" : "bitwise equal to ggml", bad);
    return bad ? 1 : 0;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void bench(int type, int cols, int rows) {
    const size_t rb = ggml_row_size((ggml_type) type, cols);
    const size_t mat = rb * rows;
    const auto w = quant_rows(type, rows, cols, 3);
    // cold: 1 GiB of copies, THP
    const int nmat = (int) (((size_t) 1 << 30) / mat);
    uint8_t* big = (uint8_t*) mmap(nullptr, (size_t) nmat * mat, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(big, (size_t) nmat * mat, MADV_HUGEPAGE);
    for (int i = 0; i < nmat; ++i) std::memcpy(big + (size_t) i * mat, w.data(), mat);
    const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
    std::vector<std::vector<uint8_t>> acts;
    const void* ap[8];
    for (int t = 0; t < 8; ++t) acts.push_back(quant_act(type, cols, t, 9));
    for (int t = 0; t < 8; ++t) ap[t] = acts[(size_t) t].data();
    std::vector<float> out((size_t) 8 * rows);
    float* op[8];
    for (int t = 0; t < 8; ++t) op[t] = out.data() + (size_t) t * rows;
    const int hot_rows = (int) std::min<size_t>(rows, (256 << 10) / rb);   // ~256 KB: L2
    for (int nt = 1; nt <= 8; nt = nt < 4 ? nt + 1 : nt * 2) {
        auto ggml_run = [&](const uint8_t* m, int nr) {
            for (int r = 0; r < nr; ++r)
                for (int t = 0; t < nt; ++t) tc->vec_dot(cols, op[t] + r, 0, m + r * rb, 0, ap[t], 0, 1);
        };
        auto kq_run = [&](const uint8_t* m, int nr) { cpu::kq512_rows(type, m, rb, cols, ap, nt, op, 0, nr); };
        double hot[2] = {1e9, 1e9}, cold[2] = {0, 0};
        for (int k = 0; k < 2; ++k) {
            for (int it = 0; it < 200; ++it) {
                const double a = now_s();
                k ? kq_run(w.data(), hot_rows) : ggml_run(w.data(), hot_rows);
                hot[k] = std::min(hot[k], now_s() - a);
            }
            const double a = now_s();
            for (int i = 0; i < nmat; ++i) k ? kq_run(big + (size_t) i * mat, rows) : ggml_run(big + (size_t) i * mat, rows);
            cold[k] = (now_s() - a) / nmat;
        }
        const double hb = (double) hot_rows * rb, cb = (double) mat;
        std::printf("bench %s n %d nt %d one thread | L2: ggml %5.1f GB/s  kq512 %5.1f GB/s (%.2fx) | DRAM: ggml %5.1f  kq512 %5.1f GB/s (%.2fx)\n",
                    ggml_type_name((ggml_type) type), cols, nt, hb / hot[0] / 1e9, hb / hot[1] / 1e9, hot[0] / hot[1],
                    cb / cold[0] / 1e9, cb / cold[1] / 1e9, cold[0] / cold[1]);
    }
    munmap(big, (size_t) nmat * mat);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (!cpu::cpu_avx512_ok()) {
        std::printf("kq512_parity: no AVX-512 on this CPU, nothing to test\n");
        return 0;
    }
    ggml_cpu_init();
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    int fail = 0;
    if (!do_bench) {
        fail += check(13, 2560, 640, 1);   // Q5_K gate/up rows
        fail += check(13, 256, 64, 2);     // one superblock
        fail += check(8, 640, 2560, 3);    // Q8_0 down rows (20 blocks)
        fail += check(8, 32 * 21, 64, 4);  // an odd block count: the 256-bit tail
        fail += check(8, 2560, 640, 5);    // Q8_0 gate/up
    } else {
        bench(13, 2560, 640);
        bench(8, 640, 2560);
    }
    std::printf("kq512_parity: %s\n", fail ? "FAILED" : "ok");
    return fail ? 1 : 0;
}
