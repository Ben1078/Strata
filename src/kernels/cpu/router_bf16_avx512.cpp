// src/kernels/cpu/router_bf16_avx512.cpp - q5opt r6-mem: the router lookahead's BF16 dot on AVX512-BF16 (Zen 4:
// vdpbf16ps, 32 products per instruction, no widening of the weights).  The activations are rounded to BF16 once per
// call (round-to-nearest-even), so the logits differ in the last bits from the AVX2 path - this only PREDICTS which
// experts to prefetch, it computes nothing the model uses.
#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels::cpu {

bool cpu_avx512bf16_ok() {
#if defined(__GNUC__) || defined(__clang__)
    static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bf16");
    return ok;
#else
    return false;
#endif
}

void bf16_rows_dot_multi_avx512bf16(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    static thread_local std::vector<__m512bh> xb;   // the activations as BF16, 32 per vector, nt rows
    const int nv = cols / 32;
    xb.resize((size_t) nt * (size_t) nv);
    for (int t = 0; t < nt; ++t)
        for (int v = 0; v < nv; ++v) {
            const float* p = x + (size_t) t * cols + (size_t) v * 32;
            xb[(size_t) t * nv + v] = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
        }
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m512 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm512_setzero_ps();
        for (int v = 0; v < nv; ++v) {
            const __m512bh wv = (__m512bh) _mm512_loadu_si512((const void*) (wr + (size_t) v * 32));
            for (int t = 0; t < nt; ++t) acc[t] = _mm512_dpbf16_ps(acc[t], wv, xb[(size_t) t * nv + v]);
        }
        for (int t = 0; t < nt; ++t) out[(size_t) t * rows + r] = _mm512_reduce_add_ps(acc[t]);
    }
}

}  // namespace strata::kernels::cpu
