// src/kernels/cpu/kq_avx512.cpp - Unsloth UD-Q5_K_XL's expert rows (Q5_K gate/up, Q8_0 down) for 1..8 tokens at once
// on AVX-512 (Zen 4: VNNI, BW, VL), BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built -march=native, so on an AVX-512 machine its vec_dot for these formats is still the __AVX2__ branch
// of ggml-cpu/arch/x86/quants.c (ggml_vec_dot_q5_K_q8_K, ggml_vec_dot_q8_0_q8_0), with AVX512-VNNI's vpdpbusd inside
// mul_sum_us8_pairs_float.  Its arithmetic splits into two parts:
//   * integers: per 256-value superblock (Q5_K) or 32-value block (Q8_0), eight int32 lanes; lane m sums the products
//     of bytes 4m..4m+3 of every 32-byte sub-block (times the sub-block's 6-bit scale for Q5_K), and the Q5_K mins
//     term, one int32.  Integer sums are exact, so any instruction sequence that forms the same lanes - here 64-byte
//     vpmaddubsw + vpdpwssd for Q5_K, vpdpbusd for Q8_0, two (sub-)blocks per zmm - gives the same lanes;
//   * floats, which are reproduced in ggml's order: per (super)block d = y.d * x.d, the lanes converted and FMA'd into
//     an 8-lane accumulator, the mins term FMA'd into a scalar (`summs`, contracted by GCC -O3), then ggml's
//     hsum_float_8 (+ summs).
// So each token's result is ggml's to the bit, for any number of tokens: the group size never changes an answer
// (native_expert_parity checks both).  What the kernel saves is ggml's per-token weight work (unpacking the 5-bit
// quants and the 6-bit scales, |x| and the sign of Q8_0), done once per row for all the tokens of a verify window, and
// half the instructions of the integer part (64-byte vectors).
#include "strata/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int MAXT = 8;

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}
// ggml's hsum_float_8, verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline int hsum_i32_8(const __m256i a) {
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i s2 = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(s2, _mm_shuffle_epi32(s2, _MM_SHUFFLE(2, 3, 0, 1))));
}

// ---- Q5_K x Q8_K, nt tokens of one row.  res[t] = ggml_vec_dot_q5_K_q8_K(row, act[t]).
// Per superblock: 4 zmm of 5-bit quants, zmm j = [sub-block 2j | sub-block 2j+1] (32 bytes each), and 4 zmm of
// int16 scales to match (sc[2j] in words 0..15, sc[2j+1] in words 16..31).
template <int NT>
void q5k_dot(const block_q5_K* x, int nb, const block_q8_K* const* y, float* res) {
    constexpr int nt = NT;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i m16 = _mm512_set1_epi8(0x10);
    // vpsrlvw counts: the low 256 bits keep the low nibbles, the high 256 bits shift the high ones down
    const __m512i nib_shift = _mm512_inserti64x4(_mm512_setzero_si512(), _mm256_set1_epi16(4), 1);
    // the fifth bit of sub-block 2j is bit 2j of qh, of sub-block 2j+1 bit 2j+1
    __m512i hbit[4];
    for (int j = 0; j < 4; ++j)
        hbit[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8((char) (1u << (2 * j)))),
                                     _mm256_set1_epi8((char) (1u << (2 * j + 1))), 1);
    // vpshufb controls on the 8 int16 scales broadcast to every 128-bit lane: word 2j in lanes 0-1, 2j+1 in 2-3
    __m512i sshuf[4];
    for (int j = 0; j < 4; ++j) {
        const int a = 2 * (2 * j), b = 2 * (2 * j + 1);
        const __m128i ca = _mm_set1_epi16((short) (a | ((a + 1) << 8)));
        const __m128i cb = _mm_set1_epi16((short) (b | ((b + 1) << 8)));
        sshuf[j] = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(ca), ca, 1), cb, 2), cb, 3);
    }
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        // utmp[0..1]: the 8 scales, utmp[2..3]: the 8 mins (bytes)
        const __m128i sc16 = _mm_cvtepu8_epi16(_mm_set_epi32(0, 0, (int) utmp[1], (int) utmp[0]));
        const __m512i sc_b = _mm512_broadcast_i32x4(sc16);
        // mins, each duplicated (one per bsum of 16): words [m0 m0 m1 m1 ... m7 m7]
        const __m128i mn8 = _mm_set_epi32(0, 0, (int) utmp[3], (int) utmp[2]);
        const __m256i mins_dup = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mn8, mn8));
        const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) x[i].qh));
        __m512i q5[4], scl[4];
        for (int j = 0; j < 4; ++j) {
            const __m512i bits = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j)));
            __m512i q = _mm512_and_si512(_mm512_srlv_epi16(bits, nib_shift), m4);
            const __mmask64 hk = _mm512_test_epi8_mask(qh, hbit[j]);
            q5[j] = _mm512_mask_add_epi8(q, hk, q, m16);
            scl[j] = _mm512_shuffle_epi8(sc_b, sshuf[j]);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* yb = y[t] + i;
            const float d = yb->d * xd;
            const float dmin = -yb->d * xdmin;
            const int8_t* q8 = yb->qs;
            __m512i s = _mm512_setzero_si512();
            for (int j = 0; j < 4; ++j) {
                const __m512i p = _mm512_maddubs_epi16(q5[j], _mm512_loadu_si512((const void*) (q8 + 64 * j)));
                s = _mm512_dpwssd_epi32(s, p, scl[j]);
            }
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64(s, 1));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
            const __m256i bs = _mm256_loadu_si256((const __m256i*) yb->bsums);
            const int m = hsum_i32_8(_mm256_madd_epi16(bs, mins_dup));
            summs[t] = std::fma(dmin, (float) m, summs[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0, nt tokens of one row.  res[t] = ggml_vec_dot_q8_0_q8_0(row, act[t]).
// Two blocks per zmm for the integers; the float chain stays one 8-lane FMA per block, in order.
template <int NT>
void q8_0_dot(const block_q8_0* x, int nb, const block_q8_0* const* y, float* res) {
    constexpr int nt = NT;
    __m256 acc[NT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 2 <= nb; ib += 2) {
        const float xd0 = h2f(x[ib].d), xd1 = h2f(x[ib + 1].d);
        const __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) x[ib].qs)),
                                              _mm256_loadu_si256((const __m256i*) x[ib + 1].qs), 1);
        const __m512i ax = _mm512_abs_epi8(qx);
        const __mmask64 neg = _mm512_movepi8_mask(qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* yb = y[t] + ib;
            const __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) yb[0].qs)),
                                                  _mm256_loadu_si256((const __m256i*) yb[1].qs), 1);
            // ggml's _mm256_sign_epi8(qy, qx): -qy where qx < 0, 0 where qx == 0 (ax is 0 there anyway)
            const __m512i sy = _mm512_mask_sub_epi8(qy, neg, _mm512_setzero_si512(), qy);
            const __m512 q = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ax, sy));
            const float d0 = xd0 * h2f(yb[0].d), d1 = xd1 * h2f(yb[1].d);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm512_castps512_ps256(q), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(q), 1)), acc[t]);
        }
    }
    for (; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* yb = y[t] + ib;
            const __m256i sy = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) yb->qs), qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd * h2f(yb->d)), q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// one row, m (1..8) tokens
void dot_row(int type, const uint8_t* row, int n, const void* const* a, int m, float* res) {
    if (type == 13) {
        const block_q5_K* x = (const block_q5_K*) row;
        const block_q8_K* const* y = (const block_q8_K* const*) a;
        const int nb = n / QK_K;
        switch (m) {
            case 1: q5k_dot<1>(x, nb, y, res); break;
            case 2: q5k_dot<2>(x, nb, y, res); break;
            case 3: q5k_dot<3>(x, nb, y, res); break;
            case 4: q5k_dot<4>(x, nb, y, res); break;
            case 5: q5k_dot<5>(x, nb, y, res); break;
            case 6: q5k_dot<6>(x, nb, y, res); break;
            case 7: q5k_dot<7>(x, nb, y, res); break;
            default: q5k_dot<8>(x, nb, y, res); break;
        }
    } else {
        const block_q8_0* x = (const block_q8_0*) row;
        const block_q8_0* const* y = (const block_q8_0* const*) a;
        const int nb = n / QK8_0;
        switch (m) {
            case 1: q8_0_dot<1>(x, nb, y, res); break;
            case 2: q8_0_dot<2>(x, nb, y, res); break;
            case 3: q8_0_dot<3>(x, nb, y, res); break;
            case 4: q8_0_dot<4>(x, nb, y, res); break;
            case 5: q8_0_dot<5>(x, nb, y, res); break;
            case 6: q8_0_dot<6>(x, nb, y, res); break;
            case 7: q8_0_dot<7>(x, nb, y, res); break;
            default: q8_0_dot<8>(x, nb, y, res); break;
        }
    }
}

}  // namespace

bool kq512_supported(int type) noexcept { return type == 13 || type == 8; }

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        const void* const* a = act + t0;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_row(type, blob + (size_t) r * gu_row, n, a, m, g);
            dot_row(type, blob + up_off + (size_t) r * gu_row, n, a, m, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        const void* const* a = act + t0;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_row(type, w + (size_t) r * row_bytes, n, a, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

}  // namespace strata::kernels::cpu
