// src/kernels/cuda/q8_wave32.cuh - gfx10+ HIP (wave32): one wave's R rows of a Q8_0 x q8_1 product for NC columns, bitwise
// native_small_mmvq_kernel / native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, 4, R> (native_mmvq.cu, the EXACT
// layouts).  Those kernels give a block's four warps the four virtual threads (w, l) of each lane l (Q8_0 blocks
// 8 w + l / 4 + 32 i, ints 2 (l % 4) and + 1), each summing its blocks in order; warp 0 then adds warp 1, 2 and 3's sums
// through shared memory and runs the xor tree.  Here one wave carries the four virtual threads of a lane in registers
// and adds them in that order (warp 0 + warp 1 + warp 2 + warp 3), then the same xor tree on DPP: no shared memory, no
// barrier, four times the loads in flight per wave.  The weight ints are single dword loads at 2-byte alignment
// (unaligned global access is on for gfx10+ under ROCm).
#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include "strata/kernels/dp4a.hpp"

namespace strata::kernels::q8w {
struct Q80 { half d; int8_t qs[32]; };
struct Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q80) == 34 && sizeof(Q81) == 36, "GGUF Q8_0 / ggml q8_1 blocks");

template <int O> __device__ __forceinline__ float xmov(float v) {
    if constexpr (O == 16)
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u,
                                                           false, false));
    else return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + O, 0xf, 0xf, false));
}
/// `for (o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o)` on DPP / permlanex16: the same pairs, the same sums
__device__ __forceinline__ float warp_sum(float v) {
    v += xmov<16>(v); v += xmov<8>(v); v += xmov<4>(v); v += xmov<2>(v); v += xmov<1>(v);
    return v;
}
/// rows [row0, row0 + R) (clamped to n_out) of y[j * n_out + row] for the NC columns of x (q8_1, n_in / 32 blocks each)
template <int NC, int R>
__device__ __forceinline__ void rows(const Q80* __restrict__ w, const Q81* __restrict__ x, float* __restrict__ y, int n_in,
                                     int n_out, int row0, int lane) {
    const int bpr = n_in / 32;
    const int kqs = 2 * (lane & 3);
    float tmp[4][NC][R];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) tmp[v][j][i] = 0.0f;
    for (int base = lane >> 2; base < bpr; base += 32) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int kbx = base + 8 * v;
            if (kbx < bpr) {
                int wq0[R], wq1[R];
                float wd[R];
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const int row = row0 + i < n_out ? row0 + i : row0;
                    const Q80* b = w + std::size_t(row) * bpr + kbx;
                    wq0[i] = *reinterpret_cast<const int*>(b->qs + 4 * kqs);
                    wq1[i] = *reinterpret_cast<const int*>(b->qs + 4 * kqs + 4);
                    wd[i] = b->d;
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const Q81* xb = x + std::size_t(j) * bpr + kbx;
                    const int u0 = reinterpret_cast<const int*>(xb->qs)[kqs];
                    const int u1 = reinterpret_cast<const int*>(xb->qs)[kqs + 1];
                    const float d1 = __low2float(xb->ds);
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        int sumi = STRATA_DP4A(wq0[i], u0, 0);
                        sumi = STRATA_DP4A(wq1[i], u1, sumi);
                        tmp[v][j][i] += wd[i] * d1 * float(sumi);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float sum = tmp[0][j][i];
            sum += tmp[1][j][i];
            sum += tmp[2][j][i];
            sum += tmp[3][j][i];
            sum = warp_sum(sum);
            if (lane == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        }
}
/// whether the wave32 kernels run (STRATA_MMVQ_Q8W=0: the old kernels)
bool enabled();
}  // namespace strata::kernels::q8w
#endif
