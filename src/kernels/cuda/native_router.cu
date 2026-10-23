// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
// The xor butterflies of the top-10 below.  On gfx10+ HIP the cross-lane moves are DPP row_xmask / permlanex16 (an ALU
// operand) instead of ds_bpermute (an LDS round trip per move): the same lane pairs, so the same values compared and
// added in the same order - bitwise the shuffle kernel.  The 10 rounds of 5 dependent moves made the window's top-10
// a 12.7 us single-block kernel on the RX 7900 XTX.
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
template <int O> __device__ __forceinline__ int xmov_i(int v) {
    if constexpr (O == 16) return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
    else return __builtin_amdgcn_update_dpp(0, v, 0x160 + O, 0xf, 0xf, false);
}
#else
template <int O> __device__ __forceinline__ int xmov_i(int v) { return __shfl_xor_sync(0xffffffffu, v, O, 32); }
#endif
template <int O> __device__ __forceinline__ float xmov_f(float v) { return __int_as_float(xmov_i<O>(__float_as_int(v))); }
__device__ __forceinline__ float warp_sum(float value) {
    value += xmov_f<16>(value); value += xmov_f<8>(value); value += xmov_f<4>(value);
    value += xmov_f<2>(value); value += xmov_f<1>(value);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
    value = fmaxf(value, xmov_f<16>(value)); value = fmaxf(value, xmov_f<8>(value)); value = fmaxf(value, xmov_f<4>(value));
    value = fmaxf(value, xmov_f<2>(value)); value = fmaxf(value, xmov_f<1>(value));
    return value;
}
// one level of the (probability, expert) arg-max butterfly: the larger probability, the lower expert on a tie
template <int O> __device__ __forceinline__ void best_step(float& best, int& expert) {
    const float other = xmov_f<O>(best);
    const int other_id = xmov_i<O>(expert);
    if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
}
// one token's top-10 (the body of `route` / `route_multi`); `s_ids` (optional) also gets the ids
__device__ __forceinline__ void route_token(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                            float* __restrict__ weights, int lane, int32_t* s_ids) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    // (not unrolled: in the engine the kernel starts with a cold instruction cache, and the straight-line code it would
    // be costs more there than the loop)
#pragma unroll 1
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
        best_step<16>(best, expert); best_step<8>(best, expert); best_step<4>(best, expert);
        best_step<2>(best, expert); best_step<1>(best, expert);
        if ((expert & 31) == lane) {
            const int slot = expert >> 5;
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (i == slot) values[i] = -INFINITY;
            ids[rank] = expert;
            if (s_ids != nullptr) s_ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    route_token(logits, ids, weights, (int) threadIdx.x, nullptr);
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    route_token(logits + (size_t) tk * 512, ids + (size_t) tk * 10, weights + (size_t) tk * 10, (int) threadIdx.x, nullptr);
}
// route_multi for up to 8 tokens (one warp each), then doorbell_publish_res (elementwise.cu) in the same block: the
// ids to the host, the activations too when any expert is not in VRAM, the system fences, the ring.  One launch
// instead of two on the verify window's critical path; every value written is the two kernels'.
__launch_bounds__(256, 1)
__global__ void route_doorbell(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                               int n_tok, const float* __restrict__ x, const int32_t* __restrict__ d_res, int n_expert,
                               int n, float* x_out, int32_t* ids_out, uint32_t* seq, uint32_t ring) {
    __shared__ int32_t s_ids[8 * 10];
    const int warp = (int) threadIdx.x >> 5, lane = (int) threadIdx.x & 31;
    if (warp < n_tok)
        route_token(logits + (size_t) warp * 512, ids + (size_t) warp * 10, weights + (size_t) warp * 10, lane,
                    s_ids + warp * 10);
    __syncthreads();
    const int k = n_tok * 10;
    int any_miss = 0;
    if ((int) threadIdx.x < k) {
        const int32_t id = s_ids[threadIdx.x];
        ids_out[threadIdx.x] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    if (__syncthreads_or(any_miss)) {
        // the same bytes as the doorbell's float copy, 16 at a time when both ends allow it (fewer PCIe writes)
        if ((n & 3) == 0 && ((((uintptr_t) x) | ((uintptr_t) x_out)) & 15u) == 0) {
            const float4* x4 = reinterpret_cast<const float4*>(x);
            float4* o4 = reinterpret_cast<float4*>(x_out);
            for (int i = (int) threadIdx.x; i < n / 4; i += (int) blockDim.x) o4[i] = x4[i];
        } else {
            for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) x_out[i] = x[i];
        }
        __threadfence_system();
    } else if ((int) threadIdx.x < k) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // `ring` (this step's number, known at capture) STORED: the value the read-modify-write would leave, without
        // the read of host memory across PCIe (#649's doorbell_publish_value); 0: the read-modify-write
        if (ring != 0u) *(volatile uint32_t*) seq = ring;
        else *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
        __threadfence_system();
#endif
    }
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_doorbell(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                  const int32_t* d_res, int n_expert, int64_t n, float* x_out, int32_t* ids_out,
                                  uint32_t* d_seq, uint32_t ring, void* stream) {
    if (!stream || n_tok < 1 || n_tok > 8 || !valid(logits, (size_t) n_tok * 512 * 4) ||
        !valid(ids, (size_t) n_tok * 10 * 4) || !valid(weights, (size_t) n_tok * 10 * 4) || !x || !x_out || !ids_out ||
        !d_seq || n < 0 || n > (int64_t) INT32_MAX)
        throw std::invalid_argument("native router + doorbell requires a stream, 1..8 rows and the doorbell's buffers");
    route_doorbell<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok, x, d_res, n_expert,
                                                                     (int) n, x_out, ids_out, d_seq, ring);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
