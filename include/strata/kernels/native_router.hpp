#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// native_router_top10_multi (n_tok <= 8) and then doorbell_publish_res (elementwise.hpp) of its ids in ONE launch:
/// ids/weights as the top-10, ids_out / x_out (the n floats of x, when an id is not resident in d_res) / the ring
/// d_seq as the doorbell.  Bitwise the two launches.  ring != 0: d_seq is set to ring (the step's number, which the
/// read-modify-write of a ring reset at the window's start leaves) without reading it back over PCIe.
void native_router_top10_doorbell(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                  const int32_t* d_res, int n_expert, int64_t n, float* x_out, int32_t* ids_out,
                                  uint32_t* d_seq, uint32_t ring, void* stream);
}
