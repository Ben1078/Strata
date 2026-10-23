# Title

`hip: clamp the three q8_1 scale/sum emit sites (port of upstream #606 / #1448)`

# Summary

This branch is based on **v0.1.40.3** (`d5ea7133`). Upstream clamped the last three unclamped q8_1
activation emit sites in **v0.1.41** — `7a0e3023` ("clamp the three unclamped q8_1 scale/sum emit
sites via q8_1_ds", #606) and `e372d4ff` ("q8_1 emit sites: clamp the scale and quantize against it",
#1448) — and added a non-finite-logits guard in **v0.1.43** (`f332a855`, #879). The base commit
`d5ea7133` still carries the unclamped lines at all three sites, and the branch's own commit
`23d2fc04` (the fused decode kernels) rewrites parts of `fused_gr.cu` and `verify_kernels.cu` but does
not touch these emit sites (`iq_kernels.cu` is untouched entirely), so all three still carry the
pre-fix code.

This ports the two clamp commits verbatim, at the same three sites and with the same helper calls the
rest of the tree already uses (`include/strata/kernels/q8_1_finite.hpp` is already present in this
tree; `iq_kernels.cu` already uses `q8_1_ds` at two other sites).

Without it the branch **degenerates into a persistent local loop once the context passes ~115K–150K
tokens**, which makes it unusable for long agent sessions — the workload it was built for. The full
bug report is in `BUG-q8_1-deep-context-loop.md`.

Three sites, three lines each:

| File | Function |
| --- | --- |
| `src/kernels/cuda/fused_gr.cu` | `gr_q8_tail` — the fused gate-reduce tail |
| `src/kernels/cuda/iq_kernels.cu` | `s26_swiglu_q8_1_kernel` — the S26 swiglu path |
| `src/kernels/cuda/verify_kernels.cu` | `gdn_q8_1_store` — the batched GDN verify store |

```cpp
const float d = q8_1_finite(amax / 127.0f);   // was: amax / 127.0f
const int8_t q = q8_1_quant(xi, d, amax);     // was: amax == 0.0f ? 0 : roundf(xi / d)
...ds = q8_1_ds(d, sum);                      // was: make_half2(d, sum)
```

plus `#include "strata/kernels/q8_1_finite.hpp"` in `fused_gr.cu` and `verify_kernels.cu`
(`iq_kernels.cu` already includes it). The change is a straight port — the code is byte-identical to
upstream's at these sites, so a later rebase onto v0.1.41+ should apply cleanly or be dropped.

# Why this is the right fix and not a guess

A q8_1 block stores its scale `d = amax / 127` and the sum of its 32 raw activations as **fp16**. A
sum past 65504 rounds to `inf`, and `d` overflows past `amax = 8.3M`; the dot products then read
`inf * 0 = NaN`, and the NaN ends at the sampler's all-NaN fallback. This tree's own
`q8_1_finite.hpp` documents exactly that, and names the same defect in llama.cpp (ggml-org/llama.cpp
#23606).

The depth dependence follows: the GDN/QSA recurrent-state magnitude grows with context, so a block
eventually crosses the fp16 range. The persistence follows too: the poison sits in the conversation's
K/V and recurrent state, so with the prompt cache every later turn stays poisoned until the prompt is
read again from token 0.

The base tree has **zero** raw `make_half2(d, sum)` sites. This branch had three. That is the whole
difference, and clamping them removes it.

# Test setup

- **GPUs:** RX 9060 XT 16 GB (`gfx1200`, CUDA0) + RX 6650 XT 8 GB (`gfx1032`, CUDA1), 24 GB total
- **CPU/RAM:** Ryzen 7 7800X3D, 30.4 GiB usable
- **OS / ROCm:** Arch Linux, ROCm 10.1.0 at `/opt/rocm-10`
- **Model:** Qwen3.8-Flash-Next Coder IQ1_M (2 shards, `--native`), MTP draft layer
- **Engine:** built for `gfx1032;gfx1200`, HIP backend, this branch's own `build_engine_hip` flags
  (`STRATA_PREFILL_MMQ=ON`, `STRATA_Q6K_EXPERTS=OFF`, `STRATA_MMQ_KQUANTS=OFF`)
- **Config:** `"gpu": [0,1]`, `"layer_split": 35`, `--max-context 262144 --kv int8 --resident-experts
  --vram-reserve-mib 100 --remote-expert-opt --pipeline-windows 2`, `en` MTP draft subset
- **Workload:** Zoo Code agent session over a ~7.9K-line Python repo (tool-calling, long agent turns)

The failing build and the fixed build are otherwise identical: every other change tried during the
investigation (base `serve/frontend.py`, the whole base `serve/`, build-flag changes, the two Q8_0
fused-kernel switches) was reverted before the fixed build was made, so the only delta is the clamp.

# Results

**Before:** every run degenerated into the loop after the context passed ~115K–150K tokens; never
before, and it did not clear once started. Frontend swap, fused-kernel switches and build-flag changes
all still looped.

**After:** a full agent run crossed the entire failure band and completed coherently:

```
prompt 123464 tokens = 100011 reused + 23453 read ... 262 generated in 3991 ms (65.7 tok/s)
prompt 134325 tokens = 123459 reused + 10866 read ... 319 generated in 4989 ms (63.9 tok/s)
prompt 143296 tokens = 134320 reused +  8976 read ... 228 generated in 3270 ms (69.7 tok/s)
prompt 148177 tokens = 143291 reused +  4886 read ... 354 generated in 6469 ms (54.7 tok/s)
prompt 155474 tokens = 148172 reused +  7302 read ... 12577 generated in 341541 ms (36.8 tok/s)
prompt 158058 tokens = 155469 reused +  2589 read ... 2138 generated in 38808 ms (55.1 tok/s)
```

Deepest request: **158,058 tokens of context, 12,577 tokens generated, drafts accepted 6,934 of
11,756.** Peak decode 69.7 tok/s, ~36.8 tok/s sustained across the long generation. No degeneration.

# Notes

- The patch is in this directory: `q8_1-clamp-0.1.40.3.patch`
  (`sha256 406cc12c852ce3780b26254e9df80a746c43b068b8d870d3050fd38d62922048`), apply with
  `git apply` and rebuild the engine.
- **`f332a855` (the #879 NaN guard) is deliberately not included.** With the producer clamped there
  should be no NaN to catch; the guard is a safety net for a different overflow path. It is worth a
  separate port if you want the branch to self-heal rather than trust the producer — it is the larger
  change (it touches `generate.cpp`, `verify.cpp`, `verify_kernels.cu` and `serve/server.py`).
- **Numerics:** a block that was finite before is stored bit for bit as before, so outputs on short
  contexts are unchanged. A block that would have overflowed now produces an approximate product
  instead of NaN — that is the point.
- **Untested here:** CUDA and SYCL builds; other models and quants; single-card depth runs. The defect
  is in shared `src/kernels/cuda/*.cu`, compiled for every backend, but only HIP was verified on this
  rig.
- **Separate concern:** building this branch for the RX 6650 XT (`gfx1032`) needs a small local
  `setup.py` / `cmake/hip_backend.cmake` enablement (upstream added it in 0.1.44+). It is unrelated to
  this fix and is not part of this commit.
