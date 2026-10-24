# [Bug]: `rdna3-moe-offload-perf` degenerates into a local loop past ~115K context (base v0.1.40.3)

## What happened

On a two-card layer split (RX 9060 XT 16 GB + RX 6650 XT 8 GB), running Qwen3.8-Flash-Next Coder
IQ1_M at 256K context with an agent client (Zoo Code), this branch answers coherently up to roughly
**115K–150K tokens of context**, then falls into a **self-reinforcing loop** and never comes out of it.

The loop is a reasoning/planning attractor, not a text repetition: the model re-enters the same
bookkeeping step — *"Let me continue reading the remaining files: X, Y, Z"* — issues another pair of
`readFile` tool calls, and returns to the same step. One capture shows **6 restarts** of that sentence
and ~20 near-identical re-entries, after which it re-enumerates a findings list it had already
produced ~10K tokens earlier. The per-pass text is *different* each time, so surface n-gram loop
detection does not fire (`reasoning_recoveries: 0` throughout).

**Fingerprint (the useful part):** every failure begins only after the context passes ~115K–150K
tokens. Never before. Once it starts, it persists across turns even though each turn's prompt differs,
because the poison lives in the conversation's carried state and the prompt cache replays it.

**Expected:** coherent continuation at every context depth up to `--max-context`.
**Got:** a loop that begins at a depth threshold and does not clear.

## Version, GPU, OS

- **Strata:** fork branch `rdna3-moe-offload-perf` (`Ben1078/Strata`), HEAD `d1754f73`, **base
  `d5ea7133` = v0.1.40.3** (2026-10-07). Upstream's clamp for this defect first shipped in **v0.1.41**
  (`7a0e3023` + `e372d4ff`, 2026-10-08) and the NaN guard in **v0.1.43** (`f332a855`, 2026-10-11) —
  the branch's base predates all three, and none were picked up by its 19 commits.
- **Engine:** built locally for `gfx1032;gfx1200`, HIP backend, ROCm 10.1.0 at `/opt/rocm-10`
  (`STRATA_PREFILL_MMQ=ON`, `STRATA_Q6K_EXPERTS=OFF`, `STRATA_MMQ_KQUANTS=OFF` — the branch's own
  `build_engine_hip` flags). The fix was verified on a rebuild with the identical flags.
- **GPU:** RX 9060 XT 16 GB (`gfx1200`, CUDA0) + RX 6650 XT 8 GB (`gfx1032`, CUDA1), plain PCIe.
- **CPU / RAM:** Ryzen 7 7800X3D, 30.4 GiB usable.
- **OS / runtime:** Arch Linux, ROCm 10.1.0.
- **Model:** Qwen3.8-Flash-Next Coder IQ1_M (2 shards, `--native`), MTP draft layer (`en` subset).
- **Config:** `"gpu": [0,1]`, `"layer_split": 35`, `--max-context 262144 --kv int8 --resident-experts
  --vram-reserve-mib 100 --remote-expert-opt --pipeline-windows 2`, env `STRATA_SPLIT_OWN=1
  STRATA_STAGE_TRIM=1 STRATA_RESIDENT_PIN=0`.

## Where it comes from

Three q8_1 activation emit sites in this branch store the block's `(d, sum)` pair with a raw
`make_half2(d, sum)` and quantize against an unclamped `d`, bypassing the finite helper
(`include/strata/kernels/q8_1_finite.hpp`) that the rest of the tree already uses:

| File | Function |
| --- | --- |
| `src/kernels/cuda/fused_gr.cu` | `gr_q8_tail` — the fused gate-reduce tail |
| `src/kernels/cuda/iq_kernels.cu` | `s26_swiglu_q8_1_kernel` — the S26 swiglu path |
| `src/kernels/cuda/verify_kernels.cu` | `gdn_q8_1_store` — the batched GDN verify store |

A q8_1 block stores its scale `d = amax / 127` and the sum of its 32 raw activations as **fp16**. A
sum past 65504 — one massive-activation dimension among the 32 is enough — rounds to `inf`, and `d`
itself overflows past `amax = 8.3M`. Nothing downstream tests for it: the dot products read
`inf * 0 = NaN`, every sum downstream of the block is poisoned, and a non-finite logit row ends at the
sampler's all-NaN fallback. The branch's own header states the failure exactly:

> "A sum past 65504 (one massive-activation dimension among 32 values is enough) rounds to inf, and d
> does past amax = 8.3M; the dot products then read inf * 0 = NaN, and a NaN in the residual ends as
> one token repeated."

Why it only starts at depth: the GDN/QSA recurrent-state magnitude grows with context, so an
activation block eventually crosses the fp16 range. Before that threshold the branch is bit-for-bit
fine — which is why every short run passed and only long ones failed. And why it persists: the poison
sits in the conversation's K/V and recurrent state, so with the prompt cache **every later turn stays
poisoned until the prompt is read again from token 0**.

The base tree has **zero** raw `make_half2(d, sum)` sites; this branch has three. The fix is upstream's
two commits, ported verbatim:

```cpp
const float d = q8_1_finite(amax / 127.0f);   // was: amax / 127.0f
const int8_t q = q8_1_quant(xi, d, amax);     // was: roundf(xi / d)
...ds = q8_1_ds(d, sum);                      // was: make_half2(d, sum)
```

`q8_1_ds()` clamps both halves into the fp16-finite range, producing the same bits the CPU pool and
the already-clamped GPU sites produce for the same block. A block that was finite before is stored bit
for bit as before.

## How it was isolated

Each arm was a separate server start with the same model, config and agent task; only the named change
differed. Every arm that did **not** touch the q8_1 emit sites still looped:

| arm | change | result |
| --- | --- | --- |
| baseline | branch as shipped | loops |
| frontend | base v0.1.45 `serve/frontend.py` copied in | loops |
| A | `STRATA_GDN_INPROJ=0 STRATA_MMVQ_Q8W=0` (both Q8_0 fused decode kernels off) | loops |
| B | rebuilt with `-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON` | loops (placement unchanged: 218 native projections / 4703+926 slots, byte-identical to the branch's own build) |
| C | the whole base v0.1.45 `serve/` copied in, branch engine kept | not concluded |

Everything above was then **reverted** — `serve/` restored to the branch's HEAD, build flags restored,
config restored — so that the fixed build differs from the failing build by exactly the three clamped
lines. On that build, a full agent run crossed the whole failure band:

```
prompt 123464 tokens = 100011 reused + 23453 read ... 262 generated in 3991 ms (65.7 tok/s)
prompt 134325 tokens = 123459 reused + 10866 read ... 319 generated in 4989 ms (63.9 tok/s)
prompt 143296 tokens = 134320 reused +  8976 read ... 228 generated in 3270 ms (69.7 tok/s)
prompt 148177 tokens = 143291 reused +  4886 read ... 354 generated in 6469 ms (54.7 tok/s)
prompt 155474 tokens = 148172 reused +  7302 read ... 12577 generated in 341541 ms (36.8 tok/s)
prompt 158058 tokens = 155469 reused +  2589 read ... 2138 generated in 38808 ms (55.1 tok/s)
```

**Result: coherent to 158,058 tokens, 12,577 generated in the deepest request, drafts accepted 6,934
of 11,756.** Every prior failure began in the 115K–150K band; this run crossed the entire band and
completed. The only change from the failing build is the clamp.

## Workaround

None that preserves the workload. A smaller `--max-context` (below the depth at which the first block
overflows) avoids the trigger but does not fix it, and the threshold moves with the model's state
magnitude. Apply the clamp:

```
git apply patches/q8_1-clamp-0.1.40.3.patch     # then rebuild the engine
```

## What was not tested

- **One model, one quant.** Coder IQ1_M only. Not the IQ2_XS / IQ3_S / Unsloth packs.
- **One card pair.** 9060 XT + 6650 XT. No single-card run at depth, no equal pair, no 3+ cards.
- **One backend.** HIP only. CUDA and SYCL were not built or run; the defect is in shared
  `src/kernels/cuda/*.cu` compiled for both, but only HIP was verified here.
- **One sampler path.** The loop was observed with greedy/`temperature 0` agent decoding and MTP
  speculation. Sampling paths were not separately exercised.
- **The `f332a855` NaN guard was not ported.** With the producer clamped there should be no NaN to
  catch; the guard is a safety net for a different overflow path and was deliberately left out of
  this fix so the test isolates the clamp.
- **Noise.** The depth band is from repeated observation (three separate sessions), not a controlled
  sweep; the exact first-failure token is not pinned.

## Related

- Upstream **#606** — the q8_1 finite scale/sum defect (`q8_1_finite.hpp`), fixed for the other emit
  sites earlier; these three were missed.
- Upstream **#1448** — clamp the scale and quantize against it (`e372d4ff`), the second half.
- Upstream **#879** — non-finite logits, the guard with self-healing (`f332a855`).
- Upstream `docs/Q8_1_FINITE_DS.md` and `docs/DEGENERATION_RCA.md` — the wider carried-state RCA,
  referenced by `7a0e3023` (not present in this tree).
