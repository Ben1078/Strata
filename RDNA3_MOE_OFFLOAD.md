# Qwen3.8-Flash-Next at a usable quant on one RX 7900 XTX: ~58 tok/s decode at UD-Q5_K_XL

This is a fork branch of [Strata](https://github.com/Niko1221/Strata) (by Niko1221, MIT) with
18 commits of performance work for running a large MoE model with its experts split between
one 24 GB RDNA3 GPU, system RAM and the CPU.

**Why this exists:** most fast single-24 GB-card numbers you'll see for Qwen3.8-Flash-Next use
heavily quantized experts (2-4 bit: IQ3_S, IQ4_XS, ...). This branch targets a quant you'd
actually want to use every day: **Unsloth UD-Q5_K_XL** (Q5_K gate/up, Q8_0 down projections,
~92 GiB of experts), and still gets:

- **~58 tok/s decode** on general prompts (stock Strata 0.1.40.3 on the same box: ~36 tok/s),
  ~70 tok/s on agent/coding output and ~100 tok/s while thinking (observed in real use;
  predictable text means longer accepted speculative windows)
- **~2-3x faster turns in long multi-turn chats** (time to first token on a 100K-token
  conversation with a few hundred new tokens: ~2.2-3.1 s -> ~0.8-1.1 s)
- huge conversations brought back from NVMe in under a second instead of a full re-read

Hardware: Ryzen 7 7800X3D, RX 7900 XTX 24 GB, 128 GB DDR5-5600, ROCm 7.2, Linux. Every new
behaviour is behind an env var or flag (default off) unless a commit says otherwise; see the
per-feature commits and the tables below. These changes were written and tested with AI coding
agents on this one machine - expect different numbers on other hardware, and treat the code as
a starting point for review, not a drop-in.

---

## Full benchmark notes: before/after, what helped, what did not

Branch: 18 commits on top of Strata 0.1.40.3 (`d5ea7133`), one commit per feature, every new
behaviour behind an env var or flag (default off) unless noted in the commit.

## Machine and model (the same for every number below)

| | |
|---|---|
| CPU | Ryzen 7 7800X3D (8C/16T, Zen 4, AVX-512 incl. BF16, 96 MiB L3) |
| GPU | RX 7900 XTX 24 GB (gfx1100), PCIe 4.0 x16, ROCm 7.2.x, Linux |
| RAM | **96 GB DDR5-5600 for the stock baseline, 128 GB DDR5-5600 (FCLK 2100) from Oct 21** |
| Model | Qwen3.8-Flash-Next, Unsloth UD-Q5_K_XL (experts Q5_K gate/up, Q8_0 down; 48 layers x 512 experts, 10 routed + 1 shared; 91.6 GiB of experts) |
| Placement | ~3.5-4.4K hottest experts in VRAM, the rest locked in RAM, CPU expert pool for misses, a share of misses read by the GPU over PCIe; MTP draft head |
| Use | one user at a time (agent clients + chat), greedy unless the client sets sampling |

Measurement tools: `run.py` = one engine start, 6 fixed prompts, 512 tokens, greedy, 2 rounds,
mean of per-prompt medians (run-to-run noise about +-3%); fresh prompt-processing runs at 16K/64K;
`prof_bench` = 2-3 rounds of "std" (general) and "agent" (tool-calling) prompt sets; `turns` = a
~100K-token conversation plus 8 turns of 500-1,900 new tokens.

## Headline: stock vs now

| | Stock 0.1.40.3 (Oct 14, 96 GB) | Now (Oct 22, 128 GB) | Change |
|---|---|---|---|
| Decode, 6 standard prompts (run.py) | 35.2 / 36.3 tok/s (two runs) | **58.3** tok/s | +63% |
| Decode, live production (no restart, same prompts) | - | 55.8 tok/s | |
| Decode, prof_bench std / agent | - | 54.5 / 56.9 tok/s | |
| Decode, agent text / while thinking (user-observed in the client, not benchmarked) | - | ~70 / ~100 tok/s | |
| Fresh prompt, 16K tokens | 697 tok/s | ~1,400 tok/s | ~2.0x |
| Fresh prompt, 64K tokens | 1,231 tok/s | ~1,790 tok/s | ~1.45x |

Fresh-prompt "now" numbers are from the reference run just before rounds 6-7 (1,400 / 1,786 tok/s);
rounds 6-7 targeted decode and turns on long chats, not fresh prompts, and did not re-measure them.
| Turn on a 100K-token chat (500-2,000 new tokens) | - (see note) | **0.54x** the time of the Oct 21 engine | |
| Bring back a 100K-token chat after another client used the engine | full re-read (~50 s on the current engine without parking) | 0.45 s (from NVMe) | |
| Max context in the config | 128K | 1M (YaRN x4) | config |

Note on turns: the turn benchmark did not exist at the stock baseline. Its reference is the Oct 21
production engine (already 128 GB, already rounds 1-5): 500 new tokens 3,078 -> 1,650 ms, 2,000 new
tokens 4,462 -> 2,419 ms.

Draft acceptance went from 0.63 (stock config, `--spec` default) to 0.82-0.85.

## How it got there (each stage measured with the configuration it was deployed with)

The configuration changed between stages (context length, vision encoder on the GPU, RAM), so the
stages are not a clean ablation. Numbers are mean of per-prompt medians unless stated.

| Stage (date, RAM) | Decode | Prompt 16K / 64K | What changed |
|---|---|---|---|
| Stock (Oct 14, 96 GB) | 35.2 / 36.3 | 697 / 1,231 | 0.1.40.3 + the two patches needed to run this model here (GGUF shard fix, ROCm mlock) |
| Round 1-2 (Oct 15, 96 GB) | 43.7 | 978 / 1,832 | GPU launch fusions, L3 prefetch, KV mlock (code); `--spec 6 --spec-min-p 0.85 --pcie-frac 0.30 --pool-tasks 96 --kv-resident 32768`, upstream env flags `STRATA_QFUSE STRATA_MMVF_ROWS STRATA_ATTN_LANECELL STRATA_HIP_WMMA STRATA_PREFILL_RING=384` (config) |
| Round 3 (Oct 19, 96 GB) | 44.2 | 1,154 / 1,526 (1,630 at 130K) | gfx1100 QSA scorer (code); buffered pread for the file tier (upstream flags); config had meanwhile gained the GPU vision encoder (-527 expert slots) and lost `GPU_MAX_HW_QUEUES=8` (-2%, see tips) |
| 128 GB (Oct 21, config only) | 45.5 (unchanged) | live: 895 -> 1,502 (16K), 1,364 -> 1,852 (65K), 1,210 -> 1,914 (128K) | every non-VRAM expert locked in RAM (`--resident-budget-gib 78`), no file-tier reads during decode, bigger page cache |
| Round 5 (Oct 21, config) | prof_bench std 46.0 -> 48.8, agent 47.0 -> 49.0 | - | `--adapt-every 4 --adapt-async 1 --vram-reserve-mib 800`; then 1M context (decode unchanged, -358 slots) |
| Reference before round 6 (Oct 22) | 48.2 | 1,400 / 1,786 | the above |
| Round 6-7 release (Oct 22) | **58.3** | ~1,400 / ~1,790 | helper-thread fix, multi-thread L3 prefetch, cost-based spec windows, draft vocab subset, resident lend, prompt-path CPU share, VRAM band + vision on demand + elastic context |
| Disk parking (Oct 22) | unchanged | unchanged | park files + one-conversation disk policy |

### What came from the 96 -> 128 GB RAM change

- Fresh prompt processing at 16K-128K: the config-only change on Oct 21 moved live prompt speed from
  895 / 1,364 / 1,210 to 1,502 / 1,852 / 1,914 tok/s (16K / 65K / 128K) with the same engine. Most of
  the 16K gain over stock is therefore RAM plus round 1-3 code, not round 6-7.
- Decode did not move with the RAM itself (45.5 before and after on the same prompts).
- RAM is what makes several features affordable: all experts resident (no SSD reads during decode),
  the 1M-context host KV (12.4 GiB pinned), `STRATA_RESIDENT_LEND` (+4.7 GB locked), a 10 GiB
  conversation cache. On 96 GB the system was at the edge: lowering the expert budget by 2 GiB
  (thrashing the page cache) dropped decode to ~11 tok/s.

## Per-feature contribution (where it was measured on its own)

| Feature (commit) | Default | Measured | Numerics |
|---|---|---|---|
| GGUF shard padding fix | on (bug fix) | needed to load UD-Q5_K_XL at all | - |
| mlock of ROCm-mapped expert arena / KV copies | on (HIP+Linux) | removes swap-out stalls and an amdgpu re-validation timeout under memory pressure | identical |
| GPU launch fusions (round 1) | on where shapes match (`STRATA_POST_FUSE=0`, `STRATA_ROUTE_DOORBELL=0` to disable) | round-1 GPU agent: 38.5 vs ~35.7 tok/s together with `--kv-resident` and upstream env flags | bitwise |
| L3 prefetch, 1 thread (`STRATA_L3_PREFETCH=64`) | off | 96 GB: part of the CPU agent's 37.4 vs 35.3 (with `--pool-tasks 96 --pcie-frac 0.30`); on 128 GB later: worth ~0 on its own (lookahead + prefetch off 49.1 vs on 49.3) | bitwise |
| `--spec 6 --spec-min-p 0.85` (config) | - | 38.7 vs 35.7 tok/s (+8%) | lossless |
| gfx1100 QSA scorer (`STRATA_SCORES_THREAD=1`) | off | long prompts (130K: 1,630 tok/s with the round-3 config) | bit-exact (parity test) |
| Helper threads off CPU 0 (`STRATA_HELPER_CPUS=auto`) | off | **47.5 -> 49.3 tok/s (+3.8%)**, same binary; GPU wait for the host's plan 1.52 -> 0.51 ms/window | bitwise (scheduling only) |
| L3 prefetch on 3 threads + CPU share only + BF16 lookahead | off | 49.9 vs 49.3-49.4 (+1%); prefetched 12.8 vs 6.5-7.5 MB/layer | bitwise |
| Cost-based spec windows (`--spec 8 --mtp-max-t 8 --spec-policy cost`) | off | same engine, interleaved, 3 rounds: +3.3% std, +5.2% agent; pooled over 2 engine starts +1.0% / +4.0% | lossless (window size only) |
| Draft vocab subset (data file for upstream's `--mtp-draft-vocab`) | - | draft step 1.14-1.21 -> 0.90 ms; frees ~160 MiB of VRAM (+42 expert slots) | placement only (see below) |
| VRAM band + vision on demand + elastic context (`STRATA_VRAM_BAND_MIB`, `STRATA_CTX_GROW=1`, `vision_on_demand`) | off | 3,563 -> 4,354 expert slots; decode 48.2 -> 50.8 (+5.3%) on its own; image round trip: band out 460 ms, encoder up 1.1 s, band back 240 ms | allocation only; placement |
| Resident lend (`STRATA_RESIDENT_LEND=1`, budget 82.5) | off | 100K turns: prompt 4,302 -> 3,959 ms (-8%), refill wait 245 -> 0 ms, SSD reads 761 -> 23 MB/turn; +4.7 GB locked | same bytes |
| Prompt-path CPU share (`STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_MAXT=128 STRATA_PF_KQ512=1 STRATA_PREFILL_STREAM_MIN=8192 STRATA_PF_STAGE_FIT=1 STRATA_PARK_SKIP_TAIL=16384`) | off | 100K turns: prompt 4,896 -> 2,299 ms mean (-53%); 500 new tokens 3,086 -> 1,570 ms; 2,000: 4,342 -> 2,738 ms | CPU rounding for CPU-computed experts (below) |
| Trailing-reminder fix (server) | on (`STRATA_TRAILING_REMINDER=0` off) | an 18K agent chat: 18,271 of 18,822 tokens reused instead of falling back to the last 16K checkpoint (was 5-19 s re-reads per turn) | prompt text unchanged except where the reminder sits |
| Park files + `disk_park` | off | 100K-token chat restored in 0.45 s, 60K in 0.28 s, 30K in 0.22 s; an incremental save after one turn wrote 126 MB instead of 1.65 GB | restored state resumes with the same tokens as the RAM path |
| All of round 6-7 together | | 48.2 -> 58.3 tok/s (+21%), 38.9 ms/window at 2.26 tokens/window (was 43.2 at 2.08) | see below |

The configuration "now" runs (paths shortened; the expert profile and draft vocab are local data files):

```
args: --resident-budget-gib 82.5 --expert-profile <blend profile> --expert-cache auto --prefill auto
      --spec 8 --mtp-max-t 8 --spec-policy cost --spec-min-p 0.85 --mtp-draft-vocab <subset.bin>
      --kv int8 --kv-resident 32768 --max-context 1048576 --rope-scaling yarn --rope-scale 4
      --pcie-frac 0.30 --pool-tasks 96 --adapt-every 4 --adapt-async 1 --vram-reserve-mib 800
      --vision --prompt-cache 8 --conversation-cache-mib 10240 --conversation-cache-slots 8
env:  STRATA_QFUSE=1 STRATA_MMVF_ROWS=1 STRATA_ATTN_LANECELL=1 STRATA_HIP_WMMA=1 STRATA_PREFILL_RING=384
      STRATA_HIPBLASLT_TUNING=<upstream gfx1100 tuning file> STRATA_IO_PREFETCH=1 STRATA_IO_PF_AHEAD=0
      STRATA_SCORES_THREAD=1 STRATA_L3_PREFETCH=64 STRATA_HELPER_CPUS=auto STRATA_IO_PREFETCH_DEPTH=1
      STRATA_L3_PF_THREADS=3 STRATA_L3_PF_CPU_ONLY=1 STRATA_LOOKAHEAD_BF16=1 STRATA_RESIDENT_LEND=1
      STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_MAXT=128 STRATA_PF_KQ512=1
      STRATA_PREFILL_STREAM_MIN=8192 STRATA_PF_STAGE_FIT=1 STRATA_PARK_SKIP_TAIL=16384 STRATA_CTX_GROW=1
server config: "effort_position": "end",
      "vision_on_demand": {"band_mib": 3584, "reserve_mib": 2300, "need_mib": 1600, "idle_s": 300},
      "disk_park": {... one conversation, see the serve commit ...}
```

## Numerics

- Most features are bit-identical by construction or by test (fusions, prefetch, scheduling, spec
  window size, lend, allocation changes). Parity tests: `kq512_parity`, `expert_multi_test`,
  `qsa_scores_gfx1100_parity`, `cost_policy_test`, `session_park_test`, the upstream `hip_prefill_*`
  parity tests.
- What does change the output is **expert placement**: experts that move between the GPU cache and
  the CPU pool are computed with different rounding. The release moved ~790 experts onto the GPU.
  On a deterministic teacher-forced probe (4,248 positions) the release vs the previous production:
  argmax agreement 91.1%, mean KL 0.072, perplexity without end-of-turn 15.27 -> 15.01 (slightly
  better). The attribution runs showed the new binary, the cost policy, the window cap and every
  CPU/prompt-path env var are bit-identical to the old engine on this probe; the whole shift comes
  from placement (the smaller draft head's +42 slots already gives 91.7% / 0.067).
  On this 1M-context/YaRN config that is the normal size of a placement change; it is larger than
  the 98.4% / 0.013 floor measured on the older 128K config.
- The prompt-path CPU share computes some prompt experts on the CPU (Q8_K activations, like decode):
  after a 2K-token prompt-path read, argmax 96.6%, KL 0.013, ppl +0.5% vs the GPU-only path. Reading
  the same tokens through the decode windows (what upstream does for short parts) moves it more
  (96.6%, KL 0.016).
- Greedy outputs differ from run to run anyway with `--adapt-async` (the adaptive expert tier moves
  experts in the background).

## Dead ends (measured, not used)

| Idea | Result |
|---|---|
| Route tail skip (port of upstream 0.1.42, `STRATA_ROUTE_TAIL_SKIP=K`) | decode K=9/8/7: 48.1 / 52.1 / 56.0 vs 44.9 tok/s, but KL 0.020 / 0.028 / 0.041 vs a 0.011 placement floor; lossy, **not used and not in the branch** |
| GPU hyper-connection kernel fusion (one launch per read, in-kernel role sync) | worked after fixing a gfx1100 GL1 stale-read hang, but per read slower at the usual 3-4 window rows (T3 33.8 -> 38.5 us, T4 36.2 -> 41.6 us); CONVREC + MERGE_FAST flags 47.6 vs 47.7 tok/s: neutral |
| Device-side PCIe "gap fetch" of predicted experts (graph branch) | 47.9 -> 37.0 tok/s (-23%): a per-layer cross-stream fork/join inside a HIP graph costs more than it hides; ring hit rate only 14% |
| Q8 copy of the hyper-connection weights (`STRATA_HC_Q8`) to free VRAM | 46.7 vs 48.2 tok/s and KL 0.018 |
| MALL (Infinity Cache) prefetch of dense weights | estimated <= 2 ms/window; the window test was interrupted, never showed a gain; dropped |
| gemv retuning for the dense side | 48.5 vs 48.2 tok/s (+0.6%, noise); micro-bench +2.9% / -2.1% by mode |
| CPU pool rework (`STRATA_POOL2_FUSE`) | 48.3 vs 48.2 tok/s (+0.2%), neutral |
| CPU writes its expert rows straight into VRAM through the BAR | estimated <= +0.5-0.8%; test run failed; dropped |
| `--pcie-frac 0.35` (more PCIe share) | 47.5 vs 49.4 tok/s |
| Blocking adaptive tier (`--adapt-every 4` without `--adapt-async`) | std/agent 43.7 / 41.3 vs 46.0 / 47.0 |
| `--sfx-any` (lookup drafts when the MTP head disagrees) | lookup right only 8-12% of the time in those cases |
| Window cap 8 forced (T=8 every window) | 32.7 tok/s vs 45-47 at T=2-4: one verify row costs ~9-11 ms here |
| AVX-512 multi-token Q5_K/Q8_0 kernels for decode | +2.5% in isolation, ~0 end-to-end: the pool is DRAM-bound (~59-66 GB/s) |
| Static CPU expert set for long prompts, grouping on cached table copies | no gain at 2K, worse at 4K; 16K read 9,550 vs 5,336 ms |
| `--pcie-mode dma/direct`, `--prompt-cache-every 8192`, new expert profiles, ROCm 10.2 nightly | noise / mixed / equal / -1.5% |
| Upstream 0.1.45 on this config | 40.4 vs ~44 tok/s at the time |
| Async MTP draft during verify | expected +0.1-0.4%, below noise |

Also learned: the CPU pool vs PCIe DMA "collapse" seen on the 96 GB setup did not reproduce on
128 GB / FCLK 2100: CPU 51-54 GB/s + GPU 19-27 GB/s = 74-78 GB/s, the DRAM's practical ceiling.

## Caveats

- Tuned for one machine and one model. The knobs exist because this box is CPU/DRAM-bound with a
  24 GB card and a 92 GiB expert set; on a bigger GPU or a smaller model several of them (CPU share,
  L3 prefetch, lend) may do nothing or hurt. Defaults are off so nothing changes unless you opt in.
- Written by AI coding agents (Claude, with some Codex work) under the owner's direction, and tested
  on this one box only. Builds and CPU unit tests pass; the GPU paths were exercised on gfx1100 only.
  CUDA builds were not tested.
- Speed numbers are single-machine, +-3% run-to-run; the stages above mix code and config changes.
  "User-observed" numbers come from watching the client, not from a benchmark.
- The draft vocab subset used in production is a data file built from the owner's own transcripts;
  it is not part of the branch (anyone can build one from their own outputs).
- Code comments still carry internal round tags ("q5opt r6-mem", "r8") from the work log.
- `immintrin.h` is included unconditionally in `expert_source.cpp` (L3 prefetch): non-x86 builds
  need a guard before this goes upstream.

## Tips for other ROCm users (no code needed)

- If the 7900 XTX sits at 100% busy / ~110 W while idle, count the compute queues held by ROCm
  processes: more than 8 in total (engine + vision helper) kept the GPU spinning here. Starting the
  vision helper with `GPU_MAX_HW_QUEUES=2` fixed it (idle 0% / ~10-15 W).
- ROCm maps host memory for the GPU without pinning it. With a large resident expert arena, give the
  service `LimitMEMLOCK=infinity` and use the mlock commits, or zram/swap will stall the GPU.
