// src/kernels/r6mem_bench.cpp - q5opt round 6 (r6-mem): the CPU expert pool, the DRAM and the PCIe tier on this PC,
// measured in isolation (a port of round 1's cpu_pool_bench with the engine's own PCIe copy kernel).
//
//   r6mem_bench bw [threads...]                    CPU read bandwidth; thread i pinned to logical CPU i
//                                                  (8..15 are the SMT siblings of 0..7 on a 7800X3D)
//   r6mem_bench gpu KIND [seconds]                 the PCIe tier alone: KIND 1 = copy engine (hipMemcpyAsync H2D of
//                                                  3.99 MB blobs), 2 = the engine's fetch_blobs kernel reading the
//                                                  mapped page-locked arena (the decode's --pcie-mode auto)
//   r6mem_bench mix THREADS KIND [blob_kb]         CPU read threads (1..8) and the GPU load together: both rates
//   r6mem_bench pool LAYERS EXPERTS TASKS KIND     the engine's pool (7 workers + host, run_split_multi_native) on
//                                                  synthetic Q5_K/Q8_0 layers, the decode's token mix; KIND 0 = alone
//   r6mem_bench split LAYERS MISSES FRAC GAP_US PF_MB HIT PF_THREADS KIND
//                                                  a decode layer modelled: MISSES RAM experts, the last FRAC of them
//                                                  over PCIe (KIND as above), the rest on the pool; GAP_US of GPU-
//                                                  serial time between layers in which PF_THREADS threads read the
//                                                  next layer's CPU experts (HIT of the list right) into the L3
// The arena is 6 GiB, page-locked and mapped (cudaHostAlloc + mlock: the engine's RAM copy), filled with copies of
// 16 synthetic experts quantized by ggml, so no layer finds its experts in the 96 MiB L3 by accident.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <immintrin.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
using clk = std::chrono::steady_clock;

namespace {
constexpr int64_t H = 2560, FF = 640;

int type_of(const std::string& name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n && name == n) return t;
    }
    std::fprintf(stderr, "unknown type %s\n", name.c_str());
    std::exit(2);
}

void pin(int core) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(core, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// the engine's RAM copy: page-locked and mapped, mlock'd (ROCm maps it without pinning)
struct Arena {
    uint8_t* base = nullptr;
    const uint8_t* dev = nullptr;
    size_t stride = 0;
    int n = 0;
};

std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    return blob;
}

Arena make_arena(const cpu::NativeFmt& f, size_t bytes) {
    Arena a;
    a.stride = f.bytes;   // packed like the engine's copy (blobs back to back)
    a.n = (int) (bytes / a.stride);
    void* p = nullptr;
    if (cudaHostAlloc(&p, (size_t) a.n * a.stride, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        std::fprintf(stderr, "cudaHostAlloc of %.2f GiB failed\n", (double) a.n * a.stride / 1073741824.0);
        std::exit(1);
    }
    if (mlock(p, (size_t) a.n * a.stride) != 0) std::perror("mlock");
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) { std::fprintf(stderr, "no device alias\n"); std::exit(1); }
    a.base = (uint8_t*) p;
    a.dev = (const uint8_t*) d;
    std::vector<std::vector<uint8_t>> src;
    for (int i = 0; i < 16; ++i) src.push_back(synthetic_blob(f, 100 + i));
    for (int i = 0; i < a.n; ++i) std::memcpy(a.base + (size_t) i * a.stride, src[(size_t) (i % 16)].data(), f.bytes);
    return a;
}

cpu::NativeFmt fmt_q5() {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(type_of("q5_K"), type_of("q8_0"), H, FF, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
    return f;
}

// ------------------------------------------------------------------------------------------------ CPU read threads
struct CpuReaders {
    std::atomic<bool> stop{false};
    std::atomic<long long> bytes{0};
    std::vector<std::thread> th;
    void start(const uint8_t* buf, size_t per, int nt, int first_cpu = 0) {
        for (int t = 0; t < nt; ++t)
            th.emplace_back([this, buf, per, t, first_cpu] {
                pin(first_cpu + t);
                const uint8_t* p = buf + (size_t) t * per;
                __m512i acc = _mm512_setzero_si512();
                while (!stop.load(std::memory_order_relaxed)) {
                    for (size_t o = 0; o + 256 <= per && !stop.load(std::memory_order_relaxed); o += 256) {
                        acc = _mm512_add_epi64(acc, _mm512_loadu_si512(p + o));
                        acc = _mm512_add_epi64(acc, _mm512_loadu_si512(p + o + 64));
                        acc = _mm512_add_epi64(acc, _mm512_loadu_si512(p + o + 128));
                        acc = _mm512_add_epi64(acc, _mm512_loadu_si512(p + o + 192));
                        if ((o & ((1u << 20) - 1)) == 0) bytes.fetch_add(1 << 20, std::memory_order_relaxed);
                    }
                }
                volatile long long sink = _mm512_reduce_add_epi64(acc);
                (void) sink;
            });
    }
    void end() { stop.store(true); for (auto& t : th) t.join(); th.clear(); }
};

int cmd_bw(int argc, char** argv) {
    std::vector<int> counts;
    for (int i = 0; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
    if (counts.empty()) counts = {1, 2, 3, 4, 6, 8, 16};
    const size_t per = (size_t) 256 << 20;
    const int maxt = *std::max_element(counts.begin(), counts.end());
    uint8_t* buf = (uint8_t*) mmap(nullptr, per * (size_t) maxt, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, per * (size_t) maxt, MADV_HUGEPAGE);
    std::memset(buf, 1, per * (size_t) maxt);
    for (int nt : counts) {
        CpuReaders r;
        r.start(buf, per, nt);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const long long b0 = r.bytes.load();
        const auto t0 = clk::now();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        const double gbs = (double) (r.bytes.load() - b0) / secs(t0, clk::now()) / 1e9;
        r.end();
        std::printf("bw %2d threads (CPUs 0..%d): %.1f GB/s (%.1f per thread)\n", nt, nt - 1, gbs, gbs / nt);
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ the PCIe tier load
// kind 1: the copy engine (hipMemcpyAsync H2D, 16 blobs per sync); kind 2: the engine's fetch_blobs kernel (384 x 256
// threads, 16-byte loads of the mapped arena) on `per_launch` blobs, launch after launch on one stream.
struct GpuLoad {
    std::atomic<bool> stop{false};
    std::atomic<long long> bytes{0};
    std::thread th;
    void start(const Arena& ar, size_t blob, int kind, int per_launch = 3) {
        th = std::thread([this, &ar, blob, kind, per_launch] {
            pin(15);
            void* d = nullptr;
            cudaMalloc(&d, (size_t) 64 * blob);
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            unsigned long long* hp = nullptr;   // mapped: the kernel's source pointers and count
            int32_t* hn = nullptr;
            cudaHostAlloc((void**) &hp, 64 * sizeof(unsigned long long), cudaHostAllocMapped);
            cudaHostAlloc((void**) &hn, sizeof(int32_t), cudaHostAllocMapped);
            unsigned long long* dp = nullptr;
            int32_t* dn = nullptr;
            cudaHostGetDevicePointer((void**) &dp, hp, 0);
            cudaHostGetDevicePointer((void**) &dn, hn, 0);
            std::mt19937 rng(7);
            while (!stop.load()) {
                if (kind == 1) {
                    for (int i = 0; i < 16; ++i) {
                        const size_t e = rng() % (unsigned) ar.n;
                        cudaMemcpyAsync((uint8_t*) d + (size_t) (i % 16) * blob, ar.base + e * ar.stride, blob,
                                        cudaMemcpyHostToDevice, s);
                    }
                    cudaStreamSynchronize(s);
                    bytes.fetch_add(16LL * (long long) blob);
                } else {
                    for (int i = 0; i < per_launch; ++i) hp[i] = (unsigned long long) (ar.dev + (size_t) (rng() % (unsigned) ar.n) * ar.stride);
                    *hn = per_launch;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    strata::kernels::fetch_blobs(dp, dn, (uint8_t*) d, (int64_t) blob, 64, (void*) s);
                    cudaStreamSynchronize(s);
                    bytes.fetch_add((long long) per_launch * (long long) blob);
                }
            }
            cudaStreamDestroy(s);
            cudaFree(d);
            cudaFreeHost(hp);
            cudaFreeHost(hn);
        });
    }
    void end() { if (th.joinable()) { stop.store(true); th.join(); } }
};

int cmd_gpu(int kind, double seconds) {
    const cpu::NativeFmt f = fmt_q5();
    Arena ar = make_arena(f, (size_t) 6 << 30);
    for (int per : {1, 3, 8}) {
        if (kind == 1 && per != 1) break;
        GpuLoad g;
        g.start(ar, f.bytes, kind, per);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const long long b0 = g.bytes.load();
        const auto t0 = clk::now();
        std::this_thread::sleep_for(std::chrono::milliseconds((int) (seconds * 1000)));
        const double gbs = (double) (g.bytes.load() - b0) / secs(t0, clk::now()) / 1e9;
        g.end();
        std::printf("gpu kind %d (%s)%s: %.1f GB/s\n", kind, kind == 1 ? "copy engine" : "fetch_blobs kernel, mapped",
                    kind == 2 ? (" " + std::to_string(per) + " blobs a launch").c_str() : "", gbs);
    }
    return 0;
}

int cmd_mix(int threads, int kind) {
    const cpu::NativeFmt f = fmt_q5();
    Arena ar = make_arena(f, (size_t) 6 << 30);
    const size_t per = (size_t) 256 << 20;
    uint8_t* buf = (uint8_t*) mmap(nullptr, per * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, per * 8, MADV_HUGEPAGE);
    std::memset(buf, 1, per * 8);
    auto measure = [&](bool cpu_on, bool gpu_on) {
        CpuReaders r;
        GpuLoad g;
        if (cpu_on) r.start(buf, per, threads);
        if (gpu_on) g.start(ar, f.bytes, kind, 3);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const long long c0 = r.bytes.load(), g0 = g.bytes.load();
        const auto t0 = clk::now();
        std::this_thread::sleep_for(std::chrono::seconds(3));
        const double dt = secs(t0, clk::now());
        const double cg = (double) (r.bytes.load() - c0) / dt / 1e9, gg = (double) (g.bytes.load() - g0) / dt / 1e9;
        if (cpu_on) r.end();
        g.end();
        std::printf("mix: CPU %d threads %s, GPU kind %d %s: CPU %.1f GB/s, GPU %.1f GB/s, sum %.1f GB/s\n", threads,
                    cpu_on ? "on " : "off", kind, gpu_on ? "on " : "off", cg, gg, cg + gg);
    };
    measure(true, false);
    measure(false, true);
    measure(true, true);
    return 0;
}

// ------------------------------------------------------------------------------------------------ the pool
struct Acts {
    std::vector<std::vector<uint8_t>> act, hq;
};

Acts make_acts(const cpu::NativeFmt& f, int nt, int seed) {
    Acts a;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) H), h((size_t) FF);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng);
        for (auto& v : h) v = 0.1f * nd(rng);
        a.act.emplace_back(cpu::kNativeActBytes);
        a.hq.emplace_back(cpu::kNativeHBytes);
        cpu::native_quant_act(f, x.data(), a.act.back().data());
        cpu::native_quant_h(f, h.data(), a.hq.back().data());
    }
    return a;
}

// the decode's token mix per CPU expert (round 1's measurement: 60/27/10/3 % with 1/2/3/4 tokens)
int pick_nt(std::mt19937& rng) {
    const double u = std::uniform_real_distribution<double>(0, 1)(rng);
    return u < 0.60 ? 1 : u < 0.87 ? 2 : u < 0.97 ? 3 : 4;
}

int cmd_pool(int layers, double per_layer, int tasks, int kind) {
    const cpu::NativeFmt f = fmt_q5();
    Arena ar = make_arena(f, (size_t) 6 << 30);
    const int host = cpu::planned_host_core();
    cpu::ExpertPool pool(0, true, true, cpu::PoolAffinity::All, tasks);
    const auto prev = cpu::pin_current_thread(host);
    constexpr int NTOK = 8;
    Acts a = make_acts(f, NTOK, 3);
    std::vector<std::vector<float>> outs(96 * 8, std::vector<float>((size_t) H));
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ud(0, 1);
    std::vector<cpu::ExpertJobMulti> jobs(96);
    GpuLoad g;
    if (kind > 0) g.start(ar, f.bytes, kind, 3);
    for (int warm = 0; warm < 2; ++warm) {
        double total = 0, bytes = 0;
        std::vector<double> per;
        const long long gb0 = g.bytes.load();
        const auto tw0 = clk::now();
        const double g0 = pool.ms_multi_gu, d0 = pool.ms_multi_down;
        for (int l = 0; l < layers; ++l) {
            const int n = (int) per_layer + (ud(rng) < per_layer - (int) per_layer ? 1 : 0);
            int oi = 0;
            for (int e = 0; e < n; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                j.blob = ar.base + (size_t) (rng() % (unsigned) ar.n) * ar.stride;
                j.nt = pick_nt(rng);
                for (int t = 0; t < j.nt; ++t) {
                    j.nact[t] = a.act[(size_t) ((e + t) % NTOK)].data();
                    j.out[t] = outs[(size_t) oi++].data();
                }
            }
            const auto t0 = clk::now();
            pool.run_split_multi_native(f, jobs.data(), n);
            const auto t1 = clk::now();
            total += secs(t0, t1);
            per.push_back(secs(t0, t1) * 1e3);
            bytes += (double) n * (double) f.bytes;
        }
        if (warm == 1) {
            std::sort(per.begin(), per.end());
            std::printf("pool: %d layers, %.2f experts/layer, tasks %d, GPU kind %d: %.3f ms/layer (median %.3f, p10 %.3f, "
                        "p90 %.3f), %.1f GB/s | gate/up %.3f down %.3f ms/layer", layers, per_layer, tasks, kind,
                        total * 1e3 / layers, per[per.size() / 2], per[per.size() / 10], per[per.size() * 9 / 10],
                        bytes / total / 1e9, (pool.ms_multi_gu - g0) / layers, (pool.ms_multi_down - d0) / layers);
            if (kind > 0) std::printf(" | GPU %.1f GB/s", (double) (g.bytes.load() - gb0) / secs(tw0, clk::now()) / 1e9);
            std::printf("\n");
        }
    }
    g.end();
    cpu::restore_thread_affinity(prev);
    return 0;
}

// ------------------------------------------------------------------------------------------------ the MoE step
int cmd_split(int layers, double misses, double frac, int gap_us, double pf_mb, double hit, int pf_threads, int kind) {
    const cpu::NativeFmt f = fmt_q5();
    Arena ar = make_arena(f, (size_t) 6 << 30);
    const int host = cpu::planned_host_core();
    cpu::ExpertPool pool(0, true, true, cpu::PoolAffinity::All, 96);
    const auto prev = cpu::pin_current_thread(host);
    constexpr int NTOK = 8;
    Acts a = make_acts(f, NTOK, 3);
    std::vector<std::vector<float>> outs(96 * 8, std::vector<float>((size_t) H));
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ud(0, 1);
    void* dbuf = nullptr;
    cudaMalloc(&dbuf, (size_t) 64 * f.bytes);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    unsigned long long* hp = nullptr;
    int32_t* hn = nullptr;
    cudaHostAlloc((void**) &hp, 64 * sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc((void**) &hn, sizeof(int32_t), cudaHostAllocMapped);
    unsigned long long* dp = nullptr;
    int32_t* dn = nullptr;
    cudaHostGetDevicePointer((void**) &dp, hp, 0);
    cudaHostGetDevicePointer((void**) &dn, hn, 0);
    // the prefetch threads: wait for a list, read it in 16 KB pieces (shared) until the next layer starts
    std::atomic<int64_t> phase{0};          // odd: a layer runs, even: the gap
    std::atomic<int64_t> list_gen{0};
    std::atomic<uint64_t> claim{0};
    std::mutex mu;
    std::vector<const uint8_t*> plist;
    std::atomic<bool> quit{false};
    std::atomic<long long> pf_bytes{0};
    std::vector<std::thread> pf;
    constexpr size_t kChunk = 16384;
    const size_t per = (f.bytes + kChunk - 1) / kChunk;
    for (int pt = 0; pf_mb > 0 && pt < pf_threads; ++pt)
        pf.emplace_back([&, pt] {
            pin(15 - pt);   // SMT siblings, the last first (the engine's STRATA_HELPER_CPUS=auto)
            int64_t seen = 0;
            volatile uint64_t sink = 0;
            while (!quit.load()) {
                if (list_gen.load() == seen) { _mm_pause(); continue; }
                seen = list_gen.load();
                std::vector<const uint8_t*> l;
                { std::lock_guard<std::mutex> lk(mu); l = plist; }
                while (!quit.load() && (phase.load() & 1)) _mm_pause();   // the layer before still runs
                const int64_t my = phase.load();
                const size_t total = std::min(l.size() * per, (size_t) (pf_mb * 1e6) / kChunk);
                for (;;) {
                    if (phase.load(std::memory_order_relaxed) != my) break;
                    uint64_t cur = claim.load();
                    const uint64_t g = (uint64_t) seen & 0xffffffffull;
                    size_t c;
                    if ((cur >> 32) != g) {
                        if (((g - (cur >> 32)) & 0xffffffffull) > 0x7fffffffull) break;
                        if (!claim.compare_exchange_weak(cur, (g << 32) | 1ull)) continue;
                        c = 0;
                    } else {
                        if ((cur & 0xffffffffull) >= total) break;
                        if (!claim.compare_exchange_weak(cur, cur + 1)) continue;
                        c = (size_t) (cur & 0xffffffffull);
                    }
                    if (c >= total) break;
                    const uint8_t* p = l[c / per] + (c % per) * kChunk;
                    const size_t n = std::min(kChunk, f.bytes - (c % per) * kChunk);
                    uint64_t acc = 0;
                    for (size_t q = 0; q < n; q += 256)
                        acc += *(const volatile uint64_t*) (p + q) ^ *(const volatile uint64_t*) (p + q + 64) ^
                               *(const volatile uint64_t*) (p + q + 128) ^ *(const volatile uint64_t*) (p + q + 192);
                    sink = sink + acc;
                    pf_bytes.fetch_add((long long) n, std::memory_order_relaxed);
                }
            }
        });
    std::vector<cpu::ExpertJobMulti> jobs(96);
    auto make_layer = [&](std::vector<int>& ids) {
        const int m = (int) misses + (ud(rng) < misses - (int) misses ? 1 : 0);
        ids.clear();
        while ((int) ids.size() < m) {
            const int e = (int) (rng() % (unsigned) ar.n);
            if (std::find(ids.begin(), ids.end(), e) == ids.end()) ids.push_back(e);
        }
    };
    std::vector<int> cur, next;
    make_layer(cur);
    double t_layer = 0, t_cpu = 0, t_dma = 0;
    int counted = 0;
    const int warmup = 50;
    for (int l = 0; l < layers + warmup; ++l) {
        make_layer(next);
        const int np = (int) ((cur.size() * (size_t) std::lround(frac * 256)) >> 8);   // the engine's m = nmiss*pcie_num>>8
        const int nc = (int) cur.size() - np;
        phase.fetch_add(1);   // the layer starts
        const auto t0 = clk::now();
        if (np > 0) {
            if (kind == 1) {
                for (int q = 0; q < np; ++q)
                    cudaMemcpyAsync((uint8_t*) dbuf + (size_t) q * f.bytes, ar.base + (size_t) cur[(size_t) (nc + q)] * ar.stride,
                                    f.bytes, cudaMemcpyHostToDevice, s);
            } else {
                for (int q = 0; q < np; ++q) hp[q] = (unsigned long long) (ar.dev + (size_t) cur[(size_t) (nc + q)] * ar.stride);
                *hn = np;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                strata::kernels::fetch_blobs(dp, dn, (uint8_t*) dbuf, (int64_t) f.bytes, 64, (void*) s);
            }
            cudaEventRecord(ev, s);
        }
        int oi = 0;
        for (int e = 0; e < nc; ++e) {
            cpu::ExpertJobMulti& j = jobs[(size_t) e];
            j.blob = ar.base + (size_t) cur[(size_t) e] * ar.stride;
            j.nt = pick_nt(rng);
            for (int t = 0; t < j.nt; ++t) {
                j.nact[t] = a.act[(size_t) ((e + t) % NTOK)].data();
                j.out[t] = outs[(size_t) oi++].data();
            }
        }
        if (nc > 0) pool.run_split_multi_native(f, jobs.data(), nc);
        const auto t1 = clk::now();
        if (np > 0) cudaEventSynchronize(ev);
        const auto t2 = clk::now();
        phase.fetch_add(1);   // the gap
        if (l >= warmup) {
            t_cpu += secs(t0, t1);
            t_dma += np ? secs(t0, t2) : 0;
            t_layer += secs(t0, std::max(t1, t2));
            ++counted;
        }
        if (pf_mb > 0) {   // the next layer's CPU experts in routing order, `hit` of them right
            std::vector<const uint8_t*> l2;
            const int nc2 = (int) next.size() - (int) ((next.size() * (size_t) std::lround(frac * 256)) >> 8);
            for (int e = 0; e < nc2 + 4; ++e) {
                const int id = (e < nc2 && ud(rng) < hit) ? next[(size_t) e] : (int) (rng() % (unsigned) ar.n);
                l2.push_back(ar.base + (size_t) id * ar.stride);
            }
            { std::lock_guard<std::mutex> lk(mu); plist = l2; }
            list_gen.fetch_add(1);
        }
        const auto until = clk::now() + std::chrono::microseconds(gap_us);
        while (clk::now() < until) _mm_pause();
        std::swap(cur, next);
    }
    quit.store(true);
    for (auto& t : pf) t.join();
    std::printf("split: misses %.1f, PCIe share %.2f (kind %d), gap %d us, prefetch %.0f MB on %d thread(s) (hit %.2f): "
                "MoE step %.3f ms/layer (CPU %.3f, PCIe %.3f); prefetched %.1f MB/layer; step+gap %.3f ms\n",
                misses, frac, kind, gap_us, pf_mb, pf_threads, hit, t_layer * 1e3 / counted, t_cpu * 1e3 / counted,
                t_dma * 1e3 / counted, (double) pf_bytes.load() / 1e6 / (layers + warmup),
                t_layer * 1e3 / counted + gap_us / 1e3);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s);
    cudaFree(dbuf);
    cudaFreeHost(hp);
    cudaFreeHost(hn);
    cpu::restore_thread_affinity(prev);
    return 0;
}

// the router lookahead's dot: AVX2 vs AVX512-BF16 on random BF16 weights (48 layers' worth, so DRAM-cold like the engine)
int cmd_router(int nt) {
    const int rows = 512, cols = 2560, L = 48;
    std::vector<uint16_t> w((size_t) L * rows * cols);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto& v : w) { const float f = nd(rng); uint32_t b; std::memcpy(&b, &f, 4); v = (uint16_t) (b >> 16); }
    std::vector<float> x((size_t) nt * cols), o1((size_t) nt * rows), o2((size_t) nt * rows);
    std::normal_distribution<float> nx(0.f, 1.f);
    for (auto& v : x) v = nx(rng);
    double t1 = 0, t2 = 0;
    int same = 0, total = 0;
    for (int it = 0; it < 4; ++it)
        for (int l = 0; l < L; ++l) {
            const uint16_t* wl = w.data() + (size_t) l * rows * cols;
            auto a = clk::now();
            cpu::bf16_rows_dot_multi(wl, rows, cols, x.data(), nt, o1.data());
            auto b = clk::now();
            cpu::bf16_rows_dot_multi_avx512bf16(wl, rows, cols, x.data(), nt, o2.data());
            auto c = clk::now();
            if (it > 0) { t1 += secs(a, b); t2 += secs(b, c); }
            for (int t = 0; t < nt; ++t) {   // top-10 overlap
                std::vector<int> i1(rows), i2(rows);
                for (int r = 0; r < rows; ++r) i1[r] = i2[r] = r;
                const float* p1 = o1.data() + (size_t) t * rows; const float* p2 = o2.data() + (size_t) t * rows;
                std::partial_sort(i1.begin(), i1.begin() + 10, i1.end(), [&](int u, int v) { return p1[u] > p1[v]; });
                std::partial_sort(i2.begin(), i2.begin() + 10, i2.end(), [&](int u, int v) { return p2[u] > p2[v]; });
                for (int k = 0; k < 10; ++k) same += std::find(i2.begin(), i2.begin() + 10, i1[k]) != i2.begin() + 10;
                total += 10;
            }
        }
    std::printf("router nt %d: AVX2 %.3f ms/layer, AVX512-BF16 %.3f ms/layer (bf16 ok %d); top-10 agreement %.2f%%\n", nt,
                t1 * 1e3 / (3 * L), t2 * 1e3 / (3 * L), (int) cpu::cpu_avx512bf16_ok(), 100.0 * same / total);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cpu_init();
    const std::string cmd = argc > 1 ? argv[1] : "";
    auto I = [&](int i, int d) { return argc > i ? std::atoi(argv[i]) : d; };
    auto D = [&](int i, double d) { return argc > i ? std::atof(argv[i]) : d; };
    if (cmd == "bw") return cmd_bw(argc - 2, argv + 2);
    if (cmd == "router") { pin(9); return cmd_router(I(2, 3)); }
    if (cmd == "gpu") return cmd_gpu(I(2, 2), D(3, 3.0));
    if (cmd == "mix") return cmd_mix(I(2, 8), I(3, 2));
    if (cmd == "pool") return cmd_pool(I(2, 480), D(3, 8.1), I(4, 96), I(5, 0));
    if (cmd == "split")
        return cmd_split(I(2, 480), D(3, 11.0), D(4, 0.30), I(5, 280), D(6, 0.0), D(7, 1.0), I(8, 1), I(9, 2));
    std::fprintf(stderr, "usage: r6mem_bench bw|gpu|mix|pool|split ... (see the header)\n");
    return 2;
}
