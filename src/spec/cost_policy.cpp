// src/spec/cost_policy.cpp - see include/strata/spec/cost_policy.hpp.
#include "strata/spec/cost_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::spec {
namespace {

// Draft-probability bins: finer at the top, where most drafts are.  Prior acceptance per bin from the Oct 14
// full-window logs of this model's MTP head (runs/spec-sw2, greedy, std+agent prompts, ~2,500 tested drafts): the
// head is over-confident below ~0.95 (0.85-0.95: ~0.65 accepted).  Learned online from there.
constexpr float kEdge[CostPolicy::kBins] = {0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.75f, 0.80f, 0.85f,
                                            0.88f, 0.91f, 0.93f, 0.95f, 0.97f, 0.98f, 0.99f, 0.995f, 0.998f, 2.0f};
constexpr double kPriorQ[CostPolicy::kBins] = {0.05, 0.20, 0.25, 0.35, 0.38, 0.46, 0.55, 0.62, 0.68, 0.72,
                                               0.80, 0.80, 0.85, 0.86, 0.86, 0.90, 0.90, 0.94, 0.97, 0.99};
// (re-measured Oct 22 on this engine, runs/r6-spec-cal: 1,863 full windows, depth 0)
constexpr double kPriorN = 8.0;          // worth this many observations per bin
constexpr double kCalDecay = 0.9995;     // per WINDOW, every bin (memory ~2,000 windows): evidence a starved bin
                                         // no longer gets (no draft of it enters a window) fades back to the prior
constexpr int kExplore = 16;             // after this many windows in a row without a draft, one gets in anyway
constexpr double kCostDecay = 0.997;     // per window (memory ~330 windows)
constexpr double kRateDecay = 0.99;      // per round (memory ~100 rounds)
// verify-time prior: 26 ms + 8 ms per row, worth 6 windows at T = 1 and 6 at T = 4
constexpr double kPriorA = 26.0, kPriorB = 8.0, kPriorW = 3.0;   // never decayed: regularizes the fit
constexpr double kThetaLo = 0.15, kThetaHi = 0.9;

int bin_of(float p) {
    int b = 0;
    while (b < CostPolicy::kBins - 1 && p >= kEdge[b]) ++b;
    return b;
}

}  // namespace

CostPolicy::CostPolicy() {
    dr_n_ = 4.0; dr_ms_ = 4.0 * 1.2;          // prior: ~1.2 ms per draft step (round overhead included)
    r_tok_ = 5.0; r_ms_ = 5.0 * 21.0;         // prior: ~48 tok/s
}

double CostPolicy::cal(float p, int depth, bool sampled) const {
    if (raw_) return std::clamp((double) p, 0.0, 1.0);
    const int b = bin_of(p), d = std::clamp(depth, 0, kDepths - 1);
    const Cal& c = cal_[sampled ? 1 : 0][(size_t) d][(size_t) b];
    return (c.ok + kPriorN * kPriorQ[b]) / (c.n + kPriorN);
}

double CostPolicy::rate() const { return r_tok_ / std::max(1e-9, r_ms_); }

// the decayed sums plus the prior's pseudo-windows at T = 1 and T = 4 (constant weight: with windows of one size only,
// e.g. a stretch of T = 1, the slope stays defined and drifts towards the prior instead of becoming noise)
void CostPolicy::fit(double& a, double& b) const {
    double S0 = s0_, S1 = s1_, S2 = s2_, Sy = sy_, Sxy = sxy_;
    for (int t : {1, 4}) {
        const double y = kPriorA + kPriorB * t;
        S0 += kPriorW; S1 += kPriorW * t; S2 += kPriorW * t * t; Sy += kPriorW * y; Sxy += kPriorW * t * y;
    }
    const double den = S0 * S2 - S1 * S1;
    b = den > 1e-9 ? (S0 * Sxy - S1 * Sy) / den : kPriorB;
    b = std::clamp(b, 1.0, 40.0);
    a = (Sy - b * S1) / S0;
}

double CostPolicy::dv() const {
    if (fixed_dv_ > 0) return fixed_dv_;
    double a, b;
    fit(a, b);
    return b;
}

double CostPolicy::draft_step_ms() const { return dr_ms_ / std::max(1e-9, dr_n_); }

double CostPolicy::threshold() const {
    if (fixed_theta_ >= 0) return fixed_theta_;
    return std::clamp(rate() * dv() + margin_, kThetaLo, kThetaHi);
}

bool CostPolicy::keep_drafting(double pc, int depth, bool sampled) const {
    // the next draft can only enter the window if pc x (a confident draft's acceptance) reaches theta; its own step
    // costs draft_step_ms at the current rate
    const double best = cal(0.999f, depth, sampled);
    return pc * best >= threshold() + rate() * draft_step_ms();
}

int CostPolicy::window(const float* pr, int n, int cap, bool sampled) {
    const double th = threshold();
    double pc = 1.0;
    int T = 1;
    for (int j = 0; j < n && T < cap; ++j) {
        pc *= cal(pr[j], j, sampled);
        if (pc < th) break;
        ++T;
    }
    // exploration: a draft is only learned from when it enters a window, so a table that turned pessimistic (a hard
    // stretch) could keep every draft out and never learn otherwise.  Every kExplore-th window in a row that would
    // hold no draft takes its first one if the head gave it >= 0.5 (one row, rarely).
    if (T == 1 && n > 0 && cap >= 2) {
        if (++starved_ >= kExplore && pr[0] >= 0.5f) { T = 2; starved_ = 0; ++explored_; }
    } else {
        starved_ = 0;
    }
    return T;
}

void CostPolicy::observe(int T, int accepted, const float* pr, bool sampled, double verify_ms, double round_ms,
                         int tokens) {
    ++windows_;
    for (auto& tabs : cal_[sampled ? 1 : 0])
        for (Cal& c : tabs) { c.ok *= kCalDecay; c.n *= kCalDecay; }
    // acceptance: draft i was tested if every draft before it was accepted
    for (int i = 0; i < T - 1 && i <= accepted; ++i) {
        const int b = bin_of(pr[i]), d = std::min(i, kDepths - 1);
        auto& tab = cal_[sampled ? 1 : 0][(size_t) d];
        tab[(size_t) b].n += 1.0;
        tab[(size_t) b].ok += i < accepted ? 1.0 : 0.0;
    }
    if (verify_ms > 0 && verify_ms < 2000) {
        s0_ *= kCostDecay; s1_ *= kCostDecay; s2_ *= kCostDecay; sy_ *= kCostDecay; sxy_ *= kCostDecay;
        s0_ += 1; s1_ += T; s2_ += (double) T * T; sy_ += verify_ms; sxy_ += T * verify_ms;
    }
    observe_rate(round_ms, tokens);
}

void CostPolicy::observe_rate(double round_ms, int tokens) {
    if (round_ms <= 0 || round_ms > 5000 || tokens <= 0) return;
    r_tok_ = kRateDecay * r_tok_ + tokens;
    r_ms_ = kRateDecay * r_ms_ + round_ms;
}

void CostPolicy::observe_draft(int steps, double ms) {
    if (steps <= 0 || ms <= 0 || ms > 1000) return;
    dr_n_ = 0.98 * dr_n_ + steps;
    dr_ms_ = 0.98 * dr_ms_ + ms;
}

std::string CostPolicy::summary() const {
    char b[320];
    double fa, fb;
    fit(fa, fb);
    std::snprintf(b, sizeof b, "windows=%lld explored=%lld rate=%.1f tok/s dV=%.2f ms/row (fit %.1f + %.2f T) "
                               "draft=%.2f ms/step theta=%.3f q(.999,d0)=%.2f q(.96,d0)=%.2f q(.90,d0)=%.2f q(.999,d2+)=%.2f",
                  windows_, explored_, 1000.0 * rate(), dv(), fa, fb, draft_step_ms(), threshold(),
                  cal(0.999f, 0, false), cal(0.96f, 0, false), cal(0.90f, 0, false), cal(0.999f, 2, false));
    return b;
}

}  // namespace strata::spec
