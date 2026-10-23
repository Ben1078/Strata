// include/strata/spec/cost_policy.hpp - q5opt r6-spec: a cost-aware MTP window size (opt-in, --spec-policy cost).
//
// The default rule (--spec-min-p P) puts draft j into the verify window while EVERY draft up to it has a draft-layer
// probability >= P.  That ignores what a window row costs on this machine and how the drafts' probabilities add up:
// a run of 0.86s passes while a 0.84 after three 0.99s stops the window.
//
// This policy includes draft j while the CALIBRATED probability that drafts 1..j are all accepted,
//     P_j = prod_{i<=j} q(p_i, depth i)      (q: measured acceptance of drafts with that probability, learned online)
// is at least the cost of the row it adds, in tokens:
//     P_j >= theta = R* x dV,   R* = the decode rate in tokens/ms (decayed), dV = the verify ms per extra row (an
//     online least-squares fit of the verify time on the window size, decayed).
// That is the renewal-reward rule for the long-run tokens per ms (a row pays when the tokens it adds on average are
// worth more than the time it costs at the current rate).  Drafting stops when even a confident next draft could not
// reach theta.  Only the window's size changes: every token is still the verify window's own, so greedy output is the
// plain decode's (up to the window-shape rounding every window size already has) and sampled output keeps its law.
#pragma once

#include <array>
#include <string>

namespace strata::spec {

class CostPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBins = 20;     // draft probability bins of 0.05
    static constexpr int kDepths = 3;    // depth 0, 1, 2+

    CostPolicy();

    /// Calibrated acceptance of a draft with draft-layer probability p at depth d (0 = the window's first draft).
    double cal(float p, int depth, bool sampled) const;
    /// The inclusion threshold on P_j (tokens): R* x dV (+ margin), clamped to [lo, hi].
    double threshold() const;
    /// Keep drafting after a draft whose cumulative calibrated probability is `pc` (the next would be `depth`)?
    bool keep_drafting(double pc, int depth, bool sampled) const;
    /// The window size for `n` drafts with draft-layer probabilities pr[0..n-1]: 1 + the longest prefix whose P_j
    /// stays >= threshold, at most `cap` (>= 1).
    int window(const float* pr, int n, int cap, bool sampled);   // (counts starved windows: exploration)

    /// After a verified MTP window: its size, the drafts accepted, their probabilities, the verify ms (the window's
    /// run alone), the whole round's ms and the tokens it committed.
    void observe(int T, int accepted, const float* pr, bool sampled, double verify_ms, double round_ms, int tokens);
    /// A round that is not an MTP window (lookup, chained, first): its time and tokens still count for the rate.
    void observe_rate(double round_ms, int tokens);
    /// The draft chain: `steps` draft steps took `ms`.
    void observe_draft(int steps, double ms);

    /// Overrides (STRATA_SPEC_TUNE_FILE / --spec-policy-* for A/B): a fixed theta, a fixed dV, a margin on theta.
    void set_fixed_theta(double t) { fixed_theta_ = t; }
    void set_fixed_dv(double v) { fixed_dv_ = v; }
    void set_margin(double m) { margin_ = m; }
    void set_raw(bool raw) { raw_ = raw; }   // q = p itself (no calibration), for A/B

    double rate() const;        // R*, tokens per ms
    double dv() const;          // verify ms per extra row
    double draft_step_ms() const;
    std::string summary() const;

private:
    struct Cal { double ok = 0, n = 0; };
    std::array<std::array<std::array<Cal, kBins>, kDepths>, 2> cal_{};   // [sampled][depth][bin]
    // verify ms on T: decayed sums for a least-squares line, seeded with a prior
    double s0_ = 0, s1_ = 0, s2_ = 0, sy_ = 0, sxy_ = 0;
    double dr_n_ = 0, dr_ms_ = 0;            // draft: decayed steps and ms
    double r_tok_ = 0, r_ms_ = 0;            // rate: decayed tokens and ms
    double fixed_theta_ = -1, fixed_dv_ = -1, margin_ = 0;
    bool raw_ = false;
    long long windows_ = 0, explored_ = 0;
    int starved_ = 0;
    void fit(double& a, double& b) const;
};

}  // namespace strata::spec
