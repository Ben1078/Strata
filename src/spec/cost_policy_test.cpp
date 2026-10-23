// src/spec/cost_policy_test.cpp - q5opt r6-spec CostPolicy: does the window rule follow the cost and the calibration?
//
//   1. a cheap row (small dV) lets more drafts in than an expensive one, for the same probabilities;
//   2. the window never exceeds its cap and is >= 1;
//   3. drafts the head is sure of but that are always rejected stop entering windows (calibration learns);
//   4. the learned dV follows the observed verify times (a + b T);
//   5. keep_drafting stops once even a confident next draft cannot reach theta.
#include "strata/spec/cost_policy.hpp"

#include <cstdio>

using strata::spec::CostPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("cost_policy_test\n");
    const float pr[7] = {0.999f, 0.999f, 0.97f, 0.96f, 0.93f, 0.9f, 0.85f};
    {
        CostPolicy cheap, dear;
        cheap.set_fixed_dv(3.0);
        dear.set_fixed_dv(20.0);
        const int tc = cheap.window(pr, 7, 8, false), td = dear.window(pr, 7, 8, false);
        std::printf("    cheap rows: T=%d (theta %.3f), dear rows: T=%d (theta %.3f)\n", tc, cheap.threshold(), td,
                    dear.threshold());
        check(tc > td, "cheaper rows: a longer window");
        check(tc <= 8 && td >= 1, "window within [1, cap]");
        check(cheap.window(pr, 7, 3, false) <= 3, "cap respected");
        check(cheap.window(pr, 0, 8, false) == 1, "no drafts: T = 1");
    }
    {
        CostPolicy p;
        p.set_fixed_dv(5.0);
        const int before = p.window(pr, 7, 8, false);
        // the head's 0.999 drafts at depth 0 rejected every time
        for (int i = 0; i < 400; ++i) p.observe(3, 0, pr, false, 40.0, 45.0, 1);
        const int after = p.window(pr, 7, 8, false);
        std::printf("    before %d, after rejections %d (q(.999,d0) = %.2f)\n", before, after, p.cal(0.999f, 0, false));
        check(after < before && after == 1, "always-rejected confident drafts stop entering");
        int explored = 0;
        for (int i = 0; i < 64; ++i) explored += p.window(pr, 7, 8, false) > 1;
        check(explored >= 3 && explored <= 5, "starved: every 16th window still tests a draft (exploration)");
        const double q0 = p.cal(0.999f, 0, false);
        for (int i = 0; i < 3000; ++i) p.observe(1, 0, pr, false, 30.0, 35.0, 1);   // windows without drafts
        const double q1 = p.cal(0.999f, 0, false);
        for (int i = 0; i < 12000; ++i) p.observe(1, 0, pr, false, 30.0, 35.0, 1);
        const double q2 = p.cal(0.999f, 0, false);
        std::printf("    starved bin q: %.2f -> %.2f (3K windows) -> %.2f (15K windows)\n", q0, q1, q2);
        check(q1 > q0 && q2 > 0.9, "starved evidence fades back to the prior (backstop to exploration)");
        check(p.cal(0.999f, 0, true) > 0.9, "sampled table untouched by greedy windows");
    }
    {
        CostPolicy p;
        for (int i = 0; i < 2000; ++i) {
            const int T = 1 + i % 6;
            p.observe(T, T - 1, pr, false, 15.0 + 11.0 * T, 20.0 + 11.0 * T, T);
        }
        std::printf("    learned dV %.2f ms/row (true 11), rate %.1f tok/s, theta %.3f\n", p.dv(), 1000.0 * p.rate(),
                    p.threshold());
        check(p.dv() > 10.0 && p.dv() < 12.0, "dV follows the observed verify times");
        CostPolicy q;
        for (int i = 0; i < 3000; ++i) q.observe(1, 0, pr, false, 30.0, 35.0, 1);   // only T = 1: slope undefined
        std::printf("    only T=1 windows: dV %.2f (prior 8)\n", q.dv());
        check(q.dv() > 4.0 && q.dv() < 12.0, "one window size only: dV stays near the prior");
        check(!p.keep_drafting(0.1, 3, false), "keep_drafting: stops when theta is out of reach");
        check(p.keep_drafting(0.99, 1, false), "keep_drafting: continues on a sure chain");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
