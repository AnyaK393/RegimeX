import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from metrics import seed_permutation_test, hierarchical_bootstrap, paired_bootstrap, sharpe

# ============================================================
# REGIMEX - SEED-LEVEL SIGNIFICANCE TESTS (synthetic checks)
# ============================================================

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    print(f"  [{'PASS' if cond else 'FAIL'}]  {name}" + (f"  -- {detail}" if detail else ""))
    passed += bool(cond)
    failed += (not cond)


rng = np.random.default_rng(0)

print("\n1. seed_permutation_test")
a, b = rng.normal(0, 1, 10), rng.normal(0, 1, 10)
r = seed_permutation_test(a, b, 5000)
check("no effect -> not significant", r["p_value"] > 0.05, f"p = {r['p_value']:.3f}")
check("CI contains the point estimate", r["ci_low"] <= r["diff"] <= r["ci_high"])
r = seed_permutation_test(a + 2.0, b, 5000)
check("large effect -> significant", r["p_value"] < 0.01, f"p = {r['p_value']:.4f}")
r = seed_permutation_test([1.0], [0.0])
check("fewer than 2 seeds -> NaN", np.isnan(r["p_value"]))

# Null calibration: p < 0.05 in about 5 % of repeated null experiments
ps = [seed_permutation_test(rng.normal(0, 1, 10), rng.normal(0, 1, 10), 999, seed=i)["p_value"]
      for i in range(300)]
rate = np.mean(np.array(ps) < 0.05)
check("false-positive rate near 5 % under H0", 0.02 <= rate <= 0.09, f"rate = {rate:.3f}")

print("\n2. hierarchical_bootstrap vs day-level paired_bootstrap")
# Two arms with the SAME expected return but strong seed-specific drift:
# each seed's series has its own mean. A day-level test on the seed-averaged
# series cannot see that the arm means are driven by which seeds were drawn.
T, K = 400, 3
market = rng.normal(0, 0.01, (1, T))          # same market days for both arms
mk = lambda: market + rng.normal(0, 4e-4, (K, 1)) + rng.normal(0, 2e-3, (K, T))
A, B = mk(), mk()
day = paired_bootstrap(A.mean(0), B.mean(0), lambda x: x.mean() * 1e4, 2000, 10)
hier = hierarchical_bootstrap(A, B, lambda x: x.mean() * 1e4, 2000, 10)
w_day, w_hier = day["ci_high"] - day["ci_low"], hier["ci_high"] - hier["ci_low"]
check("hierarchical CI is wider when seeds disagree", w_hier > 1.5 * w_day,
      f"width day {w_day:.2f} bp vs hierarchical {w_hier:.2f} bp")
check("same point estimate as the day-level test", np.isclose(day["diff"], hier["diff"]))

same = rng.normal(5e-4, 0.01, (1, T))
r = hierarchical_bootstrap(np.repeat(same, 4, 0), np.repeat(same, 4, 0), sharpe, 500)
check("identical arms -> zero difference", abs(r["diff"]) < 1e-12 and r["p_value"] == 1.0)

print("\n" + "=" * 60)
print(f"RESULTS:  {passed} passed  |  {failed} failed")
print("=" * 60)
sys.exit(1 if failed else 0)
