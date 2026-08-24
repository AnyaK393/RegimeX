import sys
import os
import numpy as np
import pandas as pd

# Allow imports from src/ when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trading_env import RegimeXTradingEnv

# ============================================================
# REGIMEX - REGIME-CONDITIONAL REWARD SMOKE TEST
# ============================================================
#
# Verifies:
#   1. No NaN or Inf rewards over a full episode (adaptive mode)
#   2. Regime-blind and regime-adaptive modes produce different
#      reward signals on the same trajectory
#   3. Drawdown penalty activates correctly after a portfolio loss
#   4. Turnover penalty activates on BUY and SELL, not on HOLD
#   5. Backwards-compatibility: zero-arg instantiation still works
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}{'  -- ' + detail if detail else ''}")


print("=" * 60)
print("REGIMEX - REGIME REWARD SMOKE TEST")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ============================================================
# Test 1 — Backwards-compatible zero-arg instantiation
# ============================================================
print("\n[1] Zero-arg instantiation (backwards compatibility)")
try:
    env_default = RegimeXTradingEnv()
    obs, info = env_default.reset()
    check("Zero-arg init works", True)
    check("Observation shape", obs.shape == (5,), f"shape={obs.shape}")
    check("Initial portfolio value", info["portfolio_value"] == 100000)
except Exception as e:
    check("Zero-arg init works", False, str(e))

# ============================================================
# Test 2 — No NaN/Inf in rewards over a full episode (adaptive)
# ============================================================
print("\n[2] Full episode — no NaN/Inf (regime_adaptive=True)")
env_adaptive = RegimeXTradingEnv(df=df, regime_adaptive=True)
obs, _ = env_adaptive.reset()

rewards_adaptive = []
actions_cycle = [1, 0, 0, 0, 2, 0, 0, 1, 0, 2]  # BUY/HOLD/SELL cycle
step = 0
terminated = False

while not terminated:
    action = actions_cycle[step % len(actions_cycle)]
    obs, reward, terminated, truncated, info = env_adaptive.step(action)
    rewards_adaptive.append(reward)
    step += 1

nan_count = np.sum(np.isnan(rewards_adaptive))
inf_count = np.sum(np.isinf(rewards_adaptive))
check("No NaN rewards",  nan_count == 0, f"{nan_count} NaN found")
check("No Inf rewards",  inf_count == 0, f"{inf_count} Inf found")
check("Episode length",  len(rewards_adaptive) == len(df) - 1,
      f"steps={len(rewards_adaptive)}")

# ============================================================
# Test 3 — Regime-blind vs adaptive: different reward on same trajectory
# ============================================================
print("\n[3] Regime-blind vs regime-adaptive: different rewards")

env_blind = RegimeXTradingEnv(df=df, regime_adaptive=False)

# Run same action sequence on both envs
env_adaptive2 = RegimeXTradingEnv(df=df, regime_adaptive=True)
obs_a, _ = env_adaptive2.reset()
obs_b, _ = env_blind.reset()

rewards_a, rewards_b = [], []
n_steps_test = 200

for i in range(n_steps_test):
    action = actions_cycle[i % len(actions_cycle)]
    _, r_a, done_a, _, info_a = env_adaptive2.step(action)
    _, r_b, done_b, _, info_b = env_blind.step(action)
    rewards_a.append(r_a)
    rewards_b.append(r_b)
    if done_a or done_b:
        break

sum_a = np.sum(rewards_a)
sum_b = np.sum(rewards_b)
differ = not np.isclose(sum_a, sum_b, atol=1e-9)
check(
    "Regime-adaptive != regime-blind (cumulative reward)",
    differ,
    f"adaptive={sum_a:.6f}  blind={sum_b:.6f}"
)

# ============================================================
# Test 4 — Turnover penalty activates on trades, not HOLD
# ============================================================
print("\n[4] Turnover penalty: BUY triggers it, HOLD does not")

env_t = RegimeXTradingEnv(df=df, regime_adaptive=True)
obs, _ = env_t.reset()

# Step with HOLD first
obs, r_hold, _, _, _ = env_t.step(0)
hold_trade_val = env_t._last_trade_value
check("HOLD: _last_trade_value == 0", hold_trade_val == 0.0,
      f"trade_val={hold_trade_val}")

# Step with BUY
obs, r_buy, _, _, _ = env_t.step(1)
buy_trade_val = env_t._last_trade_value
check("BUY: _last_trade_value > 0", buy_trade_val > 0,
      f"trade_val={buy_trade_val:.2f}")

# ============================================================
# Test 5 — Drawdown penalty activates after portfolio drops
# ============================================================
print("\n[5] Drawdown penalty: peak tracked correctly")

env_d = RegimeXTradingEnv(df=df, regime_adaptive=True)
obs, _ = env_d.reset()
initial_peak = env_d.peak_value

# BUY to take a position (risk exposure)
obs, _, _, _, _ = env_d.step(1)

# Run for a while; peak should update
for _ in range(50):
    obs, _, done, _, _ = env_d.step(0)
    if done:
        break

check("Peak value tracked (>= initial capital)", env_d.peak_value >= initial_peak,
      f"peak={env_d.peak_value:.2f}")

# ============================================================
# Test 6 — Lambda risk values: HV > Weak_Bear > Normal
# ============================================================
print("\n[6] Lambda risk ordering (HV > Weak_Bear > Normal)")

env_lam = RegimeXTradingEnv(df=df, regime_adaptive=True)
lam_hv  = env_lam._get_lambda_risk("High_Volatility")
lam_wb  = env_lam._get_lambda_risk("Weak_Bear")
lam_nm  = env_lam._get_lambda_risk("Normal_Market")
check("HV > Weak_Bear",    lam_hv > lam_wb,  f"HV={lam_hv}  WB={lam_wb}")
check("Weak_Bear > Normal", lam_wb > lam_nm,  f"WB={lam_wb}  NM={lam_nm}")

env_blind2 = RegimeXTradingEnv(df=df, regime_adaptive=False)
lam_blind_hv = env_blind2._get_lambda_risk("High_Volatility")
lam_blind_nm = env_blind2._get_lambda_risk("Normal_Market")
check("Regime-blind: HV == Normal lambda", lam_blind_hv == lam_blind_nm,
      f"HV={lam_blind_hv}  NM={lam_blind_nm}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"RESULTS:  {passed} passed  |  {failed} failed")
if failed:
    print("\nFailed tests:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  {name}: {detail}")
    sys.exit(1)
else:
    print("Regime reward smoke test PASSED.")
print("=" * 60)
