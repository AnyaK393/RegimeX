import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from exposure_env import (
    RegimeXExposureEnv, PooledExposureEnv, blind_gamma_from, gamma_config,
    EXPOSURES, DEFAULT_GAMMA,
)
from strategies import rollout

# ============================================================
# REGIMEX - v3 EXPOSURE ENVIRONMENT SMOKE TEST
# ============================================================

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    print(f"  [{'PASS' if cond else 'FAIL'}]  {name}" + (f"  -- {detail}" if detail else ""))
    passed += bool(cond)
    failed += (not cond)


df = pd.read_csv("data/processed/RELIANCE_regimes.csv", parse_dates=["Date"])
train, _, test = get_splits(df, verbose=False)
test = test.reset_index(drop=True)
stats = obs_stats_from(train)
kw = dict(env_cls=RegimeXExposureEnv, scaled_obs=True, obs_stats=stats, blind_gamma=7.0)

print("\n[1] Gamma configuration")
cfg = gamma_config(3.0)
check("default scale reproduces DEFAULT_GAMMA", cfg == DEFAULT_GAMMA)
cfg15 = gamma_config(1.5)
check("scaling keeps 1:3:6 ratio",
      np.isclose(cfg15["Weak_Bear"] / cfg15["Normal_Market"], 3) and
      np.isclose(cfg15["High_Volatility"] / cfg15["Normal_Market"], 6))
bg = blind_gamma_from([train])
check("blind gamma lies between min and max regime gamma",
      min(DEFAULT_GAMMA.values()) <= bg <= max(DEFAULT_GAMMA.values()), f"{bg:.3f}")

print("\n[2] Mechanics")
hold = rollout(test, lambda o, t, e: 4, **kw)
check("always-100% is invested ~fully", hold["exposure"].iloc[1:].mean() > 0.95)
check("always-100% makes very few trades", (hold["trade_value"] > 0).sum() <= 3,
      f"trades={(hold['trade_value'] > 0).sum()}")
cash = rollout(test, lambda o, t, e: 0, **kw)
check("always-cash never trades and never changes value", (cash["trade_value"] == 0).all()
      and np.allclose(cash["value"], 100000))
check("always-cash reward is exactly zero", np.allclose(cash["reward"], 0))

rng = np.random.default_rng(0)
rand = rollout(test, lambda o, t, e: int(rng.integers(len(EXPOSURES))), **kw)
check("random policy: no NaN/inf reward", np.isfinite(rand["reward"]).all())
check("random policy: exposure within [0, 1]", rand["exposure"].between(-1e-9, 1 + 1e-9).all())
check("random policy: portfolio value stays positive", (rand["value"] > 0).all())

print("\n[3] Reward: regime-adaptive vs regime-blind")
ad = rollout(test, lambda o, t, e: 4, regime_adaptive=True, **kw)
bl = rollout(test, lambda o, t, e: 4, regime_adaptive=False, **kw)
check("same P&L in both modes", np.allclose(ad["value"], bl["value"]))
check("rewards differ between modes", not np.allclose(ad["reward"], bl["reward"]))
hv = ad["regime"] == "High_Volatility"
if hv.any():
    check("HV days are penalised more under adaptive than blind (gamma 18 > 7)",
          (ad.loc[hv, "reward"] < bl.loc[hv, "reward"]).mean() > 0.9)
nm = ad["regime"] == "Normal_Market"
check("Normal days are penalised less under adaptive than blind (gamma 3 < 7)",
      (ad.loc[nm, "reward"] >= bl.loc[nm, "reward"]).mean() > 0.9)

print("\n[4] Observation")
env = RegimeXExposureEnv(df=test, scaled_obs=True, obs_stats=stats)
obs, _ = env.reset()
check("observation has 7 dims", obs.shape == (7,) and env.observation_space.shape == (7,))
check("regime one-hot sums to 1", np.isclose(obs[3:6].sum(), 1.0))
obs, *_ = env.step(4)
check("exposure feature reflects the position after a BUY", obs[6] > 0.9, f"{obs[6]:.3f}")

print("\n[5] Pooled environment draws from several stocks")
d2 = pd.read_csv("data/processed/TCS_regimes.csv", parse_dates=["Date"])
pe = PooledExposureEnv([train, get_splits(d2, verbose=False)[0]], scaled_obs=True, obs_stats=stats)
seen = set()
for i in range(30):
    pe.reset(seed=i)
    seen.add(id(pe.df))
check("both stocks are sampled across resets", len(seen) == 2, f"distinct dataframes seen: {len(seen)}")

print("\n" + "=" * 60)
print(f"RESULTS:  {passed} passed  |  {failed} failed")
print("=" * 60)
sys.exit(1 if failed else 0)
