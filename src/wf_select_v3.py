import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from trading_env import obs_stats_from
from exposure_env import (
    RegimeXExposureEnv, PooledExposureEnv, blind_gamma_from, gamma_config,
)
from strategies import rollout, ppo_policy
from metrics import sharpe, total_return, max_drawdown, REGIMES

# ============================================================
# REGIMEX - WALK-FORWARD HYPER-PARAMETER SELECTION FOR v3
# ============================================================
#
# One job = one (gamma_base, agent kind, test year, seed):
#   train on all five stocks up to 31 Dec of (year-1), evaluate on `year`.
# Used ONLY on years before the final test window (2019-2023), so choosing
# gamma from these results never touches the 2024-06+ test set.
# Results are written to results/v3_select/<job>.json and aggregated by
# wf_select_report.py.
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]

parser = argparse.ArgumentParser()
parser.add_argument("--gamma-base", type=float, required=True)
parser.add_argument("--blind", action="store_true")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timesteps", type=int, default=300_000)
args = parser.parse_args()

OUT = "results/v3_select"
os.makedirs(OUT, exist_ok=True)
kind = "blind" if args.blind else "adaptive"
job = f"g{args.gamma_base}_{kind}_{args.year}_s{args.seed}"
path = f"{OUT}/{job}.json"
if os.path.exists(path):
    print("already done", job)
    sys.exit(0)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]).sort_values("Date")
        for t in TICKERS}
Y = args.year
train_dfs = [d[d["Date"] < f"{Y}-01-01"].reset_index(drop=True) for d in data.values()]
test_dfs = {t: d[(d["Date"] >= f"{Y}-01-01") & (d["Date"] < f"{Y + 1}-01-01")].reset_index(drop=True)
            for t, d in data.items()}

cfg = gamma_config(args.gamma_base)
bg = blind_gamma_from(train_dfs, cfg)
stats = obs_stats_from(pd.concat(train_dfs))
env_kw = dict(gamma_config=cfg, blind_gamma=bg, scaled_obs=True, obs_stats=stats)

env = DummyVecEnv([lambda: PooledExposureEnv(train_dfs, regime_adaptive=not args.blind, **env_kw)])
model = PPO(
    "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, ent_coef=0.01, clip_range=0.2,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    seed=args.seed, verbose=0,
)
model.learn(total_timesteps=args.timesteps)

rets, expos, regs = [], [], []
for t in TICKERS:
    lg = rollout(test_dfs[t], ppo_policy(model), regime_adaptive=not args.blind,
                 env_cls=RegimeXExposureEnv, **env_kw)
    rets.append((lg["value"] / lg["prev_value"] - 1).to_numpy())
    expos.append(lg["exposure"].to_numpy())
    regs.append(lg["regime"].to_numpy())

port = np.mean(rets, axis=0)
reg_all, e_all = np.concatenate(regs), np.concatenate(expos)
res = {
    "gamma_base": args.gamma_base, "kind": kind, "year": Y, "seed": args.seed,
    "blind_gamma": bg,
    "port_sharpe": sharpe(port), "port_return": total_return(port), "port_maxdd": max_drawdown(port),
    "exposure": {r: (float(e_all[reg_all == r].mean()) if (reg_all == r).any() else None) for r in REGIMES},
    "n_days": {r: int((reg_all == r).sum()) for r in REGIMES},
}
json.dump(res, open(path, "w"), indent=1)
print(job, "sharpe", round(res["port_sharpe"], 3))
