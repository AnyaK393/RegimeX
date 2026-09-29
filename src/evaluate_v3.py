import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from exposure_env import RegimeXExposureEnv, blind_gamma_from, gamma_config, EXPOSURES
from strategies import rollout, ppo_policy
from metrics import (
    summarize, daily_returns, paired_bootstrap, sharpe, total_return,
    max_drawdown, REGIMES,
)

# ============================================================
# REGIMEX - v3 (POSITION-SIZING) EVALUATION
# ============================================================
#
# Evaluates the v3 agents (train_pooled.py --exposure) on every stock's
# held-out split against:
#   Buy & Hold        - 100% exposure throughout
#   Constant 50%      - naive fixed de-risking
#   Vol-target        - classic rule: exposure = clip(1% / rolling vol, 0, 1)
#
# Headline questions:
#   1. Does the regime-adaptive agent take a DIFFERENT (lower) exposure in
#      High_Volatility than the regime-blind agent, whose average risk
#      aversion is identical?
#   2. Does that show up as better risk-adjusted performance / drawdown,
#      with a statistically supported difference?
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
MODEL_ROOT = "models/ppo_pooled_v3"
TARGET_VOL = 0.01

parser = argparse.ArgumentParser()
parser.add_argument("--gamma-base", type=float, default=3.0)
parser.add_argument("--split", choices=["validation", "test"], default="test")
args = parser.parse_args()
GSUF = "" if args.gamma_base == 3.0 else f"g{args.gamma_base}"
MODEL_ROOT = f"models/ppo_pooled_v3{GSUF}"
OUT = f"results/v3{GSUF}_{args.split}"
os.makedirs(OUT, exist_ok=True)


def nearest_action(w):
    return int(np.argmin(np.abs(EXPOSURES - w)))


def vol_target_policy():
    def policy(obs, t, env):
        vol = env.df["Rolling_Volatility"].iloc[t]
        return nearest_action(min(1.0, TARGET_VOL / max(vol, 1e-6)))
    return policy


def load_models():
    from stable_baselines3 import PPO
    out = {"adaptive": [], "blind": []}
    for kind in out:
        for p in sorted(glob.glob(f"{MODEL_ROOT}/{kind}_scaled_s[0-9]*/ppo_regimex_final.zip")):
            out[kind].append(PPO.load(p))
    return out


def main():
    data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]) for t in TICKERS}
    splits = {t: get_splits(d, verbose=False) for t, d in data.items()}
    train_dfs = [s[0] for s in splits.values()]
    stats = obs_stats_from(pd.concat(train_dfs))
    cfg = gamma_config(args.gamma_base)
    bg = blind_gamma_from(train_dfs, cfg)
    models = load_models()
    print("models:", {k: len(v) for k, v in models.items()}, f"| blind gamma {bg:.3f} | split={args.split}")

    idx = {"validation": 1, "test": 2}[args.split]
    env_kw = dict(env_cls=RegimeXExposureEnv, scaled_obs=True, obs_stats=stats,
                  gamma_config=cfg, blind_gamma=bg)

    rows, series, expo, regimes = [], {}, {}, {}
    for t in TICKERS:
        d = splits[t][idx].reset_index(drop=True)
        runs = {
            "Buy & Hold": [rollout(d, lambda o, i, e: 4, **env_kw)],
            "Constant 50%": [rollout(d, lambda o, i, e: 2, **env_kw)],
            "Vol-target": [rollout(d, vol_target_policy(), **env_kw)],
            "PPO-v3 regime-blind": [rollout(d, ppo_policy(m), regime_adaptive=False, **env_kw)
                                    for m in models["blind"]],
            "PPO-v3 regime-adaptive": [rollout(d, ppo_policy(m), regime_adaptive=True, **env_kw)
                                       for m in models["adaptive"]],
        }
        for label, logs in runs.items():
            for s, lg in enumerate(logs):
                m = summarize(lg)
                m.pop("win_rate_trades", None)
                rows.append({"stock": t, "strategy": label, "seed": s,
                             "mean_exposure": lg["exposure"].mean(), **m})
            series[(label, t)] = [daily_returns(lg).to_numpy() for lg in logs]
            expo[(label, t)] = [lg["exposure"].to_numpy() for lg in logs]
        regimes[t] = runs["Buy & Hold"][0]["regime"].to_numpy()

    ps = pd.DataFrame(rows)
    ps.to_csv(f"{OUT}/per_seed_per_stock.csv", index=False)
    labels = list(dict.fromkeys(ps["strategy"]))
    pd.set_option("display.width", 200, "display.max_columns", 30)

    per_stock = ps.groupby(["strategy", "stock"])[["total_return", "sharpe", "max_drawdown", "mean_exposure"]].mean().reset_index()
    per_stock.to_csv(f"{OUT}/per_stock.csv", index=False)
    print("\nSharpe per stock (seed mean):")
    print(per_stock.pivot(index="strategy", columns="stock", values="sharpe").loc[labels].round(2))

    # ---- equal-weight portfolio (per seed -> mean/std) ----
    prow = []
    for lbl in labels:
        n = len(series[(lbl, TICKERS[0])])
        vals = []
        for s in range(n):
            r = np.mean([series[(lbl, t)][s] for t in TICKERS], axis=0)
            vals.append((total_return(r), sharpe(r), max_drawdown(r), r.std(ddof=1) * np.sqrt(252)))
        v = np.array(vals)
        prow.append({"strategy": lbl, "n_seeds": n,
                     "total_return": v[:, 0].mean(), "sharpe": v[:, 1].mean(),
                     "sharpe_std": v[:, 1].std(ddof=1) if n > 1 else np.nan,
                     "max_drawdown": v[:, 2].mean(), "ann_vol": v[:, 3].mean()})
    port = pd.DataFrame(prow)
    port.to_csv(f"{OUT}/portfolio_equal_weight.csv", index=False)
    print(f"\nEqual-weight 5-stock portfolio ({args.split}):")
    print(port.round(3).to_string(index=False))

    # ---- regime-stratified: return AND behaviour (exposure) ----
    reg_all = np.concatenate([regimes[t] for t in TICKERS])
    rrows = []
    for lbl in labels:
        n = len(series[(lbl, TICKERS[0])])
        for reg in REGIMES:
            m = reg_all == reg
            per_seed_bp, per_seed_ex = [], []
            for s in range(n):
                r = np.concatenate([series[(lbl, t)][s] for t in TICKERS])
                e = np.concatenate([expo[(lbl, t)][s] for t in TICKERS])
                per_seed_bp.append(r[m].mean() * 1e4)
                per_seed_ex.append(e[m].mean())
            rrows.append({"strategy": lbl, "regime": reg, "n_stock_days": int(m.sum()),
                          "mean_daily_bp": np.mean(per_seed_bp),
                          "mean_exposure": np.mean(per_seed_ex),
                          "mean_exposure_std": np.std(per_seed_ex, ddof=1) if n > 1 else np.nan})
    rt = pd.DataFrame(rrows)
    rt.to_csv(f"{OUT}/by_regime.csv", index=False)
    print("\nMean exposure by regime (behaviour):")
    print(rt.pivot(index="strategy", columns="regime", values="mean_exposure").loc[labels][REGIMES].round(3))
    print("\nMean daily return (bp) by regime:")
    print(rt.pivot(index="strategy", columns="regime", values="mean_daily_bp").loc[labels][REGIMES].round(2))
    print("stock-days:", rt[rt.strategy == "Buy & Hold"].set_index("regime")["n_stock_days"].to_dict())

    # ---- bootstrap ----
    def stacked(lbl):
        return np.concatenate([np.mean(series[(lbl, t)], axis=0) for t in TICKERS])

    def portfolio(lbl):
        return np.mean([np.mean(series[(lbl, t)], axis=0) for t in TICKERS], axis=0)

    A = "PPO-v3 regime-adaptive"
    brows = []
    for B in ["PPO-v3 regime-blind", "Vol-target", "Constant 50%", "Buy & Hold"]:
        for stat_name, stat in [("sharpe", sharpe), ("max_drawdown", max_drawdown),
                                ("total_return", total_return)]:
            res = paired_bootstrap(portfolio(A), portfolio(B), stat, 3000, 10)
            brows.append({"A": A, "B": B, "level": "portfolio", "regime": "ALL", "stat": stat_name, **res})
        ra, rb = stacked(A), stacked(B)
        for reg in REGIMES:
            m = reg_all == reg
            res = paired_bootstrap(ra[m], rb[m], lambda x: x.mean() * 1e4, 3000, 10)
            brows.append({"A": A, "B": B, "level": "stock-days", "regime": reg,
                          "stat": "mean_daily_bp", **res})
    # behavioural test: is adaptive exposure in HV different from blind?
    ea = np.concatenate([np.mean(expo[(A, t)], axis=0) for t in TICKERS])
    eb = np.concatenate([np.mean(expo[("PPO-v3 regime-blind", t)], axis=0) for t in TICKERS])
    for reg in REGIMES:
        m = reg_all == reg
        res = paired_bootstrap(ea[m], eb[m], lambda x: x.mean(), 3000, 10)
        brows.append({"A": A, "B": "PPO-v3 regime-blind", "level": "stock-days", "regime": reg,
                      "stat": "mean_exposure", **res})
    bt = pd.DataFrame(brows)
    bt.to_csv(f"{OUT}/bootstrap.csv", index=False)
    print("\nBootstrap (A = adaptive; diff = A - B; 95% CI; two-sided p):")
    print(bt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
