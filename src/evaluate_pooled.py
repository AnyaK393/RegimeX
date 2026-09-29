import os
import sys
import glob
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from strategies import (
    rollout, buy_and_hold_policy, signal_policy, ppo_policy,
    arima_signals, fit_xgboost, xgboost_signals,
)
from metrics import summarize, daily_returns, paired_bootstrap, sharpe, total_return, REGIMES

# ============================================================
# REGIMEX - POOLED FIVE-STOCK EVALUATION
# ============================================================
#
# Agents trained on the pooled train split (train_pooled.py) are evaluated
# on every stock's held-out TEST split (2024-06-03 -> 2025-12-31), against
# Buy & Hold, ARIMA and a pooled XGBoost. Reports per-stock results, an
# equal-weight portfolio across the five stocks, regime-stratified returns
# over all stock-days, and paired block-bootstrap tests on the stacked
# stock-day returns. PPO results are averaged over seeds.
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
MODEL_ROOT = "models/ppo_pooled"
OUT = "results/pooled"
os.makedirs(OUT, exist_ok=True)

KINDS = {
    "adaptive": "PPO regime-adaptive", "blind": "PPO regime-blind",
    "adaptive_scaled": "PPO regime-adaptive (scaled)", "blind_scaled": "PPO regime-blind (scaled)",
}


def load_models():
    from stable_baselines3 import PPO
    out = {k: [] for k in KINDS}
    for kind in KINDS:
        for p in sorted(glob.glob(f"{MODEL_ROOT}/{kind}_s[0-9]*/ppo_regimex_final.zip")):
            out[kind].append(PPO.load(p))
    return out


def main():
    data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]) for t in TICKERS}
    splits = {t: get_splits(d, verbose=False) for t, d in data.items()}
    pooled_train = pd.concat([s[0] for s in splits.values()])
    stats = obs_stats_from(pooled_train)
    models = load_models()
    print("models:", {k: len(v) for k, v in models.items()})

    # xgboost on the pooled train set (same features / label as the single-stock version)
    xgb = fit_xgboost(pooled_train.reset_index(drop=True))

    per_seed = []      # rows: stock, strategy, seed, metrics
    series = {}        # (label, stock) -> list of seed return arrays
    regimes = {}       # stock -> regime array

    for t in TICKERS:
        test = splits[t][2].reset_index(drop=True)
        print(f"[{t}] test n={len(test)}")
        runs = {
            "Buy & Hold": [rollout(test, buy_and_hold_policy())],
            "ARIMA": [rollout(test, signal_policy(arima_signals(data[t], test)))],
            "XGBoost": [rollout(test, signal_policy(xgboost_signals(xgb, test)))],
        }
        for kind, label in KINDS.items():
            scaled = kind.endswith("_scaled")
            runs[label] = [
                rollout(test, ppo_policy(m), regime_adaptive=kind.startswith("adaptive"),
                        scaled_obs=scaled, obs_stats=stats if scaled else None)
                for m in models[kind]
            ]
        for label, logs in runs.items():
            for i, lg in enumerate(logs):
                per_seed.append({"stock": t, "strategy": label, "seed": i, **summarize(lg)})
            if logs:
                series[(label, t)] = [daily_returns(lg).to_numpy() for lg in logs]
        regimes[t] = runs["Buy & Hold"][0]["regime"].to_numpy()

    ps = pd.DataFrame(per_seed)
    ps.to_csv(f"{OUT}/per_seed_per_stock.csv", index=False)
    labels = list(dict.fromkeys(ps["strategy"]))

    pd.set_option("display.width", 200, "display.max_columns", 30)

    # ---- per-stock table (mean over seeds) ----
    per_stock = ps.groupby(["strategy", "stock"])[["total_return", "sharpe", "max_drawdown", "pct_in_market", "n_trades"]].mean().reset_index()
    per_stock.to_csv(f"{OUT}/per_stock.csv", index=False)
    print("\nSharpe per stock (mean over seeds):")
    print(per_stock.pivot(index="strategy", columns="stock", values="sharpe").loc[labels].round(2))
    print("\nTotal return per stock:")
    print(per_stock.pivot(index="strategy", columns="stock", values="total_return").loc[labels].round(3))

    # ---- equal-weight portfolio across the 5 stocks (per seed, then mean) ----
    port_rows = []
    for lbl in labels:
        n_seeds = len(series[(lbl, TICKERS[0])])
        vals = []
        for s in range(n_seeds):
            r = np.mean([series[(lbl, t)][s] for t in TICKERS], axis=0)
            vals.append((total_return(r), sharpe(r)))
        v = np.array(vals)
        port_rows.append({"strategy": lbl, "n_seeds": n_seeds, "total_return": v[:, 0].mean(),
                          "sharpe": v[:, 1].mean(), "sharpe_std": v[:, 1].std(ddof=1) if n_seeds > 1 else np.nan})
    port = pd.DataFrame(port_rows)
    port.to_csv(f"{OUT}/portfolio_equal_weight.csv", index=False)
    print("\nEqual-weight 5-stock portfolio (test):")
    print(port.round(3).to_string(index=False))

    # ---- regime-stratified over all stock-days (seed-mean returns) ----
    def stacked(lbl):
        return np.concatenate([np.mean(series[(lbl, t)], axis=0) for t in TICKERS])
    reg_all = np.concatenate([regimes[t] for t in TICKERS])
    rrows = []
    for lbl in labels:
        r = stacked(lbl)
        for reg in REGIMES:
            m = reg_all == reg
            rrows.append({"strategy": lbl, "regime": reg, "n_stock_days": int(m.sum()),
                          "mean_daily_bp": r[m].mean() * 1e4, "sharpe": sharpe(r[m]),
                          "cum_return_sum": total_return(r[m])})
    rt = pd.DataFrame(rrows)
    rt.to_csv(f"{OUT}/by_regime.csv", index=False)
    print("\nMean daily return (bp) by regime, all stock-days:")
    print(rt.pivot(index="strategy", columns="regime", values="mean_daily_bp").loc[labels][REGIMES].round(2))
    print("stock-days per regime:", rt[rt.strategy == "Buy & Hold"].set_index("regime")["n_stock_days"].to_dict())

    # ---- bootstrap on stacked stock-days ----
    brows = []
    pairs = [("PPO regime-adaptive", "PPO regime-blind"),
             ("PPO regime-adaptive (scaled)", "PPO regime-blind (scaled)"),
             ("PPO regime-adaptive", "Buy & Hold"),
             ("PPO regime-adaptive (scaled)", "Buy & Hold"),
             ("PPO regime-blind", "Buy & Hold"),
             ("PPO regime-blind (scaled)", "Buy & Hold")]
    for a, b in pairs:
        if a not in labels or b not in labels:
            continue
        ra, rb = stacked(a), stacked(b)
        res = paired_bootstrap(ra, rb, sharpe, 3000, 10)
        brows.append({"A": a, "B": b, "regime": "ALL", "stat": "sharpe", **res})
        res = paired_bootstrap(ra, rb, lambda x: x.mean() * 1e4, 3000, 10)
        brows.append({"A": a, "B": b, "regime": "ALL", "stat": "mean_daily_bp", **res})
        for reg in REGIMES:
            m = reg_all == reg
            res = paired_bootstrap(ra[m], rb[m], lambda x: x.mean() * 1e4, 3000, 10)
            brows.append({"A": a, "B": b, "regime": reg, "stat": "mean_daily_bp", **res})
    bt = pd.DataFrame(brows)
    bt.to_csv(f"{OUT}/bootstrap.csv", index=False)
    print("\nBootstrap (stacked stock-days):")
    print(bt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
