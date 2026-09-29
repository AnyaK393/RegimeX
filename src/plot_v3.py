import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from exposure_env import RegimeXExposureEnv, blind_gamma_from, EXPOSURES
from strategies import rollout, ppo_policy
from metrics import REGIMES

# ============================================================
# REGIMEX - v3 SUMMARY FIGURES (test split)
#   1. equal-weight five-stock equity curves, High_Volatility shaded
#   2. mean exposure by regime (the behavioural evidence)
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
MODEL_ROOT = "models/ppo_pooled_v3"
OUT = "results/figures"
COLORS = {"Buy & Hold": "#7f7f7f", "Constant 50%": "#8c564b", "Vol-target": "#9467bd",
          "PPO-v3 regime-blind": "#1f77b4", "PPO-v3 regime-adaptive": "#d62728"}


def main():
    from stable_baselines3 import PPO
    os.makedirs(OUT, exist_ok=True)
    data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]) for t in TICKERS}
    splits = {t: get_splits(d, verbose=False) for t, d in data.items()}
    train_dfs = [s[0] for s in splits.values()]
    kw = dict(env_cls=RegimeXExposureEnv, scaled_obs=True, obs_stats=obs_stats_from(pd.concat(train_dfs)),
              blind_gamma=blind_gamma_from(train_dfs))
    models = {k: [PPO.load(p) for p in sorted(glob.glob(f"{MODEL_ROOT}/{k}_scaled_s[0-9]*/ppo_regimex_final.zip"))]
              for k in ["adaptive", "blind"]}

    def vt(o, i, e):
        w = min(1.0, 0.01 / max(e.df["Rolling_Volatility"].iloc[i], 1e-6))
        return int(np.argmin(np.abs(EXPOSURES - w)))

    curves, hv_mask, dates = {}, None, None
    for t in TICKERS:
        d = splits[t][2].reset_index(drop=True)
        runs = {"Buy & Hold": [rollout(d, lambda o, i, e: 4, **kw)],
                "Constant 50%": [rollout(d, lambda o, i, e: 2, **kw)],
                "Vol-target": [rollout(d, vt, **kw)],
                "PPO-v3 regime-blind": [rollout(d, ppo_policy(m), regime_adaptive=False, **kw) for m in models["blind"]],
                "PPO-v3 regime-adaptive": [rollout(d, ppo_policy(m), regime_adaptive=True, **kw) for m in models["adaptive"]]}
        for name, logs in runs.items():
            r = np.mean([(l["value"] / l["prev_value"] - 1).to_numpy() for l in logs], axis=0)
            curves.setdefault(name, []).append(r)
        dates = pd.to_datetime(runs["Buy & Hold"][0]["date"])
        frac = (runs["Buy & Hold"][0]["regime"] == "High_Volatility").to_numpy().astype(float)
        hv_mask = frac if hv_mask is None else hv_mask + frac

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, rs in curves.items():
        port = np.mean(rs, axis=0)
        ax.plot(dates, 100000 * np.cumprod(1 + port), label=name, color=COLORS[name], lw=1.6)
    for d_, n in zip(dates, hv_mask):
        if n >= 3:                      # shade days where >= 3 of the 5 stocks are High_Volatility
            ax.axvspan(d_, d_ + pd.Timedelta(days=1), color="#fcae91", alpha=0.5, lw=0)
    ax.set_title("v3 test: equal-weight five-stock equity (shaded = High_Volatility on >= 3 stocks)")
    ax.set_ylabel("Portfolio value (Rs)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/v3_equity_test.png", dpi=140); plt.close(fig)

    rt = pd.read_csv("results/v3_test/by_regime.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = list(COLORS)
    w = 0.8 / len(labels)
    for i, lbl in enumerate(labels):
        vals = [rt[(rt.strategy == lbl) & (rt.regime == r)]["mean_exposure"].iloc[0] for r in REGIMES]
        ax.bar(np.arange(3) + i * w, vals, w, label=lbl, color=COLORS[lbl])
    ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels(REGIMES)
    ax.set_ylabel("Mean exposure (fraction of equity)"); ax.set_ylim(0, 1.05)
    ax.set_title("v3 test: how much risk each strategy takes, by regime")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/v3_exposure_by_regime_test.png", dpi=140); plt.close(fig)
    print("saved v3_equity_test.png, v3_exposure_by_regime_test.png")


if __name__ == "__main__":
    main()
