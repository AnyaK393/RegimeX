import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from data_split import get_splits
from trading_env import obs_stats_from
from exposure_env import RegimeXExposureEnv, blind_gamma_from, EXPOSURES
from strategies import rollout

# ============================================================
# REGIMEX - SHAP FOR THE v3 (POSITION-SIZING) AGENTS
# ============================================================
#
# Target explained: the policy's EXPECTED TARGET EXPOSURE
#     E[w | state] = sum_a P(a | state) * exposure_a
# a single number in [0, 1], so a positive SHAP value means "this feature
# pushed the agent toward holding more of the stock".
#
# Observation (scaled): [z(Daily_Return), z(Rolling_Volatility), z(Hurst),
#   is_Normal, is_Weak_Bear, is_High_Vol, Exposure]. The three regime
# indicators are reported both individually and grouped as "Regime".
#
# Explained set: the agent's real test-split observations across all five
# stocks (captured during a true rollout, so Exposure is the agent's own
# position). Background: observations from the same policy on the train split.
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
FEATS = ["Daily_Return", "Rolling_Volatility", "Hurst", "is_Normal", "is_Weak_Bear", "is_High_Vol", "Exposure"]
GROUPED = ["Daily_Return", "Rolling_Volatility", "Hurst", "Regime", "Exposure"]
REGIME_NAMES = ["Normal_Market", "Weak_Bear", "High_Volatility"]

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["adaptive", "blind"], default="adaptive")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model-root", default="models/ppo_pooled_v3")
parser.add_argument("--n-explain", type=int, default=400)
parser.add_argument("--n-background", type=int, default=25)
args = parser.parse_args()

OUT = "results/figures"
TAG = f"v3_{args.kind}_s{args.seed}"


def main():
    os.makedirs(OUT, exist_ok=True)
    model = PPO.load(f"{args.model_root}/{args.kind}_scaled_s{args.seed}/ppo_regimex_final.zip")

    def expected_exposure(X):
        with torch.no_grad():
            probs = model.policy.get_distribution(
                torch.as_tensor(np.asarray(X, dtype=np.float32))).distribution.probs.numpy()
        return probs @ EXPOSURES

    data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]) for t in TICKERS}
    splits = {t: get_splits(d, verbose=False) for t, d in data.items()}
    train_dfs = [s[0] for s in splits.values()]
    stats = obs_stats_from(pd.concat(train_dfs))
    kw = dict(env_cls=RegimeXExposureEnv, scaled_obs=True, obs_stats=stats,
              blind_gamma=blind_gamma_from(train_dfs), regime_adaptive=args.kind == "adaptive")

    def collect(dfs):
        """Real rollouts of the trained policy, capturing the observations it saw."""
        obs_all, reg_all, dates, tick = [], [], [], []
        for t, d in dfs.items():
            seen = []

            def policy(obs, i, env):
                seen.append(obs.copy())
                return model.predict(obs, deterministic=True)[0]

            lg = rollout(d.reset_index(drop=True), policy, **kw)
            obs_all.append(np.array(seen))
            reg_all.append(lg["regime"].to_numpy())
            dates.append(lg["date"].to_numpy())
            tick += [t] * len(lg)
        return (np.concatenate(obs_all), np.concatenate(reg_all),
                np.concatenate(dates), np.array(tick))

    X_bg_all, _, _, _ = collect({t: splits[t][0] for t in TICKERS})
    X_te, reg_te, date_te, tick_te = collect({t: splits[t][2] for t in TICKERS})

    rng = np.random.default_rng(0)
    background = shap.kmeans(X_bg_all[rng.choice(len(X_bg_all), 2000, replace=False)], args.n_background)
    # Explain a stratified sample so High_Volatility days are well represented
    idx = []
    for r in REGIME_NAMES:
        pool = np.where(reg_te == r)[0]
        idx += list(rng.choice(pool, min(len(pool), args.n_explain // 3), replace=False))
    idx = np.array(sorted(idx))

    explainer = shap.KernelExplainer(expected_exposure, background)
    sv = np.asarray(explainer.shap_values(X_te[idx], nsamples=200, silent=True))   # (n, 7)
    Xs, regs, dts, ticks = X_te[idx], reg_te[idx], date_te[idx], tick_te[idx]
    print("Explained", len(idx), "test observations; base value", round(float(explainer.expected_value), 3))

    # ---- grouped attribution ----
    grouped = np.column_stack([sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3:6].sum(axis=1), sv[:, 6]])
    imp = pd.Series(np.abs(grouped).mean(axis=0), index=GROUPED)
    imp.to_csv(f"results/shap_importance_{TAG}.csv", header=["mean_abs_shap"])
    print("\nMean |SHAP| on expected exposure:\n", imp.round(4))

    by_reg = pd.DataFrame(grouped, columns=GROUPED).assign(regime=regs).groupby("regime").mean().loc[REGIME_NAMES]
    by_reg.to_csv(f"results/shap_by_regime_{TAG}.csv")
    print("\nMean signed SHAP by regime (positive = pushes exposure up):\n", by_reg.round(3))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    imp.sort_values().plot.barh(ax=ax, color="#d62728" if args.kind == "adaptive" else "#1f77b4")
    ax.set_xlabel("Mean |SHAP| on expected target exposure")
    ax.set_title(f"Feature importance - v3 {args.kind} agent (test)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/shap_importance_{TAG}.png", dpi=140); plt.close(fig)

    plt.figure()
    shap.summary_plot(sv, Xs, feature_names=FEATS, show=False)
    plt.title(f"SHAP summary - expected exposure - v3 {args.kind}")
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_{TAG}.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 4))
    by_reg.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Mean SHAP (signed)")
    ax.set_title(f"What pushes exposure up/down, by regime - v3 {args.kind}")
    plt.xticks(rotation=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/shap_by_regime_{TAG}.png", dpi=140); plt.close(fig)

    # ---- example per-day explanations: highest and lowest predicted exposure in High_Volatility ----
    pred = expected_exposure(Xs)
    hv = np.where(regs == "High_Volatility")[0]
    picks = [("HV_max_exposure", hv[np.argmax(pred[hv])]), ("HV_min_exposure", hv[np.argmin(pred[hv])])]
    nm = np.where(regs == "Normal_Market")[0]
    picks.append(("Normal_max_exposure", nm[np.argmax(pred[nm])]))
    for name, j in picks:
        contrib = pd.Series(grouped[j], index=GROUPED)
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.barh(GROUPED, contrib.values, color=["#d62728" if v > 0 else "#1f77b4" for v in contrib])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{ticks[j]} {str(dts[j])[:10]} ({regs[j]}) -> E[exposure]={pred[j]:.2f}", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{OUT}/shap_day_{TAG}_{name}.png", dpi=140); plt.close(fig)
        print(f"{name}: {ticks[j]} {str(dts[j])[:10]} {regs[j]} E[exp]={pred[j]:.2f} | "
              + ", ".join(f"{k}={v:+.3f}" for k, v in contrib.items()))


if __name__ == "__main__":
    main()
