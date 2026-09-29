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
from strategies import rollout, ppo_policy

# ============================================================
# REGIMEX - SHAP EXPLAINABILITY FOR THE TRAINED PPO POLICY
# ============================================================
#
# Explains the policy's action probabilities P(HOLD), P(BUY), P(SELL) as a
# function of the five state features the agent observes:
#   [Daily_Return, Rolling_Volatility, Hurst, Regime, Holdings]
#
# KernelExplainer is model-agnostic and exact enough for 5 features. The
# background set is a k-means summary of the TRAIN observations; the
# explained set is the held-out TEST observations.
# ============================================================

FEATURE_NAMES = ["Daily_Return", "Rolling_Volatility", "Hurst", "Regime", "Holdings"]
ACTIONS = ["HOLD", "BUY", "SELL"]
OUT = "results/figures"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/ppo_regimex/adaptive_s0/ppo_regimex_final.zip")
parser.add_argument("--tag", default="adaptive_s0")
parser.add_argument("--n-background", type=int, default=25)
parser.add_argument("--n-examples", type=int, default=4)
args = parser.parse_args()


def observations(df):
    """State the env would show the agent, without the Holdings dynamics."""
    return df[["Daily_Return", "Rolling_Volatility", "Hurst", "Regime"]].to_numpy(dtype=np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = PPO.load(args.model)

    def action_probs(X):
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(X, dtype=np.float32))
            return model.policy.get_distribution(obs).distribution.probs.numpy()

    df = pd.read_csv("data/processed/RELIANCE_regimes.csv", parse_dates=["Date"])
    train_df, _, test_df = get_splits(df, verbose=False)

    # Real rollout on test => the holdings feature is the agent's actual position
    log = rollout(test_df, ppo_policy(model), regime_adaptive="adaptive" in args.tag)
    holdings = (log["shares"].shift(1).fillna(0).to_numpy() > 0).astype(np.float32)

    X_test = np.column_stack([observations(test_df.iloc[: len(log)]), holdings])

    # Background: train observations (holdings sampled from both states)
    rng = np.random.default_rng(0)
    Xb = observations(train_df)
    Xb = np.column_stack([Xb, rng.integers(0, 2, size=len(Xb)).astype(np.float32)])
    background = shap.kmeans(Xb, args.n_background)

    explainer = shap.KernelExplainer(action_probs, background)
    sv = explainer.shap_values(X_test, nsamples=200, silent=True)
    # shap versions differ: list[n_actions] of (n, f)  or  array (n, f, n_actions)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (n, f, a)

    # ---- Mean |SHAP| per feature per action ----
    imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=FEATURE_NAMES, columns=ACTIONS)
    imp.to_csv("results/shap_importance.csv")
    print("Mean |SHAP| per feature and action:\n", imp.round(5))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Mean |SHAP| on action probability"); ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Feature importance by action - {args.tag} (test split)")
    fig.tight_layout(); fig.savefig(f"{OUT}/shap_importance_{args.tag}.png", dpi=140); plt.close(fig)

    # ---- Beeswarm summaries ----
    for i, action in enumerate(ACTIONS):
        if np.abs(sv[:, :, i]).max() < 1e-9:
            continue
        plt.figure()
        shap.summary_plot(sv[:, :, i], X_test, feature_names=FEATURE_NAMES, show=False)
        plt.title(f"SHAP summary - P({action}) - {args.tag}")
        plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary_{args.tag}_{action}.png", dpi=140)
        plt.close()

    # ---- Per-decision explanations: sample the days where the agent traded ----
    traded = np.where(log["trade_value"].to_numpy() > 0)[0]
    pick = traded if len(traded) <= args.n_examples else rng.choice(traded, args.n_examples, replace=False)
    if len(pick) == 0:
        print("Agent made no trades on test split - no per-trade explanations produced.")
    lines = []
    for j in sorted(pick):
        a = int(log["action"].iloc[j])
        contrib = pd.Series(sv[j, :, a], index=FEATURE_NAMES)
        regime = log["regime"].iloc[j]
        title = f"{log['date'].iloc[j].date()}  {ACTIONS[a]}  (regime={regime})"
        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib]
        ax.barh(FEATURE_NAMES, contrib.values, color=colors)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title); ax.set_xlabel(f"SHAP contribution to P({ACTIONS[a]})")
        fig.tight_layout()
        fig.savefig(f"{OUT}/shap_trade_{args.tag}_{log['date'].iloc[j].date()}_{ACTIONS[a]}.png", dpi=140)
        plt.close(fig)
        lines.append(f"{title}: " + ", ".join(f"{k}={v:+.3f}" for k, v in contrib.items()))
    print("\nPer-trade explanations:")
    print("\n".join(lines))

    # ---- Does the agent act differently by regime? ----
    by_regime = pd.crosstab(log["regime"], log["action"].map(dict(enumerate(ACTIONS))), normalize="index")
    by_regime.to_csv("results/action_mix_by_regime.csv")
    print("\nAction mix by regime (test):\n", by_regime.round(3))


if __name__ == "__main__":
    main()
