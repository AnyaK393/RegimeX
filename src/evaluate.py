import os
import sys
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from strategies import (
    rollout, buy_and_hold_policy, signal_policy, ppo_policy,
    arima_signals, fit_xgboost, xgboost_signals,
)
from metrics import (
    summarize, regime_summary, daily_returns, paired_bootstrap,
    sharpe, total_return, REGIMES,
)

# ============================================================
# REGIMEX - BACKTEST, METRICS, REGIME-STRATIFIED EVALUATION,
#           SIGNIFICANCE TESTS AND PLOTS
# ============================================================
#
# Backtests every strategy on the held-out validation and test splits
# through the identical trading environment, then reports:
#   - overall metrics (return, Sharpe, Sortino, max DD, turnover, win rate)
#   - the same metrics broken down by regime
#   - paired block-bootstrap CIs for the differences that matter
#   - equity curves, regime overlay and per-regime charts
#
# PPO results are averaged over training seeds. Bootstrap tests use the
# seed-averaged daily return series (an equal-weight ensemble of the seeds).
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"
MODEL_ROOT = "models/ppo_regimex"
OUT = "results"
os.makedirs(f"{OUT}/logs", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

N_BOOT = 5000
BLOCK = 10


def load_ppo_models():
    """Returns {'adaptive': [(seed, model), ...], 'blind': [...]} for models on disk."""
    from stable_baselines3 import PPO
    found = {"adaptive": [], "blind": [], "adaptive_scaled": [], "blind_scaled": []}
    for kind in found:
        for path in sorted(glob.glob(f"{MODEL_ROOT}/{kind}_s[0-9]*/ppo_regimex_final.zip")):
            seed = int(os.path.basename(os.path.dirname(path)).split("_s")[-1])
            found[kind].append((seed, PPO.load(path)))
    return found


def run_all(df_full, splits, ppo_models):
    """Returns logs[strategy][split] -> rollout log DataFrame."""
    train_df = splits["train"]
    xgb = fit_xgboost(train_df)
    logs = {}

    for split_name in ["validation", "test"]:
        d = splits[split_name].reset_index(drop=True)
        print(f"\n[{split_name}] running strategies ...")

        logs.setdefault("Buy & Hold", {})[split_name] = rollout(d, buy_and_hold_policy())
        logs.setdefault("ARIMA", {})[split_name] = rollout(
            d, signal_policy(arima_signals(df_full, d)))
        logs.setdefault("XGBoost", {})[split_name] = rollout(
            d, signal_policy(xgboost_signals(xgb, d)))

        stats = obs_stats_from(train_df)
        for kind in ["blind", "adaptive", "blind_scaled", "adaptive_scaled"]:
            scaled = kind.endswith("_scaled")
            for seed, model in ppo_models[kind]:
                name = f"PPO-{kind}_s{seed}"
                logs.setdefault(name, {})[split_name] = rollout(
                    d, ppo_policy(model), regime_adaptive=kind.startswith("adaptive"),
                    scaled_obs=scaled, obs_stats=stats if scaled else None)
    return logs


def strategy_groups(logs):
    """Maps display name -> list of underlying strategy keys (seeds grouped)."""
    groups = {"Buy & Hold": ["Buy & Hold"], "ARIMA": ["ARIMA"], "XGBoost": ["XGBoost"]}
    for kind, label in [("blind", "PPO regime-blind"), ("adaptive", "PPO regime-adaptive"),
                        ("blind_scaled", "PPO regime-blind (scaled)"),
                        ("adaptive_scaled", "PPO regime-adaptive (scaled)")]:
        keys = sorted(k for k in logs if re.fullmatch(rf"PPO-{kind}_s\d+", k))
        if keys:
            groups[label] = keys
    return groups


def ensemble_returns(logs, keys, split):
    """Seed-averaged daily return series for a strategy group."""
    return pd.concat([daily_returns(logs[k][split]) for k in keys], axis=1).mean(axis=1)


def main():
    df_full = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    train_df, val_df, test_df = get_splits(df_full, verbose=False)
    splits = {"train": train_df, "validation": val_df, "test": test_df}

    ppo_models = load_ppo_models()
    print("PPO models found:", {k: len(v) for k, v in ppo_models.items()})

    logs = run_all(df_full, splits, ppo_models)
    groups = strategy_groups(logs)

    for name, per_split in logs.items():
        for split, lg in per_split.items():
            safe = name.replace(" ", "_").replace("&", "and")
            lg.to_csv(f"{OUT}/logs/{safe}__{split}.csv", index=False)

    # --------------------------------------------------------
    # Overall + regime-stratified tables (mean over seeds, std reported)
    # --------------------------------------------------------
    overall_rows, regime_rows = [], []
    for split in ["validation", "test"]:
        for label, keys in groups.items():
            per_seed = pd.DataFrame([summarize(logs[k][split]) for k in keys])
            row = {"split": split, "strategy": label, "n_seeds": len(keys)}
            row.update(per_seed.mean().to_dict())
            if len(keys) > 1:
                row["sharpe_std"] = per_seed["sharpe"].std(ddof=1)
                row["total_return_std"] = per_seed["total_return"].std(ddof=1)
            overall_rows.append(row)

            reg = pd.concat([regime_summary(logs[k][split]) for k in keys])
            reg = reg.groupby("regime", sort=False).mean(numeric_only=True).reset_index()
            reg.insert(0, "strategy", label)
            reg.insert(0, "split", split)
            regime_rows.append(reg)

    overall = pd.DataFrame(overall_rows)
    regime_tbl = pd.concat(regime_rows, ignore_index=True)
    overall.to_csv(f"{OUT}/metrics_overall.csv", index=False)
    regime_tbl.to_csv(f"{OUT}/metrics_by_regime.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n" + "=" * 70 + "\nOVERALL METRICS\n" + "=" * 70)
    print(overall.round(4).to_string(index=False))
    print("\n" + "=" * 70 + "\nREGIME-STRATIFIED METRICS\n" + "=" * 70)
    print(regime_tbl.round(4).to_string(index=False))

    # --------------------------------------------------------
    # Bootstrap significance (test split)
    # --------------------------------------------------------
    boot_rows = []
    for A, B, tag in [("PPO regime-adaptive", "PPO regime-blind", ""),
                      ("PPO regime-adaptive (scaled)", "PPO regime-blind (scaled)", "s")]:
        if A not in groups or B not in groups:
            continue
        for split in ["validation", "test"]:
            series = {lbl: ensemble_returns(logs, keys, split) if lbl.startswith("PPO")
                      else daily_returns(logs[keys[0]][split])
                      for lbl, keys in groups.items()}
            regimes = logs["Buy & Hold"][split]["regime"].to_numpy()

            comparisons = [(A, B), (A, "Buy & Hold"), (A, "XGBoost"), (A, "ARIMA"),
                           (B, "Buy & Hold")]
            for a, b in comparisons:
                ra, rb = series[a].to_numpy(), series[b].to_numpy()
                for stat_name, stat in [("sharpe", sharpe), ("total_return", total_return)]:
                    res = paired_bootstrap(ra, rb, stat, N_BOOT, BLOCK)
                    boot_rows.append({"split": split, "regime": "ALL", "A": a, "B": b,
                                      "stat": stat_name, **res})
                # Regime-specific: mean daily return difference, on that regime's days
                for reg in REGIMES:
                    m = regimes == reg
                    if m.sum() >= 5:
                        res = paired_bootstrap(ra[m], rb[m], lambda x: x.mean() * 1e4,
                                               N_BOOT, BLOCK)
                        boot_rows.append({"split": split, "regime": reg, "A": a, "B": b,
                                          "stat": "mean_daily_bp", **res})

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(f"{OUT}/bootstrap_tests.csv", index=False)
    if len(boot):
        print("\n" + "=" * 70 + "\nBOOTSTRAP TESTS (A minus B, 95% CI, two-sided p)\n" + "=" * 70)
        print(boot.round(4).to_string(index=False))

    make_plots(df_full, splits, logs, groups, regime_tbl)
    print(f"\nResults written to {OUT}/")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

COLORS = {
    "Buy & Hold": "#7f7f7f", "ARIMA": "#8c564b", "XGBoost": "#9467bd",
    "PPO regime-blind": "#1f77b4", "PPO regime-adaptive": "#d62728",
    "PPO regime-blind (scaled)": "#17becf", "PPO regime-adaptive (scaled)": "#ff7f0e",
}
REGIME_COLORS = {"Normal_Market": "#c7e9c0", "Weak_Bear": "#fdd0a2", "High_Volatility": "#fcae91"}


def make_plots(df_full, splits, logs, groups, regime_tbl):
    for split in ["validation", "test"]:
        # Equity curves with High_Volatility shaded
        fig, ax = plt.subplots(figsize=(11, 5))
        ref = logs["Buy & Hold"][split]
        dates = pd.to_datetime(ref["date"])
        for label, keys in groups.items():
            r = ensemble_returns(logs, keys, split) if len(keys) > 1 else daily_returns(logs[keys[0]][split])
            ax.plot(r.index, 100000 * (1 + r).cumprod(), label=label,
                    color=COLORS.get(label), lw=1.6)
        hv = (ref["regime"] == "High_Volatility").to_numpy()
        for d in dates[hv]:
            ax.axvspan(d, d + pd.Timedelta(days=1), color="#fcae91", alpha=0.5, lw=0)
        ax.set_title(f"Equity curves - {split} split (shaded = High_Volatility days)")
        ax.set_ylabel("Portfolio value (Rs)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/equity_{split}.png", dpi=140); plt.close(fig)

    # Regime overlay on the full price series
    fig, ax = plt.subplots(figsize=(12, 5))
    d = df_full.sort_values("Date")
    ax.plot(d["Date"], d["Close"], color="black", lw=0.8)
    ymin, ymax = d["Close"].min(), d["Close"].max()
    for reg, col in REGIME_COLORS.items():
        m = (d["Market_Regime"] == reg).to_numpy()
        ax.fill_between(d["Date"], ymin, ymax, where=m, color=col, alpha=0.6, lw=0, label=reg)
    for x, name in [(splits_end("train"), "train | val"), (splits_end("validation"), "val | test")]:
        ax.axvline(pd.Timestamp(x), color="k", ls="--", lw=0.8)
    ax.set_title("RELIANCE close with causal regime labels")
    ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(f"{OUT}/figures/regime_overlay.png", dpi=140); plt.close(fig)

    # Per-regime Sharpe and mean daily return on the test split
    t = regime_tbl[regime_tbl["split"] == "test"]
    for metric, ylabel, fname in [("mean_daily_bp", "Mean daily return (bp)", "regime_mean_return_test"),
                                  ("cum_return", "Cumulative return within regime", "regime_cum_return_test")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = list(groups.keys())
        w = 0.8 / len(labels)
        for i, lbl in enumerate(labels):
            vals = [t[(t["strategy"] == lbl) & (t["regime"] == r)][metric].iloc[0] for r in REGIMES]
            ax.bar(np.arange(len(REGIMES)) + i * w, vals, w, label=lbl, color=COLORS.get(lbl))
        ax.set_xticks(np.arange(len(REGIMES)) + 0.4 - w / 2)
        ax.set_xticklabels(REGIMES); ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{ylabel} by regime - test split")
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/{fname}.png", dpi=140); plt.close(fig)


def splits_end(name):
    from data_split import DEFAULT_TRAIN_END, DEFAULT_VAL_END
    return DEFAULT_TRAIN_END if name == "train" else DEFAULT_VAL_END


if __name__ == "__main__":
    main()
