import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from trading_env import RegimeXTradingEnv
from strategies import (
    rollout, buy_and_hold_policy, signal_policy, ppo_policy,
    arima_signals, fit_xgboost, xgboost_signals,
)
from metrics import summarize, regime_summary, daily_returns, paired_bootstrap, sharpe, total_return

# ============================================================
# REGIMEX - WALK-FORWARD VALIDATION
# ============================================================
#
# Repeats train -> test across expanding historical windows instead of
# relying on the single static split. For each test year Y:
#
#     train = all data up to 31 Dec (Y-1)        (expanding window)
#     test  = calendar year Y                    (never seen in training)
#
# and every strategy (Buy & Hold, ARIMA, XGBoost, PPO regime-blind,
# PPO regime-adaptive) is refit on the fold's train data and evaluated
# on the fold's test year. Regime labels are causal, so no fold sees
# any information from its future.
#
# PPO uses fewer timesteps per fold than the main run to keep the total
# compute manageable; results are therefore a robustness check on the
# adaptive-vs-blind conclusion, not a replacement for the main run.
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"
OUT = "results"

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=150_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test-years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
args = parser.parse_args()


def train_ppo(train_df, adaptive, seed, timesteps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: RegimeXTradingEnv(df=train_df, regime_adaptive=adaptive)])
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01, clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        seed=seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    return model


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    fold_rows, regime_rows, oos = [], [], {}

    for year in args.test_years:
        train_df = df[df["Date"] < f"{year}-01-01"].reset_index(drop=True)
        test_df = df[(df["Date"] >= f"{year}-01-01") & (df["Date"] < f"{year + 1}-01-01")].reset_index(drop=True)
        if len(test_df) < 20:
            print(f"Skipping {year}: not enough test data")
            continue

        n_hv = int((test_df["Market_Regime"] == "High_Volatility").sum())
        print(f"\n=== Fold: test {year}  (train n={len(train_df)}, test n={len(test_df)}, HV days={n_hv}) ===")

        logs = {
            "Buy & Hold": rollout(test_df, buy_and_hold_policy()),
            "ARIMA": rollout(test_df, signal_policy(arima_signals(df, test_df))),
            "XGBoost": rollout(test_df, signal_policy(xgboost_signals(fit_xgboost(train_df), test_df))),
        }
        for label, adaptive in [("PPO regime-blind", False), ("PPO regime-adaptive", True)]:
            print(f"  training {label} ({args.timesteps} steps) ...")
            model = train_ppo(train_df, adaptive, args.seed, args.timesteps)
            logs[label] = rollout(test_df, ppo_policy(model), regime_adaptive=adaptive)

        for label, lg in logs.items():
            fold_rows.append({"test_year": year, "hv_days": n_hv, "strategy": label, **summarize(lg)})
            rs = regime_summary(lg)
            rs.insert(0, "strategy", label)
            rs.insert(0, "test_year", year)
            regime_rows.append(rs)
            oos.setdefault(label, []).append(daily_returns(lg))

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(f"{OUT}/walk_forward_folds.csv", index=False)
    pd.concat(regime_rows).to_csv(f"{OUT}/walk_forward_by_regime.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + "=" * 70 + "\nWALK-FORWARD: per-fold Sharpe / total return\n" + "=" * 70)
    print(folds.pivot(index="test_year", columns="strategy", values="sharpe").round(3))
    print(folds.pivot(index="test_year", columns="strategy", values="total_return").round(3))

    summary = folds.groupby("strategy")[["total_return", "sharpe", "sortino", "max_drawdown", "turnover"]].agg(["mean", "std"])
    summary.to_csv(f"{OUT}/walk_forward_summary.csv")
    print("\nMean over folds:\n", summary.round(3))

    # Adaptive vs blind: fold wins and pooled out-of-sample bootstrap
    piv = folds.pivot(index="test_year", columns="strategy", values="sharpe")
    wins = int((piv["PPO regime-adaptive"] > piv["PPO regime-blind"]).sum())
    print(f"\nAdaptive beats blind on Sharpe in {wins}/{len(piv)} folds")

    pooled = {k: pd.concat(v).to_numpy() for k, v in oos.items()}
    rows = []
    for other in ["PPO regime-blind", "Buy & Hold"]:
        for stat_name, stat in [("sharpe", sharpe), ("total_return", total_return)]:
            res = paired_bootstrap(pooled["PPO regime-adaptive"], pooled[other], stat, 5000, 10)
            rows.append({"A": "PPO regime-adaptive", "B": other, "stat": stat_name, **res})
    boot = pd.DataFrame(rows)
    boot.to_csv(f"{OUT}/walk_forward_bootstrap.csv", index=False)
    print("\nPooled out-of-sample bootstrap (concatenated test years):")
    print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
