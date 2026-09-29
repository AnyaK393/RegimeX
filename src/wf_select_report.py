import os
import sys
import glob
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from trading_env import obs_stats_from
from exposure_env import RegimeXExposureEnv, EXPOSURES
from strategies import rollout
from metrics import sharpe, total_return, max_drawdown

# ============================================================
# REGIMEX - AGGREGATE WALK-FORWARD SELECTION RESULTS (v3)
# ============================================================
#
# Selection rule (fixed BEFORE running):
#   choose gamma_base maximising the ADAPTIVE agent's mean portfolio Sharpe
#   across the 2019-2023 folds and both seeds. If a smaller gamma is not better
#   by more than 0.05 Sharpe than the pre-specified default (3.0), keep 3.0.
# Every grid point is reported, including the losers.
# The 2024-06+ test window is never used here.
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
DEFAULT_GAMMA_BASE = 3.0
MARGIN = 0.05
YEARS = [2019, 2020, 2021, 2022, 2023]

rows = [json.load(open(p)) for p in glob.glob("results/v3_select/g*.json")]
df = pd.DataFrame(rows)
print(f"{len(df)} job results")

# ---- baselines per fold (same env, same costs) ----
data = {t: pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"]).sort_values("Date")
        for t in TICKERS}


def vol_target_action(vol):
    w = min(1.0, 0.01 / max(vol, 1e-6))
    return int(np.argmin(np.abs(EXPOSURES - w)))


base_rows = []
for Y in YEARS:
    train = [d[d["Date"] < f"{Y}-01-01"] for d in data.values()]
    stats = obs_stats_from(pd.concat(train))
    tests = {t: d[(d["Date"] >= f"{Y}-01-01") & (d["Date"] < f"{Y + 1}-01-01")].reset_index(drop=True)
             for t, d in data.items()}
    kw = dict(env_cls=RegimeXExposureEnv, scaled_obs=True, obs_stats=stats)
    pols = {
        "Buy & Hold": lambda o, i, e: 4,
        "Constant 50%": lambda o, i, e: 2,
        "Vol-target": lambda o, i, e: vol_target_action(e.df["Rolling_Volatility"].iloc[i]),
    }
    for name, pol in pols.items():
        rets = [(lambda lg: (lg["value"] / lg["prev_value"] - 1).to_numpy())(rollout(tests[t], pol, **kw))
                for t in TICKERS]
        port = np.mean(rets, axis=0)
        base_rows.append({"year": Y, "strategy": name, "port_sharpe": sharpe(port),
                          "port_return": total_return(port), "port_maxdd": max_drawdown(port)})
base = pd.DataFrame(base_rows)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\nBaselines per fold (portfolio Sharpe):")
print(base.pivot(index="strategy", columns="year", values="port_sharpe").round(2))
print("mean:", base.groupby("strategy")["port_sharpe"].mean().round(3).to_dict())

# ---- grid results ----
g = df.groupby(["gamma_base", "kind", "year"])[["port_sharpe", "port_return", "port_maxdd"]].mean().reset_index()
summary = g.groupby(["gamma_base", "kind"])[["port_sharpe", "port_return", "port_maxdd"]].mean().reset_index()
print("\nGrid: mean over folds/seeds")
print(summary.round(3).to_string(index=False))

piv = g.pivot_table(index=["gamma_base", "kind"], columns="year", values="port_sharpe")
print("\nPortfolio Sharpe by fold:")
print(piv.round(2))

exp = df.copy()
for r in ["Normal_Market", "Weak_Bear", "High_Volatility"]:
    exp[r] = exp["exposure"].apply(lambda d: d.get(r))
print("\nMean exposure by regime (mean over folds/seeds):")
print(exp.groupby(["gamma_base", "kind"])[["Normal_Market", "Weak_Bear", "High_Volatility"]].mean().round(3))

# ---- selection ----
ad = summary[summary["kind"] == "adaptive"].set_index("gamma_base")["port_sharpe"]
best = float(ad.idxmax())
chosen = best if ad[best] - ad[DEFAULT_GAMMA_BASE] > MARGIN else DEFAULT_GAMMA_BASE
diff = (g[g.kind == "adaptive"].set_index(["gamma_base", "year"])["port_sharpe"]
        - g[g.kind == "blind"].set_index(["gamma_base", "year"])["port_sharpe"])
print("\nAdaptive - blind Sharpe per fold, by gamma_base:")
print(diff.unstack().round(2))
print("mean:", diff.groupby("gamma_base").mean().round(3).to_dict(),
      "| folds where adaptive wins:", (diff > 0).groupby("gamma_base").sum().to_dict())
print(f"\nBest adaptive mean Sharpe at gamma_base={best} ({ad[best]:.3f}); default {DEFAULT_GAMMA_BASE} = {ad[DEFAULT_GAMMA_BASE]:.3f}")
print(f"==> SELECTED gamma_base = {chosen}")

os.makedirs("results/v3_select_summary", exist_ok=True)
summary.to_csv("results/v3_select_summary/grid_summary.csv", index=False)
piv.to_csv("results/v3_select_summary/grid_by_fold.csv")
base.to_csv("results/v3_select_summary/baselines_by_fold.csv", index=False)
json.dump({"selected_gamma_base": chosen, "best_gamma_base": best,
           "adaptive_mean_sharpe": {str(k): float(v) for k, v in ad.items()}},
          open("results/v3_select_summary/selection.json", "w"), indent=1)
