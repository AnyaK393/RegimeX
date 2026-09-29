import os
import sys

import numpy as np
import pandas as pd
from hurst import compute_Hc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_split import DEFAULT_TRAIN_END

# ============================================================
# REGIMEX - MULTI-STOCK REGIME DATASETS
# ============================================================
#
# Builds <TICKER>_regimes.csv for the other four stocks with the same
# schema as RELIANCE_regimes.csv, so the pooled agent can train on all
# five. Method (same hybrid as RELIANCE, but stricter on leakage):
#
#   1. Features: Daily_Return, 20-day Rolling_Volatility, 100-day Hurst.
#   2. KMeans (k=3) is fitted on TRAIN-PERIOD rows only, then used to
#      label all rows. (The original RELIANCE labels were fitted on the
#      full sample; here no test-period data touches the clustering or
#      its standardisation.)
#   3. Cluster roles: highest mean volatility -> High_Volatility base;
#      of the other two, lower mean return -> Weak_Bear, else Normal_Market.
#   4. High_Volatility is additionally set on any day where volatility
#      >= causal trailing P90 (504-day window, expanding fallback with
#      >= 126 prior days; see regime_relabel.py).
# ============================================================

TICKERS = ["TCS", "HDFCBANK", "ICICIBANK", "INFY"]
REGIME_INT = {"Weak_Bear": 0, "Normal_Market": 1, "High_Volatility": 2}
TRAILING, MIN_TRAIL, MIN_EXP, PCT = 504, 252, 126, 0.90


def hurst_value(series):
    try:
        return compute_Hc(series, kind="price", simplified=True)[0]
    except Exception:
        return np.nan


def build(ticker):
    df = pd.read_csv(f"data/processed/{ticker}_clean.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Rolling_Volatility"] = df["Daily_Return"].rolling(20).std()
    df["Hurst"] = df["Adj Close"].rolling(100).apply(hurst_value, raw=False)
    df = df.dropna().reset_index(drop=True)

    feats = ["Daily_Return", "Rolling_Volatility", "Hurst"]
    train = df["Date"] <= DEFAULT_TRAIN_END

    scaler = StandardScaler().fit(df.loc[train, feats])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(df.loc[train, feats]))
    df["cluster"] = km.predict(scaler.transform(df[feats]))

    # Cluster roles from TRAIN-period statistics only
    stats = df[train].groupby("cluster")[["Rolling_Volatility", "Daily_Return"]].mean()
    hv_c = stats["Rolling_Volatility"].idxmax()
    rest = stats.drop(hv_c)
    bear_c = rest["Daily_Return"].idxmin()
    role = {c: "Normal_Market" for c in stats.index}
    role[hv_c], role[bear_c] = "High_Volatility", "Weak_Bear"
    df["Market_Regime"] = df["cluster"].map(role)

    # Causal rolling P90 High_Volatility override
    vol = df["Rolling_Volatility"].shift(1)
    thr = vol.rolling(TRAILING, min_periods=MIN_TRAIL).quantile(PCT).combine_first(
        vol.expanding(min_periods=MIN_EXP).quantile(PCT))
    df.loc[df["Rolling_Volatility"] >= thr, "Market_Regime"] = "High_Volatility"

    df["Regime"] = df["Market_Regime"].map(REGIME_INT)
    return df.drop(columns=["cluster"])


if __name__ == "__main__":
    for t in TICKERS:
        d = build(t)
        d.to_csv(f"data/processed/{t}_regimes.csv", index=False)
        tr = d["Date"] <= DEFAULT_TRAIN_END
        va = (d["Date"] > DEFAULT_TRAIN_END) & (d["Date"] <= "2024-05-31")
        te = d["Date"] > "2024-05-31"
        print(f"\n{t}: {len(d)} rows  {d['Date'].min().date()} -> {d['Date'].max().date()}")
        for name, m in [("train", tr), ("val", va), ("test", te)]:
            c = d.loc[m, "Market_Regime"].value_counts(normalize=True).mul(100).round(1).to_dict()
            print(f"  {name:5s} n={m.sum():5d} {c}")
