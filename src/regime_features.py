import pandas as pd
import numpy as np
from hurst import compute_Hc


print("="*60)
print("REGIMEX - REGIME FEATURE ENGINEERING")
print("="*60)


# Load cleaned data
df = pd.read_csv(
    "data/processed/RELIANCE_clean.csv"
)

df["Date"] = pd.to_datetime(df["Date"])


# -----------------------------
# Daily Returns
# -----------------------------

df["Daily_Return"] = (
    df["Adj Close"]
    .pct_change()
)


# -----------------------------
# Rolling Volatility
# -----------------------------

df["Rolling_Volatility"] = (
    df["Daily_Return"]
    .rolling(window=20)
    .std()
)


# -----------------------------
# Rolling Hurst Exponent
# -----------------------------

def hurst_value(series):

    try:
        H, _, _ = compute_Hc(
            series,
            kind="price",
            simplified=True
        )
        return H

    except:
        return np.nan


df["Hurst"] = (
    df["Adj Close"]
    .rolling(window=100)
    .apply(
        hurst_value,
        raw=False
    )
)


# -----------------------------
# Remove missing rows
# -----------------------------

df = df.dropna()


# Save feature dataset
df.to_csv(
    "data/processed/RELIANCE_regime_features.csv",
    index=False
)


print("\nFeature dataset created")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)