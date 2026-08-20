import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


print("="*60)
print("REGIMEX - MARKET REGIME DETECTION")
print("="*60)


# -----------------------------
# Load Feature Dataset
# -----------------------------

df = pd.read_csv(
    "data/processed/RELIANCE_regime_features.csv"
)

df["Date"] = pd.to_datetime(df["Date"])


print("\nDataset loaded")
print(df.head())


# -----------------------------
# Select Features
# -----------------------------

features = [
    "Daily_Return",
    "Rolling_Volatility",
    "Hurst"
]


X = df[features]


# -----------------------------
# Standardization
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------
# KMeans Clustering
# -----------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


df["Regime"] = kmeans.fit_predict(X_scaled)


print("\nRegime counts:")
print(df["Regime"].value_counts())


# -----------------------------
# Save Results
# -----------------------------

df.to_csv(
    "data/processed/RELIANCE_regimes.csv",
    index=False
)


print("\nSaved:")
print("data/processed/RELIANCE_regimes.csv")


# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(14,6))


for regime in sorted(df["Regime"].unique()):

    temp = df[df["Regime"] == regime]

    plt.scatter(
        temp["Date"],
        temp["Rolling_Volatility"],
        s=10,
        label=f"Regime {regime}"
    )


plt.title(
    "RegimeX - Detected Market Regimes"
)

plt.xlabel("Date")

plt.ylabel(
    "Rolling Volatility"
)

plt.legend()

plt.grid(True)

plt.show()