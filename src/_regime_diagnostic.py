"""
RegimeX — Regime Detection Diagnostic
Deep-dives into the existing 3-cluster solution and tests alternatives.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# Load feature dataset
# ============================================================
df = pd.read_csv("data/processed/RELIANCE_regime_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("=" * 60)
print("REGIME DETECTION DIAGNOSTIC")
print("=" * 60)
print(f"\nTotal observations: {len(df)}")
print(f"Date range: {df['Date'].min().date()} --> {df['Date'].max().date()}")

features = ["Daily_Return", "Rolling_Volatility", "Hurst"]

# ============================================================
# Feature statistics BEFORE scaling
# ============================================================
print("\n" + "=" * 60)
print("RAW FEATURE STATISTICS (before scaling)")
print("=" * 60)
print(df[features].describe().round(6))

# ============================================================
# Check: was scaling actually effective?
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
X_scaled_df = pd.DataFrame(X_scaled, columns=[f"{f}_scaled" for f in features])
print("\nScaled feature means (should be ~0):")
print(X_scaled_df.mean().round(6))
print("\nScaled feature stds (should be ~1):")
print(X_scaled_df.std().round(6))

# ============================================================
# Reproduce original 3-cluster solution
# ============================================================
print("\n" + "=" * 60)
print("ORIGINAL 3-CLUSTER SOLUTION")
print("=" * 60)
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels3 = km3.fit_predict(X_scaled)
df["Regime_3"] = labels3

print("\nCluster sizes:")
print(df["Regime_3"].value_counts().sort_index())

print("\nCluster centroids (in original feature space):")
centroids_orig = scaler.inverse_transform(km3.cluster_centers_)
centroid_df = pd.DataFrame(centroids_orig, columns=features)
centroid_df.index.name = "Cluster"
print(centroid_df.round(6))

print("\nPer-cluster statistics:")
for c in sorted(df["Regime_3"].unique()):
    sub = df[df["Regime_3"] == c]
    print(f"\n  Cluster {c} (n={len(sub)}):")
    print(sub[features].describe().loc[["mean","std","min","max"]].round(6))

sil3 = silhouette_score(X_scaled, labels3)
print(f"\nSilhouette score (k=3): {sil3:.4f}")

# ============================================================
# Identify known high-volatility periods
# ============================================================
print("\n" + "=" * 60)
print("KNOWN HIGH-VOLATILITY PERIODS IN DATA")
print("=" * 60)

# COVID crash
covid = df[(df["Date"] >= "2020-02-01") & (df["Date"] <= "2020-05-31")]
print(f"\nCOVID period (Feb–May 2020): n={len(covid)}")
print(f"  Regime_3 distribution: {dict(covid['Regime_3'].value_counts())}")
print(f"  Max Rolling_Volatility: {covid['Rolling_Volatility'].max():.6f}")

# 2022 rate hike turbulence
rate22 = df[(df["Date"] >= "2022-01-01") & (df["Date"] <= "2022-12-31")]
print(f"\n2022 rate-hike year: n={len(rate22)}")
print(f"  Regime_3 distribution: {dict(rate22['Regime_3'].value_counts())}")
print(f"  Max Rolling_Volatility: {rate22['Rolling_Volatility'].max():.6f}")

# 2018 trade-war
tw18 = df[(df["Date"] >= "2018-01-01") & (df["Date"] <= "2018-12-31")]
print(f"\n2018 trade-war year: n={len(tw18)}")
print(f"  Regime_3 distribution: {dict(tw18['Regime_3'].value_counts())}")
print(f"  Max Rolling_Volatility: {tw18['Rolling_Volatility'].max():.6f}")

# What dates ARE High_Volatility in the current labelling?
hv = df[df["Regime_3"] == df.groupby("Regime_3")["Rolling_Volatility"].mean().idxmax()]
print(f"\nHigh-volatility cluster dates (top cluster by mean Rolling_Volatility):")
print(hv[["Date","Rolling_Volatility","Daily_Return","Hurst"]].to_string())

# ============================================================
# Test k=4
# ============================================================
print("\n" + "=" * 60)
print("4-CLUSTER ALTERNATIVE")
print("=" * 60)
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels4 = km4.fit_predict(X_scaled)
df["Regime_4"] = labels4

print("\nCluster sizes:")
print(df["Regime_4"].value_counts().sort_index())

print("\nCluster centroids (in original feature space):")
centroids4 = scaler.inverse_transform(km4.cluster_centers_)
centroid4_df = pd.DataFrame(centroids4, columns=features)
centroid4_df.index.name = "Cluster"
print(centroid4_df.round(6))

sil4 = silhouette_score(X_scaled, labels4)
print(f"\nSilhouette score (k=4): {sil4:.4f}")

print(f"\nSilhouette comparison: k=3 → {sil3:.4f}  |  k=4 → {sil4:.4f}")

print("\nCOVID period regime coverage with k=4:")
print(f"  {dict(df[(df['Date'] >= '2020-02-01') & (df['Date'] <= '2020-05-31')]['Regime_4'].value_counts())}")

print("\n2022 rate-hike coverage with k=4:")
print(f"  {dict(df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]['Regime_4'].value_counts())}")

# ============================================================
# Test k=2, 5 for silhouette comparison
# ============================================================
print("\n" + "=" * 60)
print("SILHOUETTE SCORES FOR k=2..6")
print("=" * 60)
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lbl)
    counts = pd.Series(lbl).value_counts().sort_values(ascending=False).tolist()
    print(f"  k={k}  silhouette={sil:.4f}  cluster sizes: {counts}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
