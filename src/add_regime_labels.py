import pandas as pd

INPUT_PATH = "data/processed/RELIANCE_regimes.csv"
OUTPUT_PATH = "data/processed/RELIANCE_regimes.csv"

print("=" * 60)
print("REGIMEX - ADD MARKET REGIME LABELS")
print("=" * 60)

# Load regime dataset
df = pd.read_csv(INPUT_PATH)

# Convert numerical regimes into meaningful names
regime_labels = {
    0: "Weak_Bear",
    1: "Normal_Market",
    2: "High_Volatility"
}

df["Market_Regime"] = df["Regime"].map(regime_labels)

# Display result
print("\nFirst 5 rows:")
print(df[["Date", "Regime", "Market_Regime"]].head())

print("\nRegime Distribution:")
print(df["Market_Regime"].value_counts())

# Save updated dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved updated dataset:")
print(OUTPUT_PATH)

print("\nColumns:")
print(df.columns.tolist())