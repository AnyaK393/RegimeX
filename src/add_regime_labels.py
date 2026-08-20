import pandas as pd


df = pd.read_csv(
    "data/processed/RELIANCE_regimes.csv"
)


regime_mapping = {
    0: "Weak_Bear",
    1: "Normal_Market",
    2: "High_Volatility"
}


df["Market_Regime"] = df["Regime"].map(regime_mapping)


df.to_csv(
    "data/processed/RELIANCE_final_regime_dataset.csv",
    index=False
)


print(df[[
    "Date",
    "Regime",
    "Market_Regime"
]].head())


print("\nRegime Distribution:")
print(df["Market_Regime"].value_counts())