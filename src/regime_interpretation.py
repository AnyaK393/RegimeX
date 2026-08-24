import pandas as pd


print("="*60)
print("REGIMEX - REGIME INTERPRETATION")
print("="*60)


df = pd.read_csv(
    "data/processed/RELIANCE_regimes.csv"
)


summary = df.groupby("Regime")[
    [
        "Daily_Return",
        "Rolling_Volatility",
        "Hurst"
    ]
].mean()


print("\nRegime Characteristics:")
print(summary)


print("\nNumber of days in each regime:")
print(df["Regime"].value_counts())