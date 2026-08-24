import pandas as pd

df = pd.read_csv("data/processed/RELIANCE_regimes.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Total rows:", len(df))
print("Date range:", df["Date"].min().date(), "-->", df["Date"].max().date())
print()
print("Regime distribution:")
print(df["Market_Regime"].value_counts())
print()

n = len(df)
t = int(n * 0.70)
v = int(n * 0.15)
te = n - t - v

print("Proposed splits (70/15/15):")
print(f"  Train:      rows 0:{t}  n={t}  {df.iloc[0]['Date'].date()} to {df.iloc[t-1]['Date'].date()}")
print(f"  Validation: rows {t}:{t+v}  n={v}  {df.iloc[t]['Date'].date()} to {df.iloc[t+v-1]['Date'].date()}")
print(f"  Test:       rows {t+v}:{n}  n={te}  {df.iloc[t+v]['Date'].date()} to {df.iloc[-1]['Date'].date()}")
print()
print("Regime coverage per split:")
for label, (s, e) in [("Train", (0, t)), ("Validation", (t, t+v)), ("Test", (t+v, n))]:
    print(f"  {label}: {dict(df.iloc[s:e]['Market_Regime'].value_counts())}")
print()
print("Columns in dataset:")
print(list(df.columns))
