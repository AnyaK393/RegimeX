import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/RELIANCE_regime_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

TRAIN_END = "2022-10-25"
VAL_END   = "2024-05-31"

train = df[df["Date"] <= TRAIN_END]
val   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)]
test  = df[df["Date"] > VAL_END]

print("Rolling_Volatility percentiles by split:")
for name, s in [("Train", train), ("Val", val), ("Test", test)]:
    percs = np.percentile(s["Rolling_Volatility"], [50, 70, 80, 85, 90, 95])
    print(f"  {name}: P50={percs[0]:.5f} P70={percs[1]:.5f} P80={percs[2]:.5f} P85={percs[3]:.5f} P90={percs[4]:.5f} P95={percs[5]:.5f}")

print()
p90_train = np.percentile(train["Rolling_Volatility"], 90)
print(f"TRAIN P90 = {p90_train:.6f}")
print("What pct of each split is above TRAIN P90?")
for name, s in [("Train", train), ("Val", val), ("Test", test)]:
    n = (s["Rolling_Volatility"] >= p90_train).sum()
    print(f"  {name}: {n}/{len(s)} = {100*n/len(s):.1f}%")

print()
p80_train = np.percentile(train["Rolling_Volatility"], 80)
print(f"TRAIN P80 = {p80_train:.6f}")
print("What pct of each split is above TRAIN P80?")
for name, s in [("Train", train), ("Val", val), ("Test", test)]:
    n = (s["Rolling_Volatility"] >= p80_train).sum()
    print(f"  {name}: {n}/{len(s)} = {100*n/len(s):.1f}%")

print()
p75_train = np.percentile(train["Rolling_Volatility"], 75)
print(f"TRAIN P75 = {p75_train:.6f}")
print("What pct of each split is above TRAIN P75?")
for name, s in [("Train", train), ("Val", val), ("Test", test)]:
    n = (s["Rolling_Volatility"] >= p75_train).sum()
    print(f"  {name}: {n}/{len(s)} = {100*n/len(s):.1f}%")

print()
print("Val top 10 vol days:")
print(val.nlargest(10, "Rolling_Volatility")[["Date", "Rolling_Volatility"]].to_string())
print()
print("Test top 10 vol days:")
print(test.nlargest(10, "Rolling_Volatility")[["Date", "Rolling_Volatility"]].to_string())
