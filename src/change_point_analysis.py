import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt


print("="*60)
print("REGIMEX - CHANGE POINT DETECTION")
print("="*60)


# -----------------------------
# Load cleaned data
# -----------------------------

file_path = "data/processed/RELIANCE_clean.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

print("\nDataset loaded")
print(df.head())


# -----------------------------
# Calculate daily returns
# -----------------------------

df["Daily_Return"] = df["Adj Close"].pct_change()

df = df.dropna()


returns = df["Daily_Return"].values


print("\nReturn observations:", len(returns))


# -----------------------------
# Change Point Detection
# -----------------------------

# Reshape for ruptures
signal = returns.reshape(-1,1)


# PELT algorithm
model = rpt.Binseg(
    model="rbf"
)

model.fit(signal)


# Number of expected change points
change_points = model.predict(
    n_bkps=5
)


print("\nDetected change points:")
print(change_points)


# Remove final endpoint
change_points = change_points[:-1]


# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(16,7))


plt.plot(
    df["Date"],
    returns,
    label="Daily Returns"
)


for cp in change_points:

    plt.axvline(
        df["Date"].iloc[cp],
        linestyle="--"
    )


plt.title(
    "RegimeX - RELIANCE Daily Returns with Detected Change Points"
)

plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()