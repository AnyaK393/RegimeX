import pandas as pd
import os
import matplotlib.pyplot as plt
from hurst import compute_Hc

# Stocks
stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY"
]

processed_folder = "data/processed"

# Hurst calculation window
WINDOW = 100

plt.figure(figsize=(12, 6))

for stock in stocks:

    print(f"\nCalculating Hurst for {stock}...")

    file_path = os.path.join(
        processed_folder,
        f"{stock}_clean.csv"
    )

    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Store rolling Hurst values
    hurst_values = [None] * len(data)

    # Calculate Hurst over rolling windows
    for i in range(WINDOW, len(data) + 1):

        window_data = data["Adj Close"].iloc[i-WINDOW:i].values

        try:
            H, c, _ = compute_Hc(
                window_data,
                kind="price",
                simplified=True
            )

            hurst_values[i - 1] = H

        except Exception:
            hurst_values[i - 1] = None

    data["Hurst_100D"] = hurst_values

    # Print basic information
    valid_hurst = data["Hurst_100D"].dropna()

    print(f"Valid Hurst observations: {len(valid_hurst)}")
    print(f"Mean Hurst: {valid_hurst.mean():.4f}")
    print(f"Minimum Hurst: {valid_hurst.min():.4f}")
    print(f"Maximum Hurst: {valid_hurst.max():.4f}")

    # Plot
    plt.plot(
        data["Date"],
        data["Hurst_100D"],
        label=stock
    )

# Reference line: H = 0.5
plt.axhline(
    y=0.5,
    linestyle="--",
    label="H = 0.5"
)

plt.title("RegimeX - 100-Day Rolling Hurst Exponent")
plt.xlabel("Date")
plt.ylabel("Hurst Exponent")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()