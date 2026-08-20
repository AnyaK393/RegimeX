import pandas as pd
import os
import matplotlib.pyplot as plt

# Stocks in our project
stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY"
]

# Location of cleaned data
processed_folder = "data/processed"

# Create figure
plt.figure(figsize=(12, 6))

for stock in stocks:

    file_path = os.path.join(
        processed_folder,
        f"{stock}_clean.csv"
    )

    # Load cleaned data
    data = pd.read_csv(file_path)

    # Convert Date
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort chronologically
    data = data.sort_values("Date")

    # Calculate daily returns
    data["Daily_Return"] = data["Adj Close"].pct_change()

    # Calculate 20-day rolling volatility
    data["Rolling_Volatility_20D"] = (
        data["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    # Plot rolling volatility
    plt.plot(
        data["Date"],
        data["Rolling_Volatility_20D"],
        label=stock
    )

# Formatting
plt.title("RegimeX - 20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("20-Day Rolling Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()