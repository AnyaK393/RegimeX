import pandas as pd
import os

# Stocks in our dataset
stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY"
]

# Input and output folders
raw_folder = "data/raw"
processed_folder = "data/processed"

# Create processed folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

print("\n" + "=" * 60)
print("REGIMEX - DATA CLEANING")
print("=" * 60)

for stock in stocks:

    print(f"\nCleaning {stock}...")

    # File paths
    input_file = os.path.join(raw_folder, f"{stock}.csv")
    output_file = os.path.join(processed_folder, f"{stock}_clean.csv")

    # Read raw CSV
    data = pd.read_csv(input_file)

    # The first two rows are Yahoo Finance header information.
    # Remove them because actual market data starts from row 2.
    data = data.iloc[2:].copy()

    # Rename columns
    data.columns = [
        "Date",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    # Convert Date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Convert numerical columns to numeric values
    numeric_columns = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Sort by date
    data = data.sort_values("Date")

    # Remove duplicate dates
    data = data.drop_duplicates(subset="Date")

    # Remove rows with missing values
    data = data.dropna()

    # Reset index
    data = data.reset_index(drop=True)

    # Save cleaned dataset
    data.to_csv(output_file, index=False)

    # Print validation information
    print(f"Saved: {output_file}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate dates: {data['Date'].duplicated().sum()}")
    print(f"Date range: {data['Date'].min().date()} → {data['Date'].max().date()}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)