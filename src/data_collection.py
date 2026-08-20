import pandas as pd
import os

# Stocks we are working with
stocks = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY"
]

# Folder containing our raw CSV files
raw_folder = "data/raw"

print("\n" + "=" * 60)
print("REGIMEX - RAW DATA VALIDATION")
print("=" * 60)

# Check every stock
for stock in stocks:

    file_path = os.path.join(raw_folder, f"{stock}.csv")

    print("\n" + "=" * 60)
    print(f"Checking: {stock}")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found!")
        continue

    # Read existing CSV
    data = pd.read_csv(file_path)

    # Basic information
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nColumn names:")
    print(data.columns.tolist())

    # Missing values
    print("\nMissing values:")
    print(data.isnull().sum())

    # Duplicate rows
    print("\nDuplicate rows:", data.duplicated().sum())

    # First 3 rows
    print("\nFirst 3 rows:")
    print(data.head(3))

    # Last 3 rows
    print("\nLast 3 rows:")
    print(data.tail(3))

print("\n" + "=" * 60)
print("RAW DATA VALIDATION COMPLETE")
print("=" * 60)