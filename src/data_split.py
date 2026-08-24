import pandas as pd
import numpy as np
import os


# ============================================================
# REGIMEX - CHRONOLOGICAL DATA SPLIT
# ============================================================
#
# Provides a single reusable function get_splits() that slices
# the regime dataset by date boundary. Used by:
#   - train_ppo.py (this phase)
#   - baseline models (next phase)
#   - walk-forward validation (next phase)
#
# Split boundaries (70 / 15 / 15):
#   Train:      2015-05-29 -> 2022-10-25   (n=1831)
#   Validation: 2022-10-27 -> 2024-05-31   (n=392)
#   Test:       2024-06-03 -> 2025-12-31   (n=394)
#
# NO shuffling. Strict temporal order is preserved throughout.
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"

DEFAULT_TRAIN_END = "2022-10-25"
DEFAULT_VAL_END   = "2024-05-31"


def get_splits(
    df: pd.DataFrame,
    train_end_date: str = DEFAULT_TRAIN_END,
    val_end_date:   str = DEFAULT_VAL_END,
    verbose:        bool = True,
) -> tuple:
    """
    Chronological train / validation / test split by date boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Full regime dataset with a 'Date' column (datetime or str).
    train_end_date : str
        Last date (inclusive) of the training split.
    val_end_date : str
        Last date (inclusive) of the validation split.
        Test set is everything after this date.
    verbose : bool
        If True, prints a summary table.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
        Each DataFrame is sorted by date with a fresh integer index.
        Original index is dropped to avoid step-indexing bugs in the env.
    """
    # Ensure datetime and chronological order
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    train_end = pd.to_datetime(train_end_date)
    val_end   = pd.to_datetime(val_end_date)

    train_df = df[df["Date"] <= train_end].reset_index(drop=True)
    val_df   = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].reset_index(drop=True)
    test_df  = df[df["Date"] > val_end].reset_index(drop=True)

    if verbose:
        _print_split_summary(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _print_split_summary(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Prints a concise split verification table."""

    print("\n" + "=" * 60)
    print("REGIMEX - DATA SPLIT SUMMARY")
    print("=" * 60)

    total = len(train_df) + len(val_df) + len(test_df)

    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:

        n   = len(split)
        pct = 100 * n / total

        date_range = (
            f"{split['Date'].min().date()} -> {split['Date'].max().date()}"
            if n > 0 else "empty"
        )

        print(f"\n  {name} ({pct:.0f}%):  n={n}  [{date_range}]")

        if "Market_Regime" in split.columns:
            regime_counts = split["Market_Regime"].value_counts()
            for regime, count in regime_counts.items():
                regime_pct = 100 * count / n
                print(f"    {regime:<20s} {count:5d} ({regime_pct:5.1f}%)")

    print()

    # Sanity checks
    _check_no_overlap(train_df, val_df, test_df)


def _check_no_overlap(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Asserts no date leaks between splits."""

    train_dates = set(train_df["Date"])
    val_dates   = set(val_df["Date"])
    test_dates  = set(test_df["Date"])

    tv_overlap = train_dates & val_dates
    te_overlap = train_dates & test_dates
    ve_overlap = val_dates   & test_dates

    all_ok = True
    for pair, overlap in [
        ("Train/Val",  tv_overlap),
        ("Train/Test", te_overlap),
        ("Val/Test",   ve_overlap),
    ]:
        if overlap:
            print(f"  ERROR: {len(overlap)} overlapping dates in {pair} split!")
            all_ok = False

    if all_ok:
        print("  Overlap check: PASSED (no date leakage between splits)")

    print("=" * 60)


# ============================================================
# Run as script: print split summary and save CSVs
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("REGIMEX - DATA SPLIT")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    train_df, val_df, test_df = get_splits(df)

    # Optionally save splits for inspection
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/RELIANCE_train.csv",      index=False)
    val_df.to_csv(  "data/processed/RELIANCE_validation.csv", index=False)
    test_df.to_csv( "data/processed/RELIANCE_test.csv",       index=False)

    print("Saved split files:")
    print("  data/processed/RELIANCE_train.csv")
    print("  data/processed/RELIANCE_validation.csv")
    print("  data/processed/RELIANCE_test.csv")
