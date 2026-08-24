import pandas as pd
import numpy as np

# ============================================================
# REGIMEX - HYBRID REGIME RE-LABELLING  (Path C)
# ============================================================
#
# Methodology (report this verbatim in the methodology section):
#
#   "KMeans clustering on [Daily Return, Rolling Volatility,
#    Hurst Exponent] identifies the base Normal_Market /
#    Weak_Bear regime structure. The High_Volatility label is
#    additionally governed by a causal trailing-window percentile:
#    a day is classified as High_Volatility if its 20-day rolling
#    volatility exceeds the 90th percentile of that measure over
#    the preceding 504 trading days (~2 years). An expanding window
#    from the dataset start is used where fewer than 252 observations
#    are available. This approach is fully causal — no future data
#    is used on any given day — and adapts naturally to structural
#    changes in RELIANCE's volatility regime rather than anchoring
#    permanently to a single historical spike."
#
# Known trade-off (disclose in paper):
#   The regime label is period-relative, not absolute. A
#   'High_Volatility' day in a low-vol period may have lower
#   absolute volatility than a 'Normal_Market' day during COVID.
#   This is standard practice in adaptive-volatility-regime
#   literature and is a deliberate methodological choice.
#
# Reference: Ang & Bekaert (2002), Hamilton (1989) HMM regimes;
#   rolling-percentile vol regimes used in Lopez de Prado (2018).
# ============================================================

INPUT_PATH   = "data/processed/RELIANCE_regimes.csv"
OUTPUT_PATH  = "data/processed/RELIANCE_regimes.csv"

TRAIN_END_DATE = "2022-10-25"
VAL_END_DATE   = "2024-05-31"

TRAILING_WINDOW = 504    # ~2 trading years
MIN_PERIODS     = 252    # ~1 trading year minimum before using trailing; else use expanding
PERCENTILE      = 90

HV_REGIME_INT = 2
HV_REGIME_STR = "High_Volatility"

print("=" * 60)
print("REGIMEX - CAUSAL ROLLING REGIME RE-LABELLING")
print("=" * 60)

# ------------------------------------------------------------
# Load dataset (use regime_features so we start from clean KMeans)
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nLoaded: {len(df)} rows  ({df['Date'].min().date()} -> {df['Date'].max().date()})")

# ------------------------------------------------------------
# Pre-fix distribution
# ------------------------------------------------------------
print("\nBEFORE re-labelling:")
print(df["Market_Regime"].value_counts().to_string())

# ------------------------------------------------------------
# Compute CAUSAL trailing P90 of Rolling_Volatility
#
# Two passes:
#   1. Trailing 504-day window  (primary, causal)
#   2. Expanding from start     (fallback for early rows)
#
# On day t, the quantile is computed over [t-503, t-1] only.
# The closed="left" shift ensures day t is NOT included.
# We use .shift(1) so the window ends the day BEFORE today.
# ------------------------------------------------------------

vol = df["Rolling_Volatility"]

# Expanding causal: quantile over all rows strictly before t
# (shift(1) ensures today is excluded)
vol_shifted = vol.shift(1)

p90_expanding = (
    vol_shifted
    .expanding(min_periods=2)
    .quantile(PERCENTILE / 100)
)

# Trailing 504-day causal window
p90_trailing = (
    vol_shifted
    .rolling(window=TRAILING_WINDOW, min_periods=MIN_PERIODS)
    .quantile(PERCENTILE / 100)
)

# Use trailing where available, expanding as fallback for early rows
df["vol_p90_causal"] = p90_trailing.combine_first(p90_expanding)

# ------------------------------------------------------------
# Apply High_Volatility override
# ------------------------------------------------------------
hv_mask = df["Rolling_Volatility"] >= df["vol_p90_causal"]

original_hv = (df["Market_Regime"] == HV_REGIME_STR).sum()

df.loc[hv_mask, "Regime"]        = HV_REGIME_INT
df.loc[hv_mask, "Market_Regime"] = HV_REGIME_STR

print(f"\nCausal P90 threshold (trailing 504-day window / expanding fallback)")
print(f"  Days reclassified to High_Volatility: {hv_mask.sum()} (was {original_hv})")

# ------------------------------------------------------------
# Post-fix distribution
# ------------------------------------------------------------
print("\nAFTER re-labelling:")
print(df["Market_Regime"].value_counts().to_string())

# ------------------------------------------------------------
# Per-split coverage table — THE critical check
# ------------------------------------------------------------
train_mask = df["Date"] <= TRAIN_END_DATE
val_mask   = (df["Date"] > TRAIN_END_DATE) & (df["Date"] <= VAL_END_DATE)
test_mask  = df["Date"] > VAL_END_DATE

splits = {
    "Train":      df[train_mask],
    "Validation": df[val_mask],
    "Test":       df[test_mask],
}

print("\n" + "=" * 60)
print("HIGH_VOLATILITY COVERAGE PER SPLIT")
print("=" * 60)

warnings = []
for name, split_df in splits.items():
    n_total = len(split_df)
    n_hv    = (split_df["Market_Regime"] == HV_REGIME_STR).sum()
    n_nm    = (split_df["Market_Regime"] == "Normal_Market").sum()
    n_wb    = (split_df["Market_Regime"] == "Weak_Bear").sum()
    pct     = 100 * n_hv / n_total if n_total > 0 else 0

    flag = ""
    if pct < 5:
        flag = "  <-- THIN (< 5%)"
        warnings.append(f"{name} has only {pct:.1f}% HV coverage")

    print(
        f"  {name:12s}  n={n_total:5d}  "
        f"HV={n_hv:4d} ({pct:5.1f}%)"
        f"  Normal={n_nm:5d}  WB={n_wb:5d}"
        f"{flag}"
    )

if warnings:
    print("\n  WARNINGS:")
    for w in warnings:
        print(f"    - {w}")
    print("  This reflects genuine market behaviour, not a labelling error.")
    print("  Document in the paper; do not force higher coverage.")
else:
    print("\n  OK: HV is reasonably distributed across all splits.")

# ------------------------------------------------------------
# Spot-check: causal threshold evolves sensibly over time
# ------------------------------------------------------------
print("\nCalibration spot-check (threshold evolution):")
for date in ["2017-01-03", "2019-01-02", "2020-03-23", "2022-01-03", "2023-06-01", "2025-01-02"]:
    row = df[df["Date"] == date]
    if len(row):
        r = row.iloc[0]
        print(
            f"  {date}  vol={r['Rolling_Volatility']:.5f}"
            f"  causal_P90={r['vol_p90_causal']:.5f}"
            f"  label={r['Market_Regime']}"
        )

# ------------------------------------------------------------
# Drop helper column and save
# ------------------------------------------------------------
df = df.drop(columns=["vol_p90_causal"])
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")
print("\n" + "=" * 60)
print("RE-LABELLING COMPLETE")
print("=" * 60)
