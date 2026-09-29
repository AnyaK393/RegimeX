import numpy as np
import pandas as pd

# ============================================================
# REGIMEX - PERFORMANCE METRICS
# ============================================================
#
# All metrics are computed from a per-step rollout log with columns
#   date, regime, action, value, prev_value, trade_value, shares
# produced by strategies.rollout(). The return on row t is the
# realised close(t) -> close(t+1) portfolio return after the action
# taken on day t, and `regime` is the regime known at decision time t
# (causal attribution: a return is credited to the regime the agent
# actually saw when it acted).
#
# Risk-free rate is taken as 0. 252 trading days per year.
# ============================================================

TRADING_DAYS = 252
REGIMES = ["Normal_Market", "Weak_Bear", "High_Volatility"]


def daily_returns(log: pd.DataFrame) -> pd.Series:
    return pd.Series(
        (log["value"] / log["prev_value"] - 1.0).to_numpy(),
        index=pd.to_datetime(log["date"]),
    )


def total_return(r) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.prod(1.0 + r) - 1.0) if len(r) else np.nan


def annualized_return(r) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return np.nan
    growth = np.prod(1.0 + r)
    return float(growth ** (TRADING_DAYS / len(r)) - 1.0) if growth > 0 else -1.0


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0


def sortino(r) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 1e-12 else 0.0


def max_drawdown(r) -> float:
    """Maximum peak-to-trough decline of the compounded equity curve (<= 0)."""
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float((equity / peak - 1.0).min())


def round_trips(log: pd.DataFrame) -> list:
    """
    Pairs each executed BUY with the next executed SELL. Returns a list of
    round-trip P&L values (proceeds after costs minus outlay incl. costs).
    A position still open at the end is not counted (it is unrealised).
    """
    pnl, entry = [], None
    for a, tv in zip(log["action"], log["trade_value"]):
        if tv <= 0:
            continue
        if a == 1 and entry is None:
            entry = tv
        elif a == 2 and entry is not None:
            pnl.append(tv - entry)
            entry = None
    return pnl


def summarize(log: pd.DataFrame) -> dict:
    r = daily_returns(log).to_numpy()
    in_market = (log["shares"].to_numpy() > 0)
    trips = round_trips(log)
    avg_equity = float(log["value"].mean())

    return {
        "total_return":   total_return(r),
        "ann_return":     annualized_return(r),
        "sharpe":         sharpe(r),
        "sortino":        sortino(r),
        "max_drawdown":   max_drawdown(r),
        # traded notional / average equity, over the whole period
        "turnover":       float(log["trade_value"].sum() / avg_equity),
        "n_trades":       int((log["trade_value"] > 0).sum()),
        "win_rate_trades": float(np.mean(np.array(trips) > 0)) if trips else np.nan,
        "win_rate_days":  float(np.mean(r[in_market] > 0)) if in_market.any() else np.nan,
        "pct_in_market":  float(in_market.mean()),
    }


def regime_summary(log: pd.DataFrame) -> pd.DataFrame:
    """Metrics computed separately on the days falling in each regime."""
    r = daily_returns(log)
    regime = log["regime"].to_numpy()
    in_market = (log["shares"].to_numpy() > 0)
    rows = []
    for reg in REGIMES:
        m = regime == reg
        rr = r.to_numpy()[m]
        rows.append({
            "regime":        reg,
            "n_days":        int(m.sum()),
            "cum_return":    total_return(rr),
            "mean_daily_bp": float(rr.mean() * 1e4) if len(rr) else np.nan,
            "sharpe":        sharpe(rr),
            "max_drawdown":  max_drawdown(rr),
            "win_rate_days": float(np.mean(rr[in_market[m]] > 0)) if in_market[m].any() else np.nan,
            "pct_in_market": float(in_market[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Bootstrap significance testing
# ------------------------------------------------------------

def _block_indices(n: int, block: int, rng) -> np.ndarray:
    """Circular block bootstrap indices (preserves short-range dependence)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def paired_bootstrap(
    r_a, r_b, stat, n_boot=5000, block=10, seed=0,
) -> dict:
    """
    Paired circular-block bootstrap of stat(r_a) - stat(r_b).
    The same resampled day indices are applied to both series, so the
    comparison is on identical market days.
    Returns point estimate, 95% percentile CI and a two-sided p-value
    (2 * min(P(diff <= 0), P(diff >= 0))).
    """
    r_a, r_b = np.asarray(r_a, float), np.asarray(r_b, float)
    n = len(r_a)
    assert n == len(r_b), "series must be aligned"
    if n < 5:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": n}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(n, min(block, n), rng)
        diffs[i] = stat(r_a[idx]) - stat(r_b[idx])

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "diff":    float(stat(r_a) - stat(r_b)),
        "ci_low":  float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n":       n,
    }
