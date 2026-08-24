# RegimeX — Project Plan & Alignment Check
**Repo:** github.com/AnyaK393/RegimeX
**Status check date:** Aug 24, 2026

---

## 1. Alignment vs. Our Redesigned Objectives

| Our Objective | Repo Status | Verdict |
|---|---|---|
| 0. EDA & statistical foundation before modeling | `eda.py` — returns, volatility, distributions, normalized performance | ✅ Done |
| 1. Indian-market data (NSE/BSE) | `yfinance` pull on RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY (2015–2025); dev/testing currently on RELIANCE only | ✅ Done (single-stock scope for now) |
| 2. Realistic execution — dynamic slippage & market impact | `trading_env.py` — 0.1% transaction cost, volatility-scaled slippage, market impact term, tested via `test_trading_env.py` | ✅ Done |
| 3. Regime detection inside the pipeline | `regime_features.py` (daily return, rolling volatility, Hurst exponent) → `change_point_analysis.py` (ruptures) → `regime_detection.py` (KMeans) → labeled as Normal / Weak Bear / High Volatility | ✅ Done — feeds into **state**, not yet into **reward** (see gap below) |
| **Regime-conditioned REWARD** (not just regime-aware state) | Current reward = `portfolio_value_t − portfolio_value_t-1`, i.e. plain PnL. Regime label is in the *observation*, but nothing in the reward function changes weight based on regime yet | ⚠️ **Gap** — this was our core novelty claim. Regime awareness in state ≠ regime-adaptive reward. Needs explicit reward reshaping when PPO is built. |
| 4. Baseline ladder (classical stats → ML → RL) + significance testing | Repo's planned baselines: Buy & Hold, possibly Random Trading only | ⚠️ **Gap** — no ARIMA/GARCH or XGBoost baseline planned, no bootstrap significance testing mentioned |
| 5. Explainability (SHAP) | Listed in upcoming stage, tools already in `requirements.txt` | 🔜 Planned, not started |
| 6. Regime-stratified evaluation (not just pooled metrics) | Repo lists standard Sharpe/Sortino/MDD/turnover metrics, but no mention of splitting results *by regime* | ⚠️ **Gap** — needs to be explicit in the backtesting script, not just an afterthought |

**Bottom line:** infra (data → EDA → regime detection → environment) is genuinely complete and well-built — that part of the pitch is accurate. But "only training is remaining" is an understatement. Three things from our objective set aren't in the repo's own TODO list: **regime-conditioned reward**, **baseline ladder**, and **regime-stratified evaluation**. Skipping these quietly loses the novelty argument we built the pitch around — without them this reverts to "PPO + Buy&Hold benchmark," which is exactly what the lit review said was already saturated.

---

## 2. What's Actually Left — Full Scope

| # | Task | How | Owner (fill in) |
|---|---|---|---|
| 1 | **Reshape reward function** to weight risk/turnover penalties by current regime (e.g. tighter drawdown penalty in High_Volatility, looser in Normal_Market) instead of flat PnL | Modify `trading_env.py`'s `step()` reward calculation; add regime-conditional coefficients (λ_risk, λ_fee) keyed off the regime label already in the observation | |
| 2 | **Build PPO agent** | `stable-baselines3` PPO on the existing Gymnasium env; hyperparameter tuning (learning rate, batch size, entropy coef) via TensorBoard logging | |
| 3 | **Train/Validation/Test split** — strict chronological, no shuffling | Slice `RELIANCE_regimes.csv` by date: e.g. 2015–2022 train, 2023 validation, 2024–2025 test | |
| 4 | **Add baseline ladder** | (a) Buy & Hold — already planned. (b) Add a simple ARIMA or GARCH volatility-only baseline. (c) Add XGBoost predicting next-day direction as a supervised sanity check. Run all three + PPO on identical splits | |
| 5 | **Backtest PPO agent** | Roll trained policy forward on held-out test period, log portfolio value, trades, regime at each step | |
| 6 | **Compute performance metrics** | Total/annualized return, Sharpe, Sortino, MDD, turnover, win rate — standard `pandas`/`numpy` calc from the equity curve | |
| 7 | **Regime-stratified reporting** | Don't just report one pooled Sharpe — split test-period results by the regime label already generated (Normal / Weak_Bear / High_Volatility) and report metrics per regime, plus regime-blind vs. regime-aware comparison | |
| 8 | **Statistical significance testing** | Bootstrap resampling on returns to get confidence intervals on Sharpe/CR differences between PPO and Buy & Hold — proves the improvement (if any) is real, not noise | |
| 9 | **SHAP explainability** | Apply SHAP (or a simpler feature-attribution method compatible with the PPO policy network) over `[Daily Return, Rolling Volatility, Hurst, Regime, Holdings]` to explain individual BUY/HOLD/SELL decisions | |
| 10 | **Walk-forward validation** | Repeat train→validate→test in rolling windows (not just one static split) to confirm robustness across different historical periods | |
| 11 | **Final visualizations & write-up** | Equity curves, regime overlays, per-regime performance tables, SHAP summary plots, final report/paper draft | |

---

## 3. Suggested Sequencing

1. **Reward reshaping (Task 1)** — do this *before* PPO training starts, since it changes what the agent optimizes for. Retrofitting it after training means retraining anyway.
2. **PPO build + train (Tasks 2–3)** in parallel with **baseline ladder (Task 4)** — different team members, no dependency between them.
3. **Backtest + metrics + regime-stratified reporting (Tasks 5–7)** once PPO is trained.
4. **Significance testing + SHAP (Tasks 8–9)** — can start as soon as Task 5 produces an equity curve.
5. **Walk-forward validation (Task 10)** last, once the single-split pipeline is proven to work.
6. **Write-up (Task 11)** throughout, finalized at the end.

---

## 4. One thing to raise with your teammate

Worth a quick, non-confrontational check-in: confirm whether the reward function is *intended* to become regime-conditioned during PPO development, or whether it was left as plain PnL deliberately. If it's the latter, that's the one conversation worth having before training starts — it's the difference between "PPO trading bot" and "the regime-adaptive contribution we pitched to faculty."
