# RegimeX - Phase 2 Results (RELIANCE, single-stock)

Everything below is reproducible from the scripts in `src/`. All strategies are
executed through the same `RegimeXTradingEnv` (0.1% cost, volatility-scaled
slippage and market impact, Rs 100,000 start), so they face identical frictions.

## 1. Pipeline fixes made before Phase 2 could be trusted

| Issue found | Effect | Fix |
|---|---|---|
| `RELIANCE_regimes.csv` still held the original KMeans labels (31 High_Volatility days, all in train, none in val/test) | Regime-adaptive claim untestable out of sample | Ran the causal rolling-P90 relabel (`src/regime_relabel.py`), now reproducible from a saved copy of the KMeans labels |
| `trading_env.step()` valued the portfolio before and after the action at the **same** close | Reward saw only trading costs, never price moves; the Phase 1 agent chose SELL 100% of the time with zero reward | Reward is now measured close(t) to close(t+1) after the action |

High_Volatility coverage after relabelling: train 8.2%, validation 2.3% (thin,
genuinely calm period - documented, not forced), test 11.7%.

## 2. Setup

- Splits: train 2015-05-29 to 2022-10-25, validation to 2024-05-31, test 2024-06-03 to 2025-12-31.
- Baseline ladder: Buy & Hold, ARIMA(1,0,1) walk-forward, XGBoost next-day direction, PPO regime-blind, PPO regime-adaptive.
- PPO: 500k steps, 3 seeds per variant, results averaged over seeds. Two observation variants: **raw** (v1) and **scaled** (v2: z-scored features from train stats, one-hot regime).
- Significance: paired circular block bootstrap (block 10, 5,000 resamples) on daily returns; PPO tests use the seed-averaged return series.

## 3. Test-split results (393 trading days, 46 High_Volatility)

| Strategy | Total return | Sharpe | Sortino | Max drawdown | % time in market |
|---|---|---|---|---|---|
| Buy & Hold | +3.7% | 0.22 | 0.31 | -27.4% | 100% |
| ARIMA | -40.0% | -1.98 | -2.56 | -40.0% | 52% |
| XGBoost | -17.5% | -0.69 | -1.02 | -35.1% | 63% |
| PPO regime-blind (raw) | +4.4% | 0.21 | 0.31 | -18.3% | 52% |
| PPO regime-adaptive (raw) | +10.1% | 0.41 | 0.61 | -27.3% | 99% |
| PPO regime-blind (scaled) | +2.5% | 0.11 | 0.17 | -9.1% | 32% |
| PPO regime-adaptive (scaled) | +1.2% | 0.10 | 0.15 | -15.8% | 28% |

ARIMA and XGBoost trade 130-245 times over the period; costs and slippage
dominate their results.

## 4. Regime-stratified evaluation (test, mean daily return in bp)

| Strategy | Normal_Market | Weak_Bear | High_Volatility |
|---|---|---|---|
| Buy & Hold | -3.2 | -8.5 | +53.2 |
| PPO blind (raw) | -2.2 | -2.7 | +30.8 |
| PPO adaptive (raw) | +0.1 | -8.5 | +48.3 |
| PPO blind (scaled) | +0.0 | -2.8 | +14.4 |
| PPO adaptive (scaled) | -3.6 | +0.5 | +21.2 |

## 5. Statistical significance (test, adaptive minus blind)

| Variant | Sharpe diff [95% CI] | p | High_Volatility mean daily bp diff | p |
|---|---|---|---|---|
| raw | +0.08 [-0.32, +0.49] | 0.70 | +17.5 | 0.005 |
| scaled | -0.18 [-1.24, +0.89] | 0.75 | +6.8 | 0.22 |

The raw-variant High_Volatility difference is significant, but it is
explained by the adaptive agents being ~99% invested during a rising stretch
of the test period (they behave like Buy & Hold), while the blind agents held
cash about half the time. It is not evidence of regime-aware behaviour, and it
does not survive in the scaled variant.

## 6. Walk-forward validation (expanding window, test years 2021-2025, PPO 150k steps, 1 seed)

| Strategy | Mean total return | Mean Sharpe | Mean max DD |
|---|---|---|---|
| Buy & Hold | +10.9% | 0.59 | -17.2% |
| PPO regime-adaptive | +1.7% | 0.12 | -3.0% |
| PPO regime-blind | +1.5% | 0.11 | -8.2% |
| XGBoost | -12.4% | -0.71 | -21.1% |
| ARIMA | -25.6% | -1.73 | -28.4% |

Adaptive beat blind on Sharpe in 2 of 5 folds; pooled out-of-sample bootstrap
of the difference: p = 0.88. In many folds the PPO agents stayed entirely in
cash (Sharpe exactly 0). The 2021 and 2023 test years contain no
High_Volatility days at all.

## 7. Explainability (SHAP, adaptive raw agent, seed 0)

Mean |SHAP| on action probability: `Holdings` (0.11-0.38) and `Hurst` /
`Regime` (0.02-0.09) dominate; `Daily_Return` and `Rolling_Volatility` are
about 0.001. The agent's action mix by regime is BUY 76% / SELL 24% in
High_Volatility, BUY 99.6% in Normal_Market and HOLD 100% in Weak_Bear -
so the regime input does change behaviour, but mostly as a lever toward
"stay invested". The near-zero attribution of return and volatility motivated
the scaled-observation variant, which did not improve results.

## 8. Conclusions and limitations

- The regime pipeline is now sound and causal, with High_Volatility present in train and test.
- No configuration showed a statistically significant benefit from the regime-conditional reward on RELIANCE alone. This is reported as a negative result, not tuned away.
- PPO agents trained on ~1,800 daily bars of one stock tend to collapse to always-invested or always-cash policies; the training signal is weak relative to trading costs.
- The validation split has only 9 High_Volatility days, so validation regime tests (and their bootstrap intervals) are not meaningful.
- Walk-forward PPO used 150k steps and one seed per fold; it is a robustness check, not a full replication.
- Seed-to-seed variance is large (Sharpe std about 0.1-0.2), comparable to the effects being measured.
- The base Normal_Market / Weak_Bear labels for RELIANCE come from a KMeans fitted on the full 2015-2025 sample, so only the High_Volatility override is strictly causal. The four additional stocks (section 9) fit KMeans on the train period only.
- Pooling five stocks (section 9) was tried to address the small-data problem; the conclusion did not change.

## 9. Extension: agents trained on all five stocks

`src/build_multistock.py` builds regime datasets for TCS, HDFCBANK, ICICIBANK and
INFY (KMeans fitted on the train period only, plus the same causal rolling-P90
High_Volatility rule). `src/train_pooled.py` trains one agent on the pooled train
splits (~9,150 bars, a random stock per episode, 1M steps, 3 seeds per variant).
`src/evaluate_pooled.py` tests on every stock's held-out test split
(2024-06-03 to 2025-12-31): 1,965 stock-days, of which 205 are High_Volatility.

Equal-weight five-stock portfolio, test:

| Strategy | Total return | Sharpe |
|---|---|---|
| Buy & Hold | +10.2% | 0.52 |
| ARIMA | -39.4% | -3.92 |
| XGBoost (pooled) | -23.4% | -1.67 |
| PPO regime-blind (raw) | +2.7% | 0.21 |
| PPO regime-adaptive (raw) | +0.0% | 0.02 |
| PPO regime-blind (scaled) | +3.1% | 0.25 |
| PPO regime-adaptive (scaled) | +6.2% | 0.57 |

Adaptive minus blind, paired block bootstrap on stacked stock-days:

| Variant | Sharpe diff [95% CI] | p | High_Volatility mean daily bp diff | p |
|---|---|---|---|---|
| raw | -0.16 [-0.49, +0.13] | 0.31 | -9.4 | 0.001 |
| scaled | +0.17 [-0.07, +0.42] | 0.18 | -4.4 | 0.19 |

Findings: with pooled data and scaled inputs the adaptive agent roughly matches
Buy & Hold on risk-adjusted terms (Sharpe diff -0.01, p = 0.88) and is the best
PPO variant, but it is not significantly better than the regime-blind agent, and
no PPO variant beats Buy & Hold. On High_Volatility days every PPO variant
earns significantly less than Buy & Hold, because the test period's volatile
stretches were rallies and the agents were not fully invested. The
regime-conditioned reward therefore did not produce a demonstrable out-of-sample
advantage on either one stock or five.

## 10. Reproducing

```
python src/regime_relabel.py
python src/train_ppo.py --seed 0            # adaptive; add --blind, --scaled as needed
python src/evaluate.py                       # tables, bootstrap, figures -> results/
python src/walk_forward.py
python src/explain_shap.py
python src/build_multistock.py                # other four stocks
python src/train_pooled.py --scaled --seed 0  # add --blind for the blind agent
python src/evaluate_pooled.py
```
