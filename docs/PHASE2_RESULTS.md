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

## 10. v3: position-sizing agent with a dense mean-variance regime reward

**Why.** Looking at why v1/v2 collapsed to always-invested or always-cash, three design flaws were identified
(`src/exposure_env.py` documents them):

1. The reward charged `0.01 x turnover` per trade, about 20x a typical daily return, and double-counted costs already deducted from portfolio value. Removed.
2. The risk term (`lambda x drawdown-from-peak`) is zero whenever the agent is in cash or at a new high, so adaptive and blind agents got nearly the same reward. Replaced by a dense per-step penalty `0.5 x gamma(regime) x r_t^2`.
3. All-in/all-out actions cannot express "take less risk in a volatile regime". The agent now picks a target exposure in {0, 25, 50, 75, 100}% (5% no-trade band).

Design choices fixed **before** looking at results and not tuned: gamma ratio Normal : Weak_Bear : High_Volatility = 1 : 3 : 6
(gamma = 3 / 9 / 18, following the ordering already used in v1); the regime-blind agent uses the
train-frequency-weighted mean gamma (7.72), so both agents have identical *average* risk aversion and differ only in whether
it depends on the regime. Scaled observations, pooled five-stock training (1M steps, 3 seeds each). Validation was run first
as a sanity check, then the test split once. (Note: the test split had already been viewed while evaluating v1/v2, though it
was not used to tune anything; v3 was motivated by diagnostics of the training behaviour and reward design.)

Added baselines: Constant 50% exposure and a volatility-targeting rule (`exposure = clip(1% / rolling vol, 0, 1)`).

**Test, equal-weight five-stock portfolio (mean over 3 seeds for PPO):**

| Strategy | Total return | Sharpe | Max drawdown | Ann. vol |
|---|---|---|---|---|
| Buy & Hold | +10.2% | 0.52 | -15.3% | 14.0% |
| Constant 50% | +5.4% | 0.50 | -8.1% | 7.2% |
| Vol-target | +2.6% | 0.20 | -14.2% | 11.2% |
| PPO-v3 regime-blind | -1.1% | -0.08 | -8.2% | 5.4% |
| PPO-v3 regime-adaptive | +1.4% | 0.18 | -7.8% | 6.6% |

**Behaviour: mean exposure by regime (test).** This is the first configuration where the agents visibly manage risk by regime:

| Strategy | Normal_Market | Weak_Bear | High_Volatility |
|---|---|---|---|
| Buy & Hold | 0.99 | 0.99 | 0.99 |
| Vol-target | 0.87 | 0.84 | 0.55 |
| PPO-v3 regime-blind | 0.53 | 0.26 | 0.13 |
| PPO-v3 regime-adaptive | 0.66 | 0.31 | 0.16 |

Adaptive minus blind exposure: +0.13 in Normal_Market (95% CI [0.10, 0.15], p < 0.001) and +0.06 in Weak_Bear
([0.03, 0.09], p < 0.001); no significant difference in High_Volatility (both agents are ~85% de-risked there).
So with equal average risk aversion, the regime-conditioned reward makes the agent take *more* risk in calm markets, rather than
less in volatile ones (the regime-blind agent is over-cautious everywhere).

**Adaptive vs blind (paired block bootstrap, portfolio level, n = 393):** Sharpe +0.29 [-0.05, +0.67], p = 0.096;
total return +2.5 pts, p = 0.25; max drawdown no different (p = 0.44). On validation the sign agrees (+0.06) but is far from significant (p = 0.82).
The direction is consistent, but the evidence is suggestive, not conclusive at the 5% level.

**Versus simple baselines:** the adaptive agent has a significantly shallower drawdown than Buy & Hold (-7.8% vs -15.3%, p = 0.001)
and than Vol-target (p < 0.001), but it does **not** beat Constant 50% (same drawdown, lower Sharpe 0.18 vs 0.50, p = 0.31) and its
Sharpe is not significantly different from Buy & Hold (p = 0.29). On High_Volatility days it earns significantly less than Buy & Hold and
Constant 50% (p < 0.001) because the test period's volatile stretches were rallies.

**Reading.** The redesign fixed the *behavioural* problem (agents now differentiate regimes and control risk) and gave the first
directionally positive, though not significant, adaptive-vs-blind result. It did not produce an agent that beats naive
de-risking in this test window, mainly because the agents are over-cautious (validation shows the same pattern, with vol-target
Sharpe 0.73 vs 0.16 for adaptive). Further tuning of gamma or exposure levels on this test window would be overfitting to it; a fair next
step is to select those on validation or via walk-forward, on more history.

## 11. Reproducing

```
python src/regime_relabel.py
python src/train_ppo.py --seed 0            # adaptive; add --blind, --scaled as needed
python src/evaluate.py                       # tables, bootstrap, figures -> results/
python src/walk_forward.py
python src/explain_shap.py
python src/build_multistock.py                # other four stocks
python src/train_pooled.py --scaled --seed 0  # add --blind for the blind agent
python src/evaluate_pooled.py
python src/train_pooled.py --exposure --scaled --seed 0   # v3; add --blind for the blind agent
python src/evaluate_v3.py --split validation
python src/evaluate_v3.py --split test
```
