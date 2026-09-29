# RegimeX: Does Regime Awareness Help a Reinforcement-Learning Trader? Evidence from Indian Equities

*Draft research report. Detailed tables and caveats are in [PHASE2_RESULTS.md](PHASE2_RESULTS.md). All numbers are reproducible from the scripts in `src/`.*

## Abstract

We ask whether a reinforcement-learning (RL) trading agent that conditions its risk preference on the market regime achieves better out-of-sample risk-adjusted performance than an otherwise identical regime-blind agent. Using daily data for five Nifty stocks (2015-2025), we build a causal regime labelling (KMeans base regimes plus a trailing-percentile High_Volatility rule), a cost-aware Gymnasium trading environment, PPO agents, a baseline ladder (Buy & Hold, ARIMA, XGBoost, constant exposure, volatility targeting), and a statistical evaluation using paired block bootstrap and walk-forward validation. With the original all-in BUY/HOLD/SELL formulation we find **no significant benefit**. Diagnosing why, we identify three design flaws and introduce a position-sizing formulation with a dense mean-variance regime-conditioned reward. Under this formulation the agents visibly manage risk by regime (about 13-16 % exposure in High_Volatility versus 99 % for Buy & Hold) and roughly halve drawdown, and the adaptive agent leads the blind agent on Sharpe (+0.29, p = 0.096), a suggestive but not statistically significant difference. Neither agent beats a naive constant-50 %-exposure rule on Sharpe. A walk-forward selection of the risk-aversion scale on pre-2024 data (chosen by a rule fixed in advance) did not transfer to the test window: with the selected setting adaptive and blind agents are statistically indistinguishable (Sharpe difference -0.04, p = 0.79), although the adaptive agent does hold significantly less in High_Volatility (p = 0.004). We report this as a mixed, largely negative result.

## 1. Research question

> Can a regime-aware RL trading agent achieve better risk-adjusted performance than a regime-blind trading strategy?

The comparison is controlled: the two agents share the same data, environment, costs, algorithm, seeds and, in the final design, the same *average* risk aversion; they differ only in whether the risk penalty depends on the regime.

## 2. Data and regime labelling

Daily OHLCV for RELIANCE, TCS, HDFCBANK, ICICIBANK and INFY (yfinance, 2015-01-01 to 2025-12-31), split chronologically: train to 2022-10-25, validation to 2024-05-31, test from 2024-06-03 (about 70/15/15). Features: daily return, 20-day rolling volatility and a 100-day Hurst exponent.

**Regime detection is a hybrid method.** KMeans (k = 3) on the standardised features establishes the base Normal_Market / Weak_Bear structure. **High_Volatility is not a cluster boundary but a causal rule:** a day is High_Volatility if its rolling volatility is at or above the 90th percentile of the preceding 504 trading days (expanding window, requiring at least 126 prior days, before that). The reason is empirical: the pure-KMeans High_Volatility cluster held 31 of 2,617 days, all in the 2020 COVID crash and none in validation or test, which makes the research question untestable. Per-split percentiles were rejected as look-ahead; the trailing percentile uses only past data. Coverage after relabelling (RELIANCE): train 8.2 %, validation 2.3 %, test 11.7 %. The regime label is period-relative rather than absolute.

Caveat: for RELIANCE the KMeans base labels were fitted on the full sample, so only the High_Volatility rule is strictly causal; for the other four stocks KMeans was fitted on the training period only.

## 3. Environment and methods

The environment charges 0.1 % transaction cost, volatility-scaled slippage and market impact, starting from Rs 100,000. Reward is the realised close-to-close log-return after the action.

* **v1:** actions BUY/HOLD/SELL (all-in / all-out); reward = log-return minus a regime-weighted drawdown penalty minus a turnover penalty.
* **v3:** the agent selects a target exposure in {0, 25, 50, 75, 100} %; reward = 100 x [log-return - 0.5 x gamma(regime) x r^2], with gamma = 3 / 9 / 18 for Normal / Weak_Bear / High_Volatility (fixed a priori). The regime-blind agent uses the train-frequency-weighted mean gamma so average risk aversion is equal.

PPO (Stable-Baselines3, 64-64 networks, 500k-1M steps) is trained with three seeds per variant. Baselines run through the identical environment: Buy & Hold, ARIMA(1,0,1) (walk-forward), XGBoost next-day direction, Constant 50 %, and volatility targeting. Evaluation reports return, Sharpe, Sortino, maximum drawdown, turnover and win rate, overall and per regime. Significance uses a paired circular block bootstrap (block 10, 3-5k resamples). Robustness uses expanding-window walk-forward validation. Explanations use SHAP.

## 4. Two corrections that mattered

1. The stored regime file still contained the original KMeans labels; the causal relabelling had never been applied.
2. The v1 environment valued the portfolio before and after each action at the *same* close, so price moves never entered the reward; the first trained agent chose SELL 100 % of the time with zero reward. Reward is now measured close(t) to close(t+1).

## 5. Results

**v1 (all-in), test split.** The regime-adaptive agent's return (+10.1 % on RELIANCE) is explained by being about 99 % invested through a rising period; the adaptive-versus-blind Sharpe difference is not significant (p = 0.70). Walk-forward over five test years: adaptive wins on Sharpe in 2 of 5 folds (pooled p = 0.88); many folds have agents fully in cash. Pooling all five stocks and scaling inputs did not change the conclusion (adaptive minus blind Sharpe +0.17, p = 0.18; no PPO variant beats Buy & Hold).

**Diagnosis.** The turnover penalty was about 20 times a typical daily return and double-counted costs; the drawdown term is zero when in cash or at a peak, so the two agents received nearly the same reward; and all-in actions cannot express "take less risk when volatile".

**v3 (position sizing), five-stock equal-weight portfolio, test:**

| Strategy | Return | Sharpe | Max drawdown |
|---|---|---|---|
| Buy & Hold | +10.2 % | 0.52 | -15.3 % |
| Constant 50 % | +5.4 % | 0.50 | -8.1 % |
| Vol-target | +2.6 % | 0.20 | -14.2 % |
| PPO-v3 regime-blind | -1.1 % | -0.08 | -8.2 % |
| PPO-v3 regime-adaptive | +1.4 % | 0.18 | -7.8 % |

Mean exposure by regime (adaptive / blind): Normal 0.66 / 0.53, Weak_Bear 0.31 / 0.26, High_Volatility 0.16 / 0.13. The adaptive-minus-blind exposure difference is significant in Normal and Weak_Bear (p < 0.001), i.e. with equal average risk aversion the regime-conditioned agent spends more risk in calm markets instead of being uniformly cautious. Adaptive minus blind Sharpe: +0.29 (95 % CI [-0.05, +0.67], p = 0.096). Adaptive drawdown is about half of Buy & Hold's (p = 0.001) but not different from Constant 50 %.

**Explainability.** SHAP on the expected target exposure attributes the decision mainly to the regime (mean |SHAP| 0.121), the current position (0.106) and volatility (0.096); previous-day return (0.010) and Hurst (0.013) barely matter. The regime input raises exposure in Normal_Market (+0.13) and lowers it in Weak_Bear (-0.12) and High_Volatility (-0.11).

**Hyper-parameter selection without touching the test set.** The base risk-aversion scale was selected on walk-forward folds for 2019-2023 with a rule fixed in advance (grid 0.75 / 1.5 / 3.0). The rule chose 0.75 (mean adaptive Sharpe 0.84 vs 0.74), but lower risk aversion improved the blind agent equally, so the adaptive-versus-blind gap did not widen (mean -0.08 Sharpe). Evaluated once on the test window, the selected setting gave adaptive Sharpe 0.13 versus blind 0.12 (difference -0.04, p = 0.79) and was slightly worse than the default on drawdown (-11.8 % vs -7.8 %). The improvement did not transfer out of sample.

## 6. Discussion

The original hypothesis, that a regime-conditioned reward yields a clear out-of-sample advantage, is **not supported**. The most defensible positive statements are behavioural and risk-related: with a well-posed reward, agents differentiate regimes as intended, and they cut drawdown substantially relative to Buy & Hold. The performance advantage of the adaptive over the blind agent is suggestive (p about 0.10) but not conclusive, and both RL agents are over-cautious and do not beat simple constant de-risking in this test window. The test window was also unusually favourable to Buy & Hold: its High_Volatility stretches were rallies, so any strategy that reduces exposure in volatile regimes is penalised there.

## 7. Limitations

* One test window of 393 days per stock; effects are small relative to seed-to-seed variance (Sharpe standard deviation about 0.1-0.2).
* The test split was viewed while evaluating v1/v2, although it was never used for tuning; v3 design choices were made from training-behaviour diagnostics and fixed before viewing v3 results. Any further tuning uses only pre-2024 walk-forward folds.
* The validation split has 9 High_Volatility days for RELIANCE, so regime tests there are not meaningful.
* KMeans base labels for RELIANCE are not strictly causal (Section 2).
* Regime labels are relative to the trailing two years; a "High_Volatility" day in a calm period can be less volatile in absolute terms than a "Normal" day in a crisis.
* Single market (five large-cap Indian stocks), daily frequency, no short selling, Yahoo Finance data.

## 8. Reproducibility

Scripts, seeds, splits and tables are listed in the README and in `PHASE2_RESULTS.md` (section 11). Tests: `src/test_regime_reward.py`, `src/test_exposure_env.py`.

*Not financial advice; academic research only.*
