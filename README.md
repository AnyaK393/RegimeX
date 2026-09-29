
# RegimeX 📈🤖

## Regime-Aware Reinforcement Learning Trading System

RegimeX is an AI-powered quantitative trading research project that explores whether **market regime awareness can improve reinforcement learning-based trading strategies** in Indian equity markets.

The system combines **Data Science, Financial Analytics, Machine Learning, Reinforcement Learning, and Explainable AI** to build an adaptive trading agent that learns to make **Buy / Hold / Sell** decisions under changing market conditions.

> **⚠️ Academic Project:** RegimeX is developed for research and educational purposes only. It is not financial advice and should not be used for real-world trading.

---

## 🎯 Research Objective

Financial markets constantly change between different conditions such as stable, trending, bearish, and highly volatile periods.

RegimeX investigates:

> **Can a regime-aware Reinforcement Learning trading agent achieve better risk-adjusted performance than a regime-blind trading strategy?**

---

## 🧠 Project Pipeline

```text
Historical NSE Stock Data
          ↓
Data Collection
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Daily Returns
Rolling Volatility
Hurst Exponent
          ↓
Change Point Detection
          ↓
Market Regime Detection
          ↓
Regime Interpretation & Labelling
          ↓
Gymnasium Trading Environment
          ↓
Transaction Costs
Dynamic Slippage
Market Impact
          ↓
PPO Reinforcement Learning Agent
          ↓
Backtesting
          ↓
Performance Evaluation
          ↓
SHAP Explainability
```

---

# 📊 Dataset

Historical Indian equity market data is collected using **Yahoo Finance through `yfinance`**.

### Selected Stocks

- RELIANCE
- TCS
- HDFCBANK
- ICICIBANK
- INFY

The current development and trading-environment pipeline is primarily being tested on:

**RELIANCE**

### Time Period

**2015-01-01 → 2025-12-31**

### Dataset Features

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Trading Volume

### Storage

```text
data/
├── raw/
└── processed/
```

---

# 🔹 1. Data Collection

Historical stock data is downloaded using `yfinance`.

Main script:

```text
src/data_collection.py
```

The raw datasets are stored inside:

```text
data/raw/
```

---

# 🔹 2. Data Cleaning

The collected data is cleaned and validated before further analysis.

Operations include:

- Missing value checking
- Duplicate checking
- Date conversion
- Data validation
- Processed dataset generation

Main script:

```text
src/data_cleaning.py
```

Processed datasets are stored inside:

```text
data/processed/
```

---

# 🔹 3. Exploratory Data Analysis

EDA is performed to understand the behaviour of the selected stocks.

Analysis includes:

- Price statistics
- Daily returns
- Return distributions
- Volatility
- Price movement
- Normalized stock performance
- Market behaviour

Main script:

```text
src/eda.py
```

---

# 🔹 4. Feature Engineering

Three important market features are currently used for regime detection.

## Daily Return

Daily return measures the percentage change in price.

```text
Daily Return =
(Current Price - Previous Price) / Previous Price
```

---

## Rolling Volatility

Rolling volatility measures the variability of returns over a moving window.

Higher volatility indicates greater market uncertainty and risk.

---

## Hurst Exponent

The Hurst exponent is used to identify the persistence and behaviour of market movements.

| Hurst Value | Interpretation |
|---|---|
| H < 0.5 | Mean-reverting behaviour |
| H ≈ 0.5 | Random / weakly persistent behaviour |
| H > 0.5 | Trending behaviour |

Main script:

```text
src/regime_features.py
```

Generated dataset:

```text
data/processed/RELIANCE_regime_features.csv
```

---

# 🔹 5. Change Point Detection

Change-point analysis is used to identify points where the statistical behaviour of the market changes.

The project uses the `ruptures` library for this analysis.

Main script:

```text
src/change_point_analysis.py
```

This provides additional insight into structural changes in market behaviour.

---

# 🔹 6. Market Regime Detection

RegimeX uses **KMeans clustering** to identify different market conditions.

### Input Features

- Daily Return
- Rolling Volatility
- Hurst Exponent

The clustering algorithm groups observations with similar market characteristics.

Main script:

```text
src/regime_detection.py
```

---

# 📌 Current Market Regimes

The detected clusters are interpreted as:

| Market Regime | Description |
|---|---|
| Normal_Market | Relatively stable market conditions |
| Weak_Bear | Weak or negative market behaviour |
| High_Volatility | Strong market fluctuations and elevated risk |

Additional interpretation is performed using:

```text
src/regime_interpretation.py
```

Regime labels are added using:

```text
src/add_regime_labels.py
```

Final dataset:

```text
data/processed/RELIANCE_regimes.csv
```

---

# 🔹 7. Trading Environment

A custom trading environment has been implemented using **Gymnasium**.

The environment simulates an investor trading RELIANCE using historical market data.

Main implementation:

```text
src/trading_env.py
```

A previous experimental implementation is also retained:

```text
src/trading_environment.py
```

---

# 💰 Initial Portfolio

The environment starts with:

**₹100,000**

The portfolio tracks:

- Cash
- Shares
- Stock price
- Portfolio value

---

# 🎮 Trading Actions

The agent has three possible actions:

```text
0 → HOLD
1 → BUY
2 → SELL
```

---

# 🧠 State Representation

The trading agent observes the following state:

```text
[
    Daily Return,
    Rolling Volatility,
    Hurst Exponent,
    Market Regime,
    Current Holdings
]
```

This allows the agent to consider both the current market condition and its existing portfolio position.

---

# 💸 Trading Costs & Market Realism

The trading environment includes realistic trading frictions.

## Transaction Cost

A **0.1% transaction cost** is applied when trades are executed.

---

## Dynamic Slippage

Slippage changes according to market volatility.

The execution price is adjusted using:

```text
Slippage =
Base Slippage +
Volatility × Slippage Multiplier
```

This means more volatile markets result in greater execution uncertainty.

---

## Market Impact

Market impact is also incorporated into the execution price.

This simulates the effect of a trade influencing the effective price at which the order is executed.

These mechanisms make the environment more realistic than assuming perfect market execution.

---

# 🎁 Reward Function

The reward is the realised log-return of the portfolio from the close of day *t* to the close of day *t+1*
(after the action), minus a regime-conditional risk penalty. Transaction costs, slippage and market impact are
deducted from portfolio value, so they reduce the reward automatically.

**v1 (`src/trading_env.py`)** – BUY / HOLD / SELL, all-in or all-out:

```text
Reward = ln(V[t+1] / V[t]) - lambda_risk(regime) * drawdown - lambda_fee * turnover
```

**v3 (`src/exposure_env.py`)** – the agent chooses a target exposure (0 / 25 / 50 / 75 / 100 % of equity) and is
penalised by a dense mean-variance term whose risk aversion depends on the regime:

```text
Reward = 100 * [ ln(V[t+1] / V[t]) - 0.5 * gamma(regime) * r[t]^2 ]
gamma : Normal_Market = 3,  Weak_Bear = 9,  High_Volatility = 18
```

The regime-blind control uses the train-frequency-weighted mean gamma, so both agents have the same *average* risk
aversion and differ only in whether it depends on the regime. See `docs/PHASE2_RESULTS.md` for why v3 was introduced.

---

# 🧪 Gymnasium Environment Testing

The environment has been successfully tested using:

```text
src/test_trading_env.py
```

The environment successfully supports:

- Environment initialization
- Observation generation
- BUY action
- HOLD action
- SELL action
- Portfolio tracking
- Reward calculation
- Transaction costs
- Dynamic slippage
- Market impact
- Regime information

The test action sequence used was:

```text
BUY → HOLD → HOLD → SELL → HOLD
```

The environment currently uses:

```text
Action Space: Discrete(3)
Observation Space: Box(...)
```

---

# 🤖 8. PPO Reinforcement Learning

## ✅ Implemented

A **PPO (Proximal Policy Optimization)** trading agent (Stable-Baselines3) learns trading decisions instead of
using manually defined rules. Agents are trained per regime-adaptive / regime-blind variant, with several seeds,
either on RELIANCE alone or pooled across all five stocks (`src/train_ppo.py`, `src/train_pooled.py`).

The learning loop is:

```text
Market State
      ↓
PPO Agent
      ↓
BUY / HOLD / SELL
      ↓
Trading Environment
      ↓
Portfolio Update
      ↓
Reward
      ↓
PPO Policy Update
      ↓
Repeat
```

The goal is for the agent to learn how different market regimes affect trading decisions.

---

# 📈 9. Backtesting

After training, every strategy is evaluated through the same trading environment on chronological
validation / test data that was not used during training (`src/evaluate.py`, `src/evaluate_pooled.py`, `src/evaluate_v3.py`).

Performance metrics will include:

### Return Metrics

- Total Return
- Annualized Return

### Risk Metrics

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Portfolio Volatility

### Trading Metrics

- Number of Trades
- Win Rate
- Turnover
- Transaction Cost Impact

---

# 🆚 10. Benchmark Comparison

The trained PPO strategies are compared against a baseline ladder, all run through the same environment:

```text
Buy & Hold            passive benchmark
ARIMA(1,0,1)          classical time-series, walk-forward
XGBoost               supervised next-day direction on the same features
Constant 50%          naive fixed de-risking
Volatility targeting  exposure = clip(1% / rolling volatility, 0, 1)
```

This comparison will help determine whether the learned strategy provides meaningful improvement over simpler approaches.

---

# 🔍 11. Explainable AI

SHAP (`src/explain_shap.py`) is used to understand the factors influencing the agent's trading decisions.

Features explained:

- Daily Return
- Rolling Volatility
- Hurst Exponent
- Market Regime
- Current Holdings

The objective is to answer:

> **Why did the agent choose BUY, HOLD, or SELL?**

SHAP will help analyse the contribution of individual features to the agent's decisions.

---

# 🔄 12. Walk-Forward Validation

Financial data is time-dependent, so all evaluation is chronological (`src/data_split.py`, `src/walk_forward.py`,
`src/wf_select_v3.py`).

```text
Historical Data
      ↓
Training Period
      ↓
Validation Period
      ↓
Testing Period
```

This helps reduce look-ahead bias and provides a more realistic evaluation of the trading strategy.

---

# 📁 Project Structure

```text
RegimeX/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── RegimeX_Analysis.ipynb
│
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── change_point_analysis.py
│   ├── regime_features.py
│   ├── regime_detection.py
│   ├── regime_interpretation.py
│   ├── add_regime_labels.py
│   ├── regime_relabel.py          # causal rolling-P90 High_Volatility labels
│   ├── build_multistock.py        # regime datasets for the other four stocks
│   ├── trading_environment.py
│   ├── trading_env.py             # v1 environment (BUY/HOLD/SELL)
│   ├── exposure_env.py            # v3 environment (position sizing)
│   ├── pooled_env.py
│   ├── data_split.py
│   ├── train_ppo.py / train_pooled.py
│   ├── strategies.py / metrics.py
│   ├── evaluate.py / evaluate_pooled.py / evaluate_v3.py
│   ├── walk_forward.py / wf_select_v3.py / wf_select_report.py
│   ├── explain_shap.py
│   └── test_trading_env.py / test_regime_reward.py
│
├── docs/
│   └── PHASE2_RESULTS.md
├── results/                       # tables and figures
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AnyaK393/RegimeX.git
cd RegimeX
```

## 2. Create a Virtual Environment

Python **3.11 or 3.12** is recommended.

```bash
python3.11 -m venv .venv
```

## 3. Activate the Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the data pipeline in the following order:

```bash
python src/data_collection.py
python src/data_cleaning.py
python src/eda.py
python src/change_point_analysis.py
python src/regime_features.py
python src/regime_detection.py
python src/regime_interpretation.py
python src/add_regime_labels.py
```

After generating the final regime dataset, test the trading environment:

```bash
python src/test_trading_env.py
```

A successful environment test should end with:

```text
Environment test completed successfully!
```

---

# 📓 Research Notebook

The Jupyter notebook documents the analysis and reasoning behind the project.

It contains:

- Data loading
- Data validation
- Data cleaning
- Exploratory Data Analysis
- Daily Returns
- Rolling Volatility
- Hurst Exponent
- Change Point Detection
- Feature Engineering
- Market Regime Detection
- Regime Interpretation
- Final Regime Dataset

The Python scripts contain the reusable implementation, while the notebook provides the research documentation, analysis, visualizations, and explanations.

---

# 🚧 Project Status

## Completed

- [x] Historical Data Collection
- [x] Data Cleaning
- [x] Exploratory Data Analysis
- [x] Daily Return
- [x] Rolling Volatility
- [x] Hurst Exponent
- [x] Change Point Detection
- [x] Regime Feature Engineering
- [x] KMeans Regime Detection
- [x] Regime Interpretation
- [x] Regime Labelling
- [x] Final Regime Dataset
- [x] Trading Environment
- [x] BUY / HOLD / SELL Actions
- [x] Portfolio Tracking
- [x] Reward Function
- [x] Transaction Costs
- [x] Dynamic Slippage
- [x] Market Impact
- [x] Gymnasium Environment
- [x] Environment Testing
- [x] Train / Validation / Test Split
- [x] PPO Agent
- [x] PPO Training

- [x] Causal rolling-percentile High_Volatility labelling
- [x] Multi-stock regime datasets (TCS, HDFCBANK, ICICIBANK, INFY)
- [x] Backtesting and baseline ladder (Buy & Hold, ARIMA, XGBoost, Constant 50%, Vol-target)
- [x] Performance metrics (return, Sharpe, Sortino, max drawdown, turnover, win rate)
- [x] Regime-stratified evaluation
- [x] Bootstrap significance testing
- [x] Walk-forward validation
- [x] SHAP explainability
- [x] Position-sizing environment (v3) with regime-conditioned risk reward

## 🚀 Upcoming

- [x] Walk-forward selection of the v3 risk-aversion scale (no test data used; did not transfer to test)
- [x] SHAP explanations for the v3 agents (`src/explain_shap_v3.py`)
- [x] Final report draft (`docs/REPORT.md`)

## 📦 Models and data in this repo

The five `data/processed/*_regimes.csv` files and every final trained model (`models/**/ppo_regimex_final.zip`, about 5 MB in total) are tracked, so the evaluation, SHAP and figure scripts run right after cloning (see `RUN_ME.txt`). Training checkpoints, logs and raw/intermediate data are not tracked and are regenerated by the pipeline.

## 📊 Results summary

Full details, tables and caveats: [`docs/PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md).

- The regime pipeline is causal and High_Volatility appears in train and test on all five stocks.
- With the original all-in BUY/HOLD/SELL formulation, **no significant benefit from the regime-conditioned reward** was found, on one stock or five (reported as a negative result).
- The v3 position-sizing agents do adapt to regimes (about 13–16 % exposure in High_Volatility versus 99 % for Buy & Hold) and cut drawdown by about half versus Buy & Hold. The adaptive-versus-blind Sharpe gain is suggestive (p ≈ 0.10) but not significant, and neither agent beats a naive Constant 50 % rule on Sharpe in the test window.

---

# 🛠️ Technology Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly

### Financial Data

- yfinance

### Machine Learning

- Scikit-learn
- KMeans

### Change Point Detection

- ruptures

### Reinforcement Learning

- Gymnasium
- Stable-Baselines3
- PPO

### Explainable AI

- SHAP

### Development

- Jupyter Notebook
- Git
- GitHub

---

# 👥 Team

## Team RegimeX

A collaborative academic research project combining:

**Data Science + Quantitative Finance + Machine Learning + Reinforcement Learning + Explainable AI**

---

# ⚠️ Disclaimer

RegimeX is developed strictly for academic and research purposes.

It is **not financial advice** and should not be used for live trading or investment decisions.


