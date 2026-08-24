
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

The current reward is based on the change in portfolio value.

```text
Reward =
Current Portfolio Value
-
Previous Portfolio Value
```

Therefore:

- Profitable portfolio changes produce positive rewards.
- Portfolio losses produce negative rewards.
- Transaction costs reduce rewards.
- Slippage reduces effective returns.
- Market impact reduces effective returns.

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

## 🚧 Next Major Stage

The next stage of RegimeX is implementing a **PPO (Proximal Policy Optimization)** trading agent.

The agent will learn trading decisions instead of using manually defined trading rules.

The learning loop will be:

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

After training the PPO agent, the strategy will be evaluated on historical data that was not used during training.

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

The trained PPO strategy will be compared against simple baseline strategies.

The primary benchmark will be:

```text
Buy & Hold
```

Additional baselines may include:

```text
Random Trading
```

This comparison will help determine whether the learned strategy provides meaningful improvement over simpler approaches.

---

# 🔍 11. Explainable AI

SHAP will be integrated to understand the factors influencing the agent's trading decisions.

Important features will include:

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

Financial data is time-dependent, so the final system will use chronological validation.

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
│   ├── trading_environment.py
│   ├── trading_env.py
│   └── test_trading_env.py
│
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

## 🚀 Upcoming

- [x] Train / Validation / Test Split
- [x] PPO Agent
- [x] PPO Training
- [ ] Backtesting
- [ ] Buy & Hold Benchmark
- [ ] Performance Metrics
- [ ] SHAP Explainability
- [ ] Walk-Forward Validation
- [ ] Final Visualizations
- [ ] Statistical Evaluation
- [ ] Final Research Results

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


