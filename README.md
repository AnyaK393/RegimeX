# RegimeX 📈🤖

## Regime-Aware Reinforcement Learning Trading System for Indian Equity Markets

RegimeX is an AI-powered quantitative trading research project that investigates whether **market regime awareness can improve reinforcement-learning-based trading strategies** in Indian equity markets.

The system combines:

- Financial Data Science
- Market Regime Detection
- Reinforcement Learning
- Realistic Trading Simulation
- Explainable AI
- Backtesting

The ultimate goal is to build a PPO-based trading agent that adapts its decisions according to changing market conditions.

---

# 🎯 Research Objective

Financial markets do not behave the same way at all times.

Market conditions can shift between:

- Normal/stable periods
- Bearish periods
- Trending periods
- High-volatility periods

RegimeX investigates:

> **Can a regime-aware reinforcement learning agent make better trading decisions than a regime-blind strategy?**

---

# 🧠 System Pipeline

```text
Historical NSE Stock Data
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Financial Feature Engineering
          ↓
Daily Returns
          ↓
Rolling Volatility
          ↓
Hurst Exponent
          ↓
Change Point Detection
          ↓
Market Regime Detection
          ↓
Regime Interpretation
          ↓
Trading Environment
          ↓
Transaction Costs
          ↓
Dynamic Slippage
          ↓
Dynamic Market Impact
          ↓
Reward Function
          ↓
Gymnasium Environment
          ↓
PPO Agent
          ↓
SHAP Explainability
          ↓
Backtesting
          ↓
Performance Evaluation

📊 Data
Historical Indian equity market data is collected using Yahoo Finance.
Initial stocks:
RELIANCE
TCS
HDFCBANK
ICICIBANK
INFY
Current development and environment testing primarily use:
RELIANCE
Historical period:
2015-01-01 → 2025-12-31
Data contains:
Open
High
Low
Close
Adjusted Close
Volume
🔧 Feature Engineering
RegimeX currently creates the following features.
Daily Return
Measures the percentage price movement between consecutive trading days.
Daily Return =
(Current Price - Previous Price) / Previous Price
Rolling Volatility
Measures recent market uncertainty using rolling standard deviation of returns.
Higher volatility indicates larger and less stable price movements.
Hurst Exponent
The Hurst exponent is used to study the persistence of price behaviour.
H < 0.5  → Mean-reverting behaviour

H ≈ 0.5  → Random-walk-like behaviour

H > 0.5  → Persistent / trending behaviour
🔍 Change Point Detection
RegimeX uses change point detection to identify points where the statistical behaviour of the market changes.
The project currently uses:
ruptures
Change points provide additional information about structural changes in market behaviour.
📈 Market Regime Detection
The engineered features are used to identify different market conditions.
Input features:
Daily Return
Rolling Volatility
Hurst Exponent
The current implementation uses unsupervised clustering to group observations into statistically similar market states.
The numerical clusters are interpreted as:
Regime	Interpretation
0	Weak Bear
1	Normal Market
2	High Volatility


The exact characteristics of each regime are determined from the historical data rather than assumed beforehand.
🏦 Trading Environment
The detected market regimes are passed into a simulated trading environment.
Starting capital:
₹100,000
The environment tracks:
Cash
Shares
Stock price
Portfolio value
Market regime
Trading actions
Rewards
🎮 Trading Actions
The agent has three possible actions:
0 → HOLD
1 → BUY
2 → SELL
👀 Observation / State
The current trading environment provides five observations:
[
    Daily Return,
    Rolling Volatility,
    Hurst Exponent,
    Market Regime,
    Holding Status
]
Where:
Holding Status:
0 → No shares held
1 → Shares currently held
💰 Reward Function
The current reward is based on the change in portfolio value.
Reward =
Current Portfolio Value
-
Previous Portfolio Value
Trading costs are reflected in portfolio value and therefore affect the reward.
💸 Transaction Costs
The environment includes transaction costs to make trading more realistic.
Current transaction cost:
0.1%
The cost is applied to BUY and SELL transactions.
📉 Dynamic Slippage
The environment models execution price differences using volatility-dependent slippage.
Conceptually:
Slippage =
Base Slippage
+
Slippage Multiplier × Volatility
BUY trades receive a slightly higher execution price.
SELL trades receive a slightly lower execution price.
🌊 Market Impact
RegimeX also models dynamic market impact.
The execution price therefore considers:
Market Price
      ↓
Slippage
      ↓
Market Impact
      ↓
Execution Price
      ↓
Transaction Cost
This creates a more realistic trading environment than assuming every trade executes exactly at the market price.
🤖 Reinforcement Learning
The next major component is a PPO agent.
Algorithm:
Proximal Policy Optimization (PPO)
The agent will learn:
Observe Market State
        ↓
Choose Action
        ↓
Execute Trade
        ↓
Receive Reward
        ↓
Update Policy
        ↓
Repeat
🔬 Explainable AI
SHAP will be used to analyze the trained agent.
The goal is to understand:
Why did the agent choose BUY, SELL, or HOLD?

Potential influential factors include:
Daily return
Volatility
Hurst exponent
Market regime
Current holdings
📊 Evaluation
The final strategy will be evaluated using:
Return Metrics
Total Return
Annualized Return
Risk Metrics
Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Trading Metrics
Number of Trades
Turnover
Transaction Costs
Slippage Impact
The trained agent will eventually be compared against appropriate baseline strategies.
📁 Project Structure
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
🚀 Current Progress
Completed

Historical data collection

Data cleaning

Data validation

Exploratory Data Analysis

Daily return analysis

Normalized performance analysis

Rolling volatility

Hurst exponent

Change point detection

Regime feature engineering

Market regime detection

Regime interpretation

Final regime-labelled dataset

Trading environment

BUY / HOLD / SELL actions

Portfolio accounting

Reward function

Transaction costs

Dynamic slippage

Dynamic market impact

Gymnasium environment

Environment testing
In Progress / Upcoming

PPO agent

PPO training

Model evaluation

Baseline comparison

SHAP explainability

Walk-forward backtesting

Risk analysis

Statistical significance testing
⚙️ Installation
Clone the repository:
git clone https://github.com/AnyaK393/RegimeX.git
cd RegimeX
Create a virtual environment:
python -m venv .venv
Activate it on macOS/Linux:
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
▶️ Running the Project
Run the individual pipeline stages from the project root.
Data Collection
python src/data_collection.py
Data Cleaning
python src/data_cleaning.py
Exploratory Analysis
python src/eda.py
Change Point Detection
python src/change_point_analysis.py
Feature Engineering
python src/regime_features.py
Regime Detection
python src/regime_detection.py
Regime Interpretation
python src/regime_interpretation.py
Add Market Regime Labels
python src/add_regime_labels.py
Test Trading Environment
python src/trading_environment.py
Test Gymnasium Environment
python src/test_trading_env.py
📓 Notebook
The main notebook documents the complete research workflow.
It contains:
Data
 ↓
Cleaning
 ↓
EDA
 ↓
Feature Engineering
 ↓
Regime Detection
 ↓
Regime Interpretation
 ↓
Trading Environment
 ↓
Gymnasium Environment
The notebook is intended to provide a readable research record, while the Python files contain the reusable implementation.

⚠️ Disclaimer
RegimeX is an academic and research project.
It is not financial advice and should not be used as a real-world automated trading system without extensive additional validation, risk controls, and regulatory consideration.

---

# 16. One README correction from your old version

Your old README says:

> `Stock price, Daily returns, Volatility, Hurst exponent, Market regime, Portfolio balance, Current holdings`

as the state.

That's **not currently accurate**.

Your actual state is:

```text
[Daily_Return,
 Rolling_Volatility,
 Hurst,
 Regime,
 Holding_Status]
So I deliberately corrected that above.
Also, don't claim PPO, SHAP, or backtesting are completed yet. They're roadmap items. This makes the repo much more credible.