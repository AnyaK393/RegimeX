# RegimeX 📈🤖

## Regime-Aware AI Trading System for Indian Equity Markets

RegimeX is an AI-powered quantitative trading research framework that explores how **market regime awareness** can improve reinforcement learning-based trading strategies.

The project combines **Data Science, Financial Analytics, Reinforcement Learning, and Explainable AI** to build an adaptive trading agent capable of making **Buy / Sell / Hold** decisions based on changing market conditions.

---

# Research Objective

Traditional trading models often assume that market behaviour remains constant.

However, financial markets continuously shift between different conditions:

- Stable markets
- Trending markets
- High volatility periods
- Uncertain or bearish phases

RegimeX investigates:

> Can a regime-aware Reinforcement Learning trading agent achieve better risk-adjusted performance compared to regime-blind trading strategies?

---

# System Pipeline

```
Stock Market Data (NSE)
            |
            ↓
Data Collection & Cleaning
            |
            ↓
Exploratory Data Analysis
            |
            ↓
Financial Feature Engineering
            |
            ↓
Market Behaviour Analysis
            |
            ↓
Regime Detection
            |
            ↓
Regime-Aware Trading Environment
            |
            ↓
PPO Reinforcement Learning Agent
            |
            ↓
Explainable AI (SHAP)
            |
            ↓
Backtesting & Performance Evaluation
```

---

# Current Implementation Status ✅

## 1. Data Collection

- Collected historical NSE stock data using Yahoo Finance API
- Initial stocks:

```
RELIANCE
TCS
HDFCBANK
ICICIBANK
INFY
```

Data includes:

- Open Price
- High Price
- Low Price
- Close Price
- Adjusted Close
- Trading Volume

---

## 2. Data Cleaning

Performed:

- Missing value handling
- Duplicate removal
- Date formatting
- Data validation
- Structured processed datasets

---

## 3. Exploratory Data Analysis

Performed financial analysis:

- Price movement analysis
- Return distribution analysis
- Daily return statistics
- Volatility analysis

---

## 4. Feature Engineering

Created financial indicators:

### Daily Returns

Measures daily price movement:

```
Return = (Today's Price - Yesterday's Price) / Yesterday's Price
```

---

### Rolling Volatility

Measures market uncertainty using rolling standard deviation.

---

### Hurst Exponent

Measures market behaviour:

- H < 0.5 → Mean reverting behaviour
- H ≈ 0.5 → Random behaviour
- H > 0.5 → Trending behaviour

---

# Market Regime Detection

RegimeX uses unsupervised learning to identify different market states.

## Method

KMeans Clustering

Input Features:

```
Daily Return
Rolling Volatility
Hurst Exponent
```

The model identifies statistically similar market conditions.

Detected regimes are interpreted as:

| Regime | Meaning |
|---|---|
| Normal Market | Stable trading conditions |
| Weak Bear | Negative/uncertain market behaviour |
| High Volatility | Extreme market movements |

---

# Reinforcement Learning Trading Agent 🚧

Upcoming implementation:

The trading agent will use:

## Algorithm

PPO (Proximal Policy Optimization)

The agent learns:

```
Market State
      ↓
Trading Action
      ↓
Reward
      ↓
Policy Improvement
```

Actions:

```
0 → Hold
1 → Buy
2 → Sell
```

---

# Trading Environment Design

The environment will include:

## State Space

The agent observes:

- Stock price
- Daily returns
- Volatility
- Hurst exponent
- Market regime
- Portfolio balance
- Current holdings


## Reward Function

Initial:

- Portfolio returns

Advanced:

- Regime-conditioned reward
- Risk penalties
- Market impact cost

---

# Explainable AI

To understand trading decisions, SHAP explainability will be integrated.

The system will answer:

> Why did the AI decide to Buy/Sell/Hold?

Example explanations:

- High volatility increased risk
- Strong trend supported buying
- Weak momentum caused selling

---

# Evaluation Metrics

The final system will be evaluated using:

### Return Metrics

- Total Return
- Annualized Return

### Risk Metrics

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown

### Trading Metrics

- Turnover
- Transaction Cost Impact

---

# Technology Stack

## Programming

- Python

## Data Science

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Financial Analysis

- Yahoo Finance API
- Technical Indicators

## Machine Learning

- KMeans Clustering
- Change Point Detection

## Reinforcement Learning

- Stable-Baselines3
- PPO

## Explainable AI

- SHAP

---

# Project Structure

```
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
│   └── add_regime_labels.py
│
├── README.md
└── requirements.txt
```

---

# Future Roadmap

- [x] Data Collection
- [x] Data Cleaning
- [x] Exploratory Analysis
- [x] Financial Feature Engineering
- [x] Hurst Analysis
- [x] Change Point Detection
- [x] Market Regime Detection

Next:

- [ ] Trading Environment
- [ ] PPO Agent
- [ ] Dynamic Reward Function
- [ ] Market Impact Simulation
- [ ] SHAP Explainability
- [ ] Walk Forward Backtesting
- [ ] Statistical Significance Testing

---

# Authors

Team RegimeX

---

## Disclaimer

This project is developed for academic and research purposes only.

It is not financial advice and should not be used for real-world trading decisions.
