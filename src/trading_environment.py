import pandas as pd
import os

# ============================================================
# REGIMEX - TRADING ENVIRONMENT
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"

print("=" * 60)
print("REGIMEX - TRADING ENVIRONMENT")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

print("\nDataset loaded successfully!")
print(f"Rows: {len(df)}")

# ------------------------------------------------------------
# 2. Initial portfolio
# ------------------------------------------------------------

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001

# Dynamic slippage parameters
BASE_SLIPPAGE = 0.0005
SLIPPAGE_MULTIPLIER = 0.05

# Dynamic market impact parameters
BASE_MARKET_IMPACT = 0.0001
MARKET_IMPACT_MULTIPLIER = 0.01

cash = INITIAL_CAPITAL
shares = 0

# Current stock price
current_price = df.iloc[0]["Close"]

# Portfolio value
portfolio_value = cash + (shares * current_price)

print("\n" + "=" * 60)
print("INITIAL PORTFOLIO")
print("=" * 60)

print(f"Starting Capital: ₹{INITIAL_CAPITAL:,.2f}")
print(f"Cash:             ₹{cash:,.2f}")
print(f"Shares:           {shares}")
print(f"Stock Price:      ₹{current_price:,.2f}")
print(f"Portfolio Value:  ₹{portfolio_value:,.2f}")

# ------------------------------------------------------------
# 3. Trading Actions
# ------------------------------------------------------------

HOLD = 0
BUY = 1
SELL = 2

actions = {
    HOLD: "HOLD",
    BUY: "BUY",
    SELL: "SELL"
}

print("\n" + "=" * 60)
print("AVAILABLE TRADING ACTIONS")
print("=" * 60)

for action_id, action_name in actions.items():
    print(f"{action_id} → {action_name}")


# ------------------------------------------------------------
# 4. Execute Trading Action
# ------------------------------------------------------------

def execute_trade(
    action,
    cash,
    shares,
    price,
    transaction_cost,
    volatility
):

    # --------------------------------------------------------
    # Calculate dynamic slippage
    # --------------------------------------------------------

    slippage = BASE_SLIPPAGE + (SLIPPAGE_MULTIPLIER * volatility)

    # --------------------------------------------------------
    # Calculate dynamic market impact
    # --------------------------------------------------------

    market_impact = (
        BASE_MARKET_IMPACT
        + (MARKET_IMPACT_MULTIPLIER * volatility)
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if action == BUY:

        # Buyer pays a higher execution price
        execution_price = price * (
            1 + slippage + market_impact
        )

        shares_to_buy = int(
            cash // (execution_price * (1 + transaction_cost))
        )

        if shares_to_buy > 0:

            trade_value = shares_to_buy * execution_price
            cost = trade_value * transaction_cost

            shares += shares_to_buy
            cash -= trade_value + cost

            print("\nBUY executed")
            print(f"Market price: ₹{price:.2f}")
            print(f"Execution price: ₹{execution_price:.2f}")
            print(f"Slippage: {slippage:.4%}")
            print(f"Market impact: {market_impact:.4%}")
            print(f"Shares bought: {shares_to_buy}")
            print(f"Trade value: ₹{trade_value:.2f}")
            print(f"Transaction cost: ₹{cost:.2f}")

        else:
            print("\nNot enough cash to buy a share.")

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    elif action == SELL:

        if shares > 0:

            # Seller receives a lower execution price
            execution_price = price * (
                1 - slippage - market_impact
            )

            trade_value = shares * execution_price
            cost = trade_value * transaction_cost

            cash += trade_value - cost

            print("\nSELL executed")
            print(f"Market price: ₹{price:.2f}")
            print(f"Execution price: ₹{execution_price:.2f}")
            print(f"Slippage: {slippage:.4%}")
            print(f"Market impact: {market_impact:.4%}")
            print(f"Shares sold: {shares}")
            print(f"Trade value: ₹{trade_value:.2f}")
            print(f"Transaction cost: ₹{cost:.2f}")

            shares = 0

        else:
            print("\nNo shares available to sell.")

    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    elif action == HOLD:

        print("\nHOLD executed")
        print("No trade performed.")

    return cash, shares


# ------------------------------------------------------------
# 5. Current market information
# ------------------------------------------------------------

#current_regime = df.iloc[0]["Market_Regime"]

#print("\n" + "=" * 60)
#print("CURRENT MARKET STATE")
#print("=" * 60)

#print(f"Date:             {df.iloc[0]['Date'].date()}")
#print(f"Stock Price:      ₹{current_price:,.2f}")
#print(f"Market Regime:    {current_regime}")

#print("\nTrading environment initialized successfully!")


# ------------------------------------------------------------
# 6. Test BUY action
# ------------------------------------------------------------

# action = BUY #change it to sell or hold to test other actions

# cash, shares = execute_trade(
#     action,
#     cash,
#     shares,
#     current_price,
#     TRANSACTION_COST,
#     df.iloc[0]["Rolling_Volatility"]
# )

# portfolio_value = cash + (shares * current_price)

# print("\n" + "=" * 60)
# print("PORTFOLIO AFTER TRADE")
# print("=" * 60)

# print(f"Cash:             ₹{cash:,.2f}")
# print(f"Shares:           {shares}")
# print(f"Stock Price:      ₹{current_price:,.2f}")
# print(f"Portfolio Value:  ₹{portfolio_value:,.2f}")


# ------------------------------------------------------------
# 7. Create Market State
# ------------------------------------------------------------

def get_state(data, shares):

    state = [
        data["Daily_Return"],
        data["Rolling_Volatility"],
        data["Hurst"],
        data["Regime"],
        1 if shares > 0 else 0
    ]

    return state


# ------------------------------------------------------------
# 8. Calculate Reward
# ------------------------------------------------------------

def calculate_reward(previous_value, current_value):

    reward = current_value - previous_value

    return reward


# ------------------------------------------------------------
# 9. Move through trading days
# ------------------------------------------------------------

current_step = 0

# Store the portfolio value before the first trading day
previous_portfolio_value = INITIAL_CAPITAL

# Test actions for the first 5 trading days
test_actions = [
    BUY,
    HOLD,
    HOLD,
    SELL,
    HOLD
]

print("\n" + "=" * 60)
print("TRADING DAY SIMULATION")
print("=" * 60)

for current_step in range(5):

    current_data = df.iloc[current_step]

    current_price = current_data["Close"]
    current_regime = current_data["Market_Regime"]
    current_date = current_data["Date"]

    # --------------------------------------------------------
    # Create current market state
    # --------------------------------------------------------

    state = get_state(current_data, shares)

    # Select test action
    action = test_actions[current_step]

    # --------------------------------------------------------
    # Execute action
    # --------------------------------------------------------

    cash, shares = execute_trade(
        action,
        cash,
        shares,
        current_price,
        TRANSACTION_COST,
        current_data["Rolling_Volatility"]
    )

    # --------------------------------------------------------
    # Calculate portfolio value
    # --------------------------------------------------------

    portfolio_value = cash + (shares * current_price)

    # --------------------------------------------------------
    # Calculate reward
    # --------------------------------------------------------

    reward = calculate_reward(
        previous_portfolio_value,
        portfolio_value
    )

    print("\n----------------------------------------")
    print(f"Day: {current_step + 1}")
    print(f"Date: {current_date.date()}")
    print(f"Price: ₹{current_price:.2f}")
    print(f"Market Regime: {current_regime}")

    print(f"State: {state}")

    print(f"Action: {actions[action]}")

    print(f"Cash: ₹{cash:.2f}")
    print(f"Shares: {shares}")
    print(f"Portfolio Value: ₹{portfolio_value:.2f}")

    print(f"Reward: ₹{reward:.2f}")

    # Update portfolio value for next day
    previous_portfolio_value = portfolio_value