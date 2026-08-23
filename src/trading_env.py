import gymnasium as gym
from gymnasium import spaces

import pandas as pd
import numpy as np
import os


# ============================================================
# REGIMEX - GYMNASIUM TRADING ENVIRONMENT
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"


class RegimeXTradingEnv(gym.Env):

    # --------------------------------------------------------
    # Environment metadata
    # --------------------------------------------------------

    metadata = {"render_modes": ["human"]}

    # --------------------------------------------------------
    # Initialize environment
    # --------------------------------------------------------

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        self.df = pd.read_csv(DATA_PATH)

        self.df["Date"] = pd.to_datetime(self.df["Date"])

        # ----------------------------------------------------
        # Trading parameters
        # ----------------------------------------------------

        self.initial_capital = 100000

        # Transaction cost = 0.1%
        self.transaction_cost = 0.001

        # Dynamic slippage
        self.base_slippage = 0.0005
        self.slippage_multiplier = 0.05

        # Dynamic market impact
        self.base_market_impact = 0.0001
        self.market_impact_multiplier = 0.01

        # ----------------------------------------------------
        # Action space
        # ----------------------------------------------------

        # 0 = HOLD
        # 1 = BUY
        # 2 = SELL

        self.action_space = spaces.Discrete(3)

        # ----------------------------------------------------
        # Observation space
        # ----------------------------------------------------

        # State:
        #
        # 0 → Daily Return
        # 1 → Rolling Volatility
        # 2 → Hurst
        # 3 → Market Regime
        # 4 → Holding Position
        #
        # Values are kept within a broad range.

        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, 0, 0, 0, 0],
                dtype=np.float32
            ),
            high=np.array(
                [np.inf, np.inf, 1, 2, 1],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Environment state
        # ----------------------------------------------------

        self.current_step = 0

        self.cash = self.initial_capital
        self.shares = 0

        self.previous_portfolio_value = self.initial_capital

    # ========================================================
    # GET OBSERVATION
    # ========================================================

    def _get_observation(self):

        row = self.df.iloc[self.current_step]

        observation = np.array(
            [
                row["Daily_Return"],
                row["Rolling_Volatility"],
                row["Hurst"],
                row["Regime"],
                1 if self.shares > 0 else 0
            ],
            dtype=np.float32
        )

        return observation

    # ========================================================
    # CALCULATE PORTFOLIO VALUE
    # ========================================================

    def _get_portfolio_value(self):

        price = self.df.iloc[self.current_step]["Close"]

        return self.cash + (self.shares * price)

    # ========================================================
    # EXECUTE TRADE
    # ========================================================

    def _execute_trade(self, action):

        row = self.df.iloc[self.current_step]

        price = row["Close"]
        volatility = row["Rolling_Volatility"]

        # ----------------------------------------------------
        # Dynamic slippage
        # ----------------------------------------------------

        slippage = (
            self.base_slippage
            + self.slippage_multiplier * volatility
        )

        # ----------------------------------------------------
        # Dynamic market impact
        # ----------------------------------------------------

        market_impact = (
            self.base_market_impact
            + self.market_impact_multiplier * volatility
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if action == 1:

            execution_price = price * (
                1 + slippage + market_impact
            )

            shares_to_buy = int(
                self.cash
                // (
                    execution_price
                    * (1 + self.transaction_cost)
                )
            )

            if shares_to_buy > 0:

                trade_value = (
                    shares_to_buy * execution_price
                )

                transaction_cost = (
                    trade_value * self.transaction_cost
                )

                self.shares += shares_to_buy

                self.cash -= (
                    trade_value
                    + transaction_cost
                )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif action == 2:

            if self.shares > 0:

                execution_price = price * (
                    1 - slippage - market_impact
                )

                trade_value = (
                    self.shares * execution_price
                )

                transaction_cost = (
                    trade_value * self.transaction_cost
                )

                self.cash += (
                    trade_value
                    - transaction_cost
                )

                self.shares = 0

        # ----------------------------------------------------
        # HOLD
        # ----------------------------------------------------

        elif action == 0:

            pass

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        self.cash = self.initial_capital

        self.shares = 0

        self.previous_portfolio_value = (
            self.initial_capital
        )

        observation = self._get_observation()

        info = {
            "portfolio_value": self.initial_capital,
            "cash": self.cash,
            "shares": self.shares
        }

        return observation, info

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        # ----------------------------------------------------
        # Portfolio value before action
        # ----------------------------------------------------

        previous_value = self._get_portfolio_value()

        # ----------------------------------------------------
        # Execute action
        # ----------------------------------------------------

        self._execute_trade(action)

        # ----------------------------------------------------
        # Calculate current portfolio value
        # ----------------------------------------------------

        current_value = self._get_portfolio_value()

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = current_value - previous_value

        # ----------------------------------------------------
        # Move to next trading day
        # ----------------------------------------------------

        self.current_step += 1

        # ----------------------------------------------------
        # Check episode termination
        # ----------------------------------------------------

        terminated = (
            self.current_step >= len(self.df) - 1
        )

        truncated = False

        # ----------------------------------------------------
        # Get next observation
        # ----------------------------------------------------

        if not terminated:

            observation = self._get_observation()

        else:

            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        info = {
            "portfolio_value": current_value,
            "cash": self.cash,
            "shares": self.shares
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        row = self.df.iloc[self.current_step]

        print("=" * 50)
        print("REGIMEX TRADING ENVIRONMENT")
        print("=" * 50)

        print(f"Date: {row['Date'].date()}")
        print(f"Price: ₹{row['Close']:.2f}")
        print(f"Market Regime: {row['Market_Regime']}")
        print(f"Cash: ₹{self.cash:.2f}")
        print(f"Shares: {self.shares}")
        print(
            f"Portfolio Value: "
            f"₹{self._get_portfolio_value():.2f}"
        )