import gymnasium as gym
from gymnasium import spaces

import pandas as pd
import numpy as np
import os


# ============================================================
# REGIMEX - GYMNASIUM TRADING ENVIRONMENT  (v2)
# ============================================================
#
# Changes from v1:
#   - __init__ accepts a pre-sliced DataFrame (df=) instead of
#     loading from a hardcoded path. Backwards-compatible: if
#     df=None, loads from DATA_PATH as before.
#   - regime_adaptive=True toggle: when False, lambda_risk is
#     fixed at the Normal_Market baseline across all regimes.
#     This flag enables the regime-blind vs. regime-adaptive
#     comparison in the next phase without any code changes.
#   - Reward reshaped to: log_return - lambda_risk * drawdown
#     - lambda_fee * turnover  (see _compute_reward).
#   - Tracks self.peak_value for drawdown, self._last_trade_value
#     for turnover.
#   - Reward is measured close(t) -> close(t+1) after the action, so
#     price moves are reflected in the reward (earlier draft valued
#     both sides at the same close and only saw trading costs).
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"

# ------------------------------------------------------------
# Default lambda configuration
# ------------------------------------------------------------
#
# lambda_risk: drawdown penalty weight, keyed by Market_Regime.
#   High_Volatility → tighter (agent should be conservative)
#   Weak_Bear       → moderate (biased against long exposure)
#   Normal_Market   → baseline / looser
#
# lambda_fee: turnover penalty weight (fixed, penalises churn).
#
# These are exposed as constructor args so the team can tune
# without touching the class internals.
# ------------------------------------------------------------

DEFAULT_LAMBDA_RISK = {
    "High_Volatility": 0.30,
    "Weak_Bear":        0.15,
    "Normal_Market":    0.05,
}
REGIME_BLIND_LAMBDA = 0.05   # flat rate used when regime_adaptive=False
DEFAULT_LAMBDA_FEE  = 0.01


REGIME_ORDER = ["Normal_Market", "Weak_Bear", "High_Volatility"]
OBS_FEATURES = ["Daily_Return", "Rolling_Volatility", "Hurst"]


def obs_stats_from(train_df) -> dict:
    """Mean/std of the continuous features, from TRAIN data only (no leakage)."""
    return {
        c: (float(train_df[c].mean()), float(train_df[c].std(ddof=0) or 1.0))
        for c in OBS_FEATURES
    }


class RegimeXTradingEnv(gym.Env):

    # --------------------------------------------------------
    # Environment metadata
    # --------------------------------------------------------

    metadata = {"render_modes": ["human"]}

    # --------------------------------------------------------
    # Initialize environment
    # --------------------------------------------------------

    def __init__(
        self,
        df=None,
        regime_adaptive: bool = True,
        lambda_risk_config: dict = None,
        lambda_fee: float = DEFAULT_LAMBDA_FEE,
        scaled_obs: bool = False,
        obs_stats: dict = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame or None
            Pre-sliced regime dataset. If None, loads from DATA_PATH
            (backwards-compatible with the original no-arg call).
        regime_adaptive : bool
            If True, lambda_risk scales by the current Market_Regime.
            If False, lambda_risk is fixed at REGIME_BLIND_LAMBDA
            (Normal_Market baseline) regardless of regime.
            Keep this toggle — it drives the regime-blind vs. adaptive
            comparison in the next phase.
        lambda_risk_config : dict or None
            Override the default lambda_risk values. Keys must be
            'High_Volatility', 'Weak_Bear', 'Normal_Market'.
        lambda_fee : float
            Turnover penalty weight (applied to trade_value / portfolio_value).
        scaled_obs : bool
            False (default, v1): raw 5-dim observation
                [Daily_Return, Rolling_Volatility, Hurst, Regime(int), Holdings].
            True (v2): 7-dim observation with the three continuous features
                z-scored (clipped to +/-5) using `obs_stats`, the regime as a
                one-hot vector (Normal, Weak_Bear, High_Vol) and Holdings.
                Raw features are ~0.01 in scale next to 0/1/2-valued inputs,
                which starves the policy network of the return/volatility signal.
        obs_stats : dict or None
            {feature: (mean, std)} from obs_stats_from(train_df). Required when
            scaled_obs=True; MUST come from the training split only.
        """
        super().__init__()

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        if df is None:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            df["Date"] = pd.to_datetime(df["Date"])

        self.df = df.reset_index(drop=True)

        self.scaled_obs = scaled_obs
        self.obs_stats  = obs_stats
        if scaled_obs and obs_stats is None:
            raise ValueError("scaled_obs=True requires obs_stats (from train split)")

        # ----------------------------------------------------
        # Reward configuration
        # ----------------------------------------------------

        self.regime_adaptive  = regime_adaptive
        self.lambda_risk_cfg  = lambda_risk_config or DEFAULT_LAMBDA_RISK
        self.lambda_fee       = lambda_fee

        # Validate config keys
        required = {"High_Volatility", "Weak_Bear", "Normal_Market"}
        missing  = required - set(self.lambda_risk_cfg.keys())
        if missing:
            raise ValueError(
                f"lambda_risk_config is missing keys: {missing}"
            )

        # ----------------------------------------------------
        # Trading parameters
        # ----------------------------------------------------

        self.initial_capital = 100000

        # Transaction cost = 0.1%
        self.transaction_cost    = 0.001

        # Dynamic slippage
        self.base_slippage       = 0.0005
        self.slippage_multiplier = 0.05

        # Dynamic market impact
        self.base_market_impact          = 0.0001
        self.market_impact_multiplier    = 0.01

        # ----------------------------------------------------
        # Action space  (0=HOLD, 1=BUY, 2=SELL)
        # ----------------------------------------------------

        self.action_space = spaces.Discrete(3)

        # ----------------------------------------------------
        # Observation space
        #
        # [Daily_Return, Rolling_Volatility, Hurst, Regime, Holdings]
        # ----------------------------------------------------

        if scaled_obs:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, 1, 2, 1], dtype=np.float32),
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Episode state (initialised in reset())
        # ----------------------------------------------------

        self.current_step            = 0
        self.cash                    = self.initial_capital
        self.shares                  = 0
        self.peak_value              = self.initial_capital
        self._last_trade_value       = 0.0
        self.previous_portfolio_value = self.initial_capital

    # ========================================================
    # GET OBSERVATION
    # ========================================================

    def _get_observation(self):

        row = self.df.iloc[self.current_step]
        holding = 1 if self.shares > 0 else 0

        if self.scaled_obs:
            z = [
                np.clip((row[c] - self.obs_stats[c][0]) / self.obs_stats[c][1], -5, 5)
                for c in OBS_FEATURES
            ]
            onehot = [1.0 if row["Market_Regime"] == r else 0.0 for r in REGIME_ORDER]
            return np.array(z + onehot + [holding], dtype=np.float32)

        return np.array(
            [
                row["Daily_Return"],
                row["Rolling_Volatility"],
                row["Hurst"],
                row["Regime"],
                holding
            ],
            dtype=np.float32
        )

    # ========================================================
    # CALCULATE PORTFOLIO VALUE
    # ========================================================

    def _get_portfolio_value(self):

        price = self.df.iloc[self.current_step]["Close"]
        return self.cash + (self.shares * price)

    # ========================================================
    # LAMBDA RISK LOOKUP
    # ========================================================

    def _get_lambda_risk(self, regime_label: str) -> float:
        """
        Returns the drawdown penalty weight for the current regime.
        When regime_adaptive=False, always returns the regime-blind
        constant so reward is identical across regimes.
        """
        if not self.regime_adaptive:
            return REGIME_BLIND_LAMBDA
        return self.lambda_risk_cfg.get(regime_label, REGIME_BLIND_LAMBDA)

    # ========================================================
    # COMPUTE REWARD
    # ========================================================

    def _compute_reward(
        self,
        previous_value: float,
        current_value:  float,
        regime_label:   str,
    ) -> float:
        """
        Regime-conditional reward:

            R = log(V_t / V_{t-1})
              - lambda_risk(regime) * drawdown_penalty
              - lambda_fee          * turnover_penalty

        log_return        — normalises for portfolio size; numerically
                            stable vs. raw delta.
        drawdown_penalty  — (peak - current) / peak; tracks running
                            peak within the episode.
        turnover_penalty  — trade_value / portfolio_value; penalises
                            unnecessary churn.
        """
        # Log return (guard against zero/negative values)
        eps = 1e-8
        log_return = np.log(
            max(current_value, eps) / max(previous_value, eps)
        )

        # Update running peak
        self.peak_value = max(self.peak_value, current_value)

        # Drawdown from peak
        drawdown = max(
            0.0,
            (self.peak_value - current_value) / max(self.peak_value, eps)
        )

        # Turnover fraction
        turnover = self._last_trade_value / max(previous_value, eps)

        # Penalty weights
        lambda_risk = self._get_lambda_risk(regime_label)

        reward = (
            log_return
            - lambda_risk * drawdown
            - self.lambda_fee * turnover
        )

        return float(reward)

    # ========================================================
    # EXECUTE TRADE
    # ========================================================

    def _execute_trade(self, action):

        row       = self.df.iloc[self.current_step]
        price     = row["Close"]
        volatility = row["Rolling_Volatility"]

        # Reset turnover tracker each step
        self._last_trade_value = 0.0

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
                self.cash   -= (trade_value + transaction_cost)

                # Record for turnover penalty
                self._last_trade_value = trade_value

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

                self.cash  += (trade_value - transaction_cost)
                self.shares = 0

                # Record for turnover penalty
                self._last_trade_value = trade_value

        # ----------------------------------------------------
        # HOLD — no action, no turnover
        # ----------------------------------------------------

        elif action == 0:
            pass

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step             = 0
        self.cash                     = self.initial_capital
        self.shares                   = 0
        self.peak_value               = self.initial_capital
        self._last_trade_value        = 0.0
        self.previous_portfolio_value = self.initial_capital

        observation = self._get_observation()

        info = {
            "portfolio_value": self.initial_capital,
            "cash":            self.cash,
            "shares":          self.shares,
        }

        return observation, info

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        # ----------------------------------------------------
        # Portfolio value and regime BEFORE action
        # (current_step has not incremented yet)
        # ----------------------------------------------------

        previous_value = self._get_portfolio_value()
        regime_label   = self.df.iloc[self.current_step]["Market_Regime"]

        # ----------------------------------------------------
        # Execute action (updates cash, shares, _last_trade_value)
        # ----------------------------------------------------

        self._execute_trade(action)

        # ----------------------------------------------------
        # Advance to next trading day, THEN value the portfolio.
        #
        # The trade executes at today's close; the reward is the
        # realised change in portfolio value by the next close. This
        # is what lets price moves (not just trading costs) reach the
        # reward. The agent only ever observes day-t data when acting,
        # so this introduces no look-ahead.
        # ----------------------------------------------------

        self.current_step += 1

        current_value = self._get_portfolio_value()

        # ----------------------------------------------------
        # Regime-conditional reward (regime at decision time)
        # ----------------------------------------------------

        reward = self._compute_reward(
            previous_value,
            current_value,
            regime_label,
        )

        # ----------------------------------------------------
        # Episode termination
        # ----------------------------------------------------

        terminated = (self.current_step >= len(self.df) - 1)
        truncated  = False

        # ----------------------------------------------------
        # Next observation
        # ----------------------------------------------------

        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Info dict
        # ----------------------------------------------------

        info = {
            "portfolio_value": current_value,
            "cash":            self.cash,
            "shares":          self.shares,
            "regime":          regime_label,
            "regime_adaptive": self.regime_adaptive,
        }

        return observation, reward, terminated, truncated, info

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        row = self.df.iloc[self.current_step]

        print("=" * 50)
        print("REGIMEX TRADING ENVIRONMENT")
        print("=" * 50)
        print(f"Date:           {row['Date'].date()}")
        print(f"Price:          Rs.{row['Close']:.2f}")
        print(f"Market Regime:  {row['Market_Regime']}")
        print(f"Regime-Adaptive: {self.regime_adaptive}")
        print(f"Cash:           Rs.{self.cash:.2f}")
        print(f"Shares:         {self.shares}")
        print(f"Peak Value:     Rs.{self.peak_value:.2f}")
        print(
            f"Portfolio Value: "
            f"Rs.{self._get_portfolio_value():.2f}"
        )