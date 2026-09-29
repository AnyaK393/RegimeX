import numpy as np
from gymnasium import spaces

from trading_env import RegimeXTradingEnv

# ============================================================
# REGIMEX - POSITION-SIZING ENVIRONMENT  (v3)
# ============================================================
#
# Why v3 exists (diagnosed from the v1/v2 negative result):
#   1. The v1 reward charged lambda_fee * turnover = 0.01 for a full trade.
#      That is ~20x a typical daily return AND double-counts costs that are
#      already deducted from portfolio value. It pushes agents to never trade
#      (collapse to always-invested / always-cash). Removed here.
#   2. The v1 risk term was lambda * drawdown-from-peak, which is zero
#      whenever the agent is in cash or at a new high, so adaptive and blind
#      agents received nearly identical rewards. Replaced by a dense
#      per-step mean-variance penalty  0.5 * gamma(regime) * r_t^2.
#   3. All-in/all-out actions cannot express "take less risk in a volatile
#      regime". The agent now chooses a TARGET EXPOSURE.
#
# Actions : Discrete(5) -> target exposure in {0, .25, .5, .75, 1} of equity.
# Reward  : scale * [ ln(V_t+1/V_t) - 0.5 * gamma(regime_t) * r_t^2
#                     - lambda_fee * turnover ]          (lambda_fee = 0)
# Regime-adaptive: gamma depends on the regime known at decision time.
# Regime-blind   : gamma is the train-frequency-weighted mean of the three
#                  regime values, so average risk aversion is IDENTICAL and
#                  the only difference is regime-conditioning.
#
# Gamma ratios (Normal : Weak_Bear : High_Vol = 1 : 3 : 6) follow the ordering
# already used by the v1 lambda_risk config. Values were fixed a priori and
# are NOT tuned on validation or test data.
# ============================================================

EXPOSURES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
DEFAULT_GAMMA = {"Normal_Market": 3.0, "Weak_Bear": 9.0, "High_Volatility": 18.0}
NO_TRADE_BAND = 0.05          # ignore rebalances smaller than 5% of equity


def blind_gamma_from(train_dfs, gamma_cfg=None) -> float:
    """Train-frequency-weighted mean gamma (equal average risk aversion)."""
    import pandas as pd
    gamma_cfg = gamma_cfg or DEFAULT_GAMMA
    d = pd.concat(train_dfs)
    freq = d["Market_Regime"].value_counts(normalize=True)
    return float(sum(freq.get(r, 0.0) * g for r, g in gamma_cfg.items()))


class RegimeXExposureEnv(RegimeXTradingEnv):

    def __init__(self, df=None, regime_adaptive=True, gamma_config=None,
                 blind_gamma=None, reward_scale=100.0, lambda_fee=0.0, **kwargs):
        super().__init__(df=df, regime_adaptive=regime_adaptive,
                         lambda_fee=lambda_fee, **kwargs)
        self.action_space = spaces.Discrete(len(EXPOSURES))
        self.gamma_cfg = gamma_config or DEFAULT_GAMMA
        self.blind_gamma = (blind_gamma if blind_gamma is not None
                            else float(np.mean(list(self.gamma_cfg.values()))))
        self.reward_scale = reward_scale
        self._last_exposure = 0.0

    # observation: last element = current exposure fraction (not just 0/1)
    def _exposure(self):
        v = self._get_portfolio_value()
        return float(self.shares * self.df.iloc[self.current_step]["Close"] / v) if v > 0 else 0.0

    def _get_observation(self):
        obs = super()._get_observation()
        obs[-1] = self._exposure()
        return obs

    def _gamma(self, regime_label):
        if not self.regime_adaptive:
            return self.blind_gamma
        return self.gamma_cfg.get(regime_label, self.blind_gamma)

    def _compute_reward(self, previous_value, current_value, regime_label):
        eps = 1e-8
        r = current_value / max(previous_value, eps) - 1.0
        log_ret = np.log(max(current_value, eps) / max(previous_value, eps))
        turnover = self._last_trade_value / max(previous_value, eps)
        reward = (log_ret
                  - 0.5 * self._gamma(regime_label) * r * r
                  - self.lambda_fee * turnover)
        return float(self.reward_scale * reward)

    def _execute_trade(self, action):
        row = self.df.iloc[self.current_step]
        price, vol = row["Close"], row["Rolling_Volatility"]
        self._last_trade_value = 0.0

        slip = self.base_slippage + self.slippage_multiplier * vol
        impact = self.base_market_impact + self.market_impact_multiplier * vol

        equity = self.cash + self.shares * price
        w = float(EXPOSURES[int(action)])
        delta = w * equity - self.shares * price

        if w == 0.0 and self.shares > 0:                       # full exit
            n = self.shares
            exec_price = price * (1 - slip - impact)
            tv = n * exec_price
            self.cash += tv * (1 - self.transaction_cost)
            self.shares = 0
            self._last_trade_value = tv

        elif delta > NO_TRADE_BAND * equity:                    # buy up to target
            exec_price = price * (1 + slip + impact)
            n = int(min(delta / exec_price,
                        self.cash // (exec_price * (1 + self.transaction_cost))))
            if n > 0:
                tv = n * exec_price
                self.cash -= tv * (1 + self.transaction_cost)
                self.shares += n
                self._last_trade_value = tv

        elif -delta > NO_TRADE_BAND * equity and self.shares > 0:   # trim to target
            exec_price = price * (1 - slip - impact)
            n = min(self.shares, int(-delta / price))
            if n > 0:
                tv = n * exec_price
                self.cash += tv * (1 - self.transaction_cost)
                self.shares -= n
                self._last_trade_value = tv

        self._last_exposure = float(self.shares * price / max(self.cash + self.shares * price, 1e-8))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["exposure"] = self._last_exposure
        return obs, reward, terminated, truncated, info


class PooledExposureEnv(RegimeXExposureEnv):
    """Each episode is played on a randomly chosen stock's train split."""

    def __init__(self, dfs, **kwargs):
        self.dfs = [d.reset_index(drop=True) for d in dfs]
        super().__init__(df=self.dfs[0], **kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.df = self.dfs[int(self.np_random.integers(len(self.dfs)))]
        return super().reset(seed=None)
