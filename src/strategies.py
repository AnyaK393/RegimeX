import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from trading_env import RegimeXTradingEnv

# ============================================================
# REGIMEX - STRATEGIES / BASELINE LADDER
# ============================================================
#
# Every strategy - Buy & Hold, ARIMA, XGBoost and the PPO agents - is
# executed through the SAME RegimeXTradingEnv, so all of them face
# identical transaction costs, slippage, market impact and capital.
# A strategy is just a policy: (obs, t, env) -> action in {0,1,2}.
#
# Baseline ladder:
#   1. Buy & Hold          - passive benchmark
#   2. ARIMA(1,0,1)        - classical time-series; long when the one-step
#                            forecast of the next daily return is > 0
#   3. XGBoost             - supervised ML; predicts P(next-day return > 0)
#                            from the same state features the RL agent sees
#   4. PPO regime-blind    - RL, constant risk penalty
#   5. PPO regime-adaptive - RL, regime-conditional risk penalty
#
# All signals use information available at the close of day t only.
# ============================================================

FEATURES = ["Daily_Return", "Rolling_Volatility", "Hurst", "Regime"]


# ------------------------------------------------------------
# Rollout
# ------------------------------------------------------------

def rollout(df: pd.DataFrame, policy, regime_adaptive: bool = True, env_cls=None, **env_kwargs) -> pd.DataFrame:
    """Runs one full episode of `policy` over `df` and returns a per-step log."""
    df = df.reset_index(drop=True)
    env = (env_cls or RegimeXTradingEnv)(df=df, regime_adaptive=regime_adaptive, **env_kwargs)
    obs, _ = env.reset()

    rows, done, t = [], False, 0
    while not done:
        prev_value = env._get_portfolio_value()
        regime = df["Market_Regime"].iloc[t]
        date = df["Date"].iloc[t]

        action = int(policy(obs, t, env))
        obs, reward, done, _, info = env.step(action)

        rows.append({
            "date":        date,
            "regime":      regime,
            "action":      action,
            "prev_value":  prev_value,
            "value":       info["portfolio_value"],
            "trade_value": env._last_trade_value,
            "shares":      info["shares"],
            "exposure":    info.get("exposure", float(info["shares"] > 0)),
            "reward":      reward,
        })
        t += 1

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Policies
# ------------------------------------------------------------

def buy_and_hold_policy():
    return lambda obs, t, env: 1 if t == 0 else 0


def signal_policy(desired):
    """
    desired[t] in {0,1}: target position (flat / fully long) decided at the
    close of day t. Converts to BUY/SELL/HOLD given the current holding.
    """
    desired = np.asarray(desired)

    def policy(obs, t, env):
        holding = env.shares > 0
        if desired[t] == 1 and not holding:
            return 1
        if desired[t] == 0 and holding:
            return 2
        return 0

    return policy


def ppo_policy(model):
    return lambda obs, t, env: model.predict(obs, deterministic=True)[0]


# ------------------------------------------------------------
# ARIMA baseline
# ------------------------------------------------------------

def arima_signals(full_df, split_df, order=(1, 0, 1), refit_every=21, window=750):
    """
    Walk-forward ARIMA signal. For each day t of `split_df`, fits on daily
    returns up to and including day t (last `window` observations, refit every
    `refit_every` days, parameters re-used in between) and goes long if the
    one-step-ahead forecast is positive. Zero-mean model (trend='n') so the
    signal comes from return dynamics rather than the sample drift.
    """
    from statsmodels.tsa.arima.model import ARIMA

    full_df = full_df.sort_values("Date").reset_index(drop=True)
    ret = full_df["Daily_Return"].to_numpy() * 100.0   # % units, better conditioned
    start = int(np.where(full_df["Date"].to_numpy() == split_df["Date"].iloc[0])[0][0])

    signals, res = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(len(split_df)):
            endog = ret[max(0, start + t + 1 - window): start + t + 1]
            try:
                if res is None or t % refit_every == 0:
                    res = ARIMA(endog, order=order, trend="n").fit()
                else:
                    res = res.apply(endog, refit=False)
                forecast = float(np.asarray(res.forecast(1))[0])
            except Exception:
                forecast = 0.0
            signals.append(int(forecast > 0))
    return np.array(signals)


# ------------------------------------------------------------
# XGBoost baseline
# ------------------------------------------------------------

def fit_xgboost(train_df):
    """
    Supervised next-day-direction classifier on the same state features.
    Label for day t = 1 if Daily_Return on day t+1 is positive.
    """
    from xgboost import XGBClassifier

    train_df = train_df.reset_index(drop=True)
    X = train_df[FEATURES].iloc[:-1].to_numpy()
    y = (train_df["Daily_Return"].shift(-1).iloc[:-1] > 0).astype(int).to_numpy()

    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=0,
        eval_metric="logloss", n_jobs=1,
    )
    model.fit(X, y)
    return model


def xgboost_signals(model, split_df):
    proba = model.predict_proba(split_df[FEATURES].to_numpy())[:, 1]
    return (proba > 0.5).astype(int)
