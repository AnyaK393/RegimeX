import numpy as np

from trading_env import RegimeXTradingEnv

# ============================================================
# REGIMEX - POOLED MULTI-STOCK TRAINING ENVIRONMENT
# ============================================================
# Same trading mechanics and reward as RegimeXTradingEnv, but each
# episode is played on a randomly chosen stock's TRAIN split. Pooling the
# five stocks multiplies the training data ~5x; the observation is
# stock-agnostic (returns, volatility, Hurst, regime, holdings).
# ============================================================


class PooledTradingEnv(RegimeXTradingEnv):

    def __init__(self, dfs, **kwargs):
        self.dfs = [d.reset_index(drop=True) for d in dfs]
        super().__init__(df=self.dfs[0], **kwargs)

    def reset(self, seed=None, options=None):
        # Seed first so self.np_random is initialised, then pick the stock
        super().reset(seed=seed)
        self.df = self.dfs[int(self.np_random.integers(len(self.dfs)))]
        return super().reset(seed=None)
