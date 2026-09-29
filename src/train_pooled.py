import os
import sys
import argparse

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(__file__))

from data_split import get_splits
from trading_env import obs_stats_from
from pooled_env import PooledTradingEnv

# ============================================================
# REGIMEX - PPO TRAINING ON THE POOLED FIVE-STOCK TRAIN SET
# ============================================================

TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]

parser = argparse.ArgumentParser()
parser.add_argument("--blind", action="store_true")
parser.add_argument("--scaled", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timesteps", type=int, default=1_000_000)
args = parser.parse_args()

TAG = ("blind" if args.blind else "adaptive") + ("_scaled" if args.scaled else "") + f"_s{args.seed}"
MODEL_DIR = f"models/ppo_pooled/{TAG}"
os.makedirs(MODEL_DIR, exist_ok=True)

train_dfs = []
for t in TICKERS:
    df = pd.read_csv(f"data/processed/{t}_regimes.csv", parse_dates=["Date"])
    train_dfs.append(get_splits(df, verbose=False)[0])
print(f"Pooled train: {sum(len(d) for d in train_dfs)} bars across {len(train_dfs)} stocks")

stats = obs_stats_from(pd.concat(train_dfs)) if args.scaled else None
env = DummyVecEnv([lambda: PooledTradingEnv(
    train_dfs, regime_adaptive=not args.blind, scaled_obs=args.scaled, obs_stats=stats)])

model = PPO(
    "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, ent_coef=0.01, clip_range=0.2,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    tensorboard_log="logs/ppo_pooled", seed=args.seed, verbose=1,
)
model.learn(
    total_timesteps=args.timesteps,
    callback=CheckpointCallback(save_freq=100_000, save_path=MODEL_DIR, name_prefix="ppo_model"),
    tb_log_name=f"PPO_{TAG}",
)
model.save(os.path.join(MODEL_DIR, "ppo_regimex_final"))
print("saved", MODEL_DIR)
