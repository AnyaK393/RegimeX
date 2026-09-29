import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trading_env import RegimeXTradingEnv, obs_stats_from
from data_split import get_splits

# ============================================================
# REGIMEX - PPO TRAINING
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--blind", action="store_true",
                    help="train the regime-blind agent (regime_adaptive=False)")
parser.add_argument("--scaled", action="store_true",
                    help="v2 observation: z-scored features + one-hot regime")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timesteps", type=int, default=500_000)
args = parser.parse_args()

# Adaptive agent keeps the original paths (seed 0); other runs get a suffix.
TAG       = ("blind" if args.blind else "adaptive") + ("_scaled" if args.scaled else "") + f"_s{args.seed}"
DATA_PATH = "data/processed/RELIANCE_regimes.csv"
MODEL_DIR = f"models/ppo_regimex/{TAG}"
LOG_DIR   = "logs/ppo_regimex"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 60)
print("REGIMEX - PPO TRAINING")
print("=" * 60)

# 1. Load Data & Get Splits
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df_full = pd.read_csv(DATA_PATH, parse_dates=["Date"])
train_df, val_df, test_df = get_splits(df_full, verbose=True)

print(f"\nTraining on {len(train_df)} days (Train split).")

# 2. Create Environment
# We use regime_adaptive=True so the agent experiences the
# regime-conditioned reward during training.
def make_env():
    return RegimeXTradingEnv(
        df=train_df, regime_adaptive=not args.blind,
        scaled_obs=args.scaled,
        obs_stats=obs_stats_from(train_df) if args.scaled else None,
    )

env = DummyVecEnv([make_env])

# 3. Setup PPO Agent
# Hyperparameters for financial time series
policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
    tensorboard_log=LOG_DIR,
    seed=args.seed,
    verbose=1,
)

# 4. Checkpoint Callback
# Save model every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=MODEL_DIR,
    name_prefix="ppo_model"
)

# 5. Train
TOTAL_TIMESTEPS = args.timesteps
print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    tb_log_name=f"PPO_{TAG}"
)

# 6. Save final model
final_model_path = os.path.join(MODEL_DIR, "ppo_regimex_final")  # -> models/ppo_regimex/<TAG>/ppo_regimex_final.zip
model.save(final_model_path)
print(f"\nTraining complete. Final model saved to: {final_model_path}.zip")
print("=" * 60)
