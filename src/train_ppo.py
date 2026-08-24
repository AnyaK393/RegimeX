import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trading_env import RegimeXTradingEnv
from data_split import get_splits

# ============================================================
# REGIMEX - PPO TRAINING
# ============================================================

DATA_PATH = "data/processed/RELIANCE_regimes.csv"
MODEL_DIR = "models/ppo_regimex"
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
    return RegimeXTradingEnv(df=train_df, regime_adaptive=True)

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
TOTAL_TIMESTEPS = 500_000
print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    tb_log_name="PPO_run"
)

# 6. Save final model
final_model_path = os.path.join(MODEL_DIR, "ppo_regimex_final")
model.save(final_model_path)
print(f"\nTraining complete. Final model saved to: {final_model_path}.zip")
print("=" * 60)
