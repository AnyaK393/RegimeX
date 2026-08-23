from trading_env import RegimeXTradingEnv


# ============================================================
# REGIMEX - TRADING ENVIRONMENT TEST
# ============================================================

print("=" * 60)
print("REGIMEX - GYMNASIUM ENVIRONMENT TEST")
print("=" * 60)


# Create environment
env = RegimeXTradingEnv()


# ------------------------------------------------------------
# Test action and observation spaces
# ------------------------------------------------------------

print("\nAction space:")
print(env.action_space)

print("\nObservation space:")
print(env.observation_space)


# ------------------------------------------------------------
# Reset environment
# ------------------------------------------------------------

observation, info = env.reset()

print("\nInitial observation:")
print(observation)

print("\nInitial information:")
print(info)


# ------------------------------------------------------------
# Test a few actions
# ------------------------------------------------------------

actions = [1, 0, 0, 2, 0]

print("\n" + "=" * 60)
print("TESTING TRADING ACTIONS")
print("=" * 60)


for action in actions:

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    print("\n----------------------------------------")

    print(f"Action: {action}")

    print(f"Observation: {observation}")

    print(f"Reward: ₹{reward:.2f}")

    print(f"Portfolio Value: ₹{info['portfolio_value']:.2f}")

    print(f"Cash: ₹{info['cash']:.2f}")

    print(f"Shares: {info['shares']}")

    if terminated or truncated:
        break


print("\nEnvironment test completed successfully!")