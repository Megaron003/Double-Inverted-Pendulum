import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Acrobot-v1")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=200_000)

model.save("ppo_acrobot")