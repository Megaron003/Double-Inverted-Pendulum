import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==============================
# CONFIGURAÇÕES
# ==============================
ENV_NAME = "InvertedDoublePendulum-v4"
MODEL_PATH = "ppo_double_pendulum"

TRAIN = True          # True = treina | False = só roda modelo salvo
TIMESTEPS = 500_000   # Pode aumentar depois

# ==============================
# CRIAR AMBIENTE
# ==============================
def make_env():
    return gym.make(ENV_NAME)

env = DummyVecEnv([make_env])

# ==============================
# TREINAMENTO
# ==============================
if TRAIN:
    print("Treinando modelo PPO...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save(MODEL_PATH)
    print("Modelo salvo!")

else:
    print("Carregando modelo...")
    model = PPO.load(MODEL_PATH)

# ==============================
# EXECUÇÃO COM RENDER
# ==============================
print("Executando simulação...")

env_render = gym.make(ENV_NAME, render_mode="human")
obs, _ = env_render.reset()

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env_render.step(action)

    if done or truncated:
        obs, _ = env_render.reset()

env_render.close()