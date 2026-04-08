import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# ==============================
# CONFIGURAÇÕES
# ==============================
ENV_NAME = "Acrobot-v1"
MODEL_PATH = "ppo_acrobot"
TRAIN = True            # True = treinar | False = só executar
TIMESTEPS = 1_000_000

# ==============================
# CRIAR AMBIENTE (TREINO)
# ==============================
env = gym.make(ENV_NAME, max_episode_steps=500)

# ==============================
# TREINAMENTO
# ==============================
if TRAIN:
    print("🚀 Treinando PPO no Acrobot...")

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
        ent_coef=0.01,  # mantém exploração
        policy_kwargs=dict(net_arch=[256, 256])
    )

    model.learn(total_timesteps=TIMESTEPS)

    model.save(MODEL_PATH)
    print("✅ Modelo salvo!")

else:
    print("📂 Carregando modelo...")
    model = PPO.load(MODEL_PATH)

env.close()

# ==============================
# EXECUÇÃO COM RENDER
# ==============================
print("🎮 Executando simulação...")

env_render = gym.make(ENV_NAME, render_mode="human", max_episode_steps=500)

obs, info = env_render.reset()

for step in range(2000):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env_render.step(action)

    # LOG DO EPISÓDIO
    if terminated:
        print(f"🏆 Sucesso no step {step}!")
    
    if truncated:
        print(f"⏱️ Tempo esgotado no step {step}")

    # RESET
    if terminated or truncated:
        obs, info = env_render.reset()

env_render.close()