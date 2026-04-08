import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from stable_baselines3 import PPO


class DoublePendulumEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ===== MUJOCO MODEL =====
        self.model = mujoco.MjModel.from_xml_path(
            r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Project_Final\Acrobot\MuJoCo\model_double_inverted_pendulum.xml"
        )
        self.data = mujoco.MjData(self.model)

        # ===== ACTION SPACE (NORMALIZADO) =====
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ===== OBSERVATION SPACE =====
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # ===== ESTADO INICIAL =====
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # pequena perturbação só nos ângulos
        self.data.qpos[1:] += np.random.uniform(-0.1, 0.1, size=2)

        return self._get_obs(), {}

    def step(self, action):
        # ===== ESCALA DA FORÇA =====
        force = 10.0 * float(action[0])
        self.data.ctrl[0] = force

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # ===== ESTADOS =====
        x = self.data.qpos[0]
        theta1 = self.data.qpos[1]
        theta2 = self.data.qpos[2]

        x_dot = self.data.qvel[0]

        # ===== REWARD CORRIGIDO =====
        reward = (
            np.cos(theta1) +
            np.cos(theta2)
            - 0.1 * (x ** 2)              # manter no centro
            - 0.001 * (x_dot ** 2)        # evitar drift
            - 0.01 * (force ** 2)         # suavidade
        )

        # ===== TERMINAÇÃO =====
        terminated = False

        # evita fugir infinitamente
        truncated = abs(x) > 2.0

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel
        ]).astype(np.float32)


# ==============================
# TREINAMENTO
# ==============================

env = DoublePendulumEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
)

model.learn(total_timesteps=500_000)

model.save("ppo_double_pendulum")

print("✅ Modelo treinado e salvo!")


# ==============================
# SIMULAÇÃO
# ==============================

obs, _ = env.reset()

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    mujoco.mj_step(env.model, env.data)

    if step % 50 == 0:
        print(f"Step {step} | Reward: {reward:.3f}")

    if truncated:
        print("⚠️ Episódio truncado (saiu do limite)")
        obs, _ = env.reset()