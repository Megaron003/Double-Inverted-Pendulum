"""
Carrega o ppo_mujoco.pt e renderiza o pêndulo em tempo real.
Ajuste apenas PPO_CHECKPOINT para o caminho do seu .pt.
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# ── Caminho do checkpoint ──────────────────────────────────────────────────
PPO_CHECKPOINT = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\ppo_mujoco.pt"   # ← altere para o caminho correto

# ── Configuração da sessão ─────────────────────────────────────────────────
N_EPISODES = 5      # quantos episódios renderizar
MAX_STEPS  = 1000   # máximo de steps por episódio (~2s de simulação)
SEED       = 42


# ── Arquitetura (deve ser idêntica à do treino) ────────────────────────────
class Actor(nn.Module):
    def __init__(self, obs_dim=11, act_dim=1, hidden=(256, 256)):
        super().__init__()
        dims   = [obs_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        self.trunk    = nn.Sequential(*layers)
        self.mu_head  = nn.Linear(hidden[-1], act_dim)
        self.log_std  = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu  = torch.tanh(self.mu_head(self.trunk(obs)))
        std = self.log_std.exp().expand_as(mu).clamp(0.05, 1.0)
        return mu, std

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """Ação determinística (sem ruído de exploração)."""
        t = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self(t)
        return mu.squeeze().numpy()


# ── Carregar política ──────────────────────────────────────────────────────
ckpt  = torch.load(PPO_CHECKPOINT, map_location="cpu", weights_only=False)
actor = Actor()
actor.load_state_dict(ckpt["actor"])
actor.eval()
print(f"[OK] Política carregada: {PPO_CHECKPOINT}")

# ── Ambiente com renderização ──────────────────────────────────────────────
env = gym.make("InvertedDoublePendulum-v4", render_mode="human")

print(f"\nExecutando {N_EPISODES} episódios — feche a janela para encerrar.\n")
print(f"{'Ep':>4}  {'Steps':>6}  {'Recomp.':>10}  Resultado")
print("─" * 38)

for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=SEED + ep)
    ep_reward = 0.0

    for step in range(MAX_STEPS):
        action    = actor.act(obs)
        obs, rew, terminated, truncated, _ = env.step(np.atleast_1d(action))
        ep_reward += rew
        if terminated or truncated:
            break

    survived  = (step + 1 == MAX_STEPS)
    resultado = "equilibrou!" if survived else "caiu"
    print(f"{ep+1:>4}  {step+1:>6}  {ep_reward:>10.1f}  {resultado}")

env.close()
print("\nEncerrado.")
