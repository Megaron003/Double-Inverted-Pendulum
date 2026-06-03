"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PPO — Proximal Policy Optimization                                      ║
║  Pêndulo Invertido Duplo · Objetivo: equilibrar em θ₁ = θ₂ = 0           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Arquitetura:                                                            ║
║  · Ambiente  → MLP treinada  f(x,τ) → [α₁,α₂]  (substituta MuJoCo)       ║
║  · Ator      → política π_φ(x) → distribuição Gaussiana sobre τ          ║
║  · Crítico   → V_ψ(x) → escalar (função valor)                           ║
║                                                                          ║
║  Fluxo PPO:                                                              ║
║  1. Coletar trajetórias rodando π_φ no ambiente-MLP                      ║
║  2. Calcular retornos e vantagens (GAE)                                  ║
║  3. Atualizar ator com clipping (ε=0.2) — impede passos grandes          ║
║  4. Atualizar crítico minimizando erro de valor                          ║
║  5. Repetir                                                              ║
║                                                                          ║
║  Recompensa de equilíbrio:                                               ║
║  r(x,τ) = exp(−α·(θ₁²+θ₂²)) − β·(ω₁²+ω₂²) − γ·‖τ‖²                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ── Reprodutibilidade ──────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# ── Caminhos ───────────────────────────────────────────────────────────────
CHECKPOINT_MLP  = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\MLP_Results\\mlp_tanh_pendulo.pt"   # ← modelo treinado
OUT_POLICY      = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\PPO_Results\\ppo_policy.pt"
OUT_FIG         = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\PPO_Results\\ppo_treinamento.png"


# ══════════════════════════════════════════════════════════════════════════
# 1. AMBIENTE — MLP como substituta do MuJoCo
# ══════════════════════════════════════════════════════════════════════════

class MLPtanh(nn.Module):
    """Mesma arquitetura usada no treino — necessária para carregar o .pt."""
    def __init__(self, dim_in=8, dim_out=2, hidden=(128,128,64)):
        super().__init__()
        dims = [dim_in] + list(hidden) + [dim_out]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PendulumEnv:
    """
    Ambiente do pêndulo duplo usando a MLP como dinâmica.

    Estado interno: [θ₁, θ₂, ω₁, ω₂]  (4 variáveis físicas)
    Observação    : [sinθ₁, cosθ₁, sinθ₂, cosθ₂, ω₁, ω₂]  (6 features)
    Ação          : [τ₁, τ₂]  (torques em N·m)

    O estado de equilíbrio alvo é θ₁=θ₂=0, ω₁=ω₂=0.
    """

    # Limites físicos razoáveis para o pêndulo duplo
    THETA_MAX  = np.pi          # ângulo máximo (rad)
    OMEGA_MAX  = 20.0           # velocidade angular máxima (rad/s)
    TAU_MAX    = 5.0            # torque máximo (N·m)
    DT         = 0.002          # passo de tempo (s) — igual ao MuJoCo

    # Limites para detecção de estado inválido (episódio termina)
    OMEGA_CLIP = 50.0

    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        cfg  = ckpt.get("config", {})

        self.dyn = MLPtanh(
            dim_in  = 8,
            dim_out = 2,
            hidden  = cfg.get("hidden", (128,128,64)),
        ).to(DEVICE)
        self.dyn.load_state_dict(ckpt["model_state_dict"])
        self.dyn.eval()

        # Scalers para normalizar a entrada da MLP dinâmica
        self.X_mean  = torch.tensor(ckpt["scaler_X_mean"],  dtype=torch.float32)
        self.X_scale = torch.tensor(ckpt["scaler_X_scale"], dtype=torch.float32)
        self.y_mean  = torch.tensor(ckpt["scaler_y_mean"],  dtype=torch.float32)
        self.y_scale = torch.tensor(ckpt["scaler_y_scale"], dtype=torch.float32)

        self.theta1 = self.theta2 = 0.0
        self.omega1 = self.omega2 = 0.0
        self.rng = np.random.default_rng(SEED)

    # ── Dinâmica ──────────────────────────────────────────────────────────

    def _predict_alpha(self, tau1: float, tau2: float):
        """Chama a MLP de dinâmica e retorna (α₁, α₂) em rad/s²."""
        x_raw = torch.tensor([[
            np.sin(self.theta1), np.cos(self.theta1),
            np.sin(self.theta2), np.cos(self.theta2),
            self.omega1, self.omega2,
            tau1, tau2,
        ]], dtype=torch.float32)
        x_norm = (x_raw - self.X_mean) / self.X_scale
        with torch.no_grad():
            y_norm = self.dyn(x_norm)
        alpha = (y_norm * self.y_scale + self.y_mean).squeeze()
        return float(alpha[0]), float(alpha[1])

    def _step_euler(self, tau1: float, tau2: float):
        """Integração de Euler de um passo."""
        a1, a2 = self._predict_alpha(tau1, tau2)
        self.omega1 += a1 * self.DT
        self.omega2 += a2 * self.DT
        self.theta1 += self.omega1 * self.DT
        self.theta2 += self.omega2 * self.DT
        # Normalizar ângulos para (−π, π]
        self.theta1 = ((self.theta1 + np.pi) % (2*np.pi)) - np.pi
        self.theta2 = ((self.theta2 + np.pi) % (2*np.pi)) - np.pi

    # ── Interface Gym-like ────────────────────────────────────────────────

    def reset(self, random_init: bool = True):
        """
        Reseta o ambiente.
        random_init=True: condição inicial aleatória próxima do equilíbrio.
        random_init=False: começa exatamente em θ=0, ω=0.
        """
        if random_init:
            # Pequena perturbação ao redor do equilíbrio
            scale_th = 0.3   # rad
            scale_om = 0.5   # rad/s
            self.theta1 = float(self.rng.uniform(-scale_th, scale_th))
            self.theta2 = float(self.rng.uniform(-scale_th, scale_th))
            self.omega1 = float(self.rng.uniform(-scale_om, scale_om))
            self.omega2 = float(self.rng.uniform(-scale_om, scale_om))
        else:
            self.theta1 = self.theta2 = 0.0
            self.omega1 = self.omega2 = 0.0
        return self._obs()

    def step(self, action: np.ndarray):
        """
        action: array [τ₁, τ₂] já em escala física (N·m).
        Retorna: obs, reward, done, info
        """
        tau1 = float(np.clip(action[0], -self.TAU_MAX, self.TAU_MAX))
        tau2 = float(np.clip(action[1], -self.TAU_MAX, self.TAU_MAX))

        self._step_euler(tau1, tau2)

        obs    = self._obs()
        reward = self._reward(tau1, tau2)
        done   = self._terminal()

        return obs, reward, done, {}

    def _obs(self) -> np.ndarray:
        """Observação: [sinθ₁, cosθ₁, sinθ₂, cosθ₂, ω₁, ω₂]."""
        return np.array([
            np.sin(self.theta1), np.cos(self.theta1),
            np.sin(self.theta2), np.cos(self.theta2),
            self.omega1 / self.OMEGA_MAX,   # normalizado
            self.omega2 / self.OMEGA_MAX,
        ], dtype=np.float32)

    def _reward(self, tau1: float, tau2: float) -> float:
        """
        Recompensa para equilíbrio em θ₁=θ₂=0:

        r = exp(−3·(θ₁²+θ₂²))           ← máximo quando ângulos = 0
          − 0.1·(ω₁²+ω₂²)/ω_max²        ← penaliza velocidades altas
          − 0.05·(τ₁²+τ₂²)/τ_max²       ← penaliza esforço de controle

        Intervalo aproximado: [−0.15, 1.0]
        """
        angle_cost = np.exp(-3.0 * (self.theta1**2 + self.theta2**2))
        vel_cost   = 0.1  * (self.omega1**2 + self.omega2**2) / self.OMEGA_MAX**2
        ctrl_cost  = 0.05 * (tau1**2 + tau2**2) / self.TAU_MAX**2
        return float(angle_cost - vel_cost - ctrl_cost)

    def _terminal(self) -> bool:
        """Termina se velocidades explodirem (dinâmica instável)."""
        return (abs(self.omega1) > self.OMEGA_CLIP or
                abs(self.omega2) > self.OMEGA_CLIP)

    @property
    def obs_dim(self) -> int:
        return 6

    @property
    def act_dim(self) -> int:
        return 2


# ══════════════════════════════════════════════════════════════════════════
# 2. REDES — ATOR E CRÍTICO
# ══════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    Política Gaussiana:  π_φ(x) = N(μ_φ(x), σ_φ(x))

    · Saída: média μ (uma por ação) + log(σ) aprendido
    · tanh na saída de μ para limitar ao intervalo (−1, +1),
      depois escalado para (−TAU_MAX, +TAU_MAX)
    · σ é mantido positivo via softplus
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: tuple = (128, 128)):
        super().__init__()
        dims = [obs_dim] + list(hidden)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*layers)

        self.mu_head      = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)

        # Inicialização menor para saída — política começa conservadora
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1.0)   # σ inicial ≈ 0.37

    def forward(self, obs: torch.Tensor):
        feat    = self.trunk(obs)
        mu      = torch.tanh(self.mu_head(feat))          # ∈ (−1, +1)
        log_std = self.log_std_head(feat).clamp(-3, 0.5)  # limita std
        std     = log_std.exp()
        return mu, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std)

    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Amostra uma ação dado obs numpy."""
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            mu, std = self(obs_t)
        if deterministic:
            action_norm = mu
        else:
            action_norm = Normal(mu, std).sample()
        action_norm = action_norm.clamp(-1, 1)
        # Escalar para espaço físico (N·m)
        action = action_norm.squeeze().numpy() * PendulumEnv.TAU_MAX
        return action


class Critic(nn.Module):
    """
    Função valor: V_ψ(x) → escalar
    Estima o retorno esperado a partir do estado x.
    """

    def __init__(self, obs_dim: int, hidden: tuple = (128, 128)):
        super().__init__()
        dims = [obs_dim] + list(hidden) + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# 3. BUFFER DE TRAJETÓRIAS
# ══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Armazena uma época de interações com o ambiente.
    Calcula retornos e vantagens usando GAE (Generalized Advantage Estimation).
    """

    def __init__(self):
        self.obs     = []
        self.actions = []
        self.rewards = []
        self.dones   = []
        self.values  = []
        self.log_probs = []

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantages(self,
                                        last_value: float,
                                        gamma: float = 0.99,
                                        lam: float   = 0.95):
        """
        GAE — Generalized Advantage Estimation:

        δₜ  = rₜ + γ·V(sₜ₊₁) − V(sₜ)         ← erro TD
        Âₜ  = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...  ← vantagem suavizada

        γ controla horizonte temporal (0.99 = longo prazo)
        λ controla viés-variância (0.95 = baixa variância)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv   = 0.0

        values_np = np.array(self.values + [last_value], dtype=np.float32)

        for t in reversed(range(n)):
            mask      = 1.0 - float(self.dones[t])
            delta     = self.rewards[t] + gamma * values_np[t+1] * mask - values_np[t]
            last_adv  = delta + gamma * lam * mask * last_adv
            advantages[t] = last_adv

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalizar vantagens (reduz variância do gradiente)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.from_numpy(np.array(self.obs,      dtype=np.float32)),
            torch.from_numpy(np.array(self.actions,  dtype=np.float32)),
            torch.from_numpy(returns),
            torch.from_numpy(advantages),
            torch.from_numpy(np.array(self.log_probs, dtype=np.float32)),
        )

    def clear(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════════════
# 4. PPO
# ══════════════════════════════════════════════════════════════════════════

class PPO:
    """
    Proximal Policy Optimization (Schulman et al., 2017)

    Objetivo do ator com clipping:
        L_CLIP = E[min(r_t·Â_t, clip(r_t, 1−ε, 1+ε)·Â_t)]
        r_t = π_φ(aₜ|sₜ) / π_φ_old(aₜ|sₜ)   ← razão de probabilidades

    O clipping impede que a política se afaste demais da política anterior
    em cada atualização — isso é o que torna o PPO estável.
    """

    def __init__(self, env: PendulumEnv, cfg: dict):
        self.env = env
        self.cfg = cfg

        self.actor  = Actor(env.obs_dim,  env.act_dim,
                            hidden=cfg["actor_hidden"]).to(DEVICE)
        self.critic = Critic(env.obs_dim,
                             hidden=cfg["critic_hidden"]).to(DEVICE)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),
                                            lr=cfg["lr_actor"])
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                            lr=cfg["lr_critic"])

        self.buffer  = RolloutBuffer()
        self.history = {
            "ep_reward": [], "ep_len": [], "actor_loss": [],
            "critic_loss": [], "entropy": [], "ep_reward_mean": [],
        }

    # ── Coleta de trajetórias ─────────────────────────────────────────────

    def collect_rollouts(self, n_steps: int):
        """
        Roda n_steps interações com o ambiente e armazena no buffer.
        """
        obs = self.env.reset()
        ep_rewards, ep_lens = [], []
        ep_r, ep_l = 0.0, 0

        for _ in range(n_steps):
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                dist  = self.actor.get_dist(obs_t)
                # Amostra em espaço normalizado
                act_norm = dist.sample().clamp(-1, 1)
                log_prob = dist.log_prob(act_norm).sum(-1)
                value    = self.critic(obs_t).item()

            action = act_norm.squeeze().numpy() * PendulumEnv.TAU_MAX

            next_obs, reward, done, _ = self.env.step(action)

            self.buffer.add(obs, act_norm.squeeze().numpy(),
                            reward, done, value, log_prob.item())

            ep_r += reward
            ep_l += 1
            obs   = next_obs

            if done:
                ep_rewards.append(ep_r)
                ep_lens.append(ep_l)
                ep_r, ep_l = 0.0, 0
                obs = self.env.reset()

        # Valor do último estado (para bootstrap)
        obs_t      = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            last_val = self.critic(obs_t).item()

        return last_val, ep_rewards, ep_lens

    # ── Atualização PPO ───────────────────────────────────────────────────

    def update(self, obs_b, act_b, ret_b, adv_b, old_lp_b):
        """
        Executa n_epochs épocas de gradiente sobre o buffer coletado.
        Usa mini-batches aleatórios dentro de cada época.
        """
        cfg      = self.cfg
        n        = len(obs_b)
        a_losses, c_losses, entropies = [], [], []

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(n)

            for start in range(0, n, cfg["batch_size"]):
                mb = idx[start:start + cfg["batch_size"]]

                obs_mb  = obs_b[mb]
                act_mb  = act_b[mb]
                ret_mb  = ret_b[mb]
                adv_mb  = adv_b[mb]
                olp_mb  = old_lp_b[mb]

                # ── Ator ──────────────────────────────────────────────────
                dist     = self.actor.get_dist(obs_mb)
                new_lp   = dist.log_prob(act_mb).sum(-1)
                entropy  = dist.entropy().sum(-1).mean()

                # Razão de probabilidades r_t = π_new / π_old
                ratio    = (new_lp - olp_mb).exp()

                # Objetivo clipped
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1 - cfg["clip_eps"],
                                     1 + cfg["clip_eps"]) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= cfg["entropy_coef"] * entropy  # bônus de entropia

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                          cfg["max_grad_norm"])
                self.opt_actor.step()

                # ── Crítico ───────────────────────────────────────────────
                value_pred  = self.critic(obs_mb)
                critic_loss = F.mse_loss(value_pred, ret_mb)

                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                          cfg["max_grad_norm"])
                self.opt_critic.step()

                a_losses.append(actor_loss.item())
                c_losses.append(critic_loss.item())
                entropies.append(entropy.item())

        return np.mean(a_losses), np.mean(c_losses), np.mean(entropies)

    # ── Loop principal ────────────────────────────────────────────────────

    def train(self, total_steps: int):
        steps_done = 0
        iteration  = 0
        t0         = time.time()

        print(f"\n{'Iter':>6}  {'Steps':>8}  {'Recomp. média':>14}  "
              f"{'L_ator':>10}  {'L_crítico':>10}  {'Entropia':>10}  {'s':>6}")
        print("─" * 74)

        while steps_done < total_steps:
            # Coletar trajetórias
            last_val, ep_rews, ep_lens = self.collect_rollouts(
                self.cfg["n_steps"])

            steps_done += self.cfg["n_steps"]
            iteration  += 1

            # Calcular retornos e vantagens
            obs_b, act_b, ret_b, adv_b, old_lp_b = \
                self.buffer.compute_returns_and_advantages(last_val)
            self.buffer.clear()

            # Atualizar redes
            al, cl, ent = self.update(obs_b, act_b, ret_b, adv_b, old_lp_b)

            # Registrar histórico
            if ep_rews:
                mean_r = np.mean(ep_rews)
                self.history["ep_reward"].extend(ep_rews)
                self.history["ep_len"].extend(ep_lens)
                self.history["ep_reward_mean"].append(mean_r)
            else:
                mean_r = float("nan")
                self.history["ep_reward_mean"].append(float("nan"))

            self.history["actor_loss"].append(al)
            self.history["critic_loss"].append(cl)
            self.history["entropy"].append(ent)

            elapsed = time.time() - t0
            if iteration % 5 == 0 or iteration == 1:
                print(f"{iteration:>6}  {steps_done:>8}  {mean_r:>14.4f}  "
                      f"{al:>10.4f}  {cl:>10.4f}  {ent:>10.4f}  "
                      f"{elapsed:>5.0f}s")

        print(f"\nTreinamento concluído: {steps_done:,} steps em "
              f"{time.time()-t0:.1f}s")
        return self.history

    # ── Avaliação ─────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 10,
                 max_steps: int = 500,
                 random_init: bool = True) -> dict:
        """Avalia a política determinística (μ sem ruído)."""
        rewards, lens, survived = [], [], []

        for _ in range(n_episodes):
            obs  = self.env.reset(random_init=random_init)
            ep_r = 0.0
            for step in range(max_steps):
                action = self.actor.act(obs, deterministic=True)
                obs, r, done, _ = self.env.step(action)
                ep_r += r
                if done:
                    break
            rewards.append(ep_r)
            lens.append(step + 1)
            survived.append(step + 1 == max_steps)

        return {
            "mean_reward":   float(np.mean(rewards)),
            "std_reward":    float(np.std(rewards)),
            "mean_len":      float(np.mean(lens)),
            "survival_rate": float(np.mean(survived)),
        }

    # ── Salvar e carregar ─────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor_state":  self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "config":       self.cfg,
            "history":      self.history,
        }, path)
        print(f"[OK] Política salva: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        print(f"[OK] Política carregada: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 5. FIGURA DE DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════

def plot_training(history: dict, eval_result: dict):
    C = {"bg":"#0D1117","panel":"#161B22","grid":"#21262D","text":"#E6EDF3",
         "a1":"#58A6FF","a2":"#F87171","a3":"#3FB950","a4":"#D2A8FF",
         "a5":"#FFA657","slate":"#64748B"}

    def sa(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["slate"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(C["grid"]); sp.set_linewidth(0.6)
        ax.grid(True, color=C["grid"], lw=0.4, alpha=0.7)
        if title:  ax.set_title(title,   color=C["text"],  fontsize=9, fontweight="bold", pad=5)
        if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8)

    fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # (a) Recompensa por episódio
    ax = fig.add_subplot(gs[0, 0:2])
    sa(ax, "(a) Recompensa média por iteração PPO", "Iteração", "Recompensa")
    iters = range(1, len(history["ep_reward_mean"])+1)
    vals  = history["ep_reward_mean"]
    ax.plot(iters, vals, color=C["a3"], lw=1.5, alpha=0.5)
    # Média móvel
    w = min(20, len(vals)//4 or 1)
    ma = np.convolve(vals, np.ones(w)/w, mode="valid")
    ax.plot(range(w, len(vals)+1), ma, color=C["a3"], lw=2.5,
            label=f"Média móvel ({w} iter)")
    ax.axhline(0, color=C["slate"], lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

    # (b) Perdas ator e crítico
    ax = fig.add_subplot(gs[0, 2])
    sa(ax, "(b) Perdas PPO", "Iteração", "")
    ax.plot(history["actor_loss"],  color=C["a1"], lw=1.5, label="L ator")
    ax.plot(history["critic_loss"], color=C["a2"], lw=1.5, label="L crítico")
    ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

    # (c) Entropia da política
    ax = fig.add_subplot(gs[1, 0])
    sa(ax, "(c) Entropia da política", "Iteração", "Entropia")
    ax.plot(history["entropy"], color=C["a4"], lw=1.5)
    ax.text(0.04, 0.94, "Entropia alta → exploração\nEntropia baixa → política determinística",
            transform=ax.transAxes, color=C["a4"], fontsize=7.5,
            va="top", style="italic")

    # (d) Distribuição de recompensas por episódio
    ax = fig.add_subplot(gs[1, 1])
    sa(ax, "(d) Distribuição de recompensas", "Recompensa por episódio", "Frequência")
    ep_r = [r for r in history["ep_reward"] if not np.isnan(r)]
    if ep_r:
        ax.hist(ep_r, bins=40, color=C["a5"], alpha=0.75, edgecolor="none",
                density=True)
        ax.axvline(np.mean(ep_r), color=C["a3"], lw=2,
                   label=f"Média = {np.mean(ep_r):.2f}")
        ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

    # (e) Resultado da avaliação
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(C["panel"]); ax.axis("off")
    ax.set_title("(e) Avaliação final (política determinística)",
                 color=C["text"], fontsize=9, fontweight="bold", pad=5)
    rows = [
        ("Métrica",            "Valor"),
        ("Recomp. média",      f"{eval_result['mean_reward']:.3f}"),
        ("Desvio padrão",      f"{eval_result['std_reward']:.3f}"),
        ("Duração média (ep)", f"{eval_result['mean_len']:.0f} steps"),
        ("Taxa de sobreviv.",  f"{eval_result['survival_rate']*100:.1f}%"),
    ]
    cw = [0.55, 0.45]; rh = 0.16
    for ri, (c1, c2) in enumerate(rows):
        yc = 1.0 - (ri+1)*rh
        bg = C["a1"] if ri == 0 else ("#1A2535" if ri%2==0 else "#131C2A")
        for ci, (cell, w) in enumerate(zip([c1,c2], cw)):
            x = sum(cw[:ci])
            fc = plt.Rectangle((x,yc), w, rh*0.9,
                               transform=ax.transAxes,
                               facecolor=bg, edgecolor=C["grid"], lw=0.4)
            ax.add_patch(fc)
            col = (C["panel"] if ri==0 else
                   C["a3"] if "%" in cell and ri>0 else C["text"])
            ax.text(x+w/2, yc+rh*0.45, cell, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5,
                    color=col, fontweight="bold" if ri==0 else "normal")

    fig.suptitle(
        "PPO — Treinamento da Política de Equilíbrio · Pêndulo Invertido Duplo\n"
        "Ambiente: MLP tanh treinada  ·  Objetivo: θ₁ = θ₂ = 0",
        color=C["text"], fontsize=11, fontweight="bold", y=0.998
    )
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"[OK] Figura: {OUT_FIG}")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():

    # ── Verificar checkpoint MLP ──────────────────────────────────────────
    if not Path(CHECKPOINT_MLP).exists():
        print(f"[ERRO] Modelo MLP não encontrado: {CHECKPOINT_MLP}")
        print("       Altere a variável CHECKPOINT_MLP para o caminho correto.")
        return

    # ── Configuração PPO ──────────────────────────────────────────────────
    CFG = {
        # Arquitetura das redes de política/valor
        "actor_hidden":  (128, 128),
        "critic_hidden": (128, 128),

        # Hiperparâmetros PPO
        "clip_eps":      0.2,       # ε do clipping — padrão industrial
        "entropy_coef":  0.01,      # incentivo à exploração
        "max_grad_norm": 0.5,       # clipping do gradiente

        # Cálculo de vantagens
        "gamma":         0.99,      # desconto temporal
        "lam":           0.95,      # GAE λ

        # Treinamento
        "n_steps":       2048,      # passos por iteração de coleta
        "batch_size":    256,       # tamanho do mini-batch
        "n_epochs":      10,        # épocas de gradiente por iteração
        "lr_actor":      3e-4,
        "lr_critic":     1e-3,

        # Duração
        "total_steps":   300_000,   # aumentar para melhor convergência
    }

    print("=" * 60)
    print("  PPO — Pêndulo Invertido Duplo")
    print("  Objetivo: equilibrar em θ₁ = θ₂ = 0")
    print("=" * 60)

    # ── Criar ambiente e agente ───────────────────────────────────────────
    env   = PendulumEnv(CHECKPOINT_MLP)
    agent = PPO(env, CFG)

    print(f"\n  Ator  : obs_dim={env.obs_dim} → {CFG['actor_hidden']} → act_dim={env.act_dim}")
    print(f"  Crítico: obs_dim={env.obs_dim} → {CFG['critic_hidden']} → 1")
    n_a = sum(p.numel() for p in agent.actor.parameters())
    n_c = sum(p.numel() for p in agent.critic.parameters())
    print(f"  Parâmetros: ator={n_a:,}  crítico={n_c:,}")
    print(f"  Total steps: {CFG['total_steps']:,}")
    print(f"  Iterações  : {CFG['total_steps']//CFG['n_steps']}")

    # ── Treinar ───────────────────────────────────────────────────────────
    history = agent.train(CFG["total_steps"])

    # ── Avaliar ───────────────────────────────────────────────────────────
    print("\n  Avaliando política treinada (50 episódios)...")
    eval_result = agent.evaluate(n_episodes=50, max_steps=500)
    print(f"  Recompensa média : {eval_result['mean_reward']:.3f} ± "
          f"{eval_result['std_reward']:.3f}")
    print(f"  Duração média    : {eval_result['mean_len']:.0f} steps")
    print(f"  Taxa sobrevivência: {eval_result['survival_rate']*100:.1f}%")

    # ── Salvar e plotar ───────────────────────────────────────────────────
    agent.save(OUT_POLICY)
    plot_training(history, eval_result)

    print("\n" + "="*60)
    print("  Concluído.")
    print("="*60)


if __name__ == "__main__":
    main()
