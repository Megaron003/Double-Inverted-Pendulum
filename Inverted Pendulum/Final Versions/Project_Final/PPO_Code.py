"""
PPO + Behavior Cloning para controle de pêndulo invertido duplo simulado no MuJoCo.
- O ambiente é um pêndulo duplo fixo na base, com dois atuadores.
- O dataset de BC contém estados e ações (tau1, tau2) do sistema natural (sem controle).
- Treinamento combinado: PPO (recompensa) + BC (regularização).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData

# Tentar importar o visualizador (pode não estar disponível em alguns ambientes)
try:
    from mujoco.viewer import MjViewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False
    MjViewer = None

# -------------------------------------------------------------------
# 1. Caminho do modelo MuJoCo (arquivo XML com dois atuadores)
# -------------------------------------------------------------------
MODEL_XML_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum_with_actuators.xml"

if os.path.exists(MODEL_XML_PATH):
    file_size = os.path.getsize(MODEL_XML_PATH)
    print(f"Arquivo XML encontrado, tamanho: {file_size} bytes")
    if file_size == 0:
        raise ValueError(f"Arquivo XML está vazio: {MODEL_XML_PATH}")
else:
    raise FileNotFoundError(f"Arquivo XML não encontrado: {MODEL_XML_PATH}")

# -------------------------------------------------------------------
# 2. Ambiente MuJoCo com interface Gymnasium
# -------------------------------------------------------------------
class DoublePendulumMuJoCoEnv(gym.Env):
    """Ambiente do pêndulo duplo fixo na base simulado com MuJoCo."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        # Carrega o modelo do arquivo XML
        if not os.path.exists(MODEL_XML_PATH):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_XML_PATH}")
        self.model = MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = MjData(self.model)

        # Verifica se existem atuadores
        nu = self.model.nu
        if nu == 0:
            raise ValueError("O modelo MuJoCo não possui atuadores. Verifique o arquivo XML.")
        elif nu != 2:
            print(f"Aviso: O modelo possui {nu} atuadores, mas o ambiente espera 2. Usando os primeiros 2.")
        
        # Obtém os limites dos atuadores (ctrlrange) se definidos, senão usa valores padrão
        if hasattr(self.model, 'actuator_ctrlrange') and self.model.actuator_ctrlrange is not None:
            low = self.model.actuator_ctrlrange[:, 0]
            high = self.model.actuator_ctrlrange[:, 1]
            if len(low) >= 2:
                low = low[:2]
                high = high[:2]
            else:
                low = np.array([-10.0, -10.0])
                high = np.array([10.0, 10.0])
        else:
            low = np.array([-10.0, -10.0])
            high = np.array([10.0, 10.0])
        
        # Espaços
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        
        print(f"Ambiente inicializado com {nu} atuador(es).")
        print(f"Limites de ação: low={low}, high={high}")

    def _get_obs(self):
        """Extrai observação: sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2"""
        q1 = self.data.qpos[0]   # θ1
        q2 = self.data.qpos[1]   # θ2
        dq1 = self.data.qvel[0]  # ω1
        dq2 = self.data.qvel[1]  # ω2
        return np.array([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), dq1, dq2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reinicia o estado (posição aleatória próxima ao equilíbrio)
        self.data.qpos[:] = [0.1, -0.1]   # pequena perturbação inicial
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Aplica torques
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action[:self.model.nu]  # aplica aos atuadores existentes
        mujoco.mj_step(self.model, self.data)

        # Recompensa: penaliza desvio da vertical e esforço de controle
        theta1 = self.data.qpos[0]
        theta2 = self.data.qpos[1]
        reward = - (theta1**2 + theta2**2) - 0.01 * (action[0]**2 + action[1]**2)

        # Critério de término (queda): ângulo > 60° (≈1 rad)
        terminated = bool(abs(theta1) > 1.0 or abs(theta2) > 1.0)
        truncated = False   # sem limite de tempo (ou podemos colocar um número máximo de passos)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human" and VIEWER_AVAILABLE:
            if self.viewer is None:
                self.viewer = MjViewer(self.model, self.data)
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            # Retornar imagem (implementação simplificada)
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# -------------------------------------------------------------------
# 3. Redes neurais (Ator e Crítico)
# -------------------------------------------------------------------
class Actor(nn.Module):
    """Política estocástica: saída = média e log_std"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

class Critic(nn.Module):
    """Valor do estado (V(s))"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------------------------------------------------
# 4. Carregamento do dataset de Behavior Cloning
# -------------------------------------------------------------------
def load_bc_dataset(csv_path, scaler_X=None, scaler_y=None):
    """Carrega dataset de BC do arquivo CSV.
    Espera colunas: sin_theta1, cos_theta1, sin_theta2, cos_theta2, omega1, omega2, tau1_dynamics, tau2_dynamics
    """
    df = pd.read_csv(csv_path)
    input_cols = ['sin_theta1', 'cos_theta1', 'sin_theta2', 'cos_theta2', 'omega1', 'omega2']
    output_cols = ['tau1_dynamics', 'tau2_dynamics']   # dois torques

    # Verifica se todas as colunas existem
    missing_cols = [c for c in input_cols + output_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    if scaler_X is not None:
        X = scaler_X.transform(X)
    if scaler_y is not None:
        y = scaler_y.transform(y)

    return X, y

# -------------------------------------------------------------------
# 5. Funções auxiliares do PPO (GAE, coleta de trajetórias)
# -------------------------------------------------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def collect_rollouts(env, actor, critic, steps, device):
    """
    Coleta um rollout de 'steps' passos usando a política atual.
    Retorna dicionário com arrays numpy.
    """
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    state, _ = env.reset()   # Gymnasium: retorna (obs, info)
    for _ in range(steps):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, std = actor(state_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = critic(state_t).cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        done = terminated or truncated

        states.append(state)
        actions.append(action.cpu().numpy()[0])
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        state = next_state
        if done:
            state, _ = env.reset()

    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'log_probs': np.array(log_probs, dtype=np.float32),
        'advantages': np.array(advantages, dtype=np.float32),
        'returns': np.array(returns, dtype=np.float32)
    }

# -------------------------------------------------------------------
# 6. Função principal de treinamento (PPO + BC)
# -------------------------------------------------------------------
def train_ppo_with_bc(env, actor, critic, bc_loader, num_iterations=500,
                      steps_per_iter=2048, epochs=10, batch_size=64,
                      lr_actor=3e-4, lr_critic=3e-4, clip_epsilon=0.2,
                      lambda_bc=0.01, device='cpu'):
    """
    Treina a política usando PPO com regularização de Behavior Cloning.
    bc_loader: DataLoader que fornece (estados, ações) do sistema natural.
    lambda_bc: peso da perda de BC na função objetivo total.
    """
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    bc_iter = iter(bc_loader)

    for iteration in range(num_iterations):
        # 1. Coletar dados com a política atual
        rollout = collect_rollouts(env, actor, critic, steps_per_iter, device)

        # 2. Converter para tensores
        states = torch.tensor(rollout['states'], dtype=torch.float32, device=device)
        actions = torch.tensor(rollout['actions'], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(rollout['log_probs'], dtype=torch.float32, device=device)
        advantages = torch.tensor(rollout['advantages'], dtype=torch.float32, device=device)
        returns = torch.tensor(rollout['returns'], dtype=torch.float32, device=device)

        # 3. Dataset e DataLoader para os rollouts
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 4. Múltiplas épocas de atualização
        for epoch in range(epochs):
            # Amostrar um batch do dataset de BC
            try:
                bc_batch = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                bc_batch = next(bc_iter)
            bc_states, bc_actions = bc_batch
            bc_states = bc_states.to(device)
            bc_actions = bc_actions.to(device)

            # Atualizar o crítico (usando mini-batches do rollout)
            for batch in loader:
                s, a, old_lp, adv, ret = batch
                s, a, old_lp, adv, ret = s.to(device), a.to(device), old_lp.to(device), adv.to(device), ret.to(device)
                value = critic(s)
                critic_loss = nn.MSELoss()(value, ret)

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

            # Atualizar o ator (PPO + BC)
            # Calcula PPO loss sobre todos os mini-batches do rollout
            ppo_loss_total = 0.0
            num_batches = 0
            for batch in loader:
                s, a, old_lp, adv, _ = batch
                s, a, old_lp, adv = s.to(device), a.to(device), old_lp.to(device), adv.to(device)
                mean, std = actor(s)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(a).sum(dim=-1)
                ratio = torch.exp(new_log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * adv
                ppo_loss = -torch.min(surr1, surr2).mean()
                ppo_loss_total += ppo_loss
                num_batches += 1
            ppo_loss_avg = ppo_loss_total / num_batches

            # Perda de BC: MSE entre a média da política e a ação do dataset natural
            mean_bc, _ = actor(bc_states)
            bc_loss = nn.MSELoss()(mean_bc, bc_actions)

            # Perda total do ator
            actor_loss = ppo_loss_avg + lambda_bc * bc_loss

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

        # Log a cada iteração
        if (iteration+1) % 50 == 0:
            print(f"Iteração {iteration+1}/{num_iterations} | PPO loss: {ppo_loss_avg.item():.4f} | BC loss: {bc_loss.item():.4f}")

    print("Treinamento concluído.")

# -------------------------------------------------------------------
# 7. Execução principal
# -------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Criar ambiente MuJoCo (usando o arquivo XML com dois atuadores)
    env = DoublePendulumMuJoCoEnv(render_mode=None)  # Para visualizar, altere para 'human'

    # Inicializar redes
    state_dim = 6
    action_dim = 2   # dois torques
    actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)
    critic = Critic(state_dim, hidden_dim=256).to(device)

    # Caminho do dataset de BC
    bc_csv_path = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"

    # Verificar se o arquivo existe
    if not os.path.exists(bc_csv_path):
        raise FileNotFoundError(f"Arquivo de dataset não encontrado: {bc_csv_path}")

    # Carregar dataset de BC
    bc_X, bc_y = load_bc_dataset(bc_csv_path)
    bc_dataset = TensorDataset(torch.tensor(bc_X, dtype=torch.float32),
                               torch.tensor(bc_y, dtype=torch.float32))
    bc_loader = DataLoader(bc_dataset, batch_size=64, shuffle=True)
    print(f"Dataset BC carregado: {len(bc_X)} amostras")

    # Parâmetros de treinamento
    train_ppo_with_bc(
        env=env,
        actor=actor,
        critic=critic,
        bc_loader=bc_loader,
        num_iterations=500,
        steps_per_iter=2048,
        epochs=10,
        batch_size=64,
        lr_actor=3e-4,
        lr_critic=3e-4,
        clip_epsilon=0.2,
        lambda_bc=0.01,
        device=device
    )

    # Salvar o modelo treinado
    torch.save(actor.state_dict(), "actor_ppo_bc.pth")
    torch.save(critic.state_dict(), "critic_ppo_bc.pth")
    print("Modelos salvos: actor_ppo_bc.pth e critic_ppo_bc.pth")