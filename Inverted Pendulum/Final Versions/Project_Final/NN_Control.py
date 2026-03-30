"""
PPO + Behavior Cloning para controle de pêndulo invertido duplo
- BC usa dados do sistema natural (sem controle externo)
- A rede neural aprende uma política que maximiza a recompensa enquanto permanece próxima das ações observadas no dataset natural
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
import gym
from gym import spaces
import os

# -------------------------------------------------------------------
# 1. Ambiente (substitua pela sua implementação real)
# -------------------------------------------------------------------
class DoublePendulumEnv(gym.Env):
    """
    Ambiente do pêndulo invertido duplo.
    OBS: Substitua esta classe pelo seu ambiente real.
    Deve fornecer:
      - observation_space: Box(6,) -> [sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2]
      - action_space: Box(2,) -> [τ1, τ2]
      - reset() -> estado inicial
      - step(action) -> (next_state, reward, done, info)
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.state = None

    def reset(self):
        # Exemplo: estado próximo ao equilíbrio (θ1=0, θ2=0) com pequena perturbação
        self.state = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        # Aqui você deve implementar a dinâmica real do pêndulo duplo
        dt = 0.02
        # Simulação dummy: aplica ação e retorna recompensa simples
        # Substitua pelas equações de movimento do seu sistema
        self.state += np.random.randn(6) * 0.01  # ruído dummy
        # Recompensa: penalidade por afastamento da vertical (θ1=0, θ2=0)
        theta1 = np.arctan2(self.state[0], self.state[1])
        theta2 = np.arctan2(self.state[2], self.state[3])
        reward = - (theta1**2 + theta2**2) - 0.01 * (action[0]**2 + action[1]**2)
        done = False  # critério de término (ex: queda)
        return self.state.copy(), reward, done, {}


# -------------------------------------------------------------------
# 2. Redes neurais (Ator e Crítico)
# -------------------------------------------------------------------
class Actor(nn.Module):
    """Política estocástica: saída = média (tanh opcional) e log_std"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Parâmetro de log_desvio padrão (independente do estado)
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
# 3. Carregamento do dataset de Behavior Cloning
#    (dados do sistema natural – sem controle)
# -------------------------------------------------------------------
def load_bc_dataset(csv_path, scaler_X=None, scaler_y=None):
    """
    Carrega dataset de BC do arquivo CSV.
    Espera colunas: sin_theta1, cos_theta1, sin_theta2, cos_theta2, omega1, omega2, tau1, tau2
    """
    df = pd.read_csv(csv_path)
    input_cols = ['sin_theta1', 'cos_theta1', 'sin_theta2', 'cos_theta2', 'omega1', 'omega2']
    output_cols = ['tau1', 'tau2']   # ajuste se os nomes forem diferentes

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    if scaler_X is not None:
        X = scaler_X.transform(X)
    if scaler_y is not None:
        y = scaler_y.transform(y)

    return X, y


# -------------------------------------------------------------------
# 4. Funções auxiliares do PPO
# -------------------------------------------------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (GAE)"""
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

    state = env.reset()
    for _ in range(steps):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, std = actor(state_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = critic(state_t).cpu().numpy()[0]

        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

        states.append(state)
        actions.append(action.cpu().numpy()[0])
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        state = next_state
        if done:
            state = env.reset()

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
# 5. Função principal de treinamento (PPO + BC)
# -------------------------------------------------------------------
def train_ppo_with_bc(env, actor, critic, bc_loader, num_iterations=1000,
                      steps_per_iter=2048, epochs=10, batch_size=64,
                      lr_actor=3e-4, lr_critic=3e-4, clip_epsilon=0.2,
                      gamma=0.99, lam=0.95, lambda_bc=0.01, device='cpu'):
    """
    Treina a política usando PPO com regularização de Behavior Cloning.
    bc_loader: DataLoader que fornece (estados, ações) do sistema natural.
    lambda_bc: peso da perda de BC na função objetivo total.
    """
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # Iterador para o dataset de BC (será reiniciado automaticamente)
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
            # Amostrar um batch do dataset de BC (pode ser o mesmo para todos os mini-batches do rollout)
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
# 6. Execução principal
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Criar ambiente (substitua pelo seu)
    env = DoublePendulumEnv()

    # Inicializar redes
    state_dim = 6   # sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2
    action_dim = 2  # τ1, τ2
    actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)
    critic = Critic(state_dim, hidden_dim=256).to(device)

    # Carregar dataset de Behavior Cloning (dados do sistema natural)
    # Ajuste o caminho do arquivo CSV conforme necessário
    bc_csv_path = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"   # <-- substitua pelo seu arquivo
    if os.path.exists(bc_csv_path):
        bc_X, bc_y = load_bc_dataset(bc_csv_path)
        bc_dataset = TensorDataset(torch.tensor(bc_X, dtype=torch.float32),
                                   torch.tensor(bc_y, dtype=torch.float32))
        bc_loader = DataLoader(bc_dataset, batch_size=64, shuffle=True)
        print(f"Dataset BC carregado: {len(bc_X)} amostras")
    else:
        # Se não existir, crie um dataset vazio (apenas para exemplo)
        print("Arquivo de BC não encontrado. Usando dataset vazio (apenas PPO).")
        bc_loader = DataLoader(TensorDataset(torch.zeros(1, state_dim), torch.zeros(1, action_dim)), batch_size=1)

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