import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 10

# =========================
# Rede Neural
# =========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def act(self, state):
        x = self.actor(state)
        mu = torch.tanh(self.mu(x))

        std = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0)
        dist = Normal(mu, std)

        action = dist.sample()
        action = torch.clamp(action, -1, 1)

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, state, action):
        x = self.actor(state)
        mu = torch.tanh(self.mu(x))

        std = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0)
        dist = Normal(mu, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(state)

        return log_prob, entropy, value

# =========================
# Buffer
# =========================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()

# =========================
# PPO
# =========================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)

        return action.detach().cpu().numpy()

    def update(self):
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        logprobs = torch.stack(self.buffer.logprobs)

        rewards = []
        discounted = 0

        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted = 0
            discounted = reward + GAMMA * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(K_EPOCHS):
            new_logprobs, entropy, state_values = self.policy.evaluate(states, actions)

            ratios = torch.exp(new_logprobs - logprobs.detach())

            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) \
                   + 0.5 * (state_values.squeeze() - rewards)**2 \
                   - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# =========================
# TREINO (SEM RENDER)
# =========================
env = gym.make("InvertedDoublePendulum-v4")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPO(state_dim, action_dim)

for ep in range(1000):
    state, _ = env.reset()
    total_reward = 0

    for t in range(1000):
        action = agent.select_action(state)
        state, reward, done, truncated, _ = env.step(action)

        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(done or truncated)

        total_reward += reward

        if done or truncated:
            break

    agent.update()

    print(f"Episode {ep} | Reward: {total_reward:.2f}")

env.close()

# =========================
# TESTE COM RENDER
# =========================
env = gym.make("InvertedDoublePendulum-v4", render_mode="human")

for ep in range(3):
    state, _ = env.reset()

    action = env.action_space.sample()

    for t in range(1000):
        state_tensor = torch.from_numpy(state).float()

        if t % 3 == 0:  # 🔥 MUITO IMPORTANTE
            with torch.inference_mode():
                action, _ = agent.policy_old.act(state_tensor)

        state, _, done, truncated, _ = env.step(action.numpy())

        if done or truncated:
            print(f"Episode terminou em t={t}")
            break

input("Pressione ENTER para fechar...")
env.close()