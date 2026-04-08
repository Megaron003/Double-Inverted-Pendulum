import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.optim import Adam

# =========================
# Actor-Critic Networks
# =========================
class Actor(nn.Module):
    def _init_(self, obs_dim, act_dim):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def _init_(self, obs_dim):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# PPO Agent
# =========================
class PPO:
    def _init_(self, env, render=True):
        self.env = env
        self.render = render  # 🔥 controle de renderização

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)

        self.actor_optim = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=3e-4)

        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.vf_coef = 0.5
        self.batch_size = 64
        self.n_steps = 2048
        self.n_epochs = 10

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        mean, std = self.actor(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().numpy(), log_prob.detach()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def rollout(self):
        obs = self.env.reset()[0]

        obs_list, act_list, logp_list = [], [], []
        rew_list, done_list, val_list = [], [], []

        for _ in range(self.n_steps):

            # 🔥 RENDER AQUI
            if self.render:
                self.env.render()

            value = self.critic(torch.tensor(obs, dtype=torch.float32)).item()
            action, logp = self.get_action(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            rew_list.append(reward)
            done_list.append(done)
            val_list.append(value)

            obs = next_obs

            if done:
                obs = self.env.reset()[0]

        advantages = self.compute_gae(rew_list, val_list, done_list)
        returns = np.array(advantages) + np.array(val_list)

        return (
            torch.tensor(obs_list, dtype=torch.float32),
            torch.tensor(act_list, dtype=torch.float32),
            torch.tensor(logp_list, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def update(self, obs, acts, old_logp, returns, advs):
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(self.n_epochs):
            idx = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                b_obs = obs[batch_idx]
                b_acts = acts[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_returns = returns[batch_idx]
                b_advs = advs[batch_idx]

                mean, std = self.actor(b_obs)
                dist = Normal(mean, std)

                logp = dist.log_prob(b_acts).sum(axis=1)
                entropy = dist.entropy().sum(axis=1).mean()

                ratio = torch.exp(logp - b_old_logp)

                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_advs

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = ((self.critic(b_obs) - b_returns) ** 2).mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optim.state_dict(),
            'critic_optimizer': self.critic_optim.state_dict()
        }, path)
        print(f"Modelo salvo em {path}")


    def load(self, path):
        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        self.actor_optim.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer'])

        print(f"Modelo carregado de {path}")

    def train(self, total_steps, save_path=None, save_interval=50000):
        steps = 0

        while steps < total_steps:
            obs, acts, logp, returns, advs = self.rollout()
            self.update(obs, acts, logp, returns, advs)

            steps += self.n_steps
            print(f"Steps: {steps}")

            # 🔥 salvar periodicamente
            if save_path and steps % save_interval == 0:
                self.save(save_path)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    agent = PPO(env, render=True)
    agent.env = env
    agent.render = True
    agent.train(200000)


    obs = env.reset()[0]

    for _ in range(1000):
        action, _ = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs = env.reset()[0]

    env.close()
    agent.train(200000, save_path="ppo_pendulum.pth")

    # salvar final
    agent.save("ppo_pendulum_final.pth")

    print("Treinamento concluído e modelo salvo em: {0}!".format("ppo_pendulum_final.pth"))