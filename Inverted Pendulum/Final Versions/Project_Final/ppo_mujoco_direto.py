"""
PPO treinado DIRETAMENTE no MuJoCo InvertedDoublePendulum-v4.
Usa obs/ação/reward nativas do ambiente — sem a MLP como intermediária.
Após convergir, salva e renderiza.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, time
from pathlib import Path

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cpu")

OUT_POLICY = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\PPO_Results\\ppo_policy.pt"
OUT_FIG    = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\PPO_Results\\ppo_mujoco_treinamento.png"

# ── Dimensões do InvertedDoublePendulum-v4 ─────────────────────────────
OBS_DIM = 11   # obs nativa do MuJoCo
ACT_DIM = 1    # força no cart ∈ [-1, +1]

class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=(256,256)):
        super().__init__()
        dims = [obs_dim]+list(hidden)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i],dims[i+1]), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # std global
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.01)

    def forward(self, obs):
        feat = self.trunk(obs)
        mu   = torch.tanh(self.mu_head(feat))
        std  = self.log_std.exp().expand_as(mu).clamp(0.05, 1.0)
        return mu, std

    def act(self, obs_np, deterministic=False):
        t = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mu, std = self(t)
        if deterministic:
            return mu.squeeze().numpy()
        return Normal(mu, std).sample().clamp(-1,1).squeeze().numpy()

class Critic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=(256,256)):
        super().__init__()
        dims = [obs_dim]+list(hidden)+[1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i < len(dims)-2: layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, obs): return self.net(obs).squeeze(-1)

def collect(env, actor, critic, n_steps):
    obs_list,act_list,rew_list,done_list,val_list,lp_list = [],[],[],[],[],[]
    obs,_ = env.reset()
    for _ in range(n_steps):
        ot = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mu, std = actor(ot)
            dist = Normal(mu, std)
            a_raw = dist.sample().clamp(-1,1)
            lp = dist.log_prob(a_raw).sum(-1)
            val = critic(ot).item()
        act = a_raw.squeeze().numpy()
        next_obs, rew, term, trunc, _ = env.step(np.atleast_1d(act))
        done = term or trunc
        obs_list.append(obs); act_list.append(act); rew_list.append(rew)
        done_list.append(done); val_list.append(val); lp_list.append(lp.item())
        obs = next_obs if not done else env.reset()[0]
    ot = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad(): last_val = critic(ot).item()
    return obs_list,act_list,rew_list,done_list,val_list,lp_list,last_val

def gae(rewards,dones,values,last_val,gamma=0.99,lam=0.95):
    n = len(rewards)
    adv = np.zeros(n,dtype=np.float32)
    vals = np.array(values+[last_val],dtype=np.float32)
    la = 0.0
    for t in reversed(range(n)):
        m = 1.0-float(dones[t])
        delta = rewards[t]+gamma*vals[t+1]*m-vals[t]
        la = delta+gamma*lam*m*la
        adv[t] = la
    ret = adv+np.array(values,dtype=np.float32)
    adv = (adv-adv.mean())/(adv.std()+1e-8)
    return torch.from_numpy(ret), torch.from_numpy(adv)

def update(actor,critic,opt_a,opt_c,
           obs_b,act_b,ret_b,adv_b,old_lp_b,
           clip=0.2,n_ep=10,bs=256,coef_ent=0.005,max_gn=0.5):
    n = len(obs_b)
    als,cls,ents = [],[],[]
    for _ in range(n_ep):
        idx = torch.randperm(n)
        for s in range(0,n,bs):
            mb = idx[s:s+bs]
            ob = obs_b[mb]; ab = act_b[mb]
            rb = ret_b[mb]; advb = adv_b[mb]; olp = old_lp_b[mb]
            mu,std = actor(ob)
            dist = Normal(mu,std)
            nlp  = dist.log_prob(ab).sum(-1)
            ent  = dist.entropy().sum(-1).mean()
            ratio = (nlp-olp).exp()
            s1 = ratio*advb; s2 = ratio.clamp(1-clip,1+clip)*advb
            al = -torch.min(s1,s2).mean() - coef_ent*ent
            opt_a.zero_grad(); al.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),max_gn)
            opt_a.step()
            vp = critic(ob)
            cl = F.mse_loss(vp,rb)
            opt_c.zero_grad(); cl.backward()
            nn.utils.clip_grad_norm_(critic.parameters(),max_gn)
            opt_c.step()
            als.append(al.item()); cls.append(cl.item()); ents.append(ent.item())
    return np.mean(als),np.mean(cls),np.mean(ents)

def evaluate(env, actor, n_ep=10, max_steps=1000):
    rewards, survived = [], []
    for ep in range(n_ep):
        obs,_ = env.reset(seed=SEED+ep)
        r = 0.0
        for st in range(max_steps):
            act = actor.act(obs, deterministic=True)
            obs,rew,term,trunc,_ = env.step(np.atleast_1d(act))
            r += rew
            if term or trunc: break
        rewards.append(r); survived.append(st+1==max_steps)
    return np.mean(rewards), np.mean(survived)

def train(total_steps=500_000):
    env = gym.make("InvertedDoublePendulum-v4")
    actor  = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    opt_a  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    opt_c  = torch.optim.Adam(critic.parameters(), lr=1e-3)

    N_STEPS  = 2048
    history  = {"reward":[],"al":[],"cl":[],"ent":[]}
    steps    = 0; it = 0; t0 = time.time()

    print(f"\n{'Iter':>5}  {'Steps':>8}  {'Recomp':>10}  {'Surv%':>7}  "
          f"{'L_ator':>9}  {'Entropia':>9}  {'s':>6}")
    print("─"*65)

    while steps < total_steps:
        ob,ac,rw,dn,vl,lp,lv = collect(env,actor,critic,N_STEPS)
        steps += N_STEPS; it += 1

        obs_b = torch.from_numpy(np.array(ob,dtype=np.float32))
        act_b = torch.from_numpy(np.array(ac,dtype=np.float32))
        if act_b.dim()==1: act_b = act_b.unsqueeze(-1)
        old_lp_b = torch.tensor(lp, dtype=torch.float32)
        ret_b, adv_b = gae(rw,dn,vl,lv)

        al,cl,ent = update(actor,critic,opt_a,opt_c,
                           obs_b,act_b,ret_b,adv_b,old_lp_b)

        history["al"].append(al); history["cl"].append(cl)
        history["ent"].append(ent)

        if it % 10 == 0 or it == 1:
            mr, ms = evaluate(env, actor, n_ep=5)
            history["reward"].append(mr)
            print(f"{it:>5}  {steps:>8}  {mr:>10.2f}  {ms*100:>6.0f}%  "
                  f"{al:>9.4f}  {ent:>9.4f}  {time.time()-t0:>5.0f}s")
        else:
            history["reward"].append(float("nan"))

    env.close()

    # Salvar
    torch.save({"actor":actor.state_dict(),"critic":critic.state_dict()},
               OUT_POLICY)
    print(f"[OK] {OUT_POLICY}")

    # Figura
    fig, axes = plt.subplots(1,3,figsize=(18,5),facecolor="#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        for sp in ax.spines.values(): sp.set_color("#21262D")
        ax.tick_params(colors="#64748B")
        ax.grid(True,color="#21262D",lw=0.4)

    vals = [v for v in history["reward"] if not np.isnan(v)]
    iters_r = [i*10 for i in range(1,len(vals)+1)]
    axes[0].plot(iters_r, vals, color="#3FB950", lw=2)
    axes[0].set_title("Recompensa média", color="#E6EDF3", fontsize=9)
    axes[0].set_xlabel("Iteração", color="#64748B", fontsize=8)

    axes[1].plot(history["al"], color="#58A6FF", lw=1.5, label="Ator")
    axes[1].plot(history["cl"], color="#F87171", lw=1.5, label="Crítico")
    axes[1].set_title("Perdas", color="#E6EDF3", fontsize=9)
    axes[1].legend(fontsize=8, facecolor="#161B22", labelcolor="#E6EDF3")

    axes[2].plot(history["ent"], color="#D2A8FF", lw=1.5)
    axes[2].set_title("Entropia", color="#E6EDF3", fontsize=9)

    fig.suptitle("PPO → MuJoCo InvertedDoublePendulum-v4",
                 color="#E6EDF3", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] {OUT_FIG}")

    return actor

if __name__ == "__main__":
    # Treinar com steps reduzidos para teste — aumente para 1_000_000
    actor = train(total_steps=100_000)

    # Avaliação final com renderização
    print("\nAvaliação final com renderização...")
    env_render = gym.make("InvertedDoublePendulum-v4", render_mode="human")
    mr, ms = evaluate(env_render, actor, n_ep=3, max_steps=1000)
    env_render.close()
    print(f"Recompensa: {mr:.1f}  |  Sobrevivência: {ms*100:.0f}%")
