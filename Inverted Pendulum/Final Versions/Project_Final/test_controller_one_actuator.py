import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

MODEL_XML = "acrobot.xml"

model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)

# =========================
# ACTOR (MESMA ARQUITETURA)
# =========================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = torch.tanh(self.net(x))
        return mean

actor = Actor()
actor.load_state_dict(torch.load("actor_acrobot.pth"))
actor.eval()

# =========================
# ESTADO (COMPATÍVEL COM GYM)
# =========================
def get_state():
    q1, q2 = data.qpos[:2]
    dq1, dq2 = data.qvel[:2]

    return np.array([
        np.cos(q1),
        np.sin(q1),
        np.cos(q2),
        np.sin(q2),
        dq1,
        dq2
    ], dtype=np.float32)

# =========================
# RESET
# =========================
def reset():
    data.qpos[:] = np.random.uniform(-0.1, 0.1, size=2)
    data.qvel[:] = np.random.uniform(-0.1, 0.1, size=2)
    mujoco.mj_forward(model, data)

reset()

# =========================
# LOOP
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        state = torch.tensor(get_state()).unsqueeze(0)

        with torch.no_grad():
            action = actor(state).item()

        data.ctrl[0] = np.clip(action, -1, 1)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)