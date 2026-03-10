import mujoco
import mujoco.viewer
import numpy as np
import csv
import os
import time

MODEL_PATH = "double_inverted_pendulum.xml"
OUTPUT = "data/pendulum_dataset.csv"

EPISODES = 5
SIM_TIME = 10  # segundos por episódio

# =========================
# Carregar modelo
# =========================
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

os.makedirs("data", exist_ok=True)

# =========================
# Função de randomização
# =========================
def randomize_state():
    # Ângulos iniciais
    data.qpos[0] = np.random.uniform(-0.2, 0.2)
    data.qpos[1] = np.random.uniform(-0.2, 0.2)

    # Velocidades iniciais
    data.qvel[0] = np.random.uniform(-1.0, 1.0)
    data.qvel[1] = np.random.uniform(-1.0, 1.0)

    mujoco.mj_forward(model, data)


# =========================
# Energia do sistema
# =========================
def compute_energy():
    kinetic = np.sum(data.qvel ** 2) * 0.5
    potential = data.energy[0] if hasattr(data, "energy") else 0
    return kinetic, potential


# =========================
# Coleta de dados
# =========================
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "episode",
        "time",
        "theta1",
        "theta2",
        "omega1",
        "omega2",
        "kinetic_energy",
        "potential_energy"
    ])

    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("Iniciando coleta de dados...")

        for ep in range(EPISODES):

            print(f"Episódio {ep+1}/{EPISODES}")

            mujoco.mj_resetData(model, data)
            randomize_state()

            start_time = data.time

            while viewer.is_running() and (data.time - start_time < SIM_TIME):

                step_start = time.time()

                mujoco.mj_step(model, data)

                theta1 = data.qpos[0]
                theta2 = data.qpos[1]

                omega1 = data.qvel[0]
                omega2 = data.qvel[1]

                kinetic, potential = compute_energy()

                writer.writerow([
                    ep,
                    data.time,
                    theta1,
                    theta2,
                    omega1,
                    omega2,
                    kinetic,
                    potential
                ])

                viewer.sync()

                # manter tempo real
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

print("Dataset salvo em:", OUTPUT)