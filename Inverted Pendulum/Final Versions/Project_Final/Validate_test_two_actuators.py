import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import random
import msvcrt
import matplotlib.pyplot as plt
import os
import csv

# ========== CONFIGURAÇÕES ==========
MODEL_XML_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum_with_actuators.xml"
ACTOR_PATH = "actor_ppo_bc.pth"
state_dim = 6
action_dim = 2
device = torch.device("cpu")

THETA_RANGE = (-1, 1)        # rad
OMEGA_RANGE = (-1.0, 1.0)    # rad/s
FALL_ANGLE = 1.0             # rad (critério de queda)

# Limiares para análise
STABLE_ANGLE_THRESH = 0.1    # rad (considera estabilizado quando abaixo disso)
MIN_STABLE_DURATION = 1.0    # segundos de permanência abaixo do limiar para considerar estabilização

# Diretório para salvar os dados
SAVE_DIR = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Project_Final\Validate"
# ====================================

# Criar diretório se não existir
os.makedirs(SAVE_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

from PPO_Code import Actor
actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
actor.eval()

# Estruturas para armazenar dados
time_data = []
theta1_data = []
theta2_data = []
omega1_data = []
omega2_data = []
tau1_data = []
tau2_data = []
bias1_data = []
bias2_data = []
energy_data = []

def get_obs(data):
    q1 = data.qpos[0]; q2 = data.qpos[1]
    dq1 = data.qvel[0]; dq2 = data.qvel[1]
    return np.array([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), dq1, dq2], dtype=np.float32)

def randomize_state():
    data.qpos[0] = random.uniform(*THETA_RANGE)
    data.qpos[1] = random.uniform(*THETA_RANGE)
    data.qvel[0] = random.uniform(*OMEGA_RANGE)
    data.qvel[1] = random.uniform(*OMEGA_RANGE)
    mujoco.mj_forward(model, data)
    print(f"Randomizado: θ1={data.qpos[0]:.2f}, θ2={data.qpos[1]:.2f}, ω1={data.qvel[0]:.2f}, ω2={data.qvel[1]:.2f}")

def reset():
    randomize_state()
    # Limpa os buffers
    time_data.clear()
    theta1_data.clear()
    theta2_data.clear()
    omega1_data.clear()
    omega2_data.clear()
    tau1_data.clear()
    tau2_data.clear()
    bias1_data.clear()
    bias2_data.clear()
    energy_data.clear()
    print("Simulação reiniciada e buffers limpos.")

def key_pressed():
    if msvcrt.kbhit():
        return msvcrt.getch() in (b'r', b'R')
    return False

# ---------- Início da simulação ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    reset()
    step = 0
    start_time = time.time()
    print("Simulação iniciada. Pressione 'r' para reiniciar manualmente.")
    print("Feche a janela do visualizador para encerrar e gerar análise.")

    while viewer.is_running():
        if key_pressed():
            reset()
            step = 0
            start_time = time.time()
            time.sleep(0.2)

        # Coleta dados a cada passo
        t = time.time() - start_time
        time_data.append(t)
        theta1_data.append(data.qpos[0])
        theta2_data.append(data.qpos[1])
        omega1_data.append(data.qvel[0])
        omega2_data.append(data.qvel[1])
        bias1_data.append(data.qfrc_bias[0])
        bias2_data.append(data.qfrc_bias[1])
        if hasattr(data, 'energy') and len(data.energy) > 0:
            energy_data.append(data.energy[0])

        obs = get_obs(data)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = actor(obs_t)
            action = mean.cpu().numpy()[0]

        tau1_data.append(action[0])
        tau2_data.append(action[1])

        data.ctrl[:] = np.clip(action, -10.0, 10.0)
        mujoco.mj_step(model, data)
        viewer.sync()

        theta1, theta2 = data.qpos[0], data.qpos[1]
        if abs(theta1) > FALL_ANGLE or abs(theta2) > FALL_ANGLE:
            print(f"Queda após {step} passos. Reiniciando...")
            reset()
            step = 0
            start_time = time.time()
            time.sleep(0.5)
        else:
            step += 1

        time.sleep(model.opt.timestep)

    print("Simulação encerrada. Gerando análise e salvando dados...")

# ---------- Salvar dados em CSV ----------
def save_csv(filename, headers, data_columns):
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*data_columns):
            writer.writerow(row)
    print(f"Dados salvos em: {filepath}")

if len(time_data) > 0:
    save_csv("simulation_data.csv",
             ["time", "theta1", "theta2", "omega1", "omega2", "tau1", "tau2", "bias1", "bias2"],
             [time_data, theta1_data, theta2_data, omega1_data, omega2_data, tau1_data, tau2_data, bias1_data, bias2_data])
    if energy_data:
        save_csv("energy_data.csv", ["time", "energy"], [time_data, energy_data])
else:
    print("Nenhum dado coletado.")

# ---------- Análise após a simulação ----------
if len(time_data) == 0:
    print("Nenhum dado coletado.")
    exit()

time_arr = np.array(time_data)
theta1_arr = np.array(theta1_data)
theta2_arr = np.array(theta2_data)
omega1_arr = np.array(omega1_data)
omega2_arr = np.array(omega2_data)
tau1_arr = np.array(tau1_data)
tau2_arr = np.array(tau2_data)
bias1_arr = np.array(bias1_data)
bias2_arr = np.array(bias2_data)

# 1. Estatísticas dos torques
tau1_max = np.max(np.abs(tau1_arr))
tau2_max = np.max(np.abs(tau2_arr))
tau1_sat = np.mean(np.abs(tau1_arr) >= 9.9) * 100
tau2_sat = np.mean(np.abs(tau2_arr) >= 9.9) * 100
print("\n=== ANÁLISE DE CONTROLE ===")
print(f"Torque máximo τ1: {tau1_max:.2f} Nm (limite: 10 Nm) - Saturação: {tau1_sat:.1f}%")
print(f"Torque máximo τ2: {tau2_max:.2f} Nm (limite: 10 Nm) - Saturação: {tau2_sat:.1f}%")

# 2. Tempo de estabilização
stable_mask = (np.abs(theta1_arr) < STABLE_ANGLE_THRESH) & (np.abs(theta2_arr) < STABLE_ANGLE_THRESH)
stable_time = None
if np.any(stable_mask):
    stable_start_idx = None
    stable_duration = 0.0
    for i in range(len(time_arr)):
        if stable_mask[i]:
            if stable_start_idx is None:
                stable_start_idx = i
            stable_duration = time_arr[i] - time_arr[stable_start_idx]
            if stable_duration >= MIN_STABLE_DURATION:
                stable_time = time_arr[stable_start_idx]
                break
        else:
            stable_start_idx = None
if stable_time is not None:
    print(f"Tempo de estabilização (|θ|<{STABLE_ANGLE_THRESH} rad por {MIN_STABLE_DURATION}s): {stable_time:.2f} s")
else:
    print(f"Não estabilizou dentro do critério (|θ|<{STABLE_ANGLE_THRESH} rad por {MIN_STABLE_DURATION}s).")

# 3. Overshoot
theta1_abs = np.abs(theta1_arr)
theta2_abs = np.abs(theta2_arr)
zero_cross1 = None
for i in range(1, len(theta1_arr)):
    if theta1_arr[i-1] * theta1_arr[i] < 0:
        zero_cross1 = i
        break
if zero_cross1 is not None and zero_cross1 < len(theta1_arr)-1:
    overshoot1 = np.max(theta1_abs[zero_cross1:])
else:
    overshoot1 = np.max(theta1_abs)
zero_cross2 = None
for i in range(1, len(theta2_arr)):
    if theta2_arr[i-1] * theta2_arr[i] < 0:
        zero_cross2 = i
        break
if zero_cross2 is not None and zero_cross2 < len(theta2_arr)-1:
    overshoot2 = np.max(theta2_abs[zero_cross2:])
else:
    overshoot2 = np.max(theta2_abs)

print(f"Overshoot máximo (θ1): {overshoot1:.3f} rad")
print(f"Overshoot máximo (θ2): {overshoot2:.3f} rad")

# 4. Torque extra RMS
extra1 = tau1_arr - bias1_arr
extra2 = tau2_arr - bias2_arr
extra1_rms = np.sqrt(np.mean(extra1**2))
extra2_rms = np.sqrt(np.mean(extra2**2))
print(f"\nRMS do torque extra (além do necessário para equilíbrio):")
print(f"τ1_extra RMS: {extra1_rms:.2f} Nm")
print(f"τ2_extra RMS: {extra2_rms:.2f} Nm")

# 5. Energia
if len(energy_data) > 0:
    energy_arr = np.array(energy_data)
    energy_dissipated = energy_arr[0] - energy_arr[-1]
    print(f"\nEnergia inicial: {energy_arr[0]:.2f} J")
    print(f"Energia final: {energy_arr[-1]:.2f} J")
    print(f"Energia dissipada: {energy_dissipated:.2f} J")
else:
    print("\nDados de energia não disponíveis.")

# 6. Plots
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(time_arr, theta1_arr, label='θ1 (rad)')
plt.plot(time_arr, theta2_arr, label='θ2 (rad)')
plt.axhline(y=STABLE_ANGLE_THRESH, color='gray', linestyle='--', label='Limiar estabilidade')
plt.axhline(y=-STABLE_ANGLE_THRESH, color='gray', linestyle='--')
plt.ylabel('Ângulo (rad)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_arr, tau1_arr, label='τ1 aplicado')
plt.plot(time_arr, tau2_arr, label='τ2 aplicado')
plt.plot(time_arr, bias1_arr, '--', alpha=0.6, label='τ1_equilíbrio')
plt.plot(time_arr, bias2_arr, '--', alpha=0.6, label='τ2_equilíbrio')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_arr, extra1, label='τ1 extra')
plt.plot(time_arr, extra2, label='τ2 extra')
plt.ylabel('Torque extra (Nm)')
plt.xlabel('Tempo (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()