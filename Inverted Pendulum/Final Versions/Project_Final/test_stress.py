import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import random

# -------------------------------------------------------------------
# Configurações
# -------------------------------------------------------------------
MODEL_XML_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum_with_actuators.xml"
ACTOR_PATH = "actor_ppo_bc.pth"          # modelo treinado
state_dim = 6
action_dim = 2
device = torch.device("cpu")

# Intervalos de randomização
THETA_RANGE = (-2.0, 2.0)      # radianos (≈ -115° a 115°)
OMEGA_RANGE = (-3.0, 3.0)      # rad/s

# -------------------------------------------------------------------
# Carregar modelo MuJoCo e política
# -------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# Importar a classe Actor do seu código de treinamento
from PPO_Code import Actor

actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
actor.eval()

# -------------------------------------------------------------------
# Função para extrair observação (sinθ1, cosθ1, sinθ2, cosθ2, ω1, ω2)
# -------------------------------------------------------------------
def get_obs(data):
    q1 = data.qpos[0]
    q2 = data.qpos[1]
    dq1 = data.qvel[0]
    dq2 = data.qvel[1]
    return np.array([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), dq1, dq2], dtype=np.float32)

# -------------------------------------------------------------------
# Randomizar estado inicial (ângulos e velocidades)
# -------------------------------------------------------------------
def randomize_state():
    # Ângulos aleatórios dentro do intervalo definido
    data.qpos[0] = random.uniform(*THETA_RANGE)
    data.qpos[1] = random.uniform(*THETA_RANGE)
    # Velocidades angulares aleatórias
    data.qvel[0] = random.uniform(*OMEGA_RANGE)
    data.qvel[1] = random.uniform(*OMEGA_RANGE)
    # Atualiza as estruturas internas do MuJoCo
    mujoco.mj_forward(model, data)

# -------------------------------------------------------------------
# Reset da simulação (chamado no início e a cada queda)
# -------------------------------------------------------------------
def reset():
    randomize_state()
    print(f"Novo estado: θ1={data.qpos[0]:.2f} rad, θ2={data.qpos[1]:.2f} rad, "
          f"ω1={data.qvel[0]:.2f} rad/s, ω2={data.qvel[1]:.2f} rad/s")

# -------------------------------------------------------------------
# Execução com visualizador
# -------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    reset()
    step = 0
    print("Simulação iniciada. Feche a janela do visualizador para encerrar.")

    while viewer.is_running():
        # 1. Obter observação atual
        obs = get_obs(data)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # 2. Calcular ação (média da política)
        with torch.no_grad():
            mean, _ = actor(obs_t)
            action = mean.cpu().numpy()[0]  # ação determinística

        # 3. Aplicar torque (limitar para segurança)
        data.ctrl[:] = np.clip(action, -10.0, 10.0)

        # 4. Avançar um passo de simulação
        mujoco.mj_step(model, data)

        # 5. Atualizar visualização
        viewer.sync()

        # 6. Verificar queda: ângulo > 1 rad (~57°) indica queda
        theta1 = data.qpos[0]
        theta2 = data.qpos[1]
        if abs(theta1) > 1.0 or abs(theta2) > 1.0:
            print(f"Queda após {step} passos. Reiniciando com novo estado aleatório...")
            reset()
            step = 0
            time.sleep(0.5)   # pausa para não reiniciar instantaneamente
        else:
            step += 1

        # Controla a velocidade da animação (opcional)
        time.sleep(model.opt.timestep)

    print("Simulação encerrada.")