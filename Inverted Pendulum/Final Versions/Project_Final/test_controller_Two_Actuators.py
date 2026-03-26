import torch
import numpy as np
import mujoco
import mujoco.viewer
import time

# Importa a classe Actor do seu código de treinamento (PPO_Code.py)
from PPO_Code import Actor

# -------------------------------------------------------------------
# Configurações
# -------------------------------------------------------------------
MODEL_XML_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum_with_actuators.xml"
ACTOR_PATH = "actor_ppo_bc.pth"  # caminho do modelo treinado

state_dim = 6
action_dim = 2
device = torch.device("cpu")

# -------------------------------------------------------------------
# Carregar modelo MuJoCo
# -------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------------------------
# Carregar política treinada
# -------------------------------------------------------------------
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
# Reset da simulação (posição inicial com pequena perturbação)
# -------------------------------------------------------------------
def reset():
    data.qpos[:] = [0.1, -0.1]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

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

        # 6. (Opcional) Verificar se o pêndulo caiu
        theta1 = data.qpos[0]
        theta2 = data.qpos[1]
        if abs(theta1) > 1.0 or abs(theta2) > 1.0:
            print(f"Queda detectada após {step} passos. Reiniciando...")
            reset()
            step = 0
        else:
            step += 1

        # Pequena pausa para controlar a velocidade da animação
        time.sleep(model.opt.timestep)

    print("Simulação encerrada.")