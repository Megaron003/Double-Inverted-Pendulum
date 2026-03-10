import pandas as pd
import numpy as np
import os

arquivo = r"data/pendulum_dataset.csv"
pasta_saida = r"data_IA_process"
saida = os.path.join(pasta_saida, "pendulum_ai_dataset.csv")

# Cria a pasta se ela não existir
os.makedirs(pasta_saida, exist_ok=True)

df = pd.read_csv(arquivo)

episode = df.iloc[:, 0]
time = df.iloc[:, 1]
theta1 = df.iloc[:, 2]
theta2 = df.iloc[:, 3]

# ===== Converter para radianos (caso esteja em graus) =====
# Se seus dados já estão em radianos, pode comentar isso
# theta1 = np.deg2rad(theta1)
# theta2 = np.deg2rad(theta2)

# ===== Velocidade angular =====
omega1 = np.gradient(theta1, time)
omega2 = np.gradient(theta2, time)

# ===== Representação trigonométrica =====
sin_theta1 = np.sin(theta1)
cos_theta1 = np.cos(theta1)

sin_theta2 = np.sin(theta2)
cos_theta2 = np.cos(theta2)

dataset_ai = pd.DataFrame({
    "episode": episode,
    "time": time,
    "sin_theta1": sin_theta1,
    "cos_theta1": cos_theta1,
    "sin_theta2": sin_theta2,
    "cos_theta2": cos_theta2,
    "omega1": omega1,
    "omega2": omega2
})

dataset_ai.to_csv(saida, index=False)

print("Dataset para IA salvo em:")
print(saida)