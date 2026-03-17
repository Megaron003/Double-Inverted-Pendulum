import pandas as pd
import numpy as np
import os

arquivo = r"Inverted Pendulum/Final Versions/Data/pendulum_dataset_ZMP.csv"
saida = r"Inverted Pendulum/Final Versions/Data Processed/pendulum_dataset_tidy_ZMP_Velocities.csv"

os.makedirs("Data Processed", exist_ok=True)

# =========================
# Importação
# =========================
df = pd.read_csv(arquivo)

print("Estrutura original:")
print(df.head())

# =========================
# Construção do dataset tidy
# =========================
tidy = pd.DataFrame()

tidy["episode"] = df["episode"]
tidy["time"] = df["time"]

# representação trigonométrica
tidy["sin_theta1"] = np.sin(df["theta1"])
tidy["cos_theta1"] = np.cos(df["theta1"])

tidy["sin_theta2"] = np.sin(df["theta2"])
tidy["cos_theta2"] = np.cos(df["theta2"])

# velocidades angulares (já existentes)
tidy["omega1"] = df["omega1"]
tidy["omega2"] = df["omega2"]

# torques
tidy["tau1_dynamics"] = df["tau1_dynamics"]
tidy["tau2_dynamics"] = df["tau2_dynamics"]

# salvar dataset
tidy.to_csv(saida, index=False)

print("\nDataset tidy criado:")
print(tidy.head())

print("\nDataset salvo em:", saida)