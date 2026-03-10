import pandas as pd
import numpy as np
import os

arquivo = r"data/pendulum_dataset.csv"
saida = r"dataset_projeto0/pendulum_dataset_tabular.csv"

os.makedirs("dataset_projeto0", exist_ok=True)

# ===== Importação (Aquisição de Dados) =====
df = pd.read_csv(arquivo)

print("Estrutura original:")
print(df.head())

# ===== Organização das variáveis =====
dataset = pd.DataFrame()

dataset["episode"] = df.iloc[:, 0]
dataset["time"] = df.iloc[:, 1]

dataset["theta1_rad"] = df.iloc[:, 2]
dataset["theta2_rad"] = df.iloc[:, 3]

# ===== Transformação trigonométrica =====
dataset["sin_theta1"] = np.sin(dataset["theta1_rad"])
dataset["cos_theta1"] = np.cos(dataset["theta1_rad"])

dataset["sin_theta2"] = np.sin(dataset["theta2_rad"])
dataset["cos_theta2"] = np.cos(dataset["theta2_rad"])

# ===== Salvar dataset final =====
dataset.to_csv(saida, index=False)

print("\nDataset final estruturado:")
print(dataset.head())

print("\nDataset salvo em:", saida)