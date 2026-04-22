import pandas as pd
import numpy as np

# =========================
# Caminhos
# =========================
arquivo = r"Inverted Pendulum/Final Versions/Data Processed/pendulum_dataset_tidy_non_ZMP_Velocities.csv"
saida = r"Inverted Pendulum/Final Versions/Data Processed/pendulum_dataset_tidy_with_acceleration.csv"

# =========================
# Leitura
# =========================
df = pd.read_csv(arquivo)

print("Dataset original:")
print(df.head())

# =========================
# Remover coluna de energia cinética (se existir)
# =========================
if "kinetic_energy" in df.columns:
    df = df.drop(columns=["kinetic_energy"])

# =========================
# Ordenar (IMPORTANTE)
# =========================
df = df.sort_values(by=["episode", "time"])

# =========================
# Calcular acelerações angulares
# =========================
df["angle_accel1"] = df.groupby("episode")["omega1"].diff() / df.groupby("episode")["time"].diff()
df["angle_accel2"] = df.groupby("episode")["omega2"].diff() / df.groupby("episode")["time"].diff()

# =========================
# Remover NaNs (primeira linha de cada episódio)
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# Salvar
# =========================
df.to_csv(saida, index=False)

print("\nDataset com aceleração:")
print(df.head())

print("\nArquivo salvo em:", saida)