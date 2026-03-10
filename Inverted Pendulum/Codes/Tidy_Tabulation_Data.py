import pandas as pd
import numpy as np
import os

# =========================
# Paths
# =========================
arquivo = r"data/pendulum_dataset.csv"
saida = r"dataset_projeto0/pendulum_dataset_tidy.csv"

os.makedirs("dataset_projeto0", exist_ok=True)

# =========================
# 1 — Data Acquisition
# =========================
df = pd.read_csv(arquivo)

print("Original dataset structure:")
print(df.head())

# =========================
# 2 — Rename columns (important for tidy data)
# =========================
df.columns = [
    "episode",
    "time",
    "theta1_rad",
    "theta2_rad",
    "omega1",
    "omega2",
    "kinetic_energy",
    "potential_energy"
]

# =========================
# 3 — Trigonometric representation
# =========================
df["sin_theta1"] = np.sin(df["theta1_rad"])
df["cos_theta1"] = np.cos(df["theta1_rad"])

df["sin_theta2"] = np.sin(df["theta2_rad"])
df["cos_theta2"] = np.cos(df["theta2_rad"])

# =========================
# 4 — Organize tidy dataset
# =========================
tidy_dataset = df[[
    "episode",
    "time",
    "theta1_rad",
    "theta2_rad",
    "sin_theta1",
    "cos_theta1",
    "sin_theta2",
    "cos_theta2",
    "omega1",
    "omega2",
    "kinetic_energy",
    "potential_energy"
]]

# =========================
# 5 — Save tidy dataset
# =========================
tidy_dataset.to_csv(saida, index=False)

print("\nTidy dataset created successfully:")
print(tidy_dataset.head())

print("\nDataset saved at:", saida)