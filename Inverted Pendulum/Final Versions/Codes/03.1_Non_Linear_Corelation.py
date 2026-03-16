import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CAMINHO DO DATASET
# =========================

arquivo = r"Inverted Pendulum/Final Versions/Data Processed/pendulum_dataset_tidy_ZMP.csv"

df = pd.read_csv(arquivo)

df.columns = df.columns.str.strip()

# =========================
# CRIAR TORQUE FUTURO
# =========================

df["tau2_next"] = df["tau2_dynamics"].shift(-1)

df = df.dropna()

# =========================
# ENTRADAS DA REDE
# =========================

X = df[
    [
        "sin_theta1",
        "cos_theta1",
        "sin_theta2",
        "cos_theta2",
        "tau1_dynamics"
    ]
]

y = df["tau2_next"]

dataset_nn = pd.concat([X, y], axis=1)

# =========================
# SALVAR DATASET
# =========================

dataset_nn.to_csv("dataset_rede_neural.csv", index=False)

print("\nDataset para rede neural criado!\n")

# =========================
# ESTATÍSTICAS
# =========================

print("Estatísticas descritivas:\n")
print(dataset_nn.describe())

# =========================
# MATRIZ DE CORRELAÇÃO
# =========================

corr = dataset_nn.corr()

print("\nMatriz de correlação:\n")
print(corr)

# =========================
# HEATMAP DE CORRELAÇÃO
# =========================

plt.figure(figsize=(8,6))

plt.imshow(corr, interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Mapa de Correlação")

plt.tight_layout()

plt.show()

# =========================
# HISTOGRAMAS
# =========================

dataset_nn.hist(
    figsize=(10,8),
    bins=40
)

plt.suptitle("Distribuição das Variáveis")

plt.show()

# =========================
# RELAÇÃO ENTRE TORQUES
# =========================

plt.figure(figsize=(6,6))

plt.scatter(
    dataset_nn["tau1_dynamics"],
    dataset_nn["tau2_next"],
    s=5,
    alpha=0.5
)

plt.xlabel("tau1")
plt.ylabel("tau2_next")

plt.title("Influência de tau1 no tau2")

plt.show()

# =========================
# RELAÇÃO ÂNGULO → TORQUE
# =========================

plt.figure(figsize=(6,6))

plt.scatter(
    dataset_nn["sin_theta1"],
    dataset_nn["tau2_next"],
    s=5,
    alpha=0.5
)

plt.xlabel("sin(theta1)")
plt.ylabel("tau2_next")

plt.title("Influência do ângulo no torque")

plt.show()

# =========================
# SÉRIE TEMPORAL
# =========================

plt.figure(figsize=(12,6))

plt.plot(df["time"], df["tau1_dynamics"], label="tau1")
plt.plot(df["time"], df["tau2_dynamics"], label="tau2")

plt.legend()

plt.xlabel("Tempo")
plt.ylabel("Torque")

plt.title("Evolução temporal dos torques")

plt.show()