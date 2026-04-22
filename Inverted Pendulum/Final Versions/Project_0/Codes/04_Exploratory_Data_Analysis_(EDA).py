import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CARREGAR DATASET
# =========================

arquivo = r"Inverted Pendulum/Final Versions/Data Processed/pendulum_dataset_tidy.csv"

df = pd.read_csv(arquivo)
df.columns = df.columns.str.strip()

# =========================
# PARÂMETROS DO EMBEDDING
# =========================

delay = 10
embedding_dim = 3

# =========================
# FUNÇÃO DELAY EMBEDDING
# =========================

def takens_embedding(series, delay, dimension):

    N = len(series)

    M = N - (dimension - 1) * delay

    embedded = np.zeros((M, dimension))

    for i in range(dimension):

        embedded[:, i] = series[i * delay : i * delay + M]

    return embedded

# =========================
# AUTOCORRELAÇÃO
# =========================

def autocorrelation(x, lag):

    return np.corrcoef(x[:-lag], x[lag:])[0,1]

# =========================
# EPISÓDIOS
# =========================

episodios = sorted(df["episode"].unique())

pasta_base = "analise_dinamica_avancada"

os.makedirs(pasta_base, exist_ok=True)

# =========================
# LOOP POR EPISÓDIO
# =========================

for ep in episodios:

    print("\n================================")
    print("Analisando episódio", ep)
    print("================================")

    df_ep = df[df["episode"] == ep]

    pasta_ep = f"{pasta_base}/episodio_{ep}"
    os.makedirs(pasta_ep, exist_ok=True)

    # =========================
    # MATRIZ DE CORRELAÇÃO
    # =========================

    corr = df_ep[
        [
            "sin_theta1",
            "cos_theta1",
            "sin_theta2",
            "cos_theta2",
            "tau1_dynamics",
            "tau2_dynamics"
        ]
    ].corr()

    plt.figure(figsize=(7,6))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(f"Correlação — Episódio {ep}")
    plt.tight_layout()
    plt.savefig(f"{pasta_ep}/correlacao.png")
    plt.close()

    # =========================
    # RETRATO DE FASE
    # =========================

    plt.figure(figsize=(6,6))

    plt.scatter(
        df_ep["sin_theta1"],
        df_ep["cos_theta1"],
        s=2
    )

    plt.xlabel("sin(theta1)")
    plt.ylabel("cos(theta1)")
    plt.title("Retrato de fase θ1")

    plt.savefig(f"{pasta_ep}/fase_theta1.png")
    plt.close()

    # =========================
    # DENSIDADE DA DINÂMICA
    # =========================

    plt.figure(figsize=(6,6))

    plt.hist2d(
        df_ep["sin_theta1"],
        df_ep["sin_theta2"],
        bins=100
    )

    plt.colorbar()

    plt.xlabel("sin(theta1)")
    plt.ylabel("sin(theta2)")
    plt.title("Densidade da dinâmica")

    plt.savefig(f"{pasta_ep}/densidade_dinamica.png")
    plt.close()

    # =========================
    # AUTOCORRELAÇÃO TORQUE
    # =========================

    tau1 = df_ep["tau1_dynamics"].values

    lags = 200
    ac = [autocorrelation(tau1, i) for i in range(1, lags)]

    plt.figure(figsize=(8,4))
    plt.plot(ac)
    plt.title("Autocorrelação τ1")
    plt.xlabel("lag")
    plt.ylabel("correlação")

    plt.savefig(f"{pasta_ep}/autocorrelacao_tau1.png")
    plt.close()

    # =========================
    # CORRELAÇÃO CRUZADA
    # =========================

    tau2 = df_ep["tau2_dynamics"].values

    cross = np.correlate(
        tau1 - np.mean(tau1),
        tau2 - np.mean(tau2),
        mode="full"
    )

    plt.figure(figsize=(8,4))
    plt.plot(cross)
    plt.title("Correlação cruzada τ1 vs τ2")

    plt.savefig(f"{pasta_ep}/correlacao_cruzada.png")
    plt.close()

    # =========================
    # DELAY EMBEDDING (ATRATOR)
    # =========================

    series = df_ep["tau1_dynamics"].values

    emb = takens_embedding(series, delay, embedding_dim)

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        emb[:,0],
        emb[:,1],
        emb[:,2],
        s=1
    )

    ax.set_title("Atrator reconstruído (Takens)")

    plt.savefig(f"{pasta_ep}/atrator_takens.png")

    plt.close()

    # =========================
    # MAPA τ(t) → τ(t+1)
    # =========================

    plt.figure(figsize=(6,6))

    plt.scatter(
        tau1[:-1],
        tau1[1:],
        s=2
    )

    plt.xlabel("τ(t)")
    plt.ylabel("τ(t+1)")
    plt.title("Mapa dinâmico do torque")

    plt.savefig(f"{pasta_ep}/mapa_dinamico_tau.png")

    plt.close()

print("\nAnálise dinâmica avançada finalizada.")