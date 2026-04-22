import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Carregar dataset
# =========================

arquivo = r"Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_ZMP.csv"

df = pd.read_csv(arquivo)

# =========================
# Episódios disponíveis
# =========================

episodios = df["episode"].unique()

# =========================
# Loop por episódio
# =========================

for ep in episodios:

    df_ep = df[df["episode"] == ep]

    # remover coluna episode da correlação
    df_corr = df_ep.drop(columns=["episode"])

    # =========================
    # Pearson
    # =========================

    pearson_corr = df_corr.corr(method="pearson", numeric_only=True)

    plt.figure(figsize=(12,8))

    sns.heatmap(
        pearson_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(f"Correlação de Pearson — Episódio {ep}")
    plt.tight_layout()
    plt.show()

    # =========================
    # Spearman
    # =========================

    spearman_corr = df_corr.corr(method="spearman", numeric_only=True)

    plt.figure(figsize=(12,8))

    sns.heatmap(
        spearman_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(f"Correlação de Spearman — Episódio {ep}")
    plt.tight_layout()
    plt.show()