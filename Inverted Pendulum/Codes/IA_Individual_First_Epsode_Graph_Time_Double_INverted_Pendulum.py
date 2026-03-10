import pandas as pd
import matplotlib.pyplot as plt
import os

arquivo = r"data_IA_process/pendulum_ai_dataset.csv"
pasta_saida = "IA_graficos_episodios"

# cria a pasta se não existir
os.makedirs(pasta_saida, exist_ok=True)

df = pd.read_csv(arquivo)

episodes = df["episode"].unique()

for ep in episodes:
    dados = df[df["episode"] == ep]

    # cria grade 2x2 (cada gráfico ocupa 1/4 da imagem)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # canto superior esquerdo
    axs[0, 0].plot(dados["time"], dados["sin_theta1"], color="tab:blue")
    axs[0, 0].set_title("sin(theta1)")
    axs[0, 0].grid(True)

    # canto superior direito
    axs[0, 1].plot(dados["time"], dados["cos_theta1"], color="tab:orange")
    axs[0, 1].set_title("cos(theta1)")
    axs[0, 1].grid(True)

    # canto inferior esquerdo
    axs[1, 0].plot(dados["time"], dados["sin_theta2"], color="tab:green")
    axs[1, 0].set_title("sin(theta2)")
    axs[1, 0].set_xlabel("Tempo")
    axs[1, 0].grid(True)

    # canto inferior direito
    axs[1, 1].plot(dados["time"], dados["cos_theta2"], color="tab:red")
    axs[1, 1].set_title("cos(theta2)")
    axs[1, 1].set_xlabel("Tempo")
    axs[1, 1].grid(True)

    fig.suptitle(f"Episódio {ep} - Representações Angulares", fontsize=14)

    plt.tight_layout()

    # salvar imagem
    nome_arquivo = f"{pasta_saida}/episodio_{ep}_4_graficos.png"
    plt.savefig(nome_arquivo, dpi=300)

    plt.show()
    plt.close()

print("Gráficos organizados em 4 quadrantes salvos com sucesso!")