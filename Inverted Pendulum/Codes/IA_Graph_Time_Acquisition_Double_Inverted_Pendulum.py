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

    plt.figure(figsize=(10,6))

    plt.plot(dados["time"], dados["sin_theta1"], label="sin(theta1)")
    plt.plot(dados["time"], dados["cos_theta1"], label="cos(theta1)")
    plt.plot(dados["time"], dados["sin_theta2"], label="sin(theta2)")
    plt.plot(dados["time"], dados["cos_theta2"], label="cos(theta2)")

    plt.title(f"Episódio {ep} - Representação Angular")
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    # salvar imagem
    nome_arquivo = f"{pasta_saida}/episodio_{ep}.png"
    plt.savefig(nome_arquivo, dpi=300)

    # mostrar (opcional)
    plt.show()

    # fechar figura para evitar sobrecarga
    plt.close()

print("Gráficos salvos com sucesso!")