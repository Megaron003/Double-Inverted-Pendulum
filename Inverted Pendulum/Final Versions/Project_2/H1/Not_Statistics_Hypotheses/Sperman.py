import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Caminho do arquivo
file_path = r"D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"

# Carregar TODAS as colunas de uma vez
data = np.loadtxt(file_path, delimiter=",", skiprows=1)

# Nomes das variáveis (ajuste conforme seu CSV real)
labels = ['sin_1', 'cos_1', 'sin_2', 'cos_2', 'a_1', 'a_2', 'w_1', 'w_2']

# ==============================
# 🔥 MATRIZ DE SPEARMAN
# ==============================
corr_matrix, p_matrix = stats.spearmanr(data)

# ==============================
# 📊 PLOT HEATMAP
# ==============================
plt.figure(figsize=(8, 6))

im = plt.imshow(corr_matrix)

plt.colorbar(im)

plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

plt.title("Spearman Correlation Matrix")

# Mostrar valores dentro da matriz (opcional, mas MUITO útil)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{corr_matrix[i, j]:.2f}",
                 ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.show()