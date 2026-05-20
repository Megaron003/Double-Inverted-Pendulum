import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregamento dos Dados
df = pd.read_csv('D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv')

# Extraindo a aceleração (angle_accel1) para os Episódios 0 e 1 
# Selecionamos os primeiros 5000 passos para parear os tamanhos
ep0 = df[df['episode'] == 0]['angle_accel1'].dropna().values[:5000]
ep1 = df[df['episode'] == 1]['angle_accel1'].dropna().values[:5000]

# 2. Cálculo da Diferença Observada (Real)
dif_observada = np.mean(ep0) - np.mean(ep1)

# 3. Configurações da Permutação por Blocos
dados_combinados = np.concatenate([ep0, ep1])
n_total = len(dados_combinados)
tamanho_bloco = 100
n_blocos = n_total // tamanho_bloco
n_permutacoes = 1000

diferencias_permutadas = []
np.random.seed(42) # Garantir reprodutibilidade

# 4. Execução das Permutações
for _ in range(n_permutacoes):
    indices_blocos = np.arange(n_blocos)
    np.random.shuffle(indices_blocos)
    
    # Reconstrói a série baseada nos blocos embaralhados
    dados_permutados = np.concatenate([dados_combinados[i*tamanho_bloco : (i+1)*tamanho_bloco] for i in indices_blocos])
    
    # Divide na metade para os grupos "falsos"
    metade = len(dados_permutados) // 2
    falso_ep0 = dados_permutados[:metade]
    falso_ep1 = dados_permutados[metade:]
    
    # Salva a diferença média dos grupos falsos
    diferencias_permutadas.append(np.mean(falso_ep0) - np.mean(falso_ep1))

# 5. Cálculo do P-valor
diferencias_permutadas = np.array(diferencias_permutadas)
p_valor = np.sum(np.abs(diferencias_permutadas) >= np.abs(dif_observada)) / n_permutacoes

# Impressão de Resultados
print("=== Resultados do Teste de Permutação por Blocos ===")
print(f"Média Ep 0: {np.mean(ep0):.5f} rad/s²")
print(f"Média Ep 1: {np.mean(ep1):.5f} rad/s²")
print(f"Diferença Observada: {dif_observada:.5f} rad/s²")
print(f"P-valor: {p_valor:.4f}")

# ==========================================
# 6. Plotagem do Histograma
# ==========================================
plt.figure(figsize=(10, 6))

# Histograma da distribuição permutada (O "Acaso")
plt.hist(diferencias_permutadas, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Distribuição Nula (Permutações)')

# Linha vertical mostrando a nossa diferença observada (Real)
plt.axvline(dif_observada, color='red', linestyle='dashed', linewidth=2.5, label=f'Diferença Observada ({dif_observada:.4f})')
plt.axvline(-dif_observada, color='red', linestyle='dashed', linewidth=2.5, alpha=0.5) # Espelho para teste bicaudal

# Customização do Gráfico
plt.title('Teste de Permutação por Blocos: Aceleração (Ep 0 vs Ep 1)', fontsize=14, fontweight='bold')
plt.xlabel('Diferença de Médias de Aceleração (rad/s²)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Exibe o gráfico
plt.tight_layout()
plt.show()