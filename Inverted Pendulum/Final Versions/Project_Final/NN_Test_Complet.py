import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Carregar os dados
df = pd.read_csv(r'C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_non_ZMP_Velocities.csv')

# Visualizar as primeiras linhas
print(df.head())
print(df.info())

# Selecionar colunas de entrada e saída
input_cols = [
    'sin_theta1',
    'cos_theta1',
    'sin_theta2',
    'cos_theta2',
    'omega1',
    'omega2',
]
output_cols = ['tau1_dynamics', 'tau2_dynamics']

X = df[input_cols].values
y = df[output_cols].values

# Normalizar usando média e desvio padrão dos dados de treino
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Dividir em treino e teste (considerando que os episódios são independentes)
# Podemos usar uma divisão simples, mas, como há mais de um episódio, sugere-se separar por episódio.
# Caso os dados sejam sequenciais pode-se realizar uma divisão aleatória.
episodes = df['episode'].unique()

train_episodes, test_episodes = train_test_split(
    episodes, test_size=0.2, random_state=42
)

train_idx = df['episode'].isin(train_episodes)
test_idx = df['episode'].isin(test_episodes)

X_train = df.loc[train_idx, input_cols].values
y_train = df.loc[train_idx, output_cols].values

X_test = df.loc[test_idx, input_cols].values
y_test = df.loc[test_idx, output_cols].values

# Normalização dos dados de entrada e saída usando os parâmetros calculados apenas no conjunto de treino
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Tensores do PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Dataloaders para treino e teste
batch_size = 64

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Uso de uma rede neural simples do tipo feedforward com duas camadas ocultas
# Como o problema se palta em um problema de regressão, usá-se a função de ativação ReLU nas camadas ocultas e nenhuma função de ativação na camada de saída.
class PendulumController(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2):
        super(PendulumController, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = PendulumController()
print(model)

# Definição da função de perda (MSE) e do otimizador (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_X.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Avaliar no teste
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_test_loss += loss.item() * batch_X.size(0)
    epoch_test_loss /= len(test_loader.dataset)
    test_losses.append(epoch_test_loss)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}')

# Plotar as curvas de perda
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).numpy()
    y_true = y_test

# Desnormalizar as previsões e os valores reais para comparar nas escalas originais
y_pred_orig = scaler_y.inverse_transform(y_pred)
y_true_orig = scaler_y.inverse_transform(y_true)

mse = mean_squared_error(y_true_orig, y_pred_orig)
r2 = r2_score(y_true_orig, y_pred_orig)
print(f'MSE: {mse:.6f}')
print(f'R²: {r2:.4f}')

# Visualizar algumas previsões vs reais
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y_true_orig[:100,0], label='Real tau1')
plt.plot(y_pred_orig[:100,0], label='Predito tau1')
plt.legend()
plt.title('tau1')
plt.subplot(1,2,2)
plt.plot(y_true_orig[:100,1], label='Real tau2')
plt.plot(y_pred_orig[:100,1], label='Predito tau2')
plt.legend()
plt.title('tau2')
plt.show()

# SALVAR MODELO E SCALERS
torch.save(model.state_dict(), "pendulum_model.pth")

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("\nModelo e scalers salvos com sucesso!")