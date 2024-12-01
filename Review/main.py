import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Carregando o conjunto de dados de habitação da Califórnia
california = fetch_california_housing()

# Criando um DataFrame a partir dos dados
df = pd.DataFrame(california.data, columns=california.feature_names)

# Definindo as variáveis X e y
X = df[['AveRooms']]  # Número médio de quartos
y = california.target  # Preço das casas

# Inicializando o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo
model.fit(X, y)

# Prevendo os valores de y com base nos valores de X
y_pred = model.predict(X)

# Plotando os dados
plt.scatter(X, y, alpha=0.4, label="Dados reais")
plt.plot(X, y_pred, color='red', label="Linha de regressão")

# Adicionando título e rótulos
plt.title("California Housing Dataset - Regressão Linear")
plt.xlabel("Número Médio de Quartos (AveRooms)")
plt.ylabel("Preço das Casas ($)")
plt.legend()

# Exibindo o gráfico
plt.show()
