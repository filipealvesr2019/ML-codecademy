import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Carregando o dataset
df = pd.read_csv("honeyproduction.csv")
print(df.head())

# Agrupando os dados por ano e calculando a produção média de mel por ano
prod_per_year = df.groupby('year').totalprod.mean()

# Definindo as variáveis independentes (ano) e dependentes (produção total de mel)
X = prod_per_year.index.values.reshape(-1, 1)  # Usando o ano como variável independente
y = prod_per_year.values  # Produção total de mel como variável dependente

# Criando o modelo de regressão linear
model = linear_model.LinearRegression()

# Ajustando o modelo aos dados
model.fit(X, y)

# Imprimindo a inclinação (slope) e o intercepto da linha
print("Inclinação:", model.coef_[0])
print("Intercepto:", model.intercept_)

# Prevendo os valores de y com base nos valores de X
y_predict = model.predict(X)

# Plotando a linha de regressão
plt.plot(X, y_predict, color='red', label="Linha de Regressão")

# Plotando os dados reais
plt.scatter(X, y, alpha=0.4, label="Dados reais")

# Adicionando título, rótulos e legenda
plt.title("Produção de Mel por Ano - Regressão Linear")
plt.xlabel("Ano")
plt.ylabel("Produção Total de Mel")
plt.legend()

# Exibindo o gráfico
plt.show()

# Prevendo o futuro (2050)
X_future = np.array(range(2013, 2051))  # De 2013 até 2050
X_future = X_future.reshape(-1, 1)  # Ajustando para o formato correto para o scikit-learn
future_predict = model.predict(X_future)  # Predizendo a produção de mel para o futuro

# Plotando as previsões futuras
plt.plot(X_future, future_predict, color='green', label="Previsão de Produção de Mel")
plt.scatter(X, y, alpha=0.4, label="Dados reais")
plt.title("Previsão de Produção de Mel para 2050")
plt.xlabel("Ano")
plt.ylabel("Produção Total de Mel")
plt.legend()
plt.show()

# Prevendo a produção de mel no ano 2050
honey_2050 = future_predict[-1]  # O último valor da previsão será a produção de mel em 2050
print(f"Produção de mel em 2050: {honey_2050}")
