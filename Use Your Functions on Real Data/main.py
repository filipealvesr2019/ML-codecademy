from gradient_descent_funcs import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

X = df["height"]
y = df["weight"]

plt.plot(X, y, 'o')
#plot your line here:
b, m = gradient_descent(X, y, learning_rate=0.0001, num_iterations=1000)
y_predictions = [m * x_val + b for x_val in X]
plt.plot(X, y_predictions)
plt.show()