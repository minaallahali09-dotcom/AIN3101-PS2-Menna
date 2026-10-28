import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data[:500]
y = data.target[:500]
X_b = np.c_[np.ones((len(X),1)), X]

theta_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(theta_normal)

def gd_with_history(X_b, y, lr, n_iter=200):
    m = len(y)
    theta = np.zeros(X_b.shape[1])
    costs = []
    for i in range(n_iter):
        y_pred = X_b @ theta
        cost = np.mean((y_pred - y)**2)
        costs.append(cost)
        grad = (1/m) * X_b.T @ (y_pred - y)
        theta -= lr * grad
    return theta, costs

theta1, cost1 = gd_with_history(X_b, y, lr=0.01)
theta2, cost2 = gd_with_history(X_b, y, lr=0.1)

plt.plot(cost1, label='lr=0.01')
plt.plot(cost2, label='lr=0.1')
plt.xlabel('Iteration')
plt.ylabel('MSE Cost')
plt.legend()
plt.title('Cost vs Iterations')
plt.savefig('cost_plot.png')
