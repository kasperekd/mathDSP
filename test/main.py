import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# Генерация данных
mean = [0, 0]  # Мат. ожидания
cov = [[1, 0.7], [0.7, 1]]  # Ковариационная матрица (корреляция 0.7)
x, y = np.random.multivariate_normal(mean, cov, 5000).T  # 5000 точек

# Создание сетки для 2D-распределения
x_grid, y_grid = np.mgrid[-4:4:0.1, -4:4:0.1]
pos = np.dstack((x_grid, y_grid))
rv = multivariate_normal(mean, cov)
z = rv.pdf(pos)

# Построение графиков
plt.figure(figsize=(12, 6))

# 1. Двумерное распределение (heatmap)
plt.subplot(1, 2, 1)
plt.contourf(x_grid, y_grid, z, levels=20, cmap='viridis')
plt.colorbar(label='Плотность вероятности')
plt.scatter(x, y, alpha=0.2, color='red', label='Реализации')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Двумерное нормальное распределение')
plt.legend()

# 2. Одномерные проекции (маргинальные распределения)
plt.subplot(2, 2, 2)
sns.histplot(x, kde=True, color='blue', bins=30)
plt.title('Одномерное распределение X1')

plt.subplot(2, 2, 4)
sns.histplot(y, kde=True, color='green', bins=30)
plt.title('Одномерное распределение X2')

plt.tight_layout()
plt.show()