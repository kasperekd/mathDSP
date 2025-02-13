# 1.
# Выражение плотности вероятности нормального СП W(x)
# W(x) = (1)/(root(2*pi*sigma^2)) exp((-(x-m_x)^2)/(2*sigma^2))
# Параметры распределения – матожидание (среднее значение) m_x и дисперсия sigma^2
# Постройте график плотности распределения для значений x от -5 до 5 с шагом 0.01.
# Приведите графики плотности с параметрами m_x =0, sigma^2=1 , m_x =0, sigma^2=3 , m_x =0, sigma^2=0.2 , m_x =-1, sigma^2=1. 
# 2.
# Получите вектор значений СВ с нормальным распределением. 
# Параметры m, s1 = root(sigma^2)- среднеквадратичное отклонение из предыдущего задания. 
# t=np.linspace(0,3,1000)
# xn=np.random.normal(m,s1,len(t))
# Приведите графики выборки и соответствующие плотности распределения.
# 3.
# Приведите графики выборки и соответствующие плотности распределения.
#     3. Построение гистограммы распределения (эмпирической плотности распределения)
#     1. Диапазон значений СВ разбивается на большое количество сегментов (бинов). Создайте вектор с границами бинов (вектор от начального до конечного значений диапазона СВ с необходимым шагом используя np.arange
#     2. Определите центральное значение каждого сегмента
#     3. Для каждого сегмента задайте счетчик количества попаданий значений СВ в сегмент. 
#     4. Для каждого значения вектора СВ с нормальным распределение определите, в какой сегмент оно попадает инкрементируйте счетчик попаданий каждого сегмента. 
#     5. После проверки всех значений случайного вектора, значения счетчика попаданий нормируются на (общее количество значений СВ * шаг значений)
# Постойте графики эмпирической ПВ (гистограммы распределений) для указанных параметров СВ.
# 4. 
# Определите числовые параметры полученной СВ 
# Матожидание: m_x = (SUM^N_(n=1)(x(n)))/(N)   
# Дисперсия: sigma^2_x = SUM^N_(n=1)(x^2(n)))/(N) - m^2_x 
# Сравните полученные значения с параметрами моделирования.
# 5. 
# Определение числовых характеристик при помощи эмпирической плотности распределения
# Матожидание вычисляется по плотности распределения по выражению: m_x = INTEGRAL(x*W(x)dx)
# Дисперсия: sigma^2_x = INTEGRAL(x^2*W(x)dx) - m^2_x
# X – вектор возможных значений СВ, заданный в 3.1.
# Сравните полученные значения с параметрами моделирования.

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm
# from scipy.integrate import trapz

def normal_pdf(x, m_x, sigma_squared):
    sigma = np.sqrt(sigma_squared)
    return (1 / (np.sqrt(2 * np.pi * sigma_squared))) * np.exp(-((x - m_x) ** 2) / (2 * sigma_squared))

# Задание 1:

x = np.arange(-5, 5, 0.01)

params = [
    {'m_x': 0, 'sigma_squared': 1},
    {'m_x': 0, 'sigma_squared': 3},
    {'m_x': 0, 'sigma_squared': 0.2},
    {'m_x': -1, 'sigma_squared': 1}
]

plt.figure(figsize=(14, 10))

for param in params:
    pdf_values = normal_pdf(x, param['m_x'], param['sigma_squared'])
    label = f"m_x={param['m_x']}, sigma^2={param['sigma_squared']}"
    plt.plot(x, pdf_values, label=label)

plt.title('Плотность вероятности нормального распределения')
plt.xlabel('x')
plt.ylabel('W(x)')
plt.legend()
plt.grid(True)
# plt.show()

# Задание 2:

x = np.arange(-5, 5, 0.01)

params = [
    {'m_x': 0, 'sigma_squared': 1},
    {'m_x': 0, 'sigma_squared': 3},
    {'m_x': 0, 'sigma_squared': 0.2},
    {'m_x': -1, 'sigma_squared': 1}
]

t = np.linspace(0, 3, 10000000)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, param in enumerate(params):
    m_x = param['m_x']
    sigma_squared = param['sigma_squared']
    sigma = np.sqrt(sigma_squared)
    
    pdf_values = normal_pdf(x, m_x, sigma_squared)
    
    xn = np.random.normal(m_x, sigma, len(t))
    
    axs[i].plot(x, pdf_values, label=f'PDF: m_x={m_x}, sigma^2={sigma_squared}', color='blue', linewidth=2)
    
    axs[i].hist(xn, bins=100, density=True, alpha=0.6, color='gray', edgecolor='black', label='Выборка')
    
    axs[i].set_title(f'm_x={m_x}, sigma^2={sigma_squared}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('Плотность вероятности')
    axs[i].legend()

for ax in axs:
    ax.grid(True)

# plt.tight_layout()
# plt.show()

# Задание 3:

params = [
    {'m_x': 0, 'sigma_squared': 1},
    {'m_x': 0, 'sigma_squared': 3},
    {'m_x': 0, 'sigma_squared': 0.2},
    {'m_x': -1, 'sigma_squared': 1}
]

num_bins = 30
bin_step = (5 - (-5)) / num_bins

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, param in enumerate(params):
    m_x = param['m_x']
    sigma_squared = param['sigma_squared']
    sigma = np.sqrt(sigma_squared)
    
    xn = np.random.normal(m_x, sigma, len(t))
    
    bin_edges = np.arange(-5, 5 + bin_step, bin_step)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    counts, _ = np.histogram(xn, bins=bin_edges)
    empirical_pdf = counts / (len(xn) * bin_step)
    
    pdf_values = normal_pdf(x, m_x, sigma_squared)
    axs[i].plot(x, pdf_values, label=f'PDF: m_x={m_x}, sigma^2={sigma_squared}', color='blue', linewidth=2)
    
    axs[i].bar(bin_centers, empirical_pdf, width=bin_step, alpha=0.6, color='gray', edgecolor='black', label='Эмпирическая PDF')
    
    axs[i].set_title(f'm_x={m_x}, sigma^2={sigma_squared}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('Плотность вероятности')
    axs[i].legend()

for ax in axs:
    ax.grid(True)

# plt.tight_layout()
# plt.show()

# Задание 4:

params = [
    {'m_x': 0, 'sigma_squared': 1},
    {'m_x': 0, 'sigma_squared': 3},
    {'m_x': 0, 'sigma_squared': 0.2},
    {'m_x': -1, 'sigma_squared': 1}
]

# t = np.linspace(0, 3, 1000)

print("Задание 4:")
print("----------------------------------------------------")

for param in params:
    m_x = param['m_x']
    sigma_squared = param['sigma_squared']
    sigma = np.sqrt(sigma_squared)
    
    xn = np.random.normal(m_x, sigma, len(t))
    
    m_x_estimated = np.mean(xn)
    
    sigma_squared_estimated = np.var(xn)
    
    print(f"m_x={m_x}, sigma^2={sigma_squared}")
    print(f"m_x={m_x_estimated:.4f}, sigma^2={sigma_squared_estimated:.4f}")
    print("----------------------------------------------------")


# Задание 5:

params = [
    {'m_x': 0, 'sigma_squared': 1},
    {'m_x': 0, 'sigma_squared': 3},
    {'m_x': 0, 'sigma_squared': 0.2},
    {'m_x': -1, 'sigma_squared': 1}
]

# t = np.linspace(0, 3, 1000)

print("Задание 5:")
print("--------------------------------------------------------------------------------------")

for param in params:
    m_x = param['m_x']
    sigma_squared = param['sigma_squared']
    sigma = np.sqrt(sigma_squared)
    
    xn = np.random.normal(m_x, sigma, len(t))
    
    bin_edges = np.arange(-5, 5 + bin_step, bin_step)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    counts, _ = np.histogram(xn, bins=bin_edges)
    empirical_pdf = counts / (len(xn) * bin_step)
    
    m_x_empirical = np.sum(bin_centers * empirical_pdf * bin_step)
   
    sigma_squared_empirical = np.sum((bin_centers**2) * empirical_pdf * bin_step) - m_x_empirical**2
    
    print(f"m_x={m_x}, sigma^2={sigma_squared}")
    print(f"m_x={m_x_empirical:.4f}, sigma^2={sigma_squared_empirical:.4f}")
    print("--------------------------------------------------------------------------------------")

plt.tight_layout()
plt.show()