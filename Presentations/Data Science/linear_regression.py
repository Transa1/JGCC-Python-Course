import numpy as np
import matplotlib.pyplot as plt

# 1. Generar dataset sintético (x, y) con ruido
np.random.seed(42)  # Para reproducibilidad
x = np.random.rand(50) * 10  # 50 puntos entre 0 y 10
y = 2.5 * x + np.random.randn(50) * 2  # Relación lineal con ruido

# 2. Calcular la regresión lineal usando la fórmula de mínimos cuadrados
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calcular los coeficientes de la línea: y = mx + b
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

# 3. Crear la línea de regresión
x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + b

# 4. Plotear los datos y la línea de regresión
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Datos originales')
plt.plot(x_line, y_line, color='red', label=f'Regresión lineal: y = {m:.2f}x + {b:.2f}')
plt.title('Regresión Lineal Manual')
plt.xlabel('Variable independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.grid(True)
plt.show()
