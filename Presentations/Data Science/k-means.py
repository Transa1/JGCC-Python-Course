"""
Problema: Segmentación de clientes basada en datos de gasto anual y frecuencia de compra.
El objetivo es agrupar a los clientes de un centro comercial en diferentes segmentos según sus hábitos,
como clientes de alto valor, compradores regulares y clientes ocasionales.

Usaremos el algoritmo K-Means para identificar patrones en los datos y agrupar a los clientes en tres segmentos.
Esta segmentación puede ser utilizada para personalizar estrategias de marketing y mejorar la experiencia del cliente.
"""

# Importar librerías necesarias
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Dataset: Gasto anual ($) y frecuencia de compra (número de visitas por año)
X = np.array([
    [20000, 12], [22000, 15], [25000, 10], [80000, 40], [85000, 38], [78000, 45],
    [30000, 18], [32000, 22], [31000, 20], [90000, 50], [91000, 52], [87000, 48],
    [5000, 5], [7000, 8], [4000, 3], [2500, 2], [3000, 1], [3500, 2]
])

# Número de clústeres
k = 3

# Crear y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Obtener las etiquetas de los clústeres
labels = kmeans.labels_

# Calcular el puntaje de silueta
silhouette_avg = silhouette_score(X, labels)

print(f"Puntaje de silueta: {silhouette_avg:.2f}")

# Visualización de los clústeres
plt.figure(figsize=(8, 6))

# Graficar los puntos del dataset y colorearlos según su clúster
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                label=f'Clúster {i + 1}', s=100, edgecolor='k')

# Graficar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', label='Centroides', marker='X')

# Añadir detalles al gráfico
plt.title("Segmentación de Clientes con K-Means")
plt.xlabel("Gasto Anual ($)")
plt.ylabel("Frecuencia de Compra (visitas/año)")
plt.legend()
plt.grid()
plt.show()
