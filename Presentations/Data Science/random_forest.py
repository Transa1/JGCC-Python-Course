"""
Problema: Clasificación de especies de flores basado en características físicas.
El dataset contiene información sobre el largo y ancho de los pétalos de flores. 
Las especies son tres: Iris Setosa, Iris Versicolor e Iris Virginica.

Usaremos un modelo Random Forest para realizar la clasificación y evaluaremos 
el desempeño del modelo utilizando validación cruzada k-fold. Este enfoque es útil
en el ámbito de la botánica para identificar especies desconocidas basándose en medidas simples.
"""

# Importar librerías necesarias
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset: Largo y ancho de los pétalos (cm) y etiquetas de especies
X = np.array([
    [1.4, 0.2], [1.5, 0.2], [1.3, 0.2], [4.5, 1.5], [4.7, 1.4], [4.6, 1.5],
    [5.5, 2.1], [5.7, 2.3], [5.8, 2.2], [1.6, 0.4], [1.7, 0.5], [1.8, 0.4],
    [4.8, 1.6], [5.0, 1.7], [4.9, 1.5], [6.0, 2.5], [6.1, 2.6], [6.3, 2.4]
])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Configurar k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 particiones

# Variables para almacenar resultados
accuracies = []

# Proceso de validación cruzada
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Predecir con los datos de prueba
    y_pred = model.predict(X_test)
    
    # Calcular precisión y guardarla
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold Accuracy: {acc:.2f}")

# Resultados finales
print("\nResultados finales:")
print(f"Precisión media: {np.mean(accuracies):.2f}")
print(f"Desviación estándar de la precisión: {np.std(accuracies):.2f}")

# Visualización del dataset y del modelo
plt.figure(figsize=(8, 6))

# Entrenar el modelo con todo el dataset para graficar la frontera
model.fit(X, y)

# Crear un meshgrid para la visualización de la frontera
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), 
                     np.linspace(y_min, y_max, 500))

# Predecir con el modelo para el meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar las regiones de decisión
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')

# Graficar los puntos del dataset
species = ['Setosa', 'Versicolor', 'Virginica']
colors = ['blue', 'orange', 'green']
for i, label in enumerate(np.unique(y)):
    plt.scatter(X[y == label][:, 0], X[y == label][:, 1], 
                label=f'{species[label]}', s=80, edgecolor='k', color=colors[label])

# Añadir detalles al gráfico
plt.title("Clasificación de Especies de Flores con Random Forest")
plt.xlabel("Largo del Pétalo (cm)")
plt.ylabel("Ancho del Pétalo (cm)")
plt.legend()
plt.grid()
plt.show()
