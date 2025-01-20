"""
Problema: Clasificación de pacientes como sanos o en riesgo de diabetes basado en medidas simples.
El dataset contiene información de pacientes con dos características: nivel de glucosa (mg/dL) y 
presión arterial (mmHg). Queremos predecir si un paciente está en riesgo de diabetes (1) o no (0).

Usaremos una Máquina de Soporte Vectorial (SVM) con kernel lineal para realizar esta clasificación
y evaluaremos el desempeño del modelo utilizando validación cruzada k-fold.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Dataset: Niveles de glucosa (mg/dL) y presión arterial (mmHg) con etiquetas
X = np.array([
    [85, 60], [90, 65], [78, 70],  # Pacientes sanos (0)
    [180, 95], [190, 100], [200, 105],  # En riesgo de diabetes (1)
    [88, 72], [92, 74], [85, 68],  # Pacientes sanos (0)
    [170, 90], [175, 85], [190, 92]   # En riesgo de diabetes (1)
])

y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])  # Etiquetas: 0 = Sano, 1 = En riesgo

model = SVC(kernel='linear')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f'Precision del Fold: {accuracy:.2f}')



# Visualización del dataset y del modelo
plt.figure(figsize=(8, 6))

# Entrenar el modelo con todo el dataset para graficar la frontera
model.fit(X, y)

# Graficar los puntos del dataset
for i, label in enumerate(np.unique(y)):
    plt.scatter(X[y == label][:, 0], X[y == label][:, 1], 
                label=f'Clase {label}', s=80, edgecolor='k')

# Crear un meshgrid para la visualización de la frontera
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), 
                     np.linspace(y_min, y_max, 500))

# Predecir con el modelo para el meshgrid
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la línea de decisión y los márgenes
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
plt.contourf(xx, yy, Z > 0, alpha=0.2, colors=['lightblue', 'lightcoral'])

# Añadir detalles al gráfico
plt.title("Clasificación de Pacientes con SVM (Kernel Lineal)")
plt.xlabel("Glucosa (mg/dL)")
plt.ylabel("Presión Arterial (mmHg)")
plt.legend()
plt.grid()
plt.show()