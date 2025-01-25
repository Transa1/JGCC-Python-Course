import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Definir las clases del dataset
class_names = [str(i) for i in range(10)]

# Crear el modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nPrecisión en el conjunto de prueba: {test_acc:.2f}")

# Función para plotear 10 imágenes al azar con sus predicciones
def plot_random_predictions(model, images, labels):
    indices = np.random.choice(len(images), 10, replace=False)
    selected_images = images[indices]
    selected_labels = labels[indices]
    predictions = model.predict(selected_images)

    plt.figure(figsize=(10, 5))
    for i, (image, label, prediction) in enumerate(zip(selected_images, selected_labels, predictions)):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        predicted_label = np.argmax(prediction)
        color = 'green' if predicted_label == label else 'red'
        plt.xlabel(f"{class_names[predicted_label]}\n(True: {class_names[label]})", color=color)
    plt.tight_layout()
    plt.show()

# Usar la función para plotear imágenes con predicciones
plot_random_predictions(model, test_images, test_labels)
