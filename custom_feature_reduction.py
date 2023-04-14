import numpy as np
import matplotlib.pyplot as plt
from unsupervised.PCA import PCA
from unsupervised.SVD import SVD
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784',parser='auto')
X = mnist.data
y = mnist.target
X_08 = X[(y == '0') | (y == '8')]
y_08 = y[(y == '0') | (y == '8')]

# Convertir las etiquetas de clase a enteros
y_08 = y_08.astype(int)

scaler = StandardScaler()
X_08_scaled = scaler.fit_transform(X_08)

# Dividir los datos transformados en conjuntos de entrenamiento y prueba
X_train, X_test, y_08_train, y_test = train_test_split(X_08_scaled, y_08, test_size=0.2, random_state=42)
X_test = np.real(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
accuracy = []
for i, (transformer, name) in enumerate([(SVD(n_components=2), 'SVD'),
                                          (PCA(n_components=2), 'PCA')]):
    X_transformed = np.real(transformer.fit_transform(X_08_scaled))
    model.fit(X_transformed, y_08)
    X_test = X_test.real
    accuracy.append(model.score(transformer.transform(X_test), y_test))
    plt.subplot(1, 2, i+1)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_08, cmap='viridis')
    plt.title(f"{name} - Precisión: {accuracy[i]:.3f}")

plt.show()