import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target
X_08 = X[(y == '0') | (y == '8')]
y_08 = y[(y == '0') | (y == '8')]

# Convertir las etiquetas de clase a enteros
y_08 = y_08.astype(int)

scaler = StandardScaler()
X_08_scaled = scaler.fit_transform(X_08)

# Dividir los datos transformados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_08_scaled, y_08, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluar el rendimiento del modelo en los datos de prueba
accuracy = model.score(X_test, y_test)
print(f"La precisión del modelo es: {accuracy:.2f}")

# Utilizar el método SVD
svd = TruncatedSVD(n_components=2)
X_08_svd = svd.fit_transform(X_08_scaled)

# Utilizar el método t-SNE
tsne = TSNE(n_components=2)
X_08_tsne = tsne.fit_transform(X_08_scaled)

# Utilizar el método PCA
pca = PCA(n_components=2)
X_08_pca = pca.fit_transform(X_08_scaled)

# Crear las gráficas y agregar la precisión en el título
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

for i, (X_transformed, title) in enumerate(zip([X_08_svd, X_08_tsne, X_08_pca], ['SVD', 't-SNE', 'PCA'])):
    axs[i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_08, cmap='viridis')
    axs[i].set_title(f"{title} - Precisión: {accuracy:.2f}")
    
plt.show()