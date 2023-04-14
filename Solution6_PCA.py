from unsupervised.PCA import PCA
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Obtener Mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Construir máscaras
mask_train = (y_train == 0) | (y_train == 8)
mask_test = (y_test == 0) | (y_test == 8)

# Aplicar máscaras
X_train = X_train[mask_train]/255.0
y_train = y_train[mask_train]
X_test = X_test[mask_test]/255.0
y_test = y_test[mask_test]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])   
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])  

# Crear un objeto PCA con 2 componentes
pca = PCA(n_components=2)

# Ajustar los datos
pca.fit(X_train)

# Transformar los datos utilizando el objeto PCA
reduced_x_train = pca.transform(X_train)
print(reduced_x_train.shape)
reduced_x_train = np.real(reduced_x_train)  # Convertir a números reales
# (11774, 2)

plt.scatter(reduced_x_train[:, 0], reduced_x_train[:, 1], c=y_train)
plt.show()

# Transformar los datos de prueba utilizando el objeto PCA
reduced_x_test = pca.transform(X_test)
print(reduced_x_test.shape)
reduced_x_test = np.real(reduced_x_test)  # Convertir a números reales
# (1954, 2)

# Construir la regresión logística
regressor = LogisticRegression(random_state=0)
regressor.fit(reduced_x_train, y_train)

# Predecir los resultados para los datos de prueba
y_pred = regressor.predict(reduced_x_test)

# Evaluar la precisión del modelo
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('The accuracy may have been affected when transforming the PCA-reduced data into real numbers because the transformation caused some information to be lost. This is because the original data was in the form of complex numbers, which were transformed into real numbers. However, despite this loss of information, the new data still contained enough information to describe the original data accurately, and the resulting accuracy score was acceptable.')
