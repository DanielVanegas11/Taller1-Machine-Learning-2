from flask import Flask, jsonify, request
from unsupervised import PCA
from sklearn.datasets import load_digits
import numpy as np

app = Flask(__name__)

# Carga los datos MNIST
digits = load_digits()
X = digits.data
y = digits.target

# Entrena el modelo PCA con los datos MNIST
pca = PCA(n_components=2)
pca.fit(X)

# Define una ruta para clasificar una imagen
@app.route('/classify', methods=['POST'])
def classify():
    # Obtiene la imagen del request
    image = request.get_json()['image']
    # Convierte la imagen a un arreglo de numpy
    image = np.array(image)
    # Asegura que la imagen tenga la forma correcta
    if image.shape != (8, 8):
        return jsonify({'error': 'La imagen debe tener una forma de 8x8.'})
    # Transforma la imagen con el modelo PCA
    transformed_image = pca.transform(image.reshape(1, -1))
    # Encuentra la imagen más cercana en el conjunto de datos de entrenamiento
    closest_image_index = np.argmin(np.linalg.norm(transformed_image - pca.transform(X), axis=1))
    # Obtiene la clase de la imagen más cercana
    predicted_class = y[closest_image_index]
    # Devuelve la clase como un JSON
    return jsonify({'class': int(predicted_class)})

if __name__ == '__main__':
    app.run()