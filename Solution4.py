import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from matplotlib.pyplot import imread
from unsupervised.SVD import SVD

# Cargar la imagen
img = imread("my_image/DanielV.jpg")

# Convertir a escala de grises
img_gray = rgb2gray(img)

# Definir el rango de valores singulares a usar
singular_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]

# Iterar sobre los valores singulares
for i, sv in enumerate(singular_values):
    # Aplicar SVD
    svd = SVD(n_vectors=sv)
    svd.fit(img_gray)
    img_transformed = svd.transform(img_gray)
    img_reconstructed = svd.inverse_transform()
    
    # Plotear la imagen original y la transformada
    plt.subplot(3, 4, i+1)
    plt.title(f"SV = {sv}")
    plt.imshow(img_reconstructed, cmap=plt.cm.gray)
    plt.axis("off")

plt.suptitle("Images reconstructed using different numbers of singular values")
plt.figtext(0.5, 0.01, "It is observed that the image is adequately reproduced from 50 singular values", ha="center")
plt.show()