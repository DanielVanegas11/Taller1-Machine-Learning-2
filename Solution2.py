#python picture.py

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class Imagenes:
    def __init__(self, ruta_imagenes):
        self.ruta_imagenes = ruta_imagenes
        self.imagenes = self.cargar_imagenes()
        self.imagen_promedio = self.calcular_promedio()
    
    def cargar_imagenes(self):
        imagenes = []
        for filename in os.listdir(self.ruta_imagenes):
            img = cv2.imread(os.path.join(self.ruta_imagenes, filename))
            if img is not None:
                imagenes.append(img)
        return imagenes
    
    def calcular_promedio(self):
        for i in range(len(self.imagenes)):
            self.imagenes[i] = cv2.resize(self.imagenes[i], (256, 256))
        imagenes_matriz = np.array(self.imagenes)
        promedio_matriz = np.mean(imagenes_matriz, axis=0).astype(np.uint8)
        return promedio_matriz

#my_image = "my_image"
all_images="images"

# Cargar imagen de Daniel y redimensionarla
daniel = cv2.imread("my_image/DanielV.jpg")
daniel = cv2.resize(daniel, (256, 256))

# Mostrar imagen de Daniel
daniel_gris = cv2.cvtColor(daniel, cv2.COLOR_BGR2GRAY)
plt.imshow(daniel_gris, cmap="gray")
plt.title("Imagen de Daniel")
plt.show()

# Calcular y mostrar imagen promedio
imagenes = Imagenes(all_images)
plt.imshow(cv2.cvtColor(imagenes.imagen_promedio, cv2.COLOR_BGR2GRAY))
plt.title("Imagen promedio de rostros")
plt.show()



