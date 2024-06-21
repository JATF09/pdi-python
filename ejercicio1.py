# Implementar la segmentacion de una imagen utilizando el algoritmo de k-means clustering
# Pasos:
# 1. leer una imagen y convertirla a un espacion de color adecuado(por ejemplo RGB a L'a'b')
# 2. aplicar el algoritmo de k-means para segmentar la imagen de k regiones
# 3. visualizar la imagen segmentada y comparar los resultados con la original
# tips:
# utiliza la libreria de opencv para leer y mostrar la imagen
# utiliza la libreria de scikit-learn para aplicar k-means clustering

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('imagen.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
pixel_values = image_lab.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# clusters
k = 3

_, labels, centers = cv2.kmeans(
    pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image_lab.shape)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_rgb)
plt.title('Imagen Segmentada')
plt.axis('off')

plt.show()
