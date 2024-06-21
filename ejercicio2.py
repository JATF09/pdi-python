import cv2
import matplotlib.pyplot as plt

# Leer las dos imágenes
img1 = cv2.imread('imagen1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagen2.jpg', cv2.IMREAD_GRAYSCALE)

# Crear el detector de características ORB
orb = cv2.ORB_create()

# Detectar y describir las características clave en ambas imágenes
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Crear un objeto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Emparejar las características usando los descriptores
matches = bf.match(descriptors1, descriptors2)

# Ordenar los emparejamientos por la distancia (los mejores primeros)
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar los primeros 50 emparejamientos
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                              matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar los emparejamientos
plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title('Emparejamientos de características entre dos imágenes')
plt.show()
