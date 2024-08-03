import cv2

# Ruta de tu imagen
ruta_imagen = 'imagenPrueba.jpg'

# Leer la imagen en color desde el archivo
imagen_color = cv2.imread(ruta_imagen)

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Aplicar la ecualización del histograma
imagen_ecualizada = cv2.equalizeHist(imagen_gris)

# Cambia el tamaño de la ventana
imagen_ecualizada = cv2.resize(imagen_ecualizada, (800, 600))

# Guardar la imagen resultante
cv2.imwrite('imagen_ecualizada.jpg', imagen_ecualizada)

# Mostrar la imagen resultante
cv2.imshow('Imagen Ecualizada', imagen_ecualizada)
cv2.waitKey(0)
cv2.destroyAllWindows()
