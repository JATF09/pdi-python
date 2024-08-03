import cv2

# Ruta de tu imagen
ruta_imagen = 'imagenPrueba.jpg'

# Leer la imagen desde el archivo
imagen = cv2.imread(ruta_imagen)

# Cambiar el tamaño de la imagen a 300x300 píxeles
imagen_redimensionada = cv2.resize(imagen, (300, 300))

# Guardar la imagen redimensionada
cv2.imwrite('imagen_redimensionada.jpg', imagen_redimensionada)

# Mostrar la imagen redimensionada
cv2.imshow('Imagen Redimensionada', imagen_redimensionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
