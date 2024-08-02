import cv2

# Ruta de tu imagen
ruta_imagen = 'imagenPrueba.jpg'

# Leer la imagen desde el archivo
imagen = cv2.imread(ruta_imagen)

# Aplicar un filtro de suavizado gaussiano con un kernel 5x5
imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)

# Cambia el tamaño de la ventana
imagen_suavizada = cv2.resize(imagen_suavizada, (800, 600))

# Guardar la imagen suavizada
cv2.imwrite('imagen_suavizada.jpg', imagen_suavizada)

# Mostrar la imagen suavizada
cv2.imshow('Imagen Suavizada', imagen_suavizada)
cv2.waitKey(0)
cv2.destroyAllWindows()
