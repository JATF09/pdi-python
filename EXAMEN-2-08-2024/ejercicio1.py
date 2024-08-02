import cv2

# Ruta de tu imagen
ruta_imagen = 'imagenPrueba.jpg'

# Leer la imagen en color desde el archivo
imagen_color = cv2.imread(ruta_imagen)

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Cambiar el tamaño de la ventana
imagen_gris = cv2.resize(imagen_gris, (800, 600))

# Guardar la imagen en escala de grises
cv2.imwrite('imagen_gris.jpg', imagen_gris)

# Mostrar la imagen en escala de grises

cv2.imshow('Imagen en Escala de Grises', imagen_gris)
cv2.waitKey(0)
cv2.destroyAllWindows()
