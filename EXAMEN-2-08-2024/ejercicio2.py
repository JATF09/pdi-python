import cv2

# Ruta de tu imagen
ruta_imagen = 'imagenPrueba.jpg'

# Leer la imagen en escala de grises desde el archivo
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Aplicar el detector de bordes de Canny
bordes = cv2.Canny(imagen, 100, 200)

# Cambiar el tamaño de la ventana
bordes = cv2.resize(bordes, (800, 600))

# Guardar la imagen con los bordes detectados
cv2.imwrite('imagen_bordes.jpg', bordes)

# Mostrar la imagen con los bordes detectados
cv2.imshow('Bordes Detectados', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()
