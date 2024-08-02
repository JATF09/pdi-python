import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('imagen1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagen2.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                              matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title('Emparejamientos de características entre dos imágenes')
plt.show()
