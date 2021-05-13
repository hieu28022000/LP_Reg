import cv2
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import matplotlib.pyplot as plt

image = plt.imread('../test_merge.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print (image)
cv2.imshow('image', image)
cv2.waitKey(0)

coords = corner_peaks(corner_harris(image))
coords_subpix = corner_subpix(image, coords)
print (coords)
print (coords_subpix)