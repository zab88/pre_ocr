import numpy as np
import cv2, os, subprocess, math
from matplotlib import pyplot as plt

# https://github.com/danvk/oldnyc/blob/master/ocr/tess/crop_morphology.py

file = 'img/a1.JPG'

img_origin = cv2.imread(file)

# black and white
img_grey = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY )

# edges
#gray = cv2.GaussianBlur(image, (3, 3), 0)
edges = cv2.Canny(img_grey, 100, 200)
#_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hh = np.sum(np.sign(edges), 1)
print(hh)
h, w = edges.shape[0:2]
# plt.hist( hh, h, [0, h])
plt.plot(hh)
plt.show()


# morph
kernel_1 = np.ones((4, 4), np.uint8)
# kernel_3 = np.ones((3, 3), np.uint8)
# kernel_5 = np.ones((5, 5), np.uint8)
# dilation = cv2.dilate(edges, kernel_3, iterations = 1)
# dilation = cv2.erode(dilation, kernel_1, iterations = 1)
dilation = cv2.dilate(edges, kernel_1, iterations = 1)
opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_1)

# select box
#img_bin = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_OTSU)[1]

cv2.imshow("Image origin", cv2.pyrDown(img_origin))
cv2.imshow("edge", edges)
cv2.imshow("dilate", opening)
# cv2.imshow("Image e", cv2.pyrDown(img_bin))

cv2.waitKey(0)
