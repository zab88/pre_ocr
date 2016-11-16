import numpy as np
import cv2, os, subprocess, math
from matplotlib import pyplot as plt
from itertools import chain

# https://github.com/danvk/oldnyc/blob/master/ocr/tess/crop_morphology.py

file = 'img/c2.JPG'

img_origin = cv2.imread(file)
img_blur = cv2.GaussianBlur(img_origin.copy(), (3, 3), 0)
# img_blur = img_origin.copy()

# black and white
img_grey = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY )

# edges
ddepth = cv2.CV_16S
scale = 1
delta = 0
# sobelx = cv2.Sobel(img_grey.copy(), cv2.CV_64F,1,0,ksize=1)
# sobely = cv2.Sobel(img_grey.copy(), cv2.CV_64F,0,1,ksize=1)

grad_x = cv2.Sobel(img_grey.copy() ,ddepth, 1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(img_grey.copy(), ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
abs_grad_y = cv2.convertScaleAbs(grad_y)

dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
img_bin = cv2.threshold(dst.copy(), 127, 255, cv2.THRESH_OTSU)[1]

# cv2.RETR_EXTERNAL cv2.RETR_TREE
_, contours, hierarchy = cv2.findContours((img_bin.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
lala_all = []
for cnt in contours:
    # print(cnt)
    approx_area = cv2.contourArea(cnt)
    if approx_area > 10:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_origin, (x,y), (x+w,y+h), (0, 255, 0), 2)

        # histograms http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
        lala = img_origin[y:y+h, x:x+w]
        lala = cv2.cvtColor(lala, cv2.COLOR_BGR2HSV)
        lala_colors = lala[:,:,2]
        lala_colors = ( list(chain.from_iterable(lala_colors)) )
        lala_all += lala_colors
        # hist_lala = cv2.calcHist([lala], [2], None, [256], [0,256])
        # # print cv2.GetMinMaxHistValue(hist_lala)[1]
        # if hist_all is None:
        #     hist_all = hist_lala
        # else:
        #     hist_all = hist_all + hist_lala
        # print(hist_lala)
        # exit()

ccc = dict((x, lala_all.count(x)) for x in lala_all)
print( sorted(ccc.items(), key=lambda x: x[1], reverse=True))
most_color = sorted(ccc.items(), key=lambda x: x[1], reverse=True)[0][0]
print(most_color)

    # approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
    # approx_area = cv2.contourArea(approx)
    # x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(approx)
    #
    # if approx_area > 2048 and np.float32(w_cnt)/np.float32(h_cnt)>0.7 and np.float32(w_cnt)/np.float32(h_cnt)<1.4:
    #     cv2.drawContours( img_origin, [cnt], -1, (0,0,255), 1)
    #     rect = cv2.minAreaRect(approx)

# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# #gray = cv2.GaussianBlur(image, (3, 3), 0)
# edges = cv2.Canny(img_grey, 100, 200)
# #_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# hh = np.sum(np.sign(edges), 1)
# print(hh)
# h, w = edges.shape[0:2]
# # plt.hist( hh, h, [0, h])
# # plt.plot(hh)
# # plt.show()
#
#
# # morph
# kernel_1 = np.ones((4, 4), np.uint8)
# # kernel_3 = np.ones((3, 3), np.uint8)
# # kernel_5 = np.ones((5, 5), np.uint8)
# # dilation = cv2.dilate(edges, kernel_3, iterations = 1)
# # dilation = cv2.erode(dilation, kernel_1, iterations = 1)
# dilation = cv2.dilate(edges, kernel_1, iterations = 1)
# opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_1)

# select box
#img_bin = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_OTSU)[1]

lower = np.array([0, 0, 205])
upper = np.array([255, 255, 255])
hsv = cv2.cvtColor(img_origin.copy(), cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

# cv2.imshow("Image origin", cv2.pyrDown(img_origin))
cv2.imshow("Image origin", img_origin)
# cv2.imshow("edge", edges)
# cv2.imshow("dilate", opening)
# cv2.imshow("Image e", cv2.pyrDown(img_bin))
# cv2.imshow("X", sobelx)
# cv2.imshow("Y", sobely)
cv2.imshow("DST", dst)
cv2.imshow("bin", img_bin)
cv2.imshow("mask", mask)
cv2.imshow("mask_inverted", (255-mask))
# cv2.imshow("hsv", hsv)

cv2.waitKey(0)
