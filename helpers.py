import cv2, itertools
import numpy as np
from itertools import chain

def get_color_by_mask(img, mask):
    img_bin = mask.copy()
    img_origin = img.copy()
    # cv2.RETR_EXTERNAL cv2.RETR_TREE
    # older versions can be slight different
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
    # print( sorted(ccc.items(), key=lambda x: x[1], reverse=True))
    cv2.imshow('rectangles', img_origin)
    most_color = sorted(ccc.items(), key=lambda x: x[1], reverse=True)[0][0]
    print('most color detected = ' + str(most_color))
    return most_color

def get_crop_tuples(mask, axis, threshold=20, border=4):
    # if int(axis) != 1 or int(axis) != 0:
    #     print('can not use crop tuples', axis)
    #     return None
    hist = np.sum(np.sign(mask), axis)
    avg_hist = np.mean(hist)

    row_01 = [1 if x>avg_hist else 0 for x in hist]
    # print(row_01)
    # print(len(row_01))
    max_ = len(row_01)
    offset = 0
    out = []
    for n, l in itertools.groupby(row_01):
        # print(n, offset)
        offset_add = sum(1 for _ in l)
        if n == 1 and offset_add>threshold:
            out.append( (max(0, offset-border), min(max_, offset + offset_add+border)) )
        offset += offset_add
    # lala = max(sum(1 for _ in l) for n, l in itertools.groupby(row_01))
    # print(lala)
    print(out)
    return out