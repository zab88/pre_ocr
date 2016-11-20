import cv2, itertools, os
import numpy as np
from itertools import chain
from glob import glob

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
    # print(avg_hist)
    # if avg_hist < 50:
    #     return []

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
    # print(out)
    return out

class Frame():
    def __init__(self, img):
        _, img_file_name = os.path.split(img)
        self.name = img_file_name
        self.time = img[-15:-4]
        self.is_new = False
        img_origin = cv2.imread(img)
        img_grey = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        # grad_x = cv2.Sobel(img_grey.copy(), cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType = cv2.BORDER_DEFAULT)
        # grad_y = cv2.Sobel(img_grey.copy(), cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType = cv2.BORDER_DEFAULT)
        #
        # abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
        # abs_grad_y = cv2.convertScaleAbs(grad_y)
        #
        # dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # img_bin = cv2.threshold(dst.copy(), 127, 255, cv2.THRESH_OTSU)[1]

        img_bin = cv2.Canny(img_grey, 100, 200)
        height, width = img_bin.shape[:2]
        self.lines_offset = get_crop_tuples(img_bin, 1, Timeline.font_height)

        # compare with previous
        if Timeline.last_img is not None:
            fgbg = cv2.BackgroundSubtractorMOG()
            fgmask = fgbg.apply(Timeline.last_img)
            fgmask = fgbg.apply(img_origin)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            # print(current_dir + os.sep + 'tmp' + os.sep + self.name)
            non_zero = cv2.countNonZero(fgmask)
            if non_zero > height*width*0.01:
                self.is_new = True
                cv2.imwrite(current_dir + os.sep + 'tmp' + os.sep + self.name, (255-fgmask))
            # cv2.imshow('frame', fgmask)
            # cv2.waitKey(0)

        Timeline.last_img = img_origin.copy()

class Timeline():
    font_height = 40
    last_img = None
    def __init__(self, dir):
        self.dir = dir
        self.frames = []
        for f in glob(dir):
            fr = Frame(f)
            self.frames.append(fr)
    def overview(self):
        for f in self.frames:
            print(f.name)
            if len(f.lines_offset) == 0:
                print(' no text')
            else:
                print(f.lines_offset)

