import cv2, itertools, os, math
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
    current_sid = 0
    def __init__(self, img):
        _, img_file_name = os.path.split(img)
        self.name = img_file_name
        self.time = img[-15:-4]
        self.is_new = False
        self.path_to_file = img
        self.sid = Frame.current_sid
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
        # cv2.imshow('ggg', img_bin)
        # cv2.waitKey(0)
        height, width = img_bin.shape[:2]
        self.lines_offset = get_crop_tuples(img_bin, 0, Timeline.font_height)

        # compare with previous
        if Timeline.last_img is not None:
            fgbg = cv2.BackgroundSubtractorMOG()
            fgmask = fgbg.apply(Timeline.last_img)
            fgmask = fgbg.apply(img_origin)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            # print(current_dir + os.sep + 'tmp' + os.sep + self.name)
            non_zero = cv2.countNonZero(fgmask)
            if non_zero > height*width*0.01:
            # if True:
                self.is_new = True
                cv2.imwrite(current_dir + os.sep + 'tmp' + os.sep + self.name, (255-fgmask))
                Frame.current_sid += 1
                self.sid = Frame.current_sid
            # cv2.imshow('frame', fgmask)
            # cv2.waitKey(0)

        Timeline.last_img = img_origin.copy()

class Timeline():
    font_height = 40
    font_height_calc = 0
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

    def count_font_height(self):
        # all_tuples = [t[1]-t[0] for f in f.lines_offset for f in self.frames]
        all_tuples = [t[1]-t[0] for f in self.frames for t in f.lines_offset ]
        print(all_tuples)
        print( np.mean(all_tuples) )

    def get_tail_edges(self, tail=10):
        for i in range(0, len(self.frames)-tail, 1):
            current_tail = None
            for j in range(0, tail, 1):
                img_grey = cv2.imread(self.frames[i+j].path_to_file, 0)
                img_bin = cv2.Canny(img_grey, 100, 200)
                if current_tail is None:
                    current_tail = img_bin.copy()
                current_tail = np.bitwise_and(current_tail, img_bin)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            cv2.imwrite(current_dir + os.sep + 'tmp' + os.sep +'!!'+str(i)+'.png', current_tail)
    def create_sid_canny(self):
        current_sid = 0
        current_tail = None
        current_len = 0
        prev_bin = None
        th_max, th_min = 1000, 50
        current_dir = os.path.dirname(os.path.realpath(__file__))
        for frame_number, f in enumerate(self.frames):
            img_grey = cv2.imread(f.path_to_file, 0)
            img_bin = cv2.Canny(img_grey, 100, 200)

            if prev_bin is None:
                prev_bin = img_bin.copy()
                current_tail = img_bin.copy()
                f.sid = current_sid
                continue

            th_counted = cv2.countNonZero(np.bitwise_xor(img_bin, prev_bin))
            if (th_counted > th_max) or (len(self.frames)-1 == frame_number):
                print(current_sid, f.name, th_counted)
                # cv2.imshow('xor', np.bitwise_xor(img_bin, prev_bin))
                # cv2.imshow('tail', current_tail)
                # cv2.waitKey(0)
                if current_len > 5:
                    cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep + f.name, (current_tail))

                current_sid += 1
                current_tail = img_bin.copy()
                current_len = 0
            else:
                current_tail = np.bitwise_and(current_tail, img_bin)
                current_len += 1
                if cv2.countNonZero(current_tail) < th_min:
                    current_tail = img_bin.copy()
                    current_len = 0

            f.sid = current_sid
            prev_bin = img_bin.copy()

            # cv2.imshow('now', img_bin)
            # cv2.imshow('tail', current_tail)
            # cv2.imshow('bitwise_or', np.bitwise_or(img_bin, current_tail))
            # cv2.imshow('bitwise_and', np.bitwise_and(img_bin, current_tail))
            # cv2.imshow('bitwise_xor', np.bitwise_xor(img_bin, current_tail))
            # cv2.imshow('and and', np.left_shift(img_bin, current_tail))
            # cv2.waitKey(0)

