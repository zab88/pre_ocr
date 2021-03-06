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

def get_out_name(movie_name, frame_start, frame_end, fps):
    start = int(float(frame_start)/fps)
    end = int(float(frame_end)/fps)
    start_h = start/3600
    start_m = (start - start_h*3600)/60
    start_s = start%60
    end_h = end/3600
    end_m = (end - end_h*3600)/60
    end_s = end%60
    out_name = '{:0>}-{}h{:0>2}m{:0>2}s-{}h{:0>2}m{:0>2}s.png'.format(movie_name, start_h, start_m, start_s, end_h, end_m, end_s)
    return out_name

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
                    # lets' add color !
                    img_bgr = cv2.imread(self.frames[frame_number-1].path_to_file)
                    mask_color = Timeline.get_color_mask(img_bgr)
                    kernel_3 = np.ones((3, 3), np.uint8)
                    dilation = cv2.dilate(current_tail, kernel_3)
                    wow = np.bitwise_and(dilation, mask_color)
                    cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep +'_'+ f.name, (255-wow))

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

    @staticmethod
    def get_color_mask(img_bgr):
        most_color = 230
        color_threshold = 25
        lower = np.array([0, 0, max(0, most_color-color_threshold)])
        upper = np.array([250, 250, min(255, most_color+color_threshold)])
        hsv = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask

class FromVideo():
    def __init__(self):
        pass

    @staticmethod
    def create_sid_canny():
        current_sid = 0
        current_tail = None
        current_len = 0
        prev_bin = None
        prev_color = None
        th_max, th_min = 1000, 50
        current_dir = os.path.dirname(os.path.realpath(__file__))
        frame_number = 0

        cap = cv2.VideoCapture('movies/LanLing26.mp4')
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = frame[418:454, 76:776]
            frame_number += 1

            img_grey = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            img_bin = cv2.Canny(img_grey, 100, 200)

            if prev_bin is None:
                prev_bin = img_bin.copy()
                prev_color = frame.copy()
                current_tail = img_bin.copy()
                continue

            th_counted = cv2.countNonZero(np.bitwise_xor(img_bin, prev_bin))
            if (th_counted > th_max):
                print(frame_number, th_counted)
                if current_len > 8:
                    #cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep + frame_number+'.png', (current_tail))
                    # lets' add color !
                    img_bgr = prev_color.copy()
                    mask_color = Timeline.get_color_mask(img_bgr)
                    kernel_3 = np.ones((3, 3), np.uint8)
                    dilation = cv2.dilate(current_tail, kernel_3)
                    wow = np.bitwise_and(dilation, mask_color)
                    if cv2.countNonZero(wow) > (th_min*3 - 5):
                        cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep +str(frame_number)+'.png', (255-wow))

                current_sid += 1
                current_tail = img_bin.copy()
                current_len = 0
            else:
                current_tail = np.bitwise_and(current_tail, img_bin)
                current_len += 1
                if cv2.countNonZero(current_tail) < th_min:
                    current_tail = img_bin.copy()
                    current_len = 0

            prev_bin = img_bin.copy()
            prev_color = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

class FromVideo2():
    def __init__(self):
        pass

    @staticmethod
    def create_sid_canny():
        current_sid = 0
        current_tail = None
        current_tail2 = None
        current_len = 0
        prev_bin = None
        prev_color = None
        th_max, th_min = 10000, 500
        current_dir = os.path.dirname(os.path.realpath(__file__))
        frame_number = 0
        frame_number_start = 0

        text_lower = np.array([0, 0, 125])
        text_upper = np.array([200, 200, 255])
        border_lower = np.array([0, 160, 0])
        border_upper = np.array([20, 255, 100])
        # border_lower = np.array([0, 0, 0])
        # border_upper = np.array([255, 255, 121])

        cap = cv2.VideoCapture('movies/Xiang35.mp4')
        fps = cap.get(5)
        # for i in range(0, 18, 1):
        #     print(cap.get(i))
        # cap.release()
        # exit()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is False:
                # video ended
                print(frame_number)
                break
            frame = frame[340:460, 50:590]
            frame_number += 1
            if frame_number < 4800:
                continue
            # frame = cv2.blur(frame, (5, 5))
            # cv2.imshow('ggg', frame)
            # # cv2.waitKey(0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue

            img_grey = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            img_bin = cv2.Canny(img_grey, 100, 200)
            img_bin2 = cv2.Canny(img_grey, 100, 300)

            if prev_bin is None:
                prev_bin = img_bin.copy()
                prev_color = frame.copy()
                current_tail = img_bin.copy()
                current_tail2 = img_bin2.copy()
                hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
                border_mask_tail = cv2.inRange(hsv, border_lower, border_upper)
                text_mask_tail = cv2.inRange(hsv, text_lower, text_upper)
                continue

            hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
            border_mask = cv2.inRange(hsv, border_lower, border_upper)
            text_mask = cv2.inRange(hsv, text_lower, text_upper)

            fgbg = cv2.BackgroundSubtractorMOG()
            fgmask = fgbg.apply(prev_color)
            fgmask = fgbg.apply(frame)

            th_counted = cv2.countNonZero(fgmask)
            if (th_counted > th_max):
                print(frame_number_start, frame_number, th_counted)
                if current_len > 8:
                    # magic = cv2.subtract(text_mask_tail, border_mask_tail)
                    # magic = cv2.copyMakeBorder(magic, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255))
                    # cv2.floodFill(magic, None, (0, 0), (0))
                    # cv2.imshow('asdf', magic)
                    # cv2.waitKey(0)

                    kernel_9 = np.ones((9, 9), np.uint8)
                    super_board = cv2.add(border_mask_tail, current_tail2)
                    perfect_board = np.bitwise_and(current_tail2, cv2.dilate(current_tail, kernel_9))
                    perfect_board = cv2.copyMakeBorder(perfect_board, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0))
                    cv2.floodFill(perfect_board, None, (0, 0), (255))
                    cv2.imshow('board', border_mask_tail)
                    cv2.imshow('edges', current_tail)
                    cv2.imshow('edges2', current_tail2)
                    cv2.imshow('super_board', super_board)
                    cv2.imshow('perfect_board', perfect_board)
                    cv2.imshow('fgmask', fgmask)
                    cv2.imshow('text_mask_tail', text_mask_tail)
                    cv2.waitKey(0)

                    #cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep + frame_number+'.png', (current_tail))
                    # lets' add color !
                    # img_bgr = prev_color.copy()
                    # mask_color = Timeline.get_color_mask(img_bgr)
                    kernel_3 = np.ones((3, 3), np.uint8)
                    kernel_5 = np.ones((5, 5), np.uint8)
                    dilation = cv2.dilate(current_tail, kernel_5)
                    # making final image
                    #wow = cv2.subtract( np.bitwise_and(text_mask_tail, dilation), border_mask_tail)
                    wow = cv2.subtract(text_mask_tail, border_mask_tail)
                    wow = cv2.subtract(wow, (255-dilation))
                    # wow = np.bitwise_and(dilation, mask_color)
                    if cv2.countNonZero(wow) > (th_min*3 - 5):
                        if cv2.countNonZero(wow) < 30000:
                            cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep +
                                        get_out_name('Xiang', frame_number_start, frame_number, fps), (255-wow))
                            cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep +str(frame_number)+'_text.png', text_mask_tail)
                            cv2.imwrite(current_dir + os.sep + 'tmp2' + os.sep +str(frame_number)+'_super_board.png', super_board)

                current_sid += 1
                current_tail = img_bin.copy()
                current_tail2 = img_bin2.copy()
                text_mask_tail = text_mask
                border_mask_tail = border_mask
                current_len = 0
                frame_number_start = frame_number
            else:
                current_tail = np.bitwise_and(current_tail, img_bin)
                current_tail2 = np.bitwise_or(current_tail, img_bin2)
                text_mask_tail = np.bitwise_and(text_mask, text_mask_tail)
                border_mask_tail = np.bitwise_or(border_mask, border_mask_tail)
                current_len += 1
                if cv2.countNonZero(text_mask_tail) < th_min:
                    current_tail = img_bin.copy()
                    current_tail2 = img_bin2.copy()
                    current_len = 0
                    frame_number_start = frame_number

            #prev_bin = img_bin.copy()
            prev_color = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()