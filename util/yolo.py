# coding:UTF-8
from ctypes import *
import math
import random
import cv2
import glob
import os
from copy import copy
import time
import numpy as np
import uuid

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
class CstmYolo:
    libname = os.path.join(os.path.dirname(__file__),"libdarknet.so")
    lib = CDLL(libname, RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]


    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    free_net = lib.free_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)


    def __init__(self,cp_cfg, cp_weights, cp_data, nf_cfg, nf_weights, nf_data, gpu=0):
        self.cache_dir = "cache2"
        #cp car-person
        CstmYolo.set_gpu(gpu) #GPUのメモリが足りないときは分散する
        CstmYolo.net_area1 = self.load_net(cp_cfg.encode('utf-8'), cp_weights.encode('utf-8'), 0)
        CstmYolo.meta_area1 = self.load_meta(cp_data.encode('utf-8'))
        # set_gpu(1) #GPUのメモリが足りないときは分散する
        #nf number-face
        CstmYolo.net_area2 = self.load_net(nf_cfg.encode('utf-8'), nf_weights.encode('utf-8'), 0)
        CstmYolo.meta_area2 = self.load_meta(nf_data.encode('utf-8'))
    
                
        # 分割した画像を出力するキャッシュディレクトリの作成
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)


    def __detect(self, net, meta, image, thresh=.3, hier_thresh=.1, nms=.45):
        im = CstmYolo.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        CstmYolo.predict_image(net, im)
        dets = CstmYolo.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): CstmYolo.do_nms_obj(dets, num, meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        CstmYolo.free_image(im)
        CstmYolo.free_detections(dets, num)
        return res


    def __div_detect(self, file_path, net, meta, div_num=4):
        #画像を分割して検出する
        if div_num > 1:
            div_num = div_num / 2
        
        cache_dir = "cache"

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        img = cv2.imread(file_path)
        
        o_w, o_h, chanel = img.shape
        cache_imgs = []
        div_h = int(o_h/div_num)
        div_w = int(o_w/div_num)
        slide = 0.5
        for y_div in range(int(div_num/slide) - 1):
            for x_div in range(int(div_num/slide) - 1):
                div_left = int(o_w / div_num * x_div * slide)
                div_right = div_left + div_w
                div_top = int(o_h / div_num * y_div * slide)
                div_bottom = div_top + div_h
                
                cache_imgs.append([img[div_left:div_right, div_top:div_bottom],(div_left,div_top)])

        boxs = []
        for i, cache_img in enumerate(cache_imgs):
            cache_file = os.path.join(cache_dir, "{}.jpg".format(uuid.uuid4()))
            cv2.imwrite(cache_file, cache_img[0])
            results = self.__detect(net, meta, cache_file.encode('utf-8'))
            # 予測したキャッシュ画像は削除する
            os.remove(cache_file)
        
            #座標だけ取得して再計算する
            for result in results:
                name = result[0]
                c = result[1]
                c_x = result[2][0]
                c_y = result[2][1]
                w = result[2][2]
                h = result[2][3]
                
                left = int(c_x - w/2) + cache_img[1][1]
                right = int(c_x + w/2) + cache_img[1][1]
                top = int(c_y - h/2) + cache_img[1][0]
                bottom = int(c_y + h/2) + cache_img[1][0]
                boxs.append([left,top, right, bottom, name, c])

        return boxs


    # left, top, right, bottomの座標から中心座標と幅、高さに変換する
    def __get_yolo_box(self, box):
        left, top, right, bottom = box[0:4]
        c_x = (left + right) / 2
        c_y = (top + bottom) / 2
        w = abs(right - left)
        h = abs(bottom - top)
        return (c_x, c_y, w, h)


    #　面積を求める
    def __get_area(self, box):
        left, top, right, bottom = box
        w = right - left
        h = bottom - top
        return w * h


    # 重なっているかどうか
    def __is_over_lap(self, src_box, dst_box):
        over_lap = False
        src_box = self.__get_yolo_box(src_box)
        dst_box = self.__get_yolo_box(dst_box)
        x_distance = abs(src_box[0] - dst_box[0])
        y_distance = abs(src_box[1] - dst_box[1])
        x_range = abs((src_box[2] + dst_box[2])/2)
        y_range = abs((src_box[3] + dst_box[3])/2)
        if x_distance < x_range:
            if y_distance < y_range:
                # 重なっている
                over_lap = True
        return over_lap


    # 重なっている矩形をマージする
    def __merge_area(self, boxs):
        if len(boxs) == 0:
            return boxs
        rtn_box = []
    #    boxs = sorted(boxs, key=lambda x:x[0]*x[1])

        not_over_lap_boxs = boxs    
        while len(not_over_lap_boxs) > 0:
            src_box = not_over_lap_boxs[0]
            tmp_box_list = []
            not_over_lap_boxs = not_over_lap_boxs[1:]
            for i, dst_box in enumerate(not_over_lap_boxs):
                if self.__is_over_lap(src_box, dst_box):
                    # 重なっている
                    left   = int(src_box[0] if src_box[0] < dst_box[0] else dst_box[0])
                    top    = int(src_box[1] if src_box[1] < dst_box[1] else dst_box[1])
                    right  = int(src_box[2] if src_box[2] > dst_box[2] else dst_box[2])
                    bottom = int(src_box[3] if src_box[3] > dst_box[3] else dst_box[3])
                    src_box = [left, top, right, bottom]
                else:
                    tmp_box_list.append(dst_box)
     
            rtn_box.append(src_box)
            not_over_lap_boxs = copy(tmp_box_list)

                
        return rtn_box


    def __correct_box(self, box):
        # 予測した結果にマイナスが含まれることがあるので、
        # マイナスは0に補正する
        if box[0] < 0:
            box[0] = 0
        if box[1] < 0:
            box[1] = 0
        if box[2] < 0:
            box[2] = 0
        if box[3] < 0:
            box[3] = 0
        
        return box
    

    def __get_world_boxs(self, crop_box, box):
        o_left = crop_box[0] + box[0]
        o_top = crop_box[1] + box[1]
        o_right = crop_box[2] + box[0]
        o_bottom = crop_box[3] + box[1]
        
        return o_left, o_top, o_right, o_bottom


    def Blur(self, file_path, output_path, options):
        box_list = []
        file_box_list = {}
        img = cv2.imread(file_path)
        blur_img = copy(img)
        #先にぼかした画像を作成する
        blur_img = cv2.blur(blur_img, (50, 50))       
        mask_img = np.ones(img.shape)
        
        #画像１枚に対して予測
        boxs = self.__div_detect(file_path, CstmYolo.net_area1, CstmYolo.meta_area1, div_num=1)
        box_list.extend(boxs)
        boxs = self.__div_detect(file_path, CstmYolo.net_area2, CstmYolo.meta_area2, div_num=1)
        box_list.extend(boxs)
        
        #画像を４分割して予測
        boxs = self.__div_detect(file_path, CstmYolo.net_area1, CstmYolo.meta_area1, div_num=4)
        box_list.extend(boxs)
        boxs = self.__div_detect(file_path, CstmYolo.net_area2, CstmYolo.meta_area2, div_num=4)
        box_list.extend(boxs)
        #画像を８分割して予測。スピード重視なら8分割の処理はなくてもよい
        boxs = self.__div_detect(file_path, CstmYolo.net_area1, CstmYolo.meta_area1, div_num=8)
        box_list.extend(boxs)
        boxs = self.__div_detect(file_path, CstmYolo.net_area2, CstmYolo.meta_area2, div_num=8)
        box_list.extend(boxs)
        # 画像ファイル名ごとに複数の予測結果を管理する
        file_box_list[file_path] = box_list



        print("%sを予測します" % (file_path))
        img = cv2.imread(file_path)
        boxs = file_box_list[file_path]
        car_boxs = []
        person_boxs = []
        
        # 予測した結果が車、人、バイクだった場合、領域マージの対象にする
        for box in boxs:
            if options["car_op"] == 1:
                if box[4] == b'car' or box[4] == b'number' or b'bike':
                    car_boxs.append(box)
            if options["face_op"] == 1:
                if box[4] == b'person' or box[4] == b'face':
                    person_boxs.append(box)

        all_boxs = [car_boxs, person_boxs]
        
        for i, boxs in enumerate(all_boxs):
            boxs = self.__merge_area(boxs)

            for j, box in enumerate(boxs):
                if i == 0:
                    color = (0, 0, 255)
                elif i == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cache_file_name = os.path.join(self.cache_dir, "%s.jpg" % (uuid.uuid4(),))
                # 矩形でマイナス座標が返ることがあるので0に補正する
                box = self.__correct_box(box)
                
                # 矩形領域の画像を切り出す
                crop_img = copy(img[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
                cv2.imwrite(cache_file_name, crop_img)
                cache_img = cv2.imread(cache_file_name)

                # 切り出した画像に対して予測する
                crop_boxs = self.__div_detect(cache_file_name, CstmYolo.net_area2, CstmYolo.meta_area2, div_num=1)
                for crop_box in crop_boxs:
                    # 切り出した画像で検出した領域をメインの画像座標に変換する
                    o_left, o_top, o_right, o_bottom = self.__get_world_boxs(crop_box, box)

                  #  cv2.rectangle(crop_img, (int(crop_box[0]), int(crop_box[1])), (int(crop_box[2]), int(crop_box[3])), color, thickness = 2)
                  #  cv2.imwrite(cache_file_name, crop_img)
                  #  cv2.rectangle(img, (int(o_left), int(o_top)), (int(o_right), int(o_bottom)), color, thickness = 2)

                    #マスクに消す領域を作成する
                    c_x = int(abs((o_left + o_right) / 2))
                    c_y = int(abs((o_top + o_bottom) / 2))
                    w = int(abs((o_left - o_right)))
                    h = int(abs((o_top - o_bottom)))
                    if w > h:
                        rad = int(w / 2)
                    else:
                        rad = int(h / 2)
                        
                    #cv2.circle(mask_img, (c_x, c_y), rad, (0,0,0), thickness=-1)
                    cv2.rectangle(mask_img, (int(o_left), int(o_top)), (int(o_right), int(o_bottom)), (0,0,0), thickness = 50)
                    cv2.rectangle(mask_img, (int(o_left), int(o_top)), (int(o_right), int(o_bottom)), (0,0,0), thickness = -1)
                    #print((c_x, c_y))
                    
                    # 該当領域を切り出す
                    #blur_area = img[o_top:o_bottom, o_left:o_right]
                    
                    # 該当領域をぼかす
                    #print("{},{},{},{}".format(o_left, o_right, o_top,o_bottom))
                    #blur_area = cv2.blur(blur_area, (int((o_right - o_left)/2), int((o_bottom - o_top)/2)))
                    
                    #ぼかした画像を元画像に戻す
                    #img[o_top:o_bottom, o_left:o_right] = blur_area
                os.remove(cache_file_name)
            

        # マスクと合成する
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):                
                if mask_img[y][x][0] == 0 and mask_img[y][x][1] == 0 and mask_img[y][x][2] == 0:
                    img[y][x] = blur_img[y][x]
        cv2.imwrite(output_path, img)
