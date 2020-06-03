# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2

#处理解析行 给定一个训练集/测试文本 返回解析 图片路径 边界框信息 和 标签信息
def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed
    pic_path, boxes info, and label info.
    return:
        pic_path: string.  图片路径：字符型
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
            box：[N,4]，N是边界框的数目，第二维度是边界框坐标[x_min,y_min,x_max,y_max]
    '''
    #按空格进行拆分，并删除空白符（包括'\n', '\r',  '\t',  ' ')  ,返回一个列表
    line = line.decode()
    s = line.strip().split(' ')
    pic_path = s[0] #第一个元素为图片路径
    s = s[1:]  #除去图片路径的所有元素，包括box坐标和标签
    box_cnt = len(s) // 5  #每一个box对应四个坐标和一个标签  box_cnt表示边界框的数目
    boxes = []
    labels = []
    for i in range(box_cnt):
        #将每个边界框的坐标和标签分别加入到boxes列表和labels列表
        label, x_min, y_min, x_max, y_max = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels


def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        #将灰度图像转化为RGB图像
        img = np.expand_dims(img, -1)
    #原始图像大小
    ori_height, ori_width = img.shape[:2]
    #调整后的图像尺寸
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))

    # convert to float
    img = np.asarray(img, np.float32)

    # boxes
    # xmin, xmax 调整边界框坐标使之对应新的图像大小
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    # ymin, ymax
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes


def data_augmentation(img, boxes, label):
    '''
    Do your own data augmentation here.
    param:
        img: a [H, W, 3] shape RGB format image, float32 dtype
        boxes: [N, 4] shape boxes coordinate info, N is the ground truth box number,
            4 elements in the second dimension are [x_min, y_min, x_max, y_max], float32 dtype
        label: [N] shape labels, int64 dtype (you should not convert to int32)
    '''
    return img, boxes, label


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    '''
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class]
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    #[26,26,3,5+num_class]
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)
    #[52,52,3,5+num_class]
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 5 + class_num), np.float32)

    y_true = [y_true_13, y_true_26, y_true_52]

    # 将[N,2]->[N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    #真实框和锚点框的IOU
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    # [N]  九个框中最佳匹配框的索引值
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 2
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        #np.ceil()计算大于等于该值的最小整数
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        #将真实边界框缩中心坐标放到最佳锚点框的尺寸
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print feature_map_group, '|', y,x,k,c

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5+c] = 1.

    return y_true_13\
        , y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode):
    '''
    param:
        line: a line from the training/test txt file
        args: args returned from the main program
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    pic_path, boxes, labels = parse_line(line)
    #通过图片路径读取图片
    img = cv2.imread(pic_path)
    #图像转化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)

    # do data augmentation here
    #若为训练，则进行数据增强
    if mode == 'train':
        img, boxes, labels = data_augmentation(img, boxes, labels)

    # the input of yolo_v3 should be in range 0~1】
    #将输入图像像素转化为0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img, y_true_13, y_true_26, y_true_52
