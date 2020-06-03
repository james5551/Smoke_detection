# coding: utf-8

from __future__ import division, print_function

import cv2
import random

#定义颜色表函数  每个类别定义一个颜色
def get_color_table(class_num, seed=2):
    random.seed(seed)
    #定义字典
    color_table = {}
    for i in range(class_num):
        #在0~255正整数中选取三个数  作为三个颜色通道
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

#定义画框函数
def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    #线条粗细
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    #线条颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    #边界框坐标
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

