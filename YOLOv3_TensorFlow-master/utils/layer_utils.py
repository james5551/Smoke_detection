# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

#步长为1的同卷积
def conv2d(inputs, filters, kernel_size, strides=1):
    #定义步长为1的同卷积的填充函数
    def _fixed_padding(inputs, kernel_size):
        #总共填充
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    #定义残差网络函数
    def res_block(inputs, filters):
        #原始输入
        shortcut = inputs
        #将原始输入 进行1x1的卷积，滤波器32
        net = conv2d(inputs, filters * 1, 1)
        #进行 3x3 同卷积，滤波器64
        net = conv2d(net, filters * 2, 3)
        #残差连接  相加
        net = net + shortcut

        return net
    
    # first two conv2d layers conv1和conv2
    #input 256x256x3  output :256x256x32
    net = conv2d(inputs, 32,  3, strides=1)
    #input 256x256x32 output:128x128x64
    net = conv2d(net, 64,  3, strides=2)

    #第一个残差网络块 128x128x64
    # res_block * 1
    net = res_block(net, 32)

    #不使用池化层  而是用trides=2 的卷积进行降维
    #input :128x128x64 output:64x64x128
    net = conv2d(net, 128, 3, strides=2)

    #两个残差网路块  input：64x64x128 output:64x64x128
    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    #降维 input：64x64x128 output:32x32x256
    net = conv2d(net, 256, 3, strides=2)

    #8个残差网络块 input:32x32x256 output:32x32x256
    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    #作为第一个feature输出 32x32x256
    route_1 = net
    #降维  16x16x512
    net = conv2d(net, 512, 3, strides=2)

    #8个残差网络块 input : 16x16x512  output:16x16x512
    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    #第二个feature map输出  16x16x512
    route_2 = net
    #降维 8x8x1024
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    #四个残差网络块 input:8x8x1024 output:8x8x1024
    for i in range(4):
        net = res_block(net, 512)
    #第三个feature map 8x8x1024
    route_3 = net

    #返回三个feature map 的输出
    return route_1, route_2, route_3

#每个feature map后 用于处理输出的网络块
def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


#上采样层
def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs


