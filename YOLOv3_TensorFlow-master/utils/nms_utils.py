# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        iou_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')


    #因为我们在一张图片上做nms  故将boxes形状和score重构
    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    #布尔值 若score>=score_thresh 返回true  否则返回flase
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    #为每个类别做nms
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        #tf.boolean_mask(tensor,mask,name = 'boolean_mask',axis = None)
        # parms：tensor是N维度的张量，mask是k维度的，K要小于等于N,name可选项是这个操作的名字，
        #axis默认维度是0，表示从哪个维度进行mask，
        #return:N-K+1维度，也就是mask为ture的地方保存下来
        #此处表示将第i个类别为true（第i个类别的score大于指定的阈值）的boxes保存下来
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        #将第i各类别为true（）的score保存下来
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        #tf.image.non_max_suppression()非最大抑制函数
        #params:
        #return : 返回选中的那些留下来的边框在参数boxes里面的下标位置
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        #利用tf.gather()将nms保存下来的边界框的boxes的score保存为一个张量转化为标签0~num_class
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        #将保存下来的boxes放到boxes_list列表中
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        #将保存下来的score放到score_list列表中
        score_list.append(tf.gather(filter_score, nms_indices))
    #每个类别保存的边界框在第一个维度拼接
    boxes = tf.concat(boxes_list, axis=0)
    #每个类别保存的得分在第一个维度拼接
    score = tf.concat(score_list, axis=0)
    #每个类别保存的标签在第一个维度拼接
    label = tf.concat(label_list, axis=0)

    #return boxes, score, label
    return tf.identity(boxes, name='output/boxes'), tf.identity(score, name='output/scores'), tf.identity(label,name='output/labels')

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]#[n,1]
    y1 = boxes[:, 1]#[n,1]
    x2 = boxes[:, 2]#[n,1]
    y2 = boxes[:, 3]#[n,1]

    #面积  [n,]
    areas = (x2 - x1) * (y2 - y1)
    #置信度得分索引逆序排序 [n,]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #[n-1,]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        #[n-1,]
        yy1 = np.maximum(y1[i], y1[order[1:]])
        #[n-1,]
        xx2 = np.minimum(x2[i], x2[order[1:]])
        #[n-1,]
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #[n-1,]
        w = np.maximum(0.0, xx2 - xx1 + 1)
        #[n-1,]
        h = np.maximum(0.0, yy2 - yy1 + 1)
        #[n-1,]
        inter = w * h
        #[n-1,]
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #返回满足条件的下标索引
        inds = np.where(ovr <= iou_thresh)[0]
        #因为ovr为[n-1,] 故索引加1（因为需要去原来的循序表中去找对应的框）
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """
    #形状重构
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: 
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: 
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label