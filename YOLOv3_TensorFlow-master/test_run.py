# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3
class Test:
    def __init(self):
        self.anchor_path = "./data/yolo_anchors.txt"
        self.new_size = [416, 416]
        self.class_name_path = "./data/my_data/data.names"
        self.restore_path = "./checkpoint/model-step_17500_loss_0.003654_lr_0.0004995866"
        self.anchors = parse_anchors(self.anchor_path)
        self.classes = read_class_names(self.class_name_path)
        self.num_class = len(self.classes)
        self.color_tabel = get_color_table(self.num_class)
        self.img_ori = cv2.imread(self.input_image)
        self.weight_ori,self.width_ori = self.img_ori.shape[:2]
        self.img = cv2.resize(img_ori,tuple(self.new_size))
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.img = np.asarray(self.img,np.float32)
        self.img = self.img[np.newaxis,:]/255.
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

        yolo_model = yolov3(self.num_class, self.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs
        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=100, score_thresh=0.4,
                                        iou_thresh=0.5)
        self.__saver = tf.train.Saver()
        self.__saver.restore(self.__sess, self.restore_path)
        self.input_data = tf.placeholder(tf.float32, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
    def inference(self,input_image):
        boxes_, scores_, labels_ = self.__sess.run([self.boxes, self.scores, self.labels], feed_dict={self.input_data:input_image })

        # rescale the coordinates to the original image
        boxes_[:, 0] *= (width_ori / float(args.new_size[0]))
        boxes_[:, 2] *= (width_ori / float(args.new_size[0]))
        boxes_[:, 1] *= (height_ori / float(args.new_size[1]))
        boxes_[:, 3] *= (height_ori / float(args.new_size[1]))

        boxes_[:, 0] = np.ceil(boxes_[:, 0])
        boxes_[:, 2] = np.ceil(boxes_[:, 2])
        boxes_[:, 1] = np.ceil(boxes_[:, 1])
        boxes_[:, 3] = np.ceil(boxes_[:, 3])

        print("box coords:")
        print(boxes_)
        # f = open('C:/Users/Mr Lin/Desktop/YOLOv3_TensorFlow-master/YOLOv3_TensorFlow-master/data/test.txt', 'r+')
        # f.write(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
        cv2.imwrite('detection_result.jpg', img_ori)
        # cv2.waitKey(0)
if __name__ == '__main__':
    a = Test()
    a.inference('./data/test_image/7.jpg')