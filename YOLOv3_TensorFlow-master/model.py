# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf
slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer

class yolov3(object):

    def __init__(self, class_num, anchors, batch_norm_decay=0.9,use_static_shape=False):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
                         # [30, 61], [62, 45], [59,  119],
                         # [116, 90], [156, 198], [373,326]]
        #类别数目
        self.class_num = class_num
        self.anchors = anchors
        #平滑指数衰减，衰减指数0.9
        self.batch_norm_decay = batch_norm_decay
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]图像大小为 【高，宽】
        # it will be used later
        #inputs[m,height,weight,3],img_size[height,weight]
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        #归一化参数
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }
        #设置卷积、批次归一化默认参数
        with slim.arg_scope([slim.conv2d, slim.batch_norm],reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('darknet53_body'):
                    #darknet53网络框架
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    #顶层特征图13*13*
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list()if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up! 网格大小 13*13
        grid_size = feature_map.shape.as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]  # [13, 13]
        # the downscale ratio in height and weight  缩放比例 [32,32]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format! 对锚点框进行缩放 锚点框是[宽，高]
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])


        #将特征图沿着最后的维度进行切分
        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates 使用广播技巧得到网格坐标
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y) #grid_x[13,13] , grid_y [13,13]
        x_offset = tf.reshape(grid_x, (-1, 1)) #(169,1)
        y_offset = tf.reshape(grid_y, (-1, 1)) #(1,169)
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2] 网格坐标 每个特征点对应一个（x，y）
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        #相对于特征图的边界框中心坐标
        # get the absolute box coordinates on the feature_map 
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale #将边界框缩放到原始图像大小
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        #tf.clip_by_value(A,min,max):输入一个张量A，把A中的每一个元素值都压缩在min和max之间。
        #对应论文公式 求bx = exp(box_size)*anchor_size
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # rescale to the original image scale 将其缩放到原始尺寸
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h) 最后将中心点坐标和长宽连接
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        #所对应的三个尺寸的特征图
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        #对每个特征图的输出进行处理
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        #将每个特征图进行重构的子函数
        def _reshape(result):
            #网格坐标 边界框坐标 置信度 类别
            x_y_offset, boxes, conf_logits, prob_logits = result
            #网格大小
            grid_size = x_y_offset.shape.as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4] #边界框坐标
            # conf_logits: [N, 13*13*3, 1] #置信度
            # prob_logits: [N, 13*13*3, class_num] #类别
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:#取每个特征图的预测结果
            boxes, conf_logits, prob_logits = _reshape(result) #每个特征图结果进行重构处理
            confs = tf.sigmoid(conf_logits) #置信度通过sigmoid处理
            probs = tf.sigmoid(prob_logits)#通过sigmoid函数处理 可以得到多标签
            boxes_list.append(boxes)  #将三个特征图的边界框坐标放入一个列表[N,13*13*3,4],[N,26*26*3,4] [N,52*52*3,4]
            confs_list.append(confs)  #将三个特征图的置信度放入一个列表[N,13*13*3,1],[N,26*26*3,1] [N,52*52*3,1]
            probs_list.append(probs)  #将三个特征图的类别放入一个列表[N,13*13*3,num_class],[N,26*26*3,num_class)] [N,52*52*3,num_class]
        
        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        #通过中心点坐标和长宽 转化为 左上角坐标和右下角坐标
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        计算某个特征图尺寸的损失函数
        calc loss function from a certain scale
        '''
        
        # size in [h, w] format! don't get messed up! #特征图尺寸[高，宽][N,13,13,5+num_class]
        grid_size = tf.shape(feature_map_i)[1:3] #[h,w]->[13,13]
        # the downscale ratio in height and weight #特征图高宽相对于特征图缩放的比例 [32,16,8]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)
        #每个特征图的预测
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    
    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]



    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou
