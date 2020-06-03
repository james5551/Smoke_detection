import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import collections
import os
import sys
from test_batch_images_1 import YoloV3
from utils.plot_utils import get_color_table, plot_one_box
from utils.misc_utils import parse_anchors, read_class_names
sys.path.append('../')
logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# OD_PATH ='./person.txt'
# smoke_PATH = './smoke.txt'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/019.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    #os.system('python /home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/test_person.py ' + args.image)
    img_ori = cv2.imread(args.image)
    cv2.imwrite('img_ori.jpg', img_ori)
    print(img_ori.shape[:2])
    height_ori, width_ori = img_ori.shape[:2]
    # classes = read_class_names("/home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/data/coco.names")
    # color_table = get_color_table(80)
    person_model = YoloV3('./weight_yolov3.pb')
    boxes, labels, scores = person_model.run(args.image)
    index = [i for i in range(len(labels)) if labels[i] == 0]
    box_person = []
    for i in range(len(index)):
        box_person.append(list(boxes[index[i]]))
    # for i in range(len(box_person)):
    #     x0, y0, x1, y1 = box_person[i]
    #     plot_one_box(img_ori, [x0, y0, x1, y1],
    #                  label=classes[labels[0]] + ', {:.2f}%'.format(scores[i] * 100),
    #                  color=color_table[labels[0]])
    # cv2.imwrite('final_result.jpg', img_ori)
    print('=====================person test finish!=======================')
    print('box_person')
    print(box_person)
    # with open(OD_PATH,"r") as f:
    #     od_list = f.read().split("\n")
    #     for i in range(len(od_list)):
    #         od_list[i] = [int(float(j)) for j in od_list[i].split(",")]
    #     f.close()
    # print('od_list:',od_list)
    count = 0
    smoke_boxes = []
    smoke_scores = []
    for i in range(len(box_person)):
        img = np.copy(img_ori)
        person_left,person_up = box_person[i][0],box_person[i][1]
        print('person_left:',person_left)
        print('person_up:',person_up)
        crop_img = img[int(box_person[i][1]):int(box_person[i][3]), int(box_person[i][0]):int(box_person[i][2])]
        #cv2.imwrite('crop_img.jpg',crop_img)
        image = np.copy(crop_img)
        image_name = './crop_' + str(i) + '.jpg'
        cv2.imwrite(image_name, crop_img)
        # os.system('python /home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/test_single_image.py ' + image_name)
        model = YoloV3('./smoke1_frozen_graph_batch.pb')
        boxes_, labels_, scores_ = model.run(image_name)
        print('*'*30)
        print('boxes_:')
        print(boxes_)
        # new_size = [416, 416]
        # img_ori = cv2.imread(image_name)
        # height_ori, width_ori = img_ori.shape[:2]
        # img = cv2.resize(img_ori, tuple(new_size))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.asarray(img, np.float32)
        # images = img[np.newaxis, :] / 255.
        print('=============smoke test finish!=========================')
        # with open(smoke_PATH, "r") as f:
        #     smoke_list = f.read(
        #     if smoke_list:
        #         smoke_list = smoke_list.split("\n")
        #         for i in range(len(smoke_list)):
        #             smoke_list[i] = [int(float(j)) for j in smoke_list[i].split(",")]
        #         f.close()
        if len(boxes_)>0:
            w, h = model_wh(args.resize)
            if w == 0 or h == 0:
                e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
            else:
                e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
            # estimate human poses from a single image !
            # image = common.read_imgfile(args.image, None, None)
            if image is None:
                logger.error('Image can not be read, path=%s' % args.image)
                sys.exit(-1)
            t = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            elapsed = time.time() - t
            logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
            #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            #cv2.imwrite('result.jpg', image)
            npimg = np.copy(image)
            image_h, image_w = npimg.shape[:2]
            centers = {}
            print(humans)
            human = humans[0]
            #for human in humans:
            print('human:', human)  #
            keypoint = collections.defaultdict()
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                keypoint[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
            print('keypoint:', keypoint)
            keypoint.setdefault(0)#
            keypoint.setdefault(16)
            for j in range(len(boxes_)):
                print('smoke_box:',boxes_[j])
                smoke_left = boxes_[j][0]
                smoke_up = boxes_[j][1]
                print('smoke_left:',smoke_left)
                print('smoke_up:',smoke_up)
                if keypoint[0]  and abs(keypoint[0][0]-smoke_left)<50 and abs(keypoint[0][1]-smoke_up)<50 :
                    #cilp_img = npimg[keypoint[16][1]:keypoint[6][1], keypoint[16][0]:keypoint[6][0]]
                    img_name = 'cilp_img_'+str(count)+ '.jpg'
                    cv2.circle(npimg, keypoint[0], 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
                    cv2.rectangle(npimg,(int(boxes_[j][0]),int(boxes_[j][1])),(int(boxes_[j][2]),int(boxes_[j][3])),(0,0,255), 2)
                    print('Existing Smoke!')
                    cv2.imwrite(img_name, npimg)
                    boxes_[j][0]+=person_left
                    boxes_[j][2]+=person_left
                    boxes_[j][1]+=person_up
                    boxes_[j][3]+=person_up
                    smoke_boxes.append(boxes_[j])
                    smoke_scores.append(scores_[j])
                    count += 1

    classes = {0:'smoke'}
    color_table = get_color_table(1)
    for i in range(len(smoke_boxes)):
        print('change_box:',smoke_boxes[i])
        x0, y0, x1, y1 = smoke_boxes[i]
        plot_one_box(img_ori, [x0, y0, x1, y1],
                     label='smoke' + ', {:.2f}%'.format(smoke_scores[i] * 100),
                     color=color_table[0])
    cv2.imwrite('final_result.jpg', img_ori)

        # try:
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure()
    #     a = fig.add_subplot(2, 2, 1)
    #     a.set_title('Result')
    #     #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     img1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #     cv2.imwrite('./save/img1.jpg',img1)
    #
    #     bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
    #     cv2.imwrite('./save/bgimg.jpg', bgimg)
    #     # show network output
    #     a = fig.add_subplot(2, 2, 2)
    #     #plt.imshow(bgimg, alpha=0.5)
    #
    #     tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    #     #plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    #     cv2.imwrite('./save/tmp.jpg',tmp)
    #     plt.colorbar()
    #
    #     tmp2 = e.pafMat.transpose((2, 0, 1))
    #     tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    #     tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
    #
    #     a = fig.add_subplot(2, 2, 3)
    #     a.set_title('Vectormap-x')
    #     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #     #plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    #     cv2.imwrite('./save/tmp2_odd.jpg',tmp2_odd)
    #     plt.colorbar()
    #
    #     a = fig.add_subplot(2, 2, 4)
    #     a.set_title('Vectormap-y')
    #     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #     #plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    #     plt.colorbar()
    #     cv2.imwrite('./save/temp2_even.jpg',tmp2_even)
    #     #plt.show()
    # except Exception as e:
    #     logger.warning('matplitlib error, %s' % e)
    #     # cv2.imshow('result', image)
    #     # cv2.waitKey()
    #     cv2.imwrite('result.jpg',image)
