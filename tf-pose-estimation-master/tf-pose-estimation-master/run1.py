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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/smoke.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    #os.system('python /home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/test_single_image.py' + args.image)
    # with open(OD_PATH,"r") as f:
    #     od_list = f.read().split("\n")
    #     for i in range(len(od_list)):
    #         od_list[i] = [int(float(j)) for j in od_list[i].split(",")]
    #     f.close()
    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
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
    count = 0
    for human in humans:
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
            #cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
        print('keypoint:', keypoint)
        keypoint.setdefault(6)
        keypoint.setdefault(16)
        if keypoint[6] and keypoint[16]:
            print('keypoint[6]:',keypoint[6])
            print('keypoint[16]:',keypoint[16])
            cilp_img = npimg[keypoint[16][1]:keypoint[6][1], keypoint[16][0]:keypoint[6][0]]
            img_name = 'cilp_img_'+str(count)+ '.jpg'
            cv2.imwrite(img_name, cilp_img)
            #os.system('python /home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/test_single_image.py clip_img_.jpg')
            model = YoloV3('./smoke1_frozen_graph_batch.pb')
            boxes_, labels_, scores_ = model.run(img_name)
            print('boxes:',boxes_)
            classes = read_class_names("/home/dp/shuju/linhaihua3/Smoke/Smoke/YOLOv3_TensorFlow-master/data/my_data/data.names")
            color_table = get_color_table(80)
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(cilp_img, [x0, y0, x1, y1],
                             label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                             color=color_table[labels_[i]])
            # out_name = os.path.join('./data/demo_data/results', 'batch_output_' + os.path.basename(files[idx]))
            out_name = os.path.join('batch_output_' + str(count)+'.jpg')
            cv2.imwrite(out_name,cilp_img)
            count+=1
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