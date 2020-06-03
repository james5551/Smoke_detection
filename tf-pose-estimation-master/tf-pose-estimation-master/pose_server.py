import argparse
import logging
import sys
import time
import time
from PIL import Image
import base64
from flask import Flask,request,jsonify
from urllib.parse import unquote
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.estimator import Human
from tf_pose.networks import get_graph_path, model_wh
import collections
import os
import sys
from utils.plot_utils import get_color_table, plot_one_box
import json
sys.path.append('../')
logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

app = Flask(__name__)

class UserEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Human):
            return obj
        return json.JSONEncoder.default(self, obj)

def get_dataDict(data):
    data_dict = {}
    for text in data.split('&'):
        key, value = text.split('=')
        value_1 = unquote(value)
        data_dict[key] = value_1
    return data_dict
@app.route('/pose_server',methods = ['POST'])
def pose_server():
    t1 = time.time()
    data_bytes = request.get_data('img')
    data = data_bytes.decode('utf-8')
    data_dict = get_dataDict(data)
    image_base64_string = data_dict['img']
    if image_base64_string:
        try:
            image_base64_bytes = image_base64_string.encode('utf-8')
            image_bytes = base64.b64decode(image_base64_bytes)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = np.asarray(img)
            ori_height,ori_width = img.shape[:2]
            print(ori_width, ori_height)
            w, h = model_wh('0x0')
            if w == 0 or h == 0:
                e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
            else:
                e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))
            # estimate human poses from a single image !
            # image = common.read_imgfile(args.image, None, None)
            if img is None:
                logger.error('Image can not be read')
                sys.exit(-1)
            t = time.time()
            humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            elapsed = time.time() - t
            npimg = np.copy(img)
            image_h, image_w = npimg.shape[:2]
            nose_list = []
            for human in humans:
                print('human:', human)
                if 0 in human.body_parts.keys():
                    body_part = human.body_parts[0]
                    nose_list.append([int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5)])
            print('inference image: in %.4f seconds.' % elapsed)
            json_dict = {'nose_list':nose_list.tolist()}
            print(json_dict)
            return jsonify(**json_dict)
        except Exception as e:
            print(e)
#主函数
if __name__ == '__main__':
    app.run('0.0.0.0', port=14000)
