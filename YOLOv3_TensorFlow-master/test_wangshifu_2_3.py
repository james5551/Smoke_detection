# coding: utf-8
# 后台运行程序 nohup python test_ssd.py >> log_ssd.out 2>&1 &
# In[1]:
from test_batch_images_1 import YoloV3
import sys
#sys.path.append('/root/liweiwei/modelDeploy/application_layer_wangshifu/')
import json
import base64
from flask import Flask, request
import numpy as np
import tensorflow as tf
import cv2
import keras
# In[2]:
from PIL import Image
#from inference.main import ImgInference
#from inference.functions import Config

# In[ ]:
from main_wangshifu_2_1 import wangshifu_inference
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
#from utils_all.image_2_base64 import image_to_base64
app = Flask(__name__)
# 汪师傅模型
#ckpt_filename = '/root/liweiwei/modelDeploy/ssd/checkpoints/ssd_300_vgg.ckpt'
#inferencer_ssd = SSDInference(ckpt_filename)


#session = tf.Session()
#keras.backend.set_session(session)
'''
with open('/root/kuaidiyuan/img139.png', 'rb') as f:
    img_base64_data = base64.b64encode(f.read())
img = img_base64_data
img = base64.b64decode(img)
img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)


#cv2转mpimg.read
b,g,r=cv2.split(img)
img_1=cv2.merge([r,g,b])
#cv2转Image.open格式
img_2 = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
result = inferencer_ssd.ssd_inference(img_1, img_2)
#print("goint to print ans_str---------------------------------")
#print(result)
'''
@app.route("/is_img_wangshifu", methods=['post'])
def is_img_wangshifu():
    res = request.get_data()
    res_json = json.loads(res)
    img = res_json['base64']
    img_base_decode = base64.b64decode(img)
    img_base = cv2.imdecode(np.frombuffer(img_base_decode, np.uint8), cv2.IMREAD_COLOR)
    #inferencer = WangShiFu(img_base)
    result = wangshifu_inference(img_base)
    #result_json = json.loads(result)
    return result


if __name__ == '__main__':
    app.run("0.0.0.0", port=12210)
#

# In[ ]:
