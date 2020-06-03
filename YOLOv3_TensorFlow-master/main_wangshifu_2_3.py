# coding: utf-8

# !/usr/bin/python
import os
import sys

sys.path.append('/root/liweiwei/modelDeploy/')
import urllib
from urllib import request
import requests
import base64
from PIL import Image
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from ctpn.inference_ctpn.resize_image import resize_image
from utils_all.rotation import rotate
from utils_all.pic_qingxidu import getImageVar
import json
from aip import AipOcr
import time
import dlib
from flask import jsonify
import pandas as pd
import sys
import utils_all.jd_sdk as jd_sdk
from io import BytesIO
import copy
from skimage import transform
import json
from flask import Flask, request
from utils_all.image_2_base64 import image_to_base64
from utils_all.humanbody_rotation import body_rotation


def wangshifu_inference(img_data):
    global contect, logo  #将base64转为Image.open格式
    # global item
    # img_base_decode = base64.b64decode(self.img_data)
    # img_base = cv2.imdecode(np.frombuffer(img_base_decode, np.uint8), cv2.IMREAD_COLOR)
    img_base = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))#数组到图像的转换
    # print(img_base) #---------------- PIL.Image.Image
    print('===================首先看图片的清晰度=====================')
    # 把图片进行resize为固定大小
    size = (450, 600)
    image_rize = img_base.resize(size, Image.ANTIALIAS)
    image_rize_base64 = image_to_base64(image_rize)
    image_rize_decode = base64.b64decode(image_rize_base64) #decode将字符串解码成二进制字符  encode:将二进制编码成字符
    image_rize_cv2 = cv2.imdecode(np.frombuffer(image_rize_decode, np.uint8), -1)
    qingxidu = getImageVar(image_rize_cv2)#判断图像模糊度
    if qingxidu < 25:
        contect = 'NO CLEAR'
        logo = '0'
        res_back = {'code': logo, 'base64': contect}
        # en_json = json.dumps(res_back)   这是在自己调用的时候使用，flask使用的是jsonify
        # en_json = jsonify(res_back)

        return jsonify(res_back)  # en_json
    else:
        print('======================使用s3fd看图片中有没有完整的人脸==============')
        #im2 = img_data
        im2 = img_base
        im2_base64 = image_to_base64(im2)
        #with open(im2, 'rb') as f:
            #img_base64_data = base64.b64encode(f.read())
        data = {"type": 'jpg',
                'img': im2_base64}
        url = "http://172.18.131.37:12320/s3fd_server"
        result = requests.post(url, data=data)
        responseJson_dict = result.json()
        box_ndarray = np.array(responseJson_dict['box_list'], np.float)
        box_list = box_ndarray.astype('int').tolist()
        print(box_list)
        score_list = responseJson_dict['score_list']
        if len(box_list) == 0:
            contect = 'HAVE NO FACE'
            logo = '1'
            res_back = {'code': logo, 'base64': contect}
            # en_json = jsonify(res_back)
            return jsonify(res_back)  # en_json
        else:
            # 一定使用的是经过alphapose判断过后的图片
            print('============使用SSD来检查图片中有几个人===============')
            test_data_1 = {
                "type": "png",
                "img": im2_base64
            }
            req_url_ssd = "http://172.17.0.1:12470/is_img_ssd"
            result_ssd = requests.post(req_url_ssd, data=test_data_1)
            print(result_ssd)
            content_ssd = result_ssd.content.decode('utf-8')
            if len(content_ssd) < 3:
                contect = 'HUMEN N0 CLEAR'
                logo = '9'
                res_back = {'code': logo, 'base64': contect}
                # en_json = jsonify(res_back)
                return jsonify(res_back)  # en_json
            else:
                list_ssd = json.loads(result_ssd.content)
                print(list_ssd)
                lenght = len(list_ssd)
                contect = []  # 得到所有图片二值化的结果
                for i in range(lenght):
                    # 第一步，尝试对SSD抽取的对象得到个人的图片--- Image.Open格式
                    up = list_ssd[i][0]
                    left = list_ssd[i][1]
                    bottom = list_ssd[i][2]
                    right = list_ssd[i][3]
                    pic_ssd = img_base.crop((up, left, bottom, right))
                    print('============Yolov3检测是否有logo======================')
                    pic_yolov3_base64 = image_to_base64(pic_ssd)
                    #pic_yolov3_base64 = base64.b64encode(pic_ssd)
                    data = {"type": 'jpg',
                            'img': pic_yolov3_base64 }
                    url = "http://172.18.131.37:12420/yolo_server"
                    result = requests.post(url, data=data)
                    responseJson_dict = result.json()
                    # box_ndarray = np.array(responseJson_dict['box_list']) / resized_multiple
                    box_ndarray = np.array(responseJson_dict['box_list'], np.float)
                    # box_ndarray = coordinates_transform(box_ndarray,ori_size,new_size=[416,416])
                    box_list = box_ndarray.astype('int').tolist()
                    classId_list = responseJson_dict['classId_list']
                    #print(classes_dict[classId_list[0]])
                    score_list = responseJson_dict['score_list']
                    #img_ori = cv2.imread(filepath)
                    if len(box_list) > 0:
                        contect = 'HUMEN HAVR LOGO'
                        logo = '99'
                        res_back = {'code': logo, 'base64': contect}
                        # en_json = jsonify(res_back)
                        print(res_back)
                        #return jsonify(res_back)  # en_json
                        continue

                    else:
                        print('==================使用CTPN把字符抽取===================')
                        pic_ssd_base64 = image_to_base64(pic_ssd)
                        image_ssd_ctpn = base64.b64decode(pic_ssd_base64)
                        image_ssd_ctpn = cv2.imdecode(np.frombuffer(image_ssd_ctpn, np.uint8), cv2.IMREAD_COLOR)
                        # print(type(image_ssd_ctpn))
                        im_ssd_ctpn = image_ssd_ctpn[:, :, ::-1]
                        img_ssd_ctpn, (rh, rw) = resize_image(im_ssd_ctpn)
                        # print(pic_ssd_base64)
                        # 第二步，代入到CTPN的模型
                        test_data_3 = {
                            "type": "jpg",
                            "img": pic_ssd_base64
                        }
                        req_url_ctpn = "http://172.17.0.1:12490/is_img_ctpn"
                        result_ctpn = requests.post(req_url_ctpn, data=test_data_3)
                        # print(result_ctpn)
                        # 得到返回的list
                        result_ctpn = json.loads(result_ctpn.content)  # print(result_ctpn)
                        # 看返回的是几张图片
                        if len(result_ctpn) == 0:
                            contect.append([])
                        else:
                            for i in range(len(result_ctpn)):
                                left = result_ctpn[i][0]
                                right = result_ctpn[i][1]
                                down = result_ctpn[i][2]
                                up = result_ctpn[i][3]
                                # 原图#---- 由numpy.array转成PIL.Image图片类型
                                array_ctpn = img_ssd_ctpn[down:up, left:right, ::-1]
                                raw_image = Image.fromarray(np.uint8(array_ctpn))
                                # print(raw_image)
                                print('===========对每张图片进行左右180度旋转(变成Image.Open格式)======')
                                out_rotation = raw_image.transpose(Image.FLIP_LEFT_RIGHT)  # ---PIL.Image.Image
                                # print(out_rotation)
                                print('============不对图片进行霍夫变换处理======================')
                                # 对原图
                                output_buffer = BytesIO()
                                img_yuantu = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
                                result_yuantu = img_yuantu
                                result_yuantu_image = Image.fromarray(np.uint8(result_yuantu))
                                result_yuantu_image_1 = result_yuantu_image
                                result_yuantu_image_1.save(output_buffer, format='JPEG')
                                yuantu_binary_data_1 = output_buffer.getvalue()
                                nohuofu_yuantu_data_1 = base64.b64encode(yuantu_binary_data_1)
                                # 对旋转后的图片
                                output_buffer = BytesIO()
                                img_rotation = cv2.cvtColor(np.array(out_rotation), cv2.COLOR_RGB2BGR)
                                result_rotation = img_rotation
                                result_rotation_image = Image.fromarray(np.uint8(result_rotation))
                                result_rotation_image_1 = result_rotation_image
                                result_rotation_image_1.save(output_buffer, format='JPEG')
                                rotation_binary_data_1 = output_buffer.getvalue()
                                nohuofu_rotate_data_1 = base64.b64encode(rotation_binary_data_1)
                                print('===========对图片进行霍夫变换处理（变成cv2.imread格式）=========')
                                # 对原图
                                output_buffer = BytesIO()
                                img_yuantu = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
                                result_yuantu = rotate(img_yuantu)
                                result_yuantu_image = Image.fromarray(np.uint8(result_yuantu))
                                result_yuantu_image_1 = result_yuantu_image
                                result_yuantu_image_1.save(output_buffer, format='JPEG')
                                yuantu_binary_data_1 = output_buffer.getvalue()
                                yuantu_base64_data_1 = base64.b64encode(yuantu_binary_data_1)
                                # 将图片放大一倍
                                output_buffer = BytesIO()
                                (x, y) = result_yuantu_image.size
                                # print(x,y)
                                x_y = x * 2
                                y_y = y * 2
                                result_yuantu_image_2 = result_yuantu_image.resize((x_y, y_y), Image.ANTIALIAS)
                                result_yuantu_image_2.save(output_buffer, format='JPEG')
                                yuantu_binary_data_2 = output_buffer.getvalue()
                                yuantu_base64_data_2 = base64.b64encode(yuantu_binary_data_2)
                                # print(type(result_yuantu_image))
                                # result_yuantu_image.save('./res1.jpg')
                                # 对旋转后的图片
                                output_buffer = BytesIO()
                                img_rotation = cv2.cvtColor(np.array(out_rotation), cv2.COLOR_RGB2BGR)
                                result_rotation = rotate(img_rotation)
                                result_rotation_image = Image.fromarray(np.uint8(result_rotation))
                                result_rotation_image_1 = result_rotation_image
                                result_rotation_image_1.save(output_buffer, format='JPEG')
                                rotation_binary_data_1 = output_buffer.getvalue()
                                rotation_base64_data_1 = base64.b64encode(rotation_binary_data_1)
                                # 将图片放大一倍
                                output_buffer = BytesIO()
                                (x1, y1) = result_rotation_image.size
                                print(x1, y1)
                                x_r = x1 * 2
                                y_r = y1 * 2
                                result_rotation_image_2 = result_rotation_image.resize((x_r, y_r), Image.ANTIALIAS)
                                result_rotation_image_2.save(output_buffer, format='JPEG')
                                rotation_binary_data_2 = output_buffer.getvalue()
                                rotation_base64_data_2 = base64.b64encode(rotation_binary_data_2)

                                contect.append([nohuofu_yuantu_data_1, nohuofu_rotate_data_1, yuantu_base64_data_1,
                                                yuantu_base64_data_2, rotation_base64_data_1,
                                                rotation_base64_data_2])
                # print(type(contect))
                contect = str(contect)
                logo = '999'
                res_back = {'code': logo, 'base64': contect}
                print(res_back)
                # en_json = json.dumps(res_back)
                # print(type(en_json))
                # print(en_json)
                #return jsonify(res_back)  # en_json

if __name__ == '__main__':
    t1 = time.time()
    img = cv2.imread('/root/liweiwei/modelDeploy/application_layer_wangshifu/2.jpg')
    # print(img)
    # img_base64_encode = base64.b64encode(img)
    # img_base64_decode = base64.b64decode(img)
    # img_base = cv2.imdecode(np.frombuffer(img_base_decode, np.uint8), cv2.IMREAD_COLOR)
    result_wangshifu = wangshifu__inference(img)  # 自己测试版本不用jsonify,用json dumps
    print(result_wangshifu)
    t2 = time.time()
    total_time = t2 - t1
    print("总耗时:" + str(total_time))
