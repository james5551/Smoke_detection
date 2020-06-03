from test_batch_images_1 import YoloV3
# 导入常用的库
import time
import os
from PIL import Image
# 导入flask库
from flask import Flask, render_template, request, jsonify
# 加载把图片文件转换为字符串的base64库
import base64
import numpy as np
import io
import cv2
from urllib.parse import unquote
# 实例化Flask对象
server = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
server.jinja_env.auto_reload = True
server.config['TEMPLATES_AUTO_RELOAD'] = True
# 实例化检测器对象
detector = YoloV3('D:\dog_project\YOLOv3_TensorFlow-master/yolov3_frozen_graph_batch.pb')

# 根据图片文件路径获取base64编码后内容
def get_imageBase64String(imageFilePath):
    if not os.path.exists(imageFilePath):
        image_base64_string = ''
    else:
        with open(imageFilePath, 'rb') as file:
            image_bytes = file.read()
        image_base64_bytes = base64.b64encode(image_bytes)
        image_base64_string = image_base64_bytes.decode('utf-8')
    return image_base64_string

def get_dataDict(data):
    data_dict = {}
    for text in data.split('&'):
        key, value = text.split('=')
        value_1 = unquote(value)
        data_dict[key] = value_1
    return data_dict

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image



# 获取使用YOLOv3算法做目标检测的结果
def get_detectResult(image):
    startTime = time.time()
    boxed_image = letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image).astype('float') / 255
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # 模型网络结构运算
    box_ndarray, classId_ndarray, score_ndarray = detector.run(image_data)
    print(box_ndarray)
    #box_ndarray = box_ndarray[:, [1, 0, 3, 2]]
    return box_ndarray, classId_ndarray, score_ndarray

# 网络请求'/'的回调函数
@server.route('/')
def index():
    htmlFileName = '_08_yolov3.html'
    return render_template(htmlFileName)





# 网络请求'/get_drawedImage'的回调函数
@server.route('/get_detectionResult', methods=['POST'])#get_drawedImage
def anyname_you_like():
    startTime = time.time()
    data_bytes = request.get_data()
    data = data_bytes.decode('utf-8')
    data_dict = get_dataDict(data)

    # print(type(received_file))
    imageFileName = '1.jpg'
    print('imageFileName:',imageFileName)
    if 'image_base64_string' in data_dict:
        # 保存接收的图片到指定文件夹
        received_dirPath = 'D:\dog_project\ckpt-to-pb\server'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        try:
            image_base64_string = data_dict['image_base64_string']
            image_base64_bytes = image_base64_string.encode('utf-8')
            image_bytes = base64.b64decode(image_base64_bytes)
            with open(imageFilePath, 'wb') as file:
                file.write(image_bytes)
            #received_file.save(imageFilePath)
            print('接收图片文件保存到此路径：%s' % imageFilePath)
            usedTime = time.time() - startTime
            print('接收图片并保存，总共耗时%.2f秒' % usedTime)
            # 对指定图片路径的图片做目标检测，并打印耗时
            image = Image.open(imageFilePath)
            #image =  np.array(image)
            #print(image.shape)
            #image = np.array(image,np)
            #image = Image.fromarray(np.uint8(image))
            box_ndarray, classId_ndarray, score_ndarray = get_detectResult(image)
            usedTime = time.time() - startTime
            print('打开接收的图片文件并做目标检测，总共耗时%.2f秒\n' % usedTime)

            # 把目标检测结果转化为json格式的字符串
            json_dict = {
                'box_list': box_ndarray.astype('int').tolist(),
                'classId_list': classId_ndarray.tolist(),
                'score_list': score_ndarray.tolist()
            }
            return jsonify(**json_dict)
        except Exception as e:
            print(e)

            '''
            # 把目标检测结果图保存在服务端指定路径，返回指定路径对应的图片经过base64编码后的字符串
            drawed_imageFileName = 'drawed_' + os.path.splitext(imageFileName)[0] + '.jpg'
            drawed_imageFilePath = os.path.join(received_dirPath, drawed_imageFileName)
            drawed_image.save(drawed_imageFilePath)
            image_base64_string = get_imageBase64String(drawed_imageFilePath)
            return jsonify(image_base64_string=image_base64_string)
            '''

# 主函数
if __name__ == '__main__':
    server.run('127.0.0.1', port=5000)