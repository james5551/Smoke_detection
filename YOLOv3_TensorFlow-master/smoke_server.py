from test_batch_images_1 import YoloV3
import time
from PIL import Image
import base64
import numpy as np
from flask import Flask,request,jsonify
import cv2
from urllib.parse import unquote
app = Flask(__name__)
detector = YoloV3('./smoke_preview.pb')
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
def get_detection(image):
    new_size = [416,416]
    #img = image.resize(tuple(new_size),Image.BICUBIC)
    image = np.asarray(image)
    height_ori, width_ori = image.shape[:2]
    print(width_ori)
    print(height_ori)
    boxed_image = cv2.resize(image,(416,416))
    #boxed_image = letterbox_image(image, (416, 416))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(image,tuple(new_size))
    print('img_size',boxed_image.shape)
    #new_image = Image.new('RGB',tuple(new_size),(128,128,128))
    #new_image.paste(img,(0,0))
    #image_data = np.array(new_image).astype((float))/255
    image_data = np.array(boxed_image).astype(float)/255
    image_data = np.expand_dims(image_data,0)
    #img = np.asarray(img,np.float32)
    #img = img[np.newaxis,:]/255.
    #print(image_data.shape)
    boxes_, classId_ndarray, score_ndarray = detector.run(image_data)
    print(boxes_)
    #box_ndarray = coordinates_transform(box_ndarray,ori_size=[ori_width,ori_height],new_size=new_size)
    boxes_[:, 0] *= (width_ori / float(new_size[0]))
    boxes_[:, 2] *= (width_ori / float(new_size[0]))
    boxes_[:, 1] *= (height_ori / float(new_size[1]))
    boxes_[:, 3] *= (height_ori / float(new_size[1]))

    boxes_[:, 0] = np.ceil(boxes_[:, 0])
    boxes_[:, 2] = np.ceil(boxes_[:, 2])
    boxes_[:, 1] = np.ceil(boxes_[:, 1])
    boxes_[:, 3] = np.ceil(boxes_[:, 3])
    print(boxes_)
    print(classId_ndarray)
    print(score_ndarray)
    return boxes_,classId_ndarray,score_ndarray


@app.route('/smoke_server',methods = ['POST'])
def yolo_server():
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
            box_ndarray, classId_ndarray, score_ndarray= get_detection(img)
            t2 = time.time()
            print('打开接收的图片文件并做目标检测，总共耗时%.2f秒\n' % (t2-t1))
            # result_str = json.dumps(result)
            json_dict = {
                'box_list': box_ndarray.astype('int').tolist(),
                'classId_list': classId_ndarray.tolist(),
                'score_list': score_ndarray.tolist()
            }
            return jsonify(**json_dict)
        except Exception as e:
            print(e)

#主函数
if __name__ == '__main__':
    app.run('127.0.0.1', port=5000)