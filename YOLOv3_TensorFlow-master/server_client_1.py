import base64
import cv2
import  requests
import json
import os
import numpy as np
import math
# 获取绘制检测效果之后的图片
from PIL import Image, ImageDraw, ImageFont
def get_imageBase64String(imageFilePath):
    assert os.path.exists(imageFilePath), "此图片路径不存在: %" % imageFilePath
    with open(imageFilePath, 'rb') as file:
        image_bytes = file.read()
        image_base64_bytes = base64.b64encode(image_bytes)
        image_base64_string = image_base64_bytes.decode('utf-8')
    return image_base64_string

def resize_image_1(imageFilePath):
    new_size = [416,416]
    img_ori = cv2.imread(imageFilePath)
    height_ori, width_ori = img_ori.shape[:2]
    resized_image_ndarray = cv2.resize(img_ori,tuple(new_size))
    scale_ration = [width_ori/float(new_size[0]),height_ori/float(new_size[1])]
    image_dirPath, imageFileName = os.path.split(imageFilePath)
    resized_imageFileName = 'resized_' + imageFileName
    #resized_imageFileName = imageFileName
    resized_imageFilePath = os.path.join(image_dirPath, resized_imageFileName)
    cv2.imwrite(resized_imageFilePath, resized_image_ndarray)
    return resized_imageFilePath, scale_ration

def coordinates_transform(boxes_,scale_ration):

    boxes_[:,0] *= scale_ration[0]
    boxes_[:,1] *= scale_ration[1]
    boxes_[:,2] *= scale_ration[0]
    boxes_[:,3] *= scale_ration[1]

    boxes_[:, 0] = np.ceil(boxes_[:, 0])
    boxes_[:, 2] = np.ceil(boxes_[:, 2])
    boxes_[:, 1] = np.ceil(boxes_[:, 1])
    boxes_[:, 3] = np.ceil(boxes_[:, 3])
    return boxes_

def resize_image(imageFilePath, max_height=416, max_width=416):
    image_ndarray = cv2.imread(imageFilePath)
    old_height, old_width, _ = image_ndarray.shape
    if old_width > max_width or old_height > max_height:
        if old_width / old_height >= max_width / max_height:
            new_width = max_width
            resized_multiple = new_width / old_width
            new_height = math.ceil(old_height * resized_multiple)
        else:
            new_height = max_height
            resized_multiple = new_height / old_height
            new_width = math.ceil(old_width * resized_multiple)
    else:
        resized_multiple = 1
        new_width = old_width
        new_height = old_height
    resized_image_ndarray = cv2.resize(
        image_ndarray,
        (new_width, new_height),
    )
    image_dirPath, imageFileName = os.path.split(imageFilePath)
    resized_imageFileName = 'resized_' + imageFileName
    resized_imageFilePath = os.path.join(image_dirPath, resized_imageFileName)
    cv2.imwrite(resized_imageFilePath, resized_image_ndarray)
    return resized_imageFilePath, resized_multiple

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/get_detectionResult"
    classes_dict = {0: 'dog'}
    while True:
        input_content = input('输入图片路径，输入-1退出，默认值(../resources/images/person.jpg): ')
        if input_content.strip() == "":
            input_content = '../resources/images/person.jpg'
        if input_content.strip() == "-1":
            break
        elif not os.path.exists(input_content.strip()):
            print('输入图片路径不正确，请重新输入')
        else:
            imageFilePath = input_content.strip()
            resized_imageFilePath, resized_multiple = resize_image_1(imageFilePath)
            image_base64_string = get_imageBase64String(resized_imageFilePath)
            data_dict = {'image_base64_string': image_base64_string}
            # 调用request.post方法发起post请求，并接收返回结果
            response = requests.post(url, data=data_dict)
            # 处理返回的json格式数据，准备好传入get_drawedImageNdarray函数的参数
            responseJson_dict = response.json()
            image = Image.open(imageFilePath)
            # box_ndarray = np.array(responseJson_dict['box_list']) / resized_multiple
            box_ndarray = np.array(responseJson_dict['box_list'], np.float)
            box_ndarray = coordinates_transform(box_ndarray, resized_multiple)
            box_list = box_ndarray.astype('int').tolist()
            classId_list = responseJson_dict['classId_list']
            print(classes_dict[classId_list[0]])
            score_list = responseJson_dict['score_list']
            for i in range(len(box_list)):
                x0, y0, x1, y1 = box_list[i]
                print(x0,y0,x1,y1)
                image = np.array(image)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=2)
            #cv2.namedWindow('name', cv2.WINDOW_AUTOSIZE)
            # cv2.resizeWindow('test', 1000, 1000)
            #cv2.imshow('img', image)
            cv2.imwrite('D:\dog_project\ckpt-to-pb\server/result.jpg', image)
            #cv2.waitKey(0)



