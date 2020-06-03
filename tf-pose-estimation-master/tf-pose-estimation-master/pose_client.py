import requests
import base64
import numpy as np
import cv2
filepath = './1.jpg'
img_ori = cv2.imread(filepath)
#cv2.imwrite('img_ori.jpg', img_ori)
print(img_ori.shape[:2])
height_ori, width_ori = img_ori.shape[:2]
with open(filepath,'rb') as f:
    img_base64_data = base64.b64encode(f.read())
data = {"type":'jpg',
        'img':img_base64_data}
url = "http://0.0.0.0:14000/pose_server"
result = requests.post(url,data=data)
responseJson_dict = result.json()
print(responseJson_dict['nose_list'])
