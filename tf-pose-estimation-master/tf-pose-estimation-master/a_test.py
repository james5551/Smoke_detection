import cv2
import json
import os
import base64
import numpy as np
import PIL.Image
import io
box_person = [[133.,22.,295.,206.],[7.,16.,143.,198.],[247. ,66.,300.,206.],[1.,45.,19.,216.]]
data = json.load(open('D:/Mask_RCNN-master/dataes/json/1.json'))
imageData = data.get('imageData')
# img_ori = cv2.imread()
if not imageData:
    imagePath = os.path.join(os.path.dirname('D:/Mask_RCNN-master/dataes/json/'), data['imagePath'])
    print(imagePath)
    with open(imagePath, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
img_data = base64.b64decode(imageData)
f = io.BytesIO()
f.write(img_data)
img_arr = np.array(PIL.Image.open(f).convert('RGB'))
print(img_arr.shape)
if img_arr.shape[2]==4:
    img = cv2.cvtColor(img_arr,cv2.COLOR_RGBA2RGB)
    print('finish!')
# for i in range(len(box_person)):
#     print(box_person[i])
#     img = np.copy(img_ori)
#     x0,y0,x1,y1 = box_person[i]
#     print(x0,y0,x1,y1)
#     crop_img = img_ori[int(y0):int(y1),int(x0):int(x1)]
#     # cv2.imwrite('crop_img.jpg',crop_img)
#     image = np.copy(crop_img)
#     image_name = './crop_' + str(i) + '.jpg'
#     cv2.imwrite(image_name, crop_img)