import requests
import base64
import numpy as np
import cv2
filepath = './1.jpg'
#classes_dict = {0:'dog'}
ori_img = cv2.imread(filepath)
ori_height,ori_width = ori_img.shape[:2]
print(ori_width,ori_height)
ori_size = [ori_width,ori_height]
with open(filepath,'rb') as f:
    img_base64_data = base64.b64encode(f.read())
data = {"type":'jpg',
        'img':img_base64_data}
url = "http://127.0.0.1:5000/smoke_server"
result = requests.post(url,data=data)
responseJson_dict = result.json()
# box_ndarray = np.array(responseJson_dict['box_list']) / resized_multiple
box_ndarray = np.array(responseJson_dict['box_list'], np.float)
#box_ndarray = coordinates_transform(box_ndarray,ori_size,new_size=[416,416])
box_list = box_ndarray.astype('int').tolist()
classId_list = responseJson_dict['classId_list']
#print(classes_dict[classId_list[0]])
score_list = responseJson_dict['score_list']
img_ori = cv2.imread(filepath)
for i in range(len(box_ndarray)):
    x0, y0, x1, y1 = box_list[i]
    print(x0,y0,x1,y1)
    cv2.rectangle(img_ori, (int(x0), int(y0)), (int(x1),int(y1)),(0,255,0),thickness=2)
#cv2.namedWindow('name',cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow('test', 1000, 1000)
#cv2.imshow('img',img_ori)
# cv2.destroyAllWindows()
cv2.imwrite('./result.jpg',img_ori)
#cv2.waitKey(0)