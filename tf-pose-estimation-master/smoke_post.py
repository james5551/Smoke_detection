import requests
import base64
import numpy as np
import cv2
import collections
from tf_pose import common
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
boxes_ = box_ndarray.astype('int').tolist()
classId_list = responseJson_dict['classId_list']
#print(classes_dict[classId_list[0]])
scores_ = responseJson_dict['score_list']
smoke_boxes = []
smoke_scores = []
if len(boxes_)>0:
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
    nose_list = responseJson_dict['nose_list']
    npimg = np.copy(img_ori)
    image_h, image_w = npimg.shape[:2]
    for i in range(len(nose_list)):
        for j in range(len(boxes_)):
            print('smoke_box:', boxes_[j])
            smoke_left = boxes_[j][0]
            smoke_up = boxes_[j][1]
            print('smoke_left:', smoke_left)
            print('smoke_up:', smoke_up)
            if  abs(nose_list[i][0] - smoke_left) < 100 and abs(nose_list[0][1] - smoke_up) < 100:
                # cilp_img = npimg[keypoint[16][1]:keypoint[6][1], keypoint[16][0]:keypoint[6][0]]
                cv2.circle(npimg, (int(nose_list[i][0]),int(nose_list[i][1])), 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
                cv2.rectangle(npimg, (int(boxes_[j][0]), int(boxes_[j][1])), (int(boxes_[j][2]), int(boxes_[j][3])),
                              (0, 0, 255), 2)
                #print('Existing Smoke!')
                # cv2.imwrite(img_name, npimg)
                # boxes_[j][0]+=person_left
                # boxes_[j][2]+=person_left
                # boxes_[j][1]+=person_up
                # boxes_[j][3]+=person_up
                smoke_boxes.append(boxes_[j])
                smoke_scores.append(scores_[j])

if len(smoke_boxes)>0:
    print('*'*30)
    print('Existing Smoke!')
    print('*' * 30)
else:
    print('*' * 30)
    print('No Existing Smoke!')
    print('*' * 30)
classes = {0: 'smoke'}
for i in range(len(smoke_boxes)):
    x0, y0, x1, y1 = smoke_boxes[i]
    print('smoke_boxes:', smoke_boxes[i])
    cv2.rectangle(img_ori,(int(x0), int(y0)),(int(x1),int(y1)),color=(0,255,255),thickness=1)
cv2.imwrite('final_result.jpg', img_ori)
