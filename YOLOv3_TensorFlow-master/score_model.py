import os
import sys
import cv2
import rank
import numpy as np
OD_PATH = "D:/GANN/oth/hws.txt"
SCORED_PATH = "D:/GANN/oth/scored.jpg"
if __name__ == '__main__':
    img_path = sys.argv[1]
    od = os.system("python ./YOLOv3_TensorFlow-master/od.py " + img_path)
    if(od != 0):
        exit()
    with open(OD_PATH,"r") as f:
        od_list = f.read().split("\n")
        for i in range(len(od_list)):
            od_list[i] = [int(float(j)) for j in od_list[i].split(",")]
        f.close()
    #print(od_list)
    img = cv2.imread(img_path)
    img_hws = []
    gray = None
    for i in range(len(od_list)):
        gray = img[int(od_list[i][1]):int(od_list[i][3]), int(od_list[i][0]):int(od_list[i][2])]
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        s0,s1 = np.shape(gray)
        gray = np.reshape(gray,(s0,s1,1))
        img_hws.append(gray)
    # print(np.shape(img_hws[0]))
    scores = []
    for i in range(len(img_hws)):
        print("\rRanking-----" + str(int(i/len(img_hws)*100)) + "%", end="")
        res = rank.HW_rank(img_hws[i],is_file=False)
        # print(res)
        scores.append(res[-1])
    # print(np.mean(scores))
    new_img = np.array(img)
    for i in range(len(od_list)):
        new_img = cv2.rectangle(new_img, (od_list[i][0],od_list[i][1] ), (od_list[i][2],od_list[i][3]), color=0)
    new_img = cv2.putText(new_img, str(int(np.mean(scores))), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2,(0,0,0),2)
    # print(od_list[0])
    # new_img = cv2.rectangle(new_img,[int(x) for x in od_list[0]],color=0)
    cv2.imwrite(SCORED_PATH,new_img)
    cv2.imshow("a", new_img)
    cv2.waitKey(0)
