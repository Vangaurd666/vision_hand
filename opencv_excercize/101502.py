#和101501相似，在做匹配时，不仅仅可以用模板，还可以用特征匹配
#关键点包含位置、梯度方向、稳定性；描述子描述该点的信息，通过点附近的区域算出来的；匹配器是对比描述子之间的相似性，不同描述子的距离计算不同；随机采样一致性是在匹配器做完后做验证
#ORB快速，尺度+旋转不变，大尺度复杂变形弱；SIFT，尺度+旋转+亮度不变
#1500是关键点数量，找角点候选，计算响应值排行取值。
#RANSAC 随机抽4个点算出一个H单应矩阵 然后用这个矩阵解释全部点，看像素偏差小于3的比例

import cv2
import numpy as np

img1 = cv2.imread('a.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('b.jpg',cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1500)
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knn = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in knn:
    if m.distance <0.75 * n.distance:
        good.append(m)

if len(good) >=4:
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    H,mask = cv2.findHomography(src,dst,cv2.RANSAC,3.0)
    inlier_ratio = float(mask.mean())if mask is not None else 0.0
    print('候选匹配=',len(good),'内点比例 =',round(inlier_ratio,3))
    print('H=\n',H)
