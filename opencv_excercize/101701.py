#内参是相机镜头的焦距和主点，这是不变的；外参是真实世界和镜头的关系。
#真实世界坐标 通过外参映射到 镜头中；镜头坐标 通过内参映射到 像素坐标；内参是已知的，外参是通过PnP求出的，相机和真实世界相对位置出现变化就要重新求外参；真实工程不是每帧都使用PnP重新算，会使用跟踪传播对上一帧的外参微调、上一帧结果作为初值加速运算
#针孔成像为例，焦距是针孔到底片的距离；主点是针孔的轴和像素坐标的交点

import cv2
import numpy as np

object_points = np.float32(
    [
        [0,0,0],
        [1,0,0],
        [1,1,0],
        [0,1,0],
    ]
)

image_points = np.float32(
    [
        [280,210],
        [410,220],
        [400,350],
        [270,340],
    ]
)

k = np.float64([
    [800,0,320],
    [0,800,240],
    [0,0,1],
])

dist = np.zeros(5,dtype = np.float64)
ok,rvec,tvec = cv2.solvePnP(
    object_points,image_points,k,dist,
    flags= cv2.SOLVEPNP_ITERATIVE
)
R,_ = cv2.Rodrigues(rvec)
print('成功=',ok)
print('旋转矩阵R=\n',R)
print('平移矩阵t=\n',tvec)
