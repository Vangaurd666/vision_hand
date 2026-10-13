#颜色空间 RGB适合深度学习；HSV适合做规则判断；Lab将明度和颜色通道分离，适合做亮度增强；
#解决光照问题，先将RGB转为Lsb，将明度筛出来单独增强，然后转到HSV做规则；
#解决光照导致的像素漂移问题，得先做对比度增强再处理，全局直方图均衡，基于累积分布函数和整张图的直方图算出全局映射值；这会导致像素被大幅拉伸，噪音和对比度一同被放大；且放大本就高对比和低对比区域使其失去细节变化。
#CLAHE限制对比度 自适应直方图均衡；图像切分8*8，每个格子独立算直方图和映射；
#人眼识别轮廓主要靠亮度差，而非颜色差，所以灰度图适合找轮廓、形状、模板；纹理是像素亮度在局部空间的重复模式，不需要颜色信息；
#RGB是针对显示器，人眼难辨差异；Lab为人眼设计；Lab适合白平衡，理想中性白/灰a=0 b=0来衡量；

import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))
out_dir = Path('output')
out_dir.mkdir(exist_ok=True)

for path in img_path:
    print('---',path,'---')
    img = cv2.imread(str(path))
    if img is None:
        print(f'跳过无法读取的文件：{path}')
        continue
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    enhanced = cv2.cvtColor(lab2,cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced,cv2.COLOR_BGR2HSV)
    lower = np.array([35,60,40],dtype=np.uint8)
    upper = np.array([90,255,255],dtype=np.uint8)
    mask = cv2.inRange(hsv,lower,upper)
    result = cv2.bitwise_and(enhanced,enhanced,mask=mask)

    cv2.imwrite(str(out_dir/f'mask_{path.stem}.png'),mask)
    cv2.imwrite(str(out_dir/f'result_{path.stem}.png'),result)
    print(f'完成{path.name}')

