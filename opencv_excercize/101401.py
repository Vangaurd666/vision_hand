#轮廓不仅仅可以用canny+霍夫进行计算，还可以二值化+findcounters
#求最小外接圆用来对不规则结构进行定位，然后进行可读量化，工业主流用最小二乘法拟合圆

import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    gray = cv2.imread(img,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.thereshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        peri = cv2.arcLength(cnt,True)
        if peri <= 0:
            continue
        circularity = 4.0*np.pi*area/(peri*peri)
        if circularity > best_score:
            best_score = circularity
            best =cnt

    if best is not None:
        (cx,cy),radius = cv2.minEnclosingCircle(best)
        cv2.drawContours(img,[best],-1,(0,255,0),2)
        cv2.circle(img,(int(cx),int(cy)),int(radius),(255,0,0),2)
        print('圆度=',round(best_score,4))

    cv2.imwrite('07_measure.png',img)
