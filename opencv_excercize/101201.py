import cv2
from pathlib import Path
import numpy as np

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    if img is None:
        print(f'跳过无法读取的文件：{path}')
        continue
    gray = cv2.cvtColor(img,cv2.BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)


mask = cv2.threshold(
    gray,0,255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

num,labels,stats,centroids = cv2.connectedComponentsWithStats(mask,connectivity=8)

for i in range(1,num):
    x,y,w,h,area = stats[i]
    if area < 200:
        continue
    cx,cy = centroids[i]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(img,(int(cx),int(cy)),3,(0,0,255),-1)
    cv2.imwrite('05_components.png',img)