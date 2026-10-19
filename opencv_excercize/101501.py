#模版匹配，当我要在一张游戏大图中找到金币时，可以使用一张金币的图像进入大图进行匹配，金币主要由明暗和形状区分，所以转为灰度图来识别；不同图片中金币大小可能发生缩放变化，所以用多尺度


import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))
tpl0 = cv2.imread('icon.jpeg',cv2.IMREAD_GRAYSCALE)

for path in img_path:
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    best = {'score':-1.0}
    for scale in np.linspace(0.7,1.3,13):
        tw = max(4,int(tpl0.shape[1]*scale))
        th = max(4,int(tpl0.shape[1]*scale))
        tpl = cv2.resize(tpl0,(tw,th),interpolation=cv2.INTER_LINEAR)
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue

        resp = cv2.matchTemplate(gray,tpl,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,max_loc = cv2.minMaxLoc(resp)
        if max_val > best['score']:
            best = {'score':max_val,'loc':max_loc,'size':(tw,th),'scale':scale}

    x,y = best['loc']
    tw,th = best['size']
    cv2.rectangle(img,(x,y),(x+tw,y+th),(0,0,255),2)
    print(best)
    cv2.imwrite('08_match.png',img)


