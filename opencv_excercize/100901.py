import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

def letterbox(img,dst_w=640,dst_h=640):
    h,w = img.shape[:2]
    scale = min(dst_w/w,dst_h/h)
    new_w,new_h = round(w*scale),round(h*scale)
    resized = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_LINEAR)

    pad_w,pad_h =dst_w -new_w,dst_h-new_h
    left = pad_w // 2
    right = pad_w -left
    top = pad_h // 2
    bottom = pad_h - top

    out = cv2.copyMakeBorder(
        resized,top,bottom,left,right,
        borderType=cv2.BORDER_CONSTANT,value=(114,114,114)
    )
    meta = {'scale':scale,'pad_x':left,'pad_y':top}
    return out,meta

def forward_box(box,meta):
    x1,y1,x2,y2 = box
    s,px,py = meta['scale'],meta['pad_x'],meta['pad_y']
    return np.array([x1*s+px,y1*s+py,x2*s+px,y2*s+py])

def inverse_box(box,meta):
    x1,y1,x2,y2 = box
    s,px,py = meta['scale'],meta['pad_x'],meta['pad_y']
    return np.array([(x1-px)/s,(y1-py)/s,(x2-px)/s,(y2-py)/s])

for path in img_path:
    print('---',path,'---')
    img = cv2.imread(str(path))
    if img is None:
        print(f'跳过无法读取的文件：{path}')
        continue
    out,meta = letterbox(img)
    box0 =np.array([100,80,260,220],dtype=np.float32)
    box1 =forward_box(box0,meta)
    box2 =inverse_box(box1,meta)
    print('最大回映误差：',np.abs(box0-box2).max())



