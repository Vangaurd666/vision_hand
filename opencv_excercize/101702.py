#MOG2，对一个像素位置，收集在history=300帧中出现的颜色，拟合成若干个高斯分布，比较不同时间的颜色分布
#while True 是条件为真的永远为真的循环
#capture是一个维护MP4解码状态和只会前进的帧指针

import cv2
from pathlib import Path

vid_dir = Path('test')
vid_path = list(vid_dir.glob('*.mp4'))

for path in vid_path:
    cap = cv2.VideoCapture(str(path))
    back = cv2.createBackgroundSubtractorMOG2(
        history=300,varThreshold=25,detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    while True:
        ret,frame = cap.read()
        if not ret:
            break

        fg = back.apply(frame)
        _,fg = cv2.threhold(fg,200,255,cv2.THRESH_BINARY)
        fg = cv2.morphologyEX(fg,cv2.MORPH_OPEN,kernel)
        fg = cv2.dilate(fg,kernel,iterations=1)

        contours,_ = cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        cap.release()
