import cv2
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    boxes_xyxy = [
        [100,80,260,240],
        [110,90,255,235],
        [300,100,420,260],
    ]
    scores = [0.92,0.81,0.88]

    boxes_xywh = []
    for x1,y1,x2,y2 in boxes_xyxy:
        boxes_xywh.append([x1,y1,x2-x1,y2-y1])

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh,scores,
        score_threshold=0.3,
        nms_threshold=0.5
    )

    for idx in indices:
        i = int(idx)
        x,y,w,h = boxes_xywh[i]

