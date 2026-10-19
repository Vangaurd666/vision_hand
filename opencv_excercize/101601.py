#仿射是特殊的单应矩阵
#需要画布是因为需要一个清晰的范围或者说网格做映射，而不是仅仅是4个点

import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    src = np.float32(
        [
            [120,80],
            [620,110],
            [680,820],
            [90,790],
        ]
    )

    w_top = np.linalg.norm(src[1] - src[0])
    w_bottom = np.linalg.norm(src[2]-src[3])
    h_left = np.linalg.norm(src[3]-src[0])
    h_right = np.linalg.norm(src[2]-src[1])

    W = int(max(w_top,w_bottom))
    H = int(max(h_left,h_right))

    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M = cv2.getPerspectiveTransform(img,M,(W,H))
    warped = cv2.warpPerspective(img,M,(W,H))

    cv2.imwrite('10_wraped.png',warped)
    print('透视矩阵=\n',M)


#


