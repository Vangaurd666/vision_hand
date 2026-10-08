import cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    if img is None:
        print(f'{path}读取失败')
        continue

    print('-----',path,'-----')
    print('shape=',img.shape)

    print('dtype=',img.dtype)
    print('min/max=',img.min(),img.max())
    print('contiguous=',img.flags['C_CONTIGUOUS'])

    roi_view = img[20:120,30:160]
    roi_copy = img[20:120,30:160].copy()

    roi_view[:] = 0
    print('原图区域均值=',img[20:120,30:160].mean())
    print('独立拷贝均值',roi_copy.mean())
    img_float = img.astype(np.float32)/255.0
    print('float range=',img_float.min(),img_float.max())


