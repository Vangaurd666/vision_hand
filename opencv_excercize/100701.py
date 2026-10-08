from pathlib import Path
import cv2

image_dir = Path('test')
image_paths = list(image_dir.glob('*.jpeg'))

for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"读取失败：{path}")
        continue

    h,w = img.shape[:2]
    print(f'{path.name}:宽={w},高={h},类型={img.dtype}')

