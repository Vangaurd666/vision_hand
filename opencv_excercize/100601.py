import  cv2
import numpy as np

img = cv2.imread('test/test1.jpeg')
if img is None:
    raise FileNotFoundError('图像读取失败')

print()



