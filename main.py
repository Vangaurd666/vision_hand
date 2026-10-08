import cv2

img = cv2.imread('data/gray1.jpg')
h,w = img.shape[:2]
pixel = img[2,3]
print(h,w,pixel)