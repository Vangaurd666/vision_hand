#gray = cv2.cvtColor(img,cv2.BGR2GRAY)
#gray = cv2.HaussianBlur(gray,(5,5),0)

#将边缘轮廓转换为几何语言，将检测到的边缘尽量贴近真实边缘
#canny流水线：高斯平滑去噪-sobel梯度计算-非极大值抑制-双阈值分类-滞后连接
#噪音的孤立点梯度变化明显，容易误判故需要高斯平滑；sobel用一对卷积核沿着xy分别卷积后合成；
#双阈值high和low，高于high是边缘低于low非边缘，中间就交给滞后连接处理；滞后连接也就是将和强边缘有连接的弱边缘作为强边缘；
#霍夫转换，像素点投票制，直线用ρ表示长度θ表示角度，基于像素点的x，预设4张θ角度30 60 90 0（这就是参数np.pi/180来决定步长），然后看各个像素点投票出来的重合的ρ和西塔,找到最高的重合点

import  cv2
import numpy as np
from pathlib import Path

img_dir = Path('test')
img_path = list(img_dir.glob('*.jpeg'))

for path in img_path:
    img = cv2.imread(str(path))
    if img is None:
        print(f'跳过无法读取的文件：{path}')
        continue

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    edges = cv2.Canny(blur,60,160)

    lines = cv2.HoughLinesP(
        edges,1,np.pi/180,
        threshold = 60,
        minLineLength=60,
        maxLineGap=12,
    )

    if lines is not None:
        for line in lines[:,0]:
            x1,y1,x2,y2 = line
            length = np.hypot(x2-x1,y2-y1)
            if length < 80:
                continue
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('06_lines.png',img)
    cv2.imwrite('06_edges.png',edges)