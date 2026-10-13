#噪音类型决定滤波方法，这是从频域的角度来看噪音
#还可以从统计学的角度看噪音、信号关系角度、空间形态角度、来源角度、颜色角度
#高斯噪音服从正态分布零均值，加性噪音，传感热噪音，使用均值类滤波器高斯模糊、均值滤波
#椒盐噪音只有0和255双脉冲分布，脉冲替换型原始值已丢失，传输误码、坏点，使用中值滤波
#周期性滤波确定性信号，加性，电网工频干扰，使用方向滤波
#泊松噪音，泊松分布方差等于均值，独立随机事件离散分布，非加性噪音

import cv2
import numpy as np
import time
from pathlib import Path

img_dir = Path('test')
img_path =list(img_dir.glob('*.jpeg'))

img = cv2.imread(str(img_path[1]),cv2.IMREAD_GRAYSCALE)
rng = np.random.default_rng(0)

noise = rng.normal(0,18,img.shape).astype(np.flot32)
noisy = np.clip(img.astype(np.float32)+noise,0,255).astype(np.unit8)

methods ={
    'gaussian':lambda x:cv2.GaussianBlur(x,(5,5),1.2),
    'median':lambda x:cv2.medianBlur(x,5),
    'bilateral':lambda x:cv2.bilateralFilter(x,7,50,50),
}

for name,fn in methods.items():
    t0 = time.perf_counter()
    out = fn(noisy)
    cost_ms = (time.perf_counter()-t0)*1000
    mae = np.abs(out.astype(np.float32)-img.astype(np.float32)).mean()
    print(name,'MAE=',round(float(mae),2),'耗时毫秒=',round(cost_ms,2))
    cv2.imwrite(f'04_{name}.png',out)