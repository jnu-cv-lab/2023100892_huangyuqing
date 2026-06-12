import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取你拍的图片
img = cv2.imread("paper.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ 这是我专门给你算的坐标！
pts_src = np.float32([
    [165, 339],   # 左上
    [853, 361],   # 右上
    [253, 1495],  # 左下
    [1150, 1313]  # 右下
])

# 目标标准A4大小
w, h = 600, 800
pts_dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# 透视变换
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
img_corrected = cv2.warpPerspective(img, M, (w, h))

# 显示 + 保存
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img), plt.title("畸变原图"), plt.axis('off')
plt.subplot(122), plt.imshow(img_corrected), plt.title("校正后"), plt.axis('off')
plt.savefig("校正结果.png", dpi=150, bbox_inches='tight')
plt.show()