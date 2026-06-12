import cv2
import matplotlib.pyplot as plt

# 1. 读取图片（灰度模式）
img1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# 2. 创建ORB检测器
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# 保存结果图片
cv2.imwrite("box_keypoints.png", img1_kp)
cv2.imwrite("box_in_scene_keypoints.png", img2_kp)

# 5. 输出结果
print("box.png 关键点数量:", len(kp1))
print("box_in_scene.png 关键点数量:", len(kp2))
print("描述子维度:", des1.shape[1])

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img1_kp, cmap='gray')
plt.title("box.png ORB Keypoints"), plt.axis('off')
plt.subplot(122), plt.imshow(img2_kp, cmap='gray')
plt.title("box_in_scene.png ORB Keypoints"), plt.axis('off')
plt.show()