import cv2
import numpy as np

# ---------------------- 1. 读取图像 + ORB特征检测（复用前序步骤） ----------------------
img_box = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# ---------------------- 2. 暴力匹配 + RANSAC单应矩阵估计 ----------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

# 提取对应点
pts_box = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 估计Homography矩阵
H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, 5.0)

# ---------------------- 3. 获取box.png的四个角点 ----------------------
h, w = img_box.shape
corners_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# ---------------------- 4. 使用perspectiveTransform投影到场景图 ----------------------
corners_scene = cv2.perspectiveTransform(corners_box, H)

# ---------------------- 5. 在场景图中画出目标边框 ----------------------
# 把场景图转成彩色图，方便画彩色框
img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
# 画四边形边框，颜色设为红色，线宽3
cv2.polylines(img_scene_color, [np.int32(corners_scene)], True, (0, 0, 255), 3)

# 保存结果图片
cv2.imwrite("target_localization_result.png", img_scene_color)

# ---------------------- 6. 输出结果说明 ----------------------
print("="*40 + " 任务4 结果 " + "="*40)
print("✅ 目标定位已完成！")
print("结果图已保存为：target_localization_result.png")
print("\n定位说明：")
print("通过RANSAC估计的Homography矩阵，已将box.png的四个角点投影到场景图中，")
print("并成功在box_in_scene.png中画出了目标物体的红色边框，定位成功。")
print("="*95)