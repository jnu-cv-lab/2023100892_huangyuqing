import cv2
import numpy as np

# ---------------------- 1. 读取图像 + ORB特征检测（复用前序步骤） ----------------------
img_box = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# ---------------------- 2. 暴力匹配 ----------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

# ---------------------- 3. 提取对应点坐标 ----------------------
# 把匹配点的坐标提取出来，转换成findHomography需要的格式
pts_box = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ---------------------- 4. 使用RANSAC估计单应矩阵 ----------------------
# 按要求使用cv2.RANSAC，重投影误差阈值设为5.0
H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, 5.0)

# ---------------------- 5. 统计结果信息 ----------------------
total_matches = len(matches)
inlier_mask = mask.ravel().tolist()  # 把mask转成列表
num_inliers = sum(inlier_mask)
inlier_ratio = num_inliers / total_matches

# ---------------------- 6. 绘制RANSAC后的内点匹配 ----------------------
# 只绘制内点
draw_params = dict(matchColor=(0, 255, 0),  # 匹配线为绿色
                   singlePointColor=None,
                   matchesMask=inlier_mask,  # 只显示内点
                   flags=2)

img_ransac = cv2.drawMatches(img_box, kp_box, img_scene, kp_scene, matches, None, **draw_params)

# 保存结果图片
cv2.imwrite("orb_ransac_result.png", img_ransac)

# ---------------------- 7. 输出实验要求的所有信息 ----------------------
print("="*40 + " 任务3 结果 " + "="*40)
print(f"总匹配数量：{total_matches}")
print(f"RANSAC内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.4f}")
print("\nHomography 矩阵：")
print(H)
print("\n✅ RANSAC后的匹配图已保存为：orb_ransac_result.png")
print("="*95)