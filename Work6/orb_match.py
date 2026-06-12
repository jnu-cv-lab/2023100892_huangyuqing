import cv2

# ---------------------- 1. 读取图像 + ORB特征检测（复用任务1的步骤） ----------------------
img_box = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)

# 初始化ORB检测器
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# ---------------------- 2. 创建暴力匹配器 ----------------------
# 按要求使用NORM_HAMMING + crossCheck=True
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ---------------------- 3. 特征匹配 ----------------------
matches = bf.match(des_box, des_scene)

# 按匹配距离从小到大排序
matches = sorted(matches, key = lambda x:x.distance)

# ---------------------- 4. 输出结果信息 ----------------------
print("="*40 + " 任务2 结果 " + "="*40)
print(f"总匹配数量：{len(matches)}")

# ---------------------- 5. 绘制匹配结果 ----------------------
# 绘制所有匹配（初始匹配图）
img_all_matches = cv2.drawMatches(
    img_box, kp_box, img_scene, kp_scene, matches, None, flags=2
)
cv2.imwrite("orb_all_matches.png", img_all_matches)

# 绘制前50个匹配（可视化结果）
top_n = 50
img_top_matches = cv2.drawMatches(
    img_box, kp_box, img_scene, kp_scene, matches[:top_n], None, flags=2
)
cv2.imwrite("orb_top50_matches.png", img_top_matches)

print(f"✅ 初始匹配图已保存为：orb_all_matches.png")
print(f"✅ 前{top_n}个匹配图已保存为：orb_top50_matches.png")
print("="*95)