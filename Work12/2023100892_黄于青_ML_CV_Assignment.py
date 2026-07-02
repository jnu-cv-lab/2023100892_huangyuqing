import cv2
import numpy as np
import glob

# 棋盘固定参数
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 25.0
image_folder = "imgs/*.jpg"

# 亚像素角点迭代条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 构造棋盘三维世界坐标
objp = np.zeros((np.prod(CHECKERBOARD), 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

obj_points = []
img_points = []
success_list = []  # 存放识别成功的图片路径

# 读取所有标定图片
images = glob.glob(image_folder)
print(f"一共读取到 {len(images)} 张标定图片")

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)

    if ret:
        obj_points.append(objp)
        # 亚像素优化角点
        corners_opt = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_opt)
        success_list.append((img_path, img, corners_opt))
        print(f"{img_path} 角点识别成功")
    else:
        print(f"{img_path} 角点识别失败，跳过")

# 保存第3、第4张成功图片的角点绘制图
if len(success_list) >= 4:
    # 第三张成功图
    path3, img3, corner3 = success_list[2]
    cv2.drawChessboardCorners(img3, CHECKERBOARD, corner3, True)
    cv2.imwrite("corner_draw_3.jpg", img3)
    # 第四张成功图
    path4, img4, corner4 = success_list[3]
    cv2.drawChessboardCorners(img4, CHECKERBOARD, corner4, True)
    cv2.imwrite("corner_draw_4.jpg", img4)
    print("已保存第3、4张识别成功图片的角点图：corner_draw_3.jpg、corner_draw_4.jpg")
else:
    print("警告：识别成功的图片不足4张，无法生成3、4号角点图，请重拍更多清晰照片！")

# 执行相机标定
err, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 打印报告所需标定数据
print("========== 相机标定结果 ==========")
print(f"平均重投影误差：{err:.4f}")
print("\n相机内参矩阵 K：")
print(K)
print("\n畸变系数 D [k1,k2,p1,p2,k3]：")
print(D.ravel())

# 生成去畸变校正图（使用第一张有效图片）
test_img = cv2.imread(success_list[0][0])
h, w = test_img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
undist_img = cv2.remap(test_img, map1, map2, cv2.INTER_LINEAR)
cv2.imwrite("undistort_result.jpg", undist_img)
print("已生成去畸变校正图：undistort_result.jpg")