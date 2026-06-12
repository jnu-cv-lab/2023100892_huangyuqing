import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 任务1：使用OpenCV读取一张测试图片 --------------------------
# 请将test.jpg替换为你的图片路径
img = cv2.imread("/mnt/c/Users/hyq/test.jpg")
if img is None:
    raise FileNotFoundError("无法读取图片，请检查文件路径是否正确")

# -------------------------- 任务2：输出图像基本信息 --------------------------
height, width, channels = img.shape
dtype = img.dtype
print("="*50)
print("图像基本信息：")
print(f"宽度 (Width): {width} 像素")
print(f"高度 (Height): {height} 像素")
print(f"通道数 (Channels): {channels}")
print(f"数据类型 (Data Type): {dtype}")
print("="*50)

# -------------------------- 任务3：显示原图（Matplotlib方式，解决OpenCV颜色反转问题） --------------------------
# OpenCV读取的是BGR格式，Matplotlib显示需要RGB格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("原始彩色图像")
plt.axis("off")

# -------------------------- 任务4：转换为灰度图并显示 --------------------------
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 2, 2)
plt.imshow(gray_img, cmap="gray")
plt.title("灰度图像")
plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------- 任务5：保存灰度图为新文件 --------------------------
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为 gray_test.jpg")

# -------------------------- 任务6：NumPy简单操作（两种示例可选） --------------------------
# 示例1：输出某个像素值（以(100,100)坐标为例）
pixel_value = img[100, 100]
print(f"\n坐标(100,100)处的像素值（BGR格式）: {pixel_value}")
gray_pixel_value = gray_img[100, 100]
print(f"坐标(100,100)处的灰度值: {gray_pixel_value}")

# 示例2：裁剪图像左上角100x100区域并保存
crop_img = img[0:100, 0:100]  # 切片操作：[y_start:y_end, x_start:x_end]
cv2.imwrite("crop_test.jpg", crop_img)
print("图像左上角100x100区域已保存为 crop_test.jpg")