import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 创建测试图像
# --------------------------
def create_test_image():
    img_size = 500
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # 画矩形
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
    # 画圆
    cv2.circle(img, (400, 150), 50, (0, 255, 0), 2)
    # 平行线（水平）
    cv2.line(img, (50, 350), (450, 350), (255, 0, 0), 2)
    cv2.line(img, (50, 400), (450, 400), (255, 0, 0), 2)
    # 垂直线
    cv2.line(img, (200, 50), (200, 450), (0, 0, 0), 2)
    cv2.line(img, (250, 50), (250, 450), (0, 0, 0), 2)

    return img

# --------------------------
# 2. 相似变换
# --------------------------
def similarity_transform(img):
    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)
    angle = 30  # 旋转30度
    scale = 0.8
    M = cv2.getRotationMatrix2D(center, angle, scale)
    result = cv2.warpAffine(img, M, (cols, rows))
    return result

# --------------------------
# 3. 仿射变换
# --------------------------
def affine_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (cols, rows))
    return result

# --------------------------
# 4. 透视变换
# --------------------------
def perspective_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    pts2 = np.float32([[50, 50], [cols-100, 100], [100, rows-50], [cols-50, rows-100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, M, (cols, rows))
    return result

# --------------------------
# 5. 保存结果（不依赖弹出窗口）
# --------------------------
def save_results(original, similar, affine, perspective):
    plt.figure(figsize=(12, 8))
    plt.subplot(221), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(222), plt.imshow(cv2.cvtColor(similar, cv2.COLOR_BGR2RGB)), plt.title('Similarity Transform')
    plt.subplot(223), plt.imshow(cv2.cvtColor(affine, cv2.COLOR_BGR2RGB)), plt.title('Affine Transform')
    plt.subplot(224), plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB)), plt.title('Perspective Transform')
    plt.tight_layout()
    # 直接保存图片到文件，不用弹出窗口
    plt.savefig("work5_result.png")
    print("✅ 结果已保存到 work5_result.png")

# --------------------------
# 主程序入口（关键：调用所有函数）
# --------------------------
if __name__ == "__main__":
    print("开始运行...")
    # 创建测试图
    test_img = create_test_image()
    print("✅ 测试图像创建完成")

    # 三种变换
    img_similar = similarity_transform(test_img)
    print("✅ 相似变换完成")
    img_affine = affine_transform(test_img)
    print("✅ 仿射变换完成")
    img_perspective = perspective_transform(test_img)
    print("✅ 透视变换完成")

    # 保存结果
    save_results(test_img, img_similar, img_affine, img_perspective)
    print("全部完成！")