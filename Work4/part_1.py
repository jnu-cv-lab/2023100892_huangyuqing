import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------------- 1. 生成测试图 ----------------------
# 生成棋盘格 (Checkerboard)
def generate_checkerboard(size=512, square_size=32):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    board = np.mod(np.floor(x / square_size) + np.floor(y / square_size), 2)
    return board * 255

# 生成 Chirp 信号 (频率逐渐变化的图像)
def generate_chirp(size=512, min_freq=0.05, max_freq=1.0):
    x = np.linspace(0, size, size)
    # 频率从低到高线性增加
    freq = min_freq + (max_freq - min_freq) * x / size
    # 生成正弦波
    chirp = np.sin(2 * np.pi * freq * x)
    # 扩展为2D图像
    return np.tile(chirp, (size, 1)) * 127.5 + 127.5

# 生成图像
checker_img = generate_checkerboard()
chirp_img = generate_chirp()

# ---------------------- 2. 下采样函数 ----------------------
def downsample(img, scale=4, sigma=None):
    # 如果指定sigma，先进行高斯滤波
    if sigma is not None:
        img = gaussian_filter(img, sigma=sigma)
    # 直接下采样 (每隔scale个像素取一个)
    return img[::scale, ::scale]

# 执行下采样
scale_factor = 4
# 直接下采样不滤波
downsampled_no_filter = downsample(checker_img, scale_factor)
# 加高斯滤波后下采样
downsampled_with_filter = downsample(checker_img, scale_factor, sigma=1.0)

# ---------------------- 3. FFT 频谱分析 ----------------------
def fft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 计算幅度谱并取对数，便于可视化（加1e-8避免log(0)报错）
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

# ---------------------- 4. 绘制结果 ----------------------
plt.figure(figsize=(15, 10))

# 第一行：原图 + 直接下采样 + 滤波后下采样（棋盘格）
plt.subplot(3, 3, 1)
plt.imshow(checker_img, cmap='gray')
plt.title('原图 (棋盘格)')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(downsampled_no_filter, cmap='gray')
plt.title('直接下采样 (混叠严重)')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(downsampled_with_filter, cmap='gray')
plt.title('高斯滤波+下采样 (清晰)')
plt.axis('off')

# 第二行：混叠/滤波后的频谱
plt.subplot(3, 3, 4)
plt.imshow(fft_spectrum(downsampled_no_filter), cmap='jet')
plt.title('混叠图像频谱')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(fft_spectrum(downsampled_with_filter), cmap='jet')
plt.title('滤波后图像频谱 (混叠消失)')
plt.axis('off')

# 第三行：Chirp图对比
plt.subplot(3, 3, 7)
plt.imshow(chirp_img, cmap='gray')
plt.title('Chirp测试图')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(downsample(chirp_img, 4), cmap='gray')
plt.title('Chirp直接下采样 (混叠)')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(downsample(chirp_img, 4, sigma=1.0), cmap='gray')
plt.title('Chirp滤波+下采样')
plt.axis('off')

# 关键修改：直接保存图片到文件，无需弹窗口
plt.tight_layout()
# 保存为高清PNG，300dpi，适合实验报告
plt.savefig("part1_result.png", dpi=300, bbox_inches="tight")
print("✅ 第一部分结果已保存为 part1_result.png，在cv-course文件夹里！")
# 关闭图形，避免WSL残留进程
plt.close()