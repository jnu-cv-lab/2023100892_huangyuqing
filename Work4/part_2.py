import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------------- 复用基础函数 ----------------------
def generate_checkerboard(size=512, square_size=32):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    board = np.mod(np.floor(x / square_size) + np.floor(y / square_size), 2)
    return board * 255

def downsample(img, scale=4, sigma=None):
    if sigma is not None:
        img = gaussian_filter(img, sigma=sigma)
    return img[::scale, ::scale]

def fft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

# ---------------------- 实验参数 ----------------------
M = 4  # 固定下采样倍数
sigmas = [0.5, 1.0, 2.0, 4.0]  # 测试不同σ
img = generate_checkerboard()  # 用棋盘格观察边缘混叠

# ---------------------- 绘制结果 ----------------------
plt.figure(figsize=(16, 10))

# 遍历不同σ值
for i, sigma in enumerate(sigmas):
    # 滤波+下采样
    result = downsample(img, scale=M, sigma=sigma)
    
    # 绘制图像
    plt.subplot(2, 4, i+1)
    plt.imshow(result, cmap='gray')
    # 标注效果
    if sigma < 1.0:
        status = "混叠残留"
    elif sigma > 3.0:
        status = "过度模糊"
    else:
        status = "效果适中"
    plt.title(f'σ = {sigma} ({status})')
    plt.axis('off')
    
    # 绘制频谱
    plt.subplot(2, 4, i+5)
    plt.imshow(fft_spectrum(result), cmap='jet')
    plt.title(f'频谱 σ={sigma}')
    plt.axis('off')

# 理论值对比：σ=0.45*M=1.8
theoretical_sigma = 0.45 * M
result_theory = downsample(img, scale=M, sigma=theoretical_sigma)
plt.subplot(2, 4, 4)  # 替换最后一个位置放理论值
plt.imshow(result_theory, cmap='gray')
plt.title(f'理论值 σ={theoretical_sigma} (最优)')
plt.axis('off')

plt.suptitle(f"固定M={M}，不同σ对下采样效果的影响", fontsize=16)
plt.tight_layout()
# 保存图片
plt.savefig("part2_result.png", dpi=300, bbox_inches="tight")
print("✅ 第二部分结果已保存为 part2_result.png，在cv-course文件夹里！")
plt.close()