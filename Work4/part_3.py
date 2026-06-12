import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

# ---------------------- 复用基础函数 ----------------------
def generate_chirp(size=512, min_freq=0.05, max_freq=1.0):
    x = np.linspace(0, size, size)
    freq = min_freq + (max_freq - min_freq) * x / size
    chirp = np.sin(2 * np.pi * freq * x)
    return np.tile(chirp, (size, 1)) * 127.5 + 127.5

def downsample(img, scale=4, sigma=None):
    if sigma is not None:
        img = gaussian_filter(img, sigma=sigma)
    return img[::scale, ::scale]

# ---------------------- 自适应下采样核心函数 ----------------------
# 用Sobel算子计算局部梯度（估计图像复杂度）
def calculate_local_gradient(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve(img, sobel_x)
    gy = convolve(img, sobel_y)
    return np.sqrt(gx**2 + gy**2)

# 自适应下采样主函数
def adaptive_downsample(img, base_scale=4):
    h, w = img.shape
    # 计算梯度图
    grad_map = calculate_local_gradient(img)
    # 归一化梯度，映射M值范围[2, base_scale]
    max_grad = np.max(grad_map)
    if max_grad == 0:
        max_grad = 1e-8
    # 梯度大→M小（采样率高，保留细节）；梯度小→M大（采样率低，压缩）
    local_M = 2 + (grad_map / max_grad) * (base_scale - 2)
    
    # 初始化输出图像
    out_h = h // base_scale
    out_w = w // base_scale
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # 逐块处理
    for y in range(out_h):
        for x in range(out_w):
            # 提取原图对应块
            y_start = y * base_scale
            y_end = (y+1) * base_scale
            x_start = x * base_scale
            x_end = (x+1) * base_scale
            patch = img[y_start:y_end, x_start:x_end]
            
            # 取块内平均M值
            m_patch = local_M[y_start:y_end, x_start:x_end]
            m_val = np.mean(m_patch)
            # 自适应sigma=0.45*M
            sigma = 0.45 * m_val
            
            # 滤波+采样
            filtered_patch = gaussian_filter(patch, sigma=sigma)
            output[y, x] = np.mean(filtered_patch)  # 取块内均值作为采样值

    return output, local_M

# ---------------------- 实验执行 ----------------------
# 生成测试图（用Chirp图，有明显的频率/细节变化）
img = generate_chirp(size=256)
base_scale = 4

# 1. 全图统一下采样（基准对比）
unified_result = downsample(img, scale=base_scale, sigma=0.45*base_scale)

# 2. 自适应下采样
adaptive_result, local_M_map = adaptive_downsample(img, base_scale=base_scale)

# ---------------------- 可视化 ----------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('原图')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(adaptive_result, cmap='gray')
plt.title('自适应下采样结果')
plt.axis('off')

plt.subplot(2, 2, 3)
im = plt.imshow(local_M_map, cmap='jet')
plt.title('局部M值分布（红=高细节，蓝=低细节）')
plt.colorbar(im)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(unified_result, cmap='gray')
plt.title(f'全图统一下采样（M={base_scale}，σ={0.45*base_scale}）')
plt.axis('off')

plt.tight_layout()
# 保存图片
plt.savefig("part3_result.png", dpi=300, bbox_inches="tight")
print("✅ 第三部分结果已保存为 part3_result.png，在cv-course文件夹里！")
plt.close()