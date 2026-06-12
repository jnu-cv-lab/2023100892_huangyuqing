import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 第1步：读取并显示原图 ======================
img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("请把 test.jpg 放在同一文件夹")

h, w = img.shape
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title("原始图像")
plt.axis('off')
plt.savefig("original_image.png", dpi=150, bbox_inches='tight')
plt.close()
print("1/6 完成：original_image.png")

# ====================== 第2步：下采样对比（有无滤波） ======================
scale = 2
new_h, new_w = h//scale, w//scale

img_down_nofilter = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
img_blur = cv2.GaussianBlur(img, (3,3), 1)
img_down_filtered = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title("原图"); plt.axis('off')
plt.subplot(132); plt.imshow(img_down_nofilter, cmap='gray'); plt.title("无滤波下采样"); plt.axis('off')
plt.subplot(133); plt.imshow(img_down_filtered, cmap='gray'); plt.title("高斯滤波下采样"); plt.axis('off')
plt.tight_layout()
plt.savefig("downsampling_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("2/6 完成：downsampling_comparison.png")

# ====================== 第3步：三种插值恢复对比 ======================
img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
img_nn = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_NEAREST)
img_bilinear = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_LINEAR)
img_bicubic = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(12,8))
plt.subplot(231); plt.imshow(img, cmap='gray'); plt.title("原图"); plt.axis('off')
plt.subplot(232); plt.imshow(img_down, cmap='gray'); plt.title("下采样图像"); plt.axis('off')
plt.subplot(233); plt.imshow(img_nn, cmap='gray'); plt.title("最近邻插值"); plt.axis('off')
plt.subplot(234); plt.imshow(img_bilinear, cmap='gray'); plt.title("双线性插值"); plt.axis('off')
plt.subplot(235); plt.imshow(img_bicubic, cmap='gray'); plt.title("双三次插值"); plt.axis('off')
plt.tight_layout()
plt.savefig("interpolation_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("3/6 完成：interpolation_comparison.png")

# ====================== 第4步：空间域整体对比 ======================
img_restore = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_LINEAR)
plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title("原始图像"); plt.axis('off')
plt.subplot(132); plt.imshow(img_down, cmap='gray'); plt.title("下采样图像"); plt.axis('off')
plt.subplot(133); plt.imshow(img_restore, cmap='gray'); plt.title("双线性插值恢复"); plt.axis('off')
plt.tight_layout()
plt.savefig("spatial_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("4/6 完成：spatial_comparison.png")

# ====================== 第5步：傅里叶频谱分析 ======================
def get_spectrum(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    return 20 * np.log(np.abs(fft_shift) + 1)

spec_ori = get_spectrum(img)
spec_down = get_spectrum(img_down)
spec_restore = get_spectrum(img_restore)

plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(spec_ori, cmap='gray'); plt.title("原图频谱"); plt.axis('off')
plt.subplot(132); plt.imshow(spec_down, cmap='gray'); plt.title("下采样图像频谱"); plt.axis('off')
plt.subplot(133); plt.imshow(spec_restore, cmap='gray'); plt.title("双线性恢复图像频谱"); plt.axis('off')
plt.tight_layout()
plt.savefig("spectrum_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("5/6 完成：spectrum_analysis.png")

# ====================== 第6步：DCT分析 + 能量占比 ======================
def dct_energy_ratio(image, ratio=0.1):
    dct_img = cv2.dct(np.float32(image))
    h_dct, w_dct = dct_img.shape
    top_h = int(h_dct * ratio)
    top_w = int(w_dct * ratio)
    low_part = dct_img[:top_h, :top_w]
    total = np.sum(dct_img ** 2)
    low = np.sum(low_part ** 2)
    return round(low / total, 4)

print("\n=== DCT低频能量占比（10%）===")
print(f"原图:         {dct_energy_ratio(img)}")
print(f"最近邻恢复:   {dct_energy_ratio(img_nn)}")
print(f"双线性恢复:   {dct_energy_ratio(img_bilinear)}")
print(f"双三次恢复:   {dct_energy_ratio(img_bicubic)}")

plt.figure(figsize=(12,8))
plt.subplot(221); plt.imshow(np.log(np.abs(cv2.dct(np.float32(img)))+1), cmap='gray'); plt.title("原图DCT系数"); plt.axis('off')
plt.subplot(222); plt.imshow(np.log(np.abs(cv2.dct(np.float32(img_nn)))+1), cmap='gray'); plt.title("最近邻恢复DCT系数"); plt.axis('off')
plt.subplot(223); plt.imshow(np.log(np.abs(cv2.dct(np.float32(img_bilinear)))+1), cmap='gray'); plt.title("双线性恢复DCT系数"); plt.axis('off')
plt.subplot(224); plt.imshow(np.log(np.abs(cv2.dct(np.float32(img_bicubic)))+1), cmap='gray'); plt.title("双三次恢复DCT系数"); plt.axis('off')
plt.tight_layout()
plt.savefig("dct_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n6/6 完成：dct_analysis.png")
print(" 已生成所有图片")