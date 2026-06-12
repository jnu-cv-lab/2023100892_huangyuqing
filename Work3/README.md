# 图像缩小、恢复与频域分析

## 一、实验题目

使用 OpenCV 读入一幅灰度图像，先对图像进行下采样，再分别用不同内插方法恢复图像尺寸，并结合傅里叶变换和 DCT 变换对原图与恢复图进行分析比较。

## 二、多文件编译方法

当项目包含多个 `.cpp` 文件时，需要将所有源文件一起编译。以下是三种常用方法：

### 方法一：修改 tasks.json（VS Code）

将 `args` 中的 `"${file}"` 改为 `"${workspaceFolder}/*.cpp"`。

```json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ 生成活动文件",
            "command": "/usr/bin/g++",
            "args": [
                "${workspaceFolder}/*.cpp",
                "-o",
                "${workspaceFolder}/main",
                "-I/usr/include/opencv4",
                "-lopencv_core",
                "-lopencv_imgproc",
                "-lopencv_highgui",
                "-lopencv_imgcodecs"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ],
    "version": "2.0.0"
}
```

按 `Ctrl+Shift+B` 编译。

### 方法二：终端编译所有文件

```bash
g++ *.cpp -o main `pkg-config --cflags --libs opencv4`
./main
```

### 方法三：指定文件列表编译

```bash
g++ main.cpp utils.cpp -o main `pkg-config --cflags --libs opencv4`
```

## 三、实验内容与代码实现

### 1. 图像读入与预处理

- 以灰度模式读入图像，确保后续处理基于单通道数据。

### 2. 下采样

- 将原图缩小为原来的 **1/2** 和 **1/4**。
- 分别测试两种方式：
  - **直接缩小**：使用 `cv::resize` 直接下采样。
  - **高斯平滑后缩小**：先进行高斯模糊（`cv::GaussianBlur`），再下采样，模拟抗混叠滤波。

### 3. 图像恢复

- 将缩小后的图像恢复到原始尺寸，分别使用：
  - **最近邻内插**（`cv::INTER_NEAREST`）
  - **双线性内插**（`cv::INTER_LINEAR`）
  - **双三次内插**（`cv::INTER_CUBIC`）

### 4. 空间域比较

- 显示原图、缩小图、恢复图，并进行视觉比较。

### 5. 傅里叶变换分析

- 对以下图像计算二维傅里叶变换并显示频谱：
  - 原图
  - 缩小后图像
  - 双线性恢复后的图像
- 要求：
  - 将频谱中心移动到图像中心（`cv::dft` + `fftshift`）
  - 对幅度谱取对数显示（`log(1 + magnitude)`）
  - 比较高频成分差异并分析原因

### 6. DCT 分析

- 对原图和三种恢复图做二维 DCT（`cv::dct`）。
- 显示 DCT 系数图（取对数显示）。
- 统计**左上角低频区域**能量占总能量的比例。
- 比较不同恢复方法下的 DCT 能量分布差异并给出解释。

## 四、核心代码片段

### 下采样与恢复

```cpp
// 直接缩小
cv::resize(src, small_direct, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

// 高斯平滑后缩小
cv::GaussianBlur(src, blurred, cv::Size(5, 5), 1.0);
cv::resize(blurred, small_smooth, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

// 恢复到原尺寸（三种方法）
cv::resize(small, restored_nearest, src.size(), 0, 0, cv::INTER_NEAREST);
cv::resize(small, restored_linear, src.size(), 0, 0, cv::INTER_LINEAR);
cv::resize(small, restored_cubic, src.size(), 0, 0, cv::INTER_CUBIC);
```

### 傅里叶变换与频谱显示

```cpp
cv::dft(complex, complex);
cv::magnitude(planes[0], planes[1], mag);
// 中心化
int cx = mag.cols / 2, cy = mag.rows / 2;
cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
// 对数变换
mag += 1; cv::log(mag, mag);
```

### DCT 与能量统计

```cpp
cv::dct(float_img, dct_coeff);
double total_energy = cv::norm(dct_coeff, cv::NORM_L2);
total_energy = total_energy * total_energy;
double low_energy = cv::norm(dct_coeff(cv::Rect(0, 0, rows/4, cols/4)), cv::NORM_L2);
low_energy = low_energy * low_energy;
double ratio = low_energy / total_energy;
```

## 五、实验结果与分析

> 以下为理论分析，实际运行代码时会生成对应的图像和数据。

### 1. 空间域比较分析

| 恢复方法 | 视觉效果 | 边缘清晰度 | 锯齿/模糊程度 |
| :--- | :--- | :--- | :--- |
| 最近邻内插 | 像素块状明显 | 保持原始值，但出现锯齿 | 严重锯齿 |
| 双线性内插 | 平滑过渡 | 边缘略微模糊 | 无锯齿，但整体偏模糊 |
| 双三次内插 | 最接近原图 | 边缘保持较好 | 清晰度最高 |

**结论**：双三次内插在恢复质量上最优，最近邻内插速度最快但质量最差。

### 2. 傅里叶变换频谱分析

#### 频谱特点对比

| 图像类型 | 高频成分 | 低频成分 | 频谱特点 |
| :--- | :--- | :--- | :--- |
| 原图 | 丰富 | 集中 | 中心亮斑明显，高频分量散布 |
| 缩小图（无预滤波） | 严重混叠 | 丢失 | 频谱出现虚假高频，混叠伪影 |
| 缩小图（高斯预滤波） | 减少 | 保留 | 高频被抑制，混叠减轻 |
| 双线性恢复图 | 部分恢复 | 集中 | 高频有所恢复，但弱于原图 |

#### 原因分析

- **直接下采样产生混叠**：当图像缩小而不做预滤波时，高于新奈奎斯特频率的成分会"折叠"到低频区域，导致频谱中出现虚假的高频分量，表现为伪影和锯齿。
- **高斯预滤波的作用**：高斯滤波是低通滤波器，在下采样前抑制高频成分，避免混叠，使缩小图的频谱更"干净"。
- **恢复算法对频谱的影响**：
  - 最近邻恢复：频谱呈现块状周期性重复，高频信息实际是原始像素的重复。
  - 双线性/双三次恢复：相当于在频域进行低通滤波，会衰减部分真正的高频细节，导致恢复图像比原图"模糊"。

### 3. DCT 能量分布分析

#### 低频能量占比统计（示例数据）

| 图像类型 | 左上角 10% 区域能量占比 | 说明 |
| :--- | :--- | :--- |
| 原图 | 约 85% - 95% | 自然图像能量集中在低频 |
| 最近邻恢复 | 约 70% - 80% | 高频分量（伪影）增加，低频占比下降 |
| 双线性恢复 | 约 88% - 96% | 低频占比略高于原图（高频被平滑） |
| 双三次恢复 | 约 87% - 94% | 接近原图，分布更自然 |

#### 能量分布差异解释

- **DCT 的性质**：DCT 将图像能量压缩到低频区域，尤其适合分析自然图像。
- **恢复方法的影响**：
  - **最近邻内插**：引入高频噪声（锯齿），这些"假高频"会扩散到 DCT 的中高频系数中，导致低频能量占比**下降**。
  - **双线性/双三次内插**：本质是低通滤波操作，会**衰减**真正的高频细节，但也同时**抑制**了混叠产生的高频伪影。最终结果是低频能量占比可能**略高于**原图，但损失了细节。
  - **理想恢复**：应尽可能恢复原图的 DCT 系数分布，双三次内插最接近这一目标。

## 六、实验环境

- 操作系统：Windows / Linux / macOS
- 编译器：g++ 支持 C++11 及以上
- 依赖库：OpenCV 4.x（包含 core、imgproc、imgcodecs、highgui）

## 七、注意事项

- 输入图像路径需正确设置，建议使用绝对路径或确保图像在程序运行目录下。
- 傅里叶变换前最好将图像尺寸扩展到 DFT 最优尺寸（`cv::getOptimalDFTSize`）。
- DCT 要求输入图像为 32 位浮点型（`CV_32F`），且尺寸需为偶数（否则内部会填充）。
- 对数频谱显示时，需要先对所有值加 1，避免对 0 取对数。