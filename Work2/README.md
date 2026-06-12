# OpenCV 图像处理基础任务

## 多文件编译方法

当项目包含多个 `.cpp` 文件时，需要将所有源文件一起编译。以下是三种常用方法：

### 方法一：修改 tasks.json（VS Code）

将 `args` 中的 `"${file}"` 改为 `"${workspaceFolder}/*.cpp"`，表示编译当前文件夹下的所有 `.cpp` 文件。

```json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ 生成活动文件",
            "command": "/usr/bin/g++",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${workspaceFolder}/*.cpp",
                "-o",
                "${workspaceFolder}/main",
                "-I/usr/include/opencv4",
                "-lopencv_core",
                "-lopencv_imgproc",
                "-lopencv_highgui",
                "-lopencv_imgcodecs"
            ],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "problemMatcher": ["$gcc"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ],
    "version": "2.0.0"
}
```

配置完成后，按 Ctrl+Shift+B 即可编译所有文件。

方法二：终端编译所有文件

```bash
# 编译当前目录下所有 .cpp 文件
g++ *.cpp -o main `pkg-config --cflags --libs opencv4`

# 运行
./main
```

方法三：指定文件列表编译

```bash
# 只编译指定的几个文件
g++ main.cpp utils.cpp image.cpp -o main `pkg-config --cflags --libs opencv4`
```

---

实验目的

1. 掌握 OpenCV 在 C++ 环境下的配置方法
2. 学习图像的读取、显示、转换和保存
3. 熟悉 Mat 对象的基本操作

实验环境

· 操作系统：Windows / Linux / macOS
· 编译器：支持 C++11 及以上（如 g++、MSVC）
· 依赖库：OpenCV 4.x

功能说明

1. 读取图像

程序默认读取 test.jpg 文件，支持常见格式（jpg、png 等）。

2. 输出图像信息

在终端打印图像的：

· 宽度（width）
· 高度（height）
· 通道数（channels）
· 数据类型（如 CV_8UC3）

3. 显示原图

使用 OpenCV 的 imshow 显示原始图像，按任意键关闭窗口。

4. 转换为灰度图

将彩色图像转换为灰度图像，并显示灰度图。

5. 保存灰度图

将灰度图像保存为 gray_image.jpg。

6. 像素操作（NumPy 类比）

在 C++ 中实现类似 NumPy 的操作，例如：

· 输出左上角第一个像素的 BGR 值
· 裁剪图像左上角 50x50 区域并保存为 cropped.jpg

代码结构

```
main.cpp               # 主程序源码
test.jpg               # 输入图像（示例）
gray_image.jpg         # 输出灰度图
cropped.jpg            # 输出裁剪图
README.md              # 项目说明
```

核心代码片段

读取与显示

```cpp
cv::Mat img = cv::imread("test.jpg");
cv::imshow("Original Image", img);
cv::waitKey(0);
```

灰度转换

```cpp
cv::Mat gray;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
```

裁剪操作

```cpp
cv::Rect cropRegion(0, 0, 50, 50);
cv::Mat cropped = img(cropRegion);
cv::imwrite("cropped.jpg", cropped);
```

注意事项

· 确保输入图像 test.jpg 存在于程序运行目录
· 若使用 pkg-config 失败，可手动指定 OpenCV 路径，例如：
  ```bash
  g++ -o image_processing main.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui
  ```