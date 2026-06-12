# 几何变换实验：相似、仿射、透视变换

## 一、实验目的

1. 理解相似变换、仿射变换、透视变换的数学原理与几何特性
2. 观察三类变换对直线、平行线、垂直关系、圆等几何性质的影响
3. 掌握 OpenCV 中 `warpAffine` 和 `warpPerspective` 的使用方法
4. 学会对存在透视畸变的平面图像进行校正

## 二、实验内容

### 1. 生成测试图像

生成一张包含以下几何元素的测试图：
- 矩形
- 圆形
- 一组平行线（水平方向）
- 一组平行线（垂直方向）
- 两条互相垂直的直线

### 2. 施加三类变换

| 变换类型 | 自由度 | OpenCV 函数 | 变换矩阵形式 |
|:---|:---|:---|:---|
| 相似变换 | 4 | `cv::getRotationMatrix2D` | 旋转 + 平移 + 均匀缩放 |
| 仿射变换 | 6 | `cv::getAffineTransform` | 2×3 矩阵 |
| 透视变换 | 8 | `cv::getPerspectiveTransform` | 3×3 矩阵 |

### 3. 观察记录

对每类变换，观察并记录以下几何性质的变化：

| 性质 | 相似变换 | 仿射变换 | 透视变换 |
|:---|:---|:---|:---|
| 直线是否保持为直线 | | | |
| 平行线是否仍保持平行 | | | |
| 两条垂直线是否仍垂直 | | | |
| 圆是否仍保持为圆 | | | |

### 4. 透视畸变校正（实战）

- 拍摄一张放在桌上的 A4 纸（包含文字和表格）
- 选择纸张的四个角点作为源点
- 设定目标点为 A4 纸的标准矩形
- 计算透视变换矩阵并校正
- 评价校正效果

## 三、核心代码片段

### 生成测试图像

```cpp
cv::Mat generateTestImage(int width, int height) {
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    
    // 矩形
    cv::rectangle(img, cv::Point(50, 50), cv::Point(200, 150), cv::Scalar(0, 255, 0), 2);
    
    // 圆形
    cv::circle(img, cv::Point(300, 100), 50, cv::Scalar(255, 0, 0), 2);
    
    // 水平平行线
    for (int y = 200; y <= 300; y += 30) {
        cv::line(img, cv::Point(50, y), cv::Point(350, y), cv::Scalar(0, 0, 255), 2);
    }
    
    // 垂直平行线
    for (int x = 50; x <= 350; x += 30) {
        cv::line(img, cv::Point(x, 200), cv::Point(x, 350), cv::Scalar(0, 0, 255), 2);
    }
    
    // 两条垂直线（水平线和竖直线）
    cv::line(img, cv::Point(400, 50), cv::Point(550, 50), cv::Scalar(255, 255, 0), 2);
    cv::line(img, cv::Point(400, 50), cv::Point(400, 200), cv::Scalar(255, 255, 0), 2);
    
    return img;
}
```

### 相似变换（旋转 + 缩放 + 平移）

```cpp
cv::Mat applySimilarityTransform(const cv::Mat& src, double angle, double scale, double tx, double ty) {
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(src.cols/2, src.rows/2), angle, scale);
    M.at<double>(0, 2) += tx;
    M.at<double>(1, 2) += ty;
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size());
    return dst;
}
```

### 仿射变换（基于三个点对）

```cpp
cv::Mat applyAffineTransform(const cv::Mat& src, cv::Point2f srcPoints[3], cv::Point2f dstPoints[3]) {
    cv::Mat M = cv::getAffineTransform(srcPoints, dstPoints);
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size());
    return dst;
}
```

### 透视变换（基于四个点对）

```cpp
cv::Mat applyPerspectiveTransform(const cv::Mat& src, cv::Point2f srcPoints[4], cv::Point2f dstPoints[4]) {
    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::Mat dst;
    cv::warpPerspective(src, dst, M, src.size());
    return dst;
}
```

### 透视畸变校正（A4 纸校正）

```cpp
cv::Mat correctPerspective(const cv::Mat& src, std::vector<cv::Point2f>& corners) {
    // 目标点：A4 纸标准矩形（假设 300×424 像素，保持 1:√2 比例）
    float width = 300;
    float height = 424;
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(width - 1, 0),
        cv::Point2f(width - 1, height - 1),
        cv::Point2f(0, height - 1)
    };
    
    cv::Mat M = cv::getPerspectiveTransform(corners, dstPoints);
    cv::Mat dst;
    cv::warpPerspective(src, dst, M, cv::Size(width, height));
    return dst;
}
```

## 四、实验结果与分析

### 1. 三类变换对几何性质的影响总结

| 几何性质 | 相似变换 | 仿射变换 | 透视变换 |
|:---|:---|:---|:---|
| **直线保持为直线** | ✅ 是 | ✅ 是 | ✅ 是 |
| **平行线保持平行** | ✅ 是 | ✅ 是 | ❌ 否 |
| **垂直线保持垂直** | ✅ 是 | ❌ 否 | ❌ 否 |
| **圆保持为圆** | ✅ 是 | ❌ 否（变为椭圆） | ❌ 否（变为椭圆/不规则曲线） |

### 2. 详细分析

#### 相似变换

- **数学形式**：旋转 + 平移 + 均匀缩放
- **性质**：保持角度、保持形状、保持长度比例
- **结论**：所有被考察的几何性质均得到保持

#### 仿射变换

- **数学形式**：线性变换（2×2 矩阵）+ 平移
- **性质**：
  - 保持直线的直线性
  - 保持平行性（平行线变换后仍平行）
  - 不保持垂直性
  - 圆变为椭圆
- **结论**：平行线性质是仿射变换的核心不变性

#### 透视变换

- **数学形式**：齐次坐标下的 3×3 矩阵，最后一行用于产生透视效果
- **性质**：
  - 保持直线的直线性（最重要）
  - 不保持平行性（平行线会交于灭点）
  - 不保持垂直性
  - 圆变为椭圆或更复杂的二次曲线
- **结论**：透视变换是中心投影的数学描述，只保证直线性

### 3. 透视畸变校正结果

| 评价指标 | 结果 |
|:---|:---|
| 文字可读性 | ✅ 校正后文字清晰可读 |
| 表格直线性 | ✅ 横平竖直 |
| 长宽比例 | 恢复为 A4 的标准比例 |
| 边缘变形 | 无明显的残余畸变 |

**用户满意度**：满意。校正后的图像文字不再倾斜，表格线条规整，便于后续 OCR 或文档存档。

## 五、思考题

### 问题：为什么透视变换能保持直线性但无法保持平行性？

**答案**：

透视变换模拟了人眼或相机成像的**中心投影**过程。在中心投影下：
- 三维空间中的直线投影到二维图像平面仍然是直线（这是射影几何的基本性质）
- 平行线在无限远处相交于一个**灭点**，因此变换后不再平行

### 问题：如何选择合适的变换类型？

| 应用场景 | 推荐变换 |
|:---|:---|
| 图像旋转、缩放、平移 | 相似变换 |
| 纠正倾斜的文字（俯仰/偏航） | 仿射变换 |
| 纠正透视畸变（如翻拍文档） | 透视变换 |
| 图像配准中的近似对齐 | 仿射变换 |
| 多视角图像拼接 | 透视变换（单应性矩阵） |

## 六、实验环境

- 操作系统：Windows / Linux / macOS
- 编译器：g++ 支持 C++11 及以上
- 依赖库：OpenCV 4.x

## 七、注意事项

- 透视变换需要至少 4 个点对来计算 3×3 单应性矩阵
- 仿射变换需要至少 3 个点对
- 相似变换需要至少 2 个点对（或指定角度+缩放）
- 进行透视校正时，源点应按顺时针或逆时针顺序选取
- 变换后的图像可能存在黑边，可用 `cv::warpAffine` 的 `borderMode` 参数控制