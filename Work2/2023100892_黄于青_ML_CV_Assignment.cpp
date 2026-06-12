#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // -------------------------- 任务1: 读取测试图片 --------------------------
    Mat img = imread("test.jpg"); // 你的测试图片
    if (img.empty()) {
        cout << "无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // -------------------------- 任务2: 输出图像基本信息 --------------------------
    cout << "=== 图像基本信息 ===" << endl;
    cout << "宽度 (Width): " << img.cols << endl;
    cout << "高度 (Height): " << img.rows << endl;
    cout << "通道数 (Channels): " << img.channels() << endl;
    cout << "数据类型 (Type): " << typeToString(img.type()) << endl;

    // -------------------------- 任务3: 显示原图（已注释，避免WSL图形崩溃） --------------------------
    // imshow("Original Image", img);
    // waitKey(0); // 等待按键后关闭窗口

    // -------------------------- 任务4: 转换为灰度图 --------------------------
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY); // BGR转灰度

    // -------------------------- 任务5: 显示灰度图（已注释，避免WSL图形崩溃） --------------------------
    // imshow("Grayscale Image", gray_img);
    // waitKey(0);

    // -------------------------- 任务5: 保存灰度图 --------------------------
    imwrite("gray_test.jpg", gray_img); // 保存为新文件
    cout << "灰度图已保存为: gray_test.jpg" << endl;

    // -------------------------- 任务6: NumPy风格简单操作（C++版） --------------------------
    if (!gray_img.empty()) {
        // 操作1: 输出某个像素值（比如 (100, 100) 位置）
        Vec3b bgr_pixel = img.at<Vec3b>(100, 100); // 彩色图像素（BGR顺序）
        uchar gray_pixel = gray_img.at<uchar>(100, 100); // 灰度图像素
        cout << "=== 像素值信息 ===" << endl;
        cout << "位置 (100,100) 彩色 B: " << (int)bgr_pixel[0]
             << " G: " << (int)bgr_pixel[1]
             << " R: " << (int)bgr_pixel[2] << endl;
        cout << "位置 (100,100) 灰度值: " << (int)gray_pixel << endl;

        // 操作2: 裁剪左上角 100x100 区域并保存
        Rect roi(0, 0, 100, 100); // 左上角 (x,y), 宽w, 高h
        Mat crop_img = img(roi);
        imwrite("crop_test.jpg", crop_img);
        cout << "左上角裁剪图已保存为: crop_test.jpg" << endl;
    }

    destroyAllWindows(); // 关闭所有 OpenCV 窗口（已注释imshow后可保留此行）
    return 0;
}