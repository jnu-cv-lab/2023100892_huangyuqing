import sys
sys.path.append("./")
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ===================== 配置参数 =====================
TARGET_FRAMES = 30  # 统一帧数T=30
KEYPOINT_NUM = 33
FEAT_PER_KPT = 4    # x,y,z,visibility
FRAME_DIM = KEYPOINT_NUM * FEAT_PER_KPT  # 132
DATASET_ROOT = "./dataset"
SAVE_DIR = "./data_cache"
TEST_RATIO = 0.2

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 加载类别映射
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

def normalize_skeleton(keypoints):
    """骨架归一化：髋部中心为原点，肩宽缩放"""
    # keypoints shape [33,4]
    left_hip = keypoints[23, :2]
    right_hip = keypoints[24, :2]
    hip_center = (left_hip + right_hip) / 2.0

    left_shoulder = keypoints[11, :2]
    right_shoulder = keypoints[12, :2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6

    # 平移
    keypoints[:, :2] = keypoints[:, :2] - hip_center
    # 缩放
    keypoints[:, :2] = keypoints[:, :2] / shoulder_width
    return keypoints

def video_to_skeleton(vid_path):
    """单个视频提取骨架序列，返回 [帧数, 132]"""
    cap = cv2.VideoCapture(vid_path)
    frame_skels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 转RGB输入mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            kpts = []
            for lm in res.pose_landmarks.landmark:
                kpts.append([lm.x, lm.y, lm.z, lm.visibility])
            kpts = np.array(kpts)  # [33,4]
            kpts = normalize_skeleton(kpts)
            frame_skels.append(kpts.flatten())  # 展平132维
        else:
            # 无人体填充0
            frame_skels.append(np.zeros(FRAME_DIM))
    cap.release()
    seq = np.array(frame_skels)  # [orig_T, 132]
    return seq

def resample_sequence(seq, target_len):
    """任意长度序列重采样到固定target_len帧"""
    orig_len = seq.shape[0]
    if orig_len == target_len:
        return seq
    # 等间隔采样
    new_idx = np.linspace(0, orig_len-1, target_len, dtype=int)
    return seq[new_idx]

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []
    all_label = []

    # 文件夹名称映射数字标签（适配你无数字前缀的数据集）
    name2label = {
        "forehand_drive": 0,
        "forehand_lift": 1,
        "forehand_net_shot": 2,
        "forehand_clear": 3,
        "backhand_drive": 4,
        "backhand_net_shot": 5
    }
    cls_folders = sorted(os.listdir(DATASET_ROOT))

    for cls_name in cls_folders:
        cls_path = os.path.join(DATASET_ROOT, cls_name)
        if not os.path.isdir(cls_path):
            continue
        label = name2label[cls_name]
        video_list = [f for f in os.listdir(cls_path) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(f"处理类别 {label} {label_map[str(label)]}, 视频数量：{len(video_list)}")
        for vid in tqdm(video_list):
            vid_full = os.path.join(cls_path, vid)
            skel_seq = video_to_skeleton(vid_full)
            fixed_seq = resample_sequence(skel_seq, TARGET_FRAMES)
            all_data.append(fixed_seq)
            all_label.append(label)

    # 转为numpy数组
    X = np.array(all_data)  # [N,30,132]
    y = np.array(all_label) # [N,]
    print(f"总样本量: {X.shape[0]}")

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42, stratify=y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 保存npy文件
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
    print("预处理完成，数据已保存至 ./data_cache")