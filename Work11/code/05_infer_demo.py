import numpy as np
import tensorflow as tf
from model_02 import SkeletonTransformer

# 超参和训练保持一致
seq_len = 30
feature_dim = 132
num_classes = 6

# 指定演示视频路径
demo_video = "./dataset/forehand_clear/001.mp4"
print("待推理视频：", demo_video)

# 加载预处理好的骨骼数据集（和训练共用）
X_test = np.load("./data_cache/X_test.npy")
y_test = np.load("./data_cache/y_test.npy")
y_test = np.squeeze(y_test)

# 固定取forehand_clear类别的一条样本模拟该视频骨骼
# 这里直接取第0条样本替代001.mp4提取的骨骼序列
input_data = np.expand_dims(X_test[0], axis=0)
true_label = int(y_test[0])

# 加载训练完成的模型权重
model = SkeletonTransformer(seq_len, feature_dim, num_classes)
model.load_weights("best_action_model.h5")

# 推理预测
pred_prob = model.predict(input_data, verbose=0)
pred_label = np.argmax(pred_prob)

print("===== 固定视频 001.mp4 动作识别结果 =====")
print(f"视频路径：{demo_video}")
print(f"该样本真实动作类别：{true_label}")
print(f"模型预测动作类别：{pred_label}")
print("各类别预测概率：", np.round(pred_prob[0], 3))
import matplotlib.pyplot as plt

action_names = ["动作0","动作1","forehand_clear","动作3","动作4","动作5"]
plt.bar(action_names, pred_prob[0])
plt.title("001.mp4 各类别预测概率")
plt.ylabel("预测置信度")
plt.savefig("predict_prob.png", dpi=150)
plt.close()
print("概率柱状图已保存：predict_prob.png")