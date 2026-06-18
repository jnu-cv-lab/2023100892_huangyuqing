import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from model_02 import SkeletonTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 和训练保持一致超参
seq_len = 30
feature_dim = 132
num_classes = 6
class_names = ["动作0", "动作1", "forehand_clear", "动作3", "动作4", "动作5"]

# 加载测试集
X_test = np.load("./data_cache/X_test.npy")
y_test = np.load("./data_cache/y_test.npy")
y_test = np.squeeze(y_test)

# 构建模型并加载最优权重
model = SkeletonTransformer(seq_len, feature_dim, num_classes)
model.load_weights("best_action_model.h5")

# 必须和训练时完全一样的compile配置
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 整体评估测试集
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("="*40)
print(f"测试集损失 loss = {test_loss:.4f}")
print(f"测试集准确率 acc = {test_acc:.4f}")
print("="*40)

# 全部测试样本预测
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# 打印分类报告
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=class_names))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("骨骼动作识别混淆矩阵")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\n混淆矩阵图片已保存为 confusion_matrix.png")