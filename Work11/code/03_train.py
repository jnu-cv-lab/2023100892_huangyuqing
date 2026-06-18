import sys
sys.path.append("./code")

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import matplotlib.pyplot as plt

# 导入模型函数
from model_02 import SkeletonTransformer

# 超参数
seq_len = 30
feature_dim = 132
num_classes = 6
batch_size = 32
epochs = 30

# 加载数据集
X_train = np.load("./data_cache/X_train.npy")
y_train = np.load("./data_cache/y_train.npy")
X_test = np.load("./data_cache/X_test.npy")
y_test = np.load("./data_cache/y_test.npy")

# 统一压缩训练、测试标签多余维度
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# 打印形状校验
print("训练输入尺寸", X_train.shape)
print("训练标签尺寸", y_train.shape)
print("测试输入尺寸", X_test.shape)
print("测试标签尺寸", y_test.shape)

# 创建模型
model = SkeletonTransformer(
    seq_len=seq_len,
    feature_dim=feature_dim,
    num_classes=num_classes
)

model.summary()

# 模型编译
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 训练回调
checkpoint = callbacks.ModelCheckpoint(
    "best_action_model.h5",
    save_best_only=True,
    monitor="val_accuracy"
)
early_stop = callbacks.EarlyStopping(
    patience=5,
    monitor="val_accuracy",
    restore_best_weights=True
)

# 开始训练
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# 绘制并保存训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.savefig("train_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print("训练完成！")
print("最优模型权重：best_action_model.h5")
print("训练曲线图：train_curve.png")