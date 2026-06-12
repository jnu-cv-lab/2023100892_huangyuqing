# 第8课：传统机器学习方法用于图像分类
import numpy as np
import matplotlib
matplotlib.use('Agg')  # WSL无界面保存图片
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 任务1：数据准备
# ==============================
digits = load_digits()
X = digits.data       # (1797,64)
y = digits.target     # 标签0~9
images = digits.images  # (1797,8,8)

print("===== 任务1：数据准备 =====")
print("图像总数：", len(images))
print("每张图像大小：", images.shape[1:])
print("类别标签：", np.unique(y))

# 显示样本图
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(str(y[i]))
    plt.axis('off')
plt.savefig("task1_samples.png", dpi=300)
plt.close()
print("✅ 样本图已保存：task1_samples.png")

# ==============================
# 任务2：数据划分 75%/25%
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("\n===== 任务2：数据划分 =====")
print("训练集数量：", X_train.shape[0])
print("测试集数量：", X_test.shape[0])
print("训练集：用于模型学习规律")
print("测试集：用于评估泛化能力，不参与训练")

# ==============================
# 任务3：特征表示
# ==============================
print("\n===== 任务3：特征表示 =====")
print("8×8图像按行展平 → 1×64向量")
print("传统模型只能处理向量，不能直接输入图片")
print("原始像素优点：简单、无需手工设计")
print("原始像素缺点：对平移、旋转、光照极敏感")

# ==============================
# 任务4：模型训练（6种全部训练）
# ==============================
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
print("\n===== 任务4：模型准确率 =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:20s}: {acc:.4f}")

# ==============================
# 任务5：结果表格
# ==============================
print("\n===== 任务5：准确率表格 =====")
print("| 模型 | 测试准确率 |")
print("|------|------------|")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")

# ==============================
# 任务6：错误样本分析（用SVM）
# ==============================
print("\n===== 任务6：错误分析 =====")
best_model = models["SVM"]
y_pred = best_model.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("task6_confusion_matrix.png", dpi=300)
plt.close()
print("✅ 混淆矩阵已保存")

# 错误样本
wrong_idx = np.where(y_pred != y_test)[0]
plt.figure(figsize=(12,6))
for i, idx in enumerate(wrong_idx[:6]):
    plt.subplot(2,3,i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis('off')
plt.savefig("task6_wrong_samples.png", dpi=300)
plt.close()
print("✅ 错误样本图已保存")

print("\n🎉 全部任务1~6 完成！")