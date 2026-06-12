import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------- 1. 环境检查 ----------------------
print("="*30)
print("环境检查:")
print("PyTorch 版本:", torch.__version__)
print("GPU 是否可用:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)
print("="*30)

# ---------------------- 2. 数据加载与预处理 ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# ---------------------- 3. 显示训练样本 ----------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

dataiter = iter(trainloader)
images, labels = next(dataiter)

plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.tight_layout()
plt.savefig("work8_train_samples.png")
plt.close()
print("✅ 训练样本图片已保存为 work8_train_samples.png")

# ---------------------- 4. 定义 CNN 模型 ----------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)
print("\n✅ 模型结构定义完成:")
print(net)

# ---------------------- 5. 损失函数与优化器 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ---------------------- 6. 训练与验证 ----------------------
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

epochs = 5
print("\n" + "="*30)
print("开始训练:")
print("="*30)

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    net.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valloader)
    val_acc = 100 * val_correct / val_total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train | Loss: {train_loss:.3f} | Acc: {train_acc:.2f}%")
    print(f"Val   | Loss: {val_loss:.3f} | Acc: {val_acc:.2f}%\n")

print("✅ 训练完成！")

# ---------------------- 7. 测试集评估 ----------------------
net.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(testloader)
test_acc = 100 * test_correct / test_total

print("\n" + "="*30)
print("测试集最终结果:")
print("="*30)
print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ---------------------- 8. 测试集预测可视化 ----------------------
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(12, 5))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i].cpu())
    plt.title(f"True:{classes[labels[i]]}\nPred:{classes[predicted[i]]}")
plt.tight_layout()
plt.savefig("work8_test_predictions.png")
plt.close()
print("\n✅ 测试预测图片已保存为 work8_test_predictions.png")

# ---------------------- 9. 绘制 Loss/Acc 曲线 ----------------------
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss_list, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("work8_loss_curve.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc')
plt.plot(range(1, epochs+1), val_acc_list, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig("work8_acc_curve.png")
plt.close()
print("✅ 损失和准确率曲线已保存完成！")