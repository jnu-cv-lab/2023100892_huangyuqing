# ========== WSL 不弹窗 ==========
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ========== 设备 & 数据 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备：", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================
# 任务1：复用CNN模型 + 训练 + 记录过程
# ============================
print("\n====== 任务1：复用CNN模型 ======\n")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练 + 记录过程
history = {
    'train_loss':[], 'train_acc':[],
    'val_loss':[], 'val_acc':[]
}

epochs = 9
for epoch in range(epochs):
    # 训练
    model.train()
    tl, ta, n = 0,0,0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        tl += loss.item()
        ta += (out.argmax(1)==y).sum().item()
        n += y.size(0)

    train_loss = tl/len(train_loader)
    train_acc = 100*ta/n

    # 测试
    model.eval()
    vl, va, n = 0,0,0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            vl += criterion(out,y).item()
            va += (out.argmax(1)==y).sum().item()
            n += y.size(0)

    val_loss = vl/len(test_loader)
    val_acc = 100*va/n

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}")
    print(f"训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.2f}%")
    print(f"验证损失：{val_loss:.4f} | 验证准确率：{val_acc:.2f}%\n")

# 画图：任务1训练过程
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.title('任务1 - Loss曲线')
plt.legend()

plt.subplot(122)
plt.plot(history['train_acc'], label='train acc')
plt.plot(history['val_acc'], label='val acc')
plt.title('任务1 - Accuracy曲线')
plt.legend()

plt.savefig('task1_process.png', dpi=300)
plt.close()
print("✅ 任务1 训练过程已保存：task1_process.png")

# ============================
# 任务2：优化器对比 SGD / Momentum / Adam
# ============================
print("\n====== 任务2：优化器对比 ======\n")

def get_train_process(model, opt, epochs=3):
    h = {'tl':[],'ta':[],'vl':[],'va':[]}
    for e in range(epochs):
        model.train()
        tl,ta,n=0,0,0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            o=model(x)
            loss=criterion(o,y)
            loss.backward()
            opt.step()
            tl+=loss.item()
            ta+=(o.argmax(1)==y).sum().item()
            n+=y.size(0)
        h['tl'].append(tl/len(train_loader))
        h['ta'].append(100*ta/n)

        model.eval()
        vl,va,n=0,0,0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device)
                o=model(x)
                vl+=criterion(o,y).item()
                va+=(o.argmax(1)==y).sum().item()
                n+=y.size(0)
        h['vl'].append(vl/len(test_loader))
        h['va'].append(100*va/n)
    return h

model_sgd = CNN().to(device)
model_mom = CNN().to(device)
model_adam2 = CNN().to(device)

h_sgd = get_train_process(model_sgd, optim.SGD(model_sgd.parameters(), lr=0.01))
h_mom = get_train_process(model_mom, optim.SGD(model_mom.parameters(), lr=0.01, momentum=0.9))
h_adam2 = get_train_process(model_adam2, optim.Adam(model_adam2.parameters(), lr=0.001))

plt.figure(figsize=(12,5))
plt.subplot(121);plt.plot(h_sgd['va'],label='SGD');plt.plot(h_mom['va'],label='Momentum');plt.plot(h_adam2['va'],label='Adam');plt.title('优化器-准确率');plt.legend()
plt.subplot(122);plt.plot(h_sgd['vl'],label='SGD');plt.plot(h_mom['vl'],label='Momentum');plt.plot(h_adam2['vl'],label='Adam');plt.title('优化器-损失');plt.legend()
plt.savefig('task2_optimizer.png',dpi=300);plt.close()
print("✅ 任务2 完成：task2_optimizer.png")

# ============================
# 任务3：学习率对比 0.1 0.01 0.001
# ============================
print("\n====== 任务3：学习率对比 ======\n")
lrs = [0.1,0.01,0.001]
l_h = []
for lr in lrs:
    m=CNN().to(device)
    h=get_train_process(m, optim.Adam(m.parameters(), lr=lr))
    l_h.append(h)

plt.figure(figsize=(12,5))
plt.subplot(121)
for i,lr in enumerate(lrs): plt.plot(l_h[i]['va'],label=f'lr={lr}')
plt.title('学习率-准确率');plt.legend()
plt.subplot(122)
for i,lr in enumerate(lrs): plt.plot(l_h[i]['vl'],label=f'lr={lr}')
plt.title('学习率-损失');plt.legend()
plt.savefig('task3_lr.png',dpi=300);plt.close()
print("✅ 任务3 完成：task3_lr.png")

# ============================
# 任务4：卷积核可视化
# ============================
print("\n====== 任务4：卷积核 ======\n")
kernels = model_adam2.conv1.weight.data.cpu()
plt.figure(figsize=(10,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(kernels[i,0], cmap='gray')
    plt.title(f'kernel{i+1}')
    plt.axis('off')
plt.savefig('task4_kernels.png',dpi=300);plt.close()
print("✅ 任务4 完成：task4_kernels.png")

# ============================
# 任务5：特征图可视化
# ============================
print("\n====== 任务5：特征图 ======\n")
img, label = test_dataset[0]
img = img.unsqueeze(0).to(device)
with torch.no_grad():
    feat = model_adam2.relu(model_adam2.conv1(img))

plt.figure(figsize=(10,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(feat[0,i].cpu(), cmap='gray')
    plt.title(f'feat{i+1}')
    plt.axis('off')
plt.savefig('task5_feat.png',dpi=300);plt.close()
print("✅ 任务5 完成：task5_feat.png")

# ============================
# 任务6：错误样本
# ============================
print("\n====== 任务6：错误样本 ======\n")
err = []
model_adam2.eval()
with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to(device),y.to(device)
        p=model_adam2(x).argmax(1)
        idx=(p!=y).nonzero()[:,0]
        for i in idx:
            err.append((x[i].cpu(), y[i].item(), p[i].item()))
            if len(err)>=8: break
        if len(err)>=8: break

plt.figure(figsize=(10,6))
for i,(im,t,p) in enumerate(err[:8]):
    plt.subplot(2,4,i+1)
    plt.imshow(im[0], cmap='gray')
    plt.title(f'true:{t}\npred:{p}')
    plt.axis('off')
plt.savefig('task6_error.png',dpi=300);plt.close()
print("✅ 任务6 完成：task6_error.png")

# ============================
# 任务7：混淆矩阵
# ============================
print("\n====== 任务7：混淆矩阵 ======\n")
all_pred=[]
all_true=[]
model_adam2.eval()
with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        p=model_adam2(x).argmax(1).cpu()
        all_pred.extend(p.numpy())
        all_true.extend(y.numpy())

cm=confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('预测');plt.ylabel('真实')
plt.savefig('task7_cm.png',dpi=300);plt.close()
print("✅ 任务7 完成：task7_cm.png")

print("\n🎉 第9次实验 全部完成！")