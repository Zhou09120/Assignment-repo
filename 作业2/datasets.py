import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib
# 强制使用无头模式（Agg），防止在无GUI的工作站上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 1. 超参数配置 (Config)
class Config:
    batch_size = 128
    epochs = 50
    learning_rate = 0.05
    weight_decay = 5e-4
    mixup_alpha = 0.2  # Mixup 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "best_svhn_resnet.pth"

cfg = Config()

# 2. 数据增强与加载
# 训练集：AutoAugment + 基础增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

# 测试集：仅归一化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)


# 3. 改进版 ResNet 模型定义
class SVHN_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_ResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        # 针对 32x32 图像的结构修正
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity() 
        # 增加 Dropout 提升泛化
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = SVHN_ResNet().to(cfg.device)


# 4. Mixup 辅助函数
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(cfg.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 5. 训练与评估逻辑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
# 使用 OneCycleLR 策略
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, 
                                          steps_per_epoch=len(train_loader), epochs=cfg.epochs)
# AMP Scaler
scaler = torch.amp.GradScaler('cuda')

def train_epoch(epoch):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        # 应用 Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, cfg.mixup_alpha)
        
        optimizer.zero_grad()
        
        # AMP 混合精度前向传播
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        # AMP 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float() + 
                    (1 - lam) * predicted.eq(targets_b).sum().float()).item()

    return train_loss / total, 100. * correct / total

def evaluate():
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return test_loss / total, 100. * correct / total


# 6. 主循环
if __name__ == '__main__':
    best_acc = 0.0
    print(f"Starting training on {cfg.device}...")
    
    # 初始化用于保存绘图数据的列表
    history_train_loss = []
    history_test_loss = []
    history_train_acc = []
    history_test_acc = []

    for epoch in range(cfg.epochs):
        start_time = time.time()
        
        # 训练和测试
        train_loss, train_acc = train_epoch(epoch)
        test_loss, test_acc = evaluate()
        
        # 记录每轮的数据
        history_train_loss.append(train_loss)
        history_test_loss.append(test_loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), cfg.save_path)
            best_flag = "*"
        else:
            best_flag = ""
            
        time_taken = time.time() - start_time
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
              f"Time: {time_taken:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.5f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% {best_flag}")

    print(f"Training Complete. Best Test Accuracy: {best_acc:.2f}%")

    # ==========================================
    # 8. 结果可视化与图片保存 (适配工作站)
    # ==========================================
    print("正在生成损失与准确率曲线并保存至当前目录...")
    epochs_range = range(1, cfg.epochs + 1)
    
    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, history_test_loss, label='Test Loss', color='red', linewidth=2, linestyle='--')
    plt.title('Training and Test Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_curve.jpg', dpi=300)  # 保存为高分辨率 JPG 文件
    plt.close() # 关闭画布释放内存
    
    # 绘制并保存 Accuracy 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_train_acc, label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(epochs_range, history_test_acc, label='Test Accuracy', color='green', linewidth=2, linestyle='--')
    plt.title('Training and Test Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('accuracy_curve.jpg', dpi=300) # 保存为高分辨率 JPG 文件
    plt.close() # 关闭画布释放内存
    
    print("保存成功！请在当前目录下查看 'loss_curve.jpg' 和 'accuracy_curve.jpg'")