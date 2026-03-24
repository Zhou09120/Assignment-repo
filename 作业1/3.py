import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error

# 设置全局随机种子，确保你的“神级结果”每次都能完美复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 骨灰级数据预处理：业务特征构建与异常截断
# ==========================================
def load_and_engineer_data(file_path):
    print(">>> 1. 正在加载数据并执行终极特征工程...")
    df = pd.read_csv(file_path)
    
    # 统一列名，防止原始数据集表头乱码或含特殊字符
    df.columns = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 
                  'coarseaggregate', 'fineaggregate', 'age', 'csMPa']
    
    # 缺失值填补
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())
    
    # 【大招 1】：零值指示变量 (捕捉物理相变)
    df['has_slag'] = (df['slag'] > 0).astype(float)
    df['has_flyash'] = (df['flyash'] > 0).astype(float)
    df['has_superplasticizer'] = (df['superplasticizer'] > 0).astype(float)
    
    # 【大招 2】：深度土木工程特征
    total_binder = df['cement'] + df['slag'] + df['flyash']
    df['water_binder_ratio'] = df['water'] / (total_binder + 1e-5) # 水胶比
    df['sand_ratio'] = df['fineaggregate'] / (df['fineaggregate'] + df['coarseaggregate'] + 1e-5) # 砂率
    
    # 养护龄期对数变换
    df['log_age'] = np.log1p(df['age'])
    
    # 汇总最终的 13 个输入特征
    features = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 
                'coarseaggregate', 'fineaggregate', 
                'has_slag', 'has_flyash', 'has_superplasticizer', 
                'water_binder_ratio', 'sand_ratio', 'log_age']
    
    # 【大招 3】：离群点截断 (Winsorization) - 掐头去尾，消除实验噪音
    for col in features:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    X = df[features].values
    y = df['csMPa'].values.reshape(-1, 1)
    
    # 划分数据集 (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 【大招 4】：高斯化缩放 (Yeo-Johnson)
    scaler_X = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    return (torch.FloatTensor(X_train_scaled), torch.FloatTensor(X_test_scaled), 
            torch.FloatTensor(y_train), torch.FloatTensor(y_test), features)

# ==========================================
# 2. 巅峰网络构建：注意力机制 + 残差网络 (ResNet)
# ==========================================
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
    def forward(self, x):
        return x * torch.sigmoid(self.feature_weights)

# 定义标准残差块 (Residual Block)
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x # 保存输入，用于跨层相加
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out += identity # 核心逻辑：残差连接 (抄近道)
        out = self.relu2(out)
        return out

class AttentionResNet(nn.Module):
    def __init__(self, input_dim=13): 
        super(AttentionResNet, self).__init__()
        self.attention = FeatureAttention(input_dim)
        
        # 输入层映射到 128 维
        self.input_layer = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        
        # 叠加两个残差块，加深网络而不丢失梯度
        self.res_block1 = ResidualBlock(128, drop_rate=0.1)
        self.res_block2 = ResidualBlock(128, drop_rate=0.05)
        
        # 降维并输出
        self.fc_out1 = nn.Linear(128, 32)
        self.relu_out = nn.ReLU()
        self.fc_out2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.fc_out1(x)
        x = self.relu_out(x)
        x = self.fc_out2(x)
        return x

# ==========================================
# 3. 主程序：极限压榨训练
# ==========================================
if __name__ == "__main__":
    # 请确保 concrete_data.csv 文件名和路径正确
    X_train, X_test, y_train, y_test, feature_names = load_and_engineer_data('Concrete_Data_Yeh.csv')

    model = AttentionResNet(input_dim=13)
    criterion = nn.MSELoss()
    
    # 优化器与学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

    epochs = 2000
    train_losses, test_losses = [], []
    
    print("\n>>> 2. 启动 ResNet 核心，开始极限压榨训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train() 
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        scheduler.step() # 步进学习率
        
        # 评估阶段
        model.eval() 
        with torch.no_grad():
            test_outputs = model(X_test)
            t_loss = criterion(test_outputs, y_test)
            
        train_losses.append(loss.item())
        test_losses.append(t_loss.item())
        
        # 打印日志
        if (epoch+1) % 200 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}] | Train MSE: {loss.item():.2f} | Test MSE: {t_loss.item():.2f} | LR: {current_lr:.6f}')

    # ==========================================
    # 4. 终极评估与可视化报告
    # ==========================================
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test).numpy()
        y_test_np = y_test.numpy()
        final_mse = mean_squared_error(y_test_np, final_preds)
        print(f"\n==========================================")
        print(f"🚀 终极神级优化 (ResNet) MSE: {final_mse:.4f}")
        print(f"🚀 终极神级优化 (ResNet) RMSE: {np.sqrt(final_mse):.4f} MPa")
        print(f"==========================================\n")
        
        learned_weights = torch.sigmoid(model.attention.feature_weights).numpy()
        print("🔍 ResNet 自动学习的 13 维特征重要性解码:")
        for name, weight in zip(feature_names, learned_weights):
            print(f" - {name}: {weight:.4f}")

    # 绘制可视化大屏面板
    plt.figure(figsize=(15, 6))
    
    # 图 1: 残差网络学习曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(test_losses, label='Test Loss (MSE)')
    plt.title('ResNet Learning Curves (Train vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图 2: 预测能力分布
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_np, final_preds, alpha=0.7, color='indigo', edgecolor='white', s=50, label='ResNet Predictions')
    min_val = min(y_test_np.min(), final_preds.min())
    max_val = max(y_test_np.max(), final_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Fit (y=x)')
    plt.title('Ultimate Prediction Power: Target vs Output')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()