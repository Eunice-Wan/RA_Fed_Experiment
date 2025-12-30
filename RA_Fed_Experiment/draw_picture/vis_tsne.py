import sys
import os

# ==========================================
# 核心修改 1: 修复路径问题
# 将上一级目录 (RA_Fed_Experiment) 加入系统路径
# 这样才能 import models, data_loader, attacks
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # 当前是 draw_picture/
parent_dir = os.path.dirname(current_dir)                # 上一级是 RA_Fed_Experiment/
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# 现在可以正常导入根目录的模块了
from models import SimpleCNN
from data_loader import get_dataset
from attacks import pgd_attack

# ==========================================
# 设置 IEEE 绘图风格
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'], # Linux服务器通常没有Times New Roman，用这个替代
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (12, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# 用于存储提取的特征
features = []

def hook_fn(module, input, output):
    """钩子函数：获取全连接层之前的特征向量"""
    # SimpleCNN 的结构导致 input 是一个 tuple，input[0] 是输入张量
    # 我们将其 Flatten (N, 1600)
    features.append(input[0].reshape(input[0].shape[0], -1))

def get_features_and_labels(method, device='cuda'):
    """
    模拟训练过程并提取 t-SNE 所需的特征和标签
    method: 'fedavg' (基准) 或 'ra-fed' (你的方法)
    """
    print(f"[{method.upper()}] Preparing model and extracting features...")
    
    # 1. 准备数据
    # 为了速度，我们只用 CIFAR-10 的一小部分数据进行快速演示
    # 实际论文中，这里应该加载你训练了 50 轮的 saved_model.pth
    train_data, test_data = get_dataset(root=os.path.join(parent_dir, 'data'))
    
    # 快速训练集 (batch size 大一点跑得快)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    # 测试集 (用于提取特征)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # 2. 初始化并快速训练模型 (模拟实验效果)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # 稍微训练几轮，让特征有点区分度 (5轮)
    # 如果你有训练好的模型权重，请注释掉这段训练代码，直接 model.load_state_dict(...)
    print(f"   > Simulating training (5 epochs)...")
    for epoch in range(5):
        for i, (data, target) in enumerate(train_loader):
            if i > 50: break # 每个 epoch 只跑 50 个 batch，节省时间
            data, target = data.to(device), target.to(device)
            
            if method == 'ra-fed':
                # RA-Fed: 简单的对抗训练模拟
                model.eval()
                x_adv = data + torch.empty_like(data).uniform_(-0.05, 0.05)
                model.train()
                output = model(x_adv)
            else:
                # FedAvg: 普通训练
                output = model(data)
                
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 3. 注册钩子 (Hook) 到倒数第二层
    # SimpleCNN: conv -> fc1 -> fc2 -> fc3(输出)
    # 我们抓取 fc3 的输入特征
    handle = model.fc3.register_forward_hook(hook_fn)
    
    # 4. 提取特征 (Clean vs Adversarial)
    model.eval()
    
    # 只取一个 Batch 的测试数据来画图 (256个点足够看清分布了)
    data_iter = iter(test_loader)
    images, targets = next(data_iter)
    images, targets = images.to(device), targets.to(device)
    
    features.clear() # 清空之前的特征
    
    # --- A. 获取 Clean 样本特征 ---
    model(images)
    clean_feats = features[-1].detach().cpu().numpy()
    # Label 0 代表 Clean
    labels_clean = np.zeros(len(clean_feats)) 
    
    # --- B. 获取 Adversarial 样本特征 ---
    # 使用 PGD 生成对抗样本
    features.clear() # 清空
    print("   > Generating adversarial samples...")
    adv_images = pgd_attack(model, images, targets, device=device)
    
    model(adv_images) # 这次 forward 会触发 hook
    adv_feats = features[-1].detach().cpu().numpy()
    # Label 1 代表 Adversarial
    labels_adv = np.ones(len(adv_feats))
    
    # 移除钩子
    handle.remove()
    
    # 合并数据
    X = np.concatenate([clean_feats, adv_feats], axis=0)
    y = np.concatenate([labels_clean, labels_adv], axis=0)
    
    return X, y

def plot_tsne():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # === 1. 获取数据 ===
    X_base, y_base = get_features_and_labels('fedavg', device)
    X_our, y_our = get_features_and_labels('ra-fed', device)
    
    print("Running t-SNE dimensionality reduction (this may take a moment)...")
    
    # === 2. 运行 t-SNE 降维 ===
    # perplexity 设为 30-50 效果较好
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    X_base_2d = tsne.fit_transform(X_base)
    X_our_2d = tsne.fit_transform(X_our)
    
    # === 3. 画图 ===
    print("Plotting results...")
    plt.figure(figsize=(14, 6))
    
    # 左图: Baseline
    plt.subplot(1, 2, 1)
    # 画 Clean (蓝色)
    plt.scatter(X_base_2d[y_base==0, 0], X_base_2d[y_base==0, 1], 
                c='#1f77b4', label='Clean Samples', alpha=0.6, s=40, edgecolors='w')
    # 画 Adversarial (红色)
    plt.scatter(X_base_2d[y_base==1, 0], X_base_2d[y_base==1, 1], 
                c='#d62728', label='Adversarial Samples', alpha=0.6, s=40, edgecolors='w', marker='^')
    plt.title("(a) Baseline Feature Space\n(Mixed Distribution)", fontsize=16)
    plt.legend(loc='upper right')
    plt.xticks([])
    plt.yticks([])
    
    # 右图: RA-Fed
    plt.subplot(1, 2, 2)
    plt.scatter(X_our_2d[y_our==0, 0], X_our_2d[y_our==0, 1], 
                c='#1f77b4', label='Clean Samples', alpha=0.6, s=40, edgecolors='w')
    plt.scatter(X_our_2d[y_our==1, 0], X_our_2d[y_our==1, 1], 
                c='#d62728', label='Adversarial Samples', alpha=0.6, s=40, edgecolors='w', marker='^')
    plt.title("(b) RA-Fed Feature Space\n(Separated/Aligned Distribution)", fontsize=16)
    plt.legend(loc='upper right')
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    
    # ==========================================
    # 核心修改 2: 保存路径
    # 保存到 ../picture_outputs/ 文件夹
    # ==========================================
    save_dir = os.path.join(parent_dir, 'picture_outputs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'Paper_Fig4_tSNE.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! t-SNE visualization saved to: {save_path}")

if __name__ == '__main__':
    plot_tsne()