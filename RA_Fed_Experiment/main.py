import torch
import numpy as np
import argparse
import time
import os
from tqdm import tqdm

# 导入你自己定义的模块
from data_loader import get_dataset, split_noniid
from models import SimpleCNN
from client import FLClient
from server import FLServer
from attacks import pgd_attack

def evaluate_model(model, test_loader, device):
    """同时评估 Clean Accuracy 和 Adversarial Accuracy"""
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    # 这里的循环需要小心，PGD 攻击比较耗时，测试集可以只测一部分如果时间不够
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += target.size(0)

        # 1. Clean Accuracy
        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output.data, 1)
            clean_correct += (pred == target).sum().item()
        
        # 2. Adversarial Accuracy (PGD)
        # 注意：生成对抗样本需要梯度
        with torch.enable_grad():
            adv_data = pgd_attack(model, data, target, device=device)
        
        with torch.no_grad():
            adv_output = model(adv_data)
            _, adv_pred = torch.max(adv_output.data, 1)
            adv_correct += (adv_pred == target).sum().item()

    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total
    return clean_acc, adv_acc

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== Experiment: {args.method} | Agg: {args.agg} | Malicious Rate: {args.mal_rate} ===")

    # 1. 准备数据
    train_data, test_data = get_dataset()
    user_groups = split_noniid(train_data, args.num_users, alpha=0.5)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    # 2. 初始化
    global_model = SimpleCNN().to(device)
    server = FLServer(device)
    
    # 设置恶意节点
    num_malicious = int(args.num_users * args.mal_rate)
    malicious_idxs = np.random.choice(range(args.num_users), num_malicious, replace=False)
    
    clients = []
    for i in range(args.num_users):
        is_mal = i in malicious_idxs
        clients.append(FLClient(i, train_data, user_groups[i], device, is_malicious=is_mal))

    # 3. 联邦训练循环
    for round in tqdm(range(args.epochs)):
        local_weights = []
        
        # 随机选择部分客户端参与本轮
        idxs_users = np.random.choice(range(args.num_users), int(args.num_users * args.frac), replace=False)
        
        for idx in idxs_users:
            # 客户端本地训练
            w = clients[idx].train(global_model, method=args.method, lambda_reg=args.lambda_reg)
            local_weights.append(w)
        
        # 服务器聚合
        global_weights = server.aggregate(local_weights, agg_method=args.agg)
        global_model.load_state_dict(global_weights)

        # 4. 记录与评估 (每5轮)
        if (round+1) % 5 == 0:
            clean_acc, adv_acc = evaluate_model(global_model, test_loader, device)
            print(f"\n[Round {round+1}] Clean Acc: {clean_acc:.2f}% | Robust Acc (PGD): {adv_acc:.2f}%")
            
            save_dir = './results'
            # 保存结果到 CSV，格式: round, clean_acc, adv_acc
            filename = os.path.join(save_dir,f"res_{args.method}_{args.agg}_mal{args.mal_rate}.csv")
            with open(filename, "a") as f:
                f.write(f"{round+1},{clean_acc},{adv_acc}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_users', type=int, default=20)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--mal_rate', type=float, default=0.2, help="恶意节点比例 (0.0 - 0.4)")
    
    # 核心实验参数
    parser.add_argument('--method', type=str, default='fedavg', help="local training: 'fedavg' or 'ra-fed'")
    parser.add_argument('--agg', type=str, default='mean', help="aggregation: 'mean' or 'trimmed_mean'")
    parser.add_argument('--lambda_reg', type=float, default=0.05, help="RA-Fed 的正则化系数")
    
    args = parser.parse_args()
    main(args)