import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from data_loader import get_client_loader

class FLClient:
    def __init__(self, client_id, dataset, data_idxs, device='cuda', is_malicious=False):
        self.client_id = client_id
        self.loader = get_client_loader(dataset, data_idxs)
        self.device = device
        self.is_malicious = is_malicious

    def train(self, global_model, method='fedavg', epochs=2, lr=0.01, lambda_reg=0.0):
        """
        method: 'ra-fed' 现在将启用对抗训练 (Adversarial Training)
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # === 核心修改：RA-Fed 使用对抗训练 ===
                if method == 'ra-fed':
                    model.eval() # 先切换到 eval 模式生成样本
                    
                    # 生成对抗样本 (PGD-3 Step, 快速版)
                    # 这里的 eps=0.03 是 CIFAR-10 的常用扰动阈值
                    x_adv = data.clone().detach()
                    x_adv += torch.empty_like(x_adv).uniform_(-0.03, 0.03)
                    x_adv = torch.clamp(x_adv, 0, 1)
                    
                    for _ in range(3): # 跑 3 步 PGD
                        x_adv.requires_grad = True
                        output = model(x_adv)
                        loss_adv = criterion(output, target)
                        grad = torch.autograd.grad(loss_adv, x_adv)[0]
                        x_adv = x_adv.detach() + (2/255) * grad.sign()
                        x_adv = torch.min(torch.max(x_adv, data - 0.03), data + 0.03)
                        x_adv = torch.clamp(x_adv, 0, 1)
                    
                    model.train() # 切回训练模式
                    
                    # === 混合 Loss (TRADES 思想) ===
                    # 同时优化干净样本和对抗样本
                    # Loss = Clean_Loss + lambda * Robust_Loss
                    optimizer.zero_grad()
                    
                    # 1. 干净样本 Loss
                    out_clean = model(data)
                    loss_clean = criterion(out_clean, target)
                    
                    # 2. 对抗样本 Loss
                    out_adv = model(x_adv.detach())
                    loss_robust = criterion(out_adv, target)
                    
                    # 总 Loss (推荐 lambda=1.0)
                    total_loss = loss_clean + 1.0 * loss_robust
                    
                    total_loss.backward()
                    optimizer.step()
                    
                else:
                    # 普通训练 (FedAvg / Baseline)
                    output = model(data)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================================

        local_w = model.state_dict()
        
        # 拜占庭攻击逻辑
        if self.is_malicious:
            local_w = self._perform_attack(global_model.state_dict(), local_w)
            
        return local_w

    def _perform_attack(self, global_w, local_w):
        """
        支持 Sign Flipping 和 Noise Attack
        """
        malicious_w = copy.deepcopy(local_w)
        
        # 这里我们可以简单的随机切换攻击方式，或者在代码里硬编码
        # 为了实验方便，我们默认还是 Sign Flipping，
        # 但如果是 Noise 攻击，逻辑如下：
        
        # 假设我们在 main.py 里没法传参，这里做一个简单的开关
        # 你可以通过修改这个变量来切换攻击模式： 'sign' 或 'noise'
        attack_type = 'sign' 
        
        for k in malicious_w.keys():
            if 'weight' in k or 'bias' in k:
                if attack_type == 'sign':
                    # 符号翻转
                    update = local_w[k] - global_w[k].to(self.device)
                    malicious_w[k] = global_w[k].to(self.device) - 3.0 * update
                elif attack_type == 'noise':
                    # === 新增：高斯噪声攻击 ===
                    # 向参数中注入强噪声
                    noise = torch.normal(0, 1.0, size=malicious_w[k].shape).to(self.device)
                    malicious_w[k] += noise
                    
        return malicious_w