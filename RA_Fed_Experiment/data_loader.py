import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataset(root='./data'):
    """加载 CIFAR-10 数据集并进行预处理"""
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=trans_train)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=trans_test)
    return train_dataset, test_dataset

def split_noniid(train_dataset, num_clients, alpha=0.5):
    """
    核心代码：使用 Dirichlet 分布划分 Non-IID 数据。
    alpha 越小，Non-IID 程度越高（数据分布越极端）。
    """
    n_classes = 10
    label_list = np.array(train_dataset.targets)
    min_size = 0
    min_require_size = 10
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(n_classes):
            idx_k = np.where(label_list == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # 平衡性调整，避免某客户端数据过少
            proportions = np.array([p * (len(idx_j) < len(label_list) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        net_dataidx_map[j] = idx_batch[j]
    
    return net_dataidx_map

def get_client_loader(train_dataset, data_idxs, batch_size=64):
    """为特定客户端生成 DataLoader"""
    return DataLoader(Subset(train_dataset, data_idxs), batch_size=batch_size, shuffle=True)