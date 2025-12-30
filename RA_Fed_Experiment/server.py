import torch
import copy

class FLServer:
    def __init__(self, device='cuda'):
        self.device = device

    def aggregate(self, local_weights, agg_method='mean'):
        w_avg = copy.deepcopy(local_weights[0])
        
        for k in w_avg.keys():
            layer_updates = torch.stack([w[k].float() for w in local_weights], dim=0)
            
            if agg_method == 'trimmed_mean':
                beta = 0.2
                n = len(local_weights)
                m = int(n * beta)
                sorted_updates, _ = torch.sort(layer_updates, dim=0)
                w_avg[k] = torch.mean(sorted_updates[m:n-m], dim=0).type(w_avg[k].dtype)
            
            elif agg_method == 'median':
                # === 新增：Coordinate-wise Median 防御 ===
                # 取每一维度的中位数，对极端值非常鲁棒
                w_avg[k] = torch.median(layer_updates, dim=0).values.type(w_avg[k].dtype)
            
            else: 
                w_avg[k] = torch.mean(layer_updates, dim=0).type(w_avg[k].dtype)
                
        return w_avg