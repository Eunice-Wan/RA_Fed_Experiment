import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================================
# 1. 路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_dir = os.path.join(parent_dir, 'results')
save_dir = os.path.join(parent_dir, 'picture_outputs')

# 确保图片保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 2. IEEE 风格设置
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 14,
    'lines.linewidth': 3,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def get_final_accuracy(filename):
    """读取 CSV 文件的最后一行，获取 Robust Accuracy"""
    filepath = os.path.join(results_dir, filename)
    
    # 如果 results 目录里找不到，试试看是不是在当前脚本目录（兼容性处理）
    if not os.path.exists(filepath):
        if os.path.exists(filename):
            filepath = filename
        else:
            return None
            
    try:
        # CSV 格式: round, clean_acc, robust_acc
        df = pd.read_csv(filepath, header=None)
        return df.iloc[-1, 2] # 返回最后一行的第3列 (Robust Acc)
    except Exception as e:
        # print(f"Error reading {filename}: {e}") # 调试时可打开
        return None

# ==========================================
# 3. 数据读取逻辑 (核心修改点)
# ==========================================
rates = [0.1, 0.2, 0.3, 0.4]
baseline_accs = []
rafed_accs = []

print("Extracting sensitivity data...")

for r in rates:
    # --- 构造文件名 ---
    # 1. Baseline 文件名
    base_name = f"res_fedavg_trimmed_mean_mal{r}.csv"
    acc_b = get_final_accuracy(base_name)
    
    # 特殊处理：0.2 的 baseline 可能被你重命名过
    if acc_b is None and r == 0.2:
        acc_b = get_final_accuracy("res_baseline_final.csv")
    
    if acc_b is None:
        print(f"⚠️ Warning: Missing Baseline file for rate {r}. Using 0.0.")
        acc_b = 0.0
    baseline_accs.append(acc_b)

    # 2. RA-Fed 文件名
    rafed_name = f"res_ra-fed_trimmed_mean_mal{r}.csv"
    acc_r = get_final_accuracy(rafed_name)
    
    if acc_r is None:
        print(f"⚠️ Warning: Missing RA-Fed file for rate {r}. Using 0.0.")
        acc_r = 0.0
    rafed_accs.append(acc_r)

print(f"Baseline Accs: {baseline_accs}")
print(f"RA-Fed Accs:   {rafed_accs}")

# ==========================================
# 4. 画图
# ==========================================
if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    
    # Baseline
    plt.plot(rates, baseline_accs, label='Baseline (Trimmed Mean)', 
             color='#7f7f7f', linestyle='--', marker='s', markersize=8)
    
    # RA-Fed
    plt.plot(rates, rafed_accs, label='RA-Fed (Ours)', 
             color='#d62728', linestyle='-', marker='o', markersize=10)

    plt.xlabel('Malicious Client Ratio (%)')
    plt.ylabel('Robust Accuracy (%)')
    plt.title('Impact of Malicious Client Ratio')
    
    # 设置 X 轴刻度显示为百分比 (10%, 20%...)
    plt.xticks(rates, [f"{int(r*100)}%" for r in rates])
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图片为 Fig3
    full_save_path = os.path.join(save_dir, 'Paper_Fig3_Sensitivity.png')
    plt.savefig(full_save_path, dpi=300)
    print(f"✅ Saved figure to: {full_save_path}")