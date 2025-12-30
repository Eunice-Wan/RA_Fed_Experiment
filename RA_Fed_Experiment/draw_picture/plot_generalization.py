import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_dir = os.path.join(parent_dir, 'results')
save_dir = os.path.join(parent_dir, 'picture_outputs')

# === IEEE 风格 ===
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 14})

def get_acc(filename):
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path, header=None).iloc[-1, 2] # Robust Acc
    return 0.0

# === 读取数据 (Rate=0.2) ===
# 1. Trimmed Mean (之前的数据)
tm_base = get_acc("res_fedavg_trimmed_mean_mal0.2.csv")
# 如果找不到 0.2 的文件，尝试读取 baseline_final
if tm_base == 0.0: tm_base = get_acc("res_baseline_final.csv")
tm_our = get_acc("res_ra-fed_trimmed_mean_mal0.2.csv")

# 2. Median (刚才跑的数据)
med_base = get_acc("res_fedavg_median_mal0.2.csv")
med_our = get_acc("res_ra-fed_median_mal0.2.csv")

# === 画柱状图 ===
labels = ['Trimmed Mean', 'Median']
baseline_means = [tm_base, med_base]
our_means = [tm_our, med_our]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, baseline_means, width, label='Baseline', color='gray')
rects2 = ax.bar(x + width/2, our_means, width, label='RA-Fed (Ours)', color='#d62728')

ax.set_ylabel('Robust Accuracy (%)')
ax.set_title('Generalization across Aggregation Rules (Malicious Rate=0.2)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 标数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Paper_Fig5_Generalization.png'), dpi=300)
print("Saved Paper_Fig5_Generalization.png")