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
plt.rcParams.update({
    'font.family': 'serif', 
    'font.serif': ['DejaVu Serif'], 
    'font.size': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def get_last_acc(filename):
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, header=None)
            # 返回 (Clean Acc, Robust Acc)
            return df.iloc[-1, 1], df.iloc[-1, 2]
        except:
            pass
    return 0.0, 0.0

# === 读取数据 ===
# Group A: 0% 攻击 (Ideal)
clean_A, robust_A = get_last_acc("res_fedavg_mean_mal0.0.csv")

# Group B: 20% 攻击 + 无防御 (Broken)
clean_B, robust_B = get_last_acc("res_fedavg_mean_mal0.2.csv")

# Group C: 20% 攻击 + Trimmed Mean (Defended)
# 这是我们用来做 Baseline 对比的关键数据
clean_C, robust_C = get_last_acc("res_fedavg_trimmed_mean_mal0.2.csv")
# 兼容性：如果之前跑过并被重命名了，尝试读取 baseline_final
if clean_C == 0.0:
    clean_C, robust_C = get_last_acc("res_baseline_final.csv")

# === 画图 ===
labels = ['Group A\n(No Attack)', 'Group B\n(Undefended)', 'Group C\n(Trimmed Mean)']
clean_scores = [clean_A, clean_B, clean_C]
robust_scores = [robust_A, robust_B, robust_C]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
# 蓝色柱子：Clean Accuracy
rects1 = ax.bar(x - width/2, clean_scores, width, label='Clean Accuracy (Utility)', color='#1f77b4', alpha=0.85, edgecolor='black')
# 红色柱子：Robust Accuracy
rects2 = ax.bar(x + width/2, robust_scores, width, label='Robust Accuracy (Security)', color='#d62728', alpha=0.85, edgecolor='black')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Decoupling: Byzantine Defense vs. Adversarial Robustness')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 100) # 固定 Y 轴范围方便对比

# 自动标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
save_path = os.path.join(save_dir, 'Paper_Fig_Gap_Analysis.png')
plt.savefig(save_path, dpi=300)
print(f"✅ Saved figure to: {save_path}")