import matplotlib.pyplot as plt
import pandas as pd
import os

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
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def read_data(filename):
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        try:
            # CSV 格式: round, clean_acc, robust_acc
            df = pd.read_csv(path, header=None)
            # 假设第0列是轮数，第1列是Clean，第2列是Robust
            # 如果CSV没有轮数，我们需要手动生成 range
            if df.shape[1] == 3:
                rounds = df.iloc[:, 0].values
                clean = df.iloc[:, 1].values
                robust = df.iloc[:, 2].values
            else:
                # 兼容旧格式，假设只有两列数据，每行代表一轮
                clean = df.iloc[:, 0].values
                robust = df.iloc[:, 1].values
                rounds = range(1, len(clean) + 1)
            return rounds, clean, robust
        except:
            return [], [], []
    return [], [], []

# === 1. 读取三组数据 ===
# Group A: Ideal (0% Attack)
r_a, clean_a, robust_a = read_data("res_fedavg_mean_mal0.0.csv")

# Group B: Broken (20% Attack, No Defense)
r_b, clean_b, robust_b = read_data("res_fedavg_mean_mal0.2.csv")

# Group C: Defended (20% Attack, Trimmed Mean)
# 尝试读取，兼容文件名
r_c, clean_c, robust_c = read_data("res_fedavg_trimmed_mean_mal0.2.csv")
if len(r_c) == 0:
    r_c, clean_c, robust_c = read_data("res_baseline_final.csv")

# === 2. 画图 (双子图：左边 Clean，右边 Robust) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- 左图：Clean Accuracy (Utility) 的演变 ---
if len(r_a) > 0: ax1.plot(r_a, clean_a, label='Group A (No Attack)', color='#1f77b4', linestyle='-') # 蓝
if len(r_b) > 0: ax1.plot(r_b, clean_b, label='Group B (Undefended)', color='gray', linestyle=':') # 灰虚线
if len(r_c) > 0: ax1.plot(r_c, clean_c, label='Group C (Trimmed Mean)', color='#d62728', linestyle='--') # 红虚线

ax1.set_title('(a) Evolution of Clean Accuracy (Utility)')
ax1.set_xlabel('Communication Rounds')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 80)
ax1.legend()

# --- 右图：Robust Accuracy (Security) 的演变 ---
if len(r_a) > 0: ax2.plot(r_a, robust_a, label='Group A (No Attack)', color='#1f77b4', linestyle='-')
if len(r_b) > 0: ax2.plot(r_b, robust_b, label='Group B (Undefended)', color='gray', linestyle=':')
if len(r_c) > 0: ax2.plot(r_c, robust_c, label='Group C (Trimmed Mean)', color='#d62728', linestyle='--')

ax2.set_title('(b) Evolution of Robust Accuracy (Security)')
ax2.set_xlabel('Communication Rounds')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 80)
ax2.legend()

plt.tight_layout()
save_path = os.path.join(save_dir, 'Paper_Fig_Gap_Process.png')
plt.savefig(save_path, dpi=300)
print(f"✅ Saved figure to: {save_path}")