import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 路径设置 (适配新文件夹结构)
# ==========================================
# 获取当前脚本目录 (draw_picture)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取根目录 (RA_Fed_Experiment)
parent_dir = os.path.dirname(current_dir)
# 图片保存目录
save_dir = os.path.join(parent_dir, 'picture_outputs')

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 2. IEEE 风格设置
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'], # 服务器通用字体
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (8, 6),
    'lines.linewidth': 3,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# ==========================================
# 3. 数据录入 (基于之前的成功实验日志)
# ==========================================
rounds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Baseline (Trimmed Mean) 数据
base_clean = [26.83, 32.12, 36.02, 39.15, 43.02, 46.16, 49.34, 49.69, 51.53, 52.15]
base_robust = [19.26, 19.55, 19.52, 19.62, 20.14, 20.66, 20.23, 21.37, 20.71, 22.48]

# RA-Fed (Ours) 数据
our_clean = [21.30, 32.80, 36.91, 39.73, 41.92, 44.39, 47.04, 48.46, 49.90, 50.91]
our_robust = [19.40, 26.76, 30.69, 32.43, 34.21, 35.35, 36.74, 37.62, 38.17, 39.04]

# ==========================================
# 4. 画图函数
# ==========================================
def plot_graph(y_base, y_our, ylabel, title, filename, loc='lower right'):
    plt.figure()
    
    # 画 Baseline (灰色/虚线)
    plt.plot(rounds, y_base, label='Baseline (Trimmed Mean)', 
             color='#7f7f7f', linestyle='--', marker='s', markersize=8)
    
    # 画 Ours (红色/实线)
    plt.plot(rounds, y_our, label='RA-Fed (Ours)', 
             color='#d62728', linestyle='-', marker='o', markersize=10)
    
    plt.xlabel('Communication Rounds')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout()
    
    # 保存路径修改为 picture_outputs
    full_save_path = os.path.join(save_dir, filename)
    plt.savefig(full_save_path, dpi=300)
    print(f"✅ Saved figure to: {full_save_path}")

# ==========================================
# 5. 执行画图
# ==========================================
if __name__ == "__main__":
    # 图1: Robust Accuracy (核心卖点)
    plot_graph(base_robust, our_robust, 
               ylabel='Robust Accuracy (%)', 
               title='Robustness against PGD Attack', 
               filename='Paper_Fig1_Robustness.png',
               loc='lower right')

    # 图2: Clean Accuracy (Trade-off 展示)
    plot_graph(base_clean, our_clean, 
               ylabel='Test Accuracy (%)', 
               title='Clean Accuracy (Utility)', 
               filename='Paper_Fig2_Utility.png',
               loc='lower right')