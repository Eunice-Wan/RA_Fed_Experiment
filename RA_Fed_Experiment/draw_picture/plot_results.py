import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# 设置 IEEE 风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 5),
    'lines.linewidth': 2.5
})

def plot_comparison():
    # 自动寻找 results 目录下的 csv 文件
    # 注意：你需要确认你的文件名，这里根据之前的命名习惯进行匹配
    # 也可以手动指定文件名
    
    # 假设 Baseline 文件名包含 'fedavg' 和 'trimmed'
    # 假设 RA-Fed 文件名包含 'ra-fed'
    
    files = glob.glob("results/*.csv")
    baseline_file = None
    rafed_file = None
    
    print("Found files:", files)
    
    for f in files:
        if "ra-fed" in f:
            rafed_file = f
        elif "fedavg" in f and "trimmed" in f:
            baseline_file = f
            
    data_map = {}
    if baseline_file:
        data_map['Baseline (Trimmed Mean)'] = pd.read_csv(baseline_file, header=None, names=['Round', 'Clean', 'Robust'])
    if rafed_file:
        data_map['RA-Fed (Ours)'] = pd.read_csv(rafed_file, header=None, names=['Round', 'Clean', 'Robust'])
        
    if not data_map:
        print("No valid CSV files found! Check your filenames.")
        return

    # === 图 1: Robust Accuracy 对比 (核心图) ===
    plt.figure()
    for name, df in data_map.items():
        plt.plot(df['Round'], df['Robust'], label=name, marker='o', markersize=4)
    
    plt.title('Robustness Comparison under PGD Attack')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Robust Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure_robustness.png', dpi=300)
    print("Saved figure_robustness.png")

    # === 图 2: Clean Accuracy 对比 (Trade-off 图) ===
    plt.figure()
    for name, df in data_map.items():
        plt.plot(df['Round'], df['Clean'], label=name, linestyle='--')
        
    plt.title('Clean Accuracy Comparison (Utility)')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure_clean_acc.png', dpi=300)
    print("Saved figure_clean_acc.png")

if __name__ == "__main__":
    plot_comparison()