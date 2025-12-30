#!/bin/bash

echo "====== STARTING FULL PAPER EXPERIMENTS ======"

# --- 1. 敏感性分析 (Sensitivity): 不同的恶意比例 ---
# 此时固定使用 Trimmed Mean
# 我们已经跑过 0.2 了，现在补齐 0.1, 0.3, 0.4
for rate in 0.1 0.3 0.4
do
    echo ">>> Running Sensitivity: Malicious Rate $rate (Trimmed Mean)"
    # Baseline
    python main.py --method fedavg --agg trimmed_mean --mal_rate $rate --epochs 50 --gpu 0
    # Ours (RA-Fed)
    python main.py --method ra-fed --agg trimmed_mean --mal_rate $rate --epochs 50 --lambda_reg 1.0 --gpu 0
done

# --- 2. 通用性分析 (Generalization): 不同的防御算法 ---
# 固定恶意比例 0.2，尝试 Median 聚合
echo ">>> Running Generalization: Median Aggregation (Rate 0.2)"

# Baseline (Median)
python main.py --method fedavg --agg median --mal_rate 0.2 --epochs 50 --gpu 0
# Ours (RA-Fed + Median)
python main.py --method ra-fed --agg median --mal_rate 0.2 --epochs 50 --lambda_reg 1.0 --gpu 0

echo "====== ALL EXPERIMENTS FINISHED ======"