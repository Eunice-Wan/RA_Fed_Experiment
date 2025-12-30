#!/bin/bash

# 确保日志目录存在
mkdir -p logs

echo "====== STARTING GAP ANALYSIS EXPERIMENTS ======"

# 1. Group A: Control Group (0% Malicious, Mean Aggregation)
# 目的：测定模型在和平环境下的“天生”表现
echo ">>> Running Group A: No Attack (Standard FedAvg)..."
python main.py --method fedavg --agg mean --mal_rate 0.0 --epochs 50 --gpu 0 > logs/gap_A_control.log 2>&1

# 2. Group B: Broken Group (20% Malicious, Mean Aggregation)
# 目的：展示没有防御时，模型会被破坏成什么样
echo ">>> Running Group B: Attack + No Defense..."
python main.py --method fedavg --agg mean --mal_rate 0.2 --epochs 50 --gpu 0 > logs/gap_B_broken.log 2>&1

# 3. Group C: Defended Group (20% Malicious, Trimmed Mean Defense)
# 目的：关键对照组！Trimmed Mean 能救回 Clean Acc，但能救回 Robust Acc 吗？
echo ">>> Running Group C: Attack + Trimmed Mean Defense..."
python main.py --method fedavg --agg trimmed_mean --mal_rate 0.2 --epochs 50 --gpu 0 > logs/gap_C_defended.log 2>&1

echo "====== GAP ANALYSIS FINISHED ======"