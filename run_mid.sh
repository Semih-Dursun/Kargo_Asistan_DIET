#!/usr/bin/env bash
# Mid-project: Config A, MASSIVE tr-TR, seed 42/43/44 sırayla, GPU üzerinde.
# Zaten eğitilmiş modeli atlar (train_diet.py). Kullanım: bash run_mid.sh
cd "$(dirname "$0")"
source cuda_env.sh                      # CUDA 11.8 / cuDNN 8.6 yolu (.cuda11)
export TF_GPU_MEMORY_ALLOC="0:3400"      # MB; TF varsayılanı yalnızca ~2,1 GB veriyor, DIET batch büyüdükçe OOM oluyordu
mkdir -p logs models results/preds
for seed in 42 43 44; do
  PYTHONPATH=. .venv/bin/python train_diet.py massive a "$seed" > "logs/massive_a_${seed}.log" 2>&1
done
PYTHONPATH=. .venv/bin/python summarize.py > results/mid_summary.md
