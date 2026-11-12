#!/bin/bash
# Config B, seed 42, YALNIZCA dev (mid-project yöntem durumu). Test setinde çalıştırılmaz.
cd "$(dirname "$0")"
source cuda_env.sh
export TF_GPU_MEMORY_ALLOC="0:3400"
PYTHONPATH=. .venv/bin/python train_diet.py massive b 42 --dev-only
