#!/usr/bin/env bash
# DIET taraması (dev-only), GPU. Kullanım: bash run_tune.sh screen|confirm ADAY|final ADAY
cd "$(dirname "$0")"
source cuda_env.sh
export TF_GPU_MEMORY_ALLOC="0:3400"
mkdir -p logs
PYTHONPATH=. .venv/bin/python tune_diet.py "$@" > "logs/tune_$1${2:+_$2}.log" 2>&1
