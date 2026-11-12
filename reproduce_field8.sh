#!/bin/bash
# Project Card alan 8: baseline satırları (majority, TF-IDF+LR, DIET Config A x3 seed) KAYITLI TAHMİNLERDEN
# yeniden üretilir (results/preds/*.csv). Eğitimi yeniden koşturmaz; eğitim için run_mid.sh ve baselines.py.
cd "$(dirname "$0")"
PYTHONPATH=. .venv/bin/python summarize.py > results/mid_summary_repro.md && cat results/mid_summary_repro.md
