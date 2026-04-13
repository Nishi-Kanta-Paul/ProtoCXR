#!/bin/bash
set -e

echo "ProtoCXR Full Pipeline"
echo "======================"

pip install -r requirements.txt

python src/main.py \
  --dataset both \
  --ablation \
  --figures \
  --tables

python baselines/train_densenet121.py
python baselines/train_protopnet.py
python baselines/train_cbm.py
python baselines/compare_results.py

echo "Pipeline complete. Check outputs/ directory."
