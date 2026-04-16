#!/bin/bash
set -e
echo "=============================="
echo "  ProtoCXR Full Pipeline"
echo "  Dataset: VinDr-CXR (18K)"
echo "=============================="
pip install -r requirements.txt --quiet
python src/main.py --ablation --figures --tables
python baselines/train_densenet121.py
python baselines/train_protopnet.py
python baselines/train_cbm.py
python baselines/compare_results.py
echo "Pipeline complete. Check outputs/ directory."
