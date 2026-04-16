# ProtoCXR

Prototype-Based Interpretable Multi-Label Diagnosis from Chest X-Rays Using Visually Grounded Anatomical Concepts.

## Overview

ProtoCXR trains a DenseNet-121 feature extractor with class-wise prototypes for interpretable multi-label CXR diagnosis. The current implementation targets VinDr-CXR only.

Supported labels:

1. Aortic enlargement
2. Cardiomegaly
3. Pleural effusion
4. Pleural thickening
5. Pulmonary fibrosis
6. No finding

## Dataset

Dataset: VinDr-CXR (18,000 images)

Expected layout:

```text
data/
└── vindr-cxr/
    ├── train.csv
    ├── test.csv
    ├── train/
    └── test/
```

The loader supports both DICOM and PNG image files.

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/run_full_pipeline.sh
```

## Main Pipeline

```bash
python src/main.py --ablation --figures --tables
```

Optional inference:

```bash
python src/main.py --skip_train --inference_dir /path/to/images
```

## Baselines

```bash
python baselines/train_densenet121.py
python baselines/train_protopnet.py
python baselines/train_cbm.py
python baselines/compare_results.py
```

## Outputs

Training artifacts:

```text
experiments/protocxr/
├── checkpoints/best_model_seed{N}.pt
├── logs/train_log_seed{N}.jsonl
└── results.json
```

Publication artifacts:

```text
outputs/
├── figures/
│   ├── fig2_perfinding_auc.png
│   ├── fig3_auc_comparison.png
│   ├── fig4_ablation.png
│   ├── fig5_user_study.png
│   └── fig6_loss_curves.png
└── tables/
    ├── table1_auc.csv / table1_auc.txt
    ├── table2_ablation.csv / table2_ablation.txt
    └── table3_perfinding.csv / table3_perfinding.txt
```