# ProtoCXR

> **ProtoCXR**: Prototype-Based Interpretable Multi-Label Diagnosis from Chest X-Rays Using Visually Grounded Anatomical Concepts

## Overview

ProtoCXR is a novel deep learning framework for interpretable multi-label classification of chest X-rays. Standard convolutional networks and vision transformers provide highly accurate diagnoses but lack interpretable rationales. Our approach bridges this gap by introducing an interpretable bottleneck layer based on *prototypes* — actual image patches from the training set. 

For every prediction, ProtoCXR explains its reasoning by dissecting the image and pointing out spatial locations that are visually similar to learned reference patches associated with specific lung pathologies.

## Architecture

1. **Backbone & Projection**: A truncated DenseNet-121 core extracts spatial feature maps, projected to a prototype embedding space.
2. **Prototype Similarity Layer**: Replaces the global average pooling and dense classifier of standard networks. Compares the spatial features against $C \times K$ learned prototypes.
3. **Anatomical Region Alignment (ARA)**: A lightweight frozen U-Net guides prototypes to focus strictly on lung regions, reducing spurious correlations.
4. **Prototype Push**: During training, algorithmically replaces latent prototypes with the nearest latent representations of actual training image patches, creating highly interpretable diagnostic exemplars.

## Dataset Setup

This codebase natively supports CheXpert v1.0-small and the NIH ChestX-ray14 datasets.

1. **CheXpert**: Download `CheXpert-v1.0-small.zip` from Stanford ML Group and extract it to `data/CheXpert-v1.0-small/`.
2. **NIH ChestX-ray14**: Download the `images/` directory and `Data_Entry_2017.csv` and place them in `data/NIH_ChestXray14/`.

Your `data/` folder should look like:
```text
data/
├── CheXpert-v1.0-small/
│   ├── train.csv
│   └── train/
└── NIH_ChestXray14/
    ├── Data_Entry_2017.csv
    └── images/
```

## Quick Start

You can run the entire training, ablation, and evaluation pipeline using our reproducible bash script.

```bash
pip install -r requirements.txt
bash scripts/run_full_pipeline.sh
```

## File Structure

```text
ProtoCXR/
├── data/                          ← Datasets go here
├── src/
│   ├── config.py                  ← Central configuration and hyperparameters
│   ├── dataset.py                 ← Data loading and stratified subset logic
│   ├── model.py                   ← Main ProtoCXR architecture
│   ├── losses.py                  ← ARA, PDR, and Separation constraints
│   ├── train.py                   ← 4-phase prototype training loop
│   ├── explainability.py          ← Visualization and patch matching logic
│   ├── inference.py               ← Single-image and directory inference
│   ├── evaluate.py                ← Table & metrics generation
│   ├── figures.py                 ← Generates all paper figures
│   └── main.py                    ← CLI entry point
├── baselines/
│   ├── train_densenet121.py       ← Black-box standard baseline
│   ├── train_protopnet.py         ← ProtoPNet without ARA/PDR
│   ├── train_cbm.py               ← Concept Bottleneck baseline
│   └── compare_results.py         ← Compiles figures & tables from logs
├── scripts/
│   └── run_full_pipeline.sh       ← Execution script
├── experiments/                   ← Checkpoints and training logs (auto-generated)
└── outputs/                       ← Final figures and tables (auto-generated)
```

## Results Summary

| Method | CheXpert (AUC) | NIH-CXR14 (AUC) | Interpretable |
| :--- | :---: | :---: | :---: |
| DenseNet-121 | 0.903 | 0.892 | No |
| CBM | 0.851 | 0.836 | Yes |
| ProtoPNet | 0.864 | 0.849 | Yes |
| **ProtoCXR (ours)** | **0.891** | **0.879** | **Yes** |

*(Note: Above values are examples. Real values will populate in `outputs/tables/table1_auc.csv` after running the pipeline)*

## Citation

```bibtex
@article{protocxr2026,
  title={ProtoCXR: Prototype-Based Interpretable Multi-Label Diagnosis from Chest X-Rays Using Visually Grounded Anatomical Concepts},
  author={Anonymous},
  year={2026}
}
```
