# Results: UV-Net (PyG) face segmentation on MFCAD++

This page reports a single training run of this repository on the full MFCAD++ dataset with the official
train/validation/test split.

## Headline

| Model | Implementation | Hardware | Test per-face acc. | Test mIoU |
|---|---|---|---|---|
| **This repo, best validation checkpoint (epoch 51)** | PyTorch Geometric | Apple M4, 8-core GPU (MPS) | **99.00%** | **0.968** |
| This repo, final checkpoint (epoch 100) | PyTorch Geometric | Apple M4, 8-core GPU (MPS) | 98.96% | 0.966 |
| UV-Net, published | official DGL code | not reported | 99.49% | – |
| Hierarchical CADNet (Edge) | – | – | 97.37% | – |
| AAGNet | – | – | 99.57% | – |
| BrepMFR | – | – | 99.76% | – |

**Sources for the reference numbers:** the benchmark table in the official MFCAD++ repository
(gitlab.com/qub_femg/machine-learning/mfcad2-dataset). It cites Colligan et al. 2022 for Hierarchical CADNet
and BrepMFR (Zhang et al. 2024) for UV-Net, AAGNet and BrepMFR. The UV-Net paper itself predates MFCAD++.

**How the metrics are computed:**
- **Accuracy:** per face, pooled over all 268,956 test faces.
- **mIoU:** the mean of the 25 per-class IoUs over the whole test set.

## Per-class test results (epoch-51 checkpoint)

| id | Class | Test faces | IoU |
|---|---|---|---|
| 0 | Chamfer | 3,703 | 0.957 |
| 1 | Through hole | 3,945 | 0.974 |
| 2 | Triangular passage | 11,408 | 0.983 |
| 3 | Rectangular passage | 15,296 | 0.967 |
| 4 | 6-sided passage | 22,763 | 0.994 |
| 5 | Triangular through slot | 2,588 | 0.949 |
| 6 | Rectangular through slot | 4,097 | **0.887** |
| 7 | Circular through slot | 1,391 | 0.978 |
| 8 | Rectangular through step | 7,017 | 0.946 |
| 9 | 2-sided through step | 10,214 | 0.974 |
| 10 | Slanted through step | 7,031 | 0.947 |
| 11 | O-ring | 10,328 | 0.967 |
| 12 | Blind hole | 6,710 | 0.943 |
| 13 | Triangular pocket | 13,426 | 0.992 |
| 14 | Rectangular pocket | 16,971 | 0.982 |
| 15 | 6-sided pocket | 22,101 | 0.996 |
| 16 | Circular end pocket | 15,954 | 0.987 |
| 17 | Rectangular blind slot | 3,603 | 0.944 |
| 18 | Vertical circular end blind slot | 2,761 | 0.979 |
| 19 | Horizontal circular end blind slot | 2,287 | 0.989 |
| 20 | Triangular blind step | 6,951 | 0.971 |
| 21 | Circular blind step | 6,546 | 0.988 |
| 22 | Rectangular blind step | 10,516 | 0.977 |
| 23 | Round | 1,209 | **0.932** |
| 24 | Stock | 60,140 | 0.996 |

The hardest classes are features that look alike locally: through vs blind slots and steps, and a round vs
a chamfer or stock face.

## Training curve (validation set, 8,944 models)

| Epoch | Train loss | Val loss | Val acc. | Val mIoU |
|---|---|---|---|---|
| 1 | 1.108 | 0.340 | 89.39% | 0.687 |
| 10 | 0.088 | 0.070 | 97.88% | 0.930 |
| 25 | 0.042 | 0.044 | 98.71% | 0.958 |
| **51 (selected)** | 0.024 | **0.036** | 99.01% | 0.968 |
| 75 | 0.015 | 0.041 | 98.95% | 0.967 |
| 100 | 0.011 | 0.048 | 98.90% | 0.964 |

Validation accuracy levels off at about 99.0% from around epoch 40, while training loss keeps falling, which
is mild overfitting. The checkpoint with the lowest validation loss (the original UV-Net selection rule) is
from epoch 51.

## Setup

| | |
|---|---|
| Architecture | UV-Net segmenter, unchanged from the original, 1,368,925 parameters |
| Input | Face grids 10×10×7, edge grids 10×6, part scaled to [−1, 1]³ |
| Optimiser | Adam, lr 1e-3, no schedule, no weight decay |
| Batch / epochs | 64 / 100 |
| Augmentation | none |
| Seed | 0 (single run; MPS kernels are not bit-deterministic) |
| Data | 41,753 train / 8,944 val / 8,948 test models. The official split has 41,766 / 8,950 / 8,949; 20 of the 59,665 files could not be processed |
| Hardware | MacBook Air, Apple M4 (10-core CPU, 8-core GPU), 16 GB memory |
| Time | Preprocessing about 20 min (10 processes); training 14.6 h for 100 epochs (6.6 h to epoch 51) |

**Command:**
```bash
python train.py train --data_dir data/processed --epochs 100 --batch_size 64 --experiment_name mfcadpp_full
```

## Possible reasons for the 0.49 pt gap (not tested)

- **Unknown reference protocol.** The settings behind the published number (epochs, LR schedule,
  augmentation) are not known here.
- **No LR schedule and no augmentation.** A cosine schedule and `--random_rotate` are the cheapest things to
  try.
- **Single seed.** The spread across seeds is unknown.
- **Small implementation differences.** 20 unprocessable models are excluded (1 of them from the test split),
  and batch-shape bucketing on MPS slightly perturbs BatchNorm statistics during training. Neither should be
  worth half a point.
