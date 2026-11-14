# UV-Net face segmentation on MFCAD++ (reproduction)

This reproduces UV-Net per-face machining-feature segmentation on MFCAD++ with the official
split, as the from-scratch baseline for later comparisons.

**Result: 99.00% per-face test accuracy, 0.968 mean IoU. That is 0.49 pt below the published 99.49%, inside
the ~1 pt tolerance we set for a reproduction.**

## Result table

| Model | Implementation | Hardware | Test per-face acc. | Test mIoU | Source |
|---|---|---|---|---|---|
| **UV-Net (ours, `best.pt`, epoch 51)** | PyTorch Geometric port | Apple M4, 8-core GPU (MPS) | **99.00 %** | **0.968** | this run |
| UV-Net (ours, `last.pt`, epoch 100) | PyTorch Geometric port | Apple M4, 8-core GPU (MPS) | 98.96 % | 0.966 | this run |
| UV-Net (published target) | official DGL code | not reported | 99.49 % | not reported | BrepMFR (Zhang et al. 2024), via the MFCAD++ repo benchmark table |
| Hierarchical CADNet (Edge) | – | – | 97.37 % | – | Colligan et al. 2022 |
| AAGNet | – | – | 99.57 % | – | BrepMFR (Zhang et al. 2024) |
| BrepMFR | – | – | 99.76 % | – | Zhang et al. 2024 |

**Gap to target: −0.49 pt.** The target was taken from the benchmark table in the official MFCAD++ repository
(`gitlab.com/qub_femg/machine-learning/mfcad2-dataset`), which cites BrepMFR for the UV-Net number. The
UV-Net paper itself predates MFCAD++. Accuracy is per face, pooled over all test faces. mIoU is the mean of
the 25 per-class IoUs over the whole test set.

## Per-class test results (`best.pt`)

Support is the number of test faces with that label (268,956 in total).

| id | Class | Support | IoU |
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

The weakest classes are rectangular through slot (0.887), round (0.932), blind hole (0.943), rectangular
blind slot (0.944) and rectangular through step (0.946). These are pairs that look alike locally: through vs
blind slots and steps, and a round vs a chamfer or stock face. All other classes are ≥ 0.95.

## Training curve (validation, 8,944 models)

| Epoch | Train loss | Val loss | Val acc. | Val mIoU |
|---|---|---|---|---|
| 1 | 1.108 | 0.340 | 89.39 % | 0.687 |
| 10 | 0.088 | 0.070 | 97.88 % | 0.930 |
| 25 | 0.042 | 0.044 | 98.71 % | 0.958 |
| **51 (selected)** | 0.024 | **0.036** | 99.01 % | 0.968 |
| 75 | 0.015 | 0.041 | 98.95 % | 0.967 |
| 100 | 0.011 | 0.048 | 98.90 % | 0.964 |

Validation accuracy plateaus at about 99.0% from around epoch 40, while training loss keeps falling, which is
mild overfitting. Selecting by lowest validation loss (the original UV-Net rule) picks epoch 51.

## Configuration

| | |
|---|---|
| Code | this repository (PyG port of github.com/AutodeskAILab/UV-Net; original checkout untouched) |
| Architecture | UV-Net segmenter, unchanged: surface 2D-CNN and curve 1D-CNN → 64-d; 3-layer edge-conditioned GNN (NNConv, sum) → 128-d graph embedding; 3-layer MLP head. **1,368,925 parameters** |
| Input | Face UV-grids 10×10×7 (xyz, normal, trim mask), edge U-grids 10×6 (xyz, tangent), centered and scaled to [−1, 1]³ |
| Optimiser | Adam, lr 1e-3, no schedule, no weight decay |
| Batch / epochs | 64 models, 100 epochs, `drop_last` on train |
| Loss | Cross-entropy per face, unweighted |
| Augmentation | None (`random_rotate` off) |
| Model selection | Lowest validation loss |
| Seed | 0 (`torch.manual_seed`). Single run; MPS kernels are not bit-deterministic |
| Command | `python train.py train --data_dir data/processed --epochs 100 --batch_size 64 --experiment_name mfcadpp_full` |
| Test command | `python train.py test --data_dir data/processed --checkpoint results/mfcadpp_full/0930_000433/best.pt` |

## Data

| | Official | Processed | Faces |
|---|---|---|---|
| Train | 41,766 | 41,753 | 1,253,866 |
| Val | 8,950 | 8,944 | 269,213 |
| Test | 8,949 | 8,948 | 268,956 |
| **Total** | **59,665** | **59,645** | |

- **Source:** `MFCAD_dataset.zip` from the QUB Pure page (CC BY). STEP files are under `step/{train,val,test}`, with `train.txt`/`val.txt`/`test.txt` as the official split.
- **Labels:** read from the `ADVANCED_FACE` entity names in each STEP file, as integers 0–24 (see `feature_labels.txt`).
- **File loss: 20 of 59,665 (0.03%)**, well under the 5% we allowed for. 16 are non-manifold (an edge with more than two faces), 3 fail occwl's edge–face check, and 1 has no solid or shell. The list is in `data/processed/failures.json`.
- **Recovered files:** 141 files store a closed shell without a solid wrapper. These were wrapped with `BRepBuilderAPI_MakeSolid` and kept.
- **Preprocessing:** `process/step_to_pyg.py`, 10 processes, about 20 min on the M4. The output is 3.8 GB of per-model graphs plus 3.8 GB of packed per-split caches.

## Compute

| | |
|---|---|
| Machine | MacBook Air, Apple M4 (10-core CPU, 8-core GPU), 16 GB unified memory, macOS 27.0 |
| Training wall-clock | **14.6 h** for 100 epochs; 6.6 h to the selected epoch 51 |
| Epoch time | 425–942 s (median about 460 s). Epochs slowed to 600–900 s whenever the Mac was also in use |
| GPU-hours | about 14.6 M4-GPU-hours. a datacenter GPU should be several times faster |
| Peak process memory | about 7–12 GB footprint, reported by `footprint`; the upper figure likely includes the memory-mapped data cache |

## Environment

- **Packages:** conda env `uvnet_pyg` (see `environment.yml`): Python 3.11, PyTorch 2.10.0 (conda-forge, MPS), torch_geometric 2.8.0.post1, pythonocc-core 7.7.2, occwl 3.0.0 (from GitHub, `--no-deps`).
- **No compiled extensions:** DGL, torch_scatter and torch_sparse are not needed.

## Deviations from the official code, and workarounds

1. **PyTorch Geometric instead of DGL** . The GPU on this machine is Apple MPS, which DGL does not support. The mapping is `NNConv(sum)` → PyG `NNConv(aggr="add", root_weight=False, bias=False)`, `MaxPooling` → `global_max_pool`, and `repeat_interleave` → `graph_emb[batch]`. The message-passing equations are identical.
2. **MPS shape bucketing.** MPS recompiles kernels for every new tensor shape, at about 0.5 s each, which made training about 5× slower. Each batch is therefore padded to multiples of 128 faces and 512 edges with a disconnected dummy graph whose outputs are discarded. Eval outputs are bit-identical to unpadded runs (checked, max abs diff 0.0). In training, BatchNorm statistics also see the few percent of duplicated padding rows.
3. **PyTorch from conda-forge, not pip.** The pip wheel together with conda-forge OpenCascade loads two OpenMP runtimes and aborts with `OMP: Error #15`.
4. **Memory.** The packed data cache is memory-mapped, and cached MPS buffers are released every 200 steps. Without this the process reached 11 GB and pushed a 16 GB machine into swap on a nearly full disk. The first launch was stopped after one epoch for this reason and restarted; this report covers only the restarted run.
5. **Data loss.** 20 unparseable models are excluded, including 1 test model. See Data above.
6. **Logging.** Runs log to TensorBoard, not W&B.
7. **Float precision.** Training is fp32 throughout. fp16 and bf16 autocast were benchmarked and gave only 4–10% speedup, so they were not used.

## Possible reasons for the 0.49 pt gap (not tested)

- **Unknown target protocol.** BrepMFR's UV-Net settings (epochs, schedule, augmentation, split handling) have not been checked yet. That should be the next step before tuning.
- **No LR schedule and no augmentation.** A cosine schedule and `--random_rotate` are the cheapest things to try.
- **Single seed.** Variance across seeds is unknown.

## Artifacts

The run wrote these to `results/mfcadpp_full/0930_000433/` (not committed):
- `best.pt`: epoch 51, the reference "from-scratch" checkpoint, 16 MB
- `last.pt`: epoch 100
- `test_metrics_best.json` and `test_metrics_last.json`: accuracy, mIoU, per-class IoU
- `args.json` and TensorBoard event file

The full log was written to `logs/train_full.log` (not committed).

Trained checkpoints are not included in this repository.
