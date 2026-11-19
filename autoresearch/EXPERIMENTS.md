# UV-Net autoresearch: experiment summary

Branch `autoresearch/nov14`, Nov 14 to 17 2026. Karpathy's autoresearch loop applied to UV-Net
(PyTorch Geometric) face segmentation on the Fusion 360 Gallery segmentation dataset, trained on
Kaggle 2x T4 (account `aumr1101`). 108 training runs in total.

## Setup

- **Frozen harness** (`prepare.py`, tag `harness`): data, time budget, evaluation. Only `train.py` changes.
- **Budget**: 600 s of training per run on one T4 (two runs share a kernel, one per GPU).
- **Metric** (from harness commit `ac85b7d`): `val_miou` = mean IoU over the 7 classes other than
  RevolveEnd. The 8-class mean is logged as `val_miou8`. RevolveEnd has only a handful of val faces, so its
  IoU swings 0.06 to 0.27 on identical code and made the 8-class score too noisy to compare ideas with.
- **Noise**: seed spread on the 7-class score is about 0.003 to 0.007 (0.01 at worst). Every idea gets 2
  seeds; close calls get 4. An idea is kept only when its seeds clear the current best's.
- **Reference**: the published UV-Net recipe (1.37M parameters, Adam 1e-3, batch 64), 100 epochs (about
  78 min on a T4): val 8-class mIoU 0.669, test 0.677 (from `program.md`).

## Headline result

| | Training | Val 7-class | Val 8-class | Test 7-class | Test 8-class | Test acc |
|---|---|---|---|---|---|---|
| UV-Net recipe, 10 min | 600 s, 9 epochs | 0.611 | 0.535 | | | |
| UV-Net recipe, full | ~78 min, 100 epochs | | 0.669 | | 0.677 | |
| **Best (`e09d1fb`), 10 min** | 600 s, 23 epochs | **0.784** | 0.722 | | | |
| **Best (`e09d1fb`), full** | 4500 s, ~180 epochs | 0.779 to 0.784 | 0.723 to 0.728 | **0.801** | **0.748** | 0.918 |

Test numbers are the mean of 2 seeds (0.802 / 0.799 and 0.749 / 0.746), from the separate `final-test`
branch (`69b9bd9`, its `train.py` is included here as `final_test_train.py`), which uses the harness's own
scoring code and was never merged into the loop branch.
Test was run exactly once.

Per-class test IoU (seed 0 / seed 1):

| ExtrudeSide | ExtrudeEnd | CutSide | CutEnd | Fillet | Chamfer | RevolveSide | RevolveEnd |
|---|---|---|---|---|---|---|---|
| .895 / .896 | .890 / .892 | .707 / .706 | .672 / .681 | .845 / .838 | .857 / .831 | .749 / .749 | .379 / .374 |

## What was kept (the winning chain)

Scores are 10-minute means on the 7-class metric (8-class in brackets). Early rows were scored on the
8-class metric at the time and rescored later from saved per-class IoU.

| Commit | Change | Seeds | 7-class | Gain | Epochs |
|---|---|---|---|---|---|
| `3f5c54a` | Baseline: the published UV-Net recipe | 2 | 0.611 (0.535) | | 9 |
| `350c9ff` | Convexity edge features (normal angle + centroid-offset signs) | 1 | 0.647 (0.566) | +0.036 | 10 |
| `d8e3305` / `36ea467` | Cosine LR decay, peak 2e-3, 200-step warmup | 1 | 0.638 | | 9 |
| `d2b21a9` | fp16 autocast + GradScaler: 64 s to 25 s per epoch | 1 | 0.694 (0.607) | +0.05 | 16 |
| `3297c08` | Runner warms the data cache (harness fix, not a model change) | 3 | 0.735 (0.660) | | 26 |
| `ed0dc7b` | Message-passing hidden size 128 instead of 64 | 3 | 0.744 (0.675) | +0.009 | 25 |
| `3ccc6b5` | Local convexity at the edge midpoint (nearest-grid-point normals) | 2 | 0.755 (0.672) | +0.011 | 23 |
| `b8b03ed` | NNConv `root_weight=True` (a node keeps its own state) | 2 | 0.759 (0.687) | +0.004 | 23 |
| `c68e4e2` | NNConv `aggr="mean"` instead of sum | 4 | 0.767 (0.699) | +0.008 | 24 |
| `1fdedee` | Face and edge UV-grids centred on their own centroid; centroid fed separately | 4 | 0.777 (0.718) | +0.010 | 23 |
| `e09d1fb` | Grids also divided by their RMS radius (floor 0.02, clip to +-5); log size fed separately | 4 | **0.784 (0.722)** | +0.007 | 23 |

The `3297c08` jump is mostly the cache fix. Before it, the first pair of runs in every batch lost about
200 s paging the 3.5 GB training cache, so batches 0 to 2 understate their real scores.

## What did not work

10-minute, 2-seed means unless noted, compared with the best at the time.

| Idea | Result | Note |
|---|---|---|
| Inverse-sqrt class weights | 0.558 vs 0.611 | Hurts big classes far more than it helps RevolveEnd |
| 2 data-loader workers | fewer epochs | 4 CPUs are shared by two runs |
| Label smoothing 0.1 | 0.733, single seed | Within noise, then worse with seeds |
| Batch 128, LR 3e-3 | 0.735, single seed | No gain |
| Peak LR 4e-3 / 3e-3 | 0.729 / 0.760 | No gain |
| EMA of weights (0.999) | below best | |
| AdamW, weight decay 0.05 | 0.756 vs 0.755 | Within noise, twice |
| Face size input (log area + bbox) | 0.752 vs 0.755 | |
| Surface/curve embeddings 128 | 0.752, later 0.781 vs 0.784 | Tried twice |
| Wider encoders | ~+0.005 | Unclear, discarded |
| Jumping knowledge | 0.758 vs 0.759 | Root weight already gives a skip path |
| 3 rounds of message passing | 0.740 to 0.741 | Worse in fp16 and fp32; one fp16 run collapsed (NaN) |
| Message passing in fp32 | 0.755 vs 0.759 | Stable but 19 epochs instead of 23 |
| Classifier dropout 0.5 | 0.757 vs 0.766 | |
| Batch size 32 | 0.751 | Fewer epochs |
| Log face degree | 0.768 vs 0.767 (4 seeds) | Inside the spread, extra code |
| Max + mean part pooling | 0.759 | |
| Centring faces only (not edges) | 0.770 vs 0.776 | Edges matter most |
| Scale normalisation without floor | 0.31 to 0.41 | Collapsed: see lessons |
| Scale normalisation faces only | 0.777 vs 0.784 | |
| Convexity along the whole edge | 0.783 vs 0.784 | |
| Masked mean pooling in the surface CNN | 0.780 | |
| `cudnn.benchmark` | 0.753 | Re-tunes per batch shape: 8 epochs instead of 23 |
| Random 90-degree rotation (GPU) | 0.767 | Hurts even at equal epochs |
| Uniform random rotation (GPU) | 0.722 | |
| Per-axis stretch [0.8, 1.2] | 0.780 (10 min), 0.781 vs 0.785 (full length) | |
| Gaussian point jitter, sigma 0.005 | 0.778 | |
| Per-axis mirror flips | 0.762 (10 min), 0.756 vs 0.785 (full length) | Hurts CutSide, CutEnd, RevolveSide |

## Lessons

1. **The canonical frame is a feature.** Every rotation or mirror augmentation hurt. Fusion 360 designers
   mostly sketch on a base plane, extrude up and cut down, so even the sign of a face's direction carries
   label information.
2. **Local shape matters more than position.** The last two gains both separate where a face is (centroid,
   size) from what shape it has (centred, scale-normalised grid). Edges benefit most.
3. **The model is limited by data, not compute.** At full length (4500 s, ~170 epochs), train loss falls to
   0.002 while val loss sits near 0.95, and the 7-class score is the same as at 10 minutes (0.7836 vs
   0.7840). Only RevolveEnd gains from long training (IoU 0.06 to 0.18 at 10 min, 0.29 to 0.39 at full length).
   This also means the 10-minute proxy was a faithful one for architecture ideas.
4. **fp16 hazard.** One run in about 60 had a NaN loss in epoch 1. GradScaler skipped the step, but
   BatchNorm running stats had already absorbed the NaN, so eval scored about 0.07 while train loss
   looked normal. Symptom to look for: a near-zero score with a healthy train loss.
5. **Dividing by a near-zero radius wrecks BatchNorm at eval.** Per-batch statistics hide it in training;
   the running statistics used at eval do not. A floor and clipping fixed it.
6. **The validation set was not overfitted by the search.** Test came out about 0.02 above val on both
   metrics, with the two seeds within 0.004 of each other.

## Cost: training vs inference

**Training cost** is the one-off GPU time to fit the weights: many passes over the 24,215 training parts,
each a forward pass, a backward pass and a weight update. It is paid once per model, or on every retrain.
**Inference cost** is the GPU time to use the trained model: one forward pass per part, no gradients. It is
paid every time a part is labelled, so it grows with usage.

### Training: about 8x cheaper for better accuracy

| Recipe | GPU time on one T4 | Epochs | Val 8-class | Val 7-class |
|---|---|---|---|---|
| Original UV-Net recipe | ~78 min | 100 | 0.669 | |
| Original recipe, 10 min | 10 min | 9 | 0.535 | 0.611 |
| New recipe (`e09d1fb`) | **10 min** | 23 | **0.722** | **0.784** |
| New recipe, full length | 75 min | ~180 | 0.723 to 0.728 | 0.779 to 0.784 |

- 10 minutes of the new recipe beats 78 minutes of the original, so training to UV-Net's accuracy costs at
  least about 8x less GPU time. The new model was never trained for less than 10 minutes, so 8x is a
  lower bound.
- Two reasons: each epoch is about 2.5x faster (64 s to 25 s, from fp16 training), and far fewer epochs
  are needed (23 beat the original's 100, from convexity features, the cosine schedule and the encoder changes).
- Most of the speed comes from the training side (fp16, schedule), which would speed up the original
  encoder too: the original-sized model with convexity + cosine + fp16 (`3297c08`) already reached
  0.656 to 0.669 in 10 minutes. The encoder changes add accuracy at the same cost (10-minute 7-class
  0.735 to 0.784, 8-class 0.660 to 0.722, with epoch time barely changed).
- Training past 10 minutes buys almost nothing (7-class 0.7840 at 10 min vs 0.7836 at 75 min). Only
  RevolveEnd improves, so a 10-minute run is the cost-efficient choice unless RevolveEnd matters.

### Inference: 13% dearer in fp32, roughly 2x cheaper in fp16 (estimate)

Evaluation time on the same validation split, one T4, fp32, data cache warm (from the Kaggle run logs):

| Model | Parameters | Eval time | Relative |
|---|---|---|---|
| Original size, hidden 64 (`3297c08`, `c7a8ff8`, 4 runs) | 1.36M | 4.8 s | 1.00 |
| + message-passing hidden 128 (`ed0dc7b`, 3 runs) | 3.67M | 5.2 s | 1.08 |
| Final (`e09d1fb` / `69b9bd9`, 8 runs) | 3.67M | 5.5 s | 1.13 |

- A local check agrees: on the same batch the final model needs about 20% more FLOPs than the original
  (56 vs 47 GFLOPs per 40-part batch) and about 4% more CPU time.
- The model is 2.7x bigger but only about 13% slower, because most of the work is the per-face surface CNN,
  which barely changed. The extra parameters sit in NNConv's edge network, which grows with hidden size squared.
- Absolute cost is small either way: the final model scored the 5,360 test parts in 4.5 s on a T4, about
  1,200 parts per second including data loading.
- **fp16 inference and hidden 64, tested (batch 1119-100420).** Model-only forward time over the 6,056 val parts on
  a T4, batches already on the GPU, mean of 4 runs:

  | Model | Parameters | fp32 | fp16 |
  |---|---|---|---|
  | Hidden 128 (`e09d1fb`) | 3.67M | 4.38 s (1,380 parts/s) | 1.75 s (3,450 parts/s) |
  | Hidden 64 | 1.38M | 3.65 s (1,660 parts/s) | 1.54 s (3,930 parts/s) |

  - fp16 inference is 2.4 to 2.5x faster and costs no accuracy: fp16 and fp32 val mIoU differ by at most 0.0001.
  - Hidden 64 scores 0.781 on the 7-class metric (2 seeds) vs 0.779 for its same-batch control and 0.784 for
    `e09d1fb` over 4 seeds: a tie within noise, with 2.7x fewer parameters.
  - Hidden 64 in fp16 is about 2.8x faster than hidden 128 in fp32, so with these two changes the new model is
    cheaper to run than the original UV-Net in fp32, not 13 % dearer.

### Overall

- **Retraining often**: the new recipe is far cheaper (about 1/8 of the GPU time per run) and more accurate.
- **Labelling many parts with a fixed model**: run it in fp16 (2.5x faster, same accuracy), and use hidden 64 if
  2 more seeds confirm the tie: about 2.8x cheaper than hidden 128 in fp32.
- **The search itself** cost about 12 to 17 GPU-hours (estimated). At about 70 minutes saved per training run
  it pays for itself after roughly 10 to 15 retrains; anyone reusing the finished recipe gets the saving free.
- **Caveats**: one dataset (Fusion 360 segmentation), one GPU type (T4), and the 78-minute reference is a
  single run whose 8-class score carries noise of about 0.03.

## Reproducing

- Best model: the `train.py` in this folder is `e09d1fb`. Run `python ar.py run e09d1fb --seeds 0,1 --budget 600` (or `4500`).
- Test scoring: `final_test_train.py` (commit `69b9bd9` on the local `final-test` branch): `train.py` plus a
  final pass over the test split. Needs `cache/test.pt` from the preprocessing kernel.
- Weights: `runs/1116-231848/output/results/69b9bd9-s{0,1}/model.pt` (local only, not in the repo).
- Full log of every run: `results.tsv`; running notes: `notes.md`.
