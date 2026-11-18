# Research notes (autoresearch/nov14)

## Batches 0–2 (Nov 14–15)

- **Baseline** (UV-Net recipe, 600 s on a T4): val mIoU 0.534 / 0.535 (seeds 0, 1), 9 epochs.
- **Convexity edge features help most so far**: +0.032 (0.566). CutSide IoU 0.43 → 0.51 and CutEnd 0.36 → 0.47,
  which is exactly the cut-vs-extrude confusion they target. RevolveEnd is still 0.
- **fp16 autocast is the biggest single gain**: epochs go from ~64 s to ~25 s, so 16 epochs instead of 9 and
  0.607 mIoU, with peak memory 2.3 → 0.9 GB. The GPU was the bottleneck, not the CPU.
- Cosine LR + warmup looked like +0.024 on its own (batch 1, not confounded with the cache effect below).
- Didn't help: 2 data-loader workers (fewer epochs: 4 CPUs shared by two runs), inverse-sqrt class weights
  (−0.036, hurts the big classes far more than it helps RevolveEnd).

## Harness lesson

- Runs in the first pair of a batch lost ~200 s to paging the 3.5 GB train cache in from /kaggle/input
  (epoch 1 took 280 s vs 71 s for the second pair). Batch 2's label smoothing (+0.029) and 3-round
  message passing were in the warm second pair, so their gains are confounded; they're re-tested in batch 3.
  Since commit 3297c08 the kernel warms the cache before any run starts.
- Baseline and batches 1–2 all paid part of that cost, so their absolute scores are a little low. Batch 3
  re-runs the current best for a fair reference.

## Batches 3–4: the warm cache, and noise

- With the cache warm, the current best (convexity + cosine + fp16) gets 26 epochs in 600 s and 0.669 val mIoU,
  the same val mIoU as the original 100-epoch recipe.
- **Seed noise is ~0.026 mIoU** (same code: 0.669 vs 0.643). Most of it is RevolveEnd, a handful of val faces
  that swings its IoU between 0.08 and 0.20 and so moves mIoU by ±0.014 on its own. Single-seed comparisons
  under ~0.03 mean nothing. From batch 5 on: 2 seeds per idea, compare means.
- Single-seed results within that noise, worth re-testing with seeds: hidden 128 (0.663), batch 128 (0.650),
  LR 4e-3 (0.648), 3 rounds (0.642), label smoothing (0.641).
- Clearly worse: random rotation (0.593, data loading drops epochs from 26 to 19).
- Possible harness change for the human to decide: mIoU on full val is dominated by RevolveEnd noise.

## Batches 5–8: hidden 128 wins; the metric is now the bottleneck

- Hidden 128 (ed0dc7b): mean 0.675 over 3 seeds vs 0.656 over 4 for the previous best. Kept.
- No gain on the official metric (2 seeds each): wider encoders 0.669, label smoothing 0.650, AdamW 0.669,
  EMA 0.658, local convexity 0.672, face size 0.675.
- **mIoU over the 7 classes other than RevolveEnd is far less noisy** (seed spread ~0.01 vs ~0.04).
  RevolveEnd IoU on identical code ranges 0.06–0.27. On the 7-class score, local convexity (3ccc6b5) is a
  clear win: 0.755 vs 0.744, and both its seeds beat every seed of the best. AdamW, face size and wider
  encoders are ~+0.005 (unclear).
- Loop paused after batch 8 for the human to decide on the metric (prepare.py is theirs).

## Metric switch (harness commit ac85b7d, Nov 15)

- The human chose the 7-class score. `val_miou` is now mean IoU over the 7 classes other than RevolveEnd;
  the 8-class mean is logged as `val_miou8`. results.tsv was rescored from every run's saved per-class IoU
  (8-class original kept in runs/results_8class_backup.tsv).
- On the new score: baseline 0.611, previous best ed0dc7b 0.744 (3 seeds), **local convexity 3ccc6b5 0.755
  (2 seeds), the new best**. Face size 68dd24c 0.749 discarded.
- Batch 9 (2 seeds each, on local convexity): face size, AdamW wd 0.05, surface/curve embeddings 128.

## Next

- Now that epochs are cheap: more capacity (hidden 128, wider surface CNN), longer cosine tail.
- RevolveEnd (0 IoU): needs something other than reweighting, e.g. a face-level "is planar end face of a
  revolve" cue from the normals/positions relative to the part axis.

## Batches 9–10 (7-class score)

- Batch 9, on local convexity (best 0.7545): AdamW 0.7556, face size 0.7521, wider embeddings 0.7524. All
  within noise, discarded.
- Batch 10: **NNConv root weight (b8b03ed) 0.7590 (0.7583 / 0.7597), new best**; both seeds beat both of the
  previous best, and the 8-class score rises too (0.687 vs 0.672). Jumping knowledge 0.7575 (0.754 / 0.761),
  unclear. 3 rounds: 0.741 and one collapsed run (0.074).
- **fp16 hazard**: the collapsed run had a NaN loss in epoch 1. GradScaler skips that step, but BatchNorm
  running stats already absorbed the NaN, so eval is garbage while train loss looks normal. Batch 11 tests
  message passing in fp32.
- Batch 11, on root weight (best 0.759): jumping knowledge 0.758 (tight, but root weight already gives a
  skip path), fp32 message passing 0.755 (no NaN, but 19 epochs instead of 23), 3 rounds in fp32 0.740
  (stable but clearly worse: depth beyond 2 rounds doesn't pay at this budget). All discarded.
- The NaN happened in 1 of 52 runs so far (the 3-round model). Not worth a guard yet; if a run scores
  ~0.07 with a normal train loss, look for `loss nan` in its log and re-run it.
- Batch 12: mean aggregation, classifier dropout 0.5, batch size 32.
- Batch 12, on root weight: **mean aggregation in NNConv (c68e4e2) 0.7657 (0.763 / 0.768), new best**, both
  seeds above both of root weight's. Dropout 0.5 0.757, batch 32 0.751 (19 epochs). Discarded.
- Batch 13, on mean aggregation: log face degree (mean hides it), max + mean part pooling, peak LR 3e-3.
- Batch 13, on mean aggregation: face degree 0.768 (unclear), max + mean pooling 0.759, LR 3e-3 0.760.
- Batch 14, seeds 2,3: mean aggregation 4-seed mean **0.7668** (0.763–0.769); face degree 0.7682
  (0.766–0.770). +0.0014 is inside the spread and adds code: discarded.
- Batch 15: face (and edge) grids centred on their own centroid, centroid fed separately, so the CNNs see
  local shape independent of where the face sits in the part.
- Batch 15: **face + edge grids centred on their centroids (1fdedee) 0.7756 (0.778 / 0.773), new best**,
  both seeds above all four of mean aggregation's; 8-class 0.718. Faces only: 0.770, so edges matter most.
- Batch 16, on centring: also divide by each grid's RMS radius (log size fed separately), faces / faces + edges.
- Batch 16: scale normalisation **collapsed** (val 0.31–0.41) while train loss was normal or lower (0.14):
  a train/eval mismatch hitting every class. Likely degenerate or tiny faces: dividing by a ~0 radius
  blows up the grid (especially off-mask points); per-batch BatchNorm hides it in training, but the
  running stats used at eval are wrecked. Batch 17 retries with radius >= 0.02 and grids clipped to +-5.
- Batch 17: with the floor and clipping, scale normalisation works. Faces + edges (e09d1fb) 0.7825
  (0.779 / 0.786) vs centring 0.7756; faces only 0.777 (0.772 / 0.783). The low seed only just clears the
  best's high one, so batch 18 runs seeds 2,3 of both.
- Batch 18, seeds 2,3: **scale normalisation (e09d1fb) 4-seed mean 0.7840 (0.779–0.788) vs centring 0.7767
  (0.773–0.780): kept, new best.** 8-class 0.722.
- Batch 19, on scale normalisation: rotation augmentation done on the GPU per batch (the CPU version cost
  epochs), 90-degree steps vs uniform random.
- Batch 19: rotation augmentation hurts even at the same epoch count: 90-degree steps 0.767, uniform 0.722
  (vs 0.784). Fusion 360 parts have a meaningful canonical orientation (extrude directions on axes), so
  orientation is signal. Don't augment it away.
- Batch 20, on scale normalisation: convexity at every point along the edge as extra curve-CNN channels;
  surface/curve embeddings 128 (neutral before, inputs are richer now).
- Batch 20: convexity along the whole edge 0.7834, wider embeddings 0.7808 (vs 0.7840). Discarded.
- Batch 21: masked mean pooling in the surface CNN (off-face grid points excluded); cudnn.benchmark.
- Batch 21: masked surface pooling 0.7796 (discard); cudnn.benchmark re-tunes per batch shape, so 8 epochs
  instead of 23 (0.753, discard).
- **Plateau**: batches 19–21 brought nothing over e09d1fb (0.784, 4 seeds). Loop paused for the human:
  promotion run of e09d1fb at the full ~100-epoch budget, or more 10-minute ideas (Kaggle GPU quota).

## Promotion run (batch 1116-105724): e09d1fb for 4500 s (~170 epochs), seeds 0,1

- 7-class **0.7836** (0.7845 / 0.7826): the same as the 10-minute score (0.784). 8-class **0.728**
  (0.723 / 0.734) vs 0.722 at 10 min, and vs 0.669 for the original UV-Net recipe at ~100 epochs.
- The only class that gains from long training is RevolveEnd: 0.29 / 0.39 IoU (vs ~0.06–0.18 at 10 min).
- Train loss falls to 0.002 (memorised), val loss 0.94–0.97. **The model is limited by data /
  regularisation, not by compute**: the 10-minute loop is a faithful proxy, and more epochs won't help.
- Next ideas that fit this: augmentation that keeps the canonical orientation (axis-aligned mirror flips,
  small anisotropic scaling, point jitter), stronger regularisation, and RevolveEnd-specific cues.
  The test split is still untouched.

## Batch 22: augmentation (regularisation), 10-minute budget

- Stretch per axis 0.780, point jitter 0.778, mirror flips 0.762 (vs 0.784). Discarded on the 10-min score.
- **Caveat: the 10-minute proxy can't judge regularisation fairly.** At 10 min the model isn't overfitting
  yet (train loss ~0.10); it only memorises in long runs (0.002 at 170 epochs). Anything that slows
  fitting is penalised at 10 min even if it would raise the long-run plateau.
  - Mirror makes training harder (train loss 0.164 vs 0.101) and hurts CutSide / RevolveSide by ~0.05.
  - Stretch and jitter leave train loss unchanged (~0.105): at this strength they barely regularise.
- Long-run seed spread is small (0.7845 / 0.7826), so a single 4500 s seed per augmentation is a fair test.

## Batch 23: augmentation at full length (4500 s, seed 0), vs e09d1fb 0.7845

- Stretch 0.7814 (8-class 0.727), mirror 0.7555 (8-class 0.710). **Both still worse at full length**, so
  the 10-minute verdict holds: these augmentations don't help, not merely slow down fitting.
- Mirror hurts most where cut vs extrude and revolve are decided: CutSide 0.617 vs 0.680, CutEnd 0.570 vs
  0.610, RevolveSide 0.675 vs 0.720. Likely cause: Fusion 360 designers mostly sketch on a base plane and
  extrude up / cut down, so the *sign* of a face's direction carries label information. Mirroring erases it.
- Conclusion: in this dataset the absolute frame (orientation and its sign) is a feature, not a nuisance.
  Geometry-changing augmentation is a dead end here; the remaining gap is memorisation, which these don't
  fix. Remaining routes: more data, class-targeted features (CutSide / CutEnd are the weakest), and the
  final test-set check of e09d1fb.

## Final test (branch final-test, 69b9bd9 = e09d1fb + test scoring; 4500 s, ~180 epochs, 5360 test parts)
| seed | val 7cls | val 8cls | test 7cls | test 8cls | test acc |
|---|---|---|---|---|---|
| 0 | 0.7850 | 0.7286 | 0.8023 | 0.7494 | 0.9181 |
| 1 | 0.7731 | 0.7166 | 0.7989 | 0.7458 | 0.9176 |
| mean | 0.779 | 0.723 | **0.801** | **0.748** | 0.918 |

Per-class test IoU (s0 / s1): ExtrudeSide .895/.896, ExtrudeEnd .890/.892, CutSide .707/.706,
CutEnd .672/.681, Fillet .845/.838, Chamfer .857/.831, RevolveSide .749/.749, RevolveEnd .379/.374.
- Test runs ~0.02 above val on both metrics, and seeds agree within 0.004 on test, so no sign of
  overfitting the val split during the loop.
- Original UV-Net recipe in this harness: 0.669 val 8-class (promotion-length). Final model: 0.748 test 8-class.
- Weights: runs/1116-231848/output/results/69b9bd9-s{0,1}/model.pt (local only, not on W&B).
