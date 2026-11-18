# UV-Net autoresearch

You are an autonomous researcher improving **UV-Net face segmentation on the Fusion 360 Gallery
dataset** (8 modelling-operation classes per B-rep face). Training runs on Kaggle GPUs (T4); you
run on the human's machine and drive Kaggle through `ar.py`.

## Context

- `prepare.py`: the fixed harness: data, time budget, evaluation. **Read-only.**
- `train.py`: model, optimizer, loss, training loop. **The only file you edit.**
- `ar.py` + `ar_config.json`: the Kaggle runner. **Read-only.**
- `README.md`: how the setup works.

Read all of them before you start. The baseline `train.py` is the published UV-Net recipe
(1.37M parameters, Adam 1e-3, batch 64). Trained for 100 epochs (about 78 min on a T4) it reaches
val mIoU 0.669 and test mIoU 0.677. Under the 10-minute budget it gets about 12 epochs, so expect
the baseline score to be well below that.

What is known about its errors (from the full run's confusion matrix):
- **Cut faces are read as extrude faces.** 23% of CutSide go to ExtrudeSide, 19% of CutEnd go to
  ExtrudeEnd. What separates them is whether material was removed, which shows mostly in the
  convexity of the surrounding edges. The edge grids only carry points and tangents; the face grids
  carry normals, so convexity has to be inferred by comparing neighbouring faces' normals along the edge.
- **Frequent classes win ties.** ExtrudeSide is about half of all faces. Fillet, Chamfer and
  RevolveSide lose 6 to 8% of their faces to it.
- **RevolveEnd is barely learned**: very rare (73 test faces), 22% recall, mostly confused with ExtrudeEnd.
- Val loss bottoms out around epoch 40 while val IoU keeps creeping up, so it overfits in loss but not in IoU.

Available inputs per graph (see the `prepare.py` docstring): `x` (face UV-grids: xyz, normal, trim
mask), `edge_attr` (edge grids: xyz, tangent), `edge_index`, `batch`, `y`, and, when present,
`face_attr` (exact face area + bounding-box extent). You may derive anything you like from these
inside `train.py` (e.g. dihedral/convexity from the normals of the two faces at an edge). You may
not change the preprocessing.

## Rules

**You CAN** change anything in `train.py`: architecture, message passing, encoders, losses (class
weights, focal, label smoothing), optimizer, LR schedule, batch size, augmentation
(`make_train_dataset(random_rotate=True)`), mixed precision, data loader settings.

**You CANNOT**:
- modify `prepare.py`, `ar.py` or `ar_config.json` (the runner refuses commits that change `prepare.py`);
- install packages (only what is on the Kaggle image + `torch_geometric`);
- train on, or read, validation data inside `train.py`, or call `evaluate` anywhere except once,
  after training (no checkpoint selection on val: it would tune to the val split for free);
- touch the test split (it is not reachable from `prepare.py`, keep it that way).

**Metric**: `val_miou`, mean IoU on the full validation split over the 7 classes other than
RevolveEnd, **higher is better**. Reported by `prepare.evaluate`, which the runner reads directly.
RevolveEnd has a handful of val faces and its IoU swings 0.06-0.27 between seeds of identical code,
so it is left out of the score (it is still trained on). `val_miou8`, the mean over all 8 classes,
is logged too: watch it and RevolveEnd's IoU, but don't keep or discard on them.

**Budget**: training is cut after `TIME_BUDGET` = 600 s of wall clock from the first batch, data
loading included. A run that does not finish within budget + 15 min (startup + eval) is killed.

**Hardware notes (Kaggle T4, 16 GB, Turing)**: fp16 tensor cores but no fast bf16, so mixed
precision means `torch.autocast("cuda", dtype=torch.float16)` + `torch.amp.GradScaler`. Graph
batches change shape every step, so `torch.compile` usually hurts. The machine has 4 CPU cores and
runs two experiments side by side, so don't count on more than 1 to 2 data-loader workers.

**Noise**: the same code with a different seed moves val mIoU, by about 0.01 on the 7-class score.
Only call something an improvement if it beats the current best by clearly more than that spread;
for close calls, re-run it with more seeds (`--seeds 0,1`).

**Simplicity**: all else equal, simpler is better. A tiny gain that adds a lot of hacky code isn't
worth keeping; equal score with less code is a win.

## Setup (with the human)

1. Agree on a run tag (e.g. `nov14`). Create the branch: `git checkout -b autoresearch/<tag>` from `master`.
2. Check the data kernel finished: `kaggle kernels status aumr1101/uvnet-ar-fusion360-preprocess` says COMPLETE.
3. Check `python ar.py status` (should not be RUNNING).
4. Baseline batch: `python ar.py run HEAD --seeds 0,1`. This gives the starting score and the noise spread.
   Mark it: `python ar.py mark <hash> keep`.
5. Confirm with the human, then start the loop.

## The loop

Each Kaggle batch costs ~3 min of startup plus the runs, two at a time. A batch of **4 ideas** takes
about 25 min of wall clock and GPU quota. Kaggle gives roughly 30 GPU hours a week, so don't waste
batches on trivial variants.

LOOP:
1. `base=$(git rev-parse HEAD)` on `autoresearch/<tag>` (the current best).
2. For each of up to 4 ideas: `git checkout -q --detach $base`, edit `train.py`, then
   `git commit -qam "<short description of the idea>"` and note the hash. The commit subject
   becomes the description in `results.tsv`, so make it specific.
3. `git checkout -q autoresearch/<tag>`.
4. Run the batch: `python ar.py run <h1> <h2> <h3> <h4> > runs/batch.log 2>&1`, then read the table
   with `tail -n 30 runs/batch.log` (do not let the polling output flood your context). If you are
   interrupted while it runs, `python ar.py fetch` resumes.
5. For every run: `python ar.py mark <hash> keep|discard`. Crashes: read the log tail; fix and
   re-run if it was a silly bug, otherwise leave it as crash.
6. If the best idea beats the current best by more than the noise: `git reset --hard <winner>`
   (a fast-forward, since every idea is a child of `$base`). Consider combining two winners in the
   next batch (cherry-pick both onto the new base).
7. Every ~5 batches, write 3 to 5 lines to `notes.md`: what helped, what didn't, what to try next.

Ideas worth trying, roughly in order:
- throughput under the budget: mixed precision, larger batch with scaled LR, data loader workers;
- LR schedule (OneCycle / cosine over the time budget, using `train_time / TIME_BUDGET` as progress), AdamW + weight decay;
- class-balanced or focal loss, label smoothing;
- explicit convexity / dihedral-angle edge features from face normals, `face_attr` as extra face inputs;
- more message-passing rounds, wider hidden size, residual connections, attention or GATv2 in place of / next to NNConv;
- augmentation (`random_rotate`), dropout levels.

**NEVER STOP**: once the loop has begun, don't ask the human whether to continue. Keep going until
they stop you or until a batch fails because the Kaggle GPU quota is used up. In that case, write
`notes.md` and stop.

## Promoting a result (human decides)

A gain under 10 minutes doesn't guarantee a gain at convergence. When the human asks, run the best
commit with `python ar.py run <hash> --budget 4500 --seeds 0,1` (about the full 100-epoch time) and
compare it with the baseline's 0.669 val mIoU. The test split is only used outside this repo, on a
promoted model, by the human.
