# Summary: the PyTorch Geometric variation of UV-Net

This note explains how our variation of UV-Net was built, why it differs from the official code, and what
it achieved. The result table is in [`README.md`](README.md); this file is the narrative behind it.

## In one paragraph

We re-implemented UV-Net on **PyTorch Geometric (PyG)** instead of DGL, so that it trains on the Apple-silicon
GPU (MPS) of a MacBook Air (M4). The original code at `../UV-Net` was left untouched. We wrote our own MFCAD++
preprocessing (STEP → UV-grid graphs, with labels read from the STEP file) and verified the pipeline on
synthetic data and with an overfitting test. We then trained on the full official split for 100 epochs.
The result is **99.00% per-face test accuracy and 0.968 mean IoU**, 0.49 pt below the published 99.49%.
That is within the ~1 pt tolerance we set for a reproduction.

## Why a variation, and what "variation" means here

- **Why:** DGL has no Apple MPS backend, so the official code cannot use the GPU on this machine. Re-implementing the
  encoder in plain PyTorch Geometric was the way around it.
- **What stayed the same:** the architecture, dimensions, initialisation, optimiser (Adam, lr 1e-3), batch
  size (64), input format (10×10×7 face grids, 10×6 edge grids), normalisation, and model-selection rule
  (lowest validation loss). The model has 1,368,925 parameters.
- **What changed:** only the graph library and the engineering around it. The DGL → PyG mapping is:

| UV-Net (DGL) | Our variation (PyG) |
|---|---|
| `dgl.graph` with `ndata["x"]`, `edata["x"]` | `Data(x, edge_attr, edge_index, y)` |
| `dgl.batch` | `torch_geometric.loader.DataLoader` (a `batch` vector) |
| `NNConv(aggregator_type="sum", bias=False)` | `NNConv(aggr="add", root_weight=False, bias=False)` |
| `MaxPooling()` | `global_max_pool(h, batch)` |
| `graph_emb.repeat_interleave(batch_num_nodes)` | `graph_emb[batch]` |

Both compute the same edge-conditioned message passing, h_i' = Σ_{j→i} h_j · W(e_ji).

## What we did, step by step

1. **Environment.** We set up a conda env `uvnet_pyg` with Python 3.11, PyTorch 2.10 (MPS),
   torch_geometric 2.8, pythonocc-core 7.7.2 and occwl 3.0. PyTorch had to come from conda-forge: the pip
   wheel together with conda-forge OpenCascade loaded two OpenMP runtimes and crashed.
2. **Preprocessing (`process/step_to_pyg.py`).** For each STEP file:
   - read the solid with OpenCascade;
   - read each face's machining-feature label from the name of its `ADVANCED_FACE` entity, which is how
     MFCAD++ stores labels;
   - build the face-adjacency graph with occwl;
   - sample the UV-grids with `occwl.uvgrid`/`ugrid` (points, normals and trim mask for faces; points and
     tangents for edges);
   - normalise the part to [−1, 1]³;
   - save one PyG graph per model.

   The official `train/val/test.txt` split is used.
3. **Smoke test.** We generated 120 synthetic STEP files (a stock block with holes and pockets, labelled the
   same way MFCAD++ labels its files) and trained on them. The model reached 100% test accuracy, which proved
   the pipeline is wired end to end.
4. **Overfitting test.** On 64 real MFCAD++ models, validated on the same models, the model reached 100% face
   accuracy (loss 0.0023) after 200 epochs. This confirmed that the model and labels are consistent on real
   data.
5. **Full preprocessing.** 59,645 of 59,665 models converted, in about 20 minutes with 10 processes.
   - **Recovered:** 141 files stored a closed shell without a solid; these are wrapped into a solid and kept.
   - **Lost:** 20 files (0.03%) cannot be processed. 16 are non-manifold, 3 fail occwl's edge–face check,
     and 1 contains no geometry.
6. **Full training.** 100 epochs, 14.6 h on the M4 GPU. The best validation loss came at epoch 51.
7. **Evaluation** on the official test split (8,948 models, 268,956 faces), shown below.

## Results

| | Test per-face accuracy | Test mIoU |
|---|---|---|
| **Ours, `best.pt` (epoch 51)** | **99.00%** | **0.968** |
| Ours, `last.pt` (epoch 100) | 98.96% | 0.966 |
| Published UV-Net (BrepMFR, via the MFCAD++ benchmark table) | 99.49% | – |

- **Plateau:** validation accuracy levels off at about 99.0% from around epoch 40, and later epochs overfit
  slightly.
- **Weakest classes:** rectangular through slot (IoU 0.887), round (0.932) and blind hole (0.943). These are
  mostly features that look alike locally, such as through vs blind slots.

## Engineering problems we hit, and how we solved them

| Problem | Cause | Fix |
|---|---|---|
| Crash on import (`OMP: Error #15`) | pip PyTorch and conda-forge OpenCascade each bring their own OpenMP runtime | Install PyTorch from conda-forge |
| Training ~5× slower than expected on MPS | MPS recompiles kernels for every new tensor shape (~0.5 s), and graph batches change shape every step | **Shape bucketing:** pad each batch to multiples of 128 faces and 512 edges with a disconnected dummy graph whose outputs are dropped. Eval outputs are bit-identical to unpadded runs |
| 11 GB process memory, swap on a nearly full disk | Whole dataset held in RAM, plus MPS buffers cached for every batch shape | Memory-map the packed data cache (`torch.load(mmap=True)`) and release MPS caches every 200 steps. This brought the footprint to about 7 GB |
| Loading 42k small files at every run | One `.pt` file per model | Pack each split into one cache file (`data/processed/cache/<split>.pt`), built once and reused |
| 141 MFCAD++ files "have no solid" | They store a closed shell only | Wrap the shell with `BRepBuilderAPI_MakeSolid` |
| Long runs can be interrupted | 100 epochs take about 15 h | `--resume last.pt` restores the model, optimiser and best-val state |

We also benchmarked fp16 and bf16 autocast on MPS. They gave only a 4–10% speedup, so training stays in fp32
to match the original.

## Differences from the official setup that could matter

- **Graph library:** PyG instead of DGL. The maths is identical, but the kernels differ.
- **Shape bucketing:** during training, BatchNorm statistics see a few percent of duplicated padding rows.
- **Excluded models:** 20 unprocessable models, of which 1 is in the test split.
- **Logging and seeds:** TensorBoard instead of W&B, and a single seed (0). MPS kernels are not
  bit-deterministic.
- **The target number itself:** BrepMFR's exact settings for its UV-Net run (epochs, schedule, augmentation)
  have not been checked yet.

## How to reproduce

```bash
conda env create -f environment.yml && conda activate uvnet_pyg
pip install --no-deps git+https://github.com/AutodeskAILab/occwl.git
python process/step_to_pyg.py /path/to/MFCAD++_dataset data/processed --num_processes 8
python train.py train --data_dir data/processed --epochs 100 --batch_size 64 --experiment_name mfcadpp_full
python train.py test  --data_dir data/processed --checkpoint results/mfcadpp_full/<run>/best.pt
```

On a CUDA machine the same commands run unchanged; shape bucketing is off by default outside MPS.

## Open items

- **Setup docs:** write `docs/setup-training.md` so the run can be reproduced from the repo alone.
- **Closing the gap:** check BrepMFR's UV-Net protocol first, then try a cosine LR schedule and
  `--random_rotate`, and run more seeds.
