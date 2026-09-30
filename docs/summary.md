# How this port was built

This page records what was done to port UV-Net from DGL to PyTorch Geometric and train it on MFCAD++, and
the problems solved along the way. For the numbers, see [`results.md`](results.md).

## Why a port

The official UV-Net code uses **DGL**, which has no Apple-silicon (MPS) backend, so it cannot use the GPU on
a Mac. **PyTorch Geometric** is pure Python on top of PyTorch and runs wherever PyTorch runs: CUDA, MPS or CPU.
It needs no compiled extras such as `torch_scatter` or `torch_sparse`.

**What stayed identical:**
- the architecture, layer sizes and weight initialisation;
- the input format (10×10×7 face grids, 10×6 edge grids) and the normalisation;
- the optimiser (Adam, lr 1e-3), batch size (64), and model-selection rule (lowest validation loss).

**What changed:** only the graph library and the engineering around it.

| Original (DGL) | This port (PyG) |
|---|---|
| `dgl.graph` with `ndata["x"]`, `edata["x"]` | `Data(x, edge_attr, edge_index, y)` |
| `dgl.batch` | `torch_geometric.loader.DataLoader` (a `batch` vector) |
| `NNConv(aggregator_type="sum", bias=False)` | `NNConv(aggr="add", root_weight=False, bias=False)` |
| `MaxPooling()` | `global_max_pool(h, batch)` |
| `graph_emb.repeat_interleave(batch_num_nodes)` | `graph_emb[batch]` |

Both versions compute the same edge-conditioned message passing, h_i' = Σ_{j→i} h_j · W(e_ji).

## Steps

1. **Environment.** Conda with Python 3.11, PyTorch 2.10, torch_geometric 2.8, pythonocc-core 7.7.2 and
   occwl 3.0.
2. **Preprocessing (`process/step_to_pyg.py`).** For each STEP file:
   - read the solid with OpenCascade;
   - take each face's label from the name of its `ADVANCED_FACE` entity, which is where MFCAD++ stores it;
   - build the face-adjacency graph with occwl;
   - sample the UV-grids with `occwl.uvgrid` and `occwl.ugrid`;
   - normalise the part and save one PyG graph per model.

   Files are processed in parallel with `multiprocessing`.
3. **Smoke test on synthetic data (`process/make_toy_dataset.py`).** 120 generated parts (a block with
   through holes, blind holes and pockets), labelled the same way MFCAD++ labels its files. The model reached
   100% test accuracy, which shows the pipeline is wired correctly end to end.
4. **Overfitting test.** On 64 real MFCAD++ models, validated on the same models, the model reached 100% face
   accuracy (loss 0.0023) in 200 epochs. This shows the model can fit the real labels.
5. **Full preprocessing.** 59,645 of 59,665 models converted, in about 20 minutes.
6. **Full training.** 100 epochs, 14.6 h on an M4 MacBook Air. The best validation loss came at epoch 51.
7. **Test.** 99.00% per-face accuracy and 0.968 mIoU on 8,948 test models.

## Problems and fixes

| Problem | Cause | Fix |
|---|---|---|
| Crash on import: `OMP: Error #15` | The pip PyTorch wheel and conda-forge OpenCascade each ship an OpenMP runtime | Install PyTorch from conda-forge |
| Training about 5× slower than expected on MPS | MPS compiles kernels for every new tensor shape (~0.5 s), and graph batches change shape every step | **Batch-shape bucketing:** pad each batch to multiples of 128 faces and 512 edges with a disconnected dummy graph whose outputs are dropped. Eval outputs are bit-identical to unpadded runs |
| 11 GB process memory and heavy swapping on a 16 GB machine | The whole dataset was held in RAM, and MPS kept buffers for every batch shape it had seen | **Memory-map** the packed data cache (`torch.load(mmap=True)`), and release MPS caches every 200 steps. This brought the footprint to about 7 GB |
| Slow start of every run | Loading about 42k small files | Pack each split into one cache file on first load; later runs load it in seconds |
| 141 MFCAD++ files contain "no solid" | They store a closed shell without a solid wrapper | Wrap the shell with `BRepBuilderAPI_MakeSolid` |
| 20 files still fail | 16 are non-manifold (an edge with more than two faces), 3 fail occwl's edge–face check, 1 is empty | Skipped and listed in `failures.json` |
| Long runs can be interrupted | 100 epochs take about 15 h on a laptop | `--resume last.pt` restores the model, optimiser and best-validation state |

**Also tried:** fp16 and bf16 autocast on MPS gave only a 4–10% speedup, so training stays in fp32, as in
the original.

## Ideas for closing the gap to the published 99.49%

- Find out the exact protocol behind the published number before tuning.
- Add a cosine learning-rate schedule and `--random_rotate` augmentation.
- Run several seeds and report the mean and spread.
