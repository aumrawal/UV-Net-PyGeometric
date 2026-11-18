# uvnet-autoresearch

[karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted to UV-Net face segmentation
on Fusion 360 Gallery (from [UV-Net-PyGeometric](../UV-Net-PyGeometric)), with training on Kaggle GPUs.

A coding agent edits `train.py`, runs batches of ideas on Kaggle for a fixed 10-minute budget each, keeps
what raises validation mean IoU, and repeats. The human edits `program.md`.

| File | Role | Who edits |
|---|---|---|
| `prepare.py` | data, time budget, `evaluate` (val mIoU) | nobody (frozen at tag `harness`) |
| `train.py` | model, loss, optimizer, loop | the agent |
| `program.md` | the agent's instructions | the human |
| `ar.py`, `ar_config.json` | Kaggle runner | nobody |
| `kaggle/preprocess/` | one-off kernel: Fusion 360 s2.0.0 → packed graph caches | nobody |

How a batch runs: `python ar.py run <commit>...` embeds each commit's `train.py` in one Kaggle script
kernel (`aumr1101/uvnet-ar-experiments`, GPU T4 x2) that reads the data kernel's output, runs one
experiment per GPU, and writes `summary.json`. The runner waits, downloads it to `runs/<batch>/` and
appends to `results.tsv`.

```bash
python ar.py run HEAD --seeds 0,1      # baseline + noise
python ar.py mark <hash> keep
```

Local smoke test (CPU, random data):

```bash
python ar.py fake-data /tmp/fake && AR_DATA_DIR=/tmp/fake python ar.py local HEAD --budget 20
```

Then point an agent at `program.md`: "read program.md and let's set up a new run".
