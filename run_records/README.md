# Run records

The small files of every autoresearch Kaggle batch (`runs/<batch>/` locally, which is gitignored), without checkpoints,
result archives or the generated kernel sources:

- `<batch>/batch.json`: the experiments of the batch (commit, subject, seeds) and the time budget (600 s for the loop,
  4500 s for promotion and final test).
- `<batch>/output/results/<commit>-s<seed>/`: `result.json` (val mIoU, per-class IoU), `run.log` (per-epoch train loss
  and time), `train.py` (the exact code, also for discarded commits, which are not on this branch), and
  `test_result.json` for the final test.
- `local-*`: local smoke checks; `*.revs`: the commit lists of each batch; `runs_baseline.log`: the first batch's log.

`results.tsv` at the top of this folder is the one-line-per-run summary of all of these.
