# Confusion analysis: what the model gets wrong on MFCAD++

This page looks at the errors of the best-validation checkpoint (epoch 51) on the MFCAD++ test split:
268,956 faces, of which 2,689 are wrong (1.00%). The accuracy figures are in [`results.md`](results.md).

**The question:** are the weak classes confused with their through/blind partners (which would point to
missing global context), or with something else?

**The answer:** mostly something else.
- Through/blind partner swaps are **7.7%** of the wrong faces.
- The largest group, **27.5%**, is features of the same depth that differ in how many sides are open, for
  example a rectangular passage labelled as a rectangular through slot.
- **71%** of wrong feature faces belong to features whose face count differs from the usual one for
  their class, which is what happens when a neighbouring feature cuts into them.

## Files

| File | Contents |
|---|---|
| [`confusion/confusion_matrix.csv`](confusion/confusion_matrix.csv) | The 25×25 matrix of face counts. Rows are true classes, columns are predicted classes |
| [`confusion/confusion_report.html`](confusion/confusion_report.html) | The matrix as a shaded grid, with the tables below. Download it and open it in a browser |
| [`confusion/group_stats.json`](confusion/group_stats.json) | Error counts by class and by feature face count |

## Wrong faces by kind of confusion

Each wrong face is counted once, by its true and predicted class.

| Kind | Wrong faces | Share |
|---|---|---|
| Same depth, different number of open sides (passage / through slot / through step; pocket / blind slot / blind step) | 739 | 27.5% |
| Two or more attributes differ (everything not in another row) | 488 | 18.1% |
| Same feature type, different profile shape (triangular / rectangular / 6-sided; 2-sided / slanted / rectangular step) | 435 | 16.2% |
| Round blind features (O-ring / blind hole / circular end pocket) | 382 | 14.2% |
| Chamfer or round with another feature | 223 | 8.3% |
| Stock with any feature | 216 | 8.0% |
| **Through and blind partners** (same profile, with or without a floor) | 206 | 7.7% |

19.1% of errors cross the through/blind line in any way. The rest stay on one side of it, or involve
chamfer, round or stock.

## The most confused pairs

| Pair | Wrong faces | Share of all errors | Direction |
|---|---|---|---|
| Rectangular passage ↔ Rectangular through slot | 288 | 10.7% | 125 labelled as the second, 163 as the first |
| O-ring ↔ Blind hole | 253 | 9.4% | 169 labelled as the second, 84 as the first |
| 2-sided through step ↔ Slanted through step | 149 | 5.5% | 71 labelled as the second, 78 as the first |
| Rectangular pocket ↔ Rectangular blind slot | 117 | 4.4% | 51 labelled as the second, 66 as the first |
| Slanted through step ↔ Triangular blind step | 94 | 3.5% | 30 labelled as the second, 64 as the first |
| Rectangular through step ↔ Stock | 90 | 3.3% | 75 labelled as the second, 15 as the first |
| Rectangular through slot ↔ Rectangular through step | 82 | 3.0% | 44 labelled as the second, 38 as the first |
| Rectangular through step ↔ Rectangular blind step | 79 | 2.9% | 54 labelled as the second, 25 as the first |
| Triangular passage ↔ Triangular through slot | 79 | 2.9% | 34 labelled as the second, 45 as the first |
| Blind hole ↔ Circular end pocket | 76 | 2.8% | 22 labelled as the second, 54 as the first |
| Rectangular pocket ↔ Rectangular blind step | 57 | 2.1% | 37 labelled as the second, 20 as the first |
| O-ring ↔ Circular end pocket | 53 | 2.0% | 8 labelled as the second, 45 as the first |

## The five weakest classes

| Class | IoU | Its faces were labelled as | Faces wrongly given its label were | Errors with its through/blind partner |
|---|---|---|---|---|
| Rectangular through slot | 0.887 | Rectangular passage 163, Rectangular through step 44, Stock 11 | Rectangular passage 125, Rectangular through step 38, Rectangular blind slot 29 | 32 of 490 (7%) |
| Round | 0.932 | Through hole 17, Chamfer 11, O-ring 7 | Chamfer 10, Circular end pocket 5, Circular blind step 4 | no partner class (84 errors) |
| Blind hole | 0.943 | O-ring 84, Circular end pocket 22, Through hole 14 | O-ring 169, Circular end pocket 54, Through hole 16 | 30 of 394 (8%) |
| Rectangular blind slot | 0.944 | Rectangular pocket 66, Rectangular through slot 29, Rectangular blind step 17 | Rectangular pocket 51, Rectangular blind step 9, Vertical circular end blind slot 7 | 32 of 206 (16%) |
| Rectangular through step | 0.946 | Stock 75, Rectangular blind step 54, Rectangular through slot 38 | Rectangular through slot 44, Rectangular passage 33, Rectangular blind step 25 | 79 of 384 (21%) |

- **Rectangular through slot** is mostly swapped with rectangular passage, in both directions. Both go
  through the part; they differ in whether the cut is closed on all four sides.
- **Blind hole** is mostly swapped with O-ring, and **rectangular blind slot** with rectangular pocket. In
  each pair both features are blind.
- **Rectangular through step** is the one weak class with a real through/blind share, about a fifth of its
  errors. A slightly larger share is confusion with stock.
- **Round** is a different case. Its errors are with through hole and chamfer, on ordinary single-face
  rounds. There are only 84 of them, but the class is rare, so they move its IoU a lot.

## Usual and unusual feature shapes

MFCAD++ does not label feature instances in the data used here, so touching faces with the same label are
treated as one feature. For each class, the most common face count is taken as the usual shape. A different
count means a neighbouring feature removed or split faces, or two features of the same class touch. This
grouping is an approximation.

| Class | Usual face count | Error rate, usual count | Error rate, other counts | Ratio | Share of the class's missed faces on other counts |
|---|---|---|---|---|---|
| Rectangular blind slot | 4 | 1.45% | 14.11% | 10× | 65% |
| Rectangular through slot | 3 | 2.68% | 9.66% | 4× | 76% |
| Blind hole | 2 | 0.87% | 9.53% | 11× | 64% |
| Rectangular pocket | 5 | 0.19% | 7.34% | 39× | 85% |
| Rectangular through step | 2 | 0.96% | 7.21% | 8× | 82% |
| Triangular blind step | 2 | 0.87% | 7.12% | 8× | 65% |
| Triangular through slot | 2 | 0.38% | 6.97% | 18× | 92% |
| O-ring | 3 | 0.73% | 6.77% | 9× | 67% |
| Circular end pocket | 5 | 0.12% | 5.87% | 47× | 89% |
| Slanted through step | 2 | 0.95% | 5.27% | 6× | 75% |
| Vertical circular end blind slot | 4 | 0.67% | 4.83% | 7× | 53% |
| Round | 1 | 4.52% | 3.93% | 1× | 21% |
| Rectangular passage | 4 | 0.55% | 3.51% | 6× | 79% |
| Horizontal circular end blind slot | 4 | 0.20% | 3.47% | 18× | 69% |
| Triangular pocket | 4 | 0.16% | 3.33% | 21× | 62% |
| Circular through slot | 1 | 0.50% | 3.13% | 6× | 71% |
| 2-sided through step | 3 | 0.44% | 2.94% | 7× | 75% |
| Through hole | 1 | 0.49% | 2.94% | 6× | 63% |
| Circular blind step | 2 | 0.14% | 2.46% | 17× | 74% |
| Rectangular blind step | 3 | 0.25% | 2.08% | 8× | 75% |
| Triangular passage | 3 | 0.50% | 1.99% | 4× | 54% |
| Chamfer | 1 | 2.50% | 1.81% | 1× | 20% |
| 6-sided pocket | 7 | 0.03% | 1.27% | 36× | 78% |
| 6-sided passage | 6 | 0.13% | 0.67% | 5× | 68% |
| **All 24 feature classes** | | 0.46% | 4.16% | 9× | 71% |

Features with an unusual face count hold 22% of the feature faces and 71% of the feature-face
errors. Examples:

- A rectangular passage with its usual 4 faces is wrong 0.5% of the time. With 3 faces it is wrong
  30% of the time.
- An O-ring with its usual 3 faces is wrong 0.7% of the time. With 2 faces, the same count as a blind
  hole, it is wrong 27% of the time.
- A circular end pocket with 5 faces is wrong 0.1% of the time. With 2 faces it is wrong
  64% of the time.

48% of wrong faces touch a face of the same feature that the model labelled correctly, so the
model often splits one feature between two labels.

## What this suggests

- **Whether a cut goes through the part is not the main gap.** A change aimed only at that would address
  under a tenth of the errors.
- **Features cut by neighbours are the main gap.** A passage that has lost a wall looks like a slot. This
  analysis cannot tell whether the surrounding faces hold enough information to recover the right label,
  or whether some of these cases are ambiguous in the data. Looking at a sample of them in a CAD viewer
  would settle that.
- **Round is a local geometry problem**, between convex and concave curved faces and chamfers. Edge
  convexity or face curvature are the candidate inputs.
- **The usual/unusual split is a reusable metric.** Any model variant can be scored separately on the two
  subsets, which shows what it fixed.

## How to reproduce

The scripts are in `analysis/`. They are run from the repository root and expect the preprocessed data in
`data/processed`. The run folder (`results/mfcadpp_full/0930_000433`) is written into each script, so
change it there to analyse a different run.

```bash
OUT=results/mfcadpp_full/0930_000433/confusion
mkdir -p $OUT
python analysis/dump_preds.py $OUT/test_preds.npz   # per-face true label, prediction and confidence
python analysis/analyse.py                          # the matrix, per-class confusions, top pairs
python analysis/diag.py                             # kinds of confusion, usual vs unusual feature shapes
python analysis/make_page.py                        # confusion_matrix.csv and confusion_report.html
```

`dump_preds.py` needs the trained checkpoint (`best.pt`) and takes about a minute on an Apple M4. The
per-face predictions file (7.5 MB) is not stored in the repository.
