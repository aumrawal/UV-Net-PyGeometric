"""Confusion matrix of a segmentation run, optionally compared with a second run.

Reads per-face predictions from an .npz file that holds the true labels `y` and either predicted labels
(`pred_<key>`), class probabilities (`prob_<key>`) or logits (`logits`). `analysis/dump_logits.py` and the
Fusion 360 twin kernel write files in this format.

Single run (one panel, row-normalised %):

    python analysis/confusion_compare.py results/.../test_face_predictions.npz --key last \
        --out docs/fig/fusion360_confusion.png --csv docs/fig/fusion360_confusion.csv

Two runs (both panels plus their difference). The second run is either another .npz or a CSV of raw
counts (header row, then one row per true class: name, count per predicted class):

    python analysis/confusion_compare.py run_a.npz --label "UV-Net" \
        --other run_b.csv --other_label "My model" --out compare.png

Prints recall, precision and the largest off-diagonal confusions for each run.
"""
import argparse
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FUSION360_CLASSES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide",
                     "RevolveEnd"]


def load_npz(path, key):
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(int)
    if f"pred_{key}" in d.files:
        pred = d[f"pred_{key}"]
    elif f"prob_{key}" in d.files:
        pred = d[f"prob_{key}"].argmax(1)
    elif "logits" in d.files:
        pred = d["logits"].argmax(1)
    else:
        raise SystemExit(f"{path}: no pred_{key}, prob_{key} or logits array")
    classes = [str(c) for c in d["classes"]] if "classes" in d.files else None
    return y, pred.astype(int), classes


def counts(y, pred, n):
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def load_counts_csv(path):
    rows = list(csv.reader(open(path)))[1:]
    return np.array([[int(v) for v in r[1:]] for r in rows])


def norm(m):
    return m / np.maximum(m.sum(1, keepdims=True), 1)


def panel(ax, m, classes, title):
    n = norm(m)
    ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    k = len(classes)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{n[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if n[i, j] > 0.5 else "#222222")
    ax.set_title(title + "\nrow-normalised %, rows = true class", fontsize=11)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")


def report(name, m, classes):
    rec = np.diag(m) / np.maximum(m.sum(1), 1)
    pre = np.diag(m) / np.maximum(m.sum(0), 1)
    print(f"\n{name}: accuracy {np.trace(m) / m.sum():.4f} over {m.sum()} faces")
    print(f"{'class':12s} {'faces':>7s} {'recall':>7s} {'precision':>9s}")
    for i, c in enumerate(classes):
        print(f"{c:12s} {m[i].sum():7d} {rec[i] * 100:7.1f} {pre[i] * 100:9.1f}")
    off = [(m[i, j] / max(m[i].sum(), 1) * 100, classes[i], classes[j])
           for i in range(len(classes)) for j in range(len(classes)) if i != j]
    print("top confusions:", ", ".join(f"{a} -> {b} {p:.1f}%" for p, a, b in sorted(off, reverse=True)[:5]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help=".npz with y and pred_<key> / prob_<key> / logits")
    ap.add_argument("--key", default="last", help="which checkpoint's predictions to read (default: last)")
    ap.add_argument("--label", default="UV-Net (PyG)", help="title of the first panel")
    ap.add_argument("--other", help="second run: .npz (read with the same --key) or CSV of counts")
    ap.add_argument("--other_label", default="other run")
    ap.add_argument("--out", required=True, help="output figure (.png)")
    ap.add_argument("--csv", help="also write the first run's raw counts as CSV")
    a = ap.parse_args()

    y, pred, classes = load_npz(a.predictions, a.key)
    classes = classes or FUSION360_CLASSES
    m = counts(y, pred, len(classes))

    if a.other:
        if a.other.endswith(".npz"):
            yo, po, _ = load_npz(a.other, a.key)
            mo = counts(yo, po, len(classes))
        else:
            mo = load_counts_csv(a.other)
        assert mo.sum() == m.sum(), f"runs cover different faces: {m.sum()} vs {mo.sum()}"
        fig, ax = plt.subplots(1, 3, figsize=(21, 6.8))
        panel(ax[0], m, classes, a.label)
        panel(ax[1], mo, classes, a.other_label)
        dn = (norm(mo) - norm(m)) * 100
        ax[2].imshow(dn, cmap="RdBu", vmin=-25, vmax=25)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax[2].text(j, i, f"{dn[i, j]:+.0f}", ha="center", va="center", fontsize=9)
        ax[2].set_title(f"{a.other_label} minus {a.label} (points)\nblue on the diagonal = second run better",
                        fontsize=11)
        ax[2].set_xticks(range(len(classes)))
        ax[2].set_yticks(range(len(classes)))
        ax[2].set_xticklabels(classes, rotation=45, ha="right")
        ax[2].set_yticklabels(classes)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        panel(ax, m, classes, a.label)

    plt.tight_layout()
    plt.savefig(a.out, dpi=140)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + classes)
            for i, c in enumerate(classes):
                w.writerow([c] + [int(v) for v in m[i]])

    report(a.label, m, classes)
    if a.other:
        report(a.other_label, mo, classes)


if __name__ == "__main__":
    main()
