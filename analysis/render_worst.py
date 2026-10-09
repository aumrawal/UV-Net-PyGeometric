"""Render the worst-misclassified test features (ground truth vs prediction) for manual label review.

A "feature" is approximated as a connected group of faces sharing a true label (instances are not labelled
in MFCAD++). Candidates are feature groups where EVERY face is wrong, ranked by the model's mean confidence
in its wrong answer, with at most --per_pair cases per (true -> predicted) confusion.

Usage: python analysis/render_worst.py [--n 36] [--per_pair 3]
"""
import argparse, collections, html, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, ".")
from datasets.mfcadpp import MFCADPPDataset, MFCADPP_CLASSES
from process.step_to_pyg import read_step_with_labels
from occwl.graph import face_adjacency
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopAbs import TopAbs_REVERSED

RES = pathlib.Path("results/mfcadpp_full/0930_000433")
STEP = pathlib.Path("../MFCAD++_dataset/step/test")
STOCK = 24
CMAP = plt.get_cmap("tab20")
CLS_COLOR = {c: CMAP(c % 20) for c in range(24)}
CLS_COLOR[STOCK] = (0.85, 0.85, 0.85, 1.0)


def find_candidates(per_pair, n):
    d = np.load(RES / "confusion/test_preds.npz")
    P, Y, G, C = d["pred"], d["y"], d["graph"], d["conf"]
    starts = np.r_[0, np.cumsum(np.bincount(G))]
    ds = MFCADPPDataset("data/processed", "test")
    ids = json.load(open("data/processed/split.json"))["test"]
    cands = []
    for g in range(len(ds)):
        dt = ds[g]; y = dt.y.numpy(); k = len(y)
        p = P[starts[g]:starts[g] + k]; c = C[starts[g]:starts[g] + k]
        if (p == y).all():
            continue
        adj = [[] for _ in range(k)]
        for a, b in dt.edge_index.numpy().T:
            adj[a].append(b)
        seen = -np.ones(k, int)
        for s in range(k):
            if seen[s] >= 0 or y[s] == STOCK:
                continue
            comp, st = [s], [s]; seen[s] = s
            while st:
                u = st.pop()
                for v in adj[u]:
                    if seen[v] < 0 and y[v] == y[s]:
                        seen[v] = s; comp.append(v); st.append(v)
            comp = np.array(comp)
            if (p[comp] != y[s]).all():
                pm = collections.Counter(p[comp].tolist()).most_common(1)[0][0]
                cands.append(dict(g=g, model=ids[g], faces=comp.tolist(), true=int(y[s]), pred=int(pm),
                                  conf=float(c[comp].mean()), size=len(comp), n_faces=k))
    print(f"fully-flipped feature groups in test: {len(cands)}")
    cands.sort(key=lambda r: (-r["conf"], -r["size"]))
    cnt = collections.Counter(); out = []
    for r in cands:
        key = (r["true"], r["pred"])
        if cnt[key] < per_pair:
            cnt[key] += 1; out.append(r)
        if len(out) == n:
            break
    return out, (P, Y, G, C, starts), len(cands)


def raw(x):
    """occwl wrappers expose .topods_shape(); this occwl version may already hand back the OCC shape."""
    return x.topods_shape() if hasattr(x, "topods_shape") else x


def tessellate(model_id):
    solid, labels = read_step_with_labels(STEP / f"{model_id}.step")
    graph = face_adjacency(solid)
    shape = raw(solid)
    BRepMesh_IncrementalMesh(shape, 0.05, False, 0.3, True)
    node_ids = sorted(graph.nodes)
    tris = []
    for n in node_ids:
        face = raw(graph.nodes[n]["face"])
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        t = []
        if tri is not None:
            tr = loc.Transformation()
            pts = []
            for i in range(1, tri.NbNodes() + 1):
                q = tri.Node(i).Transformed(tr); pts.append((q.X(), q.Y(), q.Z()))
            pts = np.array(pts)
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                t.append(pts[[a - 1, b - 1, c - 1]])
        tris.append(np.array(t))
    lines = []
    for e in solid.edges():
        try:
            ad = BRepAdaptor_Curve(raw(e))
            f, l = ad.FirstParameter(), ad.LastParameter()
            pts = np.array([[q.X(), q.Y(), q.Z()] for q in (ad.Value(f + (l - f) * i / 24) for i in range(25))])
            lines.append(pts)
        except Exception:
            pass
    return tris, lines, [labels.get(graph.nodes[n]["face"], -1) for n in node_ids]


def draw(ax, tris, lines, colors, alphas, lo, hi, view):
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    polys, cols = [], []
    for t, c, a in zip(tris, colors, alphas):
        if len(t) == 0:
            continue
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        sh = 0.55 + 0.45 * np.abs(nrm @ light)
        for tri, s in zip(t, sh):
            polys.append(tri); cols.append((c[0] * s, c[1] * s, c[2] * s, a))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    ax.add_collection3d(Line3DCollection(lines, colors=(0, 0, 0, 0.5), linewidths=0.5))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo); ax.view_init(*view); ax.set_axis_off()


def render(r, preds, out_dir, rank):
    P, Y, G, C, starts = preds
    tris, lines, step_labels = tessellate(r["model"])
    k = r["n_faces"]; y = Y[starts[r["g"]]:starts[r["g"]] + k]; p = P[starts[r["g"]]:starts[r["g"]] + k]
    assert (np.array(step_labels) == y).all(), "face order mismatch between STEP and processed graph"
    allp = np.concatenate([t.reshape(-1, 3) for t in tris if len(t)]); lo, hi = allp.min(0), allp.max(0)
    grp = set(r["faces"])
    # camera: look at the group's centroid side, with a second view from the opposite corner
    gpts = np.concatenate([tris[i].reshape(-1, 3) for i in grp if len(tris[i])]); off = gpts.mean(0) - 0.5 * (lo + hi)
    az = np.degrees(np.arctan2(off[1], off[0])) if np.linalg.norm(off[:2]) > 1e-6 else 35
    views = [(28, az + 25), (28, az + 25 + 180)]
    fig = plt.figure(figsize=(14, 7.2))
    for row, (view) in enumerate(views):
        for col, (name, lab) in enumerate((("GROUND TRUTH", y), ("PREDICTION", p))):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            colors = [CLS_COLOR[int(lab[i])] for i in range(k)]
            alphas = [1.0 if i in grp else (0.18 if y[i] != STOCK else 0.08) for i in range(k)]
            # prediction panel: other faces keep their true colour, only the group shows the prediction
            if col == 1:
                colors = [CLS_COLOR[int(p[i])] if i in grp else CLS_COLOR[int(y[i])] for i in range(k)]
            draw(ax, tris, lines, colors, alphas, lo, hi, view)
            if row == 0:
                ax.set_title(f"{name}: {MFCADPP_CLASSES[r['true']] if col == 0 else MFCADPP_CLASSES[r['pred']]}", fontsize=11)
    fig.suptitle(f"#{rank}  model {r['model']}  |  true '{MFCADPP_CLASSES[r['true']]}' -> predicted '{MFCADPP_CLASSES[r['pred']]}'  "
                 f"|  {r['size']} face(s), mean conf {r['conf']:.2f}  |  other features faded", fontsize=11)
    fn = out_dir / f"{rank:02d}_{r['model']}.png"
    fig.savefig(fn, dpi=80, bbox_inches="tight"); plt.close(fig)
    return fn.name


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=36); ap.add_argument("--per_pair", type=int, default=3)
    a = ap.parse_args()
    out_dir = RES / "label_audit"; out_dir.mkdir(exist_ok=True)
    sel, preds, total = find_candidates(a.per_pair, a.n)
    rows = []
    for i, r in enumerate(sel, 1):
        try:
            rows.append((r, render(r, preds, out_dir, i)))
            print(i, r["model"], MFCADPP_CLASSES[r["true"]], "->", MFCADPP_CLASSES[r["pred"]], f"{r['conf']:.2f}", flush=True)
        except Exception as e:
            print("FAILED", r["model"], type(e).__name__, e)
    body = "".join(
        f'<section><h3>#{i} model {html.escape(r["model"])} &middot; true <b>{MFCADPP_CLASSES[r["true"]]}</b> &rarr; predicted '
        f'<b>{MFCADPP_CLASSES[r["pred"]]}</b> &middot; {r["size"]} face(s) &middot; conf {r["conf"]:.2f}</h3>'
        f'<img src="{fn}" style="max-width:100%"></section>' for i, (r, fn) in enumerate(rows, 1))
    (out_dir / "index.html").write_text(
        f"<html><head><meta charset=utf-8><title>Label audit</title></head><body style='font-family:sans-serif;max-width:1100px;margin:auto'>"
        f"<h2>Fully-misclassified features, most confident first</h2><p>{total} fully-flipped feature groups in the test set; "
        f"showing {len(rows)} (max per confusion pair). Judge: is the ground-truth label right?</p>{body}</body></html>")
    print("wrote", out_dir / "index.html")


main()
