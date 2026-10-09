"""Fusion 360 failure analysis, dataset comparison with MFCAD++, and a probe for topological features.

Usage: python analysis/f360_report.py [errors] [datasets] [ambiguity] [probe]     (default: all four)
"""
import collections, hashlib, json, sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "analysis")
from topo import FEATS, face_features, load_pack, nb_mean
from datasets.fusion360 import FUSION360_CLASSES as FC
from datasets.mfcadpp import MFCADPP_CLASSES as MC

F360 = "data/fusion360_processed/fusion360_processed"
MFC = "data/processed"
RUN = "results/kaggle_fusion360/results/fusion360_full/1006_010219/analysis"
OUT = {}
todo = sys.argv[1:] or ["errors", "datasets", "ambiguity", "probe"]


def miou(p, y, k):
    return float(np.mean([((p == c) & (y == c)).sum() / max(((p == c) | (y == c)).sum(), 1) for c in range(k)]))


def bucket_table(title, key, edges, labels, wrong):
    print(f"\n{title}"); rows = []
    for i, lab in enumerate(labels):
        m = (key >= edges[i]) & (key < edges[i + 1])
        if m.sum():
            rows.append((lab, int(m.sum()), float(wrong[m].mean())))
            print(f"  {lab:>14s}: {m.sum():6d} faces ({m.mean():5.1%})  error {wrong[m].mean():6.2%}  share of all errors {wrong[m].sum() / wrong.sum():5.1%}")
    return rows


# ---------------------------------------------------------------- 1. where the Fusion 360 model fails
if "errors" in todo:
    print("=" * 30, "1. FUSION 360 ERRORS (best.pt, test)")
    pk = load_pack(F360, "test"); d = np.load(f"{RUN}/test_best.npz")
    L, y, g = d["logits"], d["y"], d["graph"]; assert (y == pk["y"]).all()
    p = L.argmax(1); pr = torch.from_numpy(L).softmax(-1).numpy(); conf = pr.max(1); wrong = p != y
    print(f"faces {len(y)}, wrong {wrong.sum()} ({wrong.mean():.2%}), acc {1 - wrong.mean():.4f}, mIoU {miou(p, y, 8):.4f}")
    ytr = load_pack(F360, "train")["y"]
    cm = np.zeros((8, 8), int); np.add.at(cm, (y, p), 1)
    print("\nclass            train share  test n   recall  precision  IoU    top confusions (true -> pred)")
    for c in range(8):
        row = cm[c].copy(); row[c] = 0; top = np.argsort(-row)[:3]
        print(f"  {FC[c]:12s}  {np.mean(ytr == c):9.1%}  {cm[c].sum():6d}  {cm[c, c] / cm[c].sum():6.1%}  {cm[c, c] / max(cm[:, c].sum(), 1):8.1%}  "
              f"{cm[c, c] / (cm[c].sum() + cm[:, c].sum() - cm[c, c]):.3f}  " + ", ".join(f"{FC[t]} {row[t] / cm[c].sum():.1%}" for t in top))
    op = np.array([0, 0, 1, 1, 2, 3, 4, 4]); role = np.array([0, 1, 0, 1, 2, 2, 0, 1])   # operation, side/end role
    same_op = op[y] == op[p]; same_role = role[y] == role[p]; W = wrong.sum()
    print("\nerror types (share of all wrong faces)")
    kinds = {"right role, wrong operation (e.g. ExtrudeSide<->CutSide)": wrong & ~same_op & same_role,
             "right operation, wrong role (Side<->End)": wrong & same_op & ~same_role,
             "both wrong": wrong & ~same_op & ~same_role}
    for k, m in kinds.items():
        print(f"  {m.sum():6d} {m.sum() / W:6.1%}  {k}")
    ec = np.isin(y, [0, 1, 2, 3]) & np.isin(p, [0, 1, 2, 3]) & (op[y] != op[p])
    print(f"  of which Extrude<->Cut in either role: {ec.sum()} = {ec.sum() / W:.1%} of all errors")
    OUT["cm"] = cm.tolist(); OUT["err_kinds"] = {k: int(m.sum()) for k, m in kinds.items()}; OUT["extrude_cut"] = int(ec.sum())

    X, kind = face_features(pk); f = {k: X[:, i] for i, k in enumerate(FEATS)}
    surf = np.where(f["planar"] > 0, 0, np.where(f["single_curved"] > 0, 1, 2))
    OUT["by_surface"] = bucket_table("error by surface kind (from normals)", surf, [0, 1, 2, 3], ["planar", "single-curved", "double-curved"], wrong)
    OUT["by_model_size"] = bucket_table("error by faces in the model", f["model_faces"], [0, 7, 11, 21, 41, 81, 1e9], ["2-6", "7-10", "11-20", "21-40", "41-80", "81+"], wrong)
    OUT["by_degree"] = bucket_table("error by face degree (adjacent edges)", f["degree"], [0, 3, 5, 7, 11, 1e9], ["1-2", "3-4", "5-6", "7-10", "11+"], wrong)
    cc = np.where(f["n_concave"] == 0, 0, np.where(f["n_convex"] == 0, 2, 1))
    OUT["by_convexity"] = bucket_table("error by the face's edge convexity", cc, [0, 1, 2, 3], ["no concave", "mixed", "all concave"], wrong)

    n = len(pk["names"]); nf = np.bincount(g, minlength=n); nw = np.bincount(g, weights=wrong, minlength=n); acc = 1 - nw / nf
    order = np.argsort(-nw); cum = np.cumsum(nw[order]) / W
    print(f"\nper model: {np.mean(nw == 0):.1%} of {n} models fully correct; median model acc {np.median(acc):.3f}; "
          f"{np.mean(acc < 0.5):.1%} of models below 50%")
    print(f"  half of all wrong faces come from the worst {np.searchsorted(cum, 0.5) + 1} models ({(np.searchsorted(cum, 0.5) + 1) / n:.1%}); 80% from {(np.searchsorted(cum, 0.8) + 1) / n:.1%}")
    # models that become fully correct if Extrude and Cut are swapped everywhere in the prediction
    swap = np.array([2, 3, 0, 1, 4, 5, 6, 7])[p]; fix = np.bincount(g, weights=(swap != y), minlength=n) == 0
    merged = np.array([0, 1, 0, 1, 4, 5, 6, 7])
    print(f"  models wrong only by a global Extrude<->Cut swap: {(fix & (nw > 0)).sum()} models, {nw[fix].sum():.0f} wrong faces ({nw[fix].sum() / W:.1%} of errors)")
    print(f"  accuracy if Extrude and Cut are merged (6 classes): {np.mean(merged[p] == merged[y]):.4f}")
    src, dst = pk["ei"]
    nb_true = np.zeros(len(y), bool); np.logical_or.at(nb_true, dst, y[src] == p[dst])
    same_ok = np.zeros(len(y), bool); np.logical_or.at(same_ok, dst, (y[src] == y[dst]) & (p[src] == y[src]))
    print(f"  wrong faces whose prediction is the true class of an adjacent face: {nb_true[wrong].mean():.1%}; "
          f"with a correctly predicted same-class neighbour: {same_ok[wrong].mean():.1%}")
    print(f"  confidence: wrong mean {conf[wrong].mean():.3f}, right mean {conf[~wrong].mean():.3f}; wrong with conf > 0.9: {np.mean(conf[wrong] > 0.9):.1%}")
    OUT["per_model"] = dict(perfect=float(np.mean(nw == 0)), below50=float(np.mean(acc < 0.5)), merged_acc=float(np.mean(merged[p] == merged[y])))


# ---------------------------------------------------------------- 2. the two datasets side by side
def dataset_stats(root, names):
    pk = load_pack(root, "test"); ytr = load_pack(root, "train")["y"]; X, kind = face_features(pk)
    f = {k: X[:, i] for i, k in enumerate(FEATS)}; y = pk["y"]; src, dst = pk["ei"]; nf = np.diff(pk["fp"])
    share = np.bincount(ytr, minlength=len(names)) / len(ytr)
    ncls = np.array([len(set(y[a:b])) for a, b in zip(pk["fp"][:-1], pk["fp"][1:])])
    return collections.OrderedDict([
        ("classes", len(names)), ("train faces", len(ytr)), ("test models", len(nf)),
        ("faces per model: median", float(np.median(nf))), ("faces per model: p10 / p90", f"{np.percentile(nf, 10):.0f} / {np.percentile(nf, 90):.0f}"),
        ("faces per model: max", int(nf.max())),
        ("largest class (train share)", f"{names[share.argmax()]} {share.max():.1%}"), ("smallest class (train share)", f"{names[share.argmin()]} {share.min():.2%}"),
        ("class balance (normalised entropy)", float(-(share[share > 0] * np.log(share[share > 0])).sum() / np.log(len(names)))),
        ("classes present per model: median", float(np.median(ncls))),
        ("mean face degree", float(f["degree"].mean())),
        ("planar faces", float(f["planar"].mean())), ("single-curved faces", float(f["single_curved"].mean())), ("double-curved faces", float(f["double_curved"].mean())),
        ("edges: convex / concave / smooth", f"{np.mean(kind == 1):.1%} / {np.mean(kind == 2):.1%} / {np.mean(kind == 0):.1%}"),
        ("adjacent faces sharing a label", float(np.mean(y[src] == y[dst]))),
    ])


if "datasets" in todo:
    print("\n" + "=" * 30, "2. DATASETS")
    a, b = dataset_stats(F360, FC), dataset_stats(MFC, MC)
    print(f"{'':38s} {'Fusion 360':>28s} {'MFCAD++':>28s}")
    for k in a:
        fm = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{k:38s} {fm(a[k]):>28s} {fm(b[k]):>28s}")
    OUT["datasets"] = {"fusion360": a, "mfcadpp": b}


# ---------------------------------------------------------------- 3. is the label determined by the local shape?
def H(b):
    return hashlib.blake2b(b, digest_size=8).digest()


def signatures(pk, K, models):
    """Translation/scale-invariant k-hop signatures (normals, trim mask, edge tangents; rounded to 0.1)."""
    out = [[] for _ in range(K + 1)]
    for g in models:
        f0, f1, e0, e1 = pk["fp"][g], pk["fp"][g + 1], pk["ep"][g], pk["ep"][g + 1]
        x = np.round(pk["x"][f0:f1].float().numpy()[..., 3:], 1); ea = np.round(pk["ea"][e0:e1].float().numpy()[..., 3:], 1)
        ei = pk["ei"][:, e0:e1] - f0; n = f1 - f0
        h = [H(x[i].tobytes()) for i in range(n)]; eh = [H(ea[e].tobytes()) for e in range(e1 - e0)]
        out[0] += h
        for k in range(1, K + 1):
            nb = [[] for _ in range(n)]
            for e, (s, t) in enumerate(ei.T):
                nb[t].append(H(h[s] + eh[e]))
            h = [H(h[i] + b"".join(sorted(nb[i]))) for i in range(n)]
            out[k] += h
    return out


def ambiguity(root, name, K=3, n_train=10000):
    tr, te = load_pack(root, "train"), load_pack(root, "test")
    mt = np.random.RandomState(0).permutation(len(tr["names"]))[:n_train]; mt.sort()
    St = signatures(tr, K, mt); Se = signatures(te, K, range(len(te["names"])))
    ytr = np.concatenate([tr["y"][tr["fp"][g]:tr["fp"][g + 1]] for g in mt]); yte = te["y"]
    res = []
    for k in range(K + 1):
        tab = collections.defaultdict(collections.Counter)
        for s, l in zip(St[k], ytr):
            tab[s][int(l)] += 1
        seen = np.array([s in tab for s in Se[k]])
        # predict each test face from train faces with the same signature (majority label)
        maj = np.array([tab[s].most_common(1)[0][0] if s in tab else -1 for s in Se[k]])
        conflict = np.array([len(tab[s]) > 1 if s in tab else False for s in Se[k]])
        res.append(dict(k=k, seen=float(seen.mean()), lookup_acc=float((maj[seen] == yte[seen]).mean()) if seen.any() else None,
                        conflicted=float(conflict[seen].mean()) if seen.any() else None))
        print(f"  {name} k={k}: test faces whose signature also occurs in {len(mt)} train models: {seen.mean():6.2%} | "
              f"of those, same label as the train majority: {res[-1]['lookup_acc']:.2%} | train copies disagree among themselves: {res[-1]['conflicted']:.2%}")
    return res


if "ambiguity" in todo:
    print("\n" + "=" * 30, "3. SAME LOCAL SHAPE, SAME LABEL? (identical k-hop neighbourhoods, position and scale ignored)")
    OUT["ambiguity"] = {"fusion360": ambiguity(F360, "Fusion 360"), "mfcadpp": ambiguity(MFC, "MFCAD++   ")}


# ---------------------------------------------------------------- 4. do topological features add information?
def probe(Xtr, ytr, Xte, yte, seeds=3, epochs=60):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr); accs, ious = [], []
    for s in range(seeds):
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 8))
        opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            perm = torch.randperm(len(A))
            for i in range(0, len(A), 1024):
                j = perm[i:i + 1024]; opt.zero_grad(); torch.nn.functional.cross_entropy(net(A[j]), yt[j]).backward(); opt.step()
            sched.step()
        with torch.no_grad():
            p = net(B).argmax(1).numpy()
        accs.append((p == yte).mean()); ious.append(miou(p, yte, 8))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(ious))


if "probe" in todo:
    print("\n" + "=" * 30, "4. TOPOLOGY PROBE (small MLP fitted on val faces, scored on test faces; 3 seeds)")
    S = {}
    for sp in ("val", "test"):
        pk = load_pack(F360, sp); d = np.load(f"{RUN}/{sp}_best.npz"); assert (d["y"] == pk["y"]).all()
        X, _ = face_features(pk); X = np.log1p(X)
        L = d["logits"]
        S[sp] = dict(y=d["y"], logit=L, nb=np.hstack([nb_mean(pk, L), nb_mean(pk, nb_mean(pk, L))]), topo=X, topo_nb=nb_mean(pk, X))
    yv, yt = S["val"]["y"], S["test"]["y"]
    base = (S["test"]["logit"].argmax(1) == yt).mean()
    print(f"  UV-Net as trained:                                   acc {base:.4f}  mIoU {miou(S['test']['logit'].argmax(1), yt, 8):.4f}")
    conds = [("topological features only (no UV-Net)", ["topo", "topo_nb"]),
             ("UV-Net logits (re-fit, control)", ["logit"]),
             ("logits + neighbours' logits (1 and 2 hops)", ["logit", "nb"]),
             ("logits + topological features", ["logit", "topo", "topo_nb"]),
             ("logits + neighbours' logits + topological features", ["logit", "nb", "topo", "topo_nb"])]
    OUT["probe"] = {"uvnet": float(base)}
    for name, keys in conds:
        a, sd, io = probe(np.hstack([S["val"][k] for k in keys]), yv, np.hstack([S["test"][k] for k in keys]), yt)
        OUT["probe"][name] = [a, sd, io]
        print(f"  {name:52s} acc {a:.4f} (+-{sd:.4f}, {a - base:+.4f} vs UV-Net)  mIoU {io:.4f}")

json.dump(OUT, open(f"{RUN}/report_{'_'.join(todo)}.json", "w"), indent=1, default=str)
