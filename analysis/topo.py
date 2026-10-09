"""Topological / derived per-face features computed from the packed UV-grid caches (no STEP files needed).

None of these are given to UV-Net explicitly: it only sees raw UV-grids on faces and U-grids on edges.
Edge convexity is an approximation from the sampled grids (see edge_convexity).
"""
import numpy as np
import torch

FEATS = ["degree", "distinct_nb", "n_convex", "n_concave", "n_smooth", "n_closed_edge", "n_multi_edge",
         "planar", "single_curved", "double_curved", "mask_frac", "extent", "area", "model_faces"]


def load_pack(root, split):
    p = torch.load(f"{root}/cache/{split}.pt", weights_only=False, mmap=True)
    n = len(p["names"])
    fp, ep = p["face_ptr"].numpy()[: n + 1], p["edge_ptr"].numpy()[: n + 1]
    F, E = int(fp[-1]), int(ep[-1])
    ei = p["edge_index"][:, :E].numpy() + np.repeat(fp[:-1], np.diff(ep))[None]   # global face ids
    return dict(names=p["names"], x=p["x"], ea=p["edge_attr"], ei=ei, y=p["y"][:F].numpy(), fp=fp, ep=ep, F=F, E=E,
                graph=np.repeat(np.arange(n), np.diff(fp)))


def face_geometry(pk, chunk=50000):
    """planar / single / double curved flags, mask fraction, extent and area proxy per face."""
    F = pk["F"]; out = {k: np.zeros(F, np.float32) for k in ("planar", "single_curved", "double_curved", "mask_frac", "extent", "area")}
    for s in range(0, F, chunk):
        x = pk["x"][s:s + chunk].float().numpy(); e = slice(s, s + len(x))
        m = x[..., 6] > 0.5; w = m[..., None]; cnt = np.maximum(m.sum((1, 2)), 1)[:, None]
        n = x[..., 3:6]
        nm = (n * w).sum((1, 2)) / cnt
        dev = np.sqrt((((n - nm[:, None, None]) ** 2).sum(-1) * m).sum((1, 2)) / cnt[:, 0])    # spread of normals
        # variation of the normal along u and along v (only between visible neighbours)
        du = (np.linalg.norm(n[:, 1:] - n[:, :-1], axis=-1) * (m[:, 1:] & m[:, :-1])).sum((1, 2)) / np.maximum((m[:, 1:] & m[:, :-1]).sum((1, 2)), 1)
        dv = (np.linalg.norm(n[:, :, 1:] - n[:, :, :-1], axis=-1) * (m[:, :, 1:] & m[:, :, :-1])).sum((1, 2)) / np.maximum((m[:, :, 1:] & m[:, :, :-1]).sum((1, 2)), 1)
        planar = dev < 0.03
        lo, hi = np.minimum(du, dv), np.maximum(du, dv)
        single = ~planar & (lo < 0.25 * hi + 1e-3)
        out["planar"][e] = planar; out["single_curved"][e] = single; out["double_curved"][e] = ~planar & ~single
        out["mask_frac"][e] = m.mean((1, 2))
        p = x[..., :3]; big = np.where(w, p, np.nan)
        with np.errstate(all="ignore"):
            ext = np.nanmax(big, (1, 2)) - np.nanmin(big, (1, 2))
        out["extent"][e] = np.nan_to_num(np.linalg.norm(ext, axis=-1))
        cell = np.linalg.norm(np.cross(p[:, 1:, :-1] - p[:, :-1, :-1], p[:, :-1, 1:] - p[:, :-1, :-1]), axis=-1)
        out["area"][e] = (cell * (m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:])).sum((1, 2))
    return out


def edge_convexity(pk, chunk=40000):
    """Per directed edge a->b: 0 smooth (normals within ~10 deg), 1 convex, 2 concave; plus closed-edge flag.

    Approximation: n_a / n_b are the face normals at the visible grid point nearest the edge midpoint, d_b is the
    direction from the midpoint into face b (mean of its 8 nearest visible grid points). With outward normals,
    face b drops behind face a's tangent plane at a convex edge (d_b . n_a < 0) and rises above it at a concave one.
    """
    E = pk["E"]; kind = np.zeros(E, np.int8); closed = np.zeros(E, bool)
    for s in range(0, E, chunk):
        src, dst = pk["ei"][0, s:s + chunk], pk["ei"][1, s:s + chunk]
        ea = pk["ea"][s:s + chunk].float().numpy()
        mid = 0.5 * (ea[:, 4, :3] + ea[:, 5, :3])
        closed[s:s + len(src)] = np.linalg.norm(ea[:, 0, :3] - ea[:, -1, :3], axis=-1) < 1e-3
        xs = pk["x"]; res = []
        for f in (src, dst):
            order = np.argsort(f, kind="stable"); g = np.empty((len(f), 100, 7), np.float32)
            g[order] = xs[torch.from_numpy(f[order])].float().numpy().reshape(len(f), 100, 7)
            d = np.linalg.norm(g[..., :3] - mid[:, None], axis=-1) + 1e3 * (g[..., 6] < 0.5)
            idx = np.argsort(d, axis=1)[:, :8]
            near = np.take_along_axis(g, idx[..., None], 1)
            res.append((near[:, 0, 3:6], near[:, :, :3].mean(1) - mid))
        (na, da), (nb, db) = res
        cos = (na * nb).sum(-1); sgn = (db * na).sum(-1) + (da * nb).sum(-1)
        kind[s:s + len(src)] = np.where(cos > 0.985, 0, np.where(sgn < 0, 1, 2))
    return kind, closed


def face_features(pk):
    """F x len(FEATS) matrix plus the per-edge convexity kinds."""
    F = pk["F"]; src, dst = pk["ei"]
    geo = face_geometry(pk); kind, closed = edge_convexity(pk)
    cnt = lambda w=None: np.bincount(dst, weights=w, minlength=F).astype(np.float32)
    pair = np.unique(np.stack([dst, src]), axis=1)
    distinct = np.bincount(pair[0], minlength=F).astype(np.float32)
    deg = cnt()
    cols = dict(degree=deg, distinct_nb=distinct, n_convex=cnt(kind == 1), n_concave=cnt(kind == 2), n_smooth=cnt(kind == 0),
                n_closed_edge=cnt(closed), n_multi_edge=deg - distinct, model_faces=np.diff(pk["fp"])[pk["graph"]].astype(np.float32), **geo)
    return np.stack([cols[k] for k in FEATS], 1), kind


def nb_mean(pk, V):
    """Mean of V over each face's neighbours (incoming edges)."""
    src, dst = pk["ei"]; out = np.zeros_like(V, dtype=np.float32)
    np.add.at(out, dst, V[src])
    return out / np.maximum(np.bincount(dst, minlength=len(V)), 1)[:, None]
