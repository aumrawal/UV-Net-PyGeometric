"""
Convert MFCAD++ STEP files into PyTorch Geometric face-adjacency graphs with UV-grid features.

MFCAD++ stores the machining-feature class of every B-rep face as the name of its
ADVANCED_FACE entity in the STEP file, so labels are read straight from the STEP
transfer map. One `<stem>.pt` file (a torch_geometric.data.Data) is written per model:

    x          : F x U x V x 7  float16  face UV-grids (xyz, normal, trim mask), centered & scaled
    edge_attr  : E x U x 6      float16  edge U-grids (xyz, tangent), same normalization
    edge_index : 2 x E          int64    directed face-adjacency (both directions per B-rep edge)
    y          : F              int64    per-face machining feature label

A `split.json` ({"train": [...], "val": [...], "test": [...]}) is written next to the graphs.

Usage:
    python process/step_to_pyg.py /path/to/MFCAD++_dataset data/processed --num_processes 8
"""
import argparse
import json
import pathlib
import signal
from multiprocessing.pool import Pool

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Extend.TopologyUtils import TopologyExplorer
from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid


def read_step_with_labels(filename):
    """Return (occwl Solid, {face TShape hash -> int label}) for a single-solid STEP file."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(filename)) != IFSelect_RetDone:
        raise RuntimeError(f"Could not read {filename}")
    reader.TransferRoots()
    shape = reader.OneShape()
    treader = reader.WS().TransferReader()

    topo = TopologyExplorer(shape)
    solids = list(topo.solids())
    if not solids and topo.number_of_shells() == 1:
        # ~0.2% of MFCAD++ files store a closed shell without a solid wrapper
        solids = [BRepBuilderAPI_MakeSolid(next(iter(topo.shells()))).Solid()]
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 solid, found {len(solids)}")
    solid = Solid(solids[0])

    labels = {}
    for face in solid.faces():
        item = treader.EntityFromShapeResult(face.topods_shape(), 1)
        if item is None:
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name and name.strip().lstrip("-").isdigit():
            labels[face] = int(name)
    return solid, labels


def center_and_scale(face_feat, edge_feat):
    """Same normalization as UV-Net: fit the visible face points into [-1, 1]^3."""
    mask = face_feat[..., 6].reshape(-1) == 1
    pts = face_feat[..., :3].reshape(-1, 3)[mask]
    if len(pts) == 0:
        pts = face_feat[..., :3].reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    scale = 2.0 / max((hi - lo).max(), 1e-8)
    center = 0.5 * (lo + hi)
    face_feat[..., :3] = (face_feat[..., :3] - center) * scale
    edge_feat[..., :3] = (edge_feat[..., :3] - center) * scale
    return face_feat, edge_feat


def build_graph(solid, labels, curv_u, surf_u, surf_v):
    graph = face_adjacency(solid)
    if graph is None:
        raise RuntimeError("Non-manifold or open shape")

    # Node order -> contiguous 0..F-1
    node_ids = sorted(graph.nodes)
    remap = {n: i for i, n in enumerate(node_ids)}

    face_feat, y = [], []
    for n in node_ids:
        face = graph.nodes[n]["face"]
        points = uvgrid(face, method="point", num_u=surf_u, num_v=surf_v)
        normals = uvgrid(face, method="normal", num_u=surf_u, num_v=surf_v)
        vis = uvgrid(face, method="visibility_status", num_u=surf_u, num_v=surf_v)
        mask = np.logical_or(vis == 0, vis == 2)  # 0: inside, 1: outside, 2: on boundary
        face_feat.append(np.concatenate((points, normals, mask), axis=-1))
        if face not in labels:
            raise RuntimeError(f"Face {n} has no label in STEP file")
        y.append(labels[face])
    face_feat = np.asarray(face_feat, dtype=np.float32)

    src, dst, edge_feat = [], [], []
    for (u, v), attrs in graph.edges.items():
        edge = attrs["edge"]
        points = ugrid(edge, method="point", num_u=curv_u)
        tangents = ugrid(edge, method="tangent", num_u=curv_u)
        edge_feat.append(np.concatenate((points, tangents), axis=-1))
        src.append(remap[u])
        dst.append(remap[v])
    if not edge_feat:
        raise RuntimeError("Graph has no edges")
    edge_feat = np.asarray(edge_feat, dtype=np.float32)

    if not (np.isfinite(face_feat).all() and np.isfinite(edge_feat).all()):
        raise RuntimeError("Non-finite UV-grid values")
    face_feat, edge_feat = center_and_scale(face_feat, edge_feat)

    return Data(
        x=torch.from_numpy(face_feat).half(),
        edge_attr=torch.from_numpy(edge_feat).half(),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=len(node_ids),
    )


def process_one_file(arguments):
    fn, args = arguments
    out = pathlib.Path(args.output) / "graphs" / (fn.stem + ".pt")
    if out.exists():
        return fn.stem, None
    try:
        solid, labels = read_step_with_labels(fn)
        data = build_graph(solid, labels, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples)
        torch.save(data, out)
        return fn.stem, None
    except Exception as e:  # noqa: BLE001 - log and skip bad models
        return fn.stem, f"{type(e).__name__}: {e}"


def find_splits(input_path, step_files):
    """
    Resolve train/val/test membership. Prefers split text files (train.txt, val.txt, test.txt)
    anywhere under the dataset root; otherwise uses the parent folder name of each STEP file.
    """
    stems = {f.stem for f in step_files}
    splits = {}
    for split in ("train", "val", "test"):
        txts = [p for p in input_path.rglob(f"{split}.txt")]
        if txts:
            names = [pathlib.Path(l.strip()).stem for l in txts[0].read_text().splitlines() if l.strip()]
            splits[split] = [n for n in names if n in stems]
    if all(splits.get(s) for s in ("train", "val", "test")):
        return splits
    splits = {"train": [], "val": [], "test": []}
    aliases = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}
    for f in step_files:
        for part in reversed(f.parts[:-1]):
            if part.lower() in aliases:
                splits[aliases[part.lower()]].append(f.stem)
                break
    return splits


def initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    parser = argparse.ArgumentParser("Convert MFCAD++ STEP files to PyG UV-grid graphs")
    parser.add_argument("input", type=str, help="MFCAD++ dataset root (searched recursively for *.step)")
    parser.add_argument("output", type=str, help="Output folder (graphs/ and split.json)")
    parser.add_argument("--curv_u_samples", type=int, default=10)
    parser.add_argument("--surf_u_samples", type=int, default=10)
    parser.add_argument("--surf_v_samples", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files per split (prototyping)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    (output_path / "graphs").mkdir(parents=True, exist_ok=True)

    step_files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in (".step", ".stp"))
    print(f"Found {len(step_files)} STEP files")
    splits = find_splits(input_path, step_files)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    wanted = {s for v in splits.values() for s in v}
    step_files = [f for f in step_files if f.stem in wanted]
    print({k: len(v) for k, v in splits.items()})

    failures = {}
    with Pool(processes=args.num_processes, initializer=initializer) as pool:
        it = pool.imap_unordered(process_one_file, [(f, args) for f in step_files], chunksize=8)
        for stem, err in tqdm(it, total=len(step_files)):
            if err:
                failures[stem] = err

    # Only keep successfully processed models in the split
    done = {p.stem for p in (output_path / "graphs").glob("*.pt")}
    splits = {k: [s for s in v if s in done] for k, v in splits.items()}
    (output_path / "split.json").write_text(json.dumps(splits))
    if failures:
        (output_path / "failures.json").write_text(json.dumps(failures, indent=1))
    print(f"Processed: { {k: len(v) for k, v in splits.items()} }, failed: {len(failures)}")


if __name__ == "__main__":
    main()
