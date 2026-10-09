"""
Convert the Fusion 360 Gallery segmentation dataset (s2.0.0) into PyTorch Geometric
face-adjacency graphs with UV-grid features, in the same format as process/step_to_pyg.py.

The dataset stores one label per B-rep face in `breps/seg/<name>.seg`, one line per face in
the order the faces appear in `breps/step/<name>.stp`. Labels are the modeling operation that
created the face (see datasets/fusion360.py for the 8 classes).

The official `train_test.json` only has train and test. As in UV-Net, 20 % of train is held
out for validation with the same shuffle as sklearn's `train_test_split(random_state=42)`.

Usage:
    python process/fusion360_to_pyg.py /path/to/s2.0.0 data/fusion360_processed --num_processes 8
"""
import argparse
import json
import math
import pathlib
from multiprocessing.pool import Pool

import numpy as np
import torch
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from occwl.solid import Solid

from step_to_pyg import build_graph, initializer


def read_step_with_seg(step_file, seg_file):
    """Return (occwl Solid, {face -> int label}) for a single-solid STEP file and its .seg file."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_file)) != IFSelect_RetDone:
        raise RuntimeError(f"Could not read {step_file}")
    reader.TransferRoots()
    solids = list(TopologyExplorer(reader.OneShape()).solids())
    if len(solids) != 1:
        raise RuntimeError(f"Expected 1 solid, found {len(solids)}")
    solid = Solid(solids[0])
    seg = np.loadtxt(seg_file, dtype=np.int64, ndmin=1)
    faces = list(solid.faces())
    if len(faces) != len(seg):
        raise RuntimeError(f"{len(faces)} faces but {len(seg)} labels")
    return solid, {face: int(label) for face, label in zip(faces, seg)}


def process_one_file(arguments):
    stem, args = arguments
    root = pathlib.Path(args.input)
    out = pathlib.Path(args.output) / "graphs" / (stem + ".pt")
    if out.exists():
        return stem, None
    try:
        solid, labels = read_step_with_seg(root / "breps" / "step" / (stem + ".stp"), seg_dir(root) / (stem + ".seg"))
        data = build_graph(solid, labels, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples)
        torch.save(data, out)
        return stem, None
    except Exception as e:  # noqa: BLE001 - log and skip bad models
        return stem, f"{type(e).__name__}: {e}"


def seg_dir(root):
    """s2.0.0 keeps the labels in breps/seg; s1.0.0 kept them next to the B-reps."""
    path = root / "breps" / "seg"
    return path if path.exists() else root / "breps"


def train_val_split(names, val_fraction=0.2, seed=42):
    """Same split as sklearn.model_selection.train_test_split(names, test_size=0.2, random_state=42)."""
    n_val = math.ceil(val_fraction * len(names))
    perm = np.random.RandomState(seed).permutation(len(names))
    return [names[i] for i in perm[n_val:]], [names[i] for i in perm[:n_val]]


def main():
    parser = argparse.ArgumentParser("Convert Fusion 360 Gallery segmentation STEP files to PyG UV-grid graphs")
    parser.add_argument("input", type=str, help="Dataset root (contains breps/step, breps/seg and train_test.json)")
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

    filelist = json.loads((input_path / "train_test.json").read_text())
    train, val = train_val_split(filelist["train"])
    splits = {"train": train, "val": val, "test": filelist["test"]}
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    stems = sorted(s for v in splits.values() for s in v)
    print({k: len(v) for k, v in splits.items()})

    failures = {}
    with Pool(processes=args.num_processes, initializer=initializer) as pool:
        it = pool.imap_unordered(process_one_file, [(s, args) for s in stems], chunksize=8)
        for stem, err in tqdm(it, total=len(stems), mininterval=30):
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
