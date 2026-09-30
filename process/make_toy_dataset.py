"""
Generate a tiny synthetic MFCAD++-style dataset for smoke-testing the pipeline.

Each model is a box stock with random through holes (label 1), blind holes (label 12)
and rectangular pockets (label 14); remaining faces are stock (label 24). Labels are
written to the ADVANCED_FACE names in the STEP file, exactly like MFCAD++.

    python process/make_toy_dataset.py data/toy_step --num 120
"""
import argparse
import pathlib
import random

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.STEPConstruct import stepconstruct
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.Interface import Interface_Static
from OCC.Extend.TopologyUtils import TopologyExplorer

THROUGH_HOLE, BLIND_HOLE, RECT_POCKET, STOCK = 1, 12, 14, 24


def make_model(rng):
    L, W, H = rng.uniform(60, 100), rng.uniform(40, 80), rng.uniform(20, 40)
    shape = BRepPrimAPI_MakeBox(L, W, H).Shape()
    blind_depth = []
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["through", "blind", "pocket"])
        x, y = rng.uniform(12, L - 12), rng.uniform(12, W - 12)
        if kind == "pocket":
            a, b, d = rng.uniform(6, 14), rng.uniform(6, 14), rng.uniform(4, H * 0.6)
            tool = BRepPrimAPI_MakeBox(gp_Pnt(x - a / 2, y - b / 2, H - d), a, b, d + 1).Shape()
        else:
            r = rng.uniform(2, 5)
            z0, h = (-1, H + 2) if kind == "through" else (H - rng.uniform(4, H * 0.6), H)
            tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z0), gp_Dir(0, 0, 1)), r, h + 1).Shape()
            if kind == "blind":
                blind_depth.append(z0)
        cut = BRepAlgoAPI_Cut(shape, tool)
        cut.Build()
        if cut.IsDone():
            shape = cut.Shape()
    return shape, (L, W, H), blind_depth


def label_face(face, dims, blind_z):
    L, W, H = dims
    s = BRepAdaptor_Surface(face)
    tol = 1e-4
    if s.GetType() == GeomAbs_Cylinder:
        # Blind holes are the ones whose bottom sits inside the stock
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        z_lo = min(s.Value(u0, v0).Z(), s.Value(u0, v1).Z())
        return BLIND_HOLE if z_lo > tol else THROUGH_HOLE
    if s.GetType() == GeomAbs_Plane:
        pln = s.Plane()
        n, p = pln.Axis().Direction(), pln.Location()
        on_boundary = (
            (abs(abs(n.X()) - 1) < tol and (abs(p.X()) < tol or abs(p.X() - L) < tol))
            or (abs(abs(n.Y()) - 1) < tol and (abs(p.Y()) < tol or abs(p.Y() - W) < tol))
            or (abs(abs(n.Z()) - 1) < tol and (abs(p.Z()) < tol or abs(p.Z() - H) < tol))
        )
        if on_boundary:
            return STOCK
        if any(abs(p.Z() - z) < tol for z in blind_z) and abs(abs(n.Z()) - 1) < tol:
            return BLIND_HOLE  # hole floor
        return RECT_POCKET
    return STOCK


def write_labeled_step(shape, labels, filename):
    Interface_Static.SetCVal("write.step.schema", "AP203")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    fp = writer.WS().TransferWriter().FinderProcess()
    for face, label in labels:
        item = stepconstruct.FindEntity(fp, face)
        if item is not None:
            item.SetName(TCollection_HAsciiString(str(label)))
    writer.Write(str(filename))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--num", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = pathlib.Path(args.output)
    for i in range(args.num):
        split = "train" if i < args.num * 0.7 else ("val" if i < args.num * 0.85 else "test")
        (out / split).mkdir(parents=True, exist_ok=True)
        shape, dims, blind_z = make_model(rng)
        labels = [(f, label_face(f, dims, blind_z)) for f in TopologyExplorer(shape).faces()]
        write_labeled_step(shape, labels, out / split / f"toy_{i:04d}.step")
    print(f"Wrote {args.num} STEP files to {out}")


if __name__ == "__main__":
    main()
