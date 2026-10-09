"""Fusion 360 Gallery segmentation dataset of preprocessed PyG graphs (see process/fusion360_to_pyg.py)."""
from datasets.mfcadpp import MFCADPPDataset

# Modeling operation that created each face (Lambourne et al., BRepNet, 2021), in the order of
# the dataset's segment_names.json.
FUSION360_CLASSES = [
    "ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd",
]


class Fusion360Dataset(MFCADPPDataset):
    """Same on-disk format and loading as MFCAD++; only the label set differs."""

    num_classes = len(FUSION360_CLASSES)
