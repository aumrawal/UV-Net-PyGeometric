"""
Build the two Kaggle script kernels that run this project on the Fusion 360 Gallery segmentation
dataset. A script kernel is a single file, so the project source is embedded in it as a tarball.

    python kaggle/build_kernels.py --user <kaggle username>
    kaggle kernels push -p kaggle/build/preprocess      # CPU, ~1 h; wait for it to complete
    kaggle kernels push -p kaggle/build/train           # GPU

`--smoke` builds a second, small pair (a few thousand models, 3 epochs) under separate slugs to
check the kernels end to end before spending GPU quota on the full run.
"""
import argparse
import base64
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ["train.py", "uvnet_pyg", "datasets", "process/step_to_pyg.py", "process/fusion360_to_pyg.py"]


def source_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src in SOURCES:
            tar.add(ROOT / src, arcname=src, filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def write_kernel(folder, template, values, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    code = (ROOT / "kaggle" / template).read_text()
    for key, value in values.items():
        code = code.replace(f"__{key}__", str(value))
    (folder / "script.py").write_text(code)
    metadata = {"code_file": "script.py", "language": "python", "kernel_type": "script", "is_private": True,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "model_sources": [],
                **metadata}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))
    print(f"{folder.relative_to(ROOT)}: {metadata['id']}")


def main():
    p = argparse.ArgumentParser("Build the Kaggle kernels for the Fusion 360 Gallery run")
    p.add_argument("--user", required=True, help="Kaggle username that will own the kernels")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--smoke", action="store_true", help="Small end-to-end check: 2000 models per split, 3 epochs")
    p.add_argument("--tag", default="", help="Appended to the kernel slugs, e.g. 'attr' for a re-run that must not "
                   "replace the output of the existing kernels")
    args = p.parse_args()

    suffix, title = ("-smoke", " (smoke)") if args.smoke else ("", "")
    if args.tag:
        suffix, title = f"{suffix}-{args.tag}", f"{title} ({args.tag})"
    prep_id = f"{args.user}/euclid-uvnet-fusion360-preprocess{suffix}"
    build = ROOT / "kaggle" / (("build_smoke" if args.smoke else "build") + (f"_{args.tag}" if args.tag else ""))
    source = source_tarball()
    write_kernel(
        build / "preprocess", "preprocess_kernel.py",
        {"SOURCE_B64": source, "LIMIT": 2000 if args.smoke else None},
        {"id": prep_id, "title": f"Euclid UVNet Fusion360 preprocess{title}", "enable_gpu": False,
         "kernel_sources": []},
    )
    write_kernel(
        build / "train", "train_kernel.py",
        {"SOURCE_B64": source, "EPOCHS": 3 if args.smoke else args.epochs,
         "EXPERIMENT": "fusion360_smoke" if args.smoke else "fusion360_full"},
        {"id": f"{args.user}/euclid-uvnet-fusion360-train{suffix}", "title": f"Euclid UVNet Fusion360 train{title}",
         "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "kernel_sources": [prep_id]},
    )


if __name__ == "__main__":
    main()
