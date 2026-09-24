"""Portable experiment infrastructure; the tracking algorithm is unchanged."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .data import (
    choose_sequences, infer_dataset_name, load_gt, normalize_dataset_root,
    read_image, resolve_sequence_paths,
)
from .metrics import frame_statistics, summarize_frames

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_config(path):
    default = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    override = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(override) - set(default)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    default.update(override)
    for key in ("seed", "short_capacity", "long_capacity", "memory_supports",
                "entry_streak", "exit_patience"):
        if type(default[key]) is not int or default[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")
    if min(default["short_capacity"], default["long_capacity"]) < 2:
        raise ValueError("Memory capacities must be at least two")
    for key in ("disable_camera", "disable_redetection", "fixed_measurement_noise"):
        if type(default[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    for key in ("long_weight", "disagreement_penalty", "entry_evidence"):
        if not isinstance(default[key], (int, float)) or not np.isfinite(default[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= default["long_weight"] <= 1 or default["disagreement_penalty"] < 0:
        raise ValueError("Invalid memory weighting")
    return default


def inspect_dataset(root, sequences=None, alignment="strict", max_frames=0, decode=False):
    root = normalize_dataset_root(Path(root))
    names = choose_sequences(root, sequences, 0)
    if not names:
        raise ValueError(f"No supported sequences found under {root}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sequence names")
    inventory = []
    for name in names:
        paths, gt_path = resolve_sequence_paths(root, name)
        gt = load_gt(gt_path)
        if alignment == "strict" and len(paths) != len(gt):
            raise ValueError(f"{name}: {len(paths)} images != {len(gt)} annotations; "
                             "fix the dataset or explicitly use --alignment common")
        length = min(len(paths), len(gt))
        if max_frames > 0:
            length = min(length, max_frames)
        if not np.isfinite(gt[0]).all() or np.any(gt[0, 2:] <= 0):
            raise ValueError(f"{name}: initialization box must be finite and positive")
        if decode:
            for path in paths[:length]:
                read_image(path)
        valid = np.isfinite(gt[:length]).all(axis=1) & (gt[:length, 2:] > 0).all(axis=1)
        # Names and sizes detect accidental missing/replaced files without hashing
        # every multi-gigabyte image collection. This is not a byte-integrity proof.
        image_signature = hashlib.sha256("\n".join(
            f"{path.name}\t{path.stat().st_size}" for path in paths[:length]
        ).encode()).hexdigest()
        inventory.append({
            "sequence": name, "images": len(paths), "annotations": len(gt),
            "frames": length, "evaluated_frames": int(valid.sum()),
            "absent_frames": int(length - valid.sum()),
            "gt_sha256": sha256(gt_path), "image_names_sizes_sha256": image_signature,
        })
    return root, inventory


def read_predictions(path, expected_length):
    # Released prediction files have a header; external Nx4 files need not.
    path = Path(path)
    first = path.read_text(encoding="utf-8-sig").splitlines()[0].strip().lower()
    names = [name.strip() for name in first.split(",")]
    named_boxes = names[:4] in (["x", "y", "w", "h"], ["x", "y", "width", "height"])
    header = named_boxes or first.startswith("#")
    boxes = np.loadtxt(path, delimiter=",", skiprows=int(header), ndmin=2,
                       encoding="utf-8-sig")
    if named_boxes:
        boxes = boxes[:, :4]  # explicitly named trailing diagnostic columns are not boxes
    if boxes.shape != (expected_length, 4):
        raise ValueError(f"{path}: expected ({expected_length}, 4), got {boxes.shape}")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < 0):
        raise ValueError(f"{path}: nonfinite coordinates or negative box sizes")
    return boxes


def curve_data(stats):
    grids = {
        "success": ("iou", np.linspace(0, 1, 21), True),
        "precision": ("center_error", np.arange(0, 51), False),
        "normalized_precision": ("normalized_error", np.linspace(0, .5, 51), False),
    }
    result = {}
    for name, (key, grid, greater) in grids.items():
        values = [(stats[key] >= t).mean() if greater else
                  (stats[key] <= t).mean() for t in grid]
        result[name] = {"thresholds": grid.tolist(), "values": [float(x) for x in values]}
    return result


def score_predictions(root, inventory, predictions_dir):
    pooled = {key: [] for key in ("iou", "center_error", "normalized_error")}
    rows = []
    for item in inventory:
        name, length = item["sequence"], item["frames"]
        boxes = read_predictions(Path(predictions_dir) / f"{name}.csv", length)
        _, gt_path = resolve_sequence_paths(root, name)
        stats = frame_statistics(boxes, load_gt(gt_path)[:length])
        rows.append({key: item[key] for key in
                     ("sequence", "frames", "evaluated_frames", "absent_frames")} |
                    summarize_frames(stats))
        for key in pooled:
            pooled[key].append(stats[key])
    pooled = {key: np.concatenate(values) for key, values in pooled.items()}
    aggregate = summarize_frames(pooled)
    aggregate.update({
        "sequences": len(rows), "frames": sum(x["frames"] for x in rows),
        "evaluated_frames": len(pooled["iou"]),
        "absent_frames": sum(x["absent_frames"] for x in rows),
    })
    return {"dataset": infer_dataset_name(root), "protocol": "pooled-OPE-v1",
            "aggregate": aggregate, "per_sequence": rows, "curves": curve_data(pooled)}


def source_fingerprint(repo):
    digest = hashlib.sha256()
    for parent in (ROOT / "rlsm_immtrack", Path(repo) / "lib", Path(repo) / "experiments"):
        for path in sorted(parent.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                digest.update(str(path.relative_to(parent)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_output(output, manifest, resume):
    output = Path(output).resolve()
    existing = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"{output} is not empty. Use a new output or --resume.")
        if not existing.is_file() or json.loads(existing.read_text()) != manifest:
            raise ValueError("Resume refused: configuration, code, checkpoint, dataset, "
                             "frame limit, or runtime differs from the original run.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(existing, manifest)
    for name in ("predictions", "diagnostics", "completed"):
        (output / name).mkdir(exist_ok=True)
    return output


def evaluate(args):
    import torch
    from .safecl import ClosedLoopTracker
    from .tracker import RiskLimitedShadowTracker
    from .sutrack import configure_sutrack

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inference is required. Use 'score' for CPU-only evaluation.")
    torch.cuda.set_device(args.device)
    config = load_config(args.config)
    root, inventory = inspect_dataset(args.dataset_root, args.sequences,
                                     args.alignment, args.max_frames)
    checkpoint, repo = args.checkpoint.resolve(), args.sutrack_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = {
        "release": __version__, "tracker": args.tracker, "config": config,
        "checkpoint_sha256": sha256(checkpoint), "source_sha256": source_fingerprint(repo),
        "dataset_root": str(root), "alignment": args.alignment,
        "max_frames": args.max_frames, "sequences": inventory,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(args.device)},
    }
    output = prepare_output(args.output, manifest, args.resume)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = True  # same setting as the archived experiments
    base = None
    for index, item in enumerate(inventory, 1):
        name, length = item["sequence"], item["frames"]
        pred_path = output / "predictions" / f"{name}.csv"
        diag_path = output / "diagnostics" / f"{name}.csv"
        done_path = output / "completed" / f"{name}.json"
        print(f"[{index}/{len(inventory)}] {name}: {length} frames", flush=True)
        if args.resume and done_path.is_file():
            done = json.loads(done_path.read_text())
            if (pred_path.is_file() and diag_path.is_file()
                    and done.get("prediction_sha256") == sha256(pred_path)
                    and done.get("diagnostics_sha256") == sha256(diag_path)):
                read_predictions(pred_path, length)
                print("  verified completed sequence; skipped", flush=True)
                continue
            raise ValueError(f"{name}: completed outputs have changed; use a new output directory")
        if base is None:
            base = configure_sutrack(repo, checkpoint)
        paths, gt_path = resolve_sequence_paths(root, name)
        initial_box = load_gt(gt_path)[0].copy()  # no future ground truth enters tracking
        first = read_image(paths[0])
        torch.cuda.synchronize()
        start = time.perf_counter()
        base.initialize(first, {"init_bbox": initial_box.tolist()})
        if args.tracker == "sutrack":
            tracker = base
        else:
            common = {key: config[key] for key in
                      ("disable_camera", "disable_redetection", "fixed_measurement_noise")}
            if args.tracker == "safecl":
                tracker = ClosedLoopTracker(base, first, initial_box, **common)
            else:
                extras = {key: value for key, value in config.items()
                          if key not in ("seed", *common)}
                tracker = RiskLimitedShadowTracker(base, first, initial_box, **common, **extras)
        torch.cuda.synchronize()
        init_seconds = time.perf_counter() - start
        predictions = [initial_box]
        diagnostics = [{"frame": 0, "initialized": True, "wall_seconds": init_seconds}]
        track_seconds = 0.0
        for frame, path in enumerate(paths[1:length], 1):
            image = read_image(path)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if args.tracker == "sutrack":
                box = tracker.track(image)["target_bbox"]
                row = {}
            else:
                box, row = tracker.track(image)
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
            track_seconds += wall
            predictions.append(np.asarray(box, dtype=np.float64).copy())
            diagnostics.append(dict(row, frame=frame, wall_seconds=wall))
            if frame % 100 == 0 or frame + 1 == length:
                print(f"  {frame + 1}/{length}", flush=True)
        boxes = np.asarray(predictions)
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < 0):
            raise ValueError(f"{name}: invalid tracker output")
        temp = pred_path.with_suffix(".csv.tmp")
        np.savetxt(temp, boxes, delimiter=",", header="x,y,w,h", comments="", fmt="%.10f")
        temp.replace(pred_path)
        write_csv(diag_path, diagnostics)
        atomic_json(done_path, {
            "sequence": name, "frames": length,
            "prediction_sha256": sha256(pred_path), "diagnostics_sha256": sha256(diag_path),
            "init_seconds": init_seconds, "track_seconds": track_seconds,
            "legacy_probe_seconds": sum(float(row.get("camera_seconds", 0)) +
                                        float(row.get("visual_seconds", 0)) for row in diagnostics),
        })
    result = score_predictions(root, inventory, output / "predictions")
    done = [json.loads((output / "completed" / f"{x['sequence']}.json").read_text())
            for x in inventory]
    seconds = sum(x["track_seconds"] for x in done)
    result.update({"tracker": args.tracker, "config": config,
                   "partial_run": bool(args.max_frames or args.sequences)})
    result["timing"] = {
        "track_seconds": seconds, "initialization_seconds": sum(x["init_seconds"] for x in done),
        "fps_wall_excluding_init_io": (sum(x["frames"] - 1 for x in done) / seconds
                                      if seconds > 0 else None),
        "definition": "CUDA-synchronized whole track() calls; excludes initialization, model loading and image I/O",
    }
    atomic_json(output / "metrics.json", result)
    write_csv(output / "per_sequence.csv", result["per_sequence"])
    print(json.dumps(result["aggregate"], indent=2))


def plot_metrics(paths, labels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if labels and len(labels) != len(paths):
        raise ValueError("One label is required for each metrics file")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    def signature(report):
        return (report["dataset"], sorted((row["sequence"], row["frames"], row["evaluated_frames"])
                for row in report["per_sequence"]))
    if any(signature(report) != signature(reports[0]) for report in reports[1:]):
        raise ValueError("Comparison plots require the same dataset, sequences and frame counts")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for i, path in enumerate(paths):
        report = reports[i]
        label = labels[i] if labels else report.get("tracker", Path(path).parent.name)
        for ax, key in zip(axes, ("success", "precision", "normalized_precision")):
            curve = report["curves"][key]
            ax.plot(curve["thresholds"], curve["values"], label=label, linewidth=1.6)
    for ax, title, xlabel in zip(
        axes, ("Success", "Center precision", "Normalized precision"),
        ("IoU threshold", "Center error (pixels)", "Normalized center error")
    ):
        ax.set(title=title, xlabel=xlabel, ylabel="Fraction of valid frames", ylim=(0, 1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
