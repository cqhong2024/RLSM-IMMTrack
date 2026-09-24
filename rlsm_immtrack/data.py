"""Portable adapters for UAV2UAV and A2A sequence layouts."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

def natural_sort_key(value: str):
    """Sort video2 before video10 while remaining deterministic."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def _looks_like_dataset_root(path: Path) -> bool:
    if (path / "img").is_dir() and (path / "anno").is_dir():
        return True
    return any(
        child.is_dir()
        and (child / "img").is_dir()
        and any((child / filename).is_file() for filename in ("groundtruth.txt", "groundtruth_rect.txt"))
        for child in path.iterdir()
    ) if path.is_dir() else False


def normalize_dataset_root(path: Path) -> Path:
    """Accept either the dataset directory itself or its extraction parent."""
    path = path.resolve()
    if _looks_like_dataset_root(path):
        return path
    for child_name in ("A2ATracking", "A2A-UAV-Tracking", "UAV2UAV-2"):
        candidate = path / child_name
        if _looks_like_dataset_root(candidate):
            return candidate
    return path


def resolve_sequence_paths(dataset_root: Path, name: str) -> tuple[list[Path], Path]:
    """Resolve both UAV2UAV-2 and A2A-UAV-Tracking directory layouts."""
    legacy_images = dataset_root / "img" / name
    legacy_gt = dataset_root / "anno" / f"{name}.txt"
    if legacy_images.is_dir() and legacy_gt.is_file():
        image_dir, gt_path = legacy_images, legacy_gt
    else:
        sequence_dir = dataset_root / name
        image_dir = sequence_dir / "img"
        gt_candidates = [sequence_dir / "groundtruth.txt", sequence_dir / "groundtruth_rect.txt"]
        gt_path = next((path for path in gt_candidates if path.is_file()), gt_candidates[0])
    image_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: natural_sort_key(path.name),
    ) if image_dir.is_dir() else []
    if not image_paths or not gt_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve sequence {name!r} under {dataset_root}; "
            "expected img/<name> + anno/<name>.txt or <name>/img + <name>/groundtruth.txt"
        )
    return image_paths, gt_path


def list_sequence_names(dataset_root: Path) -> list[str]:
    names = set()
    legacy_images, legacy_annotations = dataset_root / "img", dataset_root / "anno"
    if legacy_images.is_dir() and legacy_annotations.is_dir():
        names.update(
            path.name
            for path in legacy_images.iterdir()
            if path.is_dir() and (legacy_annotations / f"{path.name}.txt").is_file()
        )
    if dataset_root.is_dir():
        names.update(
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir()
            and (path / "img").is_dir()
            and any((path / filename).is_file() for filename in ("groundtruth.txt", "groundtruth_rect.txt"))
        )
    return sorted(names, key=natural_sort_key)


def infer_dataset_name(dataset_root: Path) -> str:
    names = list_sequence_names(dataset_root)
    if names and all(re.fullmatch(r"video\d+", name, re.IGNORECASE) for name in names):
        return "A2A-UAV-Tracking"
    return dataset_root.name


def load_gt(path: Path) -> np.ndarray:
    # Both benchmarks use text boxes, but public mirrors vary between comma,
    # tab and whitespace separators.  np.fromstring avoids layout-specific I/O.
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        values = np.fromstring(re.sub(r"[,;\t]+", " ", line), sep=" ", dtype=np.float64)
        if values.size not in (4, 8):
            raise ValueError(f"{path}:{line_number}: expected 4 or 8 coordinates, got {values.size}")
        if values.size == 8:
            xs, ys = values[0::2], values[1::2]
            values = np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()])
        rows.append(values)
    if not rows:
        raise ValueError(f"{path}: empty annotation file")
    gt = np.vstack(rows)
    if gt.ndim == 1:
        gt = gt[None, :]
    if gt.shape[1] != 4:
        raise ValueError(f"{path}: expected Nx4 annotation, got {gt.shape}")
    return gt
def choose_sequences(dataset_root: Path, requested, count: int) -> list[str]:
    names = list_sequence_names(dataset_root)
    if requested:
        missing = sorted(set(requested) - set(names))
        if missing:
            raise ValueError(f"Unknown sequences: {missing}")
        return list(requested)
    if count <= 0 or count >= len(names):
        return names
    indices = np.rint(np.linspace(0, len(names) - 1, count)).astype(int)
    return [names[i] for i in indices]
def read_image(path: Path) -> np.ndarray:
    """Decode RGB, including paths with non-ASCII characters on Windows."""
    import cv2
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
