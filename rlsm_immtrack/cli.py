"""Command-line entry points; help and scoring do not import PyTorch."""
import argparse
import json
from pathlib import Path

from .experiment import (
    DEFAULT_CONFIG, ROOT, atomic_json, evaluate, inspect_dataset,
    plot_metrics, score_predictions, write_csv,
)


def dataset_options(parser):
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+", help="Omit to process ALL discovered sequences")
    parser.add_argument("--alignment", choices=("strict", "common"), default="strict",
                        help="common explicitly truncates to min(image count, annotation count)")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug prefix length; 0 means full")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RLSM-IMMTrack reproducible inference and OPE")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Check sequence structure, counts and annotations")
    dataset_options(check)
    check.add_argument("--decode", action="store_true", help="Decode every selected image (slower)")
    check.add_argument("--output", type=Path, help="Optional JSON report; must not already exist")

    run = commands.add_parser("evaluate", help="Run RLSM, SafeCL or original SUTrack on a CUDA GPU")
    dataset_options(run)
    run.add_argument("--tracker", choices=("rlsm", "safecl", "sutrack"), default="rlsm")
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--sutrack-root", type=Path, default=ROOT / "third_party" / "SUTrack")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--resume", action="store_true")

    score = commands.add_parser("score", help="Score saved Nx4 xywh CSV predictions without a GPU")
    dataset_options(score)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    plot = commands.add_parser("plot", help="Plot success, center and normalized precision")
    plot.add_argument("--metrics", type=Path, nargs="+", required=True)
    plot.add_argument("--labels", nargs="+")
    plot.add_argument("--output", type=Path, required=True)

    commands.add_parser("doctor", help="Print local Python, dependency and CUDA availability")
    args = parser.parse_args(argv)
    if getattr(args, "max_frames", 0) < 0:
        parser.error("--max-frames cannot be negative")
    try:
        if args.command == "evaluate":
            evaluate(args)
        elif args.command in ("validate", "score"):
            root, inventory = inspect_dataset(args.dataset_root, args.sequences,
                                             args.alignment, args.max_frames,
                                             getattr(args, "decode", False))
            if args.command == "validate":
                result = {
                    "dataset_root": str(root), "sequences": inventory,
                    "sequence_count": len(inventory),
                    "frames": sum(x["frames"] for x in inventory),
                    "evaluated_frames": sum(x["evaluated_frames"] for x in inventory),
                    "all_images_decoded": args.decode,
                }
                if args.output:
                    if args.output.exists():
                        raise FileExistsError(args.output)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(args.output, result)
                print(json.dumps(result, indent=2))
            else:
                if args.output.exists() and any(args.output.iterdir()):
                    raise FileExistsError(f"{args.output} is not empty")
                result = score_predictions(root, inventory, args.predictions)
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output / "metrics.json", result)
                write_csv(args.output / "per_sequence.csv", result["per_sequence"])
                print(json.dumps(result["aggregate"], indent=2))
        elif args.command == "plot":
            plot_metrics(args.metrics, args.labels, args.output)
        else:
            import importlib.metadata
            import platform
            result = {"python": platform.python_version()}
            for name in ("numpy", "torch", "torchvision", "opencv-python", "timm",
                         "easydict", "einops", "PyYAML", "openai-clip", "matplotlib"):
                try:
                    result[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    result[name] = "not installed"
            try:
                import torch
                result["cuda_available"] = torch.cuda.is_available()
                result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            except ImportError:
                result["cuda_available"] = False
            print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
