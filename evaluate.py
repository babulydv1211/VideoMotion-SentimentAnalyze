"""
Quick post-training evaluation for the semantic five-pillar checkpoint.

Run from project root after training completes:
    python evaluate.py

Prints:
- Overall accuracy
- Per-class precision / recall / F1
- Confusion matrix
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import numpy as np

CLASS_NAMES = ("Positive", "Neutral", "Negative")


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def print_confusion_matrix(matrix: np.ndarray):
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = f"{'':12s}" + "".join(f"{c:>12s}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(matrix):
        cells = "".join(f"{v:>12d}" for v in row)
        print(f"  {CLASS_NAMES[i]:<10s}{cells}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate five-pillar checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="scene_motion_llm/checkpoints/semantic_five_pillar_v3_compact/best_semantic_five_pillar.pt",
        help="Path to trained checkpoint",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run training first: .\\run_train.ps1")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path)

    # The canonical semantic trainer stores the selected validation and final
    # held-out reports under these names.  Keep the legacy spellings as a
    # fallback so this small utility remains useful for older checkpoints.
    reports = (
        ("validation_metrics", "Validation metrics"),
        ("test_metrics", "Held-out test metrics"),
        ("val_report", "Validation report (legacy)"),
        ("test_report", "Test report (legacy)"),
    )
    for key, title in reports:
        if key in ckpt:
            print(f"\n--- {title} ---")
            val = ckpt[key]
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {val}")

    if "confusion_matrix" in ckpt:
        matrix = np.array(ckpt["confusion_matrix"])
        print_confusion_matrix(matrix)

    validation = ckpt.get("validation_metrics", ckpt.get("val_report", {}))
    if isinstance(validation, dict) and "macro_f1" in validation:
        f1 = float(validation["macro_f1"])
        print(f"\n{'='*40}")
        print(f"Selected validation macro-F1: {f1:.4f} ({f1*100:.1f}%)")
        if f1 >= 0.60:
            print("Strong result: above 60% macro-F1 on a 3-class affect task.")
        elif f1 >= 0.50:
            print("Decent baseline: evaluate representation upgrades next.")
        else:
            print("Below 50%: inspect cache consistency, class behavior, and feature extraction.")

    epoch = ckpt.get("epoch")
    if epoch is not None:
        print(f"Selected epoch: {epoch}")

    label_rule = ckpt.get("label_rule", "unknown")
    split_proto = ckpt.get("split_protocol", "unknown")
    print(f"\nLabel rule: {label_rule}")
    print(f"Split protocol: {split_proto}")


if __name__ == "__main__":
    main()
