"""Reproducible held-out evaluation for a LIRIS-ACCEDE video checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.utils.config import FRAME_SIZE, FPS, MAX_FRAMES
from scene_motion_llm.utils.dataset import VideoFrameDataset, _collate_fn


CLASS_NAMES = ["Positive", "Neutral", "Negative"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 3-class LIRIS-ACCEDE checkpoint.")
    parser.add_argument("--dataset-path", required=True, help="Path to the Liris_Accede video directory")
    parser.add_argument("--labels-path", required=True, help="Official ACCEDEaffect.txt or ACCEDEranking.txt")
    parser.add_argument("--checkpoint", required=True, help="3-class LIRIS checkpoint to evaluate")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def build_test_indices(labels, test_size, seed):
    """Return a deterministic, label-stratified held-out split."""
    indices = list(range(len(labels)))
    _, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return sorted(test_indices)


@torch.no_grad()
def evaluate(model, loader, device):
    predictions, targets, rows = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        flow = batch["optical_flow"]
        if flow is not None:
            flow = flow.to(device)
        outputs = model(frames, flow)
        probabilities = outputs["probabilities"].cpu()
        batch_predictions = probabilities.argmax(dim=1).tolist()
        batch_targets = batch["label"].tolist()
        predictions.extend(batch_predictions)
        targets.extend(batch_targets)
        for video_id, target, prediction, probability in zip(
            batch["video_id"], batch_targets, batch_predictions, probabilities.tolist()
        ):
            rows.append({
                "video_id": video_id,
                "true_label": CLASS_NAMES[target],
                "predicted_label": CLASS_NAMES[prediction],
                "positive_probability": probability[0],
                "neutral_probability": probability[1],
                "negative_probability": probability[2],
            })
    return targets, predictions, rows


def main():
    args = parse_args()
    labels_path = Path(args.labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError("Official LIRIS annotations are required for valid evaluation.")
    if labels_path.name.lower() not in {"accedeaffect.txt", "accederanking.txt"}:
        raise ValueError("Use an official ACCEDEaffect.txt or ACCEDEranking.txt annotation file.")

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dataset = VideoFrameDataset(
        dataset_type="liris_accede",
        dataset_path=args.dataset_path,
        labels_path=str(labels_path),
        max_frames=MAX_FRAMES,
        frame_size=FRAME_SIZE,
        fps=FPS,
        compute_flow=True,
    )
    test_indices = build_test_indices(dataset.labels, args.test_size, args.seed)
    loader = DataLoader(
        Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device == "cuda", collate_fn=_collate_fn,
    )

    model = SceneMotionLLMModel(num_classes=3, max_frames=MAX_FRAMES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("num_classes") != 3:
        raise ValueError("Checkpoint is not a 3-class LIRIS sentiment model.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    targets, predictions, rows = evaluate(model, loader, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": {key: checkpoint.get(key) for key in ("epoch", "task", "metrics")},
        "label_source": "official_liris_annotation",
        "split": {"seed": args.seed, "test_size": args.test_size, "strategy": "stratified_holdout"},
        "samples": len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            targets, predictions, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
        "note": "This split is reproducible. It is an independent test set only when the checkpoint was trained without these videos.",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
