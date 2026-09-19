"""
SceneMotion-LLM  —  End-to-End Pipeline
========================================
Run once after (or during) precompute and leave the PC.

Steps:
  1. Wait for the feature cache to be complete.
  2. Train the five-pillar fusion model on the best available device.
  3. Evaluate the best checkpoint on the official LIRIS test set.
  4. Smoke-test the Streamlit app's inference path on a sample video.
  5. Print the final report: accuracy, macro-F1, confusion matrix,
     per-pillar evidence for the smoke-test clip.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-wait      # cache already done
    python run_pipeline.py --device cpu     # force CPU training
    python run_pipeline.py --epochs 10      # quick debug run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to the workspace root
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
PKG         = ROOT / "scene_motion_llm"
PYTHON      = ROOT / "scene_motion_llm" / ".venv" / "Scripts" / "python.exe"
CACHE_DIR   = PKG / "cache" / "semantic_five_pillar_v2"
CKPT_DIR    = PKG / "checkpoints" / "semantic_five_pillar"
BEST_CKPT   = CKPT_DIR / "best_semantic_five_pillar.pt"
VIDEO_DIR   = PKG / "dataset" / "Liris_Accede"
RANKING     = PKG / "dataset" / "annotations" / "ACCEDEranking.txt"
SETS_PATH   = Path(r"C:\Users\student\Downloads\LIRIS-ACCEDE-annotations"
                   r"\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt")

SPLITS     = {"train": 2450, "validation": 2450, "test": 4900}
TOTAL_CLIPS = sum(SPLITS.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}", flush=True)


def run(args: list, env_extra: dict | None = None, check: bool = True) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(args, env=env)
    if check and result.returncode != 0:
        log(f"ERROR: command exited with code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def count_clips(split: str) -> int:
    p = CACHE_DIR / split
    return sum(1 for f in p.iterdir() if f.suffix == ".npz") if p.exists() else 0


TRAIN_VAL_SPLITS = {"train": SPLITS["train"], "validation": SPLITS["validation"]}


def trainval_cache_complete() -> bool:
    """True once train+val features are ready (tolerate 1 missing/skipped clip)."""
    return all(count_clips(s) >= t - 5 for s, t in TRAIN_VAL_SPLITS.items())


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true", help="Skip waiting for precompute")
    parser.add_argument("--device", default="cuda" if sys.platform != "darwin" else "mps", help="Device for training (cuda/cpu/mps)")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs to train")
    parser.add_argument("--phase2", action="store_true", help="Run with Phase 2 upgraded models (CLIP/PANNs)")
    return parser.parse_args()


def main():
    args = _cli()

    log("=== SceneMotion-LLM Pipeline ===")
    
    if args.phase2:
        global CACHE_DIR, CKPT_DIR, BEST_CKPT
        CACHE_DIR = PKG / "cache" / "semantic_five_pillar_phase2"
        CKPT_DIR = PKG / "checkpoints" / "semantic_five_pillar_phase2"
        BEST_CKPT = CKPT_DIR / "best_semantic_five_pillar.pt"

    if not args.skip_wait:
        wait_for_trainval_cache()
    else:
        log("Skipping wait (assuming cache complete).")

    train(args.device, args.epochs, 16, args.phase2)
    evaluate()
    smoke_test()
    log("Pipeline finished.")


def test_cache_complete() -> bool:
    """True once test features are also ready (tolerate a few skipped clips)."""
    return count_clips("test") >= SPLITS["test"] - 10


def pbar(done: int, total: int, w: int = 28) -> str:
    frac = min(done / max(total, 1), 1.0)
    filled = int(frac * w)
    return f"[{'#'*filled}{'-'*(w-filled)}] {frac*100:5.1f}%"


# ---------------------------------------------------------------------------
# Step 1 — wait for train+val cache (test can continue in background)
# ---------------------------------------------------------------------------
def wait_for_trainval_cache():
    """Block until train and validation caches are ready.
    Training does NOT need test features, so we start as soon as
    train+val are precomputed — the test set finishes in the background
    while the GPU is busy training.
    """
    poll = 15  # seconds between checks
    t0 = time.time()

    train_done = count_clips("train") >= SPLITS["train"] - 5
    val_done   = count_clips("validation") >= SPLITS["validation"] - 5

    if train_done and val_done:
        log("Step 1/5 — Train + validation cache already complete. Proceeding immediately.")
        for s in ("train", "validation"):
            log(f"  {s}: {count_clips(s):,} / {SPLITS[s]:,} clips cached")
        return

    log("Step 1/5 — Waiting for train + validation cache (test finishes in background)...")
    while not trainval_cache_complete():
        tv_done  = count_clips("train") + count_clips("validation")
        tv_total = SPLITS["train"] + SPLITS["validation"]
        elapsed  = timedelta(seconds=int(time.time() - t0))
        print(
            f"\r  {pbar(tv_done, tv_total)}  "
            f"{tv_done:,}/{tv_total:,} train+val clips  elapsed {elapsed}  "
            f"(checking every {poll}s)",
            end="", flush=True
        )
        time.sleep(poll)

    print()
    for s in ("train", "validation"):
        log(f"  {s}: {count_clips(s):,} / {SPLITS[s]:,} clips cached")
    log("Train + validation cache complete — starting training now.")
    test_n = count_clips("test")
    log(f"  test: {test_n:,} / {SPLITS['test']:,} still precomputing in background")


def wait_for_test_cache():
    """Block until the test cache is ready (called just before evaluation)."""
    if test_cache_complete():
        log(f"  test: {count_clips('test'):,} / {SPLITS['test']:,} clips cached (already done)")
        return
    log("  Waiting for test-set cache to finish (was precomputing during training)...")
    poll = 30
    while not test_cache_complete():
        n = count_clips("test")
        t = SPLITS["test"]
        print(f"\r  {pbar(n, t)}  {n:,}/{t:,} test clips", end="", flush=True)
        time.sleep(poll)
    print()
    log(f"  test: {count_clips('test'):,} clips cached.")


# ---------------------------------------------------------------------------
# Step 2 — train
# ---------------------------------------------------------------------------
def train(device: str, epochs: int, batch: int, phase2: bool = False):
    log(f"Step 2/5 — Training on {device.upper()}  ({epochs} epochs, batch={batch}, phase2={phase2})")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    phase2_args = ["--use-pretrained-spatial", "--use-clip", "--use-panns"] if phase2 else ["--use-pretrained-spatial"]

    run([
        str(PYTHON), "-u", "-m",
        "scene_motion_llm.training.semantic_five_pillar_training",
        "--mode",           "train",
        "--video-dir",      str(VIDEO_DIR),
        "--ranking-path",   str(RANKING),
        "--sets-path",      str(SETS_PATH),
        "--cache-dir",      str(CACHE_DIR),
        "--checkpoint-dir", str(CKPT_DIR),
        "--epochs",         str(epochs),
        "--batch-size",     str(batch),
        "--device",         device,
    ] + phase2_args)

    if not BEST_CKPT.exists():
        log("ERROR: training finished but best_semantic_five_pillar.pt not found.")
        sys.exit(1)
    log(f"Best checkpoint saved: {BEST_CKPT}")


# ---------------------------------------------------------------------------
# Step 3 — evaluate
# ---------------------------------------------------------------------------
def evaluate():
    log("Step 3/5 — Evaluating checkpoint on official test set...")
    import torch

    ckpt = torch.load(str(BEST_CKPT), map_location="cpu", weights_only=False)

    print()
    print("=" * 60)
    print("  OFFICIAL TEST-SET RESULTS")
    print("=" * 60)

    for key in ("test_report", "val_report"):
        val = ckpt.get(key)
        if val:
            label = "Test" if "test" in key else "Validation"
            print(f"\n  {label} report:")
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {val}")

    best_f1  = ckpt.get("best_val_f1", ckpt.get("test_macro_f1", None))
    best_acc = ckpt.get("best_val_acc", ckpt.get("test_accuracy", None))

    if best_f1 is not None:
        print(f"\n  Macro-F1   : {float(best_f1):.4f}  ({float(best_f1)*100:.1f}%)")
    if best_acc is not None:
        print(f"  Accuracy   : {float(best_acc):.4f}  ({float(best_acc)*100:.1f}%)")

    cm = ckpt.get("confusion_matrix")
    if cm:
        import numpy as np
        m = np.array(cm)
        classes = ["Positive", "Neutral", "Negative"]
        print("\n  Confusion Matrix (rows=actual, cols=predicted):")
        header = f"  {'':12}" + "".join(f"{c:>12}" for c in classes)
        print(header)
        for i, row in enumerate(m):
            cells = "".join(f"{v:>12}" for v in row)
            print(f"  {classes[i]:<12}{cells}")

    if best_f1 is not None:
        f1 = float(best_f1)
        print()
        if f1 >= 0.60:
            print("  >> Strong result: above 60% macro-F1 on a 3-class affect task.")
        elif f1 >= 0.50:
            print("  >> Decent baseline. Add CLIP/PANNs to push higher.")
        else:
            print("  >> Below 50% — check class balance and feature extraction.")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Step 4 — smoke-test inference on a sample clip
# ---------------------------------------------------------------------------
def smoke_test():
    log("Step 4/5 — Smoke-testing inference on a sample video clip...")

    # Pick the first available video in the dataset
    clips = sorted(VIDEO_DIR.glob("*.mp4"))[:1] + sorted(VIDEO_DIR.glob("*.avi"))[:1]
    if not clips:
        log("  No sample clip found in dataset directory — skipping smoke test.")
        return

    clip = clips[0]
    log(f"  Using sample clip: {clip.name}")

    sys.path.insert(0, str(ROOT))
    try:
        from scene_motion_llm.inference.semantic_five_pillar_inference import (
            SemanticFivePillarInferencer,
        )
        inferencer = SemanticFivePillarInferencer(checkpoint_path=str(BEST_CKPT))
        result = inferencer.predict(str(clip))

        print()
        print("=" * 60)
        print("  INFERENCE SMOKE TEST")
        print("=" * 60)
        print(f"  Clip      : {clip.name}")
        print(f"  Sentiment : {result.get('sentiment', '?')}")
        print(f"  Confidence: {float(result.get('confidence', 0)):.1%}")
        print(f"  Audio     : {result.get('audio_status', 'unknown')}")

        pillar_summary = result.get("pillar_summary", {})
        if pillar_summary:
            print("\n  Pillar evidence:")
            for pillar, info in pillar_summary.items():
                if isinstance(info, dict):
                    avail = "yes" if info.get("available") else "no"
                    rel   = info.get("reliability", 0)
                    print(f"    {pillar:<12}: available={avail}  reliability={float(rel):.2f}")

        print("=" * 60)
        print()
        log("Inference smoke test PASSED.")

    except Exception as exc:
        log(f"  Smoke test error (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Step 5 — final summary
# ---------------------------------------------------------------------------
def final_summary(device: str, epochs: int):
    log("Step 5/5 — Pipeline complete.")
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Checkpoint : {BEST_CKPT}")
    print(f"  Trained on : {device.upper()}, {epochs} epochs")
    print()
    print("  To launch the Streamlit app:")
    print(r"    .\scene_motion_llm\.venv\Scripts\streamlit.exe run scene_motion_llm\app.py")
    print()
    print("  To re-evaluate anytime:")
    print(r"    python evaluate.py")
    print()
    print("  To add CLIP/PANNs and retrain (next upgrade):")
    print(r"    See PROJECT_CONTEXT.md  Section 4, Step 4")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SceneMotion-LLM end-to-end pipeline")
    ap.add_argument("--skip-wait",  action="store_true", help="Skip waiting for cache (already done)")
    ap.add_argument("--skip-train", action="store_true", help="Skip training (checkpoint already exists)")
    ap.add_argument("--device",     default="cuda",      help="cuda or cpu")
    ap.add_argument("--epochs",     type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # Auto-detect device
    sys.path.insert(0, str(ROOT))
    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            log("CUDA not available — falling back to CPU.")
            args.device = "cpu"
        else:
            gpu = torch.cuda.get_device_name(0) if args.device == "cuda" else "N/A"
            log(f"Device: {args.device.upper()}  ({gpu})")
    except ImportError:
        log("torch not found — using scene_motion_llm/.venv python for training subprocess.")

    print()
    log("SceneMotion-LLM End-to-End Pipeline Starting")
    log(f"  Cache dir      : {CACHE_DIR}")
    log(f"  Checkpoint dir : {CKPT_DIR}")
    log(f"  Device         : {args.device.upper()}")
    log(f"  Epochs         : {args.epochs}")
    print()

    if not args.skip_wait:
        wait_for_trainval_cache()
    else:
        log("Step 1/5 — Train/val cache wait skipped.")

    if not args.skip_train:
        train(args.device, args.epochs, args.batch_size)
    else:
        if not BEST_CKPT.exists():
            log("--skip-train given but checkpoint not found. Run training first.")
            sys.exit(1)
        log("Step 2/5 — Training skipped (checkpoint exists).")

    if not args.skip_wait:
        wait_for_test_cache()

    evaluate()
    smoke_test()
    final_summary(args.device, args.epochs)


if __name__ == "__main__":
    main()
