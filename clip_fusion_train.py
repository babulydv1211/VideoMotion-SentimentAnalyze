"""
CLIP Fusion Trainer: CLIP ViT-B/32 visual + audio + motion features.
Uses an MLP (neural network) to learn non-linear combinations of the modalities.

Usage:
  python clip_fusion_train.py
"""
import json
import glob
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────────
CLIP_CACHE    = Path(r".\scene_motion_llm\cache\clip_vitb32_v1")
PILLAR_CACHE  = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v2")
OUTPUT_DIR    = Path(r".\scene_motion_llm\checkpoints\clip_fusion")
RANKING_PATH  = Path(r".\scene_motion_llm\dataset\annotations\ACCEDEranking.txt")
SETS_PATH     = Path(r"C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load labels ────────────────────────────────────────────────────────────
def load_labels():
    """Parse ACCEDEranking + ACCEDEsets to get split→{video_id: label}."""
    rankings = {}
    with open(RANKING_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                rankings[vid] = int(parts[2])

    split_map = {}
    with open(SETS_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                split_code = int(parts[2])
                split = {1: "train", 2: "validation", 0: "test"}.get(split_code)
                if split:
                    split_map[vid] = split

    train_scores = sorted([rankings[v] for v in split_map if split_map[v] == "train" and v in rankings])
    lo = train_scores[len(train_scores) // 3]
    hi = train_scores[2 * len(train_scores) // 3]

    def score_to_label(score):
        if score < lo:   return 0
        elif score < hi: return 1
        else:            return 2

    result = {"train": {}, "validation": {}, "test": {}}
    for vid, split in split_map.items():
        if vid in rankings:
            result[split][vid] = score_to_label(rankings[vid])
    return result

# ── Feature loading ────────────────────────────────────────────────────────
def load_clip_feature(video_id: str, split: str) -> np.ndarray | None:
    npz = CLIP_CACHE / split / f"{video_id}.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    return d["embeddings"].mean(axis=0).astype(np.float32)

def load_pillar_feature(video_id: str, split: str, pillar: str) -> np.ndarray | None:
    pattern = str(PILLAR_CACHE / split / f"*{video_id.replace('/', '_')}*.npz")
    matches = glob.glob(pattern)
    if not matches:
        npz = PILLAR_CACHE / split / f"{video_id}.npz"
        if not npz.exists():
            return None
        matches = [str(npz)]
    
    try:
        d = np.load(matches[0])
        feat  = d[f"feature_{pillar}"]
        avail = d[f"availability_{pillar}"]
        rel   = d[f"reliability_{pillar}"]
        valid = (avail > 0) & (rel > 0)
        if not np.any(valid):
            return None
        return feat[valid].mean(axis=0).astype(np.float32)
    except Exception:
        return None

def build_features(split: str, labels: dict) -> tuple[np.ndarray, np.ndarray, list]:
    video_ids = sorted(labels[split].keys())
    X_list, y_list, valid_ids = [], [], []

    missing_clip = 0
    for vid in video_ids:
        vis  = load_clip_feature(vid, split)
        audio = load_pillar_feature(vid, split, "audio")
        motion = load_pillar_feature(vid, split, "motion")

        if vis is None:
            missing_clip += 1
            continue

        if audio is None: audio = np.zeros(20, dtype=np.float32)
        if motion is None: motion = np.zeros(14, dtype=np.float32)

        X_list.append(np.concatenate([vis, audio, motion]))
        y_list.append(labels[split][vid])
        valid_ids.append(vid)

    if missing_clip > 0:
        print(f"  [{split}] Warning: {missing_clip} clips missing CLIP cache, skipped.")

    return np.stack(X_list), np.array(y_list, dtype=np.int64), valid_ids

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CLIP Fusion Trainer (CLIP + Audio + Motion MLP)")
    print("=" * 60)

    print("\n[1/4] Loading labels...")
    labels = load_labels()
    
    print("\n[2/4] Building feature matrices...")
    X_train, y_train, _ = build_features("train", labels)
    X_val,   y_val,   _ = build_features("validation", labels)
    X_test,  y_test,  _ = build_features("test", labels)
    print(f"  Train : {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n[3/4] Training MLP classifier (Grid search on validation)...")
    best_val_f1 = -1
    best_model  = None
    best_params = None

    # Try a few MLP architectures to learn non-linear combinations
    configs = [
        {"hidden_layer_sizes": (256, 128), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.01, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (512,), "alpha": 0.001, "learning_rate_init": 0.0005},
    ]

    for cfg in configs:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                **cfg,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ))
        ])
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"  {cfg['hidden_layer_sizes']} alpha={cfg['alpha']} -> val_acc={val_acc*100:.1f}% val_f1={val_f1*100:.1f}%")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model  = pipe
            best_params = cfg

    print(f"\n  Best arch: {best_params['hidden_layer_sizes']}  Best val macro-F1: {best_val_f1*100:.1f}%")

    print("\n[4/4] Final evaluation on held-out TEST set...")
    test_pred = best_model.predict(X_test)
    test_acc  = accuracy_score(y_test, test_pred)
    test_bacc = balanced_accuracy_score(y_test, test_pred)
    test_f1   = f1_score(y_test, test_pred, average="macro")
    val_pred  = best_model.predict(X_val)
    val_acc   = accuracy_score(y_val, val_pred)
    val_bacc  = balanced_accuracy_score(y_val, val_pred)
    val_f1    = f1_score(y_val, val_pred, average="macro")

    print(f"\n  {'Metric':<25} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Accuracy':<25} {val_acc*100:>9.2f}% {test_acc*100:>9.2f}%")
    print(f"  {'Balanced Accuracy':<25} {val_bacc*100:>9.2f}% {test_bacc*100:>9.2f}%")
    print(f"  {'Macro F1':<25} {val_f1*100:>9.2f}% {test_f1*100:>9.2f}%")

    # Save report
    report = {
        "model": "clip_mlp_fusion",
        "best_params": best_params,
        "validation": {"accuracy": val_acc, "balanced_accuracy": val_bacc, "macro_f1": val_f1},
        "test":       {"accuracy": test_acc, "balanced_accuracy": test_bacc, "macro_f1": test_f1},
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")
    print("\n  DONE!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
