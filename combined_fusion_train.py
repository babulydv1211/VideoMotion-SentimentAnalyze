"""
Combined fusion trainer: ResNet18 visual + audio + motion features.
Replaces weak handcrafted color/spatial/temporal pillars with strong
ResNet18 visual embeddings while keeping audio and motion.

Usage:
  python combined_fusion_train.py

Expects:
  - scene_motion_llm/cache/resnet18_full_v1/{train,validation,test}/*.npz
  - scene_motion_llm/cache/semantic_five_pillar_v2/{train,validation,test}/*.npz
"""
import json
import glob
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────────
RESNET_CACHE  = Path(r".\scene_motion_llm\cache\resnet18_full_v1")
PILLAR_CACHE  = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v2")
OUTPUT_DIR    = Path(r".\scene_motion_llm\checkpoints\combined_fusion")
RANKING_PATH  = Path(r".\scene_motion_llm\dataset\annotations\ACCEDEranking.txt")
SETS_PATH     = Path(r"C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt")
CLASS_NAMES   = ["negative", "neutral", "positive"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load labels ────────────────────────────────────────────────────────────
def load_labels():
    """Parse ACCEDEranking + ACCEDEsets to get split→{video_id: label}."""
    # Load rankings — tab-separated with header row, col 1=name, col 2=valenceRank
    rankings = {}
    with open(RANKING_PATH) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                rankings[vid] = int(parts[2])   # valenceRank

    # Load splits — tab-separated with header, col 1=name, col 2=set code
    split_map = {}
    with open(SETS_PATH) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                split_code = int(parts[2])
                split = {1: "train", 2: "validation", 0: "test"}.get(split_code)
                if split:
                    split_map[vid] = split

    # Tertile cutoffs from train clips
    train_scores = sorted([rankings[v] for v in split_map if split_map[v] == "train" and v in rankings])
    lo = train_scores[len(train_scores) // 3]
    hi = train_scores[2 * len(train_scores) // 3]

    def score_to_label(score):
        if score < lo:   return 0  # negative
        elif score < hi: return 1  # neutral
        else:            return 2  # positive

    result = {"train": {}, "validation": {}, "test": {}}
    for vid, split in split_map.items():
        if vid in rankings:
            result[split][vid] = score_to_label(rankings[vid])
    return result


# ── Feature loading ────────────────────────────────────────────────────────
def load_resnet_feature(video_id: str, split: str) -> np.ndarray | None:
    """Mean-pool 16 ResNet18 frame embeddings → 512-dim."""
    npz = RESNET_CACHE / split / f"{video_id}.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    emb = d["embeddings"]  # (16, 512)
    return emb.mean(axis=0).astype(np.float32)

def load_pillar_feature(video_id: str, split: str, pillar: str) -> np.ndarray | None:
    """Mean-pool pillar windows using only available/reliable frames."""
    # Find .npz — filename may use underscore-escaped video_id
    pattern = str(PILLAR_CACHE / split / f"*{video_id.replace('/', '_')}*.npz")
    matches = glob.glob(pattern)
    if not matches:
        # Try exact name
        npz = PILLAR_CACHE / split / f"{video_id}.npz"
        if not npz.exists():
            return None
        matches = [str(npz)]

    npz_path = matches[0]
    try:
        d = np.load(npz_path)
        feat  = d[f"feature_{pillar}"]        # (T, D)
        avail = d[f"availability_{pillar}"]   # (T,)
        rel   = d[f"reliability_{pillar}"]    # (T,)
        valid = (avail > 0) & (rel > 0)
        if not np.any(valid):
            return None
        return feat[valid].mean(axis=0).astype(np.float32)
    except Exception:
        return None

def build_features(split: str, labels: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Build (X, y, video_ids) for one split."""
    video_ids = sorted(labels[split].keys())
    X_list, y_list, valid_ids = [], [], []

    missing_resnet = 0
    for vid in video_ids:
        vis  = load_resnet_feature(vid, split)
        audio = load_pillar_feature(vid, split, "audio")
        motion = load_pillar_feature(vid, split, "motion")

        if vis is None:
            missing_resnet += 1
            continue  # ResNet feature is mandatory

        # Audio: zero-fill if missing (mute clip)
        audio_dim = 20
        if audio is None:
            audio = np.zeros(audio_dim, dtype=np.float32)

        # Motion: zero-fill if missing
        motion_dim = 14
        if motion is None:
            motion = np.zeros(motion_dim, dtype=np.float32)

        combined = np.concatenate([vis, audio, motion])  # 512+20+14 = 546
        X_list.append(combined)
        y_list.append(labels[split][vid])
        valid_ids.append(vid)

    if missing_resnet > 0:
        print(f"  [{split}] Warning: {missing_resnet} clips missing ResNet cache, skipped.")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y, valid_ids

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Combined Fusion Trainer (ResNet18 + Audio + Motion)")
    print("=" * 60)

    print("\n[1/4] Loading labels...")
    labels = load_labels()
    for split, d in labels.items():
        counts = np.bincount([v for v in d.values()], minlength=3)
        print(f"  {split}: {len(d)} clips  neg={counts[0]} neu={counts[1]} pos={counts[2]}")

    print("\n[2/4] Building feature matrices...")
    X_train, y_train, _ = build_features("train", labels)
    X_val,   y_val,   _ = build_features("validation", labels)
    X_test,  y_test,  _ = build_features("test", labels)
    print(f"  Train : {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Feature dims: {X_train.shape[1]} (ResNet512 + audio20 + motion14)")

    print("\n[3/4] Training classifiers (C grid search on val macro-F1)...")
    best_val_f1 = -1
    best_model  = None
    best_c      = None

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=min(128, X_train.shape[1]), random_state=42)),
            ("clf",    LogisticRegression(C=c, max_iter=2000, class_weight="balanced",
                                          multi_class="multinomial", solver="lbfgs",
                                          random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  C={c:<6}  val_acc={val_acc*100:.1f}%  val_macro_f1={val_f1*100:.1f}%")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model  = pipe
            best_c      = c

    print(f"\n  Best C: {best_c}  Best val macro-F1: {best_val_f1*100:.1f}%")

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
        "model": "combined_fusion_resnet18_audio_motion",
        "best_C": best_c,
        "feature_dims": {"resnet18_visual": 512, "audio": 20, "motion": 14, "total": 546},
        "validation": {"accuracy": val_acc, "balanced_accuracy": val_bacc, "macro_f1": val_f1},
        "test":       {"accuracy": test_acc, "balanced_accuracy": test_bacc, "macro_f1": test_f1},
        "train_size": len(y_train), "val_size": len(y_val), "test_size": len(y_test),
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")
    print("\n  DONE!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
