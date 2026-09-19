"""
CLIP ViT-B/32 feature extractor for LIRIS-ACCEDE clips.
Extracts 512-dim semantic embeddings per clip (mean of 16 frames).

Usage:
  python clip_precompute.py
"""
import json
import numpy as np
import torch
import open_clip
import cv2
from pathlib import Path
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO_DIR    = Path(r".\scene_motion_llm\dataset\Liris_Accede")
RANKING_PATH = Path(r".\scene_motion_llm\dataset\annotations\ACCEDEranking.txt")
SETS_PATH    = Path(r"C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt")
CACHE_DIR    = Path(r".\scene_motion_llm\cache\clip_vitb32_v1")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLED_FRAMES = 16
MAX_DURATION   = 15.0  # seconds

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Load CLIP ──────────────────────────────────────────────────────────────
print(f"Loading CLIP ViT-B/32 on {DEVICE}...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(DEVICE).eval()
print("CLIP loaded.")

# ── Parse splits ───────────────────────────────────────────────────────────
split_map = {}
with open(SETS_PATH) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            vid = parts[1].replace(".mp4", "")
            code = int(parts[2])
            split = {1: "train", 2: "validation", 0: "test"}.get(code)
            if split:
                split_map[vid] = split

for split in ["train", "validation", "test"]:
    (CACHE_DIR / split).mkdir(exist_ok=True)

vids_by_split = {"train": [], "validation": [], "test": []}
for vid, split in split_map.items():
    vids_by_split[split].append(vid)

# ── Extract frames ─────────────────────────────────────────────────────────
def extract_frames(video_path: Path, n_frames: int = SAMPLED_FRAMES) -> list:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(total - 1, int(fps * MAX_DURATION))
    indices = np.linspace(0, max_frame, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

# ── Main loop ──────────────────────────────────────────────────────────────
total_done = 0
total_all  = sum(len(v) for v in vids_by_split.values())

for split in ["train", "validation", "test"]:
    vids = sorted(vids_by_split[split])
    split_done = 0
    split_cached = 0
    print(f"\n[{split}] {len(vids)} clips")

    for vid in vids:
        out_npz = CACHE_DIR / split / f"{vid}.npz"
        if out_npz.exists():
            split_cached += 1
            total_done += 1
            split_done += 1
            if split_done % 100 == 0:
                print(f"  {split_done}/{len(vids)} (cached)")
            continue

        # Find video file
        video_path = VIDEO_DIR / f"{vid}.mp4"
        if not video_path.exists():
            print(f"  MISSING: {vid}")
            continue

        # Extract frames
        frames = extract_frames(video_path)
        if not frames:
            print(f"  FAILED to extract frames: {vid}")
            continue

        # Run CLIP
        with torch.no_grad():
            tensors = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
            feats = model.encode_image(tensors)          # (N, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            feats = feats.cpu().float().numpy()

        np.savez_compressed(out_npz, embeddings=feats)

        split_done += 1
        total_done += 1
        if split_done % 100 == 0:
            print(f"  {split_done}/{len(vids)} extracted (cached={split_cached})")

    print(f"  [{split}] done. extracted={split_done-split_cached} cached={split_cached}")

print(f"\nAll done! {total_done}/{total_all} clips processed.")
print(f"Cache: {CACHE_DIR}")
