"""
Data Migration Script: Merges CLIP features into the Semantic 5-Pillar cache.
"""
import json
import shutil
import numpy as np
from pathlib import Path

ORIGINAL_CACHE = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v2")
CLIP_CACHE = Path(r".\scene_motion_llm\cache\clip_vitb32_v1")
MERGED_CACHE = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v3_clip")

def main():
    MERGED_CACHE.mkdir(parents=True, exist_ok=True)

    # First, let's find all the .npz files in the original cache
    # The original cache just has them right in the folder (as per the _cache_paths logic)
    # Wait, earlier I saw them in `test/`, `train/`, `validation/` subdirectories. 
    # Actually, LirisSemanticFivePillarDataset._cache_paths just looks directly in cache_dir,
    # but when I listed the directory it had test/ train/ validation/.
    # Let me recursively find them just in case.
    original_npz_files = list(ORIGINAL_CACHE.rglob("*.npz"))
    
    print(f"Found {len(original_npz_files)} original cache files.")
    
    # Also find all CLIP files
    clip_npz_files = {p.stem: p for p in CLIP_CACHE.rglob("*.npz")}
    
    merged_count = 0
    missing_clip = 0
    
    for orig_npz in original_npz_files:
        vid = orig_npz.stem
        
        # Load original
        try:
            with np.load(orig_npz, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        except Exception as e:
            print(f"Error loading {orig_npz}: {e}")
            continue
            
        # Find matching CLIP
        if vid not in clip_npz_files:
            missing_clip += 1
            continue
            
        clip_npz = clip_npz_files[vid]
        try:
            with np.load(clip_npz, allow_pickle=False) as clip_loaded:
                clip_feats = np.asarray(clip_loaded["embeddings"])
        except Exception as e:
            print(f"Error loading {clip_npz}: {e}")
            continue
            
        # Verify sequence length matches
        orig_T = arrays["feature_spatial"].shape[0]
        clip_T = clip_feats.shape[0]
        
        if orig_T != clip_T:
            print(f"Shape mismatch for {vid}: Original T={orig_T}, CLIP T={clip_T}. Skipping.")
            continue
            
        # Swap the spatial feature
        arrays["feature_spatial"] = clip_feats
        
        # Write merged npz (to flat directory to match Dataset loading)
        out_npz = MERGED_CACHE / f"{vid}.npz"
        np.savez_compressed(out_npz, **arrays)
        
        # Copy metadata JSON exactly as is
        orig_json = orig_npz.with_suffix(".json")
        out_json = MERGED_CACHE / f"{vid}.json"
        if orig_json.exists():
            shutil.copy2(orig_json, out_json)
            
        merged_count += 1
        if merged_count % 500 == 0:
            print(f"Merged {merged_count} clips...")

    print(f"Done! Successfully merged {merged_count} clips.")
    if missing_clip > 0:
        print(f"Warning: {missing_clip} clips were missing from the CLIP cache.")

if __name__ == "__main__":
    main()
