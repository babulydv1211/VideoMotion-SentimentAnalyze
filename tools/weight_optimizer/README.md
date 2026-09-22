# Weight Optimizer â€” 6-Pillar Expert System

This tool finds the best pillar weights for `clip_demo.py` by running a grid search against the LIRIS-ACCEDE ground truth annotations.

## How It Works
1. Loads ground truth from `dataset/annotations/ACCEDEranking.txt`
2. Converts continuous `valenceValue` (1â€“5 scale) to 3 classes:
   - `< 2.5` â†’ **Negative**
   - `2.5 â€“ 3.5` â†’ **Neutral**
   - `> 3.5` â†’ **Positive**
3. Samples **60 balanced videos** (20 per class)
4. Loads all models **once** (CLIP, Wav2Vec2, DistilBERT, Librosa)
5. Runs all 6 pillars on each video and stores raw votes
6. Grid-searches weight combinations (step 0.05), enforcing:
   - `CLIP_W >= 0.20` (semantic anchor must stay meaningful)
   - `ACOUS_W > SPEECH_W` (Mehrabian's rule: tone > words)
   - `COLOR_W = 0.05`, `TEMP_W = 0.05` (fixed structural pillars)
7. Prints the winning weights and accuracy

## How to Run
```powershell
cd C:\Users\student\Desktop\sentiment_analysis
.\.codex-venv\Scripts\python.exe tools\weight_optimizer\optimize_weights.py
```

## After Running
Copy the printed best weights into `clip_demo.py`:
```python
CLIP_W   = <best value>
SPEECH_W = <best value>
ACOUS_W  = <best value>
MOTION_W = <best value>
COLOR_W  = 0.05
TEMP_W   = 0.05
```

## Notes
- Requires the full LIRIS-ACCEDE dataset in `dataset/Liris_Accede/`
- Runtime: ~6â€“8 minutes on CPU
- Sample size can be changed by editing the `sample(min(20, len(g)), ...)` line

