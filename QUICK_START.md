# 🚀 Quick Reference: Get Started in 5 Minutes

## Step 1: Validate Your Setup (2 minutes)

```bash
cd scene_motion_llm

# Check everything is ready
python validate_datasets.py --validate --all

# This will show:
# ✓ Found datasets
# ✗ Missing datasets (with setup instructions)
```

## Step 2: Understand Your Training Path

### Option A: Complete 4-Step Pipeline (Recommended)
**Time: 6-10 hours on GPU | Results: Best performance**

```bash
python train_pipeline.py --all-stages \
  --epochs 50 \
  --batch-size 16 \
  --compute-flow \
  --use-llm
```

### Option B: Interactive Menu (Easiest)
```bash
python quick_start_pipeline.py
# Follow on-screen menu
```

### Option C: Individual Stages
```bash
# Stage 1: Motion Pretraining (2-3 hours)
python train_pipeline.py --stage 1 --compute-flow

# Stage 2: Sentiment Fine-tuning (1-2 hours) 
python train_pipeline.py --stage 2 --load-checkpoint checkpoints/stage1_kinetics/best_model.pt
```

## Step 3: Monitor Training

```
Watch these directories:
├── checkpoints/      (saved model weights)
├── outputs/          (training results & logs)
└── TRAINING_PIPELINE.md (detailed documentation)
```

## Step 4: Run Inference

```bash
# Single video prediction
python inference_llm.py \
  --video-path path/to/video.mp4 \
  --model-path checkpoints/stage2_accede/best_model.pt \
  --use-llm \
  --output-path results.json

# Result: 
# {
#   "sentiment": "Negative",
#   "confidence": 0.92,
#   "explanation": "Rapid motion patterns detected..."
# }
```

---

## 📊 What Was Changed

### ❌ Removed
- UCF-Crime dataset support
- All `UCF_CRIME_PATH` references
- `CRIME_NUM_CLASSES` constant

### ✅ Added
- LIRIS-ACCEDE dataset support
- Unified sentiment labels (Positive/Neutral/Negative)
- 4-step training pipeline
- LLM reasoning for explanations
- Interactive training menu
- Dataset validation tool

### 🔧 Modified Files
```
utils/config.py        - Config updates
config.py              - Export updates
utils/dataset.py       - New dataset loader
main_train.py          - Task options
```

---

## 🎯 Training Strategy Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: MOTION PRETRAINING (2-3 hours)                         │
│ ├─ UCF101: Learn 101 action classes                             │
│ └─ Kinetics: Fine-tune on 400 action classes                    │
│ Output: Strong motion encoder                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: SENTIMENT FINE-TUNING (1-2 hours)                      │
│ ├─ CMU-MOSEI: Learn continuous sentiment                        │
│ └─ LIRIS-ACCEDE: Learn discrete sentiment                       │
│ Output: Sentiment classifier                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: LLM REASONING (automatic)                              │
│ └─ Integrate explanation generation                             │
│ Output: Sentiment + Explanation                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: INFERENCE & TESTING                                    │
│ Input:  Video clip                                              │
│ Output: Sentiment (Positive/Neutral/Negative) + Explanation     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Dataset Label Mapping

### CMU-MOSEI (Continuous -3 to +3)
```
+2.5 → Positive (0)
 0.0 → Neutral (1)
-2.1 → Negative (2)
```

### LIRIS-ACCEDE (Categorical)
```
"positive" → Positive (0)
"neutral"  → Neutral (1)
"negative" → Negative (2)
```

---

## ⚡ Performance Tips

| Task | Recommendation | Impact |
|------|-----------------|--------|
| Speed | Disable `--compute-flow` | 2x faster, -3% accuracy |
| Accuracy | Enable `--compute-flow` | Better motion features |
| Memory | Reduce `--batch-size 8` | Prevents OOM |
| Quality | Use `--use-llm` | Better explanations |
| GPU | Use `--device cuda` | 6-8x faster vs CPU |

---

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| Path not found | Run `python validate_datasets.py --setup-guide` |
| Out of memory | `--batch-size 8` or disable `--compute-flow` |
| LLM fails | Falls back to rules. Install: `pip install transformers` |
| GPU not detected | Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |

---

## 📚 Full Documentation

For complete details see:
- **Training Guide**: [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- **Implementation Summary**: [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)

---

## 🎓 Example Workflow

### Day 1: Setup & Validation
```bash
# Check everything
python validate_datasets.py --all

# Create dataset directories
mkdir -p dataset/{UCF101,Kinetics,CMU_MOSEI,Liris_Accede}

# Copy datasets (follow setup guide)
# ...
```

### Day 2: Stage 1 Training
```bash
# Run motion pretraining
python train_pipeline.py --stage 1 --compute-flow --epochs 50

# Monitor progress
tail -f outputs/stage1_ucf101/training.log
```

### Day 3: Stage 2 Training
```bash
# Run sentiment fine-tuning
python train_pipeline.py --stage 2 \
  --load-checkpoint checkpoints/stage1_kinetics/best_model.pt

# Monitor progress
tail -f outputs/stage2_mosei/training.log
```

### Day 4: Inference & Testing
```bash
# Test on new videos
python inference_llm.py \
  --video-path dataset/Liris_Accede/test/ \
  --model-path checkpoints/stage2_accede/best_model.pt \
  --batch-process \
  --use-llm \
  --output-path results.json
```

---

## 🚀 One-Liner for Complete Training

```bash
# Start everything (6-10 hours)
python train_pipeline.py --all-stages \
  --ucf101-path ./dataset/UCF101 \
  --kinetics-path ./dataset/Kinetics \
  --mosei-path ./dataset/CMU_MOSEI \
  --liris-accede-path ./dataset/Liris_Accede \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --patience 5 \
  --compute-flow \
  --use-llm \
  --device cuda
```

---

## ✅ Success Checklist

- [ ] Validated datasets with `validate_datasets.py`
- [ ] Organized datasets in `dataset/` folder
- [ ] Started Stage 1 training
- [ ] Verified checkpoint saves in `checkpoints/`
- [ ] Started Stage 2 training
- [ ] Ran inference on test video
- [ ] Got sentiment prediction + explanation
- [ ] Saved results to JSON

---

## 📞 Need Help?

1. **Dataset issues**: `python validate_datasets.py --setup-guide`
2. **Training issues**: Check `TRAINING_PIPELINE.md`
3. **Environment issues**: Check `quick_start_pipeline.py --validate`
4. **Inference issues**: Check `inference_llm.py --help`

---

**Ready to train? Start with:**
```bash
python validate_datasets.py --validate
```

**or use interactive menu:**
```bash
python quick_start_pipeline.py
```
