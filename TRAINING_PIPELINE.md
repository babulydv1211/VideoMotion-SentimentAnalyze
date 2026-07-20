# SceneMotion-LLM: 4-Step Sentiment Analysis Training Pipeline

## Overview

Complete end-to-end training pipeline for video sentiment analysis with integrated LLM reasoning. The pipeline progressively trains a motion encoder for action understanding, then fine-tunes it on sentiment datasets with LLM-based explanation generation.

---

## Training Strategy

### **Stage 1: Motion Pretraining** (Motion Understanding)
Learn robust motion representations from large-scale action recognition datasets:

- **Step 1**: Train on **UCF101** (101 action classes)
  - Goal: Learn spatial-temporal motion patterns
  - Output: Motion encoder checkpoint
  
- **Step 2**: Fine-tune on **Kinetics-400** (400 action classes)
  - Goal: Continue pretraining with diverse actions
  - Output: Strong pretrained motion encoder

### **Stage 2: Sentiment Fine-Tuning** (Emotion Recognition)
Transfer motion understanding to sentiment classification:

- **Step 3**: Fine-tune on **CMU-MOSEI** (Continuous sentiment scores)
  - Goal: Learn sentiment features from facial expressions + body language
  - Label Mapping: Continuous → Discrete (Positive/Neutral/Negative)
  
- **Step 4**: Fine-tune on **LIRIS-ACCEDE** (Discrete sentiment labels)
  - Goal: Adapt to different facial expression database
  - Label Mapping: Unified (Positive/Neutral/Negative)

### **Stage 3: LLM Reasoning Layer**
Generate natural language explanations for predictions:

```
Video → Motion Encoder → Sentiment Prediction → LLM Reasoning → Output
         (Pretrained)       (3 classes)        (Phi-2 or rules)
         
Result:
  Sentiment: Negative
  Confidence: 92.3%
  Reason: "Rapid and erratic motion patterns detected. Abrupt motion changes 
           and avoidance behaviors suggest distress or fear."
```

### **Stage 4: Testing**
Unified inference pipeline with all components:
- Motion encoding
- Sentiment classification
- LLM-based explanation generation

---

## Sentiment Label Mapping

### Unified Label Space
```
0: Positive   (happy, excited, content, uplifted)
1: Neutral    (calm, balanced, standard)
2: Negative   (sad, angry, fearful, distressed)
```

### Dataset-Specific Mappings

**CMU-MOSEI** (Continuous [-3, +3] → Discrete)
```
score > 0.5  → Positive (0)
score < -0.5 → Negative (2)
else         → Neutral  (1)
```

**LIRIS-ACCEDE** (Direct sentiment labels)
```
"Positive" → 0
"Neutral"  → 1
"Negative" → 2
```

---

## Installation

### 1. Clone and Setup
```bash
cd sentiment_Analysis/scene_motion_llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install transformers  # For LLM (optional)
```

### 3. Dataset Structure
Ensure your datasets follow this structure:

```
dataset/
├── UCF101/
│   ├── train/
│   │   ├── Action1/
│   │   │   ├── video1.avi
│   │   │   └── ...
│   │   └── ...
│   ├── test/
│   └── val/
├── Kinetics/
│   ├── train/
│   ├── test/
│   └── val/
├── CMU_MOSEI/
│   ├── train.features
│   ├── test.features
│   ├── val.features
│   └── w2v.vectors
└── Liris_Accede/
    ├── positive/
    │   ├── video1.mp4
    │   └── ...
    ├── neutral/
    └── negative/
```

---

## Training

### Quick Start: Complete 4-Step Pipeline

```bash
python train_pipeline.py --all-stages \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --compute-flow \
  --use-llm
```

### Individual Stage Training

**Stage 1: Motion Pretraining**
```bash
python train_pipeline.py --stage 1 \
  --ucf101-path ./dataset/UCF101 \
  --kinetics-path ./dataset/Kinetics \
  --epochs 50 \
  --compute-flow
```

**Stage 2: Sentiment Fine-tuning**
```bash
python train_pipeline.py --stage 2 \
  --mosei-path ./dataset/CMU_MOSEI \
  --liris-accede-path ./dataset/Liris_Accede \
  --epochs 30 \
  --load-checkpoint ./checkpoints/stage1_kinetics/best_model.pt
```

**All Stages (Recommended)**
```bash
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

### Traditional Single-Task Training

```bash
# CMU-MOSEI Sentiment Only
python main_train.py --task sentiment \
  --epochs 50 \
  --batch-size 16

# UCF101 Action Recognition
python main_train.py --task auxiliary_ucf101 \
  --epochs 50 \
  --compute-flow

# LIRIS-ACCEDE Sentiment
python main_train.py --task auxiliary_liris \
  --liris-path ./dataset/Liris_Accede \
  --epochs 30
```

---

## Training Parameters

```
--epochs              Training epochs (default: 50)
--batch-size         Batch size (default: 16)
--learning-rate      Initial learning rate (default: 1e-4)
--patience           Early stopping patience (default: 5)
--num-workers        DataLoader workers (default: 0)
--device             cuda or cpu (default: cuda)
--compute-flow       Compute optical flow features (slower but better)
--use-llm            Use local LLM for reasoning (requires transformers)
--checkpoint-dir     Path to save checkpoints
--output-dir         Path to save results
--load-checkpoint    Load pretrained checkpoint
```

---

## Inference

### Single Video with LLM Reasoning

```bash
python inference_llm.py \
  --video-path ./videos/sample.mp4 \
  --model-path ./checkpoints/stage2_accede/best_model.pt \
  --use-llm \
  --output-path ./results/inference.json
```

### Batch Processing

```bash
python inference_llm.py \
  --video-path ./dataset/Liris_Accede/test/ \
  --model-path ./checkpoints/final_model.pt \
  --batch-process \
  --use-llm \
  --output-path ./results/batch_inference.json
```

### Inference Output Format

```json
{
  "video_path": "sample.mp4",
  "num_frames": 8,
  "results": {
    "sentiment_idx": 2,
    "sentiment_label": "Negative",
    "confidence": [0.05, 0.03, 0.92],
    "explanation": "Rapid and erratic motion patterns detected. Abrupt motion changes and avoidance behaviors observed. Jerky movements suggest distress or fear."
  }
}
```

---

## Model Architecture

### Motion Encoder (Stage 1)
```
Input Video → Spatial CNN → Temporal LSTM → Multi-Head Attention
              (ResNet-50)    (256 hidden)   (128 dim)
                             ↓
                        Fusion Layer (512-dim) → Action Logits
```

### Sentiment Classifier (Stage 2)
```
Motion Features → Temporal Attention → Fusion → Classification Head
(2048-dim)       (128-dim)           (512-dim)  → Sentiment Logits
                                                  (3 classes)
```

### LLM Reasoning (Stage 3)
```
Sentiment Logits → Arg Max → LLM/Rules → Natural Language
                 (confidence)         Explanation
```

---

## Label Unification

### Why Unify Labels?

Different datasets use different sentiment representations:
- CMU-MOSEI: Continuous [-3, +3] scores
- LIRIS-ACCEDE: Categorical (Positive/Neutral/Negative)
- Domain variation in facial expressions

**Solution**: Map all to unified 3-class sentiment space

### Mapping Process

```python
# CMU-MOSEI
if score > 0.5:
    label = 0  # Positive
elif score < -0.5:
    label = 2  # Negative
else:
    label = 1  # Neutral

# LIRIS-ACCEDE
label_map = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}
```

---

## Expected Results

After completing all 4 stages:

| Stage | Dataset | Metric | Expected |
|-------|---------|--------|----------|
| 1 | UCF101 | Top-1 Accuracy | 85-90% |
| 1 | Kinetics | Top-1 Accuracy | 75-80% |
| 2 | CMU-MOSEI | Accuracy | 70-75% |
| 2 | LIRIS-ACCEDE | Accuracy | 65-75% |
| 3 | Combined | F1-Score | 0.68-0.72 |

---

## File Structure

```
scene_motion_llm/
├── train_pipeline.py          # Main 4-step training
├── inference_llm.py           # Inference with LLM
├── main_train.py              # Traditional training (backward compatible)
├── config.py                  # Config exports
├── models/
│   ├── sentiment_classifier.py
│   ├── feature_sentiment.py
│   ├── attention.py
│   └── ...
├── training/
│   └── feature_training.py
├── utils/
│   ├── config.py              # Updated: LIRIS-ACCEDE path
│   ├── dataset.py             # Updated: LIRIS-ACCEDE loader
│   ├── frame_extractor.py
│   └── ...
└── checkpoints/
    ├── stage1_ucf101/
    ├── stage1_kinetics/
    ├── stage2_mosei/
    └── stage2_accede/
```

---

## Troubleshooting

### Issue: LIRIS-ACCEDE path not found
```bash
# Solution: Ensure directory structure matches
# Check dataset path in utils/config.py
LIRIS_ACCEDE_PATH = "./dataset/Liris_Accede"
```

### Issue: Out of memory (OOM)
```bash
# Solutions:
# 1. Reduce batch size
python train_pipeline.py --all-stages --batch-size 8

# 2. Disable optical flow (reduces memory)
python train_pipeline.py --all-stages  # (without --compute-flow)

# 3. Use CPU
python train_pipeline.py --all-stages --device cpu
```

### Issue: LLM loading fails
```bash
# The system gracefully falls back to rule-based explanations
# No error, but explanations will be simpler
# To use LLM:
pip install transformers
python inference_llm.py --use-llm
```

### Issue: UCF-Crime references in old code
```bash
# Solution: All UCF-Crime references have been removed
# If you still see them, ensure files are updated:
grep -r "UCF_CRIME_PATH" scene_motion_llm/
# Should return nothing
```

---

## Performance Tips

1. **GPU Memory**: Use `--compute-flow` for better accuracy (+2-3%)
2. **Speed**: Disable `--compute-flow` for faster training
3. **Accuracy**: Larger batch size (16+) generally improves convergence
4. **Transfer Learning**: Always load pretrained motion encoder for Stage 2
5. **LLM**: Local LLM improves explanation quality but requires 16GB+ VRAM

---

## Citation & References

```bibtex
@inproceedings{cmu_mosei,
  title={CMU-MOSEI: A Multimodal Sentiment Analysis Dataset and Baseline},
  year={2018}
}

@inproceedings{liris_accede,
  title={LIRIS-ACCEDE: A Large Image and Video Emotion Database},
  year={2015}
}

@inproceedings{ucf101,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  year={2012}
}

@inproceedings{kinetics400,
  title={Quo Vadis, Action Recognition? A New Model and Large-Scale Datasets},
  year={2017}
}
```

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review log files in output directory
3. Verify dataset structure and paths
4. Check CUDA/GPU availability: `torch.cuda.is_available()`

---

**Last Updated**: June 2024  
**Version**: 2.0 (4-Step Pipeline + LLM Reasoning)
