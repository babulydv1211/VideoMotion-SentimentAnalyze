# SceneMotion AI — Six-Signal Heuristic Video Analyzer

SceneMotion AI is a local Streamlit demo that analyzes the first 15 seconds of a video and returns a heuristic **Negative**, **Neutral**, or **Positive** result. Its six signals (also referred to in the code as pillars) combine pretrained visual and speech models with hand-authored audio, color, motion, and editing rules.

It does **not** need project-specific training data or a project-trained checkpoint to run. It is an experimental content-cue classifier, not a calibrated emotion-recognition system, a clinical tool, or a measure of a viewer's feelings.

## What runs locally

The video is decoded and analyzed on the machine running Streamlit; the app does not send uploaded clips to an external inference API. The active components are:

| Signal | Implementation | Base weight |
|---|---|---:|
| Spatial | OpenCLIP `ViT-B-32` zero-shot image/text matching | 40% |
| Speech | Whisper `base` transcription plus a DistilBERT emotion classifier | 10% |
| Acoustic | Librosa signal-processing rules | 25% |
| Color | OpenCV HSV color rules | 5% |
| Motion | Optical flow; abstains from valence fusion | 15% |
| Temporal | Editing-style cues; abstains from valence fusion | 5% |

These are the configured contributions for usable signals. A signal that abstains
(a uniform `33% / 33% / 33%` vote) receives no weight and the remaining base
weights are renormalized. Model certainty never increases a signal's weight.

### First-run downloads and offline use

No API key is required, but a completely fresh machine may need internet access once: OpenCLIP, Whisper, and the Hugging Face DistilBERT model can download their pretrained weights when first loaded. After the required files are cached locally, normal analysis can run offline. **🗑️ Clear Cache** clears the app's in-memory cached resources; it does not remove downloaded model files from disk.

## How it works

```text
Video file
  ↓
MoviePy decodes up to 15 seconds into 16 sampled frames + a WAV audio track
  ↓
Spatial · Speech · Acoustic · Color · Motion · Temporal distributions
  ↓
Fixed-weight fusion with abstentions
  ↓
Negative / Neutral / Positive score and per-signal vote bars
```

Every signal produces a three-value distribution in the order **Negative, Neutral, Positive**. A uniform distribution (`33% / 33% / 33%`) represents no useful preference, rather than a confident Neutral result.

## The active signals

### Spatial — CLIP (40% base)

For each sampled frame, OpenCLIP compares the image to prompt groups such as “a video scene conveying sadness,” “... joy,” and “a neutral everyday scene.” The best prompt match in each class is taken per frame, then averaged across frames and normalized into the three output classes.

The prompt groups map sadness, fear, anger, and disgust to Negative; calm, surprise, and an everyday-scene prompt to Neutral; and joy, love, and happiness to Positive. These are prompt-design choices, not a claim that the app can reliably infer a person's internal emotional state.

### Speech — Whisper + DistilBERT (10% base)

Whisper `base` transcribes the audio. Before scoring the transcript, the app uses `pyspellchecker` to reject transcripts with fewer than 60% recognized alphabetic words. A Hugging Face emotion-classification pipeline then groups its emotion scores into Negative, Neutral, and Positive buckets. The speech signal returns a uniform distribution when there is no usable audio, no speech, a rejected transcript, an error, or no class bucket reaches `0.85`. During fusion, a uniform speech output is treated as an abstention and receives zero weight. `pyspellchecker` is installed with the project requirements.

### Acoustic — Librosa (25% base)

The audio-rule signal extracts RMS energy and variation, pitch statistics, zero-crossing-rate variation, spectral flux, beat tempo, and spectral centroid. It derives a heuristic negative “distress” signal and a positive “energy” signal. Harmonic/percussive source separation reduces the distress signal when audio is predominantly harmonic, which helps avoid treating tonal music exactly like noisy vocal distress.

These are audio-feature heuristics; they do not establish the meaning, cause, or emotion of a sound.

### Color — HSV (5% base)

The color signal measures darkness, brightness, and approximate red, green, blue, and yellow pixel ratios in HSV space. Its rules favor Negative for darkness/green, Positive for brightness/yellow, and move strong blue toward Neutral. It does not use the CLIP result to select a color sentiment, so the spatial signal is not counted twice. It abstains when the clip appears black-and-white or has very low hue variation, because a global grade may be a stylistic choice.

Color associations are context-dependent design heuristics, not universal sentiment facts.

### Motion — optical flow (15% base)

Optical flow can describe visual activity, but camera movement, editing, and a person's movement are not reliable evidence of valence. The motion pillar therefore explicitly abstains from sentiment fusion rather than converting activity into a negative or positive vote. It is retained as a placeholder for a separately trained activity feature.

### Temporal — editing and geometry (5% base)

Cuts, fades, and tilted geometry are filmmaking choices that cannot reliably identify a clip's sentiment. The temporal pillar explicitly abstains from sentiment fusion rather than treating a fast-cut, low-light, or tilted shot as Negative.

## Fixed-weight fusion and mixed results

For each usable signal, the app applies its configured base weight. A uniform
distribution is an explicit abstention and receives zero weight; the usable
weights are normalized to sum to one:

```text
effective_j     = 0 if p_j is uniform else base_weight_j
final_weight_j  = effective_j / sum(effective weights)
```

### What “base” and “effective” mean

**Base weight** is the configured share a signal would have when every signal
has a usable opinion. It is a design choice, not a confidence score. For
example, Speech has a 10% base weight, so a transcript can inform the result
but cannot become the whole result simply because the text model is certain.

**Effective weight** is the share actually used after the signals that abstain
are removed and the remaining configured weights are scaled to total 100%.
It is also not a confidence score or probability of being correct.

For example, in `ACCEDE00636.mp4`, Motion and Temporal both abstain. The usable
base weights total `40% + 10% + 25% + 5% = 80%`, so each usable signal is
divided by 80%:

| Signal | Base weight | Status | Effective weight |
|---|---:|---|---:|
| Spatial | 40% | used | `40 / 80 = 50%` |
| Speech | 10% | used | `10 / 80 = 12.5%` |
| Acoustic | 25% | used | `25 / 80 = 31.25%` |
| Color | 5% | used | `5 / 80 = 6.25%` |
| Motion | 15% | abstained | 0% |
| Temporal | 5% | abstained | 0% |

If all six signals produce usable, non-uniform votes, their effective weights
are exactly their base weights because the base weights already sum to 100%.

This deliberately does not use entropy or output sharpness as a reliability
score: a model being certain does not make it more accurate. The UI always
displays the model's leading class as its prediction. When that class is fewer
than 5 percentage points ahead of the runner-up, it adds a pair-specific mixed
status beneath the prediction. For example, Positive `40.5%` and Neutral
`39.2%` displays **Positive** as the prediction plus **“Mixed: Positive /
Neutral.”** The other combinations are shown the same way, such as **“Mixed:
Negative / Positive.”** When all three scores are within five points, it adds
**“Mixed: Negative / Neutral / Positive.”** A three-way fusion does not need to
exceed 50% to have a meaningful leading class.

The header calls this **“Six-signal heuristic fusion”** and notes that scores are not calibrated confidence estimates. The displayed **“Top fusion score: NN.N%”** is the largest fused class score; it is not a calibrated probability, benchmark accuracy, or guarantee that the predicted label is correct.

## Run the app

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app selects CUDA when PyTorch can use it and otherwise runs on CPU. Runtime varies with the hardware, video codec, clip length, and whether models are already cached.

In the interface:

1. Use **“Drop a video clip (max 15s analyzed)”** to upload an `mp4`, `mov`, `avi`, or `mkv` file. A longer input is accepted, but only its first 15 seconds are analyzed.
2. Click **“✨ Analyze Sentiment.”**
3. Review the **“Top fusion score: NN.N%,”** any **“Mixed / uncertain”** warning, the **“Fusion scores (not calibrated probabilities):”** any usable transcript, and **“How each signal voted:”** Each signal card shows its configured base weight and its effective weight after abstaining signals are removed.
4. Use **“🗑️ Clear Cache”** to clear cached loaded resources and release available CPU/GPU memory for the running app.

## Scope and evaluation

The base weights and rules are manually selected engineering heuristics. The source comments explicitly state that the weights are not a validated translation of Mehrabian's 7–38–55 study into video-sentiment model weights. Labels inspired by basic-emotion or valence/arousal vocabulary are an organization scheme, not proof of scientific validity for this implementation.

Before making performance claims, evaluate this exact version of `app.py` on a labeled, held-out video set that matches the intended use case. Report class balance, the labeling protocol (for example, expressed versus viewer-induced affect), accuracy, macro-F1, a confusion matrix, and failure cases. Do not infer psychological traits, mental health status, safety, violence, or intent from these outputs.

## Historical archive

`archive/` contains the earlier trainable five-pillar neural-model code and its historical tests. It is separate from the active root `app.py` demo and is not required by the quick-start workflow above.

## References

- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
- Ekman, P. (1992). *An argument for basic emotions*.
- Russell, J. A. (1980). *A circumplex model of affect*.
- Mehrabian, A., & Ferris, S. R. (1967). *Inference of attitudes from nonverbal communication in two channels*.
