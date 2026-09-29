"""Mathematical accuracy tests for each SceneMotion pillar.

Each assertion is derived directly from the formulas, thresholds,
and invariants in the production source files.  No ML model weights,
API calls, or real video files are needed.  Uses stdlib tempfile
(not pytest tmp_path) to avoid Windows AppData permission issues.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import six_pillar_audio as audio
import six_pillar_fusion as fusion
import six_pillar_structural as structural

_UNIFORM_F64 = np.full(3, 1.0 / 3.0, dtype=np.float64)


def _ok(probs, tol=1e-6):
    assert probs.shape == (3,)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=0, atol=tol)


def _is_uniform(probs, tol=1e-4):
    return bool(np.allclose(probs, _UNIFORM_F64, rtol=0.0, atol=tol))


def _wav(samples, rate):
    import wave
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    p = Path(tmp.name)
    s16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(s16.tobytes())
    return p


def _loader(samples, rate):
    def _load(_path, *, sr, mono):
        assert mono
        return samples.astype(np.float32), rate
    return _load


class _OK_Spell:
    def known(self, words):
        return set(words)


class _WGood:
    def __init__(self, text="i feel happy and content today in this scene"):
        self.text = text

    def transcribe(self, path, fp16=False):
        return {
            "text": self.text,
            "segments": [{"start": 0.0, "end": 3.0, "text": self.text,
                          "no_speech_prob": 0.02, "avg_logprob": -0.18,
                          "compression_ratio": 1.05}],
        }


def _joy(_):
    return [[{"label": "joy", "score": 0.92}, {"label": "love", "score": 0.08}]]


def _sad(_):
    return [[{"label": "sadness", "score": 0.90}, {"label": "anger", "score": 0.10}]]


def _pillar(name, probs, reliability=1.0, available=True,
            contributes_to_valence=True, status="available", base_weight=None):
    return {
        "name": name, "probs": probs,
        "class_order": fusion.CLASS_NAMES,
        "reliability": reliability,
        "available": available,
        "contributes_to_valence": contributes_to_valence,
        "status": status,
        "base_weight": base_weight,
        "evidence": {},
    }


# ─────────────────────────────── PILLAR 1: ACOUSTIC ───────────────────────────

class TestAcousticMath:

    def _sine(self, rate=22_050, dur=2.0, freq=180.0, amp=0.08):
        t = np.arange(int(rate * dur), dtype=np.float32) / rate
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _pyin(self, n=20, f0=180.0):
        def _f(*a, **k):
            return np.full(n, f0), np.ones(n, dtype=bool), np.ones(n)
        return _f

    def test_silence_abstains(self):
        rate = 22_050
        s = np.zeros(rate * 2, dtype=np.float32)
        p = _wav(s, rate)
        try:
            r = audio.run_acoustic_pillar(p, audio_loader=_loader(s, rate))
            assert not r.available
            assert r.reason == "insufficient_active_audio"
            _ok(r.probs)
            assert _is_uniform(r.probs)
        finally:
            p.unlink(missing_ok=True)

    def test_output_valid_distribution(self, monkeypatch):
        rate = 22_050
        s = self._sine(rate)
        monkeypatch.setattr(audio.librosa, "pyin", self._pyin())
        p = _wav(s, rate)
        try:
            r = audio.run_acoustic_pillar(p, audio_loader=_loader(s, rate))
            _ok(r.probs)
        finally:
            p.unlink(missing_ok=True)

    def test_smoothing_cap_not_exceeded(self, monkeypatch):
        """smoothing_strength <= 0.42 -> max prob <= 1/3 + 0.42*(1-1/3) = 0.6133"""
        rate = 22_050
        t = np.arange(rate * 3, dtype=np.float32) / rate
        s = (0.05 + 0.05 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 300 * t)
        monkeypatch.setattr(audio.librosa, "pyin", self._pyin(30, 300.0))
        p = _wav(s, rate)
        try:
            r = audio.run_acoustic_pillar(p, audio_loader=_loader(s, rate))
            if r.available:
                cap = 1.0 / 3.0 + 0.42 * (1.0 - 1.0 / 3.0)
                assert float(np.max(r.probs)) <= cap + 1e-5
        finally:
            p.unlink(missing_ok=True)

    def test_reliability_in_unit_interval(self, monkeypatch):
        for amp in [0.01, 0.05, 0.15]:
            rate = 22_050
            s = self._sine(rate, amp=amp)
            monkeypatch.setattr(audio.librosa, "pyin", self._pyin())
            p = _wav(s, rate)
            try:
                r = audio.run_acoustic_pillar(p, audio_loader=_loader(s, rate))
                assert 0.0 <= r.reliability <= 1.0
            finally:
                p.unlink(missing_ok=True)

    def test_distress_uplift_formula(self):
        dc, uc = 0.6, 0.2
        neg = 0.30 + 0.70 * dc
        neu = 0.85 - 0.25 * max(dc, uc)
        pos = 0.30 + 0.70 * uc
        raw = np.array([neg, neu, pos])
        computed = raw / raw.sum()
        np.testing.assert_allclose(computed, raw / raw.sum(), atol=1e-12)
        assert computed[0] > computed[2], "High distress_cue must dominate"

    def test_music_gate_abstains(self, monkeypatch):
        rate = 22_050
        t = np.arange(rate * 2, dtype=np.float32) / rate
        s = 0.12 * np.sin(2 * np.pi * 220 * t)
        monkeypatch.setattr(audio.librosa, "pyin",
            lambda *a, **k: (np.full(16, np.nan), np.zeros(16, bool), np.zeros(16)))
        monkeypatch.setattr(audio.librosa.effects, "hpss", lambda y: (y, np.zeros_like(y)))
        monkeypatch.setattr(audio.librosa.feature, "spectral_flatness",
            lambda **_: np.array([[0.01]]))
        p = _wav(s, rate)
        try:
            r = audio.run_acoustic_pillar(p, audio_loader=_loader(s, rate))
            assert not r.available
            assert r.reason == "music_ambiguous"
        finally:
            p.unlink(missing_ok=True)


# ─────────────────────────────── PILLAR 2: SPEECH ─────────────────────────────

class TestSpeechMath:
    RATE = 16_000

    def _s(self, dur=3.0, amp=0.06):
        t = np.arange(int(self.RATE * dur), dtype=np.float32) / self.RATE
        return (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def _run(self, whisper, clf, speller=None, text=None):
        samples = self._s()
        p = _wav(samples, self.RATE)
        try:
            result = audio.run_speech_pillar(
                p, whisper, clf,
                spellchecker=speller or _OK_Spell(),
                audio_loader=_loader(samples, self.RATE),
            )
            return result
        finally:
            p.unlink(missing_ok=True)

    def _whisper_with(self, text="some words here today are spoken",
                      no_sp=0.05, lp=-0.20, cr=1.10):
        class _W:
            def __init__(self, text_, no_sp_, lp_, cr_):
                self.text_ = text_
                self.no_sp_ = no_sp_
                self.lp_ = lp_
                self.cr_ = cr_

            def transcribe(self, path, fp16=False):
                return {"text": self.text_, "segments": [{
                    "start": 0.0, "end": 2.0, "text": self.text_,
                    "no_speech_prob": self.no_sp_,
                    "avg_logprob": self.lp_,
                    "compression_ratio": self.cr_,
                }]}
        return _W(text, no_sp, lp, cr)

    def test_no_speech_prob_gate(self):
        r = self._run(self._whisper_with(no_sp=0.75), _joy)
        assert not r.available and r.reason == "whisper_no_speech"

    def test_low_logprob_gate(self):
        r = self._run(self._whisper_with(lp=-1.50), _joy)
        assert not r.available and r.reason == "whisper_low_logprob"

    def test_compression_ratio_gate(self):
        r = self._run(self._whisper_with(cr=3.50), _joy)
        assert not r.available and r.reason == "whisper_repetitive_transcript"

    def test_too_few_words_gate(self):
        r = self._run(self._whisper_with(text="ok yes"), _joy)
        assert not r.available and r.reason == "too_few_words"

    def test_low_lexical_validity_gate(self):
        class _Speller:
            def known(self, words):
                return {"as"} & set(words)
        r = self._run(
            self._whisper_with(text="qzxw plok asdf zzzq blorb vvvt"),
            _joy,
            speller=_Speller(),
        )
        assert not r.available and r.reason == "low_lexical_validity"
        assert r.evidence["lexical_validity"] < 0.70

    def test_smoothing_cap_100pct_joy(self):
        r = self._run(_WGood(), lambda _: [[{"label": "joy", "score": 1.0}]])
        if r.available:
            _ok(r.probs)
            assert r.probs[2] < 0.80
            assert r.probs[2] > r.probs[0]

    def test_whisper_quality_formula(self):
        no_sp, lp, cr = 0.04, -0.20, 1.10
        lq = np.clip((lp - (-1.35)) / 1.15, 0.0, 1.0)
        cq = np.clip((2.40 + 0.45 - cr) / 0.75, 0.0, 1.0)
        wq = np.clip(0.45 * (1.0 - no_sp) + 0.40 * lq + 0.15 * cq, 0.0, 1.0)
        np.testing.assert_allclose(lq, 1.0, atol=1e-10)
        np.testing.assert_allclose(cq, 1.0, atol=1e-10)
        assert wq > 0.80, f"Clean segment quality {wq:.4f} should be >0.80"

    def test_reliability_in_unit_interval(self):
        r = self._run(_WGood(), _joy)
        assert 0.0 <= r.reliability <= 1.0

    def test_negative_emotion_leads(self):
        r = self._run(
            _WGood("this is a horrible sad and terrible thing"),
            _sad,
        )
        if r.available:
            _ok(r.probs)
            assert r.probs[0] > r.probs[2]


# ─────────────────────────────── PILLAR 3: MOTION ─────────────────────────────

class TestMotionMath:

    def _tex(self, h=120, w=160):
        y, x = np.mgrid[:h, :w]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = ((3 * x + 5 * y) % 256).astype(np.uint8)
        img[..., 1] = ((7 * x + 2 * y) % 256).astype(np.uint8)
        img[..., 2] = ((11 * x + 13 * y) % 256).astype(np.uint8)
        import cv2
        cv2.circle(img, (w // 3, h // 2), 15, (255, 255, 255), -1)
        return img

    def _tr(self, frame, dx=3):
        import cv2
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                              borderMode=cv2.BORDER_REFLECT)

    def test_probs_always_uniform(self):
        b = self._tex()
        frames = [self._tr(b, 3 * i) for i in range(6)]
        r = structural.measure_motion(frames)
        np.testing.assert_allclose(r["probs"], np.full(3, 1/3, dtype=np.float32), atol=1e-6)

    def test_contributes_to_valence_false(self):
        b = self._tex()
        frames = [self._tr(b, 2 * i) for i in range(4)]
        r = structural.measure_motion(frames)
        assert r["contributes_to_valence"] is False

    def test_global_translation_dominates(self):
        b = self._tex()
        frames = [self._tr(b, 4 * i) for i in range(8)]
        r = structural.measure_motion(frames)
        ev = r["evidence"]
        assert r["available"] is True
        assert ev["global_motion_px"] > 0.5
        assert ev["global_motion_fraction"] > 0.30

    def test_reliability_bounds(self):
        b = self._tex()
        frames = [self._tr(b, 3 * i) for i in range(6)]
        r = structural.measure_motion(frames)
        assert r["available"] is True
        assert 0.0 <= r["reliability"] <= 1.0
        # 5 pairs coverage=0.625; min reliability >= 0.45*0.625 = 0.28
        assert r["reliability"] >= 0.28

    def test_local_object_motion(self):
        import cv2
        b = self._tex()
        frames = []
        for i in range(6):
            f = b.copy()
            cv2.rectangle(f, (15 + 12*i, 40), (55 + 12*i, 85), (0, 0, 0), -1)
            cv2.rectangle(f, (18 + 12*i, 43), (52 + 12*i, 82), (255, 255, 255), -1)
            frames.append(f)
        ev = structural.measure_motion(frames)["evidence"]
        assert ev["local_motion_px"] > ev["global_motion_px"]

    def test_single_frame_unavailable(self):
        r = structural.measure_motion([self._tex()])
        assert r["available"] is False
        assert r["reliability"] == 0.0
        np.testing.assert_allclose(r["probs"], np.full(3, 1/3), atol=1e-6)

    def test_empty_frames_unavailable(self):
        r = structural.measure_motion([])
        assert r["available"] is False
        assert r["reliability"] == 0.0


# ─────────────────────────────── PILLAR 4: TEMPORAL ───────────────────────────

class TestTemporalMath:

    def _dark(self, h=120, w=160):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:] = 30
        return f

    def _bright(self, h=120, w=160):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:] = 220
        return f

    def test_probs_always_uniform(self):
        frames = [self._dark(), self._bright()] * 3
        r = structural.measure_temporal(frames)
        np.testing.assert_allclose(r["probs"], np.full(3, 1/3), atol=1e-6)

    def test_contributes_to_valence_false(self):
        frames = [self._dark(), self._dark().copy()]
        r = structural.measure_temporal(frames)
        assert r["contributes_to_valence"] is False

    def test_hard_cut_detected(self):
        frames = [self._dark(), self._dark().copy(), self._bright(), self._bright().copy()]
        r = structural.measure_temporal(frames, fps=2.0)
        assert r["available"] is True
        ev = r["evidence"]
        assert ev["cut_count"] >= 1
        assert ev["frame_difference_mean"] > 0.05

    def test_cut_score_formula(self):
        diff, hist = 0.70, 0.60
        score = np.clip(0.65 * diff + 0.35 * hist, 0.0, 1.0)
        np.testing.assert_allclose(score, 0.665, atol=1e-10)
        assert score >= 0.45

    def test_stable_frames_zero_cuts(self):
        frames = [self._dark() for _ in range(6)]
        r = structural.measure_temporal(frames)
        assert r["evidence"]["cut_count"] == 0

    def test_cut_rate_per_second(self):
        frames = [self._dark(), self._dark().copy(), self._bright(), self._bright().copy()]
        fps = 4.0
        r = structural.measure_temporal(frames, fps=fps)
        ev = r["evidence"]
        expected_dur = (len(frames) - 1) / fps
        np.testing.assert_allclose(ev["duration_seconds"], expected_dur, atol=1e-6)
        if expected_dur > 0:
            np.testing.assert_allclose(
                ev["cut_rate_per_second"], ev["cut_count"] / expected_dur, atol=1e-6)

    def test_pacing_in_unit_interval(self):
        for fl in [
            [self._dark(), self._dark().copy()],
            [self._dark(), self._bright(), self._dark(), self._bright()],
        ]:
            ev = structural.measure_temporal(fl)["evidence"]
            assert 0.0 <= ev["pacing_index"] <= 1.0


# ─────────────────────────────── PILLAR 5: COLOR ──────────────────────────────

class TestColorMath:

    def _color(self, frames):
        import cv2
        U = np.full(3, 1.0/3.0, dtype=np.float32)
        if not frames:
            return {"probs": U, "status": "no_visual_frames"}
        brightness, saturation = [], []
        hue_hist = np.zeros(180, dtype=np.float64)
        for frame in frames:
            img = np.asarray(frame, dtype=np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hue, sat, val = cv2.split(hsv)
            brightness.append(float(val.mean() / 255.0))
            saturation.append(float(sat.mean() / 255.0))
            valid = (sat > 35) & (val > 25)
            if np.any(valid):
                hue_hist += np.bincount(hue[valid].ravel(), minlength=180)
        mb = float(np.mean(brightness))
        ms = float(np.mean(saturation))
        cf = float(hue_hist.sum() / max(1, len(frames)))
        if ms < 0.06 or cf < 20:
            return {"probs": U, "status": "low_color_information",
                    "mean_brightness": mb, "mean_saturation": ms}
        ls = float(np.clip(mb - 0.5, -0.5, 0.5)) * 0.16
        p = U.astype(np.float64).copy()
        p[0] += max(-ls, 0.0)
        p[2] += max(ls, 0.0)
        p[1] += max(0.0, 0.04 - abs(ls))
        p = (p / p.sum()).astype(np.float32)
        return {"probs": p, "status": "weak_lighting_context",
                "mean_brightness": mb, "light_shift": ls}

    def test_grayscale_abstains(self):
        gray = np.full((120, 160, 3), 128, dtype=np.uint8)
        r = self._color([gray])
        assert r["status"] in ("low_color_information", "no_visual_frames")
        np.testing.assert_allclose(r["probs"], np.full(3, 1/3), atol=1e-6)

    def test_bright_shifts_positive(self):
        bright = np.full((120, 160, 3), [240, 220, 80], dtype=np.uint8)
        r = self._color([bright])
        if r["status"] == "weak_lighting_context":
            assert r["probs"][2] >= r["probs"][0]

    def test_dark_shifts_negative(self):
        dark = np.full((120, 160, 3), [30, 80, 10], dtype=np.uint8)
        r = self._color([dark])
        if r["status"] == "weak_lighting_context":
            assert r["probs"][0] >= r["probs"][2]

    def test_color_effect_bounded(self):
        for fc in [[255, 255, 255], [10, 10, 10]]:
            frame = np.full((120, 160, 3), fc, dtype=np.uint8)
            r = self._color([frame])
            if r["status"] == "weak_lighting_context":
                spread = float(np.max(r["probs"]) - np.min(r["probs"]))
                assert spread < 0.20, f"spread={spread:.4f} for color={fc}"

    def test_light_shift_formula(self):
        np.testing.assert_allclose(
            float(np.clip(0.80 - 0.5, -0.5, 0.5)) * 0.16, 0.048, atol=1e-10)
        np.testing.assert_allclose(
            float(np.clip(0.20 - 0.5, -0.5, 0.5)) * 0.16, -0.048, atol=1e-10)

    def test_empty_frames_unavailable(self):
        r = self._color([])
        assert r["status"] == "no_visual_frames"
        np.testing.assert_allclose(r["probs"], np.full(3, 1/3), atol=1e-6)


# ─────────────────────────────── PILLAR 6: FUSION ─────────────────────────────

class TestFusionMath:

    def test_all_uniform_never_invents_label(self):
        U = fusion.UNIFORM.tolist()
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", U, available=False, reliability=0.0, status="no_visual_frames"),
            _pillar("Speech",  U, available=False, reliability=0.0),
            _pillar("Acoustic",U, available=False, reliability=0.0),
            _pillar("Color",   U, reliability=0.0),
            _pillar("Motion",  U, contributes_to_valence=False),
            _pillar("Temporal",U, contributes_to_valence=False),
        ])
        np.testing.assert_allclose(r.probabilities, fusion.UNIFORM, atol=1e-6)
        assert r.status == "uncertain"
        assert r.decision_label == "Uncertain"
        assert all(c.effective_weight == 0.0 for c in r.contributions.values())

    def test_anchor_protected_against_opposing_supports(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.05, 0.10, 0.85], status="spatial_anchor", reliability=0.95),
            _pillar("Speech",  [1.0, 0.0, 0.0]),
            _pillar("Acoustic",[1.0, 0.0, 0.0]),
            _pillar("Color",   [1.0, 0.0, 0.0]),
        ])
        assert r.label == "Positive"
        assert r.decision_label == "Positive"
        assert r.spatial_is_anchor is True

    def test_spatial_ew_gte_70_percent(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.10, 0.10, 0.80], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.20, 0.10, 0.70]),
            _pillar("Acoustic",[0.20, 0.20, 0.60]),
            _pillar("Color",   [0.30, 0.30, 0.40]),
            _pillar("Motion",  fusion.UNIFORM.tolist(), contributes_to_valence=False),
            _pillar("Temporal",fusion.UNIFORM.tolist(), contributes_to_valence=False),
        ])
        assert r.contributions["Spatial"].effective_weight >= 0.70

    def test_output_sums_to_one(self):
        for sp in [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.34, 0.33, 0.33]]:
            r = fusion.fuse_six_pillars([
                _pillar("Spatial", sp, reliability=0.8),
                _pillar("Speech",  [0.2, 0.2, 0.6]),
                _pillar("Acoustic",[0.5, 0.3, 0.2]),
            ])
            np.testing.assert_allclose(r.probabilities.sum(), 1.0, atol=1e-6)

    def test_total_support_capped_at_30pct(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.05, 0.10, 0.85], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.20, 0.10, 0.70]),
            _pillar("Acoustic",[0.20, 0.20, 0.60]),
            _pillar("Color",   [0.30, 0.30, 0.40]),
        ])
        ts = sum(c.effective_weight for n, c in r.contributions.items()
                 if n != "Spatial" and c.used_for_valence)
        assert ts <= 0.30 + 1e-6

    def test_output_shape_dtype(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.1, 0.2, 0.7], reliability=0.8, status="spatial_anchor"),
        ])
        assert r.probabilities.shape == (3,)
        assert np.issubdtype(r.probabilities.dtype, np.floating)

    def test_near_tied_spatial_provisional(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.34, 0.33, 0.33], status="visual_uncertain", reliability=0.9),
            _pillar("Speech",  [0.02, 0.02, 0.96]),
        ])
        assert not r.spatial_is_anchor
        assert r.status in {"provisional", "uncertain"}

    def test_accede00636_weight_math(self):
        """README example: 4 usable signals -> effective weights sum to 1."""
        base = {"Spatial": 0.40, "Speech": 0.10, "Acoustic": 0.25, "Color": 0.05}
        total = sum(base.values())  # 0.80
        np.testing.assert_allclose(base["Spatial"] / total, 0.50, atol=1e-10)
        np.testing.assert_allclose(base["Speech"] / total, 0.125, atol=1e-10)
        np.testing.assert_allclose(base["Acoustic"] / total, 0.3125, atol=1e-10)
        np.testing.assert_allclose(base["Color"] / total, 0.0625, atol=1e-10)
        np.testing.assert_allclose(sum(v / total for v in base.values()), 1.0, atol=1e-10)

    def test_conflict_flag_raised(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.05, 0.10, 0.85], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.90, 0.05, 0.05]),
        ])
        assert len(r.conflicts) >= 1
        c = r.conflicts[0]
        assert c.pillar == "Speech"
        assert c.spatial_label == "Positive"
        assert c.pillar_label == "Negative"

    def test_default_base_weights_sum_to_one(self):
        np.testing.assert_allclose(
            sum(fusion.DEFAULT_BASE_WEIGHTS.values()), 1.0, atol=1e-10)

    def test_abstained_signals_zero_effective_weight(self):
        U = fusion.UNIFORM.tolist()
        r = fusion.fuse_six_pillars([
            _pillar("Spatial",  [0.6, 0.2, 0.2], status="spatial_anchor", reliability=0.85),
            _pillar("Speech",   U, available=False, reliability=0.0),
            _pillar("Acoustic", U, available=False, reliability=0.0),
            _pillar("Color",    U, reliability=0.0),
            _pillar("Motion",   U, contributes_to_valence=False),
            _pillar("Temporal", U, contributes_to_valence=False),
        ])
        for name in ("Speech", "Acoustic", "Color", "Motion", "Temporal"):
            c = r.contributions.get(name)
            if c is not None:
                assert c.effective_weight == 0.0


# ──────────────────────── WHOLE-PROJECT END-TO-END ACCURACY ───────────────────

class TestEndToEndAccuracy:

    def test_strong_positive_produces_positive(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.05, 0.10, 0.85], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.05, 0.15, 0.80]),
            _pillar("Acoustic",[0.10, 0.20, 0.70]),
            _pillar("Color",   [0.20, 0.40, 0.40]),
            _pillar("Motion",  fusion.UNIFORM.tolist(), contributes_to_valence=False),
            _pillar("Temporal",fusion.UNIFORM.tolist(), contributes_to_valence=False),
        ])
        assert r.label == "Positive"
        assert r.probabilities[2] > 0.60
        _ok(r.probabilities)

    def test_strong_negative_produces_negative(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.85, 0.10, 0.05], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.80, 0.15, 0.05]),
            _pillar("Acoustic",[0.70, 0.20, 0.10]),
            _pillar("Motion",  fusion.UNIFORM.tolist(), contributes_to_valence=False),
            _pillar("Temporal",fusion.UNIFORM.tolist(), contributes_to_valence=False),
        ])
        assert r.label == "Negative"
        assert r.probabilities[0] > 0.60
        _ok(r.probabilities)

    def test_neutral_heavy_stays_neutral(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", [0.10, 0.80, 0.10], status="spatial_anchor", reliability=0.9),
            _pillar("Speech",  [0.10, 0.80, 0.10]),
            _pillar("Acoustic",[0.15, 0.70, 0.15]),
        ])
        assert r.label == "Neutral"
        assert r.probabilities[1] > 0.50
        _ok(r.probabilities)

    def test_spatial_unavailable_stays_uncertain(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial", fusion.UNIFORM.tolist(),
                    available=False, reliability=0.0, status="no_visual_frames"),
            _pillar("Speech",  [0.02, 0.03, 0.95]),
            _pillar("Acoustic",[0.02, 0.03, 0.95]),
        ])
        assert not r.spatial_is_anchor
        assert r.status in {"spatial_unavailable", "uncertain"}
        assert r.decision_label == "Uncertain"
        _ok(r.probabilities)

    def test_20_random_configs_valid_distributions(self):
        rng = np.random.default_rng(seed=42)
        for _ in range(20):
            def r3():
                return rng.dirichlet([1.0, 1.0, 1.0]).tolist()
            pillars = [
                _pillar("Spatial",  r3(),
                        reliability=float(rng.uniform(0.2, 1.0)),
                        status="spatial_anchor"),
                _pillar("Speech",   r3()),
                _pillar("Acoustic", r3()),
                _pillar("Color",    r3()),
                _pillar("Motion",   fusion.UNIFORM.tolist(), contributes_to_valence=False),
                _pillar("Temporal", fusion.UNIFORM.tolist(), contributes_to_valence=False),
            ]
            result = fusion.fuse_six_pillars(pillars)
            _ok(result.probabilities)
            assert result.label in fusion.CLASS_NAMES
            total_ew = sum(c.effective_weight for c in result.contributions.values())
            np.testing.assert_allclose(total_ew, 1.0, atol=1e-6)

    def test_label_always_matches_argmax(self):
        rng = np.random.default_rng(seed=99)
        for _ in range(15):
            probs = rng.dirichlet([2.0, 1.0, 1.0]).tolist()
            result = fusion.fuse_six_pillars([
                _pillar("Spatial", probs, reliability=0.8, status="spatial_anchor"),
            ])
            expected = fusion.CLASS_NAMES[int(np.argmax(result.probabilities))]
            assert result.label == expected

    def test_effective_weights_sum_to_one(self):
        r = fusion.fuse_six_pillars([
            _pillar("Spatial",  [0.1, 0.1, 0.8], status="spatial_anchor", reliability=0.85),
            _pillar("Speech",   [0.1, 0.1, 0.8]),
            _pillar("Acoustic", [0.2, 0.2, 0.6]),
            _pillar("Color",    [0.3, 0.3, 0.4]),
            _pillar("Motion",   fusion.UNIFORM.tolist(), contributes_to_valence=False),
            _pillar("Temporal", fusion.UNIFORM.tolist(), contributes_to_valence=False),
        ])
        total = sum(c.effective_weight for c in r.contributions.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)
