"""Gap 3 pipeline: screenshot OCR + dictation ASR -> template-constrained fusion.

  ocr_measurements(png)       Apple Vision OCR (ocrmac), on-screen caliper labels -> numeric fields
  transcribe(wav)             faster-whisper base.en, int8 on CPU
  extract_dictation(text)     Qwen (local) fills the template from the transcript as constrained JSON
  fuse(image, dictation)      numbers from the image win; qualitative findings from the dictation;
                              a number given by both that disagrees is flagged for the radiologist
"""
import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .templates import CATEGORICAL_FIELDS, NUMERIC_FIELDS

CONFLICT_TOLERANCE = 0.15  # same unit; values differing by more than this are a conflict


# -- image ---------------------------------------------------------------------------------
def parse_screen_lines(lines: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in lines:
        s = line.lower().replace(",", ".")
        for field, (unit, labels, _) in NUMERIC_FIELDS.items():
            for lab in labels:
                m = re.search(r"%s\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%s\b" % (lab, unit), s)
                if m and field not in out:
                    out[field] = float(m.group(1))
    return out


def ocr_measurements(png_path: str) -> Tuple[Dict[str, float], List[str]]:
    from ocrmac import ocrmac
    lines = [t for t, conf, _ in ocrmac.OCR(png_path).recognize()]
    return parse_screen_lines(lines), lines


# -- speech --------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def load_wav_16k(wav_path: str):
    """Read a 16 kHz mono 16-bit WAV into float32 [-1, 1] (avoids PyAV, whose API changed in v16)."""
    import wave

    import numpy as np
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected 16 kHz mono 16-bit WAV: %s" % wav_path)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe(wav_path: str) -> str:
    segments, _ = _whisper().transcribe(load_wav_16k(wav_path), beam_size=5, temperature=0.0, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


EXTRACT_PROMPT = """Fill a structured right-upper-quadrant ultrasound report from the radiologist's dictation.
Return ONLY a JSON object with exactly these keys; use null when the dictation does not mention a field:
{numeric}
{categorical}
Numbers must be plain numbers in the unit shown (no unit text). Do not guess values that were not dictated.

Dictation: "{text}"
"""


def extraction_prompt(text: str) -> str:
    numeric = "\n".join('- "%s": number in %s' % (f, u) for f, (u, _, _) in NUMERIC_FIELDS.items())
    categorical = "\n".join('- "%s": one of %s' % (f, json.dumps(v)) for f, v in CATEGORICAL_FIELDS.items())
    return EXTRACT_PROMPT.format(numeric=numeric, categorical=categorical, text=text)


def constrain(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only template fields with valid types/values (template-constrained output)."""
    out: Dict[str, Any] = {}
    for f in NUMERIC_FIELDS:
        v = raw.get(f)
        if isinstance(v, str):
            m = re.search(r"\d+(?:\.\d+)?", v)
            v = float(m.group(0)) if m else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f] = round(float(v), 2)
    for f, allowed in CATEGORICAL_FIELDS.items():
        v = raw.get(f)
        if isinstance(v, str) and v.strip().lower() in allowed:
            out[f] = v.strip().lower()
    return out


def extract_dictation(text: str, llm) -> Tuple[Dict[str, Any], str]:
    res = llm.chat([{"role": "user", "content": extraction_prompt(text)}])
    m = re.search(r"\{.*\}", res.content, re.S)
    try:
        raw = json.loads(m.group(0)) if m else {}
    except ValueError:
        raw = {}
    return constrain(raw if isinstance(raw, dict) else {}), res.content


# -- fusion --------------------------------------------------------------------------------
def fuse(image: Dict[str, float], dictation: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fused: Dict[str, Any] = {}
    conflicts: List[Dict[str, Any]] = []
    for f in NUMERIC_FIELDS:
        a, b = image.get(f), dictation.get(f)
        if a is not None:
            fused[f] = a
            if b is not None and abs(a - b) > CONFLICT_TOLERANCE:
                conflicts.append({"field": f, "image": a, "dictation": b,
                                  "resolution": "caliper value kept; flagged for radiologist review"})
        elif b is not None:
            fused[f] = b
    for f in CATEGORICAL_FIELDS:
        if f in dictation:
            fused[f] = dictation[f]
    return fused, conflicts
