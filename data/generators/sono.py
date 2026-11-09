"""Generate MedGuardian-Sono: synthetic right-upper-quadrant ultrasound cases (Gap 3).

    python data/generators/sono.py          # writes data/gap3/{dev,test}.jsonl + images/ + audio/

Each case has a ground-truth structured record, from which we produce:
  * an ultrasound-style screenshot with on-screen caliper measurements (a subset of the numeric fields);
  * a radiologist dictation script covering the qualitative findings, some of the numbers (partial
    overlap with the screen) and sometimes a number that is NOT on screen; in ~1/3 of cases one
    dictated number conflicts with the screen (a mis-spoken value);
  * speech audio of the dictation via macOS `say` (16 kHz mono WAV via ffmpeg).
Gold: numbers on screen come from the calipers (the screen wins a conflict); numbers only dictated
come from the dictation; qualitative findings come from the dictation; conflicts are listed.
"""
import argparse
import json
import os
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "gap3")

NUMERIC = {  # field: (screen label, spoken name, unit, spoken unit, low, high, decimals)
    "liver_span_cm": ("Liver span", "the liver measures", "cm", "centimeters", 12.0, 18.5, 1),
    "gb_wall_mm": ("GB wall", "the gallbladder wall measures", "mm", "millimeters", 1.5, 5.0, 1),
    "cbd_mm": ("CBD", "the common bile duct measures", "mm", "millimeters", 2.0, 9.0, 1),
    "r_kidney_cm": ("R kidney", "the right kidney measures", "cm", "centimeters", 9.0, 12.5, 1),
    "spleen_cm": ("Spleen", "the spleen measures", "cm", "centimeters", 8.0, 14.5, 1),
}
QUALITATIVE = {
    "liver_echotexture": {"normal": "The liver has normal echotexture.",
                          "increased": "The liver shows increased echogenicity consistent with fatty infiltration.",
                          "coarse": "The liver echotexture is coarse."},
    "gallstones": {"absent": "No gallstones are seen.", "present": "There are multiple gallstones in the gallbladder."},
    "hydronephrosis": {"none": "There is no hydronephrosis.", "mild": "There is mild right hydronephrosis.",
                       "moderate": "There is moderate right hydronephrosis."},
    "ascites": {"absent": "No free fluid.", "present": "There is a small amount of free fluid."},
}
VOICES = ["Samantha", "Daniel", "Karen", "Moira"]


def report_text(fields):
    """Reference report rendered from a structured record (also used for fused output -> BLEU)."""
    parts = []
    for f, (label, _, unit, _, *_r) in NUMERIC.items():
        if fields.get(f) is not None:
            parts.append("%s %s %s." % (label, fields[f], unit))
    for f, opts in QUALITATIVE.items():
        if fields.get(f) in opts:
            parts.append(opts[fields[f]])
    return " ".join(parts)


def render(path, screen, rng):
    w, h = 900, 640
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, 40
    ang = np.degrees(np.arctan2(xx - cx, yy - cy))
    r = np.hypot(xx - cx, yy - cy)
    fan = (np.abs(ang) < 38) & (r > 40) & (r < 560)
    speckle = rng.gamma(2.0, 30.0, size=(h, w)) * fan
    speckle *= np.clip(1.2 - r / 700.0, 0.2, 1.0)
    img = Image.fromarray(np.clip(speckle, 0, 255).astype("uint8"), "L").filter(ImageFilter.GaussianBlur(1.2)).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    d.text((14, 10), "RUQ  ABD US   C5-1   12 Hz", fill=(200, 200, 200), font=small)
    d.text((w - 220, 10), "MI 1.1  TIS 0.4", fill=(200, 200, 200), font=small)
    for k, (f, v) in enumerate(screen):
        label, _, unit, *_ = NUMERIC[f]
        # caliper marks on the image
        x0, y0 = int(rng.integers(250, 600)), int(rng.integers(150, 480))
        x1 = x0 + int(rng.integers(60, 160))
        d.line([(x0, y0), (x1, y0)], fill=(255, 255, 0), width=1)
        d.text((x0 - 6, y0 - 12), "+", fill=(255, 255, 0), font=small)
        d.text((x1 - 6, y0 - 12), "+", fill=(255, 255, 0), font=small)
        d.text((16, h - 34 * (len(screen) - k) - 8), "D%d  %s  %s %s" % (k + 1, label, v, unit),
               fill=(255, 255, 0), font=font)
    img.save(path)


def speak(text, voice, wav_path):
    aiff = wav_path[:-4] + ".aiff"
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", wav_path], check=True)
    os.remove(aiff)


def make_case(i, rng, split, nprng):
    cid = "%s_%03d" % (split, i)
    truth = {}
    for f, (_, _, _, _, lo, hi, dec) in NUMERIC.items():
        truth[f] = round(rng.uniform(lo, hi), dec)
    for f, opts in QUALITATIVE.items():
        truth[f] = rng.choice(list(opts))
    nums = list(NUMERIC)
    rng.shuffle(nums)
    on_screen = nums[:rng.randint(3, 4)]
    off_screen = nums[len(on_screen):]
    dict_only = off_screen[:rng.randint(0, 1)]           # a number only the dictation gives
    overlap = rng.sample(on_screen, rng.randint(1, 2))   # numbers both give
    conflict = rng.choice(overlap) if rng.random() < 0.35 else None

    gold = {f: truth[f] for f in on_screen + dict_only}
    gold.update({f: truth[f] for f in QUALITATIVE})
    sentences = ["Right upper quadrant ultrasound."]
    spoken = {}
    for f in overlap + dict_only:
        _, phrase, _, sunit, lo, hi, _ = NUMERIC[f]
        v = truth[f]
        if f == conflict:
            v = round(v + rng.choice([-1, 1]) * rng.uniform(0.6, 1.5), 1)
            v = round(min(max(v, lo - 1), hi + 1), 1)
        spoken[f] = v
        sentences.append("%s %s %s." % (phrase[0].upper() + phrase[1:], v, sunit))
    q = list(QUALITATIVE)
    rng.shuffle(q)
    for f in q:
        sentences.append(QUALITATIVE[f][truth[f]])
    script = " ".join(sentences)

    img = os.path.join(OUT, "images", cid + ".png")
    wav = os.path.join(OUT, "audio", cid + ".wav")
    render(img, [(f, truth[f]) for f in on_screen], nprng)
    voice = rng.choice(VOICES)
    speak(script, voice, wav)
    return {"id": cid, "image": os.path.relpath(img, ROOT), "audio": os.path.relpath(wav, ROOT), "voice": voice,
            "script": script, "screen_fields": on_screen, "dictated_numbers": spoken,
            "conflicts": [conflict] if conflict else [], "gold": gold, "reference_report": report_text(gold)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-dev", type=int, default=5)
    args = ap.parse_args()
    for sub in ("images", "audio"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    for split, n, seed in (("dev", args.n_dev, 100), ("test", args.n_test, 0)):
        rng, nprng = random.Random(seed), np.random.default_rng(seed)
        cases = [make_case(i, rng, split, nprng) for i in range(n)]
        with open(os.path.join(OUT, split + ".jsonl"), "w") as fh:
            for c in cases:
                fh.write(json.dumps(c) + "\n")
        print("%s: %d cases, %d with a conflict" % (split, n, sum(bool(c["conflicts"]) for c in cases)))


if __name__ == "__main__":
    main()
