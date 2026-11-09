"""Generate MedGuardian-Timeline: synthetic perioperative anticoagulation records (Gap 2, plan D6).

    python data/generators/timeline.py            # writes data/gap2/{dev,test}.jsonl

Each case is a FHIR-shaped bundle (Patient, Observation, MedicationAdministration, Procedure) for
one patient on one antithrombotic drug around one elective procedure. About half the cases are
clean; the rest carry 1-2 violations INJECTED BY CONSTRUCTION, and the injected list is the gold
label. The generator never calls the guardian; its interval table below is written out
separately from the guideline corpus the guardian retrieves from (same clinical source,
independent encoding), so the evaluation is not circular.

Violation types: hold_too_short, restart_too_early, preop_lab_out_of_range, missing_required_lab.
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "gap2")

DOACS = ("apixaban", "rivaroxaban", "edoxaban", "dabigatran")
DRUGS = DOACS + ("warfarin", "clopidogrel")
DOSE_EVERY_H = {"apixaban": 12, "dabigatran": 12, "rivaroxaban": 24, "edoxaban": 24, "warfarin": 24, "clopidogrel": 24}
PROCEDURES = {"low": ["Colonoscopy with polypectomy", "Cataract extraction", "Dental extraction"],
              "high": ["Total hip arthroplasty", "Laparoscopic colectomy", "Spinal laminectomy"]}
# Generator's own interval table (hours), from PAUSE / ACCP 2022.
HOLD = {"apixaban": (24, 48), "rivaroxaban": (24, 48), "edoxaban": (24, 48), "warfarin": (120, 120),
        "clopidogrel": (120, 120)}
RESUME = (24, 48)  # DOACs only
INR_MAX, INR_WITHIN = 1.5, 48
PLT_MIN, PLT_WITHIN = 50, 168
SEVERITY = {"hold_too_short": {"low": "medium", "high": "high"},
            "restart_too_early": {"low": "medium", "high": "high"},
            "preop_lab_out_of_range": "high", "missing_required_lab": "medium"}


def crcl(age, weight, scr, sex):
    v = (140 - age) * weight / (72 * scr)
    return v * 0.85 if sex == "female" else v


def hold_hours(drug, bleed, cl):
    if drug == "dabigatran":
        return (24, 48)[bleed == "high"] if cl >= 50 else (48, 96)[bleed == "high"]
    return HOLD[drug][bleed == "high"]


def iso(t):
    return t.isoformat()


def make_case(i, rng, split):
    case_id = "%s_%03d" % (split, i)
    mrn = "G2%06d" % rng.randrange(10 ** 6)
    sex = rng.choice(["male", "female"])
    age = rng.randint(50, 85)
    weight = round(rng.uniform(55, 100), 1)
    scr = round(rng.uniform(0.7, 2.4), 2)
    cl = crcl(age, weight, scr, sex)
    drug = rng.choice(DRUGS)
    bleed = rng.choice(["low", "high"])
    proc = rng.choice(PROCEDURES[bleed])
    start = datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 20))
    end = start + timedelta(hours=rng.choice([1, 2, 3]))
    birth = datetime(start.year - age, rng.randint(1, 12), rng.randint(1, 28))
    if (birth.month, birth.day) > (start.month, start.day):
        birth = birth.replace(year=birth.year - 1)  # keep the age exact at the procedure date

    applicable = ["hold_too_short", "preop_lab_out_of_range", "missing_required_lab"]
    if drug in DOACS:
        applicable.append("restart_too_early")
    violations = [] if rng.random() < 0.5 else rng.sample(applicable, rng.choice([1, 1, 2]))
    if "preop_lab_out_of_range" in violations and "missing_required_lab" in violations:
        violations.remove("missing_required_lab")

    gold = []
    hold = hold_hours(drug, bleed, cl)
    if "hold_too_short" in violations:
        last_dose = start - timedelta(hours=hold - rng.randint(4, min(20, hold - 2)))
        gold.append({"type": "hold_too_short", "item": drug, "severity": SEVERITY["hold_too_short"][bleed]})
    else:
        last_dose = start - timedelta(hours=hold + rng.randint(1, 24))
    doses = [last_dose - timedelta(hours=DOSE_EVERY_H[drug] * k) for k in range(6)]
    if drug in DOACS:
        resume = RESUME[bleed == "high"]
        if "restart_too_early" in violations:
            first_after = end + timedelta(hours=resume - rng.randint(4, 18))
            gold.append({"type": "restart_too_early", "item": drug, "severity": SEVERITY["restart_too_early"][bleed]})
        else:
            first_after = end + timedelta(hours=resume + rng.randint(0, 24))
        doses += [first_after + timedelta(hours=DOSE_EVERY_H[drug] * k) for k in range(3)]

    # Required labs: platelets for everyone, INR for warfarin.
    labs = [("PLT", "10*9/L", PLT_MIN, None, PLT_WITHIN)] + ([("INR", "", None, INR_MAX, INR_WITHIN)] if drug == "warfarin" else [])
    obs = [("CREAT", scr, "mg/dL", start - timedelta(hours=rng.randint(24, 72))),
           ("WEIGHT", weight, "kg", start - timedelta(hours=rng.randint(24, 96)))]
    lab_for_violation = rng.choice([l[0] for l in labs])
    for code, unit, lo, hi, within in labs:
        when = start - timedelta(hours=rng.randint(2, within - 2))
        if lo is not None:
            value = rng.randint(120, 350)
        else:
            value = round(rng.uniform(0.9, 1.4), 1)
        if "preop_lab_out_of_range" in violations and code == lab_for_violation:
            value = rng.randint(20, 49) if lo is not None else round(rng.uniform(1.6, 2.6), 1)
            gold.append({"type": "preop_lab_out_of_range", "item": code, "severity": SEVERITY["preop_lab_out_of_range"]})
        if "missing_required_lab" in violations and code == lab_for_violation:
            gold.append({"type": "missing_required_lab", "item": code, "severity": SEVERITY["missing_required_lab"]})
            if rng.random() < 0.5:
                continue  # never measured
            when = start - timedelta(hours=within + rng.randint(24, 240))  # measured, but too long ago
        obs.append((code, value, unit, when))

    res = [{"resourceType": "Patient", "id": mrn, "identifier": [{"value": mrn}], "gender": sex,
            "birthDate": birth.date().isoformat(), "name": [{"given": ["Test"], "family": "Patient%03d" % i}]}]
    for k, (code, value, unit, when) in enumerate(obs):
        res.append({"resourceType": "Observation", "id": "%s-obs%d" % (case_id, k), "status": "final",
                    "code": {"coding": [{"code": code}], "text": code}, "subject": {"reference": "Patient/" + mrn},
                    "effectiveDateTime": iso(when), "valueQuantity": {"value": value, "unit": unit}})
    for k, t in enumerate(sorted(doses)):
        res.append({"resourceType": "MedicationAdministration", "id": "%s-dose%d" % (case_id, k), "status": "completed",
                    "medicationCodeableConcept": {"text": drug}, "subject": {"reference": "Patient/" + mrn},
                    "effectiveDateTime": iso(t)})
    res.append({"resourceType": "Procedure", "id": "%s-proc" % case_id, "status": "completed",
                "code": {"text": proc}, "category": {"text": "%s bleeding risk" % bleed},
                "subject": {"reference": "Patient/" + mrn}, "performedPeriod": {"start": iso(start), "end": iso(end)}})
    rng.shuffle(res)  # record order carries no information
    return {"id": case_id, "drug": drug, "bleeding_risk": bleed, "crcl": round(cl, 1),
            "bundle": {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": r} for r in res]},
            "gold": gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--n-dev", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, n, seed in (("dev", args.n_dev, 100), ("test", args.n_test, 0)):
        rng = random.Random(seed)
        cases = [make_case(i, rng, split) for i in range(n)]
        with open(os.path.join(OUT_DIR, split + ".jsonl"), "w", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps(c) + "\n")
        n_viol = sum(bool(c["gold"]) for c in cases)
        print("%s: %d cases, %d with violations, %d gold labels" % (split, n, n_viol, sum(len(c["gold"]) for c in cases)))


if __name__ == "__main__":
    main()
