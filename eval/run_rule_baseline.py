"""Non-LLM baselines on the seed cases (no API key, no Docker).

B0 "no renal check": prescribes the requested standard dose as-is (what happens without grounding).
B1 "scripted tool chain": the MedGuardian tools in a fixed order (deterministic upper bound for Phase 1).
Writes eval/rule_baseline_results.md.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from demo import scripted_chain  # noqa: E402
from eval.mock_fhir_server import serve  # noqa: E402


def main():
    cases = json.load(open(os.path.join(ROOT, "eval", "data", "seed_cases.json")))
    serve(8081, background=True)
    base = "http://127.0.0.1:8081/fhir/"
    rows, b0_ok, b1_ok = [], 0, 0
    for c in cases:
        exp = c["expected"]
        b0 = float(c["standard_dose_mg"]) == float(exp["adjusted_dose_mg"]) and not exp["contraindicated"]
        ctx = scripted_chain(c, base)
        r = ctx.recommendations[0] if ctx.recommendations else None
        b1 = bool(r) and abs(r.crcl_ml_min - exp["crcl_ml_min"]) < 0.05 and r.adjusted_dose_mg == exp["adjusted_dose_mg"] \
            and r.contraindicated == exp["contraindicated"] and r.band == exp["band"]
        b0_ok += b0
        b1_ok += b1
        rows.append("| %s | %s | %s | %s / %s | %s | %s | %s |" % (
            c["case_id"], c["drug"], c["standard_dose_mg"], exp["crcl_ml_min"], exp["adjusted_dose_mg"],
            "PASS" if b0 else "FAIL", "%s / %s" % (r.crcl_ml_min, r.adjusted_dose_mg) if r else "-", "PASS" if b1 else "FAIL"))
    n = len(cases)
    md = ["# Rule-based (non-LLM) baselines on seed cases", "",
          "Run: `python eval/run_rule_baseline.py` (mock FHIR server, no API key).", "",
          "- **B0 no renal check** (requested dose given unchanged): %d/%d correct (%.0f%%)" % (b0_ok, n, 100.0 * b0_ok / n),
          "- **B1 scripted tool chain** (FHIR -> Cockcroft-Gault -> dose table, fixed order): %d/%d correct (%.0f%%)" % (b1_ok, n, 100.0 * b1_ok / n),
          "", "Ground truth is hand-computed (arithmetic in each case's `notes` in `eval/data/seed_cases.json`).",
          "B1 shares code with the tools under test, so it checks the tools against the hand-computed values; "
          "it is not an independent agent result.", "",
          "| Case | Drug | Requested mg | Expected CrCl / dose | B0 | B1 CrCl / dose | B1 |",
          "|---|---|---|---|---|---|---|"] + rows
    out = os.path.join(ROOT, "eval", "rule_baseline_results.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
