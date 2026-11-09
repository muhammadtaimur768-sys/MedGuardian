"""Facts stated in a MedAgentBench task's own text (instruction + context).

Everything here is parsed from what the clinician wrote in the task; nothing comes from the
grader. Used by the calculator and the order verifier.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class TaskFacts:
    now: Optional[datetime] = None
    mrn: Optional[str] = None
    ndc: Optional[str] = None
    loinc: Optional[str] = None
    snomed: Optional[str] = None
    flowsheet: Optional[str] = None
    lab_codes: Dict[str, str] = field(default_factory=dict)      # analyte name -> FHIR code ("MG", "K", ...)
    window_hours: Optional[float] = None                          # "within last 24 hours"
    max_age_days: Optional[float] = None                          # "greater than 1 year old"
    mg_tiers: List[Tuple[float, float, bool, float, float]] = field(default_factory=list)  # (lo, hi, hi_inclusive, grams, hours)
    k_rule: Optional[Tuple[float, float, float]] = None           # (step mEq/L, mEq per step, goal)
    next_day_hour: Optional[int] = None                           # "next day at 8am"
    free_text: Optional[str] = None                               # quoted text to put in a referral note


def _f(x: str) -> float:
    return float(x)


def parse(case: Dict[str, str]) -> TaskFacts:
    ins, ctx = case.get("instruction", ""), case.get("context", "")
    text = ins + "\n" + ctx
    t = TaskFacts()
    m = re.search(r"It's (\S+) now", ctx)
    if m:
        t.now = datetime.fromisoformat(m.group(1).rstrip("."))
    m = re.search(r"\b(S\d{6,8})\b", ins)
    if m:
        t.mrn = m.group(1)
    m = re.search(r"NDC for [^.]*? is ([\d-]+)", text)
    if m:
        t.ndc = m.group(1)
    m = re.search(r"LOINC code for [^.:]*?(?:is|:)\s*:?\s*([\d]+-[\d])", text)
    if m:
        t.loinc = m.group(1)
    m = re.search(r"SNOMED code for [^.]*? is (\d+)", text)
    if m:
        t.snomed = m.group(1)
    m = re.search(r"flowsheet ID for [^.]*? is (\w+)", text)
    if m:
        t.flowsheet = m.group(1)
    for name, code in re.findall(r"The code for ([\w ]+?) is \"(\w+)\"", text):
        t.lab_codes[name.strip().lower()] = code
    m = re.search(r"within (?:the )?last (\d+) hours", text)
    if m:
        t.window_hours = _f(m.group(1))
    m = re.search(r"greater than (\d+) years? old", text)
    if m:
        t.max_age_days = 365.0 * _f(m.group(1))
    # Tier upper bounds are inclusive unless the task writes "<" ("1.5 to 1.9" vs "1 to <1.5").
    for lo, lt, hi, g, h in re.findall(r"magnesium ([\d.]+) to (<?)([\d.]+) mg/dL\):\s*IV: ([\d.]+) g over ([\d.]+) hour", ctx):
        t.mg_tiers.append((_f(lo), _f(hi), not lt, _f(g), _f(h)))
    for hi, g, h in re.findall(r"magnesium <([\d.]+) mg/dL\):\s*IV: ([\d.]+) g over ([\d.]+) hour", ctx):
        t.mg_tiers.append((float("-inf"), _f(hi), False, _f(g), _f(h)))
    m = re.search(r"every ([\d.]+) mEq/L.*?below threshold, order ([\d.]+) mEq.*?goal of ([\d.]+)", ctx, re.S)
    if m:
        t.k_rule = (_f(m.group(1)), _f(m.group(2)), _f(m.group(3)))
    m = re.search(r"next day at (\d{1,2})\s*am", ins, re.I)
    if m:
        t.next_day_hour = int(m.group(1))
    m = re.search(r"free text of the referral, \"(.+?)\"", ins, re.S)
    if m:
        t.free_text = m.group(1)
    return t


def next_day_at(now: datetime, hour: int) -> datetime:
    return (now + timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
