"""Gap 1 typed FHIR tool layer (plan D4) and calculator (derived facts).

The base-paper agent receives raw FHIR bundles (often several thousand tokens, sometimes paged).
With `use_typed_tools` the agent instead receives every page, filtered to the clinically relevant
fields, newest first, and malformed searches are normalised or answered with a concrete hint.
With `use_calculator` deterministic derived facts are added (age as of "now", hours before now,
in-window statistics), so the model does not have to do date arithmetic itself.
"""
import json
import math
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .task_context import TaskFacts

MAX_PAGES = 20
MAX_ROWS = 30  # resources shown to the model; the calculator always uses every resource


def repaired_code_url(url: str, facts: "TaskFacts") -> Optional[Tuple[str, str]]:
    """If an Observation search uses the task's ORDERING code (e.g. LOINC 2823-3) or another code the
    EHR does not use, return the same search with the lab code the task text names, plus a note."""
    if "/Observation" not in url or len(facts.lab_codes) != 1:
        return None
    parts = urlsplit(url)
    params = parse_qsl(parts.query.replace("+", "%2B"), keep_blank_values=True)
    (name, lab), = facts.lab_codes.items()
    code = dict(params).get("code")
    if not code or code == lab:
        return None
    fixed = [(k, lab if k == "code" else v) for k, v in params]
    note = 'code=%s returned nothing; searched code=%s, which the task text gives as the code for %s' % (code, lab, name)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(fixed, safe=":/"), "")), note


# -- URL handling --------------------------------------------------------------------------
def _iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def normalise_url(url: str, drop_date_filters: bool = False) -> Tuple[str, List[str]]:
    """Fix common search mistakes. Returns (url, notes describing each fix).

    * `_sort` / `_count` are dropped: the typed tool returns every page, newest first.
    * a full name in `name=` is split into `given=` + `family=` (HAPI matches name parts).
    * malformed date ranges are rewritten; with `drop_date_filters` (calculator on) date filters are
      removed entirely, because the calculator computes the task's window from its own "now".
    """
    parts = urlsplit(url.strip())
    # A literal '+' in a timezone offset (+00:00) would otherwise be decoded as a space.
    params = parse_qsl(parts.query.replace("+", "%2B"), keep_blank_values=True)
    out, notes = [], []
    for k, v in params:
        if k == "_format":
            continue  # re-added exactly once below
        if k in ("_sort", "_count"):
            notes.append("dropped %s=%s (all results are returned, newest first)" % (k, v))
            continue
        if k == "name":
            words = [w for w in re.split(r"(?:%2B|\+|\s)+", v) if w]
            if len(words) >= 2:
                out += [("given", words[0]), ("family", words[-1])]
                notes.append("split name '%s' into given/family" % " ".join(words))
                continue
        if k in ("date", "_date") and drop_date_filters:
            notes.append("removed date filter %s=%s (see calculator for the task's time window)" % (k, v))
            continue
        if k == "date" and ("/" in v or ".." in v):
            lo, hi = re.split(r"/|\.\.", v, maxsplit=1)
            if _iso(lo) and _iso(hi):
                out += [("date", "ge" + lo), ("date", "le" + hi)]
                notes.append("rewrote date range '%s' as date=ge..&date=le.." % v)
                continue
        out.append((k, v))
    out.append(("_format", "json"))
    fixed = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(out, safe=":/"), ""))
    return fixed, notes


def empty_result_hint(url: str, facts: TaskFacts) -> Optional[str]:
    """When an Observation search finds nothing, point to the lab code the task text gives."""
    if "/Observation" not in url or not facts.lab_codes:
        return None
    q = dict(parse_qsl(urlsplit(url).query.replace("+", "%2B")))
    code = q.get("code")
    known = facts.lab_codes
    if code and code not in known.values():
        names = ", ".join('%s is "%s"' % (n, c) for n, c in known.items())
        return ("No results for code=%s. The task text says the observation code for %s; codes such as "
                "LOINC are for ordering, not searching." % (code, names))
    return None


def error_hint(url: str, error: str) -> str:
    hint = ""
    if "400" in error and "date=" in url:
        hint = (" FHIR date search uses prefixes, e.g. date=ge2023-11-12T10:15:00+00:00 "
                "(and optionally a second date=le... for an upper bound).")
    elif "400" in error:
        hint = " Check the parameter names against the function list (e.g. patient=, code=)."
    return "Error in sending the GET request: %s.%s" % (error, hint)


# -- fetching ------------------------------------------------------------------------------
def fetch_all(get: Callable[[str], Tuple[Dict[str, Any], bool]], url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch a search and every `next` page. Returns (merged bundle, error)."""
    merged: Optional[Dict[str, Any]] = None
    for _ in range(MAX_PAGES):
        res, _ = get(url)
        if "data" not in res:
            return merged, res.get("error", "unknown error")
        data = res["data"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return None, "non-JSON response"
        if merged is None:
            merged = data
            merged.setdefault("entry", [])
        else:
            merged["entry"].extend(data.get("entry", []))
        nxt = next((l["url"] for l in data.get("link", []) if l.get("relation") == "next"), None)
        if not nxt:
            break
        url = nxt if "_format=" in nxt else nxt + "&_format=json"
    return merged, None


# -- summarising ---------------------------------------------------------------------------
def _coding(cc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cc = cc or {}
    c = (cc.get("coding") or [{}])[0]
    return {k: v for k, v in {"code": c.get("code"), "display": c.get("display"),
                              "system": c.get("system"), "text": cc.get("text")}.items() if v}


def _age(birth: str, now: datetime) -> Optional[int]:
    b = _iso(birth)
    if not b:
        return None
    return now.year - b.year - ((now.month, now.day) < (b.month, b.day))


def _obs_value(r: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    if "valueQuantity" in r:
        return r["valueQuantity"].get("value"), r["valueQuantity"].get("unit")
    if "valueString" in r:
        return r["valueString"], None
    return None, None


def summarise(bundle: Dict[str, Any], facts: TaskFacts, use_calculator: bool) -> Dict[str, Any]:
    now = facts.now
    rows: List[Dict[str, Any]] = []
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        rt = r.get("resourceType")
        if rt == "Patient":
            name = (r.get("name") or [{}])[0]
            row = {"resourceType": rt, "id": r.get("id"),
                   "mrn": next((i.get("value") for i in r.get("identifier", [])), None),
                   "name": " ".join((name.get("given") or []) + [name.get("family", "")]).strip(),
                   "birthDate": r.get("birthDate"), "gender": r.get("gender")}
            if use_calculator and now and r.get("birthDate"):
                row["age_years_now"] = _age(r["birthDate"], now)
        elif rt == "Observation":
            v, u = _obs_value(r)
            row = {"resourceType": rt, "id": r.get("id"), "code": _coding(r.get("code")).get("code"),
                   "value": v, "unit": u, "effectiveDateTime": r.get("effectiveDateTime")}
            t = _iso(r.get("effectiveDateTime") or "")
            if use_calculator and now and t:
                row["hours_before_now"] = round((now - t).total_seconds() / 3600, 2)
        elif rt == "MedicationRequest":
            dose = ((r.get("dosageInstruction") or [{}])[0])
            row = {"resourceType": rt, "id": r.get("id"), "status": r.get("status"),
                   "medication": _coding(r.get("medicationCodeableConcept")),
                   "authoredOn": r.get("authoredOn"), "route": (dose.get("route") or {}).get("text"),
                   "dose": ((dose.get("doseAndRate") or [{}])[0]).get("doseQuantity")}
        elif rt in ("Procedure", "Condition", "ServiceRequest"):
            row = {"resourceType": rt, "id": r.get("id"), "code": _coding(r.get("code")),
                   "date": r.get("performedDateTime") or r.get("onsetDateTime") or r.get("recordedDate")
                   or r.get("authoredOn"), "status": r.get("status")}
        else:
            row = {"resourceType": rt, "id": r.get("id")}
        rows.append({k: v for k, v in row.items() if v not in (None, "", {})})

    def when(row):
        d = _iso(row.get("effectiveDateTime") or row.get("authoredOn") or row.get("date") or "")
        return d.timestamp() if d else -math.inf

    rows.sort(key=when, reverse=True)
    out: Dict[str, Any] = {"total": len(rows), "resources_newest_first": rows[:MAX_ROWS]}
    if len(rows) > MAX_ROWS:
        out["note"] = "showing the newest %d of %d resources; calculator values below cover all %d" % (
            MAX_ROWS, len(rows), len(rows))
    obs = [r for r in rows if r["resourceType"] == "Observation" and isinstance(r.get("value"), (int, float))]
    if use_calculator and obs:
        stats: Dict[str, Any] = {"latest_value": obs[0]["value"], "latest_time": obs[0].get("effectiveDateTime")}
        if facts.window_hours and "hours_before_now" in obs[0]:
            inw = [r["value"] for r in obs if 0 <= r.get("hours_before_now", math.inf) <= facts.window_hours]
            stats["window_hours"] = facts.window_hours
            stats["count_in_window"] = len(inw)
            stats["latest_in_window"] = inw[0] if inw else None
            stats["mean_in_window"] = round(sum(inw) / len(inw), 4) if inw else None
        out["calculator"] = stats
    return out


def latest_observation(get, fhir_base: str, mrn: str, code: str) -> Optional[Dict[str, Any]]:
    """Deterministic lookup used by the order verifier."""
    bundle, err = fetch_all(get, "%sObservation?patient=%s&code=%s&_count=200&_format=json" % (fhir_base, mrn, code))
    if err or not bundle:
        return None
    best = None
    for e in bundle.get("entry", []):
        r = e["resource"]
        t = _iso(r.get("effectiveDateTime") or "")
        v, _ = _obs_value(r)
        if t and isinstance(v, (int, float)) and (best is None or t > best["time"]):
            best = {"time": t, "value": v}
    return best
