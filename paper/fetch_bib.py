"""Build paper/references.bib from DOIs only, via doi.org content negotiation.

    python paper/fetch_bib.py

Every entry is fetched from the registrar (Crossref / DataCite), never typed by hand, so titles,
authors and venues are the official metadata and each entry carries its DOI. arXiv papers use
their DataCite DOIs (10.48550/arXiv.<id>). Failures are reported, not silently dropped.
"""
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "references.bib")

# (citation key, DOI, where it is used) -- the third field documents intent for the authors.
REFS = [
    # Phase 1 literature review (Deliverable A), carried over
    ("jiang2025medagentbench", "10.1056/AIdbp2500144", "base paper"),
    ("ferber2025oncology", "10.1038/s43018-025-00991-6", "related: clinical agents"),
    ("goodell2025calculations", "10.1038/s41746-025-01475-8", "related: tool use, Gap 1"),
    ("jin2025agentmd", "10.1038/s41467-025-64430-x", "related: clinical tool learning, Gap 1"),
    ("ong2025medsafety", "10.1016/j.xcrm.2025.102323", "related: medication safety, Gap 1"),
    ("ke2025ragfitness", "10.1038/s41746-025-01519-z", "related: RAG, Gap 2"),
    ("zaki2026anticoag", "10.1007/s00270-026-04428-0", "related: periprocedural anticoagulation, Gap 2"),
    ("gertz2026dictation", "10.1007/s00330-025-11929-y", "related: dictation to report, Gap 3"),
    ("hao2026ultrareporter", "10.1038/s41598-026-41439-w", "related: ultrasound reports, Gap 3"),
    ("ke2025periop", "10.1038/s41746-025-01858-x", "related: perioperative LLM, Gap 2"),
    # Agent methods
    ("yao2023react", "10.48550/arXiv.2210.03629", "agent loop"),
    ("wei2022cot", "10.48550/arXiv.2201.11903", "reasoning"),
    ("schick2023toolformer", "10.48550/arXiv.2302.04761", "tool use"),
    ("patil2023gorilla", "10.48550/arXiv.2305.15334", "tool/API calling"),
    ("shinn2023reflexion", "10.48550/arXiv.2303.11366", "verification / self-correction"),
    ("madaan2023selfrefine", "10.48550/arXiv.2303.17651", "verification / repair round"),
    ("lewis2020rag", "10.48550/arXiv.2005.11401", "retrieval-augmented generation"),
    ("robertson2009bm25", "10.1561/1500000019", "BM25 retrieval"),
    # Agent benchmarks
    ("liu2024agentbench", "10.48550/arXiv.2308.03688", "harness used by the base paper"),
    ("yao2024taubench", "10.48550/arXiv.2406.12045", "tool-agent reliability benchmark"),
    ("shi2024ehragent", "10.18653/v1/2024.emnlp-main.1245", "EHR agents"),
    # Models
    ("qwen2025qwen25", "10.48550/arXiv.2412.15115", "backbone model"),
    ("jiang2023mistral", "10.48550/arXiv.2310.06825", "E0 fidelity model"),
    ("singhal2023medpalm", "10.1038/s41586-023-06291-2", "medical LLMs"),
    ("ji2023hallucination", "10.1145/3571730", "hallucination / safety"),
    # Clinical and data standards
    ("bender2013fhir", "10.1109/CBMS.2013.6627810", "FHIR"),
    ("mandel2016smart", "10.1093/jamia/ocv189", "SMART on FHIR"),
    ("cockcroft1976crcl", "10.1159/000180580", "Cockcroft-Gault, Gap 1"),
    ("inker2021ckdepi", "10.1056/NEJMoa2102953", "CKD-EPI eGFR, Gap 1"),
    ("bates1995ade", "10.1001/jama.1995.03530010043033", "adverse drug events, motivation"),
    ("douketis2022accp", "10.1016/j.chest.2022.07.025", "perioperative antithrombotic guideline, Gap 2"),
    ("douketis2019pause", "10.1001/jamainternmed.2019.2431", "PAUSE trial, Gap 2"),
    ("esr2018structured", "10.1007/s13244-017-0588-8", "structured reporting, Gap 3"),
    ("radford2023whisper", "10.48550/arXiv.2212.04356", "ASR, Gap 3"),
    # Statistics
    ("mcnemar1947", "10.1007/BF02295996", "paired significance test"),
]


def fetch(doi: str) -> str:
    req = urllib.request.Request("https://doi.org/" + doi, headers={
        "Accept": "application/x-bibtex; charset=utf-8", "User-Agent": "medguardian-bib/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("no response")


# Fields the registrar's record is missing (checked against the publisher page). Only fields
# absent from the fetched entry are added; nothing fetched is overwritten.
MISSING_FIELDS = {
    "esr2018structured": {"title": "{ESR} paper on structured reporting in radiology",
                          "author": "{European Society of Radiology (ESR)}"},
}


def rekey(bib: str, key: str, doi: str) -> str:
    bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@%s{%s," % (m.group(1).lower(), key), bib, count=1)
    extra = {"doi": doi}
    extra.update(MISSING_FIELDS.get(key, {}))
    for field, value in extra.items():
        if not re.search(r"\b%s\s*=" % field, bib, re.I):
            bib = bib.rstrip().rstrip("}").rstrip().rstrip(",") + ",\n  %s = {%s}\n}" % (field, value)
    return bib


def main() -> int:
    entries, failed = [], []
    for key, doi, use in REFS:
        try:
            bib = rekey(fetch(doi), key, doi)
            title = re.search(r"\btitle\s*=\s*\{(.+?)\}\s*,", bib, re.S | re.I)
            print("ok   %-26s %s" % (key, (title.group(1) if title else "?")[:90]))
            entries.append("%% %s\n%s" % (use, bib))
        except Exception as e:  # noqa: BLE001
            print("FAIL %-26s %s (%s)" % (key, doi, e))
            failed.append(key)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/fetch_bib.py from DOIs (Crossref / DataCite). Do not edit by hand;\n"
                 "% add a DOI to REFS and re-run instead.\n\n" + "\n\n".join(entries) + "\n")
    print("\n%d entries written to %s; %d failed: %s" % (len(entries), os.path.relpath(OUT), len(failed), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
