"""Structured right-upper-quadrant ultrasound report template (the system's own knowledge).

The fusion step is constrained to these fields, units and allowed values; anything else is
dropped. Labels/synonyms here are written for the system, independently of the data generator.
"""
NUMERIC_FIELDS = {
    # field: (unit, on-screen label patterns, spoken patterns)
    "liver_span_cm": ("cm", [r"liver(?: span)?"], ["liver"]),
    "gb_wall_mm": ("mm", [r"gb wall", r"gallbladder wall"], ["gallbladder wall"]),
    "cbd_mm": ("mm", [r"cbd", r"common bile duct"], ["common bile duct", "bile duct", "cbd"]),
    "r_kidney_cm": ("cm", [r"r(?:ight)? kidney", r"rt kidney"], ["right kidney", "kidney"]),
    "spleen_cm": ("cm", [r"spleen"], ["spleen"]),
}
CATEGORICAL_FIELDS = {
    "liver_echotexture": ["normal", "increased", "coarse"],
    "gallstones": ["absent", "present"],
    "hydronephrosis": ["none", "mild", "moderate"],
    "ascites": ["absent", "present"],
}
ALL_FIELDS = list(NUMERIC_FIELDS) + list(CATEGORICAL_FIELDS)

SENTENCES = {
    "liver_echotexture": {"normal": "The liver has normal echotexture.",
                          "increased": "The liver shows increased echogenicity consistent with fatty infiltration.",
                          "coarse": "The liver echotexture is coarse."},
    "gallstones": {"absent": "No gallstones are seen.", "present": "There are multiple gallstones in the gallbladder."},
    "hydronephrosis": {"none": "There is no hydronephrosis.", "mild": "There is mild right hydronephrosis.",
                       "moderate": "There is moderate right hydronephrosis."},
    "ascites": {"absent": "No free fluid.", "present": "There is a small amount of free fluid."},
}
LABELS = {"liver_span_cm": "Liver span", "gb_wall_mm": "GB wall", "cbd_mm": "CBD", "r_kidney_cm": "R kidney",
          "spleen_cm": "Spleen"}


def render_report(fields: dict) -> str:
    """Deterministic report text from a structured record (used for BLEU against the reference)."""
    parts = []
    for f, (unit, _, _) in NUMERIC_FIELDS.items():
        if fields.get(f) is not None:
            parts.append("%s %s %s." % (LABELS[f], fields[f], unit))
    for f in CATEGORICAL_FIELDS:
        if fields.get(f) in SENTENCES[f]:
            parts.append(SENTENCES[f][fields[f]])
    return " ".join(parts)
