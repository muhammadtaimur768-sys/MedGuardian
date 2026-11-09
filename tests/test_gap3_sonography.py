"""Gap 3: screen parsing, template constraints, fusion rules, report rendering, metrics."""
import os
import sys

import pytest

from sonography import pipeline
from sonography.templates import render_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import run_e4_g3  # noqa: E402


def test_screen_lines_are_parsed_into_template_fields():
    lines = ["RUQ ABD US", "D1 R kidney 9.9 cm", "D2 Liver span 17.5 cm", "D3 CBD 4.9 mm", "D4 GB wall 4,2 mm", "MI 1.1"]
    assert pipeline.parse_screen_lines(lines) == {"r_kidney_cm": 9.9, "liver_span_cm": 17.5, "cbd_mm": 4.9,
                                                  "gb_wall_mm": 4.2}


def test_unit_must_match_the_template():
    assert pipeline.parse_screen_lines(["CBD 4.9 cm"]) == {}


def test_constrain_drops_invalid_values_and_unknown_keys():
    raw = {"cbd_mm": "6 mm", "spleen_cm": None, "gallstones": "Present", "hydronephrosis": "severe", "foo": 1,
           "liver_span_cm": True}
    assert pipeline.constrain(raw) == {"cbd_mm": 6.0, "gallstones": "present"}


def test_fusion_prefers_calipers_and_flags_disagreement():
    fused, conflicts = pipeline.fuse({"cbd_mm": 4.9, "liver_span_cm": 15.0},
                                     {"cbd_mm": 6.1, "liver_span_cm": 15.0, "spleen_cm": 11.2, "ascites": "absent"})
    assert fused == {"cbd_mm": 4.9, "liver_span_cm": 15.0, "spleen_cm": 11.2, "ascites": "absent"}
    assert [c["field"] for c in conflicts] == ["cbd_mm"]


def test_small_differences_are_not_conflicts():
    assert pipeline.fuse({"cbd_mm": 4.9}, {"cbd_mm": 5.0})[1] == []


def test_report_rendering_is_deterministic():
    r = render_report({"cbd_mm": 4.9, "gallstones": "absent"})
    assert r == "CBD 4.9 mm. No gallstones are seen."


def test_wer():
    assert run_e4_g3.wer("the liver measures 15.2 centimeters", "the liver measures 15.2 cm") == 0
    assert run_e4_g3.wer("a b c d", "a x c d") == pytest.approx(0.25)


def test_field_match_tolerance():
    assert run_e4_g3.field_match("cbd_mm", 4.93, 4.9)
    assert not run_e4_g3.field_match("cbd_mm", 5.0, 4.9)
    assert run_e4_g3.field_match("gallstones", "present", "present")
