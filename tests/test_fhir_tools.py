"""FHIR tools: unit tests with mocked HTTP, plus an integration test against the mock server."""
from unittest import mock

import pytest
import requests

from medguardian.state import PatientContext
from medguardian.tools import fhir_tools


def _resp(body):
    r = mock.Mock()
    r.json.return_value = body
    r.raise_for_status.return_value = None
    return r


def test_query_patient_mocked_http_uses_medagentbench_params():
    bundle = {"entry": [{"resource": {"id": "S123", "name": [{"given": ["A"], "family": "B"}],
                                      "birthDate": "1951-01-15", "gender": "female"}}]}
    with mock.patch.object(fhir_tools.requests, "get", return_value=_resp(bundle)) as g:
        ctx = PatientContext()
        rec = fhir_tools.query_patient(ctx, base="http://x/fhir/", given="A", family="B")
    assert g.call_args[0][0] == "http://x/fhir/Patient"
    assert g.call_args[1]["params"]["_format"] == "json"
    assert rec.success and ctx.patient_id == "S123" and ctx.age_years == 72


def test_query_observation_mocked_uses_patient_param_and_latest():
    bundle = {"entry": [
        {"resource": {"valueQuantity": {"value": 1.1, "unit": "mg/dL"}, "effectiveDateTime": "2023-11-01T00:00:00"}},
        {"resource": {"valueQuantity": {"value": 1.4, "unit": "mg/dL"}, "effectiveDateTime": "2023-11-12T00:00:00"}}]}
    with mock.patch.object(fhir_tools.requests, "get", return_value=_resp(bundle)) as g:
        ctx = PatientContext(patient_id="S123")
        rec = fhir_tools.query_observation(ctx, "CREAT", base="http://x/fhir/")
    assert g.call_args[1]["params"]["patient"] == "S123"  # patient=, not subject=
    assert ctx.serum_creatinine_mg_dl == 1.4 and rec.output["count"] == 2


def test_http_error_becomes_failed_record():
    with mock.patch.object(fhir_tools.requests, "get", side_effect=requests.ConnectionError("down")):
        ctx = PatientContext()
        rec = fhir_tools.query_patient(ctx, base="http://x/fhir/", given="A")
    assert not rec.success and "down" in rec.error and ctx.tool_calls == [rec]


def test_age_on_birthday_boundary():
    assert fhir_tools.age_on("1951-11-13") == 72
    assert fhir_tools.age_on("1951-11-14") == 71


def test_against_mock_server(mock_fhir):
    ctx = PatientContext()
    fhir_tools.query_patient(ctx, base=mock_fhir, given="Amina", family="Rauf", birthdate="1951-01-15")
    fhir_tools.query_observation(ctx, "CREAT", base=mock_fhir)
    fhir_tools.query_observation(ctx, "WEIGHT", base=mock_fhir)
    assert (ctx.patient_id, ctx.age_years, ctx.sex) == ("MG1001", 72, "female")
    assert (ctx.serum_creatinine_mg_dl, ctx.weight_kg) == (1.4, 68)


def test_mock_server_no_match(mock_fhir):
    ctx = PatientContext()
    rec = fhir_tools.query_patient(ctx, base=mock_fhir, given="Nobody")
    assert rec.success and rec.output == [] and ctx.patient_id is None
