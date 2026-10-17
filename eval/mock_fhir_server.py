"""Docker-free mock FHIR R4 server (stdlib only) for offline tests and the demo.

Serves synthetic patients built from eval/data/seed_cases.json on port 8081:
  GET /fhir/Patient?given=&family=&birthdate=&identifier=
  GET /fhir/Observation?patient=<MRN>&code=CREAT|WEIGHT
Usage: python eval/mock_fhir_server.py [--port 8081]
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seed_cases.json")


def build_resources(path: str = DATA):
    cases = json.load(open(path))
    patients, observations = [], []
    for c in cases:
        p = c["patient"]
        patients.append({
            "resourceType": "Patient", "id": p["id"],
            "identifier": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0203", "value": p["id"]}],
            "name": [{"use": "official", "given": [p["given"]], "family": p["family"]}],
            "gender": p["gender"], "birthDate": p["birthDate"],
        })
        for code, value, unit, when in (("CREAT", p["creat_mg_dl"], "mg/dL", "2023-11-12T07:00:00+00:00"),
                                        ("WEIGHT", p["weight_kg"], "kg", "2023-11-10T09:00:00+00:00")):
            observations.append({
                "resourceType": "Observation", "id": "%s-%s" % (p["id"], code.lower()), "status": "final",
                "code": {"coding": [{"system": "http://loinc.org" if code == "CREAT" else "http://snomed.info/sct",
                                     "code": code, "display": code}], "text": code},
                "subject": {"reference": "Patient/%s" % p["id"]},
                "effectiveDateTime": when, "valueQuantity": {"value": value, "unit": unit},
            })
    return patients, observations


def _bundle(resources):
    return {"resourceType": "Bundle", "type": "searchset", "total": len(resources),
            "entry": [{"resource": r} for r in resources]}


def make_handler(patients, observations):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/fhir+json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            res = u.path.rstrip("/").split("/")[-1]
            if res == "metadata":
                return self._send(200, {"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"})
            if res == "Patient":
                out = [p for p in patients
                       if ("given" not in q or q["given"].lower() in [g.lower() for g in p["name"][0]["given"]])
                       and ("family" not in q or q["family"].lower() == p["name"][0]["family"].lower())
                       and ("birthdate" not in q or q["birthdate"] == p["birthDate"])
                       and ("identifier" not in q or q["identifier"] == p["id"])
                       and ("_id" not in q or q["_id"] == p["id"])]
                return self._send(200, _bundle(out))
            if res == "Observation":
                pid = q.get("patient", "").replace("Patient/", "")
                out = [o for o in observations
                       if o["subject"]["reference"] == "Patient/" + pid
                       and ("code" not in q or q["code"].upper() == o["code"]["coding"][0]["code"])]
                return self._send(200, _bundle(out))
            return self._send(404, {"resourceType": "OperationOutcome", "issue": [{"diagnostics": "unsupported: " + u.path}]})

    return Handler


def serve(port: int = 8081, background: bool = False):
    patients, observations = build_resources()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(patients, observations))
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    print("Mock FHIR server on http://127.0.0.1:%d/fhir/ (%d patients)" % (port, len(patients)))
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    serve(ap.parse_args().port)
