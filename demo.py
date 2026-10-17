"""End-to-end Gap 1 demo: name/DOB -> patient -> creatinine/weight -> CrCl -> renal dose -> approve?

  python demo.py                        # mock FHIR server (started in-process), Gemini via .env
  python demo.py --case SC04            # pick a seed case
  python demo.py --fhir live            # MedAgentBench server at MEDAGENTBENCH_FHIR_BASE
  python demo.py --no-llm               # deterministic tool chain, no API key needed
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medguardian.llm_client import load_dotenv_if_present, make_client  # noqa: E402
from medguardian.orchestrator import Orchestrator, human_approval_gate  # noqa: E402
from medguardian.state import PatientContext  # noqa: E402
from medguardian.tools import dosing, fhir_tools  # noqa: E402

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "data", "seed_cases.json")


def request_text(case):
    p = case["patient"]
    return ("Patient %s %s, date of birth %s, is to start %s at a standard dose of %s mg. "
            "Check renal function and give the renal-adjusted dose." % (
                p["given"], p["family"], p["birthDate"], case["drug"], case["standard_dose_mg"]))


def scripted_chain(case, base):
    """The same tools in a fixed order, without an LLM."""
    p = case["patient"]
    ctx = PatientContext()
    fhir_tools.query_patient(ctx, base=base, given=p["given"], family=p["family"], birthdate=p["birthDate"])
    fhir_tools.query_observation(ctx, "CREAT", base=base)
    fhir_tools.query_observation(ctx, "WEIGHT", base=base)
    if ctx.weight_kg is None:  # live MedAgentBench may lack weight; fall back to case context
        ctx.weight_kg = p["weight_kg"]
    dosing.crcl_tool(ctx)
    dosing.dose_tool(ctx, drug=case["drug"], standard_dose_mg=case["standard_dose_mg"])
    return ctx


def main():
    load_dotenv_if_present()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="SC01")
    ap.add_argument("--fhir", choices=["mock", "live"], default="mock")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--yes", action="store_true", help="auto-approve (non-interactive)")
    a = ap.parse_args()

    case = next(c for c in json.load(open(CASES)) if c["case_id"] == a.case)
    if a.fhir == "mock":
        from eval.mock_fhir_server import serve
        serve(8081, background=True)
        base = "http://127.0.0.1:8081/fhir/"
    else:
        base = os.environ.get("MEDAGENTBENCH_FHIR_BASE", "http://localhost:8080/fhir/")

    print("Request:", request_text(case), "\n")
    if a.no_llm:
        ctx = scripted_chain(case, base)
    else:
        result = Orchestrator(make_client(), base).run(request_text(case))
        ctx = result["context"]
        print("status=%s latency=%ss usage=%s\n" % (result["status"], result["latency_s"], result["usage"]))
    print(ctx.trace(), "\n")
    print("Expected (hand-computed):", case["expected"], "\n")
    human_approval_gate(ctx, input_fn=(lambda _: "y") if a.yes else input)


if __name__ == "__main__":
    main()
