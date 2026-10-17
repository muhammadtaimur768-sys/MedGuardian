"""Single-agent tool-calling orchestrator for the Gap 1 renal dosing workflow.

Loop: LLM proposes tool calls -> we execute them against PatientContext -> results go back
to the LLM -> until it calls `finish` or max_rounds is hit. The final recommendation is held
for a human approval gate (requires_human_approval=True); nothing is written to the EHR.
"""
import json
import time
from typing import Any, Callable, Dict, List, Optional

from .state import PatientContext, ToolCallRecord, to_dict
from .tools import dosing, fhir_tools

SYSTEM_PROMPT = """You are MedGuardian, a medication-safety co-pilot working on a FHIR R4 EHR.
Never guess patient data or doses: always use the tools.
Workflow for a renal dosing request:
1. query_patient with the given name and birth date to find the patient (MRN).
2. query_observation with code CREAT (serum creatinine, mg/dL) and code WEIGHT (kg).
3. cockcroft_gault_crcl with the patient's age, weight, creatinine and sex.
4. renal_adjusted_dose with the drug, CrCl and the requested standard dose.
5. finish with a short summary: CrCl, band, adjusted dose (mg) and frequency, or contraindicated.
If a tool returns an error, report it in finish instead of inventing values."""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "query_patient", "description": "Search FHIR Patient by name and/or birth date.",
        "parameters": {"type": "object", "properties": {
            "given": {"type": "string"}, "family": {"type": "string"},
            "birthdate": {"type": "string", "description": "YYYY-MM-DD"}}}}},
    {"type": "function", "function": {
        "name": "query_observation", "description": "Latest FHIR Observation for a patient and code (CREAT, WEIGHT, MG, K, GLU, A1C).",
        "parameters": {"type": "object", "properties": {
            "patient_id": {"type": "string"}, "code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "cockcroft_gault_crcl", "description": "Cockcroft-Gault creatinine clearance in mL/min (total body weight).",
        "parameters": {"type": "object", "properties": {
            "age_years": {"type": "number"}, "weight_kg": {"type": "number"},
            "scr_mg_dl": {"type": "number"}, "sex": {"type": "string", "enum": ["male", "female"]}},
            "required": ["age_years", "weight_kg", "scr_mg_dl", "sex"]}}},
    {"type": "function", "function": {
        "name": "renal_adjusted_dose", "description": "Renal dose adjustment for metformin, enoxaparin, gabapentin, levetiracetam or dabigatran.",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string"}, "crcl_ml_min": {"type": "number"},
            "standard_dose_mg": {"type": "number"}}, "required": ["drug", "crcl_ml_min", "standard_dose_mg"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Return the final answer to the clinician.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]


def build_registry(fhir_base: str) -> Dict[str, Callable[..., ToolCallRecord]]:
    return {
        "query_patient": lambda ctx, **kw: fhir_tools.query_patient(ctx, base=fhir_base, **kw),
        "query_observation": lambda ctx, **kw: fhir_tools.query_observation(ctx, base=fhir_base, **kw),
        "cockcroft_gault_crcl": dosing.crcl_tool,
        "renal_adjusted_dose": dosing.dose_tool,
    }


class Orchestrator:
    def __init__(self, llm, fhir_base: str, max_rounds: int = 10):
        self.llm = llm
        self.registry = build_registry(fhir_base)
        self.max_rounds = max_rounds

    def run(self, request: str, ctx: Optional[PatientContext] = None) -> Dict[str, Any]:
        ctx = ctx or PatientContext()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": request}]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        t0 = time.time()
        status = "max_rounds"
        for _ in range(self.max_rounds):
            resp = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            usage["llm_calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += int(resp.usage.get(k) or 0)
            calls = resp.tool_calls
            if not calls:
                ctx.final_answer = resp.content  # model answered in plain text
                status = "finished_text"
                break
            messages.append(resp.message)
            done = False
            for call in calls:
                name, args = call["name"], call["arguments"] or {}
                if name == "finish":
                    ctx.final_answer = args.get("answer", "")
                    result = {"ok": True}
                    done = True
                elif name in self.registry:
                    try:
                        rec = self.registry[name](ctx, **args)
                        result = rec.output if rec.success else {"error": rec.error}
                    except TypeError as e:  # bad/unknown arguments from the model
                        rec = ctx.record(ToolCallRecord(tool_name=name, arguments=args, success=False, error=str(e)))
                        result = {"error": str(e)}
                else:
                    ctx.record(ToolCallRecord(tool_name=name, arguments=args, success=False, error="unknown tool"))
                    result = {"error": "unknown tool %s" % name}
                messages.append({"role": "tool", "tool_call_id": call["id"] or name, "name": name,
                                 "content": json.dumps(result, default=str)})
            if done:
                status = "finished"
                break
        return {"status": status, "context": ctx, "usage": usage,
                "latency_s": round(time.time() - t0, 2),
                "trace": [to_dict(c) for c in ctx.tool_calls]}


def human_approval_gate(ctx: PatientContext, input_fn=input) -> bool:
    """Phase 1 placeholder for the human-in-the-loop gate: a CLI yes/no."""
    if not ctx.recommendations:
        print("No dosing recommendation to approve.")
        return False
    try:
        ans = input_fn("Approve this recommendation? [y/N] ")
    except EOFError:
        ans = ""
    approved = ans.strip().lower() in ("y", "yes")
    print("APPROVED by clinician (not written to EHR in Phase 1)." if approved else "Not approved; nothing recorded.")
    return approved
