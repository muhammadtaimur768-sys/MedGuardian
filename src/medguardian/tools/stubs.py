"""Phase 1 interface stubs for Gap 1 guideline retrieval and Gap 2 peri-operative auditing.

Signatures follow the tool contract (patient_context, **kwargs) -> ToolCallRecord so they can be
registered with the orchestrator later. They return success=False with a clear "not implemented"
error rather than fabricated output.
"""
from ..state import PatientContext, ToolCallRecord


def guideline_rag(ctx: PatientContext, query: str, top_k: int = 3) -> ToolCallRecord:
    """Gap 1 (Phase 2): retrieve guideline/label passages that justify a dosing decision."""
    return ctx.record(ToolCallRecord(tool_name="guideline_rag", arguments={"query": query, "top_k": top_k},
                                     success=False, error="not implemented in Phase 1 (planned: label/KDIGO corpus + retriever)"))


def timeline_verifier(ctx: PatientContext, procedure_id: str, checklist: str = "WHO_surgical_safety") -> ToolCallRecord:
    """Gap 2 (Phase 2): check peri-operative events (antibiotic timing, anticoagulant holds) against a checklist."""
    return ctx.record(ToolCallRecord(tool_name="timeline_verifier",
                                     arguments={"procedure_id": procedure_id, "checklist": checklist},
                                     success=False, error="not implemented in Phase 1 (planned: FHIR Procedure/MedicationAdministration timeline audit)"))
