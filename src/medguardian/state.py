"""Shared state and structured contracts for MedGuardian.

Works on Python 3.9 and on pydantic v1 (upstream MedAgentBench pin) and v2.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_dict(model: BaseModel) -> Dict[str, Any]:
    """model_dump() on pydantic v2, .dict() on v1."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class ToolCallRecord(BaseModel):
    """One tool invocation: what was asked, what came back, and whether it worked."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: str = Field(default_factory=_now)  # default_factory: one value per record


class DosingRecommendation(BaseModel):
    drug: str
    crcl_ml_min: float
    band: str
    adjusted_dose_mg: float
    frequency: str
    contraindicated: bool
    rationale: str
    source: str


class PatientContext(BaseModel):
    """Everything the agent has learned about one patient during a run."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    patient_id: Optional[str] = None  # MRN; equals the FHIR Patient.id in MedAgentBench
    name: Optional[str] = None
    birth_date: Optional[str] = None
    sex: Optional[str] = None
    age_years: Optional[int] = None
    weight_kg: Optional[float] = None
    serum_creatinine_mg_dl: Optional[float] = None
    crcl_ml_min: Optional[float] = None
    labs: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[DosingRecommendation] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    final_answer: Optional[str] = None
    requires_human_approval: bool = True
    created_at: str = Field(default_factory=_now)

    def record(self, rec: ToolCallRecord) -> ToolCallRecord:
        self.tool_calls.append(rec)
        return rec

    def trace(self) -> str:
        """Human-readable trace of the context and every tool call."""
        lines = [
            "PatientContext (session %s)" % self.session_id,
            "  patient_id=%s name=%s dob=%s sex=%s age=%s" % (
                self.patient_id, self.name, self.birth_date, self.sex, self.age_years),
            "  weight_kg=%s SCr=%s mg/dL CrCl=%s mL/min" % (
                self.weight_kg, self.serum_creatinine_mg_dl, self.crcl_ml_min),
            "Tool calls:",
        ]
        for i, c in enumerate(self.tool_calls, 1):
            status = "ok" if c.success else "ERROR: %s" % c.error
            lines.append("  %d. %s(%s) -> %s [%s, %.0f ms]" % (
                i, c.tool_name, c.arguments, _short(c.output), status, c.latency_ms or 0))
        for r in self.recommendations:
            lines.append("Recommendation: %s %s mg %s (band %s, contraindicated=%s)" % (
                r.drug, r.adjusted_dose_mg, r.frequency, r.band, r.contraindicated))
            lines.append("  rationale: %s | source: %s" % (r.rationale, r.source))
        if self.final_answer:
            lines.append("Final answer: %s" % self.final_answer)
        return "\n".join(lines)


def _short(value: Any, n: int = 160) -> str:
    s = str(value)
    return s if len(s) <= n else s[:n] + "..."
