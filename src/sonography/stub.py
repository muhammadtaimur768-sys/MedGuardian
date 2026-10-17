"""Gap 3 (Module 2) interface stub: voice-driven sonography reporting.

Planned pipeline (Phase 2): audio -> speech-to-text -> structured findings -> templated report
-> clinician review. Phase 1 defines the data contract only.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SonographyFinding(BaseModel):
    organ: str
    description: str
    measurement_mm: Optional[float] = None


class SonographyReport(BaseModel):
    patient_id: str
    exam_type: str
    findings: List[SonographyFinding] = Field(default_factory=list)
    impression: Optional[str] = None
    status: str = "draft"  # draft until a clinician signs it


def transcribe(audio_path: str) -> str:
    raise NotImplementedError("Phase 2: speech-to-text (e.g. Whisper) not implemented yet")


def structure_findings(transcript: str, patient_id: str, exam_type: str) -> SonographyReport:
    raise NotImplementedError("Phase 2: LLM extraction of findings into SonographyReport not implemented yet")


def interface_summary() -> Dict[str, str]:
    return {"transcribe": "audio_path -> transcript", "structure_findings": "transcript -> SonographyReport (draft)"}
