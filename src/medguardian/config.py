"""MedGuardian component switches (plan D2). Every gap component can be turned off for ablation."""
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class MedGuardianConfig:
    # Gap 1 -- action reliability
    use_typed_tools: bool = True       # filtered/summarised FHIR results, URL normalisation, error hints
    use_calculator: bool = True        # derived facts: age, hours-before-now, in-window flags
    use_order_verifier: bool = True    # pre-POST checks + one repair round; action-format repair
    # Gap 2 -- process guardian (arms of experiments/run_e4_g2.py)
    use_timeline_verifier: bool = True  # deterministic timeline audit (off -> the LLM reads the raw record)
    use_guideline_rag: bool = True      # BM25-retrieved guideline thresholds/excerpts
    # Gap 3 -- sonography fusion (arms of experiments/run_e4_g3.py: image-only, dictation-only, fused)
    use_image_ocr: bool = True          # caliper measurements read from the screenshot
    use_dictation_asr: bool = True      # findings from Whisper transcript + template extraction

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MedGuardianConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError("unknown MedGuardian flags: %s" % sorted(unknown))
        return cls(**d)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
