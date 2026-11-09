"""Bridge to the upstream MedAgentBench code in the `medagentbench/` submodule.

Upstream pieces are reused rather than re-typed so the protocol stays byte-identical:
  * the agent prompt is parsed out of upstream `__init__.py` with `ast` (that module itself
    imports the AgentBench framework, so it is never executed);
  * `utils.send_get_request` and `refsol.py` are loaded as a synthetic package, which runs only
    those two files.
"""
import ast
import contextlib
import importlib
import io
import json
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAB_DIR = os.path.join(ROOT, "medagentbench")
TASK_PKG_DIR = os.path.join(MAB_DIR, "src", "server", "tasks", "medagentbench")
DATA_FILE = os.path.join(MAB_DIR, "data", "medagentbench", "test_data_v2.json")
FUNCS_FILE = os.path.join(MAB_DIR, "data", "medagentbench", "funcs_v1.json")
DEFAULT_FHIR_BASE = "http://localhost:8080/fhir/"

# Query tasks must not POST (their graders reject any POST); action tasks write orders/records.
QUERY_CATEGORIES = ("task1", "task2", "task4", "task6", "task7")
ACTION_CATEGORIES = ("task3", "task5", "task8", "task9", "task10")
CATEGORIES = tuple("task%d" % i for i in range(1, 11))

_PKG_NAME = "_mab_upstream"


def category(task_id: str) -> str:
    return task_id.split("_")[0]


def task_kind(task_id: str) -> str:
    return "query" if category(task_id) in QUERY_CATEGORIES else "action"


def load_cases(path: str = DATA_FILE) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_funcs(path: str = FUNCS_FILE) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def upstream_prompt() -> str:
    """Return upstream's `MedAgentBench_prompt` string exactly as written in the submodule."""
    src_path = os.path.join(TASK_PKG_DIR, "__init__.py")
    with open(src_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=src_path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MedAgentBench_prompt" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("MedAgentBench_prompt not found in %s" % src_path)


def build_prompt(case: Dict[str, Any], funcs: Sequence[Dict[str, Any]], fhir_base: str) -> str:
    return upstream_prompt().format(api_base=fhir_base, functions=json.dumps(funcs),
                                    context=case["context"], question=case["instruction"])


def _package() -> types.ModuleType:
    pkg = sys.modules.get(_PKG_NAME)
    if pkg is None:
        if not os.path.isdir(TASK_PKG_DIR):
            raise RuntimeError("medagentbench submodule missing: run `git submodule update --init`")
        pkg = types.ModuleType(_PKG_NAME)
        pkg.__path__ = [TASK_PKG_DIR]
        sys.modules[_PKG_NAME] = pkg
    return pkg


def upstream_utils() -> types.ModuleType:
    _package()
    return importlib.import_module(_PKG_NAME + ".utils")


def refsol() -> types.ModuleType:
    _package()
    if not os.path.exists(os.path.join(TASK_PKG_DIR, "refsol.py")):
        raise RuntimeError("refsol.py missing: download it from the Box link in medagentbench/README.md "
                           "and save it as medagentbench/src/server/tasks/medagentbench/refsol.py")
    return importlib.import_module(_PKG_NAME + ".refsol")


def grade(case: Dict[str, Any], result: Optional[str], history: Sequence[Tuple[str, str]],
          fhir_base: str) -> Tuple[bool, str, Optional[str]]:
    """Grade one finished task with upstream refsol.

    `history` is a sequence of (role, content) with upstream roles ("user" / "agent").
    Returns (correct, captured grader stdout, error message or None). As upstream, a task
    without a result is never graded and counts as incorrect.
    """
    if result is None:
        return False, "", None
    results = SimpleNamespace(result=result,
                              history=[SimpleNamespace(role=r, content=c) for r, c in history])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = getattr(refsol(), category(case["id"]))(case, results, fhir_base)
        return ok is True, buf.getvalue(), None
    except Exception as e:  # noqa: BLE001 - a crashing grader is recorded, not fatal
        return False, buf.getvalue(), "%s: %s" % (type(e).__name__, e)
