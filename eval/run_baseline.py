"""Upstream MedAgentBench baseline (unmodified agent) on a 30-task stratified subset, via Gemini.

Subcommands (run from the repo root):
  setup  - write the subset data file, copy overlay configs, generate the key-bearing agent yaml
  check  - check the Docker FHIR server and refsol.py
  start  - start the task controller + 1 task worker (blocks; leave it running)
  The assigner command is printed by `setup`/`check`.

Upstream files are never edited; we only add new files inside the submodule (they are untracked
there and are not part of this repo's commits). Python 3.9 compatible, stdlib only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAB = os.path.join(ROOT, "medagentbench")
OVERLAY = os.path.join(ROOT, "baseline", "configs")
FULL_DATA = os.path.join(MAB, "data", "medagentbench", "test_data_v2.json")
SUBSET_NAME = "medguardian_subset_30.json"
SUBSET_DATA = os.path.join(MAB, "data", "medagentbench", SUBSET_NAME)
PER_TYPE = 3
GENERATED_AGENT = os.path.join(MAB, "configs", "agents", "gemini-baseline.generated.yaml")
REFSOL = os.path.join(MAB, "src", "server", "tasks", "medagentbench", "refsol.py")
FHIR_METADATA = "http://localhost:8080/fhir/metadata"

# overlay file (relative to baseline/configs) -> destination (relative to medagentbench/configs)
OVERLAYS = {
    "tasks/medguardian_subset.yaml": "tasks/medguardian_subset.yaml",
    "assignments/medguardian_gemini.yaml": "assignments/medguardian_gemini.yaml",
    "medguardian_start_task.yaml": "medguardian_start_task.yaml",
}


def load_env(path=os.path.join(ROOT, ".env")):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("GEMINI_API_KEY", "LLM_MODEL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def task_sort_key(task_id):
    m = re.match(r"task(\d+)_(\d+)$", task_id)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 0)


def build_subset():
    with open(FULL_DATA) as f:
        data = json.load(f)
    by_type = OrderedDict()
    for rec in sorted(data, key=lambda r: task_sort_key(r["id"])):
        by_type.setdefault(rec["id"].split("_")[0], []).append(rec)
    subset = []
    for recs in by_type.values():
        subset.extend(recs[:PER_TYPE])
    with open(SUBSET_DATA, "w") as f:
        json.dump(subset, f, indent=2)
    return subset


def setup(_args):
    env = load_env()
    key, model = env.get("GEMINI_API_KEY"), env.get("LLM_MODEL")
    if not key or not model:
        sys.exit("ERROR: GEMINI_API_KEY and LLM_MODEL must be set in .env")
    if not os.path.exists(FULL_DATA):
        sys.exit("ERROR: submodule missing; run: git submodule update --init")

    subset = build_subset()
    print("wrote %s (%d tasks: %s)" % (os.path.relpath(SUBSET_DATA, ROOT), len(subset),
                                      ", ".join(r["id"] for r in subset[:: PER_TYPE])))

    for src, dst in OVERLAYS.items():
        dst_path = os.path.join(MAB, "configs", dst)
        shutil.copyfile(os.path.join(OVERLAY, src), dst_path)
        print("copied baseline/configs/%s -> medagentbench/configs/%s" % (src, dst))

    with open(os.path.join(OVERLAY, "agents", "gemini-chat.template.yaml")) as f:
        tmpl = f.read()
    with open(GENERATED_AGENT, "w") as f:
        f.write(tmpl.replace("__GEMINI_API_KEY__", key).replace("__LLM_MODEL__", model))
    os.chmod(GENERATED_AGENT, 0o600)
    print("generated %s (model=%s; key not printed; file is inside the submodule, never committed)"
          % (os.path.relpath(GENERATED_AGENT, ROOT), model))
    print_commands()


def check(_args):
    ok = True
    try:
        with urllib.request.urlopen(FHIR_METADATA, timeout=10) as r:
            print("[ok]   FHIR server at %s (HTTP %d)" % (FHIR_METADATA, r.status))
    except Exception as e:  # noqa: BLE001
        ok = False
        print("[FAIL] FHIR server not reachable (%s). Start it with:\n"
              "       docker pull jyxsu6/medagentbench:latest && "
              "docker run -p 8080:8080 jyxsu6/medagentbench:latest" % e)
    if os.path.exists(REFSOL):
        print("[ok]   refsol.py present")
    else:
        ok = False
        print("[FAIL] %s missing: download it from the Box link in medagentbench/README.md"
              % os.path.relpath(REFSOL, ROOT))
    for path in (SUBSET_DATA, GENERATED_AGENT):
        if not os.path.exists(path):
            ok = False
            print("[FAIL] %s missing: run `python3 eval/run_baseline.py setup`" % os.path.relpath(path, ROOT))
    print_commands()
    sys.exit(0 if ok else 1)


def start(args):
    """Start controller + 1 worker with our subset config (upstream start_task can't pass --config)."""
    py = args.python
    procs = [subprocess.Popen([py, "-m", "src.server.task_controller", "--port", "5000"], cwd=MAB)]
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:5000/api/list_workers", timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    else:
        procs[0].terminate()
        sys.exit("ERROR: controller did not start (are upstream requirements installed for %s?)" % py)
    procs.append(subprocess.Popen(
        [py, "-m", "src.server.task_worker", "medagentbench-subset",
         "--config", "configs/tasks/medguardian_subset.yaml",
         "--self", "http://localhost:5001/api", "--port", "5001",
         "--controller", "http://localhost:5000/api"], cwd=MAB))
    print("controller :5000 + worker :5001 running. In another terminal run the assigner (see below). Ctrl-C to stop.")
    print_commands()
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


def print_commands():
    print("""
Run the upstream baseline (needs Docker FHIR on :8080, refsol.py, upstream requirements):
  1. docker run -p 8080:8080 jyxsu6/medagentbench:latest
  2. python3 eval/run_baseline.py start          # controller + 1 worker (leave running)
  3. cd medagentbench && python -m src.assigner --config configs/assignments/medguardian_gemini.yaml
  4. python3 eval/summarize_baseline.py         # -> eval/baseline_results.md
""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("setup").set_defaults(func=setup)
    sub.add_parser("check").set_defaults(func=check)
    sp = sub.add_parser("start")
    sp.add_argument("--python", default=sys.executable, help="interpreter with upstream requirements")
    sp.set_defaults(func=start)
    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
