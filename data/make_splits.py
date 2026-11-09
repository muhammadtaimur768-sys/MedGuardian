"""Write fixed, stratified MedAgentBench task splits (plan D8).

    python data/make_splits.py --seed 0

Each of the 10 categories (task1..task10, 30 tasks each) is shuffled with its own seeded RNG;
the first DEV_PER_CAT ids go to dev and the next TEST_PER_CAT to test, so the splits are
disjoint and every category is equally represented. Prompt, threshold and repair-rule tuning
uses dev only; test is touched once per reported configuration.
"""
import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.medguardian import mab  # noqa: E402

DEV_PER_CAT = 2
TEST_PER_CAT = 5
OUT_DIR = os.path.join(ROOT, "data", "splits")


def make_splits(task_ids, seed=0, dev_per_cat=DEV_PER_CAT, test_per_cat=TEST_PER_CAT):
    by_cat = collections.defaultdict(list)
    for tid in task_ids:
        by_cat[mab.category(tid)].append(tid)
    dev, test = [], []
    for cat in mab.CATEGORIES:
        ids = sorted(by_cat[cat], key=lambda t: int(t.split("_")[1]))
        if len(ids) < dev_per_cat + test_per_cat:
            raise ValueError("%s has only %d tasks" % (cat, len(ids)))
        random.Random("%s-%s" % (seed, cat)).shuffle(ids)
        dev.extend(ids[:dev_per_cat])
        test.extend(ids[dev_per_cat:dev_per_cat + test_per_cat])
    return {"dev": dev, "test": test}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    ids = [c["id"] for c in mab.load_cases()]
    splits = make_splits(ids, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    for name, members in splits.items():
        path = os.path.join(args.out, name + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"source": "medagentbench/data/medagentbench/test_data_v2.json",
                       "seed": args.seed, "n": len(members), "ids": members}, fh, indent=2)
            fh.write("\n")
        print("wrote %s (%d tasks)" % (os.path.relpath(path, ROOT), len(members)))


if __name__ == "__main__":
    main()
