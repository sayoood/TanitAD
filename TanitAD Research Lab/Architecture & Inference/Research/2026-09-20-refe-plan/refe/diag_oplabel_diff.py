#!/usr/bin/env python3
"""Why does the on-policy labeller's teacher row differ from the bank's on ONE key for ONE frame?

Re-scores the teacher path of one banked frame and prints every key whose value differs from the
bank's `teacher` row, with both values. Run it under the variations that could explain it:
  REFE_SCORER_EGO_VIEW=0/1 · PYTHONHASHSEED · --old-sp (the live package's score_proposals.py)
  python diag_oplabel_diff.py --bank <train_grow> --token <tok> [--rank 0] [--old-sp]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--token", required=True)
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--old-sp", action="store_true", help="import score_proposals from /workspace/refe-plan")
a = ap.parse_args()
if a.old_sp:
    sys.path.insert(0, "/workspace/refe-plan/refe")         # ahead of this dir: the LIVE copy wins
import numpy as np  # noqa: E402

sys.path.insert(1 if a.old_sp else 0, str(Path(__file__).resolve().parent))
import onpolicy_label as OL  # noqa: E402
import score_proposals as SP  # noqa: E402

print(f"  score_proposals from {SP.__file__}  EGO_VIEW={SP.EGO_VIEW}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
bank = Path(a.bank)
sf = bank / ("scorer_targets.jsonl" if a.rank == 0 else "scorer_targets_rank1.jsonl")
ref = None
with open(sf, encoding="utf-8") as f:
    for line in f:
        if a.token in line:
            r = json.loads(line)
            if r["token"] == a.token and r.get("candidate") == "teacher":
                ref = r
                break
row = None
with open(bank / f"targets_rank{a.rank}.jsonl", encoding="utf-8") as f:
    for line in f:
        if a.token in line:
            r = json.loads(line)
            if r["token"] == a.token:
                row = r
                break
S = OL.Scorer(a.rank, 2)
sc = S.scenarios(row["log_name"], [a.token])[a.token]
tr = np.asarray(row["traj"], dtype=float)
got, nd = S.score(sc, row, [("teacher", OL._t(tr[:, :2]), OL._t(tr[:, 2]))])
mine = json.loads(json.dumps(got["teacher"]))
theirs = ref["targets"]
for k in sorted(set(mine) | set(theirs)):
    if json.dumps(mine.get(k)) != json.dumps(theirs.get(k)):
        print(f"  DIFF {k}: mine {mine.get(k)!r}  bank {theirs.get(k)!r}")
print(f"  bank row: stride {ref.get('stride')} n_signals_differing {ref.get('n_signals_differing')} "
      f"teacher_off_road {ref.get('teacher_off_road')}")
print("ZZDIFF_DONE")
