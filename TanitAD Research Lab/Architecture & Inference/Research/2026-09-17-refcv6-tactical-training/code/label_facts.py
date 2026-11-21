"""THE LABEL LIMITS EVERY TACTICAL NUMBER MUST BE REPORTED UNDER — per POLICY.

⛔⛔ THE POINT OF THIS FILE: the limits are NOT a constant of the corpus, they
are a function of the NEGATIVES POLICY, and the two live policies disagree by
4 tokens on trainability and by 6 on the pos_weight cap. Quoting "21 of 22
trainable" beside a run that used `negatives=measured` is the scope error this
programme has paid for repeatedly — a true number quoted outside its scope.

⭐ The scoreability floor is the exception and is stated as such: it counts
POSITIVES, so it is policy-INDEPENDENT (10 of 22 either way).

Usage:  TAC_LABELS_TRAIN=... TAC_COT_SIDECAR=... python label_facts.py
"""
from __future__ import annotations

import inspect
import json
import os

from tanitad.data import v7_labels as v7l
from tanitad.models import vocab_v7
from tanitad.refs import tac_goal_head as tgh

BLOB = os.environ["TAC_LABELS_TRAIN"]
COT = os.environ.get("TAC_COT_SIDECAR")

labs, man = v7l.load_v7_labels(BLOB)
# ⛔ THE CAP IS READ FROM THE FUNCTION'S OWN SIGNATURE, never retyped: it is a
# DEFAULT PARAMETER (`cap: float = 50.0`), not a module constant.
CAP = float(inspect.signature(v7l.goal_pos_weight).parameters["cap"].default)
FLOOR = int(vocab_v7.GOAL_MIN_N_FOR_METRIC)

policies = [("measured", {})]
if COT:
    got = v7l.load_cot_negative_sidecar(COT, man)
    policies.append(("cot-absence-negative",
                     {"sidecar": got[0] if isinstance(got, tuple) else got}))

out = {"blob": os.path.basename(BLOB), "md5": man.md5, "n_clips": len(labs),
       "pos_weight_cap": CAP, "scoreability_floor_n": FLOOR, "policies": {}}
for name, kw in policies:
    cen = v7l.goal_supervision_census(labs, negatives=name, **kw)
    pw = v7l.goal_pos_weight(labs, negatives=name, **kw)
    mask = tgh.mask_report(cen)
    on_cap = [t for t, x in zip(vocab_v7.TACTICAL_GOAL_TOKENS_V7, pw)
              if float(x) >= CAP - 1e-6]
    under = [t for t in vocab_v7.TACTICAL_GOAL_TOKENS_V7
             if int(cen[t]["pos"]) < FLOOR]
    out["policies"][name] = {
        "n_trainable": int(mask["n_trainable"]),
        "n_total": int(mask["n_total"]),
        "n_on_pos_weight_cap": len(on_cap),
        "on_pos_weight_cap": on_cap,
        "n_under_scoreability_floor": len(under),
        "under_scoreability_floor": under,
        "pos_weight": [round(float(x), 4) for x in pw],
        "⚠️": "for the tokens ON the cap it is the CAP, not the data, that "
              "sets the weight",
    }
out["⛔"] = ("the trainer's DEFAULT is `measured`. The 21/22 + 9-on-cap figures "
            "quoted around this programme are the `cot-absence-negative` "
            "numbers and are WRONG for a `measured` run. Name the policy.")
print(json.dumps(out, indent=1, ensure_ascii=False))
