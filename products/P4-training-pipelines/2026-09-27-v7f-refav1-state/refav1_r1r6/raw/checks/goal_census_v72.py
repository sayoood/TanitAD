"""R3 census (MEASURED, zero GPU): what the goal head would train on, per token, on the canonical
v7.2 TRAIN and EVAL blobs, under the default 'measured' negative policy and under the PI's
2026-09-16 cot-absence policy (sidecar built over the SAME blob by the unchanged builder).

Uses ONLY the shared functions the trainer uses (`v7_labels.goal_supervision_census`,
`goal_pos_weight`, `tac_goal_head.mask_report`), so these numbers are the trainer's.
Blob paths are env-overridable; the md5s are pinned (six copies of the blob circulate).
"""
import json
import os
import sys

from tanitad.data import v7_labels as v7l
from tanitad.refs.tac_goal_head import mask_report

ROOT = os.environ.get("REFAV1_R1R6_ROOT", "C:/Users/Admin/refav1_r1r6")
BLOBS = {"train": (f"{ROOT}/data/s2_labels_v7.2_train.jsonl.gz",
                   "0ff902130ce76886b8a925eceed9e3a5",
                   f"{ROOT}/raw/sidecars/cot_absence_negative_v7.2_train.json.gz"),
         "eval": (f"{ROOT}/data/s2_labels_v7.2_eval_repo.jsonl.gz",
                  "aa12c948f062181c3297265b51526ec5",
                  f"{ROOT}/raw/sidecars/cot_absence_negative_v7.2_eval.json.gz")}
rep = {}
for split, (path, md5, side) in BLOBS.items():
    labels, man = v7l.load_v7_labels(path, allow_oracle_nav=True)
    assert man.md5 == md5, (split, man.md5)
    r = {"md5": man.md5, "n_records": man.n_records}
    for pol in ("measured", "cot-absence-negative"):
        kw = {"negatives": pol}
        if pol == "cot-absence-negative":
            sc, _ = v7l.load_cot_negative_sidecar(side, man)
            v7l.assert_sidecar_matches_presence(labels, sc)
            kw["sidecar"] = sc
            r["sidecar_md5"] = sc.md5
        cen = v7l.goal_supervision_census(labels, **kw)
        mr = mask_report(cen)
        pw = v7l.goal_pos_weight(labels, **kw)
        r[pol] = {"n_trainable": mr["n_trainable"], "masked": mr["masked_why"],
                  "per_token": {t: {"pos": cen[t]["pos"], "neg": cen[t]["neg"],
                                    "ignored": cen[t]["ignored"],
                                    "pos_weight": round(float(pw[i]), 3)}
                                for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}}
    sb = [x.tac_goal_meta.get("SPEED_BAND") or {} for x in labels]
    r["speed_band_args_present"] = sum(1 for s in sb if s.get("v_lo_ms") is not None
                                       and s.get("v_hi_ms") is not None)
    rep[split] = r
json.dump(rep, sys.stdout, indent=1)
print()
for split, r in rep.items():
    for pol in ("measured", "cot-absence-negative"):
        print(f"# {split} {pol}: {r[pol]['n_trainable']}/22 BCE classes trainable; masked "
              f"{sorted(r[pol]['masked'])}; SPEED_BAND args on {r['speed_band_args_present']}/"
              f"{r['n_records']} records", file=sys.stderr)
