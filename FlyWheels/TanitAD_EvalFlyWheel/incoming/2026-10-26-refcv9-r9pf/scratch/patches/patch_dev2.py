import json
from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/deviations_r9pf.json")
d = json.loads(p.read_text(encoding="utf-8"))
by = {x["id"]: x for x in d}
by["D7"]["text"] = ("The prereg s1 EXPLORATORY calibration-tuned decel tau was not run in A2; it was run AFTER the bars (code/x_tau_fit.py: OOF regime "
    "logits on the calibration rows, the head's own wd / epochs and folds; tau = max Youden J for decel, ties -> larger tau; fs0/1/2 tau = 0.25 / "
    "0.29 / 0.28) and is reported BESIDE only (result/R9PF_X_ATTRIB.json, arm TAU). It never enters a bar.")
if "D19" not in by:
    d.append({"id": "D19", "text": "EXPLORATORY post-bar attribution (code/x_attrib_r9pf.py -> result/R9PF_X_ATTRIB.json; NOT a bar, NOT pre-registered): "
              "arms ORACLE (GT regime + R9 profiles; inadmissible), CRUISE, TAU, SOFT (posterior-weighted profile), HYB (R9 decision gating "
              "H2CN_DEPLOY8). Run under RULE ZERO to rank the next lever; every number there is labelled EXPLORATORY and none changes the verdict."})
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print([x["id"] for x in d])
