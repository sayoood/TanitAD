"""EXPLORATORY (NOT pre-registered): paired CF2 - H0 ADE cell on S2 / S6 from the follow-up arms (sizes the clean-split prize)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1_fit_score as F
import compliant_reread_r8 as CR
tools = F.RT.tools_load()
W = Path(sys.argv[1]); out = {"label": "EXPLORATORY, unregistered (computed after CF2 and H0 E1 points were seen)", "cells": {}}
first = {}
for surf in ("S2", "S6"):
    for s in (0, 1):
        E = F.load_eval(tools, CR, s, first, surf)
        A = np.load(W / "result" / "followup" / f"arms_{surf}_s{s}.npz")
        assert np.array_equal(A["clip"], E["clip"]) and np.array_equal(A["ws"], E["ws"])
        c2 = F.comp(tools, A["arm_CF2"], E["G"], E["dt"])["ade_m"]; c0 = F.comp(tools, A["arm_H0"], E["G"], E["dt"])["ade_m"]
        out["cells"][f"{surf}|s{s}"] = F.cell(tools, c2, c0, E["eid"])
(W / "result" / "followup" / "EXPLORATORY_CF2_minus_H0.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print({k: (round(v["delta"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in out["cells"].items()}, "ZZL1EXPLDONEZZ")
