"""Readout smoke on SYNTHETIC arm folders (no RA pass is run, no model is loaded): every arm folder is a copy of A0's banked SCORE plan with a small deterministic acceleration nudge on knots 1-10
(OFF = an exact copy), under a throwaway AMEND5_ROOT. Purpose: the contrast / label / md pipeline runs end to end. The numbers are meaningless and are never quoted.
usage: python smoke_readout.py <root> [n_boot]"""
import json, os, shutil, sys
sys.dont_write_bytecode = True
root = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 100
os.environ["AMEND5_ROOT"] = root
os.environ["AMEND5_BASE_SOLVER"] = f"{root}/BASE_SOLVER_AMEND5.md"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import amend5_stage as ST
os.makedirs(f"{root}/controls", exist_ok=True); os.makedirs(f"{root}/readout", exist_ok=True)
open(os.environ["AMEND5_BASE_SOLVER"], "w").write("BASE_SOLVER: A0\nBRANCH: B3\n")
shutil.copy(f"{ST.L0}/amend5/controls/CONTROLS.json", f"{root}/controls/CONTROLS.json")
for k in ("s0", "s1"):
    shutil.copy(f"{ST.L0}/amend5/readout/TRAIN_STAGE_{k}.json", f"{root}/readout/TRAIN_STAGE_{k}.json")
    Bd = ST.base_dir("score", "A0", k)
    for i, (arm, nudge) in enumerate((("RAL1", -0.3), ("MEANL1", -0.1), ("NEG", 0.5), ("SHUF", 0.8), ("ORACLE", 0.4), ("OFF", 0.0))):
        o = ST.out_dir("score", arm, "A0", k, root); os.makedirs(o)
        shutil.copy(f"{Bd}/windows.npz", f"{o}/windows.npz")
        for s in (0, 1):
            z = dict(np.load(f"{Bd}/plans_hwm_oracle_s{s}.npz"))
            if nudge:
                z["plan_raw"] = z["plan_raw"].copy(); z["plan_raw"][:, :10, 0] += nudge * (1 + 0.05 * s)
            n = len(z["plan_raw"]); z.update(ra_cost=np.full(n, 1e-8), ra_hit_box=np.zeros(n, bool))
            np.savez(f"{o}/plans_hwm_oracle_s{s}.npz", **z)
import amend5_read as RD
r = RD.score(root, nb)
print(json.dumps({x: {kk: v for kk, v in r["labels"]["arms"][x].items() if kk != "cells"} for x in r["labels"]["arms"]}, indent=1, default=float))
print("INVALID" in r["labels"], r["controls_ok"])
