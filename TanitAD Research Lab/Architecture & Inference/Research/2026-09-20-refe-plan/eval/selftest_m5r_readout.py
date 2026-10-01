#!/usr/bin/env python3
"""Measure 5r: a CODE-PATH self-test of the Stage-1 readout + RESULT writer, on STAND-IN data -- NOT a result.

M5's banked per-model logits (<m5>/evals/*.npz: A0, V3_s*, V4_s*) are re-scored against the REPAIRED truth
(m5r_eval.setup) and written to a scratch evals dir with V4_s* ALSO copied under the V4r_s* names and V3_s* under V3r_s*
(so every registered model name exists); the gate files are pointed at a scratch dir. Checks: the readout runs, reads
the gates, names a verdict, and reproduces -- for the V4 - V3 contrast on the repaired truth -- the POST-HOC number
banked in eval/raw/m5_effectiveness/families.json (E1 -0.0388), an INDEPENDENT computation (m5_families.py PART 3).
Nothing is written under eval/raw/m5r except selftest_readout.json.

    python eval/selftest_m5r_readout.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m5_finetune_eval as FE  # noqa: E402
import m5r_eval as R  # noqa: E402

SCR = "D:/Projects/TanitAD/data/refe_m5/scratch/m5r_readout_test"


def main() -> int:
    if os.path.isdir(SCR):
        shutil.rmtree(SCR)
    os.makedirs(f"{SCR}/evals")
    os.makedirs(f"{SCR}/raw")
    E = R.setup()
    np.savez(f"{SCR}/evals/_truth.npz", token=np.array(E["toks"]), pdms=E["pdms"], sub=E["sub"])
    for p in glob.glob(f"{R.M5}/evals/*.npz"):
        name = os.path.basename(p)[:-4]
        if name.startswith("_") or name.startswith("V4all"):
            continue
        L = {k: v for k, v in np.load(p).items() if k in ("pure", "masked", "plain")}
        mt = FE.metrics(L, E)
        names = [name] + ([name.replace("V3", "V3r")] if name.startswith("V3") else []) + \
            ([name.replace("V4", "V4r")] if name.startswith("V4") else [])
        for n in names:
            np.savez(f"{SCR}/evals/{n}.npz", token=np.array(E["toks"]), **L, **mt)
    for f in ("selftest_repair.json", "gate_gr2.json"):
        if os.path.exists(os.path.join(R.OUTD, f)):
            shutil.copy(os.path.join(R.OUTD, f), f"{SCR}/raw/{f}")
    json.dump({"G_R3": {"pass": True}, "G_R4": {"pass": True}}, open(f"{SCR}/raw/gate_gr34.json", "w"))
    json.dump({"pass": True}, open(f"{SCR}/raw/gate_gr5.json", "w"))
    json.dump({"pass": True}, open(f"{SCR}/raw/gate_gr6_eval.json", "w"))
    R.M5R, R.OUTD = SCR, f"{SCR}/raw"
    R.readout(None)
    res = json.load(open(f"{SCR}/raw/readout_stage1.json", encoding="utf-8"))
    post = json.load(open(os.path.join(HERE, "raw", "m5_effectiveness", "families.json"), encoding="utf-8"))
    ph = post["posthoc_repaired_truth"]["pairs"]["E1_masked"]["V4_minus_V3"]
    got = res["comparisons"]["E1_masked"]["V4r_minus_V3"]            # V4r := V4 in this stand-in
    out = {"_label": "CODE-PATH SELF-TEST of m5r_eval.readout on stand-in models (V4r := M5's V4, V3r := M5's V3) -- "
                     "NOT a result", "verdict_string": res["decision"]["verdict"], "gates_read": res["gates"],
           "E1_V4r_minus_V3_standin": got, "E1_posthoc_V4_minus_V3_independent": ph,
           "reproduces_independent": got["diff"] == ph["diff"] and got["ci95"] == ph["ci95"]}
    json.dump(out, open(os.path.join(R.OUTD.replace(SCR + "/raw", os.path.join(HERE, "raw", "m5r")),
                                     "selftest_readout.json") if False else os.path.join(HERE, "raw", "m5r",
                                                                                         "selftest_readout.json"),
                        "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"ZZM5R_READOUT_SELFTEST_{'PASS' if out['reproduces_independent'] else 'FAIL'}")
    return 0 if out["reproduces_independent"] else 1


if __name__ == "__main__":
    sys.exit(main())
