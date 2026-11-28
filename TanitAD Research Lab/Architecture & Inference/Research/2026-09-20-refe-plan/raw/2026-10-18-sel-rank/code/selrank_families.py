#!/usr/bin/env python3
"""SEL-RANK: the FOUR METRIC FAMILIES for the selectors of P2's own full-navtest table (families6.py, unchanged, run in the tanitad venv).

Selectors (each a single executed 8-pose plan per token, taken from the table's `proposals`, the same A7-repaired poses the harness scored):
  P2_pick            the table's `pick` (gate: its families must reproduce RESULT_P2's P2 row)
  ORACLE_best64      argmax realised PDMS; ties resolved by the scorer's OWN selection score (no extra information)
  ORACLE_EP / _TTC / _EP_TTC   the section-D selections (head(s) replaced by the realised sub-score)
Writes one seam npz per selector (schema = stage3_full.compute_families: token, fingerprint, poses, sampling, arm) and calls families6.py.
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe selrank_families.py [--only P2_pick]
"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selrank_attr import agg_v1_np, sigmoid64, HEADS, planner_pick_torch  # noqa: E402

TABLE = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
F6 = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/families6.py"
OUT = os.path.join(HERE, "..", "raw", "families")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); a = ap.parse_args()
    z = np.load(TABLE, allow_pickle=False)
    sub, pdms, logits, pick = z["sub"].astype(np.float64), z["pdms"].astype(np.float64), z["logits"], z["pick"]
    N = len(pick); ar = np.arange(N)
    P = sigmoid64(logits)[..., :5]
    _, agg = planner_pick_torch(logits)
    sel = {"P2_pick": pick}
    # oracle with ties resolved by the scorer's own score
    key = pdms + 1e-9 * (agg / max(agg.max(), 1e-12))          # PDMS dominates (steps are >= ~1e-5); the scorer's score only breaks exact ties
    sel["ORACLE_best64"] = key.argmax(1)
    assert np.allclose(pdms[ar, sel["ORACLE_best64"]], pdms.max(1))
    for nm, rep in (("ORACLE_EP", ("ep",)), ("ORACLE_TTC", ("ttc",)), ("ORACLE_EP_TTC", ("ep", "ttc"))):
        pm = P.copy()
        for h in rep:
            pm[..., HEADS.index(h)] = sub[..., HEADS.index(h)]
        sel[nm] = agg_v1_np(pm).argmax(1)
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for nm, k in sel.items():
        if a.only and nm != a.only:
            continue
        poses = z["proposals"][ar, k].astype(np.float32)
        sp = os.path.join(OUT, f"{nm}_seam.npz")
        np.savez(sp, token=z["token"], fingerprint=z["fingerprint"], poses=poses, sampling=np.array([8.0, 0.5]), arm=np.array("SELRANK_" + nm))
        outj = os.path.join(OUT, f"{nm}_families.json")
        r = subprocess.call([PY_T, F6, "--seam", sp, "--inputs", EXPORT, "--stage", "1", "--label", nm, "--out", outj, "--n-boot", "2000"],
                            cwd=os.path.dirname(F6), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            stdout=open(os.path.join(OUT, f"{nm}_families.log"), "w"), stderr=subprocess.STDOUT)
        meta[nm] = {"rc": r, "pdms_x100": float(pdms[ar, k].mean() * 100), "seam": sp, "families_json": outj}
        print(nm, meta[nm], flush=True)
    json.dump(meta, open(os.path.join(OUT, "families_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
