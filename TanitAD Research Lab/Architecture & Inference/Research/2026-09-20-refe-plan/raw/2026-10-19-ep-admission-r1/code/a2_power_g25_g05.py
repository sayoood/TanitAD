"""SUPPLEMENTARY to EP-ADMISSION-R2-CONFIRM: A2 power on the G25 and G05 geometries.  The banked run (and so the R2 run) has the planted-A2 conditions on G78 only; the brief asks for G25 / G05 beside it.
Same generator A, same estimator, same registered draws, same `plant_a2` builder, fresh offset.   python a2_power_g25_g05.py --workdir D:/epadm2/run --workers 12 --rep-offset 1000000"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ep_admission_sim as S                                                                  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-reps", type=int, default=1000)
    ap.add_argument("--rep-offset", type=int, default=1000000)
    a = ap.parse_args()
    S.design_guard()
    vec = S.load_vec(a.workdir)
    conds = {}
    for g in ("G25", "G05"):
        for h in S.HEADS:
            for rho in S.RHOS:
                n = f"{g}|A|PLANT_A2_{h}_{rho}"
                conds[n] = {"name": n, "gen": "A", "geom": g, "a1": False, "a2": True, "kind": "plant_a2", "kw": {"head": h, "rho": rho}, "heads": [h],
                            "vec": {s: S.build_condition(vec[g][s], "plant_a2", head=h, rho=rho) for s in S.SEEDS}}
    recs = S.run_condition_set(conds, a.n_reps, a.workers, rep_offset=a.rep_offset)
    margins = (1.00, 1.05, 1.10)
    power = {g: {h: {str(rho): {} for rho in S.RHOS} for h in S.HEADS} for g in ("G25", "G05")}
    for n, r in recs.items():
        g, _, _, h, rho = n.split("|")[0], None, None, n.split("_")[2], n.split("_")[3]
        lo = r["a2"][:, :, S.HEADS.index(h), 1]
        for t in margins:
            power[g][h][rho][str(t)] = S.rate(int((lo > t).any(axis=1).sum()), len(lo))
    np.savez_compressed(os.path.join(a.workdir, "a2_power_g25_g05_replicates.npz"), **{f"{n}|{k}".replace("|", "__"): v for n, r in recs.items() for k, v in r.items()})
    json.dump({"what": "SUPPLEMENTARY: planted-A2 power on G25 / G05 (not in the banked condition set)", "rep_offset": a.rep_offset, "n_reps": a.n_reps, "margins": list(margins), "power": power},
              open(os.path.join(a.workdir, "a2_power_g25_g05.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("done", os.path.join(a.workdir, "a2_power_g25_g05.json"))
