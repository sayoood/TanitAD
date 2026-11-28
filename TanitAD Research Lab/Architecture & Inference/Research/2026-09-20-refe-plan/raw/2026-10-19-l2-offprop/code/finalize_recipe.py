#!/usr/bin/env python3
"""Stage 3 of the recipe selection: merge the sweep shards, apply the pre-committed rule (design_sweep.py), walk DOWN the ranked list until a setting also clears C-REACH on seed 1, run the reported-not-gating
null-world diagnostics for the pick, and write the recipe as a DATA file.  If no setting clears C-REACH, write INSTRUMENT_DEFECT.json and NO recipe (exit 3).  No real unplanted label is read.

    python finalize_recipe.py [--parts "out/sweep/part_*.json"]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import l2lib  # noqa: E402
import design_sweep as ds  # noqa: E402
import synthetic  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts/l2_offprop"
OUT = "C:/Users/Admin/par18/l2/out"


def null_world_diagnostics(real, rec) -> dict:
    """REPORTED, never gating: in the null world (beta 0: no learnable within-slot ordering) H must not gain over P2 and the GLOBAL-null H-null must not exceed P2 (the VOID rule's own quantity)"""
    D = synthetic.semi_synthetic(real, beta=0.0, alpha=0.4, seed=23)
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), rec)
    a1_p2, inf_p2, _ = ds._read(D, p2)
    h = l2lib.oof_scores(D, l2lib.ArmSpec("H", "H", seed=0), rec)
    h0 = l2lib.oof_scores(D, l2lib.ArmSpec("H0", "H0", seed=0), rec)
    nulls = [l2lib.oof_scores(D, l2lib.ArmSpec(f"N{r}", "Hnull", seed=0, null_rep=r), rec) for r in (100, 101)]
    a1_h, inf_h, _ = ds._read(D, h)
    a1_0, inf_0, _ = ds._read(D, h0)
    a1_n = [ds._read(D, n)[0] for n in nulls]
    return {"a1_p2": a1_p2, "a1_h": a1_h, "a1_h_minus_p2": a1_h - a1_p2, "infan_h_minus_h0": inf_h - inf_0, "infan_p2": inf_p2, "a1_hnull_global_r100_r101": a1_n,
            "a1_hnull_global_max_minus_p2": max(a1_n) - a1_p2, "note": "REPORTED, not gating (MM rule names only C-REACH, the in-fan clause and the tie rule)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=[f"{OUT}/sweep/part_*.json"])
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    parts = sorted(p for pat in a.parts for p in glob.glob(pat))
    res = {"base": None, "rows": {}, "parts": parts}
    for p in parts:
        r = json.load(open(p))
        res["base"] = res["base"] or r["base"]
        res["rows"].update(r["rows"])
    res["n_expected_settings"] = len(ds.settings())
    res["choice"] = ds.choose(res)
    json.dump(l2lib.jsonable(res), open(f"{OUT}/sweep_semi.json", "w"), indent=1)
    ch = res["choice"]
    if len(res["rows"]) != len(ds.settings()):
        print(f"ZZFINALIZE_FAIL sweep incomplete: {len(res['rows'])} of {len(ds.settings())} settings")
        return 2
    real = l2lib.load_real(f"{ART}/feats.npz")
    walk = []
    pick = None
    for cand in ch["ranked"]:
        rec = l2lib.Recipe(lam_bce=cand["lam"], lr=cand["lr"], epochs=cand["epochs"], prox=cand["prox"])
        t0 = time.time()
        r1 = ds.reach_read(real, rec, 1)
        ok = bool(cand["reach_s0"] >= ds.REACH_BAR and r1["a1"] >= ds.REACH_BAR)
        walk.append({"key": cand["key"], "reach_s0": cand["reach_s0"], "reach_s1": r1["a1"], "both_seeds_pass": ok, "seconds": round(time.time() - t0, 1)})
        print("walk", walk[-1], flush=True)
        if ok:
            pick = cand
            break
    if pick is None:
        defect = {"verdict": "INSTRUMENT DEFECT", "why": "no setting of the grid clears C-REACH >= 0.98 on BOTH seeds with the in-fan rule applicable", "grid": {k: list(v) for k, v in ds.GRID.items()}, "extra": ds.EXTRA,
                  "choice": ch, "walk": walk, "sweep": f"{OUT}/sweep_semi.json"}
        json.dump(l2lib.jsonable(defect), open(f"{OUT}/INSTRUMENT_DEFECT.json", "w"), indent=1)
        print("ZZFINALIZE_DEFECT", json.dumps(l2lib.jsonable({"n_reach_pass_s0": ch["n_reach_pass_s0"], "walk": walk})))
        return 3
    rec = l2lib.Recipe(lam_bce=pick["lam"], lr=pick["lr"], epochs=pick["epochs"], prox=pick["prox"])
    diag = null_world_diagnostics(real, rec)
    row = res["rows"][pick["key"]]
    out = {"recipe": rec.as_dict(), "chosen_by": ch["rule"], "clause_may_bind": ch.get("clause_may_bind", False), "pick": pick, "pick_worlds": row["worlds"], "walk_down_for_seed1_reach": walk,
           "null_world_diagnostics_reported_not_gating": diag, "n_settings": ch["n_settings"], "n_reach_pass_s0": ch["n_reach_pass_s0"], "n_in_fan_clause_pass": ch["n_r1"],
           "sweep": f"{OUT}/sweep_semi.json", "sweep_sha256": l2lib.sha256_file(f"{OUT}/sweep_semi.json"),
           "world": "semi-synthetic (REAL tokens, SYNTHETIC labels); C-REACH reads the PLANTED arm only",
           "no_real_unplanted_label_was_read": True, "lambda_convention": "lambda = pairwise / in-fan weight = 1 / lam_bce; ties go to the smaller lambda = the larger lam_bce"}
    os.makedirs(ART, exist_ok=True)
    json.dump(l2lib.jsonable(out), open(f"{ART}/recipe_chosen.json", "w"), indent=1)
    json.dump(l2lib.jsonable(out), open(f"{OUT}/recipe_chosen.json", "w"), indent=1)
    print("ZZFINALIZE_OK", json.dumps(l2lib.jsonable({"pick": pick["key"], "reach_s0": pick["reach_s0"], "reach_s1": walk[-1]["reach_s1"], "rule": ch["rule"], "n_r1": ch["n_r1"]})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
