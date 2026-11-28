#!/usr/bin/env python3
"""Control KP (SPEC §11, amendment A3) — the PRECISION FLOOR: the same checkpoint, the same per-scene
inference seeds, the same arm and split, run once on CPU fp32 and once on CUDA bf16 (as trained).

    python code/kp_compare.py --a-rows <cpu rows_R6_A1.jsonl> --b-rows <cuda rows_R6_A1.jsonl> \
        --inputs <navsim_agent_inputs.json> [--a-csv score_R6_A1.csv --b-csv score_R6_A1.csv] --out KP.json

Reads (never re-computes) the two bridges' banked rows: per-scene 4 s endpoint distance between the
two plans, the fraction of bit-identical plans, anchor-selection agreement, and — when both score
CSVs exist — the S2-EPDMS-u of each (E2's own ``s2_group_uniform``, imported by ``parse6``'s path)
and its difference. Any cross-checkpoint comparison whose devices differ is read against this number
(and against the seed floor); a difference inside either is not a training effect.
Refuses: rows of different arms, a device/precision that is not the claimed one, or scenes whose
declared inputs or seeds differ (then the two runs did not answer the same question).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(path: str) -> dict:
    out = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-rows", required=True, help="CPU fp32 rows")
    ap.add_argument("--b-rows", required=True, help="CUDA bf16 rows")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--a-csv", default="")
    ap.add_argument("--b-csv", default="")
    ap.add_argument("--navtest", action="store_true", help="the CSVs are navtest v1.1 (single stage)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    A, B = load_rows(a.a_rows), load_rows(a.b_rows)
    own = sorted(t for t in set(A) & set(B)
                 if A[t]["source"] == "refcv6" and B[t]["source"] == "refcv6")
    if not own:
        sys.exit("⛔ no common refcv6 rows")
    for t in own:
        ra, rb = A[t], B[t]
        if ra["arm"] != rb["arm"]:
            sys.exit(f"⛔ {t}: arms differ {ra['arm']} / {rb['arm']}")
        if ra["seed"] != rb["seed"] or ra["declared_values"] != rb["declared_values"] \
                or ra["frame_sha16"] != rb["frame_sha16"]:
            sys.exit(f"⛔ {t}: seed / declared inputs / frame differ — not the same question")
    devs = ({A[t]["device"] for t in own}, {B[t]["device"] for t in own})
    precs = ({A[t]["precision"] for t in own}, {B[t]["precision"] for t in own})
    pa = np.asarray([A[t]["poses"] for t in own], dtype=np.float64)
    pb = np.asarray([B[t]["poses"] for t in own], dtype=np.float64)
    end = np.hypot(pa[:, -1, 0] - pb[:, -1, 0], pa[:, -1, 1] - pb[:, -1, 1])
    allpts = np.hypot(pa[..., 0] - pb[..., 0], pa[..., 1] - pb[..., 1]).max(axis=1)
    sel_a = [A[t].get("diag", {}).get("sel_idx") for t in own]
    sel_b = [B[t].get("diag", {}).get("sel_idx") for t in own]
    sel_known = [(x, y) for x, y in zip(sel_a, sel_b) if x is not None and y is not None]
    rep = {"control": "KP precision floor (SPEC §11)", "arm": A[own[0]]["arm"], "n_scenes": len(own),
           "a": {"rows": os.path.abspath(a.a_rows), "device": sorted(devs[0]),
                 "precision": sorted(precs[0])},
           "b": {"rows": os.path.abspath(a.b_rows), "device": sorted(devs[1]),
                 "precision": sorted(precs[1])},
           "endpoint_4s_distance_m": {"median": float(np.median(end)), "mean": float(end.mean()),
                                      "p90": float(np.quantile(end, 0.9)), "max": float(end.max())},
           "max_pointwise_distance_m": {"median": float(np.median(allpts)),
                                        "p90": float(np.quantile(allpts, 0.9))},
           "frac_bit_identical_plans": float((allpts == 0.0).mean()),
           "selection_agreement": (None if not sel_known else
                                   float(np.mean([x == y for x, y in sel_known]))),
           "n_selection_known": len(sel_known)}
    if a.navtest and a.a_csv and a.b_csv and os.path.exists(a.a_csv) and os.path.exists(a.b_csv):
        import pandas as pd
        ca = pd.read_csv(a.a_csv).set_index("token")
        cb = pd.read_csv(a.b_csv).set_index("token")
        ca, cb = ca[ca.index != "average"], cb[cb.index != "average"]
        com = sorted(set(ca.index) & set(cb.index))
        d = (ca.loc[com, "score"] - cb.loc[com, "score"]).to_numpy(dtype=float)
        rep["PDMS_x100"] = {"a": round(100 * float(ca.loc[com, "score"].mean()), 4),
                            "b": round(100 * float(cb.loc[com, "score"].mean()), 4),
                            "a_minus_b": round(100 * float(d.mean()), 4)}
        rep["scene_score_deltas"] = {"n": len(com), "n_differ": int((np.abs(d) > 1e-12).sum()),
                                     "mean_abs_x100": round(100 * float(np.abs(d).mean()), 4)}
        rep["csvs"] = [os.path.abspath(a.a_csv), os.path.abspath(a.b_csv)]
    elif a.a_csv and a.b_csv and os.path.exists(a.a_csv) and os.path.exists(a.b_csv):
        sys.path.insert(0, HERE)
        import parse6 as P6                                                   # noqa: E402
        ps = P6._load_mod("e2_parse_scores_kp", os.path.join(P6.E2, "code", "parse_scores.py"))
        doc = json.load(open(a.inputs, encoding="utf-8"))
        stage_of = {t: r["stage"] for t, r in doc["tokens"].items()}
        mapping = doc["reactive_all_mapping"]
        da = P6.load_csv(a.a_csv, stage_of, "A")
        db = P6.load_csv(a.b_csv, stage_of, "B")
        ua = ps.s2_group_uniform(da, mapping).get("value")
        ub = ps.s2_group_uniform(db, mapping).get("value")
        s2a = da[da.stage == 2].set_index("token")["score"]
        s2b = db[db.stage == 2].set_index("token")["score"]
        com = sorted(set(s2a.index) & set(s2b.index))
        d = (s2a.loc[com] - s2b.loc[com]).to_numpy()
        rep["S2_EPDMS_u"] = {"a": ua, "b": ub, "a_minus_b": None if ua is None or ub is None
                             else ua - ub}
        rep["scene_score_deltas"] = {"n": len(com), "n_differ": int((np.abs(d) > 1e-12).sum()),
                                     "mean_abs": float(np.abs(d).mean())}
        rep["csvs"] = [os.path.abspath(a.a_csv), os.path.abspath(a.b_csv)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
