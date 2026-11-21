#!/usr/bin/env python3
"""Checkpoint vs checkpoint on ONE split — did refcv6 move between two milestones?

    python code/step_compare.py --split warmup --a raw/milestones/step5000 --b raw/milestones/step30000 \
        --out raw/milestones/step30000/compare_vs_step5000_warmup.json

Reads the two milestones' banked score CSVs for R6_A1 (never re-scores), pairs them on identical tokens
and reports b − a with the estimator each split uses: warmup S2-EPDMS-u (no interval: 7 logs < 8);
navhard the official two-stage EPDMS with the PAIRED log-cluster bootstrap (navsim_ci); navtest PDMS
with the paired log-cluster bootstrap. ⚠️ Three questions ride on any such delta: another draw of
LOGS (the interval), another INFERENCE seed (read against both checkpoints' R6_A1_s1 seed floors, printed
beside it), another TRAINING run (untested — one run). And a DEVICE question (amendment A3): when the
two readings ran on different devices the delta is flagged and must be read against KP.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_stamp6 as M6  # noqa: E402
PKG = os.path.dirname(HERE)
E2 = os.path.abspath(os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge"))
INPUTS = {"warmup": os.path.join(E2, "raw", "navsim_agent_inputs.json"),
          "navhard": "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json",
          "navtest": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def csv_of(ms: str, split: str, arm: str = "R6_A1") -> str:
    if split == "warmup":
        return os.path.join(ms, "scores_warmup", f"score_{arm}.csv")
    if split == "navhard":
        return os.path.join(ms, "scores_navhard", f"score_{arm}__navhard_two_stage.csv")
    step = json.load(open(os.path.join(ms, "MILESTONE_SUMMARY.json"), encoding="utf-8"))["step"]
    lab = f"r6s{step}_{arm}"
    return os.path.join(ms, "scores_navtest", lab, f"{lab}.csv")


def device_of(ms: str, split: str, arm: str = "R6_A1") -> str:
    m = json.load(open(os.path.join(ms, f"bridge_{split}", f"seam_{arm}.manifest.json"), encoding="utf-8"))
    return "/".join(m["device"]) + " " + "/".join(m["precision"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("warmup", "navhard", "navtest"))
    ap.add_argument("--a", required=True, help="earlier milestone dir")
    ap.add_argument("--b", required=True, help="later milestone dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    op = gzip.open if INPUTS[a.split].endswith(".gz") else open
    with op(INPUTS[a.split], "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    toks = doc["tokens"]
    sa = json.load(open(os.path.join(a.a, "MILESTONE_SUMMARY.json"), encoding="utf-8"))
    sb = json.load(open(os.path.join(a.b, "MILESTONE_SUMMARY.json"), encoding="utf-8"))
    rep = {"what": f"R6_A1 step {sa['step']} -> step {sb['step']} on {a.split}", "a": a.a, "b": a.b,
           "step_a": sa["step"], "step_b": sb["step"],
           "device_a": device_of(a.a, a.split), "device_b": device_of(a.b, a.split)}
    rep["same_device"] = rep["device_a"] == rep["device_b"]
    rep["model_as_trained_a"], rep["model_as_trained_b"] = M6.stamp(sa["step"]), M6.stamp(sb["step"])
    sw = M6.switch_warning(sa["step"], sb["step"])
    if sw:
        rep["fix_switch_warning"] = sw
    if not rep["same_device"]:
        rep["device_warning"] = ("the two readings ran on DIFFERENT devices (amendment A3): read the delta "
                                 "against the precision floor KP before calling it a training effect")
    ca, cb = pd.read_csv(csv_of(a.a, a.split)), pd.read_csv(csv_of(a.b, a.split))
    for c in (ca, cb):
        c.drop(c[c.token.isin(["average"]) | c.token.astype(str).str.startswith("extended_pdm_score")].index,
               inplace=True)
    ca, cb = ca.set_index("token"), cb.set_index("token")
    com = sorted(set(ca.index) & set(cb.index))
    if len(com) != len(ca) or len(com) != len(cb):
        sys.exit("⛔ the two readings are not on identical tokens — refused")
    for side, ms in (("a", a.a), ("b", a.b)):
        summ = os.path.join(ms, f"summary_{a.split}.json")
        s = json.load(open(summ, encoding="utf-8")) if os.path.exists(summ) else {}
        if a.split == "navtest":
            sf = (s.get("pairs") or {}).get("R6_A1__minus__R6_A1_s1", {}).get("delta_x100")
            if sf is None:      # the full split carries no replicate: the 200-token diagnostic does
                dg = os.path.join(ms.rstrip("/\\") + "_navtest_diag", "summary_navtest.json")
                if os.path.exists(dg):
                    sf = (json.load(open(dg, encoding="utf-8")).get("pairs") or {}).get(
                        "R6_A1__minus__R6_A1_s1", {}).get("delta_x100")
                    rep[f"seed_floor_{side}_source"] = dg + " (200 tokens)"
            rep[f"PDMS_x100_{side}"] = (s.get("arms") or {}).get("R6_A1", {}).get("PDMS")
        else:
            sf = s.get("seed_floor_S2_EPDMS_u") if a.split == "warmup" else \
                (s.get("pairs") or {}).get("R6_A1__minus__R6_A1_s1", {}).get("official_two_stage_EPDMS_delta")
            r = (s.get("arms") or {}).get("R6_A1", {})
            rep[f"S2_EPDMS_u_{side}"] = (r.get("S2_EPDMS_u") or {}).get("value")
            rep[f"official_two_stage_EPDMS_{side}"] = r.get("official_two_stage_EPDMS")
        rep[f"seed_floor_{side}"] = sf
    if a.split == "warmup":
        stage = {t: r["stage"] for t, r in toks.items()}
        s2 = [t for t in com if stage[t] == 2]
        d = (cb.loc[s2, "score"] - ca.loc[s2, "score"]).to_numpy(float)
        rep["delta_S2_EPDMS_u"] = (None if rep["S2_EPDMS_u_a"] is None or rep["S2_EPDMS_u_b"] is None
                                   else rep["S2_EPDMS_u_b"] - rep["S2_EPDMS_u_a"])
        rep["scene_wtl_stage2"] = {"win": int((d > 1e-12).sum()), "tie": int((np.abs(d) <= 1e-12).sum()),
                                   "loss": int((d < -1e-12).sum()), "n": len(s2)}
        rep["interval"] = "UNAVAILABLE by design (warmup: 7 logs < the 8-cluster floor)"
        floors = [x for x in (rep["seed_floor_a"], rep["seed_floor_b"]) if x is not None]
        if floors and rep["delta_S2_EPDMS_u"] is not None:
            rep["exceeds_both_seed_floors"] = abs(rep["delta_S2_EPDMS_u"]) > max(abs(x) for x in floors)
    else:
        ci = _load("navsim_ci_sc", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
        tok2log = {t: r["log_name"] for t, r in toks.items()}
        if a.split == "navtest":
            xa = ca.loc[com, "score"].to_numpy(float)
            xb = cb.loc[com, "score"].to_numpy(float)
            rep["delta_PDMS_x100"] = round(100 * float(xb.mean() - xa.mean()), 4)
            iv = ci.paired_log_cluster_bootstrap(xb, xa, [tok2log[t] for t in com],
                                                 aggregation=ci.AGG_SINGLE_STAGE,
                                                 official_a=float(xb.mean()), official_b=float(xa.mean()))
        else:
            mapping = doc["reactive_all_mapping"]
            fa = os.path.join(a.a, "scores_navhard", "score_R6_A1__navhard_two_stage_wrapper",
                              "R6_A1__navhard_two_stage_final_scores_frame.csv")
            fb = os.path.join(a.b, "scores_navhard", "score_R6_A1__navhard_two_stage_wrapper",
                              "R6_A1__navhard_two_stage_final_scores_frame.csv")
            ra, _ = ci.rows_from_score_frame(fa)
            rb, _ = ci.rows_from_score_frame(fb)
            ka = ci.two_stage_key_contributions(ra, mapping, "score")
            kb = ci.two_stage_key_contributions(rb, mapping, "score")
            clusters = [tok2log[str(m[0])] for m in mapping]
            oa, ob = rep["official_two_stage_EPDMS_a"], rep["official_two_stage_EPDMS_b"]
            rep["delta_official_two_stage_EPDMS"] = (ob - oa) if isinstance(oa, float) and isinstance(ob, float) else None
            iv = ci.paired_log_cluster_bootstrap(kb, ka, clusters, aggregation=ci.AGG_TWO_STAGE,
                                                 official_a=ob, official_b=oa)
        rep["paired_interval_b_minus_a"] = iv
        rep["estimator"] = {"name": ci.PAIRED_ESTIMATOR, "unit": ci.CLUSTER_UNIT,
                            "question": "another draw of LOGS only; inference variance = the seed floors "
                                        "above; training variance untested (one run)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("a", "b", "paired_interval_b_minus_a")},
                     default=str))
    if "paired_interval_b_minus_a" in rep:
        iv = rep["paired_interval_b_minus_a"]
        print("paired interval:", {k: iv.get(k) for k in ("delta", "lo", "hi", "status", "separated")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
