#!/usr/bin/env python3
"""r8_gate_eval -- R8-GATE-NAVTEST (GOALS_AND_CLAIMS, pre-registered 0c5b5db, 2026-10-21 09:38 Berlin, before any refcv8 navtest score).

Two prior-fallback gates built per token from rows a NavSim read already produced, scored EXACTLY by per-token selection (the PDM
scorer scores each token independently, so a selected plan's row IS that arm's row for the token):
    G_CMD = PRIOR_ha0p where the NavSim driving_command is straight (one-hot index 1, 'forward'), else R8_RT
    G_ARC = PRIOR_ha0p where |ln(arc4(R8_RT) / max(arc4(PRIOR_ha0p), 0.05))| > THR (0.1368, fixed on navtrain), else R8_RT
    G_CMDCAP (AMENDMENT 1, 9cd2347) = PRIOR_ha0p where the command is straight, else R8PATH_MINPROG (refcv8's LEGAL path driven at
             min(R8, prior) progress; build_lever_seams_navtest.py) -- evaluated only when --cap-csv / --cap-csv-s1 are given
Seed 1 uses R8_RT_s1 with the SAME threshold.  arc4 = cumulative chord length through the origin to pose index 7 (4.0 s).

Criterion (registered): WORKS iff G - PRIOR > 0 with the paired log-cluster bootstrap (B 2000) interval excluding 0 at log_name AND
nuplan_drive (bars8 ``compact_pair`` 'positive'), at BOTH inference seeds, and the margin min(d_s0, d_s1) > 2 x |d_s0 - d_s1|;
FAILED if the point is <= 0 at either seed; otherwise NOT PROVEN.  Diagnostic lever test, NEVER a bar.

The estimator and the CSV reader are IMPORTED from navsim_bars8 (pinned navsim_ci identity), never re-implemented.
CPU only.  Tier: NavSim v1 open-loop PDM scorer (T1-family), one training seed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

BARS8 = "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py"
THR = 0.1368
IDX4 = 7
SG_MAP = "sg-one-north"


def load_bars8(path=BARS8):
    spec = importlib.util.spec_from_file_location("navsim_bars8_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["navsim_bars8_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def seam_poses(npz, toks):
    z = np.load(npz, allow_pickle=False)
    order = {str(t): i for i, t in enumerate(z["token"])}
    miss = [t for t in toks if t not in order]
    if miss:
        raise SystemExit(f"{npz}: lacks {len(miss)} tokens")
    return z["poses"][[order[t] for t in toks]].astype(np.float64)


def arc4(p):
    seg = np.diff(np.concatenate([np.zeros((p.shape[0], 1, 2)), p[:, :, :2]], 1), axis=1)
    return np.cumsum(np.hypot(seg[..., 0], seg[..., 1]), axis=1)[:, IDX4]


def select(rt: pd.DataFrame, pr: pd.DataFrame, toks, fallback: np.ndarray) -> pd.DataFrame:
    a, b = rt.loc[toks], pr.loc[toks]
    out = a.copy()
    out.loc[np.asarray(toks)[fallback]] = b.loc[np.asarray(toks)[fallback]].to_numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-csv", required=True, help="scorer CSV of R8_RT (seed 0)")
    ap.add_argument("--rt-csv-s1", required=True, help="scorer CSV of R8_RT_s1")
    ap.add_argument("--prior-csv", required=True, help="scorer CSV of PRIOR_ha0p")
    ap.add_argument("--seam-rt", required=True)
    ap.add_argument("--seam-rt-s1", required=True)
    ap.add_argument("--seam-prior", required=True)
    ap.add_argument("--inputs", required=True, help="inputs .json.gz (token -> log_name, map_name, driving_command)")
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--cap-csv", default="", help="scorer CSV of R8PATH_MINPROG (seed 0) -> G_CMDCAP")
    ap.add_argument("--cap-csv-s1", default="", help="scorer CSV of R8PATH_MINPROG_s1")
    ap.add_argument("--label", required=True, help="e.g. NAVTEST_RESULT or NAVTRAIN_KNOWN_VALUE_CONTROL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if abs(a.thr - THR) > 1e-12 and "CONTROL" not in a.label:
        raise SystemExit(f"the registered threshold is {THR}; a different one is allowed only on a CONTROL label")
    B = load_bars8()
    ci = B.load_ci()
    cfg = {**B.DEFAULT_CFG}
    rt0, rt1, pr = B.read_scores_csv(a.rt_csv), B.read_scores_csv(a.rt_csv_s1), B.read_scores_csv(a.prior_csv)
    toks = sorted(set(rt0.index) & set(rt1.index) & set(pr.index))
    if not (len(toks) == len(rt0) == len(rt1) == len(pr)):
        raise SystemExit(f"token sets differ: RT {len(rt0)} RT_s1 {len(rt1)} PRIOR {len(pr)} common {len(toks)}")
    ti = B.load_inputs(a.inputs)
    miss = [t for t in toks if t not in ti.index]
    if miss:
        raise SystemExit(f"{len(miss)} tokens absent from inputs")
    ti = ti.loc[toks]
    cmd = np.array([B.command_key(c) for c in ti["command"]])
    if (cmd == "malformed").any():
        raise SystemExit("malformed driving_command one-hot")
    straight = cmd == "forward"
    clusters = [str(x) for x in ti["log"]]
    P_pr = seam_poses(a.seam_prior, toks)
    res = {"id": "R8-GATE-NAVTEST", "label": a.label, "threshold": a.thr, "n_tokens": len(toks), "n_logs": len(set(clusters)),
           "estimator": {"paired": ci.PAIRED_ESTIMATOR, "cluster_unit": ci.CLUSTER_UNIT, "sensitivity_unit": ci.SENSITIVITY_UNIT,
                         "n_boot": cfg["n_boot"], "seed": cfg["seed"], "navsim_ci_sha256": ci.__file_sha256__},
           "tier": "NavSim v1 open-loop PDM scorer (T1-family); one training seed; diagnostic lever test, never a bar",
           "n_straight": int(straight.sum()), "arms": {}, "pairs": {}, "controls": {}}
    gates = {}
    for s, rt, seam in (("s0", rt0, a.seam_rt), ("s1", rt1, a.seam_rt_s1)):
        P_rt = seam_poses(seam, toks)
        lr = np.abs(np.log(np.maximum(arc4(P_rt) / np.maximum(arc4(P_pr), 0.05), 0.05)))
        fb_arc = lr > a.thr
        gates[f"G_CMD_{s}"] = (select(rt, pr, toks, straight), straight)
        gates[f"G_ARC_{s}"] = (select(rt, pr, toks, fb_arc), fb_arc)
        # control: a never-firing gate IS the RT arm, an always-firing gate IS the prior (exact)
        never = select(rt, pr, toks, np.zeros(len(toks), bool))
        always = select(rt, pr, toks, np.ones(len(toks), bool))
        res["controls"][f"never_fires_equals_RT_{s}"] = bool(np.array_equal(never.to_numpy(), rt.loc[toks].to_numpy()))
        res["controls"][f"always_fires_equals_PRIOR_{s}"] = bool(np.array_equal(always.to_numpy(), pr.loc[toks].to_numpy()))
        res["arms"][f"R8_RT_{s}"] = {m: round(100 * float(rt.loc[toks, m].mean()), 4) for m in B.SUBS + ("PDMS",)}
    cap = {}
    if a.cap_csv and a.cap_csv_s1:
        for s, pth in (("s0", a.cap_csv), ("s1", a.cap_csv_s1)):
            cdf = B.read_scores_csv(pth)
            if sorted(cdf.index) != toks:
                raise SystemExit(f"{pth}: token set differs ({len(cdf)} rows)")
            cap[s] = cdf
            gates[f"G_CMDCAP_{s}"] = (select(cdf, pr, toks, straight), straight)
            res["arms"][f"R8PATH_MINPROG_{s}"] = {m: round(100 * float(cdf.loc[toks, m].mean()), 4) for m in B.SUBS + ("PDMS",)}
    res["arms"]["PRIOR_ha0p"] = {m: round(100 * float(pr.loc[toks, m].mean()), 4) for m in B.SUBS + ("PDMS",)}
    for g, (df, fb) in gates.items():
        res["arms"][g] = {**{m: round(100 * float(df[m].mean()), 4) for m in B.SUBS + ("PDMS",)},
                          "fallback_frac": round(float(fb.mean()), 4), "n_fallback": int(fb.sum())}
        res["pairs"][f"{g}__minus__PRIOR"] = B.pair_record(df, pr, toks, clusters, cfg)
        s = g[-2:]
        res["pairs"][f"{g}__minus__R8_RT"] = B.pair_record(df, rt0 if s == "s0" else rt1, toks, clusters, cfg)
        # strata (beside): city and command
        for nm, mask in (("singapore", (ti["map_name"] == SG_MAP).to_numpy()), ("us", (ti["map_name"] != SG_MAP).to_numpy()),
                         ("straight", straight), ("turn", ~straight)):
            if mask.sum() >= 1:
                st = [t for t, k in zip(toks, mask) if k]
                cl = [c for c, k in zip(clusters, mask) if k]
                res["pairs"][f"{g}__minus__PRIOR__{nm}"] = B.pair_record(df, pr, st, cl, cfg)
    # control: G - G reads exactly 0
    g0 = gates["G_CMD_s0"][0]
    res["controls"]["G_minus_G_zero"] = B.pair_record(g0, g0, toks, clusters, cfg)["delta_x100"] == 0.0
    verdicts = {}
    for gname in ("G_CMD", "G_ARC") + (("G_CMDCAP",) if cap else ()):
        d0 = res["pairs"][f"{gname}_s0__minus__PRIOR"]
        d1 = res["pairs"][f"{gname}_s1__minus__PRIOR"]
        p0, p1 = d0["delta_x100"], d1["delta_x100"]
        floor = abs(p0 - p1)
        pos = d0["interval"].get("positive") and d1["interval"].get("positive")
        if p0 <= 0 or p1 <= 0:
            v = "FAILED"
        elif pos and min(p0, p1) > 2 * floor:
            v = "WORKS"
        else:
            v = "NOT PROVEN"
        verdicts[gname] = {"verdict": v, "delta_s0": p0, "delta_s1": p1, "inference_floor": round(floor, 4),
                           "both_positive_separated": bool(pos), "margin": round(min(p0, p1), 4), "needed": round(2 * floor, 4)}
    res["verdicts"] = verdicts
    res["controls"]["all_pass"] = all(v is True for k, v in res["controls"].items() if k != "all_pass")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("QQGATE " + json.dumps({"label": a.label, "verdicts": verdicts, "controls_all_pass": res["controls"]["all_pass"],
                                  "PRIOR": res["arms"]["PRIOR_ha0p"]["PDMS"], "RT_s0": res["arms"]["R8_RT_s0"]["PDMS"]}) + " QQ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
