#!/usr/bin/env python3
"""NAVTEST PORT (R8-GATE-NAVTEST AMENDMENT 1, 9cd2347): the same construction applied to the navtest bridge seams.
  python build_lever_seams_navtest.py <BRIDGE_DIR> <INPUTS_JSON_GZ> <OUT_DIR>
Only the paths and the split label differ from build_lever_seams.py (marked # NT); the arms, the K1-K6 controls and the
re-timing geometry are unchanged.  (original docstring follows)

Build the seam plans that would SCORE the levers PRIOR_GAP_DIAG cannot score offline (no navtrain metric cache / map on the dev box).
Pure CPU numpy+scipy.  Writes ONLY under OUT (a local package); touches nothing on Thor, nothing on G:.

  python build_lever_seams.py [RUN_DIR] [OUT_DIR]

All arms are built from refcv8's own banked R8_LEGAL rows (same inference seed s0 / s1) and the model's own emitted prior (`prior_emitted_path`),
using the SHIPPED re-timing stage's own geometry (retime_stage.py, sha-pinned deploy table) and E2's `knots_to_navsim`
(C2 not-a-knot cubic spline in time through the origin and the 8 knots at 0.5,1,1.5,2,3,4,5,6 s, evaluated at 0.5..4.0 s; heading = spline tangent,
held below 1 m/s) -- copied statement for statement from taniteval/bench/navsim/bridge.py and VERIFIED here to reproduce the banked seams.

A "path" is the polyline through (0,0) and the 8 knots; "progress" is the cumulative chord length along it at the 8 knot instants -- the same
decomposition SHIP-R8's re-timing stage uses.  s'(t_k) is the progress actually driven; the point is C_path(s'(t_k)).

  ARM (suffix _s1 = same construction on the seed-1 rows)         path        progress s'(t_k)
  ----------------------------------------------------------      ----------  ---------------------------------------------
  [scored] PRIOR_ha0p                                             prior       prior            (alpha=0)
  [scored] R8_LEGAL                                               R8          R8               (alpha=1)
  [scored] R8_RT                                                  R8          prior + g (R8 - prior), g = deploy table
  L0   R8PATH_PRIORPROG     keep refcv8's LATERAL path            R8          prior            (g = 0)
  L50  R8PATH_HALFPROG                                            R8          prior + 0.5 (R8 - prior)
  P1   PRIORPATH_R8PROG     keep refcv8's SPEED profile           prior       R8 (LEGAL's own chord length)
  P2   PRIORPATH_RTPROG                                           prior       the RT progress
  C1   R8PATH_MINPROG       keep refcv8's path, CAP its speed    R8          min(R8, prior)  (never faster than the prior)
  C2   R8PATH_MAXPROG       (mirror, expected-worse control)     R8          max(R8, prior)
  A25 / A50 / A75  ALPHAxxx plan = prior + alpha (R8 - prior) in knot space, both axes (the literal "scale the residual down")
"""
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

BR_ARG, INPUTS_ARG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]   # NT
NS = "C:/Users/Admin/refcv8_pf/navsim_ckpt"
STAGE = f"{NS}/thor_bundle/stage/retime/retime_stage.py"
TABLE = f"{NS}/thor_bundle/stage/retime/refcv8_retime_deploy_gains_pf.json"
DEPLOY_SHA = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"        # RT_ASSERT.json: deploy_sha256
BR = BR_ARG   # NT
SEAMDIR = os.path.join(OUT, "thor_seams")
os.makedirs(SEAMDIR, exist_ok=True)

KNOT_T = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
NAVSIM_T = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
HEADING_HOLD = 1.0


def knots_to_navsim(knots_xy, knot_t=KNOT_T, out_t=NAVSIM_T):
    """VERBATIM port of taniteval.bench.navsim.bridge.knots_to_navsim (see module docstring)."""
    k = np.asarray(knots_xy, dtype=np.float64)
    t = np.concatenate([[0.0], np.asarray(knot_t, dtype=np.float64)])
    p = np.vstack([[0.0, 0.0], k])
    cs = CubicSpline(t, p, axis=0, bc_type="not-a-knot")
    q = np.asarray(out_t, dtype=np.float64)
    xy = cs(q)
    d = cs(q, 1)
    head = np.zeros(len(q))
    prev = 0.0
    for i in range(len(q)):
        sp = float(np.hypot(d[i, 0], d[i, 1]))
        if sp >= HEADING_HOLD:
            a = float(np.arctan2(d[i, 1], d[i, 0]))
            prev = prev + ((a - prev + math.pi) % (2 * math.pi) - math.pi)
        head[i] = prev
    return np.column_stack([xy, head])


spec = importlib.util.spec_from_file_location("retime_stage_pf", STAGE)
RS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RS)
table = RS.load_gains_table(TABLE, DEPLOY_SHA)               # REFUSES a table whose bytes do not hash to the pinned sha
G = RS.gain_at(np.asarray(KNOT_T, np.float64), table)

inp = json.load(gzip.open(INPUTS_ARG))["tokens"]   # NT
rt0 = np.load(os.path.join(BR, "seam_R8_RT.npz"), allow_pickle=False)
ORDER = [str(t) for t in rt0["token"]]
N = len(ORDER)
HUM = np.array([inp[t]["human_future_poses"] for t in ORDER], np.float64)


def rows(name):
    out = {}
    for ln in open(os.path.join(BR, f"rows_{name}.jsonl")):
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r
    return out


def build(seed_tag):
    src = rows("R8_LEGAL" + seed_tag)
    legal = np.array([src[t]["knots"] for t in ORDER], np.float64)                      # [N,8,2] (the model's float32 values)
    prior = np.array([src[t]["prior_emitted_path"] for t in ORDER], np.float64)         # [N,8,2]
    legal_poses = np.array([src[t]["poses"] for t in ORDER], np.float64)
    return legal, prior, legal_poses


def seam_arrays(name_rt):
    z = np.load(os.path.join(BR, f"seam_{name_rt}.npz"), allow_pickle=False)
    assert [str(t) for t in z["token"]] == ORDER
    return z


results = {"controls": {}, "arms": {}}
for tag, rt_name, lg_name in (("", "R8_RT", "R8_LEGAL"), ("_s1", "R8_RT_s1", "R8_LEGAL_s1")):
    legal, prior, legal_poses = build(tag)
    zrt, zlg = seam_arrays(rt_name), seam_arrays(lg_name)
    zpr = np.load(os.path.join(BR, "seam_PRIOR_ha0p.npz"), allow_pickle=False)
    geo_l = RS.polyline_geometry(legal)
    geo_p = RS.polyline_geometry(prior)
    L = geo_l["L"][:, 1:]                       # R8 (LEGAL) cumulative chord length at the 8 instants
    M = RS.prior_arc(prior)                     # prior's

    def to_seam(knots):
        k32 = np.asarray(knots, np.float32)
        poses = np.stack([knots_to_navsim(k32[i]) for i in range(N)]).astype(np.float32)
        return k32, poses

    # ---- controls: the machinery reproduces the banked seams
    c = {}
    k_lg, p_lg = to_seam(legal)
    c["K1_knots_to_navsim(LEGAL knots) == banked LEGAL poses, max abs"] = float(np.abs(p_lg - zlg["poses"]).max())
    rt_knots = RS.retime_plan(legal.astype(np.float32), prior.astype(np.float32), np.asarray(KNOT_T, np.float64), table, cast=True)
    c["K2_retime_plan(LEGAL, prior, deploy table) == banked RT knots, max abs"] = float(np.abs(rt_knots.astype(np.float32) - zrt["knots"]).max())
    k_rt, p_rt = to_seam(rt_knots)
    c["K2b_RT poses rebuilt == banked RT poses, max abs"] = float(np.abs(p_rt - zrt["poses"]).max())
    k_pr, p_pr = to_seam(prior)
    c["K3_prior_emitted knots -> poses == banked PRIOR_ha0p poses, max abs"] = float(np.abs(p_pr - zpr["poses"]).max())
    P_g1, _ = RS.retime_positions(geo_l, M, np.ones(8))
    c["K4_g=1 retime_positions == LEGAL knots, max abs"] = float(np.abs(P_g1 - legal).max())
    c["K5_prior path with prior progress == prior knots, max abs"] = float(np.abs(np.stack([RS.eval_polyline(geo_p, M[:, k]) for k in range(8)], 1) - prior).max())
    c["K6_banked PRIOR_ha0p knots vs model's emitted prior, max abs (same token order)"] = float(np.abs(zpr["knots"].astype(np.float64) - prior).max())
    c["pass"] = bool(all(v < 2e-5 for kk, v in c.items() if kk != "pass"))
    results["controls"]["s0" if tag == "" else "s1"] = c

    # ---- the new arms (float64 knots; cast to float32 when written, as the bridge does)
    arms = {}
    P_L0, _ = RS.retime_positions(geo_l, M, np.zeros(8))                                   # R8 path, prior progress
    P_L50, _ = RS.retime_positions(geo_l, M, np.full(8, 0.5))
    P_P1 = np.stack([RS.eval_polyline(geo_p, L[:, k]) for k in range(8)], 1)               # prior path, R8 progress
    s_rt = RS.retime_progress(L, M, G)
    P_P2 = np.stack([RS.eval_polyline(geo_p, s_rt[:, k]) for k in range(8)], 1)            # prior path, RT progress
    arms["R8PATH_PRIORPROG"] = P_L0
    arms["R8PATH_HALFPROG"] = P_L50
    arms["PRIORPATH_R8PROG"] = P_P1
    arms["PRIORPATH_RTPROG"] = P_P2
    # CAP: never drive faster than the prior along refcv8's own path (s' = min(L, M)); MAX is its mirror and the expected-worse control
    arms["R8PATH_MINPROG"] = np.stack([RS.eval_polyline(geo_l, np.minimum(L, M)[:, k]) for k in range(8)], 1)
    arms["R8PATH_MAXPROG"] = np.stack([RS.eval_polyline(geo_l, np.maximum(L, M)[:, k]) for k in range(8)], 1)
    for a_ in (25, 50, 75):
        arms[f"ALPHA{a_:03d}"] = prior + (a_ / 100.0) * (legal - prior)
    # the two hybrid sanity arms that MUST equal banked, scored seams (identity checks, never scored again)
    results["controls"]["s0" if tag == "" else "s1"]["ALPHA000_equals_prior_max_abs"] = float(np.abs((prior + 0.0 * (legal - prior)) - prior).max())
    for nm, kn in arms.items():
        k32, poses = to_seam(kn)
        lab = nm + tag
        np.savez(os.path.join(SEAMDIR, f"seam_{lab}.npz"), token=zrt["token"], fingerprint=zrt["fingerprint"], source=zrt["source"], poses=poses, knots=k32,
                 sampling=np.asarray([8, 0.5]), arm=np.asarray(lab))
        spec_txt = {"R8PATH_PRIORPROG": "refcv8's PATH (LEGAL polyline) driven at the PRIOR's progress s'=M (g=0)",
                    "R8PATH_HALFPROG": "refcv8's PATH driven at s' = M + 0.5 (L - M)",
                    "PRIORPATH_R8PROG": "the PRIOR's path driven at refcv8's (LEGAL) progress s' = L",
                    "PRIORPATH_RTPROG": "the PRIOR's path driven at the shipped RT progress s' = M + g (L - M), g = deploy table",
                    "R8PATH_MINPROG": "refcv8's PATH driven at s' = min(L, M): never faster than the prior (a progress CAP)",
                    "R8PATH_MAXPROG": "refcv8's PATH driven at s' = max(L, M): never slower than the prior (the expected-worse mirror of the cap)",
                    "ALPHA025": "plan = prior + 0.25 (R8_LEGAL knots - prior knots)", "ALPHA050": "plan = prior + 0.50 (R8_LEGAL knots - prior knots)",
                    "ALPHA075": "plan = prior + 0.75 (R8_LEGAL knots - prior knots)"}[nm]
        man = {"split": "navtest",   # NT
                "label": "PRIOR_GAP_DIAG lever arm (POST-HOC diagnostic, decides nothing)", "arm": lab, "spec": spec_txt,
               "derived_from": f"rows_R8_LEGAL{tag}.jsonl (emitted plan knots + emitted prior); no forward", "n_expected": N, "n_rows": N, "partial": False,
               "n_missing": 0, "n_model_rows": N, "n_cv_standin_rows": 0, "retime_table_sha256": table["sha256"],
               "tool": "build_lever_seams.py (NEW; needs Master-Mind acceptance before it produces a decision row)"}
        json.dump(man, open(os.path.join(SEAMDIR, f"seam_{lab}.manifest.json"), "w"), indent=1)
        results["arms"][lab] = {"spec": spec_txt, "max_abs_knot_change_vs_LEGAL_m": float(np.abs(k32.astype(np.float64) - legal).max()),
                                "max_abs_knot_change_vs_prior_m": float(np.abs(k32.astype(np.float64) - prior).max()),
                                "mean_abs_pose_x4s_change_vs_prior_m": float(np.abs(poses[:, 7, 0].astype(np.float64) - zpr["poses"][:, 7, 0]).mean()),
                                "seam_sha256": hashlib.sha256(open(os.path.join(SEAMDIR, f"seam_{lab}.npz"), "rb").read()).hexdigest(),
                                "ade_to_human_m_mean": float(np.hypot(poses[:, :, 0] - HUM[:, :, 0], poses[:, :, 1] - HUM[:, :, 1]).mean())}

results["reference_ade_to_human_m_mean"] = {}
for nm_, key_ in (("PRIOR_ha0p", "seam_PRIOR_ha0p"), ("R8_LEGAL", "seam_R8_LEGAL"), ("R8_RT", "seam_R8_RT")):
    pz = np.load(os.path.join(BR, key_ + ".npz"))["poses"].astype(np.float64)
    results["reference_ade_to_human_m_mean"][nm_] = float(np.hypot(pz[:, :, 0] - HUM[:, :, 0], pz[:, :, 1] - HUM[:, :, 1]).mean())

lines = []   # NT: no Thor job file here (navtest arms are scored through r8nt_score.py jobs)
json.dump(results, open(os.path.join(OUT, "lever_seams_build_report.json"), "w"), indent=1)
ok = all(v["pass"] for v in results["controls"].values())
print("controls pass:", ok, {k: {kk: round(vv, 8) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in results["controls"].items()})
print("arms:", len(results["arms"]), "jobs:", len(lines))
sys.exit(0 if ok else 3)
