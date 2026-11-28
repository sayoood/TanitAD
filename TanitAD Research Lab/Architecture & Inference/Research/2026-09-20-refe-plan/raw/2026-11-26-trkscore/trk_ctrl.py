#!/usr/bin/env python3
"""TRK-SCORE controls that read the run's own artifacts (PREREG_TRKSCORE.md section 6). NO PDMS is read here.

    trk_ctrl.py --trk-dump <trk.npz> --trk-seam <seam.npz> --trk-props <seam.proposals.npz> --banked-seam <stage2_p5c_c1.npz>
                --banked-props <stage2_p5c_c1.proposals.npz> --out <ctrl.json> [--expect-n 12146] [--red-all]
                [--parity --cache-root <metric_cache_navtest> --token-log <navtest_inputs.json.gz>]

CTRL-ID   k_id == k_ship on every token.
CTRL-DET  k_ship == banked pick; ship8 == banked seam row; the run's 64 planned props == the banked dump's proposals (all bit for bit).
CTRL-RED  (a) red_d0 >= 0.5 m on every RED token, (b) k_red != k_trk on >= 1 % of RED tokens. RED tokens: all (smoke, --red-all) or sha1 mod 10.
CTRL-SUB  (the submitted row is the PLANNED proposal k_trk) seam row == props[k_trk] == to_navsim(pick20) bit for bit, props pick == k_trk.
CTRL-SET  tokens of dump, seam and props identical; == the banked token set restricted to the run's set; n == --expect-n when given.
CTRL-PARITY (--parity; navsim-crun venv, PYTHONPATH = <navsim 3e8291b>;<PK/refe>;<PK/eval/instruments>): the dump's tracked states (keep-sim) vs
          the harness route from the metric cache's ego_state with PDM-Closed in row 0: |dx|,|dy| <= 1e-4 m, |dheading| <= 1e-4 rad; s0 equal <= 1e-9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

PKG = os.environ.get("TANITAD_REFE_PKG", "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")


def idx(tokens):
    return {str(t): i for i, t in enumerate(tokens)}


def seam_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("seam_c", os.path.join(PKG, "eval", "refe_navtest_seam.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--trk-dump", "--trk-seam", "--trk-props", "--banked-seam", "--banked-props", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--expect-n", type=int, default=0)
    ap.add_argument("--red-all", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--cache-root", default="F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest")
    ap.add_argument("--token-log", default="F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
    a = ap.parse_args()
    d = np.load(a.trk_dump, allow_pickle=False)
    ts = np.load(a.trk_seam, allow_pickle=False)
    tp = np.load(a.trk_props, allow_pickle=False)
    bs = np.load(a.banked_seam, allow_pickle=False)
    bp = np.load(a.banked_props, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    toks = [str(t) for t in ts["token"]]
    di, pi, bsi, bpi = idx(d["token"]), idx(tp["token"]), idx(bs["token"]), idx(bp["token"])
    out = {"meta": {k: meta[k] for k in ("n", "code", "red_mod", "red_all", "wrapped_rc", "t_trk_mean_s") if k in meta}}
    C = {}
    # CTRL-SET
    set_ok = (set(toks) == set(di) == set(pi)) and all(t in bsi and t in bpi for t in toks) and (not a.expect_n or len(toks) == a.expect_n)
    if a.expect_n:
        set_ok = set_ok and set(toks) == set(bsi)
    C["CTRL-SET"] = {"pass": bool(set_ok), "n_seam": len(toks), "n_dump": len(di), "n_props": len(pi), "n_banked": len(bsi), "expect_n": a.expect_n}
    if not set_ok:
        toks = [t for t in toks if t in di and t in pi and t in bsi and t in bpi]
    D = np.array([di[t] for t in toks])
    P = np.array([pi[t] for t in toks])
    BS = np.array([bsi[t] for t in toks])
    BP = np.array([bpi[t] for t in toks])
    k_ship, k_id, k_trk, k_red = d["k_ship"][D], d["k_id"][D], d["k_trk"][D], d["k_red"][D]
    # CTRL-ID
    C["CTRL-ID"] = {"pass": bool((k_id == k_ship).all()), "match": int((k_id == k_ship).sum()), "n": len(toks),
                    "id_maxdiff_max": float(d["id_maxdiff"][D].max())}
    # CTRL-DET
    pick_eq = bp["pick"][BP] == k_ship
    ship_eq = np.array([np.array_equal(d["ship8"][i], bs["poses"][j]) for i, j in zip(D, BS)])
    props_eq = np.array([np.array_equal(tp["proposals"][i], bp["proposals"][j]) for i, j in zip(P, BP)])
    props_maxd = float(max(np.abs(tp["proposals"][i].astype(np.float64) - bp["proposals"][j]).max() for i, j in zip(P, BP)))
    C["CTRL-DET"] = {"pass": bool(pick_eq.all() and ship_eq.all() and props_eq.all()), "pick_eq": int(pick_eq.sum()), "ship_row_eq": int(ship_eq.sum()),
                     "props_eq": int(props_eq.sum()), "props_max_abs_diff": props_maxd, "n": len(toks)}
    # CTRL-RED
    red = k_red >= 0
    d0 =d["red_d0"][D][red]
    sw = int((k_red[red] != k_trk[red]).sum())
    nr = int(red.sum())
    C["CTRL-RED"] = {"pass": bool(nr > 0 and (d0 >= 0.5).all() and sw >= math.ceil(0.01 * nr)), "red_tokens": nr, "red_d0_min": float(d0.min()) if nr else None,
                     "k_red_ne_k_trk": sw, "need_at_least": math.ceil(0.01 * nr) if nr else None}
    # CTRL-SUB (submitted = planned proposal k_trk; CTRL-T (a) is the to_navsim(pick20) half)
    sm = seam_mod()
    row_eq_prop = np.array([np.array_equal(ts["poses"][si], tp["proposals"][pi_][k]) for si, pi_, k in
                            zip([idx(ts["token"])[t] for t in toks], P, k_trk)])
    pick_is_trk = tp["pick"][P] == k_trk
    p20_ok = bool(sm.pick20_equals_seam(tp["pick20"][P], ts["poses"][[idx(ts["token"])[t] for t in toks]]))
    C["CTRL-SUB"] = {"pass": bool(row_eq_prop.all() and pick_is_trk.all() and p20_ok), "row_eq_planned_props_k_trk": int(row_eq_prop.sum()),
                     "props_pick_eq_k_trk": int(pick_is_trk.sum()), "pick20_equals_seam": p20_ok, "n": len(toks)}
    out["switch_k_trk_ne_k_ship"] = int((k_trk != k_ship).sum())
    # CTRL-SEAM (part of CTRL-SET in the card): the seam report and log are this run's (rows, misses, rule, A7, frame control, GPU)
    rp = os.path.splitext(a.trk_seam)[0] + ".report.json"
    rep = json.load(open(rp, encoding="utf-8"))
    lg = os.path.splitext(a.trk_dump)[0] + ".seam.log"
    cuda = os.path.isfile(lg) and "device=cuda" in open(lg, encoding="utf-8", errors="replace").read()
    fc = float(rep["frame_control"]["max_m"])
    fc_ok = abs(fc - 6.18063877792441) <= 1e-6 if a.expect_n == 12146 else True       # the full set's declared Deviation-1 max (p7_read_chain.py)
    C["CTRL-SEAM"] = {"pass": bool(rep.get("rows") == len(toks) and rep.get("misses") == 0 and rep.get("rule") == "navsim_v1"
                                   and rep.get("repair_last_heading") is True and fc_ok and cuda),
                      "rows": rep.get("rows"), "misses": rep.get("misses"), "frame_control_max_m": fc, "frame_control_declared_ok": fc_ok,
                      "seam_log_device_cuda": cuda, "ckpt": rep.get("ckpt"), "wrapped_rc": meta.get("wrapped_rc")}
    # CTRL-PARITY (smoke)
    if a.parity:
        sys.path.insert(0, os.path.join(PKG, "refe"))
        sys.path.insert(0, os.path.join(PKG, "eval", "instruments"))
        import transp_score as TSC
        import navsim_pdm_targets as PT
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
        if "sim" not in d.files:
            raise SystemExit("REFUSING --parity: the dump has no `sim` (run the seam with --keep-sim)")
        S = PT._setup()
        tlog = TSC.load_token_log(a.token_log)
        mx = {"xy": 0.0, "heading": 0.0, "s0": 0.0, "s0_pose": 0.0, "s0_rest_over_f32_bar": 0.0, "nobatch_vs_batch": 0.0}
        for t, i in zip(toks, D):
            pdm_traj, ego = TSC.load_cache_parts(a.cache_root, tlog[t], t)[:2]
            s0 = ego_state_to_state_array(ego)
            ds0 = np.abs(s0 - d["s0"][i])
            mx["s0"] = max(mx["s0"], float(ds0.max()))
            mx["s0_pose"] = max(mx["s0_pose"], float(ds0[:3].max()))                       # AMENDMENT_1: pose exact
            mx["s0_rest_over_f32_bar"] = max(mx["s0_rest_over_f32_bar"],                   # AMENDMENT_1: |d| / (1.2e-7 max(1,|cache|)) <= 1
                                             float((ds0[3:] / (1.2e-7 * np.maximum(1.0, np.abs(s0[3:])))).max()))
            inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
            for p in d["props8"][i]:
                it = transform_trajectory(Trajectory(np.asarray(p, np.float32), S["samp8"]), ego)
                inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
            simh = S["sim"].simulate_proposals(np.stack(inp, 0), ego)[1:]
            simn = S["sim"].simulate_proposals(np.stack(inp[1:], 0), ego)
            mine = d["sim"][i]
            mx["xy"] = max(mx["xy"], float(np.abs(simh[..., :2] - mine[..., :2]).max()))
            mx["heading"] = max(mx["heading"], float(np.abs((simh[..., 2] - mine[..., 2] + math.pi) % (2 * math.pi) - math.pi).max()))
            mx["nobatch_vs_batch"] = max(mx["nobatch_vs_batch"], float(np.abs(simn - simh).max()))
        C["CTRL-PARITY"] = {"pass": bool(mx["xy"] <= 1e-4 and mx["heading"] <= 1e-4 and mx["s0_pose"] == 0.0 and mx["s0_rest_over_f32_bar"] <= 1.0),
                            "as_written_pass": bool(mx["xy"] <= 1e-4 and mx["heading"] <= 1e-4 and mx["s0"] <= 1e-9),
                            "bar": "AMENDMENT_1 (sha256 d5fc1d57...): tracked |dx|,|dy| <= 1e-4 m, |dh| <= 1e-4 rad; start pose exact; other start components <= 1.2e-7 max(1,|cache|)",
                            "max": mx, "n": len(toks),
                            "what": "in-seam tracked states (seam venv, planner EgoState, 64 proposals only) vs the harness route (navsim-crun, metric cache ego_state, PDM-Closed in row 0)"}
    out["controls"] = C
    out["all_pass"] = all(v["pass"] for v in C.values())
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZTRK_CTRL " + " ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in C.items()) + f" all_pass={out['all_pass']}", flush=True)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
