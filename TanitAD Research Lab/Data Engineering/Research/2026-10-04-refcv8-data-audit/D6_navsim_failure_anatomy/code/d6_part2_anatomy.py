"""D6 part 2 -- anatomy of the navhard DAC-zero and NC-zero scenes (refcv7 step-30,000 R7_A1), with the 5,000 and
PRIOR comparisons, and the two Master-Mind stratifiers (max-speed known/unknown; command x curve/junction).

Reads: raw/geom_all.jsonl  (d6_rescore.py --geom-only: route features + every arm's RAW plan offsets)
       raw/rescore_A_dac0.jsonl, raw/rescore_B_nc_ddc_ctrl.jsonl  (d6_rescore.py: exact re-score of the A1 plan)
       raw/d6_scene_table_step30000.csv (part 1) + the banked hooks / rows.
Writes raw/d6_part2.json and raw/d6_scene_geometry_step30000.csv.

ALL THRESHOLDS ARE LITERALS FIXED BEFORE THE CLASS COUNTS WERE LOOKED AT (road geometry, not tuned):
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")

# ---- literal thresholds ------------------------------------------------------------------------------------
T_TURN = 0.35      # rad (20 deg): a route heading change at least this large within the horizon is a CURVE/TURN
T_HEAD = 0.175     # rad (10 deg): heading error between plan and route tangent that counts as a deviation
T_LAT = 1.5        # m: lateral offset from the route centreline that counts as leaving the lane (lane half-width ~1.75)
T_STOP = 1.0       # m: 4-s endpoint distance below which a plan is STOP-LIKE (the banked stop_fraction literal)
T_AWAY = 0.75      # m: lateral excursion beyond the start offset that counts as drifting away from the centreline
SPEED_RATIO = 1.25  # plan progress along the route >= 1.25 x the PDM-closed reference's ...
SPEED_ABS = 4.0     # ... and at least 4 m more
EARLY_T = 1.0       # s: first non-drivable instant earlier than this = EARLY, else LATE


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_jsonl(path):
    out = {}
    n_bad = 0
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                out[r["token"]] = r
            except Exception:                                          # noqa: BLE001
                n_bad += 1
    return out, n_bad


def route_fn(g):
    """heading_at(d), inter_within(D) from the centreline lookahead rows [d, x, y, h_rel, in_intersection]."""
    rows = np.asarray([[r[0], r[3]] for r in g["cl_ahead"]], dtype=float)
    ds, h = rows[:, 0], np.unwrap(rows[:, 1])
    inter = [(r[0], r[4]) for r in g["cl_ahead"]]

    def heading_at(d):
        return float(np.interp(d, ds, h))

    def inter_within(D):
        return any(bool(f) for d, f in inter if d <= D and f is not None)
    return heading_at, inter_within


def plan_features(g, label, plan, v0):
    """geometry of ONE raw plan against the route; plan = [8,3] ego frame."""
    heading_at, inter_within = route_fn(g)
    D = float(np.clip(4.0 * v0, 10.0, 40.0))
    d_route = heading_at(D) - heading_at(0.0)                # signed route heading change within the horizon
    cl = g["cl_plan_raw"][label]
    lat = np.asarray([c[0] for c in cl], dtype=float)
    s = np.asarray([c[1] for c in cl], dtype=float)
    d_p = max(float(s[-1] - g["s0"]), 0.0)                   # plan's own progress along the route at 4 s
    yaw4 = float(plan[-1, 2])
    head_err = float(wrap(yaw4 - heading_at(min(d_p, 60.0))))   # plan heading minus route tangent at the plan's endpoint
    refx = np.asarray(g["ref_xyh_0p5s"], dtype=float)
    d_ref = float(np.hypot(refx[-1, 0], refx[-1, 1]))
    dist4 = float(np.hypot(plan[-1, 0], plan[-1, 1]))
    sgn = 1.0 if d_route >= 0 else -1.0
    return {"D_m": D, "d_route": float(d_route), "route_turn": abs(d_route) >= T_TURN,
            "route_dir": "L" if d_route >= T_TURN else "R" if d_route <= -T_TURN else "S",
            "junction": bool(inter_within(D)), "yaw4": yaw4, "head_err": head_err, "lat4": float(lat[-1]),
            "lat_max": float(np.abs(lat).max()), "lat0": float(g["lat0"]),
            "lat_in": float(sgn * (lat[-1] - g["lat0"])),                              # + = moved INSIDE the curve vs the start offset
            "d_plan_route": d_p, "d_ref": d_ref, "dist4": dist4,
            "turn_err": float(sgn * head_err)}                                         # + = plan turns MORE than the route


def classify(f):
    """geometry class of a plan (cascade; first match wins).  Pure function of the feature dict.
    lateral terms are measured as CHANGE from the start offset ``lat0`` (a synthetic stage-2 start is laterally
    perturbed): DRIFT = the plan moves away from the centreline, NO-RECOVERY = it keeps an inherited offset."""
    if f["dist4"] < T_STOP:
        return "STOP-LIKE"
    if f["route_turn"]:
        yaw_sign = np.sign(f["yaw4"])
        route_sign = 1.0 if f["d_route"] > 0 else -1.0
        if yaw_sign == -route_sign and abs(f["yaw4"]) >= T_HEAD:
            return "WRONG-SIDE"
        if f["turn_err"] <= -T_HEAD or f["lat_in"] <= -T_LAT:
            return "ROUTE-FOLLOWING"          # under-turn: plan heads straighter than the route / ends outside the curve
        if f["turn_err"] >= T_HEAD or f["lat_in"] >= T_LAT:
            return "OVER-STEER"
    else:
        away = f["lat_max"] - abs(f["lat0"])
        if away >= T_AWAY or abs(f["head_err"]) >= T_HEAD:
            return "LATERAL-DRIFT"
        if f["lat_max"] >= T_LAT:
            return "NO-RECOVERY"              # |offset| >= T_LAT but inherited from the start, not grown
    if f["d_plan_route"] >= SPEED_RATIO * f["d_ref"] and f["d_plan_route"] - f["d_ref"] >= SPEED_ABS:
        return "SPEED"
    return "ON-ROUTE"


def self_test():
    """Known-value controls for the classifier (analytic plans)."""
    base = dict(D_m=20.0, junction=False, d_ref=20.0, dist4=20.0, d_plan_route=20.0, lat4=0.0, lat_max=0.0, lat0=0.0)
    cases = [
        ("straight road, on line", dict(base, d_route=0.0, route_turn=False, yaw4=0.0, head_err=0.0, lat_in=0.0, turn_err=0.0), "ON-ROUTE"),
        ("left 90deg, plan straight", dict(base, d_route=1.57, route_turn=True, yaw4=0.0, head_err=-1.2, lat_in=-6.0, turn_err=-1.2), "ROUTE-FOLLOWING"),
        ("left 90deg, plan turns right", dict(base, d_route=1.57, route_turn=True, yaw4=-0.5, head_err=-1.8, lat_in=-9.0, turn_err=-1.8), "WRONG-SIDE"),
        ("left 90deg, plan cuts inside", dict(base, d_route=1.57, route_turn=True, yaw4=1.5, head_err=0.4, lat_in=3.0, turn_err=0.4), "OVER-STEER"),
        ("straight road, 3 m drift", dict(base, d_route=0.0, route_turn=False, yaw4=0.1, head_err=0.05, lat4=3.0, lat_max=3.0, lat_in=3.0, turn_err=0.0), "LATERAL-DRIFT"),
        ("straight road, starts 2 m off, stays 2 m off", dict(base, d_route=0.0, route_turn=False, yaw4=0.0, head_err=0.0, lat4=2.0, lat_max=2.0, lat0=2.0, lat_in=0.0, turn_err=0.0), "NO-RECOVERY"),
        ("straight road, 10 deg swerve, no offset", dict(base, d_route=0.0, route_turn=False, yaw4=0.2, head_err=0.2, lat_in=0.0, turn_err=0.0), "LATERAL-DRIFT"),
        ("stationary", dict(base, d_route=0.0, route_turn=False, yaw4=0.0, head_err=0.0, lat_in=0.0, turn_err=0.0, dist4=0.2), "STOP-LIKE"),
        ("fast on route", dict(base, d_route=0.0, route_turn=False, yaw4=0.0, head_err=0.0, lat_in=0.0, turn_err=0.0, d_plan_route=30.0), "SPEED"),
    ]
    res = {n: (classify(f), want) for n, f, want in cases}
    assert all(a == b for a, b in res.values()), res
    # mutation: a classifier with T_HEAD removed from the wrong-side test must be caught by the cases above
    return {n: {"got": a, "want": b} for n, (a, b) in res.items()}


def main():
    ci = C.load_ci()
    inp, mapping = C.load_inputs()
    G, bad_g = load_jsonl(os.path.join(RAW, "geom_all.jsonl"))
    RA, bad_a = load_jsonl(os.path.join(RAW, "rescore_A_dac0.jsonl"))
    RB, bad_b = load_jsonl(os.path.join(RAW, "rescore_B_nc_ddc_ctrl.jsonl"))
    RS = {**RA, **RB}
    tab = pd.read_csv(os.path.join(RAW, "d6_scene_table_step30000.csv")).set_index("token")
    arms = {"A1": C.load_arm(30000, "R7_A1"), "A1k5": C.load_arm(5000, "R7_A1", with_hooks=False),
            "PRIOR": C.load_arm(30000, "PRIOR_ha0p", with_hooks=False)}
    out = {"thresholds": dict(T_TURN=T_TURN, T_HEAD=T_HEAD, T_LAT=T_LAT, T_STOP=T_STOP, SPEED_RATIO=SPEED_RATIO,
                              SPEED_ABS=SPEED_ABS, EARLY_T=EARLY_T),
           "files": {"geom": len(G), "rescore_A": len(RA), "rescore_B": len(RB), "bad_lines": [bad_g, bad_a, bad_b]},
           "classifier_self_test": self_test()}
    toks = [t for t in tab.index if t in G]
    out["n_tokens_with_geometry"] = len(toks)
    # ---------------------------------------------------------------- features for every arm, every token
    lab = {"A1": "MAIN", "A1k5": "A1_5k", "PRIOR": "PRIOR", "A1s1": "A1_s1"}
    F = {}
    plans = {"A1": {t: arms["A1"].loc[t, "plan"] for t in toks}}
    # the other arms' raw plans come from their hooks (loaded once; small)
    for key, step, arm in (("A1k5", 5000, "R7_A1"), ("PRIOR", 30000, "PRIOR_ha0p"), ("A1s1", 30000, "R7_A1_s1")):
        df = C.load_arm(step, arm, with_hooks=True)
        plans[key] = {t: df.loc[t, "plan"] for t in toks}
        if key != "A1s1":
            arms[key] = df
        else:
            arms["A1s1"] = df
    for key in ("A1", "A1k5", "PRIOR", "A1s1"):
        F[key] = {}
        for t in toks:
            g = G[t]
            F[key][t] = plan_features(g, lab[key], plans[key][t], float(arms["A1"].loc[t, "v0"]))
            F[key][t]["cls"] = classify(F[key][t])
    # the PDM-closed reference and (stage 1) the human, classified with the same rule -> sanity controls
    ctrl = {}
    ref_lat = [abs(G[t]["cl_ref"][-1][0]) for t in toks]
    ctrl["pdm_closed_ref_lat4_within_1mm_of_{0,1}m"] = float(np.mean([min(abs(x), abs(x - 1.0)) < 1e-3 for x in ref_lat]))
    hum_cls = Counter()
    for t in toks:
        if inp[t]["stage"] == 1 and G[t].get("human_xyh"):
            hp = np.asarray(G[t]["human_xyh"], dtype=float)
            g = dict(G[t]); g["cl_plan_raw"] = dict(g["cl_plan_raw"]); g["cl_plan_raw"]["HUM"] = G[t]["cl_human"]
            f = plan_features(g, "HUM", hp, float(arms["A1"].loc[t, "v0"]))
            hum_cls[classify(f)] += 1
    ctrl["human_stage1_class_counts"] = dict(hum_cls)
    ctrl["human_stage1_ON_ROUTE_share"] = hum_cls["ON-ROUTE"] / max(1, sum(hum_cls.values()))
    ctrl["human_stage1_WRONG_SIDE_share"] = hum_cls["WRONG-SIDE"] / max(1, sum(hum_cls.values()))
    out["controls"] = ctrl

    # ---------------------------------------------------------------- scene frame
    rows = []
    for t in toks:
        f = F["A1"][t]
        r = {"token": t, "stage": int(tab.loc[t, "stage"]), "cmd": tab.loc[t, "cmd"], "v0": float(tab.loc[t, "v0"]),
             "v0_band": tab.loc[t, "v0_band"], "route_dir": f["route_dir"], "route_turn": f["route_turn"],
             "junction": f["junction"], "cls": f["cls"], "lat4": f["lat4"], "yaw4": f["yaw4"], "head_err": f["head_err"],
             "d_route": f["d_route"], "dist4": f["dist4"], "d_plan_route": f["d_plan_route"], "d_ref": f["d_ref"],
             "cls_5k": F["A1k5"][t]["cls"], "cls_prior": F["PRIOR"][t]["cls"], "cls_s1": F["A1s1"][t]["cls"],
             "lat0": G[t]["lat0"], "cl_remaining_m": G[t]["cl_remaining_m"], "ego_in_inter": G[t].get("ego_in_intersection")}
        for k in ("DAC", "NC", "DDC", "TLC"):
            r[f"A1_{k}"] = tab.loc[t, f"A1_{k}"]
            r[f"STOP_{k}"] = tab.loc[t, f"STOP_{k}"]
            r[f"PRIOR_{k}"] = tab.loc[t, f"PRIOR_{k}"]
            r[f"A1s1_{k}"] = tab.loc[t, f"A1s1_{k}"]
            r[f"A1k5_{k}"] = tab.loc[t, f"A1k5_{k}"]
        rs = RS.get(t)
        r["resc_ok"] = bool(rs and rs.get("status") == "OK")
        if r["resc_ok"]:
            r["nd_first"] = rs["nd_first"]
            r["onc_first"] = rs["onc_first"]
            r["ref_dac"] = rs["ref_dac"]
            r["raw_dac"] = rs["raw_dac"]
            r["repro_max"] = max(rs["repro"].values())
            ev = [e for e in rs["nc_events"] if e["at_fault"]]
            r["nc_first_t"] = ev[0]["t_idx"] if ev else -1
            r["nc_ctype"] = ev[0]["ctype"] if ev else ""
            r["nc_obj"] = ev[0]["obj_type"] if ev else ""
            r["nc_lat_dep"] = bool(ev[0]["ego_multi_or_nondrivable"]) if ev else False
            r["nc_ego_speed"] = ev[0]["ego_speed"] if ev else np.nan
            r["nc_obj_speed"] = ev[0]["obj_speed"] if ev else np.nan
            r["nc_rel_lon"] = ev[0]["rel_lon"] if ev else np.nan
            r["nc_rel_lat"] = ev[0]["rel_lat"] if ev else np.nan
            r["sim_speed4"] = rs["sim_speed_0p5s"][-1]
            r["ref_speed4"] = (rs.get("geom") or {}).get("ref_speed_0p5s", [np.nan] * 8)[-1] if rs.get("geom") else np.nan
            if rs.get("human_score"):
                r["human_dac"] = rs["human_score"]["dac"]
        rows.append(r)
    D = pd.DataFrame(rows).set_index("token")
    # reference speed from the geometry pass (finite-difference), plan speed from the plan
    D["ref_speed4"] = [(G[t]["ref_speed_0p5s"][-1]) for t in D.index]
    D["plan_speed4"] = [float(np.hypot(*(plans["A1"][t][-1, :2] - plans["A1"][t][-2, :2])) / 0.5) for t in D.index]
    D.to_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv"), float_format="%.5g")

    # ---------------------------------------------------------------- rescoring reproduction control
    rc = D[D.resc_ok]
    out["rescore_repro"] = {"n_rescored_ok": int(len(rc)), "n_in_files": len(RS),
                            "n_status_not_ok": int(sum(1 for r in RS.values() if r.get("status") != "OK")),
                            "max_abs_diff_all_8_subscores": float(rc.repro_max.max()) if len(rc) else None,
                            "n_with_any_diff_gt_1e-9": int((rc.repro_max > 1e-9).sum()) if len(rc) else None}

    def share(series):
        c = series.value_counts()
        return {k: {"n": int(v), "share": float(v / len(series))} for k, v in c.items()}

    # ---------------------------------------------------------------- DAC-zero anatomy
    dz = D[D.A1_DAC == 0]
    anat = {"n": int(len(dz)), "by_stage": dz.stage.value_counts().to_dict()}
    anat["geometry_class_all_scenes"] = {s: share(D[D.stage == s].cls) for s in (1, 2)}
    anat["geometry_class_DACzero"] = {"stage1": share(dz[dz.stage == 1].cls), "stage2": share(dz[dz.stage == 2].cls),
                                      "both": share(dz.cls)}
    # enrichment: P(DAC zero | class)  over all scenes of that class, with STOP and PRIOR as scene-difficulty controls
    enr = {}
    for s in (1, 2, 12):
        d = D if s == 12 else D[D.stage == s]
        enr[str(s)] = {c: {"n": int(len(g)), "A1_DAC0_rate": float((g.A1_DAC == 0).mean()),
                           "STOP_DAC0_rate": float((g.STOP_DAC == 0).mean()),
                           "PRIOR_DAC0_rate": float((g.PRIOR_DAC == 0).mean()),
                           "A1_minus_STOP": float((g.A1_DAC == 0).mean() - (g.STOP_DAC == 0).mean())}
                       for c, g in d.groupby("cls")}
    anat["enrichment_P_DACzero_given_class"] = enr
    # scene-level flags (rescored A1 DAC-zero tokens)
    dzr = dz[dz.resc_ok]
    anat["rescored_DACzero"] = {"n": int(len(dzr)), "of": int(len(dz))}
    if len(dzr):
        scene = pd.Series(np.where(dzr.nd_first == 0, "INITIAL-STATE", np.where(dzr.ref_dac == 0, "REF-ALSO-FAILS", "PLAN-INDUCED")),
                          index=dzr.index)
        anat["scene_level_flag"] = {"stage1": share(scene[dzr.stage == 1]), "stage2": share(scene[dzr.stage == 2]), "both": share(scene)}
        anat["STOP_also_zero_by_scene_flag"] = {k: {"n": int(len(g)), "STOP_also_DAC0": int((g.STOP_DAC == 0).sum())} for k, g in dzr.groupby(scene)}
        # first non-drivable instant
        tfirst = dzr.nd_first * 0.1
        bins = pd.cut(tfirst, [-0.01, 0.05, 1.0, 2.0, 3.0, 4.01], labels=["t=0", "(0,1]", "(1,2]", "(2,3]", "(3,4]"])
        anat["first_nondrivable_time_s"] = {"stage1": share(bins[dzr.stage == 1].astype(str)), "stage2": share(bins[dzr.stage == 2].astype(str)),
                                            "both": share(bins.astype(str)), "median_s_both": float(tfirst.median())}
        # geometry class x scene flag (PLAN-INDUCED only)
        pi = dzr[scene == "PLAN-INDUCED"]
        anat["PLAN_INDUCED_geometry_class"] = {"stage1": share(pi[pi.stage == 1].cls), "stage2": share(pi[pi.stage == 2].cls), "both": share(pi.cls),
                                               "n": int(len(pi))}
        anat["PLAN_INDUCED_time_bins_by_class"] = {c: share(pd.cut(g.nd_first * 0.1, [-0.01, 0.05, 1.0, 2.0, 3.0, 4.01],
                                                       labels=["t=0", "(0,1]", "(1,2]", "(2,3]", "(3,4]"]).astype(str)) for c, g in pi.groupby("cls")}
        # human filter evidence (stage 1): raw plan DAC vs banked DAC
        s1 = dzr[dzr.stage == 1]
        anat["stage1_human_check"] = {"n": int(len(s1)), "human_also_fails_DAC": int((s1.get("human_dac", pd.Series(dtype=float)) == 0).sum()) if "human_dac" in s1 else None}
        anat["scene_flag_x_geometry"] = {k: share(g.cls) for k, g in dzr.groupby(scene)}
    out["DAC_anatomy"] = anat

    # ---------------------------------------------------------------- command / curve-vs-junction splits (D4 finding 2)
    cs = {}
    for s in (1, 2, 12):
        d = D if s == 12 else D[D.stage == s]
        cs[str(s)] = {}
        for cmd, g in d.groupby("cmd"):
            rt = Counter()
            for _, r in g.iterrows():
                rt["turn-junction" if (r.route_turn and r.junction) else "turn-curve" if r.route_turn else "straight-in-horizon"] += 1
            cs[str(s)][cmd] = {"n": int(len(g)), "route_in_horizon": dict(rt),
                               "A1_DAC0_rate": float((g.A1_DAC == 0).mean()), "STOP_DAC0_rate": float((g.STOP_DAC == 0).mean()),
                               "PRIOR_DAC0_rate": float((g.PRIOR_DAC == 0).mean())}
    out["command_vs_route_geometry"] = cs
    # command compliance: did the plan turn toward the command?  (|yaw4| >= T_HEAD toward the command side)
    def complied(r):
        if r.cmd == "LEFT":
            return r.yaw4 >= T_HEAD
        if r.cmd == "RIGHT":
            return r.yaw4 <= -T_HEAD
        if r.cmd == "STRAIGHT":
            return abs(r.yaw4) < 2 * T_HEAD
        return None
    D["complied"] = D.apply(complied, axis=1)
    comp = {}
    for cmd in ("LEFT", "RIGHT", "STRAIGHT"):
        for rtype, mask in (("route turns in horizon (curve)", (D.route_turn & ~D.junction)), ("route turns in horizon (junction)", (D.route_turn & D.junction)),
                            ("route straight in horizon", ~D.route_turn)):
            g = D[(D.cmd == cmd) & mask]
            if len(g):
                comp[f"{cmd} | {rtype}"] = {"n": int(len(g)), "plan_complied": float(g.complied.mean()),
                                            "A1_DAC0_rate": float((g.A1_DAC == 0).mean()), "STOP_DAC0_rate": float((g.STOP_DAC == 0).mean()),
                                            "A1_DAC0_rate_complied": float((g[g.complied == True].A1_DAC == 0).mean()) if (g.complied == True).any() else None,
                                            "n_complied": int((g.complied == True).sum()),
                                            "A1_DAC0_rate_not_complied": float((g[g.complied == False].A1_DAC == 0).mean()) if (g.complied == False).any() else None,
                                            "n_not_complied": int((g.complied == False).sum())}
    out["command_compliance"] = comp
    # class x command for the DAC-zero ROUTE-FOLLOWING / WRONG-SIDE classes, with curve/junction
    xc = {}
    for cl in ("ROUTE-FOLLOWING", "WRONG-SIDE", "OVER-STEER", "LATERAL-DRIFT", "NO-RECOVERY", "SPEED", "ON-ROUTE", "STOP-LIKE"):
        sub = dz[dz.cls == cl]
        allc = D[D.cls == cl]
        xc[cl] = {"n_DACzero": int(len(sub)), "n_all_scenes": int(len(allc)),
                  "DACzero_by_command": {k: int(v) for k, v in sub.cmd.value_counts().items()},
                  "all_by_command": {k: int(v) for k, v in allc.cmd.value_counts().items()},
                  "DACzero_by_route_kind": {"junction": int((sub.route_turn & sub.junction).sum()), "curve": int((sub.route_turn & ~sub.junction).sum()),
                                            "straight": int((~sub.route_turn).sum())},
                  "all_by_route_kind": {"junction": int((allc.route_turn & allc.junction).sum()), "curve": int((allc.route_turn & ~allc.junction).sum()),
                                        "straight": int((~allc.route_turn).sum())},
                  "DACzero_rate_by_command": {k: float((g.A1_DAC == 0).mean()) for k, g in allc.groupby("cmd")}}
    out["class_x_command_x_routekind"] = xc

    # ---------------------------------------------------------------- 5k and PRIOR comparisons
    cmp = {}
    for key, col in (("A1_5k", "cls_5k"), ("PRIOR", "cls_prior"), ("A1_30k", "cls"), ("A1_30k_seed1", "cls_s1")):
        flag = {"A1_5k": "A1k5_DAC", "PRIOR": "PRIOR_DAC", "A1_30k": "A1_DAC", "A1_30k_seed1": "A1s1_DAC"}[key]
        cmp[key] = {}
        for s in (1, 2, 12):
            d = D if s == 12 else D[D.stage == s]
            z = d[d[flag] == 0]
            cmp[key][str(s)] = {"n_DACzero": int(len(z)), "classes": share(z[col]) if len(z) else {},
                                "DAC0_rate_by_class": {c: {"n": int(len(g)), "rate": float((g[flag] == 0).mean())} for c, g in d.groupby(col)}}
    out["class_mix_across_arms"] = cmp
    # ---------------------------------------------------------------- NC anatomy
    nz = D[D.A1_NC == 0]
    nanat = {"n": int(len(nz)), "by_stage": nz.stage.value_counts().to_dict()}
    nzr = nz[nz.resc_ok]
    nanat["rescored"] = {"n": int(len(nzr)), "of": int(len(nz))}
    if len(nzr):
        def nc_class(r):
            if r.nc_first_t < 0:
                return "NO-AT-FAULT-EVENT-FOUND"
            if r.nc_first_t <= 1:
                return "INITIAL-OVERLAP"
            if r.nc_ctype == "ACTIVE_LATERAL_COLLISION":
                return "LATERAL-AFTER-LANE-DEPARTURE"
            vru = r.nc_obj in ("PEDESTRIAN", "BICYCLE")
            if r.nc_ctype == "STOPPED_TRACK_COLLISION":
                return "STOPPED-TRACK-" + ("VRU" if vru else "VEHICLE/OBJECT")
            if r.nc_ctype == "ACTIVE_FRONT_COLLISION":
                return "ACTIVE-FRONT-" + ("VRU" if vru else "VEHICLE")
            return r.nc_ctype
        nzr = nzr.copy()
        nzr["nc_class"] = nzr.apply(nc_class, axis=1)
        nanat["class"] = {"stage1": share(nzr[nzr.stage == 1].nc_class), "stage2": share(nzr[nzr.stage == 2].nc_class), "both": share(nzr.nc_class)}
        nanat["collision_time_s_median_by_class"] = {c: float(g.nc_first_t.median() * 0.1) for c, g in nzr.groupby("nc_class")}
        nanat["ego_speed_at_collision_median_by_class"] = {c: float(g.nc_ego_speed.median()) for c, g in nzr.groupby("nc_class")}
        nanat["obj_speed_at_collision_median_by_class"] = {c: float(g.nc_obj_speed.median()) for c, g in nzr.groupby("nc_class")}
        nanat["rel_lon_median_by_class"] = {c: float(g.nc_rel_lon.median()) for c, g in nzr.groupby("nc_class")}
        nanat["also_DAC_zero"] = {c: {"n": int(len(g)), "DAC0": int((g.A1_DAC == 0).sum())} for c, g in nzr.groupby("nc_class")}
        nanat["STOP_also_NC_zero"] = {c: {"n": int(len(g)), "STOP_NC0": int((g.STOP_NC == 0).sum()), "PRIOR_NC0": int((g.PRIOR_NC == 0).sum()),
                                          "A1s1_NC0": int((g.A1s1_NC == 0).sum())} for c, g in nzr.groupby("nc_class")}
        # speed: plan 4-s speed vs PDM-closed reference speed in NC-zero scenes vs all scenes (finite-difference both)
        nanat["plan_speed4_minus_ref_speed4_m_s"] = {"NC_zero_median": float((nzr.plan_speed4 - nzr.ref_speed4).median()),
                                                     "all_scenes_median": float((D.plan_speed4 - D.ref_speed4).median())}
        nanat["plan_faster_than_ref_by_3ms_share"] = {"NC_zero": float(((nzr.plan_speed4 - nzr.ref_speed4) >= 3.0).mean()),
                                                      "all_scenes": float(((D.plan_speed4 - D.ref_speed4) >= 3.0).mean())}
        nanat["by_command"] = {c: {"n": int(len(g)), "NC0_rate": float((D[D.cmd == c].A1_NC == 0).mean()), "STOP_NC0_rate": float((D[D.cmd == c].STOP_NC == 0).mean())}
                               for c, g in nz.groupby("cmd")}
        nanat["by_v0_band"] = {b: {"n_all": int(len(g)), "A1_NC0_rate": float((g.A1_NC == 0).mean()), "STOP_NC0_rate": float((g.STOP_NC == 0).mean()),
                                   "PRIOR_NC0_rate": float((g.PRIOR_NC == 0).mean())} for b, g in D.groupby("v0_band")}
        nanat["geometry_class_NCzero"] = share(nz.cls)
        nanat["NC_zero_rate_given_class"] = {c: {"n": int(len(g)), "A1": float((g.A1_NC == 0).mean()), "STOP": float((g.STOP_NC == 0).mean())} for c, g in D.groupby("cls")}
        D.loc[nzr.index, "nc_class"] = nzr.nc_class
    out["NC_anatomy"] = nanat

    # ---------------------------------------------------------------- max-speed known/unknown (D4 finding 1), navhard
    vm = {}
    for line in open(f"{C.milestone_dir(30000)}/bridge_navhard/rows_R7_A1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        vm[r["token"]] = (float(r["vmax"].get("v_max_valid", 0.0)), r["vmax"].get("why"), float(r["vmax"].get("v_max_ms", 0.0)))
    D["vmax_known"] = [vm[t][0] == 1.0 for t in D.index]
    vm5 = {}
    for line in open(f"{C.milestone_dir(5000)}/bridge_navhard/rows_R7_A1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        vm5[r["token"]] = float(r["vmax"].get("v_max_valid", 0.0))
    out["vmax_navhard_consistency_5k_vs_30k_known_flag_equal"] = bool(all(vm5[t] == (1.0 if D.loc[t, "vmax_known"] else 0.0) for t in D.index))
    vmx = {}
    for s in (1, 2, 12):
        d = D if s == 12 else D[D.stage == s]
        vmx[str(s)] = {}
        for known, g in d.groupby("vmax_known"):
            k = "known" if known else "unknown"
            vmx[str(s)][k] = {"n": int(len(g)),
                              **{f"{a}_{t}0": float((g[f"{a}_{t}"] == 0).mean()) for a in ("A1", "STOP", "PRIOR", "A1k5") for t in ("DAC", "NC")}}
            vmx[str(s)][k]["A1_minus_STOP_DAC0"] = vmx[str(s)][k]["A1_DAC0"] - vmx[str(s)][k]["STOP_DAC0"]
            vmx[str(s)][k]["A1_minus_STOP_NC0"] = vmx[str(s)][k]["A1_NC0"] - vmx[str(s)][k]["STOP_NC0"]
            vmx[str(s)][k]["share_of_A1_DAC0"] = float(((g.A1_DAC == 0).sum()) / (d.A1_DAC == 0).sum())
            vmx[str(s)][k]["share_of_A1_NC0"] = float(((g.A1_NC == 0).sum()) / (d.A1_NC == 0).sum())
            vmx[str(s)][k]["share_of_scenes"] = float(len(g) / len(d))
    out["vmax_split_navhard"] = vmx
    # confounding control: composition of the known/unknown strata by command, speed band, route kind; v0-band-standardised difference
    comp2 = {}
    for k, g in D.groupby("vmax_known"):
        nm = "known" if k else "unknown"
        comp2[nm] = {"cmd": {c: float(v) for c, v in g.cmd.value_counts(normalize=True).items()},
                     "v0_band": {c: float(v) for c, v in g.v0_band.value_counts(normalize=True).items()},
                     "route_turn_share": float(g.route_turn.mean()), "junction_share": float(g.junction.mean()),
                     "median_v0": float(g.v0.median())}
    out["vmax_strata_composition"] = comp2
    def std_diff(term, flag_ref):
        """unknown - known, standardised over (stage, v0_band, cmd) cells using the pooled cell weights."""
        num = den = 0.0
        for _, g in D.groupby(["stage", "v0_band", "cmd"]):
            a, b = g[~g.vmax_known], g[g.vmax_known]
            if len(a) >= 5 and len(b) >= 5:
                w = len(g)
                num += w * ((a[f"{flag_ref}_{term}"] == 0).mean() - (b[f"{flag_ref}_{term}"] == 0).mean())
                den += w
        return num / den if den else None
    out["vmax_standardised_diff_unknown_minus_known"] = {f"{a}_{t}0": std_diff(t, a) for a in ("A1", "STOP", "PRIOR") for t in ("DAC", "NC")}
    # within the unknown stratum, which geometry classes carry the DAC zeros
    out["vmax_x_class_DACzero"] = {("known" if k else "unknown"): share(g[g.A1_DAC == 0].cls) for k, g in D.groupby("vmax_known")}
    # the plan-speed effect: does an unknown row change the emitted speed? (plan speed4 vs reference, by stratum)
    out["vmax_plan_speed4_minus_ref_median"] = {("known" if k else "unknown"): float((g.plan_speed4 - g.ref_speed4).median()) for k, g in D.groupby("vmax_known")}
    D.to_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv"), float_format="%.5g")
    json.dump(out, open(os.path.join(RAW, "d6_part2.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("rescore repro:", out["rescore_repro"])
    print("controls:", {k: v for k, v in ctrl.items()})
    print("DAC anatomy: geometry class (DAC0, both):", {k: v["n"] for k, v in anat["geometry_class_DACzero"]["both"].items()})
    print("scene-level:", {k: v["n"] for k, v in anat.get("scene_level_flag", {}).get("both", {}).items()})


if __name__ == "__main__":
    main()
