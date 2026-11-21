"""read_agerule_decomp.py -- card PREREG_V8F_PROOF_ADD71 (AGERULE10K), part B: the cost of rollout age of the pair AGE-K-NAVLOGGED (aged run R0KNlP30Hb20 minus fresh run R0KNlHb20, model seeds 0 and 1)
split into a POSE part (where the window starts) and a STATE part (the motion state the model reads after its own driving). numpy; the engines of the earlier cards are reused through
read_longroll10k.py (its engine(), loaders, restriction and cell helpers), unchanged.

Three tracks per window, all [N, 30, 2] metres (x, y) in the frame of the LOGGED pose at t0 (the wrapper maps the executed knots t0+1 .. t0+30 into that frame, 5 Hz cache steps of 0.2 s):
  f = fresh run track, a = aged run track, d = f moved rigidly by the offset of the aged run at t0 (rotate by dpsi about the logged origin, then add (dx, dy)); speeds and path lengths of d are those of f.
POSE = pair (d, f), STATE = pair (a, d), TOTAL = pair (a, f), all read by the family engine (Q3.pair_cells, same-sign rule on, spread of run R0Hb20 on the same windows, B = --n-boot).
STATE minus POSE per named cell: the paired two-level bootstrap of the per-window value a - 2d + f, read as x = a + f against y = 2d through RH.seed_arrays and RH.read_cell (the interval of the mean
of x - y); a window enters when a, d and f are all finite there.
Names (families LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC is not decomposed), per named cell: a part is "above 0 with its interval excluding 0" when D > 0 and the lower end of its interval is above 0.
  STATE-LARGER: on every named cell of the family STATE is above 0 with its interval excluding 0 and the interval of (STATE minus POSE) excludes 0 on the positive side.
  POSE-LARGER:  the mirror case (POSE above 0 with its interval excluding 0 and the interval of (STATE minus POSE) below 0).
  BOTH:         not one of the two above, and on every named cell the interval of the POSE part and the interval of the STATE part both exclude 0 (either side).
  NOT-SEPARATED: everything else.
Controls b1 to b6 (card section 4, part B). If one fails the controls are printed and written and no name and no part value is written.
Tier: T1-kin/hb0.2 (tier line of read_longroll10k plus the heading bound). No training, no rollout. Never run on data by the tool builder.
Usage: python read_agerule_decomp.py --pull PULL --code CODEDIR --grid-dir GDIR --add70-json READ_LONGROLL10K.json --out OUT.json [--n-boot 2000]
       python read_agerule_decomp.py --selftest EMPTYFOLDER      (synthetic arrays only; last line ZZ SELFTEST-DECOMP ok: True|False (n of m))"""
import argparse
import io
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_longroll10k as R
import read_longroll10k_n4 as N4                                                              # note N1 of the card: the engines take the own start point of a track (note N4 of card ADD70)                                                                  # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD71", "AGERULE10K"
SEEDS = R.SEEDS
FRESH, AGED, SPREAD = "R0KNlHb20", "R0KNlP30Hb20", "R0Hb20"
PAIR = "AGE-K-NAVLOGGED"
TIER_LINE = R.TIER_LINE + "; heading bound 0.2 1/m"
FAMS3 = ("LONGITUDINAL", "LATERAL", "TACTICAL")
NAMED = {f: R.NAMED[f] for f in FAMS3}
STRAT_TEXT = "not decomposed (card section 3)"
DT_STEP = 0.2                                                                                 # cache step in seconds (hierrep_read.DT)
TOL_SUM = 1e-9                                                                                # control b4
B5_SLACK_M, B5_SHARE = 0.5, 0.99                                                              # control b5
B5_CROSS_TOL = 1e-3                                                                           # offsets file cross-check: metres and degrees
TICK_FILE = "ticks_%s_m10_causal_heads.npz"


# ============================================================================================ geometry
def wrap(a):
    return (np.asarray(a, np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def displace(cl, dx, dy, dpsi):
    """rigid move of tracks cl [N, T, 2]: rotate by dpsi [N] (radians) about the origin, then add (dx, dy) [N]. Speeds and lengths are unchanged; headings turn by dpsi."""
    cl = np.asarray(cl, np.float64)
    dx, dy, dpsi = (np.asarray(v, np.float64).reshape(-1, 1) for v in (dx, dy, dpsi))
    c, s = np.cos(dpsi), np.sin(dpsi)
    x, y = cl[..., 0], cl[..., 1]
    return np.stack([c * x - s * y + dx, s * x + c * y + dy], axis=-1)


def displace_reg(cl, dx, dy, dpsi):
    """displace(), and the displaced track is registered with its own start point (dx, dy): the engines then take its step-derived cells from the track moved to that start (note N1)"""
    d = displace(cl, dx, dy, dpsi)
    p0 = np.stack([np.broadcast_to(np.asarray(dx, np.float64).reshape(-1), (len(d),)), np.broadcast_to(np.asarray(dy, np.float64).reshape(-1), (len(d),))], axis=1).copy()
    N4.REG[id(d)] = (p0, N4.sig(d))
    return d


def offsets_from_ticks(T, keys):
    """offset of the own pose from the logged pose at t0, in the logged frame, at the first tick (tick 0) of each window, in the order of `keys`.
    T: tick-file arrays ci, t0, tick, pose [rows, >= 4] (world x, y, yaw, speed of the executed state at the tick), pose_w0 [rows, >= 3] (logged world pose at t0).
    Returns (dict dx, dy, dpsi, v_own, dpos, report); report = restriction report of the tick-0 rows."""
    tick = np.asarray(T["tick"], np.int64)
    m0 = np.nonzero(tick == 0)[0]
    idx, rep = R.restrict_idx(np.asarray(T["ci"], np.int64)[m0], np.asarray(T["t0"], np.int64)[m0], keys)
    if rep["missing"]:
        return None, rep
    rows = m0[idx]
    own = np.asarray(T["pose"], np.float64)[rows]
    lg = np.asarray(T["pose_w0"], np.float64)[rows]
    d = own[:, :2] - lg[:, :2]
    c, s = np.cos(-lg[:, 2]), np.sin(-lg[:, 2])
    dx, dy = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    return {"dx": dx, "dy": dy, "dpsi": wrap(own[:, 2] - lg[:, 2]), "v_own": own[:, 3], "dpos": np.hypot(dx, dy)}, rep


# ============================================================================================ cells and the name rule
def pos_excl(c):
    """a part above 0 with its interval excluding 0"""
    return bool(c["D"] > 0 and c["interval"][0] > 0)


def excl(c):
    return bool(c["interval"][0] > 0 or c["interval"][1] < 0)


def name_family(fam, parts):
    """parts[cell] = {"POSE": {D, interval}, "STATE": {...}, "SMP": {...}} for the named cells of the family -> the name of card section 3"""
    cells = [parts[c] for c in NAMED[fam]]
    if all(pos_excl(c["STATE"]) and c["SMP"]["interval"][0] > 0 for c in cells):
        k = "STATE-LARGER"
    elif all(pos_excl(c["POSE"]) and c["SMP"]["interval"][1] < 0 for c in cells):
        k = "POSE-LARGER"
    elif any(excl(c["POSE"]) and excl(c["STATE"]) for c in cells):                             # card section 3: "on a named cell" (note N2, item 4)
        k = "BOTH"
    else:
        k = "NOT-SEPARATED"
    return "%s-DECOMP-%s-%s" % (PRE, fam, k)


def all_names(parts):
    names = {f: name_family(f, parts) for f in FAMS3}
    names["STRATEGIC"] = STRAT_TEXT
    return names


def assemble(controls, parts, extra):
    """the result record. When a control fails no name, no part value and no share is written."""
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": ok, "names": None, "parts": None, "share_pose_of_total": None}
    res.update(extra)
    if ok:
        res["parts"] = parts
        res["names"] = all_names({c: parts[c] for f in FAMS3 for c in NAMED[f]})
        sh = {}
        for f in FAMS3:
            for c in NAMED[f]:
                t = parts[c]["TOTAL"]["D"]
                sh[c] = (parts[c]["POSE"]["D"] / t) if abs(t) > 1e-12 else None
        res["share_pose_of_total"] = sh
    return res


def cell_rec(c):
    return {"D": float(c["D"]), "interval": [float(c["A_lo"]), float(c["A_hi"])], "per_seed": [float(x) for x in c["d_s"]], "verdict": c.get("verdict")}


# ============================================================================================ controls
def ctl_b1(comps, disp=displace, n=40, seed=1):
    """b1: with a zero offset d equals f on every cell exactly (cells from the family engine's `comps`), and the track itself is identical"""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal([1.0, 0.1], 0.3, (n, 30, 2)), axis=1)
    g = np.cumsum(rng.normal([1.0, 0.0], 0.3, (n, 30, 2)), axis=1)
    z = np.zeros(n)
    d = disp(f, z, z, z)
    cf, cd = comps(f, g), comps(d, g)
    same_cells = all(np.array_equal(cf[k], cd[k], equal_nan=True) for k in cf)
    return {"PASS": bool(same_cells and np.array_equal(f, d)), "cells_compared": sorted(cf), "track_identical": bool(np.array_equal(f, d))}


def straight(n=20, v=6.0, k=30):
    t = DT_STEP * np.arange(1, k + 1)
    return np.repeat(np.stack([v * t, np.zeros(k)], axis=-1)[None], n, axis=0)


def ctl_b2(comps, disp=displace):
    """b2: a planted offset of 1 m ahead on a straight synthetic track changes the along-track error by 1 m and no lateral cell (cross, heading, curvature, yaw rate)"""
    g = straight()
    z = np.zeros(len(g))
    c0, c1 = comps(g, g), comps(disp(g, z + 1.0, z, z), g)
    da = float(np.nanmean(c1["along_mae_m"]) - np.nanmean(c0["along_mae_m"]))
    lat = {k: float(np.nanmax(np.abs(c1[k] - c0[k]))) for k in ("cross_mae_m", "heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps")}
    ok = abs(da - 1.0) <= 1e-9 and all(v <= 1e-9 for v in lat.values())
    return {"PASS": bool(ok), "along_change_m": da, "lateral_cells_max_abs_change": lat}


def ctl_b3(comps, disp=displace):
    """b3: a planted heading offset of 90 deg turns an along-track run into a cross-track one of the same length (x near 0, y equal to the old x; path length unchanged; heading error 90 deg)"""
    g = straight()
    z = np.zeros(len(g))
    d = disp(g, z, z, z + math.pi / 2.0)
    len0 = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(g), 1, 2)), g], 1), axis=1), axis=-1).sum(1)
    len1 = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(d), 1, 2)), d], 1), axis=1), axis=-1).sum(1)
    xmax, ydev, ldev = float(np.abs(d[..., 0]).max()), float(np.abs(d[..., 1] - g[..., 0]).max()), float(np.abs(len1 - len0).max())
    hd = float(np.nanmean(comps(d, g)["heading_mae_deg"]))
    ok = xmax <= 1e-9 and ydev <= 1e-9 and ldev <= 1e-9 and abs(hd - 90.0) <= 1e-9
    return {"PASS": bool(ok), "max_abs_x": xmax, "max_abs_y_minus_old_x": ydev, "max_length_difference_m": ldev, "heading_error_deg": hd}


def ctl_b4(parts, banked_cells):
    """b4: POSE + STATE equals the banked AGE-K-NAVLOGGED cell, and TOTAL equals it, on the differences of the means to 1e-9"""
    worst_sum, worst_tot, missing = 0.0, 0.0, []
    for cell, p in parts.items():
        b = (banked_cells or {}).get(cell)
        if b is None or "D" not in b:
            missing.append(cell)
            continue
        worst_sum = max(worst_sum, abs(p["POSE"]["D"] + p["STATE"]["D"] - b["D"]))
        worst_tot = max(worst_tot, abs(p["TOTAL"]["D"] - b["D"]))
    return {"PASS": bool(not missing and parts and worst_sum <= TOL_SUM and worst_tot <= TOL_SUM), "largest_abs_dev_sum_of_parts": worst_sum, "largest_abs_dev_total": worst_tot,
            "cells_missing_in_bank": missing, "tolerance": TOL_SUM, "n_cells": len(parts)}


def ctl_b5(a_tracks, off, cross=None):
    """b5: the first executed point of the aged track lies within 0.5 m plus one step of travel of the offset position for at least 99 percent of the windows, per model seed.
    a_tracks / off: dicts seed -> [N, 30, 2] / offsets dict (dx, dy, v_own). One step of travel = own speed at the tick times 0.2 s. cross: dict seed -> (dpos, dpsi_deg_abs) read from the offsets file
    of the run, matched to the windows; checked against the tick-file offset to 1e-3 (metres, degrees)."""
    rec, ok = {}, True
    for s, a in a_tracks.items():
        o = off[s]
        dist = np.hypot(a[:, 0, 0] - o["dx"], a[:, 0, 1] - o["dy"])
        bound = B5_SLACK_M + np.abs(o["v_own"]) * DT_STEP
        share = float(np.mean(dist <= bound))
        rec["seed%d" % s] = {"share_within": share, "max_dist_m": float(np.nanmax(dist)), "n": int(len(dist))}
        ok &= bool(share >= B5_SHARE)
        if cross is not None:
            cr = cross.get(s)
            if cr is None:
                rec["seed%d" % s]["offsets_file"] = "absent: cross-check not made"
            else:
                dd = float(np.nanmax(np.abs(cr[0] - o["dpos"]))) if len(cr[0]) else float("inf")
                dh = float(np.nanmax(np.abs(cr[1] - np.abs(np.degrees(o["dpsi"]))))) if len(cr[1]) else float("inf")
                rec["seed%d" % s]["offsets_file"] = {"max_abs_dpos_diff_m": dd, "max_abs_dpsi_diff_deg": dh}
                ok &= bool(dd <= B5_CROSS_TOL and dh <= B5_CROSS_TOL)
    return {"PASS": bool(ok and rec), "required_share": B5_SHARE, "slack_m": B5_SLACK_M, "per_seed": rec}


def ctl_b6(dumps, keys, tick_reports):
    """b6: window identity of the three tracks. dumps: {(run, seed): restricted dump dict}; every dump holds the windows of `keys` in the order of `keys`, and g, ws, eid, clip_index, v0 are identical
    across the fresh, aged and spread runs and the two seeds; the tick-0 rows of the aged files hold exactly the windows of the grid (tick_reports: {seed: restriction report})."""
    ref = next(iter(dumps.values()))
    order_ok = all(np.array_equal(np.asarray(D["clip_index"], np.int64), keys[:, 0]) and np.array_equal(np.asarray(D["ws"], np.int64), keys[:, 1]) for D in dumps.values())
    same = all(np.array_equal(D[k], ref[k]) for D in dumps.values() for k in ("g", "ws", "eid", "clip_index", "v0"))
    ticks_ok = bool(tick_reports) and all(r.get("holds_exactly") is True for r in tick_reports.values())
    return {"PASS": bool(order_ok and same and ticks_ok), "n_dumps": len(dumps), "n_windows": int(len(keys)), "order_of_the_grid": bool(order_ok), "g_ws_eid_clip_v0_identical": bool(same),
            "aged_tick0_rows_exactly_the_grid": bool(ticks_ok)}


# ============================================================================================ the read
def make_cells(RH, cl, g, glon, glat, joint_g):
    c = RH.comps(cl, g)
    c["_lon"], c["_lat"] = RH.classify(*RH.kinematics(cl))
    c["lon_err_rate"] = (c["_lon"] != glon).astype(float)
    c["lat_err_rate"] = (c["_lat"] != glat).astype(float)
    c["joint_err_rate"] = ((c["_lon"] * 3 + c["_lat"]) != joint_g).astype(float)
    return c


def common_support(V, seeds, tags=("F", "D", "A")):
    """note N2 of the card: each cell is kept only on the windows where it is finite in all three tracks (per seed), so the three pairs are read on one set of windows and
    POSE + STATE is TOTAL on the differences of the means. The engine pairs windows where the difference is finite; without this the POSE pair has more windows than the other two.
    Class arrays (keys starting with "_") are left as they are. Returns the count report of the cells that have a window without a value."""
    rep = {}
    for s in seeds:
        for k in sorted(V[(tags[0], s)]):
            if k.startswith("_"):
                continue
            arrs = [np.asarray(V[(t, s)][k], np.float64) for t in tags]
            fin = [np.isfinite(x) for x in arrs]
            ok = fin[0] & fin[1] & fin[2]
            if not all(bool(f.all()) for f in fin):
                r = {"finite_" + t: int(f.sum()) for t, f in zip(tags, fin)}
                r.update({"common": int(ok.sum()), "n": int(ok.size)})
                rep["seed%d:%s" % (s, k)] = r
            for t, x in zip(tags, arrs):
                V[(t, s)][k] = np.where(ok, x, np.nan)
    return rep


def read_parts(E, V, ref, n_boot, extra_keys):
    """the three pairs and STATE minus POSE through the family engine. V holds keys ("F", s), ("D", s), ("A", s) and (SPREAD, s) for the two spread seeds."""
    H, Q3, RH = E
    uniq, inv = RH.episode_units(ref["eid"])
    n_ep = len(uniq)
    glon, glat = RH.classify(*RH.kinematics(ref["g"]))
    joint_g = glon * 3 + glat
    crit = joint_g != RH.KEEP_STRAIGHT
    masks = {"critical": crit, "bulk": ~crit, "all": np.ones(len(crit), bool)}
    keys_all = [(k, "all") for _, k in RH.FAM_ROWS] + list(Q3.EXTRA)
    lv = {s: {kk + "|" + m: float(np.nanmean(V[(SPREAD, s)][kk][masks[m]])) for kk, m in keys_all} for s in SEEDS}
    fam_spread = {}
    for kk, m in keys_all:
        lvs = [lv[s][kk + "|" + m] for s in SEEDS]
        d = [abs(x - y) for x, y in itertools.combinations(lvs, 2)]
        fam_spread[kk + "|" + m] = {"max": float(max(d)), "median": float(np.median(d)), "n_pairs": len(d), "levels": lvs}
    pr = {}
    for nm, (A, B) in (("POSE", ("D", "F")), ("STATE", ("A", "D")), ("TOTAL", ("A", "F"))):
        fc, _ = Q3.pair_cells(V, masks, inv, n_ep, A, B, list(SEEDS), n_boot, len(SEEDS), fam_spread, True)
        pr[nm] = fc
    parts = {c: {nm: cell_rec(pr[nm][c]) for nm in pr} for c in pr["TOTAL"]}
    for fam in FAMS3:
        for cell in NAMED[fam]:
            key = cell.split(":", 1)[1]
            xs = [V[("A", s)][key] + V[("F", s)][key] for s in SEEDS]
            ys = [2.0 * V[("D", s)][key] for s in SEEDS]
            sE, cE = RH.seed_arrays(xs, ys, masks["all"], inv, n_ep)
            parts[cell]["SMP"] = cell_rec(RH.read_cell(sE, cE, n_boot, len(SEEDS)))
    return parts, {"n_windows": int(len(ref["g"])), "n_episodes": int(n_ep), "spread_run": SPREAD, "n_boot": int(n_boot)}


def run(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    if g30 is None:
        print("STOP: grid_p30.json missing in %s" % a.grid_dir)
        return 3
    E = R.engine(a)
    H, Q3, RH = E
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    controls, parts, extra = {}, None, {}
    dumps, why, ticks, trep, offs, cross = {}, [], {}, {}, {}, {}
    for run_, ss in ((FRESH, SEEDS), (AGED, SEEDS), (SPREAD, SEEDS)):
        for s in ss:
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is None:
                why.append("dump missing: %s" % R.key_of(run_, s))
                continue
            idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
            if rep["missing"] or rep["duplicated_windows"]:
                why.append("%s: %d windows missing, %d twice" % (R.key_of(run_, s), rep["missing"], rep["duplicated_windows"]))
                continue
            dumps[(run_, s)] = R.cut(raw, idx)
    for s in SEEDS:
        z = R.npz_load(os.path.join(R.tick_dir(a.pull, AGED, s), TICK_FILE % R.key_of(AGED, s)))
        if z is None:
            why.append("tick file missing: %s" % R.key_of(AGED, s))
            continue
        T = {k: np.asarray(z[k]) for k in ("ci", "t0", "tick", "pose", "pose_w0")}
        o, rep = offsets_from_ticks(T, keys)
        trep[s] = rep
        if o is None:
            why.append("tick-0 rows missing windows: %s" % R.key_of(AGED, s))
            continue
        offs[s] = o
        po = R.npz_load(os.path.join(R.tick_dir(a.pull, AGED, s), "preroll_offsets_%s.npz" % R.key_of(AGED, s)))
        if po is not None:
            m = np.asarray(po["kind"]).astype(str) == "model"
            pos = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(po["ci"])[m], np.asarray(po["t0"])[m]))}
            sel = [pos.get((int(c), int(t)), -1) for c, t in keys.tolist()]
            if min(sel) >= 0:
                sel = np.asarray(sel)
                cross[s] = (np.asarray(po["dpos"], np.float64)[m][sel], np.abs(np.asarray(po["dpsi_deg"], np.float64)[m][sel]))
            else:
                cross[s] = (np.zeros(0), np.zeros(0))
    complete = not why and all((r, s) in dumps for r in (FRESH, AGED, SPREAD) for s in SEEDS) and all(s in offs for s in SEEDS)
    controls["b1_zero_offset_identity"] = ctl_b1(RH.comps, disp=displace_reg)
    controls["b2_one_metre_ahead"] = ctl_b2(RH.comps, disp=displace_reg)
    controls["b3_ninety_degrees"] = ctl_b3(RH.comps, disp=displace_reg)
    controls["n3_planted_start_offset"] = dict(N4.PLANT) if N4.PLANT else {"PASS": False, "why": "the planted check of note N4 did not run"}
    if not complete:
        for k in ("b4_sum_of_parts_equals_bank", "b5_first_point_at_offset", "b6_window_identity"):
            controls[k] = {"PASS": False, "why": "; ".join(why) or "inputs incomplete"}
    else:
        ref = dumps[(FRESH, 0)]
        controls["b6_window_identity"] = ctl_b6(dumps, keys, trep)
        glon, glat = RH.classify(*RH.kinematics(ref["g"]))
        joint_g = glon * 3 + glat
        V, off_log = {}, {}
        for s in SEEDS:
            f, ag = dumps[(FRESH, s)]["cl"], dumps[(AGED, s)]["cl"]
            o = offs[s]
            d = displace_reg(f, o["dx"], o["dy"], o["dpsi"])
            for tag, cl in (("F", f), ("D", d), ("A", ag)):
                V[(tag, s)] = make_cells(RH, cl, ref["g"], glon, glat, joint_g)
            V[(SPREAD, s)] = make_cells(RH, dumps[(SPREAD, s)]["cl"], ref["g"], glon, glat, joint_g)
            off_log["seed%d" % s] = {"position_m": R.off_stats(o["dpos"]), "heading_abs_deg": R.off_stats(np.abs(np.degrees(o["dpsi"]))), "own_speed_mps": R.off_stats(o["v_own"])}
        controls["b5_first_point_at_offset"] = ctl_b5({s: dumps[(AGED, s)]["cl"] for s in SEEDS}, offs, cross)
        extra["common_windows_note_n2"] = common_support(V, SEEDS)
        parts, counts = read_parts(E, V, ref, a.n_boot, None)
        extra["offsets_at_t0_from_tick_files"], extra["counts"] = off_log, counts
        bank = R.jl(a.add70_json)
        banked = ((bank or {}).get("cells") or {}).get(PAIR)
        controls["b4_sum_of_parts_equals_bank"] = ctl_b4(parts, banked) if banked else {"PASS": False, "why": "no cells[%s] in %s" % (PAIR, a.add70_json)}
    order = ["b1_zero_offset_identity", "b2_one_metre_ahead", "b3_ninety_degrees", "b4_sum_of_parts_equals_bank", "b5_first_point_at_offset", "b6_window_identity"]
    controls = {k: controls[k] for k in order}
    res = assemble(controls, parts, extra)
    print(TIER_LINE)
    print("== controls b1 to b6")
    for k, v in controls.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x != "PASS"}))[:600]))
    if res["controls_all_pass"]:
        print_table(res)
    else:
        print("== a control did not pass: no name and no part value is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


def print_table(res):
    print("== names")
    for f, n in res["names"].items():
        print("   %-13s %s" % (f, n))
    print("== named cells: D [interval] per part (POSE = d minus f, STATE = a minus d, TOTAL = a minus f, SMP = STATE minus POSE), share POSE/TOTAL")
    for f in FAMS3:
        for c in NAMED[f]:
            p = res["parts"][c]
            sh = res["share_pose_of_total"][c]
            print("   %-32s " % c + "  ".join("%s %+.4f [%+.4f,%+.4f]" % (k, p[k]["D"], p[k]["interval"][0], p[k]["interval"][1]) for k in ("POSE", "STATE", "TOTAL", "SMP")) + "  share %s" % ("n/a" if sh is None else "%.2f" % sh))


# ============================================================================================ selftest (synthetic arrays only)
def mini_comps(X, G):
    """numpy stand-in of the engine's cells for the selftest: along / cross = x / y of the track, heading from the step displacement (the origin is prepended), speed = step length / 0.2 s"""
    out = {}

    def geom(w):
        p = np.concatenate([np.zeros_like(w[:, :1]), w], 1)
        dd = p[:, 1:] - p[:, :-1]
        ds = np.linalg.norm(dd, axis=-1)
        hd = np.arctan2(dd[..., 1], dd[..., 0])
        dh = wrap(hd[:, 1:] - hd[:, :-1])
        return {"speed": ds / DT_STEP, "heading": hd, "along": p[:, 1:, 0], "cross": p[:, 1:, 1], "yaw": dh / DT_STEP, "curv": dh / (0.5 * (ds[:, 1:] + ds[:, :-1]) + 1e-9)}
    P, Q = geom(np.asarray(X, np.float64)), geom(np.asarray(G, np.float64))
    out["speed_mae_mps"] = np.abs(P["speed"] - Q["speed"]).mean(1)
    out["along_mae_m"] = np.abs(P["along"] - Q["along"]).mean(1)
    out["cross_mae_m"] = np.abs(P["cross"] - Q["cross"]).mean(1)
    out["heading_mae_deg"] = np.degrees(np.abs(wrap(P["heading"] - Q["heading"])).mean(1))
    out["yaw_rate_mae_degps"] = np.degrees(np.abs(P["yaw"] - Q["yaw"]).mean(1))
    out["curvature_mae_1pm"] = np.abs(P["curv"] - Q["curv"]).mean(1)
    return out


def cell(D, lo, hi):
    return {"D": D, "interval": [lo, hi], "per_seed": [D, D], "verdict": None}


def selftest(folder):
    res = []

    def chk(name, ok):
        res.append((name, bool(ok)))
        print("   %-110s %s" % (name, bool(ok)))
    print(TIER_LINE)
    if os.path.isdir(folder) and os.listdir(folder):
        print("STOP: the selftest folder is not empty")
        return 2
    os.makedirs(folder, exist_ok=True)
    # ---- rigid displacement
    p = np.array([[[1.0, 0.0], [2.0, 0.5]]])
    q = displace(p, [10.0], [20.0], [math.pi / 2])
    chk("displace: rotation by 90 deg about the origin then translation", np.allclose(q, [[[10.0, 21.0], [9.5, 22.0]]], atol=1e-12))
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.normal(1.0, 0.2, (50, 30, 2)), 1)
    dd = displace(t, rng.normal(0, 5, 50), rng.normal(0, 5, 50), rng.uniform(-3, 3, 50))
    st = lambda w: np.linalg.norm(np.diff(w, axis=1), axis=-1)                               # noqa: E731
    chk("displace: step lengths are unchanged for any offset (speeds of d are those of f)", np.allclose(st(t), st(dd), atol=1e-9))
    chk("displace: zero offset returns the same array bit for bit", np.array_equal(displace(t, np.zeros(50), np.zeros(50), np.zeros(50)), t))
    # ---- offsets from a synthetic tick file (3 ticks per window, shuffled; yaw wrap)
    keys = np.array([[0, 7], [0, 11], [1, 7], [2, 3]], np.int64)
    n = len(keys)
    lw = np.column_stack([rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), np.array([0.3, 3.1, -1.0, 2.0])])
    own_dx, own_dy = np.array([3.0, -2.0, 0.5, 6.0]), np.array([1.0, 4.0, -0.5, 0.0])
    dpsi = np.array([0.1, -0.2, 0.05, 0.0])
    rows = {"ci": [], "t0": [], "tick": [], "pose": [], "pose_w0": []}
    for tk in (2, 0, 1):
        for i in (3, 1, 0, 2):
            c, s = np.cos(lw[i, 2]), np.sin(lw[i, 2])
            wx = lw[i, 0] + c * own_dx[i] - s * own_dy[i]
            wy = lw[i, 1] + s * own_dx[i] + c * own_dy[i]
            yaw = lw[i, 2] + dpsi[i]
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi                                  # the own yaw is stored wrapped
            junk = 99.0 if tk else 0.0                                                       # later ticks must be ignored
            rows["ci"].append(keys[i, 0]); rows["t0"].append(keys[i, 1]); rows["tick"].append(tk)
            rows["pose"].append([wx + junk, wy + junk, yaw, 5.0 + i]); rows["pose_w0"].append(lw[i])
    T = {k: np.asarray(v) for k, v in rows.items()}
    np.savez(os.path.join(folder, "ticks_synth.npz"), **T)
    o, rep = offsets_from_ticks(dict(np.load(os.path.join(folder, "ticks_synth.npz"))), keys)
    chk("offsets_from_ticks: logged-frame offset (dx, dy, dpsi) recovered from tick 0 only, windows in the order of the grid, yaw wrap handled",
        o is not None and np.allclose(o["dx"], own_dx, atol=1e-9) and np.allclose(o["dy"], own_dy, atol=1e-9) and np.allclose(o["dpsi"], dpsi, atol=1e-9) and np.allclose(o["v_own"], [5, 6, 7, 8]) and rep["holds_exactly"])
    keep = ~((T["tick"] == 0) & (T["ci"] == keys[2, 0]) & (T["t0"] == keys[2, 1]))
    o2, rep2 = offsets_from_ticks({k: v[keep] for k, v in T.items()}, keys)
    chk("offsets_from_ticks: a window without a tick-0 row is reported missing", o2 is None and rep2["missing"] == 1)
    # ---- b1, b2, b3 on the stand-in cells, with planted faults
    chk("b1 passes with the rigid displacement", ctl_b1(mini_comps)["PASS"])
    chk("b1 fails when the displacement moves a track by 1e-6 m at zero offset", not ctl_b1(mini_comps, disp=lambda c, x, y, p_: displace(c, x + 1e-6, y, p_))["PASS"])
    chk("b2 passes: 1 m ahead changes the along-track error by 1 m and no lateral cell", ctl_b2(mini_comps)["PASS"])
    chk("b2 fails when the planted shift is 1 m sideways", not ctl_b2(mini_comps, disp=lambda c, x, y, p_: displace(c, 0 * x, x, p_))["PASS"])
    chk("b3 passes: 90 deg turns the along-track run into a cross-track one of the same length", ctl_b3(mini_comps)["PASS"])
    chk("b3 fails when the planted turn is 80 deg", not ctl_b3(mini_comps, disp=lambda c, x, y, p_: displace(c, x, y, p_ * 80 / 90))["PASS"])
    # ---- b4
    parts = {"c1": {"POSE": cell(0.3, 0.1, 0.5), "STATE": cell(0.2, 0.0, 0.4), "TOTAL": cell(0.5, 0.2, 0.8), "SMP": cell(-0.1, -0.3, 0.1)}}
    chk("b4 passes when POSE + STATE and TOTAL equal the banked D", ctl_b4(parts, {"c1": {"D": 0.5}})["PASS"])
    chk("b4 fails on a deviation of 1e-6", not ctl_b4(parts, {"c1": {"D": 0.5 + 1e-6}})["PASS"])
    chk("b4 fails when the sum differs from TOTAL (planted 1e-6 in STATE)", not ctl_b4({"c1": dict(parts["c1"], STATE=cell(0.2 + 1e-6, 0, 0.4))}, {"c1": {"D": 0.5}})["PASS"])
    chk("b4 fails when the banked cell is missing", not ctl_b4(parts, {})["PASS"])
    # ---- note N2: one set of windows for the three pairs

    def pm(x, y):
        return float(np.nanmean(np.asarray(x, np.float64) - np.asarray(y, np.float64)))
    Vt = {(t, 0): {"c": np.array(v), "_lon": np.zeros(4, np.int64)} for t, v in (("F", [1.0, 2.0, np.nan, 4.0]), ("D", [2.0, 5.0, np.nan, 5.0]), ("A", [3.0, np.nan, 7.0, 9.0]))}
    dev = lambda: pm(Vt[("D", 0)]["c"], Vt[("F", 0)]["c"]) + pm(Vt[("A", 0)]["c"], Vt[("D", 0)]["c"]) - pm(Vt[("A", 0)]["c"], Vt[("F", 0)]["c"])   # noqa: E731
    before = dev()
    rep_n2 = common_support(Vt, (0,))
    chk("note N2: with pairwise windows POSE + STATE is away from TOTAL on the planted case", abs(before) > 0.5)
    chk("note N2: on the common windows POSE + STATE is TOTAL and the count report names the cell", abs(dev()) < 1e-12 and rep_n2.get("seed0:c") == {"finite_F": 3, "finite_D": 3, "finite_A": 3, "common": 2, "n": 4})
    chk("note N2: class arrays are left as they are", Vt[("A", 0)]["_lon"].dtype.kind == "i" and "seed0:_lon" not in rep_n2)
    # ---- b5
    N = 200
    off = {0: {"dx": rng.normal(0, 5, N), "dy": rng.normal(0, 5, N), "dpsi": rng.normal(0, 0.2, N), "v_own": np.full(N, 6.0)}}
    off[0]["dpos"] = np.hypot(off[0]["dx"], off[0]["dy"])
    a = np.zeros((N, 30, 2))
    a[:, 0, 0], a[:, 0, 1] = off[0]["dx"] + 1.0, off[0]["dy"]                                # 1.0 m from the offset; bound 0.5 + 6 * 0.2 = 1.7 m
    cr = {0: (off[0]["dpos"].copy(), np.abs(np.degrees(off[0]["dpsi"])))}
    chk("b5 passes: every first point within 0.5 m plus one step of the offset, offsets file agrees", ctl_b5({0: a}, off, cr)["PASS"])
    a2 = a.copy()
    a2[:6, 0, 0] += 5.0                                                                       # 3 percent of the windows far away
    chk("b5 fails when 3 percent of the first points are 5 m from the offset", not ctl_b5({0: a2}, off, cr)["PASS"])
    a3 = a.copy()
    a3[:1, 0, 0] += 5.0                                                                       # 0.5 percent: still passes
    chk("b5 passes with 0.5 percent of the windows outside (limit is 99 percent)", ctl_b5({0: a3}, off, cr)["PASS"])
    cr2 = {0: (off[0]["dpos"] + 0.01, cr[0][1])}
    chk("b5 fails when the offsets file disagrees with the tick file by 1 cm", not ctl_b5({0: a}, off, cr2)["PASS"])
    # ---- b6
    g = rng.normal(0, 1, (4, 30, 2))
    D1 = {"g": g, "ws": keys[:, 1], "eid": np.array(["e0", "e0", "e1", "e2"]), "clip_index": keys[:, 0], "v0": np.arange(4.0)}
    D2 = {k: v.copy() for k, v in D1.items()}
    rp = {0: {"holds_exactly": True}}
    chk("b6 passes on identical windows in the order of the grid", ctl_b6({("a", 0): D1, ("b", 0): D2}, keys, rp)["PASS"])
    D3 = {k: v.copy() for k, v in D1.items()}
    D3["g"][2, 5, 0] += 1e-9
    chk("b6 fails when one logged track differs", not ctl_b6({("a", 0): D1, ("b", 0): D3}, keys, rp)["PASS"])
    D4 = {k: v[::-1].copy() for k, v in D1.items()}
    chk("b6 fails when the windows are in another order", not ctl_b6({("a", 0): D1, ("b", 0): D4}, keys, rp)["PASS"])
    chk("b6 fails when the aged tick-0 rows are not exactly the grid", not ctl_b6({("a", 0): D1, ("b", 0): D2}, keys, {0: {"holds_exactly": False}})["PASS"])
    # ---- the name rule: each outcome, on every family
    big = lambda D: cell(D, D * 0.5, D * 1.5)                                                # noqa: E731
    zero = cell(0.01, -0.1, 0.1)

    def parts_for(kind):
        P = {}
        for c in sum(NAMED.values(), []):
            if kind == "state":
                P[c] = {"POSE": cell(0.05, -0.05, 0.1), "STATE": big(0.5), "SMP": cell(0.45, 0.2, 0.7)}
            elif kind == "pose":
                P[c] = {"POSE": big(0.5), "STATE": cell(0.05, -0.05, 0.1), "SMP": cell(-0.45, -0.7, -0.2)}
            elif kind == "both":
                P[c] = {"POSE": big(0.4), "STATE": big(0.4), "SMP": cell(0.0, -0.1, 0.1)}
            else:
                P[c] = {"POSE": zero, "STATE": zero, "SMP": zero}
            P[c]["TOTAL"] = cell(P[c]["POSE"]["D"] + P[c]["STATE"]["D"], 0.0, 1.0)
        return P
    for kind, k in (("state", "STATE-LARGER"), ("pose", "POSE-LARGER"), ("both", "BOTH"), ("none", "NOT-SEPARATED")):
        nm = all_names(parts_for(kind))
        chk("name rule: %s on LONGITUDINAL, LATERAL and TACTICAL" % k, all(nm[f] == "%s-DECOMP-%s-%s" % (PRE, f, k) for f in FAMS3))
    chk("name rule: STRATEGIC is the not-decomposed string", all_names(parts_for("none"))["STRATEGIC"] == STRAT_TEXT)
    mixed = parts_for("state")
    mixed["LONGITUDINAL:along_mae_m"] = parts_for("none")["LONGITUDINAL:along_mae_m"]
    chk("name rule: one named cell not separated makes the family NOT-SEPARATED", all_names(mixed)["LONGITUDINAL"].endswith("NOT-SEPARATED"))
    one = parts_for("state")
    one["LONGITUDINAL:along_mae_m"] = parts_for("both")["LONGITUDINAL:along_mae_m"]
    chk("name rule (note N2): both parts excluding 0 on one named cell, with no part larger on every cell, makes the family BOTH", all_names(one)["LONGITUDINAL"].endswith("-BOTH"))
    ident = parts_for("state")
    ident["LONGITUDINAL:speed_mae_mps"] = {"POSE": cell(0.0, 0.0, 0.0), "STATE": big(0.5), "SMP": big(0.5), "TOTAL": big(0.5)}
    chk("name rule (note N2): a POSE part that is 0 by identity on one cell leaves STATE-LARGER reachable", all_names(ident)["LONGITUDINAL"].endswith("STATE-LARGER"))
    neg = parts_for("state")
    neg["TACTICAL:ade_tac_m"]["STATE"] = cell(-0.5, -0.8, -0.2)
    chk("name rule: a STATE part below 0 is never STATE-LARGER (it excludes 0 on the wrong side)", not all_names(neg)["TACTICAL"].endswith("STATE-LARGER"))
    sl = parts_for("state")
    sl["LATERAL:heading_mae_deg"]["SMP"] = cell(0.1, -0.1, 0.3)
    chk("name rule: STATE above 0 but STATE minus POSE interval touching 0 is not STATE-LARGER", not all_names(sl)["LATERAL"].endswith("STATE-LARGER"))
    # ---- assemble: a failed control writes no name and no part value
    ok_c = {"b1": {"PASS": True}, "b2": {"PASS": True}}
    full = assemble(ok_c, parts_for("state"), {})
    chk("assemble: all controls pass -> names, parts and shares are written", full["controls_all_pass"] and full["names"] is not None and full["parts"] is not None and full["share_pose_of_total"] is not None)
    bad = assemble({"b1": {"PASS": True}, "b4": {"PASS": False}}, parts_for("state"), {"counts": {"n": 1}})
    chk("assemble: a failed control -> no name, no part value, no share (controls and counts still written)",
        (not bad["controls_all_pass"]) and bad["names"] is None and bad["parts"] is None and bad["share_pose_of_total"] is None and bad["controls"]["b4"]["PASS"] is False and bad["counts"] == {"n": 1})
    chk("assemble: no controls at all is a failure", assemble({}, parts_for("state"), {})["names"] is None)
    wr = os.path.join(folder, "out_failed.json")
    json.dump(R.conv(bad), io.open(wr, "w", encoding="utf-8", newline="\n"))
    back = json.load(io.open(wr, encoding="utf-8"))
    chk("assemble: the written file of a failed read holds no name and no part", back["names"] is None and back["parts"] is None)
    sh = full["share_pose_of_total"]
    chk("assemble: share POSE/TOTAL = POSE D over TOTAL D", abs(sh["LONGITUDINAL:speed_mae_mps"] - 0.05 / 0.55) < 1e-12)
    # ---- the text of this file
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [w for w in words if w in src])
    chk("tier line is the reader's tier line plus the heading bound", TIER_LINE == R.TIER_LINE + "; heading bound 0.2 1/m")
    chk("names are unique over families and outcomes", len({"%s-DECOMP-%s-%s" % (PRE, f, k) for f in FAMS3 for k in ("STATE-LARGER", "POSE-LARGER", "BOTH", "NOT-SEPARATED")}) == 12)
    ok = all(v for _, v in res)
    print("ZZ SELFTEST-DECOMP ok: %s (%d of %d)" % (ok, sum(v for _, v in res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        i = argv.index("--selftest")
        if i + 1 >= len(argv):
            print("usage: --selftest EMPTYFOLDER")
            return 2
        return selftest(argv[i + 1])
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "add70-json", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
