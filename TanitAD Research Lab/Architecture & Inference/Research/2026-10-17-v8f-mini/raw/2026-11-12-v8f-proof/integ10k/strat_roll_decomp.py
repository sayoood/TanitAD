#!/usr/bin/env python
"""strat_roll_decomp.py -- note N1 to card PREREG_V8F_PROOF_ADD66 (STRATROLL10K): a descriptive split of the rollout read into the head's own output
(pos_cam: the other session's T0 checkpoint formula at the head's tick step, in the frame of the logged pose the camera stood at) and the pose the rollout
places it in (the offset of the executed pose from that logged pose). Names nothing. Reads the same tick files and the same join as strat_roll_read.py,
whose functions it imports unchanged (same folder). Pod CPU only; no model.

usage: strat_roll_decomp.py --out DIR [--with-cand]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_roll_read as R  # noqa: E402

PAIRS = [("R0G0X", "R0"), ("R0C10", "R0"), ("R0G0", "R0"), ("R0Bk10", "R0"), ("RLc", "R0"), ("RQ50", "R0")]
CAND_PAIRS = [("RLcKF5", "RLc"), ("RQ50KF5", "RQ50"), ("RLcK", "RLc"), ("RQ50K", "RQ50")]
PSI_BINS = ((0.0, 2.0), (2.0, 5.0), (5.0, 20.0), (20.0, 1e9))
V_BINS = ((0.0, 2.0), (2.0, 8.0), (8.0, 1e9))


def per_run(ctx, run, s):
    """pos (anchored, as the card), pos_cam, lat on label-valid rows, raw s_ckpt, pose offsets, speed -- [3 ticks, N windows, ...]."""
    p = R.tick_path(ctx.W, run, s)
    T = R.load_ticks(p)
    z = np.load(p, allow_pickle=True)
    v0 = np.asarray(z["v0"], np.float64)
    rbt = R.window_order(T)
    names, J = T["clip_names"], ctx.J
    cells, _, anc_ds = R.compute_cells(ctx, T)
    n = R.N_WIN
    pos_cam = np.full((3, n, 3), np.nan)
    off = np.full((3, n, 3), np.nan)                 # along, sideways (m), heading (deg), executed minus logged, in the logged frame
    raw = np.zeros((3, n, 3, 2))
    spd = np.zeros((3, n))
    for k in range(3):
        for i, row in enumerate(rbt[k]):
            ci = int(T["ci"][row])
            G, _ = ctx.grid(str(names[ci]))
            step = int(T["j"][row]) - int(T["age_s"][row])
            ck = T["s_ckpt"][row].astype(np.float64)
            raw[k, i] = ck[:, :2]
            spd[k, i] = v0[row]
            if 0 <= step < int(J["n"][ci]) and step < len(G) and np.isfinite(G[step]).all():
                idx = int(J["off"][ci]) + step
                rc, ok = np.asarray(J["rc"][idx], np.float64), np.asarray(J["rc_ok"][idx], bool)
                pos_cam[k, i] = np.where(ok, np.linalg.norm(ck[:, :2] - rc[:, :2], axis=1), np.nan)
                sp = T["s_pose"][row].astype(np.float64)
                d = sp[:2] - G[step, :2]
                c, sn = math.cos(-G[step, 2]), math.sin(-G[step, 2])
                off[k, i] = (d[0] * c - d[1] * sn, d[0] * sn + d[1] * c, float(R.wrap(sp[2] - G[step, 2])) * 180.0 / math.pi)
    lat_v = np.where(np.isfinite(cells["pos"]), cells["lat"], np.nan)
    return {"pos": cells["pos"], "pos_cam": pos_cam, "lat_v": lat_v, "raw": raw, "off": off, "spd": spd, "t0": T["t0"][rbt[0]], "T": T}


def stat(v):
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()) if len(v) else None, "median": float(np.median(v)) if len(v) else None, "n": int(len(v))}


def roll_both(D, run, eid, t0, tick):
    """ROLL and ROLLCAM of a run at a tick on exactly the rows valid for both; per arc dict."""
    a_pos, a_cam = [], []
    for s in R.SEEDS:
        d = D[(run, s)]
        dp = R.roll_diff(d["pos"][tick], d["pos"][0], t0, eid, tick)
        dc = R.roll_diff(d["pos_cam"][tick], d["pos_cam"][0], t0, eid, tick)
        both = np.isfinite(dp) & np.isfinite(dc)
        a_pos.append(np.where(both, dp, np.nan))
        a_cam.append(np.where(both, dc, np.nan))
    bp, bc = R.boot_cols(a_pos, eid), R.boot_cols(a_cam, eid)
    out = []
    for m, A in enumerate(R.ARCS):
        share = (bc[m]["mean"] / bp[m]["mean"]) if (bp[m]["mean"] not in (None, 0.0) and bc[m]["mean"] is not None) else None
        out.append({"arc_m": A, "ROLL": bp[m], "ROLLCAM": bc[m], "share_in_head_output": share})
    return out, a_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", default=R.DEF_W)
    ap.add_argument("--tracks", default=R.DEF_TRACKS)
    ap.add_argument("--join", default=R.DEF_JOIN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-cand", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, "DECOMP_STRATROLL10K.json")
    if os.path.exists(outp):
        raise SystemExit("STOP: output exists: " + outp)
    ctx = R.Ctx(a)
    ctx.load_static()
    runs = list(R.BASE_RUNS)
    cand = False
    if a.with_cand:
        cand = all(os.path.isfile(R.tick_path(ctx.W, r, s)) for r in R.CAND_RUNS for s in R.SEEDS)
        if cand:
            runs += list(R.CAND_RUNS)
    D = {}
    for run in runs:
        for s in R.SEEDS:
            D[(run, s)] = per_run(ctx, run, s)
            print(f"loaded {run}_s{s}", flush=True)
    T0 = D[("R0", 0)]["T"]
    t0 = D[("R0", 0)]["t0"]
    eid, _ = R.episode_map(ctx.W, T0)
    res = {"note": "N1 to card PREREG_V8F_PROOF_ADD66 (descriptive; names nothing)", "tier": R.TIER, "runs": runs, "cand_runs_read": cand}
    # controls d1, d2
    d1 = d2 = 0.0
    for key, d in D.items():
        x, y = d["pos"][0], d["pos_cam"][0]
        if not (np.isnan(x) == np.isnan(y)).all():
            d1 = float("inf")
        else:
            g = np.isfinite(x)
            d1 = max(d1, float(np.abs(x[g] - y[g]).max()))
        o = d["off"][0]
        d2 = max(d2, float(np.nanmax(np.abs(o))))
    res["controls"] = {"d1_first_tick_pos_cam_equals_pos": {"pass": bool(d1 <= 1e-9), "largest_difference_m": d1},
                       "d2_first_tick_pose_offset_zero": {"pass": bool(d2 <= 1e-6), "largest_abs": d2}}
    ok = all(v["pass"] for v in res["controls"].values())
    res["controls_all_pass"] = ok
    print("d1 first-tick pos_cam equals pos: %s (%.3e m); d2 first-tick pose offset 0: %s (%.3e)" % (res["controls"]["d1_first_tick_pos_cam_equals_pos"]["pass"], d1,
                                                                                                    res["controls"]["d2_first_tick_pose_offset_zero"]["pass"], d2))
    if not ok:
        print("ZZ STRATROLL-DECOMP STOP a control failed; nothing reported")
        with open(outp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=1)
        return 2
    # levels
    lv = {}
    for (run, s), d in D.items():
        e = {}
        for k in range(3):
            for m, A in enumerate(R.ARCS):
                for c in ("pos", "pos_cam", "lat_v"):
                    e[f"tick{k}_{int(A)}m_{c}"] = stat(d[c][k][:, m])
            o = d["off"][k]
            g = np.isfinite(o[:, 2])
            e[f"tick{k}_pose_offset"] = {"mean_abs_along_m": float(np.abs(o[g, 0]).mean()), "mean_abs_sideways_m": float(np.abs(o[g, 1]).mean()),
                                         "mean_abs_heading_deg": float(np.abs(o[g, 2]).mean()), "median_abs_heading_deg": float(np.median(np.abs(o[g, 2]))),
                                         "share_heading_above_5deg": float((np.abs(o[g, 2]) > 5).mean()), "share_heading_above_20deg": float((np.abs(o[g, 2]) > 20).mean()),
                                         "n": int(g.sum())}
        lv[f"{run}_s{s}"] = e
    res["levels"] = lv
    # ROLL / ROLLCAM on the same rows
    roll = {}
    for run in runs:
        roll[run] = {}
        for tick in (1, 2):
            roll[run][f"tick{tick}"], _ = roll_both(D, run, eid, t0, tick)
    res["roll"] = roll
    r2 = roll["R0"]["tick2"]
    worse = sum(x["ROLLCAM"]["verdict"] == "WORSE" for x in r2)
    res["HEAD"] = {"arcs_where_ROLLCAM2_of_R0_reads_WORSE": int(worse),
                   "reading": ("the head's own output degrades on the rollout (ROLLCAM2 is its size)" if worse >= 2 else
                               ("the name DEGRADES is carried by the pose alone: the head's output keeps its T0 quality in the camera frame" if worse == 0 else "mixed (one arc)"))}
    # pairs on pos_cam and raw output distance
    pairs = {}
    for A, B in PAIRS + (CAND_PAIRS if cand else []):
        ent = {}
        for k in (1, 2):
            arrs = [D[(A, s)]["pos_cam"][k] - D[(B, s)]["pos_cam"][k] for s in R.SEEDS]
            arrs_p = [D[(A, s)]["pos"][k] - D[(B, s)]["pos"][k] for s in R.SEEDS]
            rawd = [float(np.linalg.norm(np.concatenate([D[(A, s)]["raw"][k][:, m] - D[(B, s)]["raw"][k][:, m] for s in R.SEEDS]), axis=1).mean()) for m in range(3)]
            ent[f"tick{k}"] = {"pos_cam": R.boot_cols(arrs, eid), "pos": R.boot_cols(arrs_p, eid), "raw_output_distance_m": rawd}
        pairs[f"{A}-{B}"] = ent
    res["pairs"] = pairs
    # ROLL2 of R0 by heading-offset bin; heading offset by speed bin
    bins = []
    _, a_pos = roll_both(D, "R0", eid, t0, 2)
    for lo, hi in PSI_BINS:
        row = {"heading_offset_deg": [lo, hi if hi < 1e8 else None], "per_arc": []}
        for m, A in enumerate(R.ARCS):
            vals = []
            for si, s in enumerate(R.SEEDS):
                h = np.abs(D[("R0", s)]["off"][2][:, 2])
                g = np.isfinite(a_pos[si][:, m]) & (h >= lo) & (h < hi)
                vals.append(a_pos[si][g, m])
            v = np.concatenate(vals)
            row["per_arc"].append({"arc_m": A, "mean_ROLL2_m": float(v.mean()) if len(v) else None, "n": int(len(v))})
        bins.append(row)
    res["ROLL2_of_R0_by_heading_offset"] = bins
    vb = []
    for lo, hi in V_BINS:
        hs = []
        for s in R.SEEDS:
            d = D[("R0", s)]
            h, v = np.abs(d["off"][2][:, 2]), d["spd"][2]
            g = np.isfinite(h) & (v >= lo) & (v < hi)
            hs.append(h[g])
        h = np.concatenate(hs)
        vb.append({"speed_mps": [lo, hi if hi < 1e8 else None], "mean_abs_heading_offset_deg": float(h.mean()) if len(h) else None,
                   "median_abs_heading_offset_deg": float(np.median(h)) if len(h) else None, "n": int(len(h))})
    res["heading_offset_of_R0_third_tick_by_speed"] = vb
    res["elapsed_s"] = round(time.time() - t_all, 1)
    with open(outp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    # print
    print("HEAD: " + res["HEAD"]["reading"] + f" (WORSE at {worse} of 3 arcs)")
    for run in ("R0", "R0G0X", "R0C10"):
        for tick in (1, 2):
            print(f"{run} tick{tick}: " + " ; ".join("%dm ROLL %+.2f ROLLCAM %+.2f [%+.2f,%+.2f] %s share %s" % (
                x["arc_m"], x["ROLL"]["mean"], x["ROLLCAM"]["mean"], x["ROLLCAM"]["lo"], x["ROLLCAM"]["hi"], x["ROLLCAM"]["verdict"],
                ("%.2f" % x["share_in_head_output"]) if x["share_in_head_output"] is not None else "n/a") for x in roll[run][f"tick{tick}"]))
    for pn, e in pairs.items():
        b = e["tick2"]
        print(f"{pn:14s} tick2 pos_cam: " + " ; ".join("%dm %+.2f [%+.2f,%+.2f] %s" % (A, x["mean"], x["lo"], x["hi"], x["verdict"]) for A, x in zip(R.ARCS, b["pos_cam"]))
              + " | raw distance " + "/".join("%.2f" % x for x in b["raw_output_distance_m"]))
    for run in ("R0", "R0G0X", "R0C10"):
        for s in R.SEEDS:
            e = lv[f"{run}_s{s}"]
            print(f"{run}_s{s} " + " | ".join("t%d off along %.2f side %.2f head mean %.1f med %.1f >5deg %.2f >20deg %.2f" % (
                k, e[f"tick{k}_pose_offset"]["mean_abs_along_m"], e[f"tick{k}_pose_offset"]["mean_abs_sideways_m"], e[f"tick{k}_pose_offset"]["mean_abs_heading_deg"],
                e[f"tick{k}_pose_offset"]["median_abs_heading_deg"], e[f"tick{k}_pose_offset"]["share_heading_above_5deg"], e[f"tick{k}_pose_offset"]["share_heading_above_20deg"]) for k in (1, 2)))
            print(f"{run}_s{s} " + " | ".join("t%d pos mean %.2f med %.2f ; pos_cam mean %.2f med %.2f (30 m)" % (
                k, e[f"tick{k}_30m_pos"]["mean"], e[f"tick{k}_30m_pos"]["median"], e[f"tick{k}_30m_pos_cam"]["mean"], e[f"tick{k}_30m_pos_cam"]["median"]) for k in (0, 1, 2)))
    for row in bins:
        print("ROLL2 R0 heading offset %s deg: " % row["heading_offset_deg"] + " ; ".join("%dm %s (n %d)" % (x["arc_m"], ("%+.2f" % x["mean_ROLL2_m"]) if x["mean_ROLL2_m"] is not None else "n/a", x["n"]) for x in row["per_arc"]))
    for row in vb:
        print("heading offset R0 third tick, speed %s m/s: mean %s median %s (n %d)" % (row["speed_mps"], row["mean_abs_heading_offset_deg"], row["median_abs_heading_offset_deg"], row["n"]))
    print("ZZ STRATROLL-DECOMP DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
