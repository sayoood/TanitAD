#!/usr/bin/env python
"""read_longroll10k_n4.py -- the reader of card PREREG_V8F_PROOF_ADD70 by its note N4 (the own start point of a pre-rolled window), a wrapper around read_longroll10k_n3.py.

The fault (note N4): the family reader (hierrep_read.comps, hierrep_read.kinematics) and the headway reader (headway_read_r2.step_speeds) take step 1 of a track from the frame origin
(0, 0). The wrapper of card ADD70 writes the track of a pre-rolled window in the LOGGED frame at t0, where it starts at the carried offset p0 and not at the origin, so the speed, the
heading, the yaw rate, the curvature, the manoeuvre class and the time gap of step 1 were taken from a jump of the size of the offset.
The repair, and only this: every quantity that is derived from STEPS is computed on the track moved to its own start (track minus p0; steps 2 to 30 are unchanged, step 1 becomes the
own first step); every quantity that is a POSITION in the logged frame (along-track, cross-track, track error, gap, contact, falling behind) is computed on the track as written.
  p0   own position minus logged position at t0 in the logged frame, from the tick file of the run (rows of tick 0: pose, pose_w0), per dump row by (clip, t0)
  n1   every run without a pre-roll and the state-carry control: the largest |p0| is at most 1e-6 m (there this reader IS the reader of note N3)
  n2   (form of note N5) every pre-rolled run: each dump row has its tick row, and |p0| equals the position offset the wrapper wrote for that window (preroll_offsets, dpos) to 1e-6 m.
       Runs under readout K also need the continuity form: the first track point lies within 0.5 m plus one step of travel (the larger of own speed at t0 times 0.2 s and the length
       of the second step) of p0 in at least 0.99 of the windows. For the other pre-rolled runs that share is reported and not required: their tracks jump between steps
  n3   planted, through the real engines: a straight track at 6 m/s started 5 m behind the logged origin with a lead 12 m ahead. The engines as they were read speed error above 0.1 m/s
       and every window unsafe; this reader reads speed, heading and yaw-rate error 0, along-track error 5 m, no window unsafe, the same gaps; with p0 = 0 it equals the engines exactly
When a control fails the names, pairs, cells, readings and the beside block are removed from the result and not printed. Usage: the arguments of read_longroll10k.py, plus the optional
--n3-json <result of the read under note N3> (its names are kept beside the new ones as `names_as_read_under_note_n3`, with the list of names that differ).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import read_longroll10k_n3 as N3  # noqa: E402

R = N3.R
np = R.np
REG, REPORT, PLANT = {}, {}, {}
STEP_KEYS = ("speed_mae_mps", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")
TICKF = "ticks_%s_m10_causal_heads.npz"
DT_STEP, NEAR_M, NEAR_SHARE, ZERO_M = 0.2, 0.5, 0.99, 1e-6


def p0_rows(T, ci, ws):
    """(p0 [N,2], own speed at t0 [N], missing) for the dump rows (ci, ws) from tick-file arrays T (ci, t0, tick, pose [x,y,yaw,v] world, pose_w0 [x,y,yaw] logged world pose at t0)"""
    m0 = np.nonzero(np.asarray(T["tick"], np.int64) == 0)[0]
    keys = np.stack([np.asarray(ci, np.int64), np.asarray(ws, np.int64)], axis=1)
    idx, rep = R.restrict_idx(np.asarray(T["ci"], np.int64)[m0], np.asarray(T["t0"], np.int64)[m0], keys)
    if rep["missing"]:
        return None, None, int(rep["missing"])
    rows = m0[idx]
    own, lg = np.asarray(T["pose"], np.float64)[rows], np.asarray(T["pose_w0"], np.float64)[rows]
    d = own[:, :2] - lg[:, :2]
    c, s = np.cos(lg[:, 2]), np.sin(lg[:, 2])
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1), own[:, 3], 0


def load_dump_dir_n4(d, keys6=("g", "cl", "ws", "eid", "clip_index", "v0")):
    o = _orig_load(d, keys6)
    if o is None or "cl" not in keys6:
        return o
    key = os.path.basename(os.path.dirname(os.path.abspath(d)))
    n = len(o["cl"])
    z = R.npz_load(os.path.join(os.path.dirname(os.path.abspath(d)), "ticks", TICKF % key))
    rep = {"rows": int(n)}
    if z is None or "pose_w0" not in z.files or "pose" not in z.files:
        rep.update(tick_file=False)
        p0 = np.full((n, 2), np.nan)
    else:
        p0, v, miss = p0_rows(z, o["clip_index"], o["ws"])
        rep.update(tick_file=True, missing_rows=int(miss))
        if p0 is None:
            p0 = np.full((n, 2), np.nan)
        else:
            dn = np.hypot(p0[:, 0], p0[:, 1])
            first = np.asarray(o["cl"], np.float64)[:, 0, :] - p0
            clf = np.asarray(o["cl"], np.float64)
            step2 = np.hypot(clf[:, 1, 0] - clf[:, 0, 0], clf[:, 1, 1] - clf[:, 0, 1])
            near = np.hypot(first[:, 0], first[:, 1]) <= NEAR_M + np.maximum(np.abs(v) * DT_STEP, step2)
            rep.update(p0_max_m=float(dn.max()), p0_mean_m=float(dn.mean()), p0_median_m=float(np.median(dn)), share_first_point_near_p0=float(near.mean()))
            po = R.npz_load(os.path.join(os.path.dirname(os.path.abspath(d)), "ticks", "preroll_offsets_%s.npz" % key))
            if po is not None:
                mk = np.asarray(po["kind"]).astype(str) == "model"
                tab = {(int(c), int(w)): float(x) for c, w, x in zip(np.asarray(po["ci"])[mk], np.asarray(po["t0"])[mk], np.asarray(po["dpos"], np.float64)[mk])}
                dw = np.array([tab.get((int(c), int(w)), np.nan) for c, w in zip(o["clip_index"], o["ws"])], np.float64)
                rep.update(offsets_file=True, largest_norm_p0_minus_wrapper_dpos_m=(float(np.max(np.abs(dn - dw))) if np.isfinite(dw).all() else float("inf")))
            else:
                rep.update(offsets_file=False)
    REPORT[key] = rep
    o["p0"] = p0
    return o


def cut_n4(D, idx):
    out = _orig_cut(D, idx)
    if isinstance(out.get("p0"), np.ndarray) and isinstance(out.get("cl"), np.ndarray):
        REG[id(out["cl"])] = (out["p0"], sig(out["cl"]))
    return out


def sig(x):
    return (x.shape, float(x.flat[0]), float(x.flat[-1])) if x.size else (x.shape, 0.0, 0.0)


def moved(x, p0):
    return np.asarray(x, np.float64) - np.asarray(p0, np.float64)[:, None, :]


def live(x):
    ent = REG.get(id(x))
    if ent is None or not isinstance(x, np.ndarray) or ent[1] != sig(x):                      # an id of an array that is gone can come back on another array
        return None
    p0 = ent[0]
    if not np.isfinite(p0).all():
        raise RuntimeError("a track without its own start point reached an engine (note N4)")
    return p0 if np.any(p0) else None


def patch_engines(H, RH):
    if getattr(RH, "_note_n4", False):
        return
    oc, ok_, ow, oss = RH.comps, RH.kinematics, H.window_cells, H.step_speeds

    def comps(X, G):
        c = oc(X, G)
        p0 = live(X)
        if p0 is None:
            return c
        c2 = oc(moved(X, p0), G)
        for k in STEP_KEYS:
            c[k] = c2[k]
        return c

    def kinematics(traj):
        p0 = live(traj)
        return ok_(traj) if p0 is None else ok_(moved(traj, p0))

    def window_cells(g, cl, L):
        p0 = live(cl)
        if p0 is None:
            return ow(g, cl, L)
        calls = []

        def ss(x):                                                                           # the engine calls step_speeds(cl) first, then step_speeds(g)
            calls.append(1)
            return oss(moved(x, p0)) if len(calls) == 1 else oss(x)
        H.step_speeds = ss
        try:
            out = ow(g, cl, L)
        finally:
            H.step_speeds = oss
        if len(calls) != 2:
            raise RuntimeError("the headway engine called step_speeds %d times, not twice (note N4)" % len(calls))
        return out
    RH.comps, RH.kinematics, H.window_cells = comps, kinematics, window_cells
    RH._note_n4 = True
    PLANT.update(planted_n3(H, RH, oc, ow))


def planted_n3(H, RH, oc, ow):
    try:
        n, v, K = 8, 6.0, int(H.K)
        k = np.arange(1, K + 1, dtype=np.float64)
        g = np.stack([np.tile(v * DT_STEP * k, (n, 1)), np.zeros((n, K))], axis=-1)
        off = np.tile(np.array([[-5.0, 0.0]]), (n, 1))
        cl = g + off[:, None, :]
        L = {"G": np.full((n, K + 1), 12.0), "ns": np.full(n, K, np.int64), "lead": np.ones(n, bool)}
        c_old, w_old = oc(cl, g), ow(g, cl, L)
        REG[id(cl)] = (off, sig(cl))
        c_new, w_new = RH.comps(cl, g), H.window_cells(g, cl, L)
        cl0 = cl.copy()
        REG[id(cl0)] = (np.zeros_like(off), sig(cl0))
        c_zero, w_zero = RH.comps(cl0, g), H.window_cells(g, cl0, L)
        same0 = all(np.array_equal(c_zero[q], c_old[q], equal_nan=True) for q in c_old) and all(np.array_equal(np.asarray(w_zero[q]), np.asarray(w_old[q]), equal_nan=True) for q in w_old)
        out = {"engines_as_they_were": {"speed_mae_mps": float(c_old["speed_mae_mps"].mean()), "share_unsafe": float(np.mean(w_old["U"]))},
               "this_reader": {"speed_mae_mps": float(c_new["speed_mae_mps"].max()), "heading_mae_deg": float(np.nanmax(c_new["heading_mae_deg"])), "yaw_rate_mae_degps": float(np.nanmax(c_new["yaw_rate_mae_degps"])),
                               "along_mae_m": float(c_new["along_mae_m"].mean()), "share_unsafe": float(np.mean(w_new["U"])), "share_contact": float(np.mean(w_new["C"])),
                               "largest_gap_difference_m": float(np.max(np.abs(w_new["_gm"] - w_old["_gm"]))), "ade_equal": bool(np.array_equal(c_new["ade_tac_m"], c_old["ade_tac_m"]))},
               "zero_offset_equals_the_engines_exactly": bool(same0)}
        t, e = out["this_reader"], out["engines_as_they_were"]
        out["PASS"] = bool(e["speed_mae_mps"] > 0.1 and e["share_unsafe"] == 1.0 and t["speed_mae_mps"] <= 1e-9 and t["heading_mae_deg"] <= 1e-9 and t["yaw_rate_mae_degps"] <= 1e-9
                           and abs(t["along_mae_m"] - 5.0) <= 1e-9 and t["share_unsafe"] == 0.0 and t["share_contact"] == 0.0 and t["largest_gap_difference_m"] == 0.0 and t["ade_equal"] and same0)
        return out
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}


def engine_n4(a):
    E = _orig_engine(a)
    patch_engines(E[0], E[2])
    return E


def controls_n4():
    """n1, n2 from the report of the dumps this read loaded; n3 from the planted check"""
    n1, n2 = {"runs": {}}, {"runs": {}}
    ok1, ok2 = [], []
    for key, rep in sorted(REPORT.items()):
        run = key[3:].rsplit("_s", 1)[0] if key.startswith("I10") else key
        spec = R.ALL_RUNS.get(run)
        aged = bool(spec and spec[2] > 0 and spec[5] == "plan")
        if aged:
            need_cont = bool(spec[1] == "K")
            good = bool(rep.get("tick_file") and rep.get("missing_rows") == 0 and rep.get("p0_max_m", 0.0) > ZERO_M and rep.get("offsets_file") is True
                        and rep.get("largest_norm_p0_minus_wrapper_dpos_m", 1.0) <= ZERO_M and (not need_cont or rep.get("share_first_point_near_p0", 0.0) >= NEAR_SHARE))
            n2["runs"][key] = dict(rep, continuity_required=need_cont, PASS=good)
            ok2.append(good)
        else:
            good = bool(rep.get("tick_file") and rep.get("missing_rows") == 0 and rep.get("p0_max_m", 1.0) <= ZERO_M)
            n1["runs"][key] = dict(rep, PASS=good)
            ok1.append(good)
    n1["PASS"], n2["PASS"] = bool(ok1 and all(ok1)), bool(ok2 and all(ok2))
    return {"n1_no_preroll_runs_start_at_the_origin": n1, "n2_prerolled_runs_start_at_p0": n2, "n3_planted_start_offset": dict(PLANT) if PLANT else {"PASS": False, "why": "the planted check did not run"}}


def finish(res, text, n3_json=None):
    """adds n1 to n3 to a result of the reader, gates the names on them; returns (res, text to print)"""
    cn = controls_n4()
    res["controls"].update(cn)
    ok = bool(res.get("controls_all_pass") is True and all(v.get("PASS") is True for v in cn.values()))
    res["controls_all_pass"] = ok
    res["note_n4"] = "step-derived quantities on the track moved to its own start point, positions as written (wrapper read_longroll10k_n4.py)"
    lines = ["   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({q: w for q, w in v.items() if q != "runs"}))[:600]) for k, v in cn.items()]
    if not ok:
        for k in ("names", "pairs", "cells", "families", "strategic", "beside", "reported_no_bar"):
            res[k] = None
        cut = text.find("== names")
        text = (text if cut < 0 else text[:cut]) + "\n".join(lines) + "\n== a control did not pass: no name, no level, no difference; the read is reported as written\n"
        return res, text
    text = text + "\n== controls of note N4\n" + "\n".join(lines) + "\n"
    old = R.jl(n3_json) if n3_json else None
    if old and isinstance(old.get("names"), dict) and isinstance(res.get("names"), dict):
        diff = {k: {"under_note_n3": old["names"].get(k), "under_note_n4": v} for k, v in res["names"].items() if old["names"].get(k) != v}
        res["names_as_read_under_note_n3"] = old["names"]
        res["names_that_differ_from_the_read_under_note_n3"] = diff
        text += "== names that differ from the read under note N3 (%d of %d)\n" % (len(diff), len(res["names"])) + "".join("   %-34s %s  ->  %s\n" % (k, v["under_note_n3"], v["under_note_n4"]) for k, v in diff.items())
    return res, text


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    n3_json = None
    if "--n3-json" in argv:
        i = argv.index("--n3-json")
        n3_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    out_path = argv[argv.index("--out") + 1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = N3.main(argv)
    text = buf.getvalue()
    if rc != 0 or not os.path.exists(out_path):
        print(text)
        print("ZZ READ-LONGROLL-N4 rc %s (no result file)" % rc)
        return rc or 3
    res = json.load(io.open(out_path, encoding="utf-8"))
    res, text = finish(res, text, n3_json)
    json.dump(R.conv(res), io.open(out_path, "w", encoding="utf-8", newline="\n"), indent=1)
    print(text)
    print("ZZ READ-LONGROLL-N4 rc %s controls_all_pass %s" % (rc, res["controls_all_pass"]))
    return 0


def read_grid_n4(*args, **kw):
    REG.clear()                                                                              # the tracks of the grid read before are gone
    return _prev_read_grid(*args, **kw)


_orig_load, _orig_cut, _orig_engine, _prev_read_grid = R.load_dump_dir, R.cut, R.engine, R.read_grid
R.load_dump_dir, R.cut, R.engine, R.read_grid = load_dump_dir_n4, cut_n4, engine_n4, read_grid_n4

if __name__ == "__main__":
    sys.exit(main())
