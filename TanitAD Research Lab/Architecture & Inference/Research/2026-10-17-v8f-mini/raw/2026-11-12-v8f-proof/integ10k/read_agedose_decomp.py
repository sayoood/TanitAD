"""read_agedose_decomp.py -- card PREREG_V8F_PROOF_ADD72 (AGEDOSE10K) under its amendment 1: the split of the cost of rollout age under readout K into a POSE part and a STATE part at 6 s (P = 30)
and 12 s (P = 60), on the 284 windows of the P = 60 grid, and the shape of the STATE part between the two ages. numpy; the engines are reused through read_longroll10k.py and the functions of
read_agerule_decomp.py (module M, unedited).

Runs (both model seeds, banked from card ADD70): fresh I10R0KHb20, aged I10R0KP30Hb20 and I10R0KP60Hb20, spread run I10R0Hb20. The fresh, spread and P = 30 dumps hold the windows of the P = 30
grid; each is cut to the keys of grid_p60.json (R.restrict_idx, R.cut). Offsets per age from tick 0 of the tick file of that aged run, restricted to the same keys.
Per age: POSE = pair (d, f), STATE = pair (a, d), TOTAL = pair (a, f) as in read_agerule_decomp.py (d = the fresh track moved rigidly to the own pose of the aged run at t0).
Shape statistic per cell (TRACK TACTICAL:ade_tac_m, ALONG LONGITUDINAL:along_mae_m, SPEED LONGITUDINAL:speed_mae_mps, HEADING LATERAL:heading_mae_deg):
  Q = S(60) - 1.5 * S(30) = mean[(a60 - d60) - 1.5 * (a30 - d30)], read as x = a60 + 1.5 * d30 against y = d60 + 1.5 * a30 through RH.seed_arrays and RH.read_cell.
  NO-STATE-COST when S(30) is not above 0 with its interval excluding 0; else PLATEAU when the interval of Q lies below 0; COMPOUNDS when it lies above 0; else BETWEEN.
Controls: b1, b2, b3, n3 (planted start offset of note N4), b5 and b6 and b4 (POSE + STATE = TOTAL to 1e-9) at each age on the 284 windows, b4_bank (the same split for P = 30 on the full P = 30 grid
against cells[AGE-K] of the banked re-read of card ADD70 through M.ctl_b4; its parts are written beside and name nothing), g1 (window identity of the two grids and of the cut dumps), d1 (planted shape cases).
When a control fails no name, no part, no shape value and no offset statistic is written.
Usage: python read_agedose_decomp.py --pull PULL --code CODEDIR --grid-dir GDIR --add70-json READ_LONGROLL10K.json --out OUT.json [--n-boot 2000]
       python read_agedose_decomp.py --selftest EMPTYFOLDER      (synthetic arrays only; last line ZZ SELFTEST-AGEDOSE-DECOMP ok: True|False (n of m))"""
import argparse
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_agerule_decomp as M  # noqa: E402
import read_longroll10k as R  # noqa: E402
import read_longroll10k_n4 as N4  # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD72", "AGEDOSE10K"
SEEDS = R.SEEDS
FRESH, SPREAD = "R0KHb20", "R0Hb20"
AGED = {30: "R0KP30Hb20", 60: "R0KP60Hb20"}
TIER_LINE = M.TIER_LINE
C_BAR = 1.5
TOL_ID = 1e-9
SHAPE_CELLS = {"TRACK": "TACTICAL:ade_tac_m", "ALONG": "LONGITUDINAL:along_mae_m", "SPEED": "LONGITUDINAL:speed_mae_mps", "HEADING": "LATERAL:heading_mae_deg"}
SHAPE_OUTCOMES = ("PLATEAU", "COMPOUNDS", "BETWEEN", "NO-STATE-COST")
PARTS_NAMED = sum(M.NAMED.values(), [])


# ============================================================================================ small controls of this tool
def ctl_b4_identity(parts):
    """b4 at an age: POSE + STATE equals TOTAL to 1e-9 on every cell (all three pairs are read on one set of windows, note N2)"""
    worst = 0.0
    for c, p in parts.items():
        worst = max(worst, abs(p["POSE"]["D"] + p["STATE"]["D"] - p["TOTAL"]["D"]))
    return {"PASS": bool(parts and worst <= TOL_ID), "largest_abs_dev_pose_plus_state_minus_total": worst, "n_cells": len(parts), "tolerance": TOL_ID}


def ctl_g1(keys_big, keys_small, dumps, expect_n):
    """g1: every window of the smaller grid is a window of the larger one; every cut dump holds exactly the windows of the smaller grid in its order; the count is the card's"""
    big = {(int(c), int(w)) for c, w in np.asarray(keys_big).tolist()}
    ks = np.asarray(keys_small, np.int64)
    notin = [tuple(k) for k in ks.tolist() if (int(k[0]), int(k[1])) not in big]
    same = bool(dumps) and all(len(D["ws"]) == len(ks) and np.array_equal(np.asarray(D["clip_index"], np.int64), ks[:, 0]) and np.array_equal(np.asarray(D["ws"], np.int64), ks[:, 1]) for D in dumps.values())
    return {"PASS": bool(not notin and same and len(ks) == expect_n), "windows_not_in_the_larger_grid": len(notin), "cut_dumps_equal_the_keys_in_order": same, "n_windows": int(len(ks)), "expected": expect_n,
            "n_dumps": len(dumps)}


def bank_check(parts, add70):
    """b4_bank: POSE + STATE and TOTAL of the P = 30 split on the full P = 30 grid equal cells[AGE-K] of the banked re-read to 1e-9 (M.ctl_b4)"""
    if not add70 or add70.get("controls_all_pass") is not True:
        return {"PASS": False, "why": "the banked re-read of card ADD70 is missing or not admitted"}
    cells = (add70.get("cells") or {}).get("AGE-K")
    if not cells:
        return {"PASS": False, "why": "no cells[AGE-K] in the banked re-read"}
    return M.ctl_b4(parts, cells)


def tick_reports_for(trep, strict):
    """b6 asks that the tick-0 rows of an aged file hold the windows of the read. strict: exactly the grid (the file is of that grid). Not strict: the file is of a larger grid (the P = 30 run read on the
    P = 60 windows): every window present once, rows of other windows allowed"""
    return {s: {"holds_exactly": bool(r.get("holds_exactly") if strict else r.get("exact"))} for s, r in trep.items()}


# ============================================================================================ the shape statistic
def shape_name(s30, q):
    """s30, q: cell records (D, interval) -> PLATEAU | COMPOUNDS | BETWEEN | NO-STATE-COST"""
    if not M.pos_excl(s30):
        return "NO-STATE-COST"
    if q["interval"][1] < 0:
        return "PLATEAU"
    if q["interval"][0] > 0:
        return "COMPOUNDS"
    return "BETWEEN"


def shape_q(RH, V60, V30, key, inv, n_ep, ones, n_boot, c=C_BAR):
    """Q = mean[(a60 - d60) - c (a30 - d30)] read as x = a60 + c d30 against y = d60 + c a30 per seed; a window without a value at either age drops out through the finite mask of seed_arrays"""
    xs = [np.asarray(V60[("A", s)][key], np.float64) + c * np.asarray(V30[("D", s)][key], np.float64) for s in SEEDS]
    ys = [np.asarray(V60[("D", s)][key], np.float64) + c * np.asarray(V30[("A", s)][key], np.float64) for s in SEEDS]
    sE, cE = RH.seed_arrays(xs, ys, ones, inv, n_ep)
    return M.cell_rec(RH.read_cell(sE, cE, n_boot, len(SEEDS)))


def state_cell(RH, V, key, inv, n_ep, ones, n_boot):
    xs = [np.asarray(V[("A", s)][key], np.float64) for s in SEEDS]
    ys = [np.asarray(V[("D", s)][key], np.float64) for s in SEEDS]
    sE, cE = RH.seed_arrays(xs, ys, ones, inv, n_ep)
    return M.cell_rec(RH.read_cell(sE, cE, n_boot, len(SEEDS)))


class MiniRH:
    """numpy stand-in of the two engine functions the shape statistic uses (selftest only): per seed the window values of x - y over the finite windows, episode-cluster bootstrap of the mean over seeds"""

    @staticmethod
    def seed_arrays(xs, ys, mask, inv, n_ep):
        d = []
        for x, y in zip(xs, ys):
            v = np.asarray(x, np.float64) - np.asarray(y, np.float64)
            ok = np.isfinite(v) & np.asarray(mask, bool)
            d.append((np.where(ok, v, 0.0), ok.astype(float)))
        return d, (np.asarray(inv, np.int64), int(n_ep))

    @staticmethod
    def read_cell(sE, cE, n_boot, n_seeds):
        inv, n_ep = cE
        sums = np.stack([np.bincount(inv, weights=v, minlength=n_ep) for v, _ in sE])
        cnts = np.stack([np.bincount(inv, weights=o, minlength=n_ep) for _, o in sE])
        d_s = sums.sum(1) / cnts.sum(1)
        rng = np.random.default_rng(0)
        pick = rng.integers(0, n_ep, (n_boot, n_ep))
        bs = (sums[:, pick].sum(2) / cnts[:, pick].sum(2)).mean(0)
        return {"D": float(d_s.mean()), "d_s": [float(x) for x in d_s], "A_lo": float(np.percentile(bs, 2.5)), "A_hi": float(np.percentile(bs, 97.5))}


def planted_arrays(s30, s60, n_ep=40, per=10, seed=5, sd=0.3):
    """per-window arrays a, d at both ages with the given state parts (a - d = s + noise), per seed; keys of the cells: one key 'c'"""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    inv = np.repeat(np.arange(n_ep), per)
    V30, V60 = {}, {}
    for s in SEEDS:
        for V, sp in ((V30, s30), (V60, s60)):
            d = rng.normal(5.0, 1.0, n)
            V[("D", s)] = {"c": d}
            V[("A", s)] = {"c": d + sp + rng.normal(0.0, sd, n)}
    return V30, V60, inv, n_ep


def ctl_d1(RH, n_boot=300):
    """d1: planted arrays -> PLATEAU (equal state parts), COMPOUNDS (doubled), BETWEEN (1.5 times), NO-STATE-COST (0 at 6 s)"""
    out, ok = {}, True
    for case, s30, s60, want in (("equal", 1.0, 1.0, "PLATEAU"), ("doubled", 1.0, 2.0, "COMPOUNDS"), ("one_and_a_half_times", 1.0, 1.5, "BETWEEN"), ("zero_at_6s", 0.0, 0.0, "NO-STATE-COST")):
        V30, V60, inv, n_ep = planted_arrays(s30, s60)
        ones = np.ones(len(inv), bool)
        q = shape_q(RH, V60, V30, "c", inv, n_ep, ones, n_boot)
        sc = state_cell(RH, V30, "c", inv, n_ep, ones, n_boot)
        got = shape_name(sc, q)
        out[case] = {"want": want, "got": got, "S30": sc["D"], "Q": q["D"], "Q_interval": q["interval"]}
        ok &= bool(got == want)
    return dict(out, PASS=bool(ok))


def shape_block(RH, V60, V30, parts60, parts30, ref, n_boot):
    uniq, inv = RH.episode_units(ref["eid"])
    n_ep = len(uniq)
    ones = np.ones(len(inv), bool)
    out = {}
    for nm, cell in SHAPE_CELLS.items():
        key = cell.split(":", 1)[1]
        q = shape_q(RH, V60, V30, key, inv, n_ep, ones, n_boot)
        s30, s60 = parts30[cell]["STATE"], parts60[cell]["STATE"]
        ratio = (s60["D"] / s30["D"]) if abs(s30["D"]) > 1e-12 else None
        out[nm] = {"cell": cell, "name": "%s-STATE-%s-%s" % (PRE, nm, shape_name(s30, q)), "S30": s30, "S60": s60, "ratio_S60_over_S30": ratio, "Q_S60_minus_1p5_S30": q,
                   "POSE30": parts30[cell]["POSE"], "POSE60": parts60[cell]["POSE"], "TOTAL30": parts30[cell]["TOTAL"], "TOTAL60": parts60[cell]["TOTAL"],
                   "share_pose_30": (parts30[cell]["POSE"]["D"] / parts30[cell]["TOTAL"]["D"]) if abs(parts30[cell]["TOTAL"]["D"]) > 1e-12 else None,
                   "share_pose_60": (parts60[cell]["POSE"]["D"] / parts60[cell]["TOTAL"]["D"]) if abs(parts60[cell]["TOTAL"]["D"]) > 1e-12 else None}
    return out


def assemble_dose(controls, payload, extra):
    """the result record. When a control fails no name, no part, no shape value, no offset statistic and no bank part is written."""
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": ok, "names": None, "shape": None, "parts_by_age": None, "offsets_at_t0": None, "bank_split_beside": None}
    res.update(extra)
    if ok:
        res["shape"] = payload["shape"]
        res["names"] = {k: v["name"] for k, v in payload["shape"].items()}
        res["parts_by_age"] = payload["parts_by_age"]
        res["offsets_at_t0"] = payload["offsets"]
        res["bank_split_beside"] = payload["bank_parts"]
    return res


# ============================================================================================ one split
def tick_offsets(pull, run, s, keys):
    z = R.npz_load(os.path.join(R.tick_dir(pull, run, s), M.TICK_FILE % R.key_of(run, s)))
    if z is None:
        return None, None, None, "tick file missing: %s" % R.key_of(run, s)
    T = {k: np.asarray(z[k]) for k in ("ci", "t0", "tick", "pose", "pose_w0")}
    o, rep = M.offsets_from_ticks(T, keys)
    if o is None:
        return None, rep, None, "tick-0 rows missing windows: %s" % R.key_of(run, s)
    cross = None
    po = R.npz_load(os.path.join(R.tick_dir(pull, run, s), "preroll_offsets_%s.npz" % R.key_of(run, s)))
    if po is not None:
        m = np.asarray(po["kind"]).astype(str) == "model"
        pos = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(po["ci"])[m], np.asarray(po["t0"])[m]))}
        sel = [pos.get((int(c), int(t)), -1) for c, t in keys.tolist()]
        if min(sel) >= 0:
            sel = np.asarray(sel)
            cross = (np.asarray(po["dpos"], np.float64)[m][sel], np.abs(np.asarray(po["dpsi_deg"], np.float64)[m][sel]))
        else:
            cross = (np.zeros(0), np.zeros(0))
    return o, rep, cross, None


def split_one(E, keys, raws, aged, pull, n_boot, strict_ticks):
    """the split of one age on the windows `keys`. raws: {(run, seed): full dump}. Returns a dict: complete, why, controls (b5, b6, b4), parts, V, ref, dumps, offsets"""
    H, Q3, RH = E
    why, dumps, offs, trep, cross = [], {}, {}, {}, {}
    for (run_, s), raw in raws.items():
        if run_ not in (FRESH, SPREAD, aged):
            continue
        idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
        if rep["missing"] or rep["duplicated_windows"]:
            why.append("%s: %d windows missing, %d twice" % (R.key_of(run_, s), rep["missing"], rep["duplicated_windows"]))
            continue
        dumps[(run_, s)] = R.cut(raw, idx)
    for s in SEEDS:
        o, rep, cr, w = tick_offsets(pull, aged, s, keys)
        if w:
            why.append(w)
            continue
        offs[s], trep[s] = o, rep
        if cr is not None:
            cross[s] = cr
    complete = bool(not why and all((r, s) in dumps for r in (FRESH, aged, SPREAD) for s in SEEDS) and all(s in offs for s in SEEDS))
    out = {"complete": complete, "why": why, "controls": {}, "parts": None, "V": None, "ref": None, "dumps": dumps, "offsets": None, "common_windows_note_n2": None}
    if not complete:
        for k in ("b5", "b6", "b4"):
            out["controls"][k] = {"PASS": False, "why": "; ".join(why) or "inputs incomplete"}
        return out
    ref = dumps[(FRESH, 0)]
    out["controls"]["b6"] = M.ctl_b6({k: v for k, v in dumps.items() if k[0] in (FRESH, aged, SPREAD)}, keys, tick_reports_for(trep, strict_ticks))
    glon, glat = RH.classify(*RH.kinematics(ref["g"]))
    joint_g = glon * 3 + glat
    V, off_log = {}, {}
    for s in SEEDS:
        f, ag = dumps[(FRESH, s)]["cl"], dumps[(aged, s)]["cl"]
        o = offs[s]
        d = M.displace_reg(f, o["dx"], o["dy"], o["dpsi"])
        for tag, cl in (("F", f), ("D", d), ("A", ag)):
            V[(tag, s)] = M.make_cells(RH, cl, ref["g"], glon, glat, joint_g)
        V[(M.SPREAD, s)] = M.make_cells(RH, dumps[(SPREAD, s)]["cl"], ref["g"], glon, glat, joint_g)
        off_log["seed%d" % s] = {"position_m": R.off_stats(o["dpos"]), "heading_abs_deg": R.off_stats(np.abs(np.degrees(o["dpsi"]))), "own_speed_mps": R.off_stats(o["v_own"])}
    out["controls"]["b5"] = M.ctl_b5({s: dumps[(aged, s)]["cl"] for s in SEEDS}, offs, cross)
    out["common_windows_note_n2"] = M.common_support(V, SEEDS)
    parts, counts = M.read_parts(E, V, ref, n_boot, None)
    out.update(parts=parts, V=V, ref=ref, offsets=off_log, counts=counts)
    out["controls"]["b4"] = ctl_b4_identity(parts)
    return out


# ============================================================================================ the read
def run(a):
    g30, g60 = R.jl(os.path.join(a.grid_dir, "grid_p30.json")), R.jl(os.path.join(a.grid_dir, "grid_p60.json"))
    if g30 is None or g60 is None:
        print("STOP: grid_p30.json or grid_p60.json missing in %s" % a.grid_dir)
        return 3
    E = R.engine(a)
    H, Q3, RH = E
    J = H.load_join()
    names_ = [str(c) for c in J["clip"]]
    keys30, keys60 = R.grid_keys(g30, names_), R.grid_keys(g60, names_)
    raws = {}
    for run_ in (FRESH, SPREAD, AGED[30], AGED[60]):
        for s in SEEDS:
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is not None:
                raws[(run_, s)] = raw
    controls = {}
    controls["b1_zero_offset_identity"] = M.ctl_b1(RH.comps, disp=M.displace_reg)
    controls["b2_one_metre_ahead"] = M.ctl_b2(RH.comps, disp=M.displace_reg)
    controls["b3_ninety_degrees"] = M.ctl_b3(RH.comps, disp=M.displace_reg)
    controls["n3_planted_start_offset"] = dict(N4.PLANT) if N4.PLANT else {"PASS": False, "why": "the planted check of note N4 did not run"}
    sp = {60: split_one(E, keys60, raws, AGED[60], a.pull, a.n_boot, True), 30: split_one(E, keys60, raws, AGED[30], a.pull, a.n_boot, False)}
    spb = split_one(E, keys30, raws, AGED[30], a.pull, a.n_boot_control, True)
    for P in (30, 60):
        for k in ("b5", "b6", "b4"):
            controls["%s_P%d_on_%d_windows" % (k, P, len(keys60))] = sp[P]["controls"][k]
    alldumps = {("P%d" % P,) + k: v for P in (30, 60) for k, v in sp[P]["dumps"].items()}
    controls["g1_window_identity_of_the_grids"] = ctl_g1(keys30, keys60, alldumps, R.CARD_COUNTS["60"])
    bank = R.jl(a.add70_json)
    controls["b4_bank_P30_full_grid_equals_card_add70"] = dict(bank_check(sp_parts(spb), bank) if spb["complete"] else {"PASS": False, "why": "; ".join(spb["why"]) or "inputs incomplete"},
                                                                split_controls_reported_not_judged={k: v.get("PASS") for k, v in spb["controls"].items()}, n_boot_of_the_beside_parts=a.n_boot_control)
    controls["d1_shape_statistic_planted"] = ctl_d1(RH, a.n_boot_control)
    payload, extra = None, {"counts": sp[60].get("counts"), "common_windows_note_n2": {"P30": sp[30]["common_windows_note_n2"], "P60": sp[60]["common_windows_note_n2"]}}
    if all(sp[P]["complete"] for P in (30, 60)):
        payload = {"shape": shape_block(RH, sp[60]["V"], sp[30]["V"], sp[60]["parts"], sp[30]["parts"], sp[60]["ref"], a.n_boot),
                   "parts_by_age": {"P30": sp[30]["parts"], "P60": sp[60]["parts"]}, "offsets": {"P30": sp[30]["offsets"], "P60": sp[60]["offsets"]}, "bank_parts": spb["parts"]}
    else:
        controls["inputs"] = {"PASS": False, "why": "; ".join(sp[30]["why"] + sp[60]["why"])}
    res = assemble_dose(controls, payload or {}, extra)
    if res["controls_all_pass"] and payload is None:
        res["controls_all_pass"] = False
    print(TIER_LINE)
    print("== controls")
    for k, v in controls.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x != "PASS"}))[:600]))
    if res["controls_all_pass"]:
        print_table(res)
    else:
        print("== a control did not pass: no name, no part, no shape value is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-AGEDOSE-DECOMP rc 0 controls_all_pass %s" % res["controls_all_pass"])
    return 0


def sp_parts(sp):
    return sp["parts"]


def print_table(res):
    print("== names")
    for k, n in res["names"].items():
        print("   %-8s %s" % (k, n))
    print("== state part S(30), S(60), ratio, Q = S(60) - 1.5 S(30); POSE and TOTAL at both ages; share of POSE")
    f = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["D"], c["interval"][0], c["interval"][1])      # noqa: E731
    for k, v in res["shape"].items():
        print("   %-8s S30 %s  S60 %s  ratio %s  Q %s" % (k, f(v["S30"]), f(v["S60"]), "n/a" if v["ratio_S60_over_S30"] is None else "%.2f" % v["ratio_S60_over_S30"], f(v["Q_S60_minus_1p5_S30"])))
        print("   %-8s POSE30 %s POSE60 %s TOTAL30 %s TOTAL60 %s share %s / %s" % ("", f(v["POSE30"]), f(v["POSE60"]), f(v["TOTAL30"]), f(v["TOTAL60"]), v["share_pose_30"], v["share_pose_60"]))
    print("== offsets at t0 (reported, no bar)")
    for P, o in res["offsets_at_t0"].items():
        print("   %s %s" % (P, json.dumps(R.conv(o))[:900]))


# ============================================================================================ selftest (synthetic arrays only)
def selftest(folder):
    res = []

    def chk(name, ok):
        res.append((name, bool(ok)))
        print("   %-112s %s" % (name, bool(ok)))
    print(TIER_LINE)
    if os.path.isdir(folder) and os.listdir(folder):
        print("STOP: the selftest folder is not empty")
        return 2
    os.makedirs(folder, exist_ok=True)

    def cell(D, lo, hi):
        return {"D": D, "interval": [lo, hi], "per_seed": [D, D], "verdict": None}
    # ---- d1 on the stand-in engine functions
    d1 = ctl_d1(MiniRH)
    for case, v in d1.items():
        if case != "PASS":
            chk("d1 planted %-22s -> %s (S30 %+.3f, Q %+.3f)" % (case, v["got"], v["S30"], v["Q"]), v["got"] == v["want"])
    chk("d1 passes as a whole", d1["PASS"])
    # ---- shape_name on records
    s_pos, s_zero = cell(1.0, 0.5, 1.5), cell(0.0, -0.3, 0.3)
    chk("shape_name: Q interval below 0 -> PLATEAU", shape_name(s_pos, cell(-0.5, -0.8, -0.2)) == "PLATEAU")
    chk("shape_name: Q interval above 0 -> COMPOUNDS", shape_name(s_pos, cell(0.5, 0.2, 0.8)) == "COMPOUNDS")
    chk("shape_name: Q interval across 0 -> BETWEEN", shape_name(s_pos, cell(0.0, -0.2, 0.2)) == "BETWEEN")
    chk("shape_name: S30 interval touching 0 -> NO-STATE-COST whatever Q", shape_name(cell(0.4, -0.1, 0.9), cell(-0.5, -0.8, -0.2)) == "NO-STATE-COST" and shape_name(s_zero, cell(0.5, 0.2, 0.8)) == "NO-STATE-COST")
    chk("shape_name: S30 below 0 with its interval excluding 0 -> NO-STATE-COST", shape_name(cell(-1.0, -1.5, -0.5), cell(0.5, 0.2, 0.8)) == "NO-STATE-COST")
    chk("the bar is 1.5 and the four cells are those of the amendment", C_BAR == 1.5 and sorted(SHAPE_CELLS.values()) == sorted(["TACTICAL:ade_tac_m", "LONGITUDINAL:along_mae_m", "LONGITUDINAL:speed_mae_mps", "LATERAL:heading_mae_deg"]))
    # ---- the statistic itself: x and y are built as written (a60 + 1.5 d30 against d60 + 1.5 a30)
    V30, V60, inv, n_ep = planted_arrays(1.0, 2.0, sd=0.0)
    q = shape_q(MiniRH, V60, V30, "c", inv, n_ep, np.ones(len(inv), bool), 200)
    chk("shape_q: state parts 1 at 6 s and 2 at 12 s with no noise read Q = 2 - 1.5 = +0.5", abs(q["D"] - 0.5) < 1e-12)
    V30n, V60n = {k: {"c": v["c"].copy()} for k, v in V30.items()}, {k: {"c": v["c"].copy()} for k, v in V60.items()}
    V60n[("A", 0)]["c"][:7] = np.nan
    qn = shape_q(MiniRH, V60n, V30n, "c", inv, n_ep, np.ones(len(inv), bool), 200)
    chk("shape_q: a window without a value at one age drops out and the rest is unchanged", abs(qn["D"] - 0.5) < 1e-12)
    # ---- b4 identity
    parts = {"c1": {"POSE": cell(0.3, 0.1, 0.5), "STATE": cell(0.2, 0.0, 0.4), "TOTAL": cell(0.5, 0.2, 0.8)}}
    chk("b4 identity passes when POSE + STATE = TOTAL", ctl_b4_identity(parts)["PASS"])
    chk("b4 identity fails on a planted 1e-6 in STATE", not ctl_b4_identity({"c1": dict(parts["c1"], STATE=cell(0.2 + 1e-6, 0, 0.4))})["PASS"])
    chk("b4 identity fails on a planted 1e-6 in TOTAL", not ctl_b4_identity({"c1": dict(parts["c1"], TOTAL=cell(0.5 + 1e-6, 0, 0.4))})["PASS"])
    chk("b4 identity fails with no cell", not ctl_b4_identity({})["PASS"])
    # ---- b4_bank
    add70 = {"controls_all_pass": True, "cells": {"AGE-K": {"c1": {"D": 0.5}}}}
    chk("b4_bank passes when POSE + STATE and TOTAL equal the banked cell", bank_check(parts, add70)["PASS"])
    chk("b4_bank fails on a banked cell 1e-6 away", not bank_check(parts, {"controls_all_pass": True, "cells": {"AGE-K": {"c1": {"D": 0.5 + 1e-6}}}})["PASS"])
    chk("b4_bank fails when the banked re-read is not admitted", not bank_check(parts, dict(add70, controls_all_pass=False))["PASS"] and not bank_check(parts, None)["PASS"])
    chk("b4_bank fails when cells[AGE-K] is absent", not bank_check(parts, {"controls_all_pass": True, "cells": {}})["PASS"])
    # ---- g1
    k30 = np.array([[0, 5], [0, 9], [1, 4], [1, 8], [2, 7]], np.int64)
    k60 = k30[[1, 3, 4]]
    dmp = {("a", 0): {"ws": k60[:, 1], "clip_index": k60[:, 0]}, ("b", 1): {"ws": k60[:, 1].copy(), "clip_index": k60[:, 0].copy()}}
    chk("g1 passes: subset of the larger grid and cut dumps in the order of the keys", ctl_g1(k30, k60, dmp, 3)["PASS"])
    chk("g1 fails when a window of the smaller grid is not in the larger one", not ctl_g1(k30, np.array([[0, 9], [3, 3], [2, 7]], np.int64), dmp, 3)["PASS"])
    dm2 = {("a", 0): {"ws": k60[::-1, 1], "clip_index": k60[::-1, 0]}}
    chk("g1 fails when a cut dump holds the windows in another order", not ctl_g1(k30, k60, dm2, 3)["PASS"])
    chk("g1 fails when the count is not the card's", not ctl_g1(k30, k60, dmp, 284)["PASS"])
    chk("g1 fails with no dump", not ctl_g1(k30, k60, {}, 3)["PASS"])
    # ---- b6 tick reports
    rep = {"exact": True, "holds_exactly": False}
    chk("b6 ticks: a P = 30 file with extra windows passes when not strict and fails when strict",
        tick_reports_for({0: rep}, False)[0]["holds_exactly"] is True and tick_reports_for({0: rep}, True)[0]["holds_exactly"] is False)
    chk("b6 ticks: a file with a missing window fails in both forms", tick_reports_for({0: {"exact": False, "holds_exactly": False}}, False)[0]["holds_exactly"] is False)
    # ---- gating
    shape = {k: {"name": "%s-STATE-%s-PLATEAU" % (PRE, k)} for k in SHAPE_CELLS}
    payload = {"shape": shape, "parts_by_age": {"P30": 1}, "offsets": {"P30": 2}, "bank_parts": {"c": 3}}
    ok = assemble_dose({"b1": {"PASS": True}, "g1": {"PASS": True}}, payload, {"counts": {"n": 284}})
    chk("assemble: all controls pass -> names, shape, parts, offsets and bank parts are written", ok["controls_all_pass"] and ok["names"]["TRACK"] == "AGEDOSE10K-STATE-TRACK-PLATEAU" and ok["parts_by_age"] and ok["offsets_at_t0"] and ok["bank_split_beside"])
    bad = assemble_dose({"b1": {"PASS": True}, "g1": {"PASS": False}}, payload, {"counts": {"n": 284}})
    chk("assemble: a failed control blanks names, shape, parts, offsets and bank parts; controls and counts stay",
        (not bad["controls_all_pass"]) and all(bad[k] is None for k in ("names", "shape", "parts_by_age", "offsets_at_t0", "bank_split_beside")) and bad["controls"]["g1"]["PASS"] is False and bad["counts"] == {"n": 284})
    wr = os.path.join(folder, "out_failed.json")
    json.dump(R.conv(bad), io.open(wr, "w", encoding="utf-8", newline="\n"))
    back = json.load(io.open(wr, encoding="utf-8"))
    chk("assemble: the written file of a failed read holds no name and no part", back["names"] is None and back["parts_by_age"] is None and back["shape"] is None)
    chk("assemble: no controls at all is a failure", assemble_dose({}, payload, {})["names"] is None)
    # ---- tables
    chk("tables: ages 30 and 60, the aged runs are the K runs of card ADD70, the fresh and spread runs are those of the amendment",
        AGED == {30: "R0KP30Hb20", 60: "R0KP60Hb20"} and FRESH == "R0KHb20" and SPREAD == "R0Hb20" == M.SPREAD and R.CARD_COUNTS["60"] == 284)
    chk("tier line is the family reader's tier line", TIER_LINE == R.TIER_LINE + "; heading bound 0.2 1/m")
    chk("names are unique over cells and outcomes", len({"%s-STATE-%s-%s" % (PRE, c, o) for c in SHAPE_CELLS for o in SHAPE_OUTCOMES}) == 16)
    chk("every shape cell is a named cell of the family reader", all(c in PARTS_NAMED for c in SHAPE_CELLS.values()))
    # ---- the text of this file
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [w for w in words if w in src])
    ok_all = all(v for _, v in res)
    print("ZZ SELFTEST-AGEDOSE-DECOMP ok: %s (%d of %d)" % (ok_all, sum(v for _, v in res), len(res)))
    return 0 if ok_all else 1


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
