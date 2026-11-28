#!/usr/bin/env python
"""read_handsrc_decomp.py -- read of card PREREG_V8F_PROOF_ADD73 (HANDSRC10K): which input of the state hand-over at the first scored tick carries the state part of the cost of age.

Runs (P = 30 grid, readout K, form R0, heading bound 0.2 1/m, model seeds 0 and 1): fresh R0KHb20, spread R0Hb20, own R0KP30Hb20 (banked), and the three modes of --t0-state:
LOGGED R0KP30LgHb20, SPEED-LOGGED R0KP30SlHb20, PATH-LOGGED R0KP30PlHb20.

Part 1, the cells of section 3 of the card. Per run the split of card ADD71 part B (note N2) through the tool of card ADD72 Amendment 1 (read_agedose_decomp.split_one, not edited):
F fresh track, D = F moved rigidly to the own pose at t0, A the aged track. Per mode M and cell, on the windows where the cell has a value in all tracks used (F, D, A_own, A_M; per model
seed): state part of own = A_own minus D, state part of the mode = A_M minus D, removed = A_own minus A_M, Q = removed minus one half of the state part of own, read as
x = 0.5 A_own + 0.5 D against y = A_M through the family engine (two-level bootstrap). Names: NO-STATE-COST | CARRIES | PART | WORSE | NOT-SEPARATED.
Controls of part 1 (card section 4): c0 logged first segment equals D; c1 offsets of every mode equal those of own; c2 pre-roll logs name the mode and the groups; b1 to b3, b5, b6 and the
identity b4 per run (b5 in the form of record of note N3 of the card: one step of travel at the speed handed to the planner at the tick; its form in the split tool is written beside);
the banked total of the pair AGE-K of the re-read of card ADD70 for the own run; planted start offset of note N4; h1 planted name rule; q1 the two identities
state_own minus state_mode = removed and Q = removed minus one half of state_own on the differences of the means to 1e-9 (note N2 of the card).
A control of part 1 that does not pass: no name, no part, no value of part 1 is written.

Part 2, the four families for the pairs mode minus own (card section 3, last point; note N2 of the card): the grid read of the reader of card ADD71 (read_longroll10k.read_grid under
its notes N3 and N4: headway U, C, F and the family cells, with its engine controls) and its STRATEGIC read (strategic_grid, instrument of card ADD66, tick t0 + 2.0 s), on the runs of
this card. Its own controls: the engine controls of the grid read, stage A and B of the strategic read, and a1 (the pair own minus fresh read here equals the pair AGE-K of the banked
re-read of card ADD70 to 1e-9). The file controls of that reader (wrapper hash, run logs) are bound to the wrapper generation before this card and are not run; in their place stand
the pod check of the wrapper (note N1) and c1, c2 of part 1. A control of part 2 that does not pass: no family name is written, and the result says for each family what is missing.
With --p 60 the same read for card ADD74 (runs R0KP60LgHb20, R0KP60SlHb20, R0KP60PlHb20 against R0KP60Hb20 on the P = 60 grid; the own split is banked against the read of card ADD72
Amendment 1, --agedose-json; a1 against the pair AGE-K@P60 of the re-read of card ADD70). --p 30 (default) is card ADD73.
Usage: python read_handsrc_decomp.py --pull PULL --code CODE --grid-dir DIR --add70-json FILE --out OUT.json [--w W --tree TREE] [--p 30|60 --agedose-json FILE] [--n-boot 2000]   |   --selftest
Last line: ZZ READ-HANDSRC-DECOMP rc <rc> controls_all_pass <bool> family_controls_all_pass <bool>
"""
import argparse
import io
import json
import os
import sys

import read_agedose_decomp as AD

M, R, N4, np = AD.M, AD.R, AD.N4, AD.np
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
P_STEPS = 30                                                                                     # pre-roll steps of the card: 30 (card ADD73) or 60 (card ADD74), set by configure()
CARD, PRE = "PREREG_V8F_PROOF_ADD73", "HANDSRC10K"
OWN = "R0KP30Hb20"
MODES = {"LOGGED": ("R0KP30LgHb20", "logged"), "SPEED-LOGGED": ("R0KP30SlHb20", "speed_logged"), "PATH-LOGGED": ("R0KP30PlHb20", "path_logged")}
CELLS = {"TRACK": "TACTICAL:ade_tac_m", "ALONG": "LONGITUDINAL:along_mae_m", "SPEED": "LONGITUDINAL:speed_mae_mps", "HEADING": "LATERAL:heading_mae_deg"}
BESIDE = ("LATERAL:cross_mae_m", "LATERAL:yaw_rate_mae_degps", "TACTICAL:lon_err_rate", "TACTICAL:lat_err_rate", "TACTICAL:joint_err_rate")
SEG = 10                                                                                         # steps of the first replan segment
TOL_C0 = 1e-3                                                                                    # metres
TOL_ID = 1e-9
FAM_PAIRS = {name + "-VS-OWN": (run, OWN) for name, (run, _) in MODES.items()}
BANK_PAIR, BANK_NAME = "OWN-MINUS-FRESH-AS-BANKED", "AGE-K"
RUN_ROW = ("R0", "K", 30, "deadreckon", "30", "plan", None, 0.2)                                 # the table row of a P = 30 run under readout K (reader of card ADD71)


def configure(p):
    """sets the run names, the card and the name prefix for a pre-roll of p steps: 30 = card ADD73 (the values above), 60 = card ADD74 (the same read on the P = 60 grid)"""
    global P_STEPS, CARD, PRE, OWN, MODES, FAM_PAIRS, RUN_ROW
    if p not in (30, 60):
        raise SystemExit("REFUSED: --p is 30 or 60")
    P_STEPS = int(p)
    CARD, PRE = ("PREREG_V8F_PROOF_ADD73", "HANDSRC10K") if p == 30 else ("PREREG_V8F_PROOF_ADD74", "HANDSRC60")
    OWN = "R0KP%dHb20" % p
    MODES = {"LOGGED": ("R0KP%dLgHb20" % p, "logged"), "SPEED-LOGGED": ("R0KP%dSlHb20" % p, "speed_logged"), "PATH-LOGGED": ("R0KP%dPlHb20" % p, "path_logged")}
    FAM_PAIRS = {name + "-VS-OWN": (run, OWN) for name, (run, _) in MODES.items()}
    RUN_ROW = ("R0", "K", p, "deadreckon", str(p), "plan", None, 0.2)


def bank_pair(banked):
    """the pair own minus fresh of the banked re-read of card ADD70: at P = 30 the named pair AGE-K, at P = 60 the pair AGE-K@P60 written beside -> (pair record with A, B, headway; cells)"""
    if P_STEPS == 30:
        return banked["pairs"][BANK_NAME], banked["cells"][BANK_NAME]
    x = banked["beside"]["P60_pairs"][BANK_NAME + "@P60"]
    return x, x["cells"]


def bank_split(parts, agedose):
    """P = 60: POSE, STATE and TOTAL of the own split equal those of the banked read of card ADD72 Amendment 1 (parts_by_age[P60], the same windows and tool) on the differences of the means to 1e-9"""
    try:
        if not agedose or agedose.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the banked read of card ADD72 is missing or not admitted"}
        bp = agedose["parts_by_age"]["P60"]
        worst = max(abs(parts[c][k]["D"] - bp[c][k]["D"]) for c in bp for k in ("POSE", "STATE", "TOTAL"))
        return {"PASS": bool(sorted(parts) == sorted(bp) and worst <= TOL_ID), "largest_abs_dev": worst, "n_cells": len(bp), "tolerance": TOL_ID}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


# ============================================================================================ part 1
def name_rule(s_own, removed, q):
    """s_own = state part of the own run, removed = A_own minus A_M, q = removed minus one half of the own state part (cell records with D and interval) -> the outcome word of the card"""
    if not M.pos_excl(s_own):
        return "NO-STATE-COST"
    if q["interval"][0] > 0:
        return "CARRIES"
    if removed["interval"][0] > 0:
        return "PART"
    if removed["interval"][1] < 0:
        return "WORSE"
    return "NOT-SEPARATED"


def pair(RH, xs, ys, inv, n_ep, n_boot):
    ones = np.ones(len(inv), bool)
    sE, cE = RH.seed_arrays([np.asarray(x, np.float64) for x in xs], [np.asarray(y, np.float64) for y in ys], ones, inv, n_ep)
    return M.cell_rec(RH.read_cell(sE, cE, n_boot, len(SEEDS)))


def cell_block(RH, Vown, Vm, key, inv, n_ep, n_boot):
    """per cell key, on the windows where the cell has a value in A_own, D and A_M (per seed; F is in each through the split): the state part of own and of the mode, removed, Q"""
    a_own, d_own, a_m, n_common = [], [], [], []
    for s in SEEDS:
        ao, do, am, dm = (np.asarray(v, np.float64) for v in (Vown[("A", s)][key], Vown[("D", s)][key], Vm[("A", s)][key], Vm[("D", s)][key]))
        ok = np.isfinite(ao) & np.isfinite(do) & np.isfinite(am) & np.isfinite(dm)
        a_own.append(np.where(ok, ao, np.nan))
        d_own.append(np.where(ok, do, np.nan))
        a_m.append(np.where(ok, am, np.nan))
        n_common.append(int(ok.sum()))
    s_own = pair(RH, a_own, d_own, inv, n_ep, n_boot)
    s_m = pair(RH, a_m, d_own, inv, n_ep, n_boot)
    removed = pair(RH, a_own, a_m, inv, n_ep, n_boot)
    q = pair(RH, [0.5 * a + 0.5 * d for a, d in zip(a_own, d_own)], a_m, inv, n_ep, n_boot)
    share = (removed["D"] / s_own["D"]) if abs(s_own["D"]) > 1e-12 else None
    return {"state_own": s_own, "state_mode": s_m, "removed": removed, "Q_removed_minus_half_state": q, "removed_share": share, "word": name_rule(s_own, removed, q),
            "n_windows_per_seed": n_common}


def ctl_q1(blocks):
    """q1: blocks = list of cell blocks; state_own minus state_mode equals removed, and Q equals removed minus one half of state_own, on the differences of the means"""
    w1 = w2 = 0.0
    for b in blocks:
        w1 = max(w1, abs(b["state_own"]["D"] - b["state_mode"]["D"] - b["removed"]["D"]))
        w2 = max(w2, abs(b["Q_removed_minus_half_state"]["D"] - (b["removed"]["D"] - 0.5 * b["state_own"]["D"])))
    return {"PASS": bool(blocks and w1 <= TOL_ID and w2 <= TOL_ID), "largest_abs_dev_state_own_minus_state_mode_minus_removed": w1, "largest_abs_dev_Q": w2, "n_blocks": len(blocks), "tolerance": TOL_ID}


def seg_ade(cl, g, n=SEG):
    """mean over the first n steps of the distance between a track [N, T, 2] and the logged track"""
    d = np.asarray(cl, np.float64)[:, :n, :] - np.asarray(g, np.float64)[:, :n, :]
    return np.hypot(d[..., 0], d[..., 1]).mean(axis=1)


def moved(f, o):
    return M.displace(f, o["dx"], o["dy"], o["dpsi"])


def handed_speed(pull, run, s, keys):
    """the speed in the batch the model read at the first tick of each window (field v0 of the tick-0 rows of the tick file of the run), in the order of `keys`; None when rows are missing"""
    z = R.npz_load(os.path.join(R.tick_dir(pull, run, s), M.TICK_FILE % R.key_of(run, s)))
    if z is None:
        return None
    try:
        v0 = np.asarray(z["v0"], np.float64)
    except KeyError:
        return None
    m0 = np.nonzero(np.asarray(z["tick"], np.int64) == 0)[0]
    idx, rep = R.restrict_idx(np.asarray(z["ci"], np.int64)[m0], np.asarray(z["t0"], np.int64)[m0], keys)
    if rep["missing"]:
        return None
    return v0[m0[idx]]


def ctl_b5_handed(a_tracks, offs, cross, v_handed):
    """b5 in the form of record of note N3: control b5 of the split tool (M.ctl_b5, not edited) with one step of travel taken at the speed handed to the planner at the tick"""
    if not a_tracks or any(s not in offs or v_handed.get(s) is None for s in a_tracks):
        return {"PASS": False, "why": "tick-0 rows or the handed speed missing"}
    o2 = {s: dict(offs[s], v_own=np.asarray(v_handed[s], np.float64)) for s in a_tracks}
    out = M.ctl_b5(a_tracks, o2, cross)
    out["one_step_of_travel"] = "speed handed at the tick (v0) times 0.2 s (note N3)"
    out["largest_abs_handed_minus_own_speed_mps"] = {("seed%d" % s): float(np.max(np.abs(np.asarray(v_handed[s], np.float64) - np.asarray(offs[s]["v_own"], np.float64)))) for s in a_tracks}
    return out


def p2_check(family, first):
    """p2: part 2 of this read equals part 2 of the first read of the card (names, headway shares, family cells; 1e-9)"""
    try:
        f1 = (first or {}).get("family_read") or {}
        if f1.get("controls_all_pass") is not True or family.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "part 2 of this read or of the first read did not pass its controls"}
        d = []
        for nm in family["pairs"]:
            for c in ("U", "C", "F"):
                d.append(maxabs(family["pairs"][nm]["headway"][c]["pooled"], f1["pairs"][nm]["headway"][c]["pooled"]))
            d += [maxabs(family["cells"][nm][c]["D"], f1["cells"][nm][c]["D"]) for c in family["cells"][nm]]
        return {"PASS": bool(family["names"] == f1["names"] and d and max(d) <= TOL_ID), "same_names": family["names"] == f1["names"], "largest_difference_of_values": max(d), "values_compared": len(d)}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def ctl_c0(seg_dev):
    """c0: under logged the first replan segment of the aged track equals the moved fresh track; seg_dev = {seed: largest distance in metres}"""
    worst = max(seg_dev.values()) if seg_dev else float("inf")
    return {"PASS": bool(seg_dev and len(seg_dev) == len(SEEDS) and worst <= TOL_C0), "largest_distance_m_per_seed": {("seed%d" % s): float(v) for s, v in seg_dev.items()}, "tolerance_m": TOL_C0,
            "steps": SEG}


def ctl_c1(off_own, off_modes):
    """c1: the pre-roll offsets (dx, dy, dpsi at t0) of every mode equal those of the own run exactly"""
    worst, out = 0.0, {}
    for name, per_seed in off_modes.items():
        w = 0.0
        for s in SEEDS:
            if s not in per_seed or s not in off_own:
                return {"PASS": False, "why": "offsets missing: %s seed %d" % (name, s)}
            for k in ("dx", "dy", "dpsi"):
                a, b = np.asarray(per_seed[s][k], np.float64), np.asarray(off_own[s][k], np.float64)
                if a.shape != b.shape:
                    return {"PASS": False, "why": "offset shapes differ: %s seed %d" % (name, s)}
                w = max(w, float(np.abs(a - b).max()) if a.size else 0.0)
        out[name] = w
        worst = max(worst, w)
    return {"PASS": bool(off_modes and worst == 0.0), "largest_difference_per_mode": out}


def ctl_c2(logs):
    """c2: logs = {mode name: {seed: preroll log dict or None}}; each log names its mode, and the group that is not replaced differs by exactly 0 from the own batch"""
    out, ok = {}, bool(logs)
    for name, (run, mode) in MODES.items():
        for s in SEEDS:
            lg = (logs.get(name) or {}).get(s)
            key = "%s:seed%d" % (name, s)
            if not isinstance(lg, dict):
                out[key], ok = {"ok": False, "why": "no pre-roll log"}, False
                continue
            st = lg.get("t0_state_first_tick_max_abs_diff") or {}
            kinds = {}
            good = lg.get("t0_state") == mode and bool(st)
            for kind, rec in st.items():
                mx = (rec or {}).get("max_abs") or {}
                p, h, v = float(mx.get("positions_m", -1)), float(mx.get("headings_rad", -1)), float(mx.get("speed_ms", -1))
                if mode == "logged":
                    g = (p > 0 or h > 0) and v > 0
                elif mode == "speed_logged":
                    g = p == 0.0 and h == 0.0 and v > 0
                else:
                    g = v == 0.0 and (p > 0 or h > 0)
                kinds[kind] = {"positions_m": p, "headings_rad": h, "speed_ms": v, "groups_as_defined": bool(g)}
                good = good and g
            out[key] = {"ok": bool(good), "t0_state": lg.get("t0_state"), "kinds": kinds}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def planted(inv_n=40, per=10, seed=7, sd=0.3):
    rng = np.random.default_rng(seed)
    n = inv_n * per
    inv = np.repeat(np.arange(inv_n), per)

    def mk(state_own, keep):
        Vo, Vm = {}, {}
        for s in SEEDS:
            d = rng.normal(5.0, 1.0, n)
            Vo[("D", s)], Vm[("D", s)] = {"c": d}, {"c": d}
            Vo[("A", s)] = {"c": d + state_own + rng.normal(0.0, sd, n)}
            Vm[("A", s)] = {"c": d + state_own * keep + rng.normal(0.0, sd, n)}
        return Vo, Vm
    return mk, inv, inv_n


def ctl_h1(RH, n_boot=300):
    """h1: planted arrays -> CARRIES (mode removes 0.8 of the state part), PART (0.2), NOT-SEPARATED (0), WORSE (mode adds one half), NO-STATE-COST (state part 0)"""
    mk, inv, n_ep = planted()
    out, ok = {}, True
    for case, state_own, keep, want in (("removes_0.8", 1.0, 0.2, "CARRIES"), ("removes_0.2", 1.0, 0.8, "PART"), ("removes_0", 1.0, 1.0, "NOT-SEPARATED"), ("adds_one_half", 1.0, 1.5, "WORSE"),
                                        ("no_state_part", 0.0, 1.0, "NO-STATE-COST")):
        Vo, Vm = mk(state_own, keep)
        b = cell_block(RH, Vo, Vm, "c", inv, n_ep, n_boot)
        out[case] = {"want": want, "got": b["word"], "removed": b["removed"]["D"], "Q": b["Q_removed_minus_half_state"]["D"]}
        ok = ok and b["word"] == want
    return dict(out, PASS=bool(ok))


def assemble(controls, payload, extra):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": ok, "names": None, "modes": None, "own_split": None, "first_segment": None, "offsets_at_t0": None}
    res.update(extra)
    if ok and payload:
        res.update(payload)
    elif ok:
        res["controls_all_pass"] = False
    return res


# ============================================================================================ part 2
def maxabs(x, y):
    return float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))


def a1_check(here, banked):
    """the pair own minus fresh of this grid read against the pair AGE-K of the banked re-read of card ADD70 (headway shares and family cells, differences of the means to 1e-9)"""
    try:
        if not banked or banked.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the banked re-read of card ADD70 is missing or not admitted"}
        bp, bc = bank_pair(banked)
        if not (here["A"] == bp["A"] and here["B"] == bp["B"]):
            return {"PASS": False, "why": "the pair here names other runs than %s of the banked result" % BANK_NAME}
        bh = bp["headway"]
        hw_note = None
        if isinstance(bh, dict):
            d = [maxabs(here["headway"][c]["pooled"], bh[c]["pooled"]) for c in ("U", "C", "F")] + [maxabs(here["headway"][c]["per_seed"], bh[c]["per_seed"]) for c in ("U", "C", "F")]
        else:                                                                                    # the banked result reports no headway share for this pair (note N1 of card ADD74)
            d, hw_note = [], "not compared: the banked result holds no headway share for this pair (%s)" % str(bh)[:170]
        same = sorted(here["cells"]) == sorted(bc)
        d += [maxabs(here["cells"][c]["D"], bc[c]["D"]) for c in bc if c in here["cells"]] + [maxabs(here["cells"][c]["per_seed"], bc[c]["per_seed"]) for c in bc if c in here["cells"]]
        return {"PASS": bool(same and d and max(d) <= TOL_ID), "largest_difference_of_values": max(d) if d else None, "cells_compared": len(bc), "same_cell_names": same, "tolerance": TOL_ID,
                "headway": hw_note}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def p1_check(payload, first):
    """p1: the names and the cell blocks of part 1 of this read equal those of the first read of the card (1e-9)"""
    try:
        if not payload or not first or first.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "part 1 of this read or of the first read is missing or did not pass its controls"}
        d = []
        for name in payload["modes"]:
            for cell, b in payload["modes"][name].items():
                f = first["modes"][name][cell]
                d += [maxabs(b[k]["D"], f[k]["D"]) for k in ("state_own", "state_mode", "removed", "Q_removed_minus_half_state")]
        same = payload["names"] == first["names"]
        return {"PASS": bool(same and d and max(d) <= TOL_ID), "same_names": same, "largest_difference_of_values": max(d), "values_compared": len(d)}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def fam_names(pairs_out):
    """pairs_out[pair] = {"U": word, "fam": {family: word}} -> the names of the card"""
    names = {}
    for nm, rd in pairs_out.items():
        if rd["U"] is not None:
            names[nm + "_U"] = "%s-%s-U-%s" % (PRE, nm, rd["U"])
        for f, w in rd["fam"].items():
            names["%s_%s" % (nm, f)] = "%s-%s-%s-%s" % (PRE, nm, f, w)
    return names


def assemble_family(controls, payload, why=None):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    out = {"controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "pairs": None, "cells": None, "strategic": None, "levels": None, "counts": None,
           "file_controls_of_the_reader_of_card_add71": "not run: bound to the wrapper generation before this card; in their place the pod check of the wrapper (note N1) and c1, c2 of part 1"}
    if out["controls_all_pass"]:
        out.update(payload)
    else:
        out["missing_per_family"] = {f: (why or "a control of the family read did not pass; the cells of part 1 for this family stand by themselves") for f in ("LONGITUDINAL", "LATERAL", "TACTICAL")}
        out["missing_per_family"]["STRATEGIC"] = why or "a control of the family read did not pass; the route-goal read of card ADD66 is not reported for these runs"
    return out


def family_part(a, E, keys, bank, log):
    controls, payload = {}, None
    runs = [AD.SPREAD, AD.FRESH, OWN] + [v[0] for v in MODES.values()]
    pairs = dict(FAM_PAIRS)
    pairs[BANK_PAIR] = (OWN, AD.FRESH)
    G = R.read_grid(a, E, str(P_STEPS), keys, runs, AD.SPREAD, pairs, {}, log)
    controls["g_engine_controls_of_the_grid_read"] = {"PASS": bool(G["controls"].get("ALL_PASS") is True), "detail": G["controls"], "missing_or_incomplete": G.get("missing_or_incomplete")}
    if not controls["g_engine_controls_of_the_grid_read"]["PASS"]:
        return assemble_family(controls, None)
    pb = G["pairs"][BANK_PAIR]
    controls["a1_own_minus_fresh_equals_card_add70"] = a1_check({"A": pb["A"], "B": pb["B"], "headway": R.ucf(pb["headway"]), "cells": R.cells_of(pb["family"])}, bank)
    S, cells_s, eid_s = None, None, None
    if getattr(a, "w", None) and getattr(a, "tree", None):
        try:
            tab = dict(R.ALL_RUNS)
            for run_, _ in MODES.values():
                tab.setdefault(run_, RUN_ROW)
            tab.setdefault(OWN, RUN_ROW)
            R.ALL_RUNS = tab
            p_of = {"R0": 0, OWN: P_STEPS}
            p_of.update({v[0]: P_STEPS for v in MODES.values()})
            SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": bool(SR.get("stage_A_pass") and SR.get("stage_B_pass")), "detail": dict({k: v for k, v in SR["controls"].items() if k != "s3"},
                                                                                                                                      s3=SR["controls"]["s3"].get("pass"))}
        except SystemExit as ex:
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "SystemExit: %s" % ex}
        except Exception as ex:                                                               # noqa: BLE001
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
    else:
        controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "no --w and --tree given: the strategic read was not run"}
    if all(v.get("PASS") is True for v in controls.values()):
        pairs_out, cells_out, strat, rd = {}, {}, {}, {}
        H, cnt = E[0], G["counts"]
        qualified = bool(cnt["n_lead_windows"] >= H.MIN_WINDOWS and cnt["n_episodes_with_lead"] >= H.MIN_EPISODES)
        hw_text = "not reported: the headway instrument is not qualified on this grid (%d windows with a lead in %d episodes; its own bar is %d windows and %d episodes)" % (
            cnt["n_lead_windows"], cnt["n_episodes_with_lead"], H.MIN_WINDOWS, H.MIN_EPISODES)
        for nm, (A, B) in FAM_PAIRS.items():
            p = G["pairs"][nm]
            hw = R.ucf(p["headway"]) if qualified else hw_text
            cells_out[nm] = R.cells_of(p["family"])
            fam = {f: R.reading(cells_out[nm], f) for f in R.NAMED}
            pr = S.pair_read(cells_s, A, B, eid_s, ticks=(2,))
            vs = [c["verdict"] for c in pr["tick2"]["pos"]]
            strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")}
                                                            for arc, b in zip(S.ARCS, pr["tick2"]["pos"])], "reading": R.family_of(vs)}
            fam["STRATEGIC"] = strat[nm]["reading"]
            rd[nm] = {"fam": fam, "U": R.u_name(hw["U"]) if qualified else None}
            pairs_out[nm] = {"A": A, "B": B, "headway": hw}
        payload = {"names": fam_names(rd), "pairs": pairs_out, "cells": cells_out, "strategic": strat, "levels": G["levels"] if qualified else hw_text, "counts": G["counts"], "readings": rd,
                   "headway_qualified": qualified}
    return assemble_family(controls, payload)


# ============================================================================================ the read
def run(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p%d.json" % P_STEPS))
    if g30 is None:
        print("STOP: grid_p%d.json missing in %s" % (P_STEPS, a.grid_dir))
        return 3
    E = R.engine(a)
    H, Q3, RH = E
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    bank = R.jl(a.add70_json)
    try:                                                                                         # part 2 first: the grid read clears the start-point registry of note N4
        family = family_part(a, E, keys, bank, log)
    except Exception as ex:                                                                   # noqa: BLE001
        family = assemble_family({"family_read_ran": {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:400])}}, None, "the family read stopped: %s" % type(ex).__name__)
    runs = [AD.FRESH, AD.SPREAD, OWN] + [v[0] for v in MODES.values()]
    raws = {}
    for run_ in runs:
        for s in SEEDS:
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is not None:
                raws[(run_, s)] = raw
    controls = {}
    controls["b1_zero_offset_identity"] = M.ctl_b1(RH.comps, disp=M.displace_reg)
    controls["b2_one_metre_ahead"] = M.ctl_b2(RH.comps, disp=M.displace_reg)
    controls["b3_ninety_degrees"] = M.ctl_b3(RH.comps, disp=M.displace_reg)
    controls["n3_planted_start_offset"] = dict(N4.PLANT) if N4.PLANT else {"PASS": False, "why": "the planted check of note N4 did not run"}
    sp = {"OWN": AD.split_one(E, keys, raws, OWN, a.pull, a.n_boot, True)}
    for name, (run_, _) in MODES.items():
        sp[name] = AD.split_one(E, keys, raws, run_, a.pull, a.n_boot_control, True)
    b5_tool_form = {}
    run_of = dict([("OWN", OWN)] + [(n, v[0]) for n, v in MODES.items()])
    for name in sp:
        for k in ("b6", "b4"):
            controls["%s_%s" % (k, name)] = sp[name]["controls"][k]
        b5_tool_form[name] = sp[name]["controls"]["b5"]
        if not sp[name]["complete"]:
            controls["b5_%s" % name] = sp[name]["controls"]["b5"]
            continue
        o5, cr5, vh = {}, {}, {}
        for s in SEEDS:
            o, rep, cr, why = AD.tick_offsets(a.pull, run_of[name], s, keys)
            if o is not None:
                o5[s] = o
            if cr is not None:
                cr5[s] = cr
            vh[s] = handed_speed(a.pull, run_of[name], s, keys)
        controls["b5_%s" % name] = ctl_b5_handed({s: sp[name]["dumps"][(run_of[name], s)]["cl"] for s in SEEDS}, o5, cr5, vh)
    if not sp["OWN"]["complete"]:
        controls["b4_bank_own_split"] = {"PASS": False, "why": "; ".join(sp["OWN"]["why"]) or "inputs incomplete"}
    elif P_STEPS == 30:
        controls["b4_bank_own_split"] = dict(AD.bank_check(AD.sp_parts(sp["OWN"]), bank), against="pair AGE-K of the re-read of card ADD70")
    else:
        controls["b4_bank_own_split"] = dict(bank_split(AD.sp_parts(sp["OWN"]), R.jl(a.agedose_json) if a.agedose_json else None), against="parts at P60 of the read of card ADD72 Amendment 1")
    controls["h1_name_rule_planted"] = ctl_h1(RH, a.n_boot_control)
    complete = all(v["complete"] for v in sp.values())
    offs, logs, seg_dev, payload = {}, {}, {}, None
    if complete:
        for name, run_ in [("OWN", OWN)] + [(n, v[0]) for n, v in MODES.items()]:
            offs[name] = {}
            for s in SEEDS:
                o, rep, cr, why = AD.tick_offsets(a.pull, run_, s, keys)
                if o is not None:
                    offs[name][s] = o
        controls["c1_offsets_of_every_mode_equal_own"] = ctl_c1(offs.get("OWN", {}), {n: offs.get(n, {}) for n in MODES})
        for name, (run_, _) in MODES.items():
            logs[name] = {s: R.jl(os.path.join(R.tick_dir(a.pull, run_, s), "preroll_log_%s.json" % R.key_of(run_, s))) for s in SEEDS}
        controls["c2_preroll_logs_name_the_mode_and_the_groups"] = ctl_c2(logs)
        if all(s in offs.get("LOGGED", {}) for s in SEEDS):
            for s in SEEDS:
                f = sp["LOGGED"]["dumps"][(AD.FRESH, s)]["cl"]
                ag = sp["LOGGED"]["dumps"][(MODES["LOGGED"][0], s)]["cl"]
                dd = np.asarray(ag, np.float64)[:, :SEG, :] - moved(f, offs["LOGGED"][s])[:, :SEG, :]
                seg_dev[s] = float(np.hypot(dd[..., 0], dd[..., 1]).max())
        controls["c0_logged_first_segment_equals_the_moved_fresh_track"] = ctl_c0(seg_dev)
        if controls["c1_offsets_of_every_mode_equal_own"]["PASS"]:
            ref = sp["OWN"]["ref"]
            uniq, inv = RH.episode_units(ref["eid"])
            n_ep = len(uniq)
            modes, names, first, blocks = {}, {}, {}, []
            g = ref["g"]
            for name, (run_, _) in MODES.items():
                modes[name] = {}
                for tag, cell in list(CELLS.items()) + [(c, c) for c in BESIDE]:
                    key = cell.split(":", 1)[1]
                    modes[name][cell] = cell_block(RH, sp["OWN"]["V"], sp[name]["V"], key, inv, n_ep, a.n_boot)
                    blocks.append(modes[name][cell])
                    if tag in CELLS:
                        names["%s_%s" % (name, tag)] = "%s-%s-%s-%s" % (PRE, name, tag, modes[name][cell]["word"])
                Vo, Vm = {}, {}
                for s in SEEDS:
                    fo = sp["OWN"]["dumps"][(AD.FRESH, s)]["cl"]
                    Vo[("A", s)] = {"seg": seg_ade(sp["OWN"]["dumps"][(OWN, s)]["cl"], g)}
                    Vo[("D", s)] = {"seg": seg_ade(moved(fo, offs["OWN"][s]), g)}
                    Vm[("A", s)] = {"seg": seg_ade(sp[name]["dumps"][(run_, s)]["cl"], g)}
                    Vm[("D", s)] = {"seg": seg_ade(moved(fo, offs[name][s]), g)}
                first[name] = cell_block(RH, Vo, Vm, "seg", inv, n_ep, a.n_boot)
                blocks.append(first[name])
            controls["q1_identities_of_the_cell_blocks"] = ctl_q1(blocks)
            payload = {"names": names, "modes": modes, "own_split": sp["OWN"]["parts"], "first_segment": first, "offsets_at_t0": {"OWN": sp["OWN"]["offsets"]}}
    else:
        controls["inputs"] = {"PASS": False, "why": "; ".join(sum((v["why"] for v in sp.values()), []))[:900]}
    if getattr(a, "first_part1_json", None):
        controls["p1_part_1_equals_the_first_read"] = p1_check(payload, R.jl(a.first_part1_json))
    if getattr(a, "first_read_json", None):
        family["controls"]["p2_part_2_equals_the_first_read"] = p2_check(family, R.jl(a.first_read_json))
        if family["controls"]["p2_part_2_equals_the_first_read"]["PASS"] is not True:
            family = assemble_family(family["controls"], None)
    extra = {"counts": sp["OWN"].get("counts"), "common_windows_note_n2": {k: v.get("common_windows_note_n2") for k, v in sp.items()}, "family_read": family,
             "b5_in_the_form_of_the_split_tool_reported_not_judged": b5_tool_form}
    res = assemble(controls, payload, extra)
    res["card"], res["pre_roll_steps"] = CARD, P_STEPS
    print(TIER_LINE)
    print("== controls of part 1")
    for k, v in controls.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x != "PASS"}))[:500]))
    print("== controls of part 2 (family read)")
    for k, v in family["controls"].items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x not in ("PASS", "detail")}))[:400]))
    f3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["D"], c["interval"][0], c["interval"][1])       # noqa: E731
    if res["controls_all_pass"]:
        print("== names of part 1")
        for k, v in res["names"].items():
            print("   %-22s %s" % (k, v))
        print("== per mode and cell: state part own, state part of the mode, removed (own minus mode), Q = removed minus half of the own state part, removed share")
        for name in MODES:
            for cell, b in res["modes"][name].items():
                print("   %-13s %-28s own %s  mode %s  removed %s  Q %s  share %s" % (name, cell, f3(b["state_own"]), f3(b["state_mode"]), f3(b["removed"]), f3(b["Q_removed_minus_half_state"]),
                                                                                    "n/a" if b["removed_share"] is None else "%.2f" % b["removed_share"]))
            b = res["first_segment"][name]
            print("   %-13s %-28s own %s  mode %s  removed %s  share %s" % (name, "first segment track error m", f3(b["state_own"]), f3(b["state_mode"]), f3(b["removed"]),
                                                                         "n/a" if b["removed_share"] is None else "%.2f" % b["removed_share"]))
    else:
        print("== a control of part 1 did not pass: no name, no part, no value of part 1 is written")
    if family["controls_all_pass"]:
        print("== names of part 2")
        for k, v in family["names"].items():
            print("   %-34s %s" % (k, v))
        for nm in FAM_PAIRS:
            hw = family["pairs"][nm]["headway"]
            if not isinstance(hw, dict):
                print("   %-22s headway %s" % (nm, hw))
                hw = {c: {"pooled": float("nan"), "interval": [float("nan"), float("nan")]} for c in ("U", "C", "F")}
            print("   %-22s U %+.4f [%+.4f,%+.4f]  C %+.4f [%+.4f,%+.4f]  F %+.4f [%+.4f,%+.4f]" % (nm, hw["U"]["pooled"], hw["U"]["interval"][0], hw["U"]["interval"][1], hw["C"]["pooled"], hw["C"]["interval"][0],
                                                                                               hw["C"]["interval"][1], hw["F"]["pooled"], hw["F"]["interval"][0], hw["F"]["interval"][1]))
            for cell, c in family["cells"][nm].items():
                print("   %-22s %-30s %s %s" % (nm, cell, f3(c), c["verdict"]))
            for b in family["strategic"][nm]["pos_30_50_80m_tick_t0_plus_20"]:
                print("   %-22s STRATEGIC pos %sm %+.4f [%+.4f,%+.4f] %s n %s" % (nm, b["arc_m"], b["mean"], b["lo"], b["hi"], b["verdict"], b["n_per_seed"]))
    else:
        print("== a control of part 2 did not pass: no family name is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-HANDSRC-DECOMP rc 0 controls_all_pass %s family_controls_all_pass %s" % (res["controls_all_pass"], family["controls_all_pass"]))
    return 0


def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-112s %s" % (name, bool(ok)))
    cell = lambda D, lo, hi: {"D": D, "interval": [lo, hi]}                                    # noqa: E731
    pos, zero = cell(1.0, 0.5, 1.5), cell(0.0, -0.2, 0.2)
    chk("name rule: no state part -> NO-STATE-COST", name_rule(zero, pos, pos) == "NO-STATE-COST")
    chk("name rule: Q above 0 -> CARRIES", name_rule(pos, cell(0.8, 0.6, 1.0), cell(0.3, 0.1, 0.5)) == "CARRIES")
    chk("name rule: removed above 0, Q not -> PART", name_rule(pos, cell(0.2, 0.1, 0.3), cell(-0.3, -0.5, -0.1)) == "PART")
    chk("name rule: removed below 0 -> WORSE", name_rule(pos, cell(-0.5, -0.7, -0.3), cell(-1.0, -1.2, -0.8)) == "WORSE")
    chk("name rule: removed interval holds 0 -> NOT-SEPARATED", name_rule(pos, cell(0.05, -0.1, 0.2), cell(-0.45, -0.6, -0.3)) == "NOT-SEPARATED")
    h = ctl_h1(AD.MiniRH, 300)
    chk("h1 planted cases through the stand-in engine: " + ", ".join("%s=%s" % (k, v["got"]) for k, v in h.items() if k != "PASS"), h["PASS"])
    mk, inv, n_ep = planted()
    Vo, Vm = mk(1.0, 0.5)
    for s in SEEDS:                                                                              # windows without a value: different ones in each track
        for V, j in ((Vo, 3 + s), (Vm, 11 + s)):
            x = V[("A", s)]["c"].copy()
            x[j::17] = np.nan
            V[("A", s)]["c"] = x
    b = cell_block(AD.MiniRH, Vo, Vm, "c", inv, n_ep, 200)
    chk("q1 holds on tracks with different windows without a value (common windows %s)" % b["n_windows_per_seed"], ctl_q1([b])["PASS"] and max(b["n_windows_per_seed"]) < len(inv))
    b_bad = dict(b, removed=dict(b["removed"], D=b["removed"]["D"] + 1e-6))
    chk("q1 fails when removed is off by 1e-6", not ctl_q1([b_bad])["PASS"])
    o = {0: {"dx": np.ones(3), "dy": np.zeros(3), "dpsi": np.zeros(3)}, 1: {"dx": np.ones(3), "dy": np.zeros(3), "dpsi": np.zeros(3)}}
    o2 = {0: {"dx": np.ones(3) + 1e-9, "dy": np.zeros(3), "dpsi": np.zeros(3)}, 1: o[1]}
    chk("c1 passes on equal offsets and fails on a difference of 1e-9", ctl_c1(o, {"LOGGED": o})["PASS"] and not ctl_c1(o, {"LOGGED": o2})["PASS"])
    chk("c0 passes below the tolerance and fails above it or with a seed missing", ctl_c0({0: 1e-5, 1: 2e-5})["PASS"] and not ctl_c0({0: 1e-5, 1: 0.5})["PASS"] and not ctl_c0({0: 1e-5})["PASS"])

    def lg(mode, p, hd, v):
        return {"t0_state": mode, "t0_state_first_tick_max_abs_diff": {"model": {"states": 5, "max_abs": {"positions_m": p, "headings_rad": hd, "speed_ms": v}}}}
    good = {"LOGGED": {s: lg("logged", 1.0, 0.1, 2.0) for s in SEEDS}, "SPEED-LOGGED": {s: lg("speed_logged", 0.0, 0.0, 2.0) for s in SEEDS}, "PATH-LOGGED": {s: lg("path_logged", 1.0, 0.1, 0.0) for s in SEEDS}}
    chk("c2 passes when every log names its mode and its groups", ctl_c2(good)["PASS"])
    bad = dict(good)
    bad["SPEED-LOGGED"] = {s: lg("speed_logged", 0.5, 0.0, 2.0) for s in SEEDS}
    chk("c2 fails when the speed mode also moved a position", not ctl_c2(bad)["PASS"])
    bad2 = dict(good)
    bad2["PATH-LOGGED"] = {s: lg("logged", 1.0, 0.1, 0.0) for s in SEEDS}
    chk("c2 fails when a log names another mode", not ctl_c2(bad2)["PASS"])
    n5 = 200
    rng5 = np.random.default_rng(3)
    v_own5, v_log5 = np.full(n5, 2.0), rng5.uniform(5.0, 30.0, n5)
    off5 = {s: {"dx": np.zeros(n5), "dy": np.zeros(n5), "dpsi": np.zeros(n5), "v_own": v_own5, "dpos": np.zeros(n5)} for s in SEEDS}
    trk5 = np.zeros((n5, 30, 2))
    trk5[:, 0, 0] = v_log5 * 0.2                                                                 # the first step is as long as the handed (logged) speed makes it
    a5 = {s: trk5 for s in SEEDS}
    chk("b5 in the form of the split tool fails on a run that was handed another speed, and passes in the form of note N3",
        not M.ctl_b5(a5, off5, None)["PASS"] and ctl_b5_handed(a5, off5, None, {s: v_log5 for s in SEEDS})["PASS"])
    chk("b5 in the form of note N3 fails when the handed speed is missing or the first point is 9 m off", not ctl_b5_handed(a5, off5, None, {0: v_log5, 1: None})["PASS"]
        and not ctl_b5_handed({s: trk5 + np.array([9.0, 0.0]) for s in SEEDS}, off5, None, {s: v_log5 for s in SEEDS})["PASS"])
    fam5 = {"controls_all_pass": True, "names": {"a": "b"}, "pairs": {"P": {"headway": {c: {"pooled": 0.1} for c in ("U", "C", "F")}}}, "cells": {"P": {"x": {"D": 1.0}}}}
    fam6 = {"controls_all_pass": True, "names": {"a": "b"}, "pairs": {"P": {"headway": {c: {"pooled": 0.1} for c in ("U", "C", "F")}}}, "cells": {"P": {"x": {"D": 1.0 + 1e-6}}}}
    chk("p2 passes on an equal part 2 and fails on a cell that differs by 1e-6", p2_check(fam5, {"family_read": fam5})["PASS"] and not p2_check(fam5, {"family_read": fam6})["PASS"])
    tr = np.zeros((2, 30, 2))
    tr[:, :, 0] = np.arange(1, 31)
    chk("first-segment error of a track against itself is 0 and of a track 3 m aside is 3", float(seg_ade(tr, tr).max()) == 0.0 and abs(float(seg_ade(tr + np.array([0.0, 3.0]), tr).mean()) - 3.0) < 1e-12)
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}}, {})
    chk("part 1: a failed control leaves no name", a_["names"] is None and a_["controls_all_pass"] is False)
    f_ = assemble_family({"x": {"PASS": False}}, {"names": {"k": "v"}})
    chk("part 2: a failed control leaves no family name and says what is missing per family", f_["names"] is None and f_["controls_all_pass"] is False and len(f_["missing_per_family"]) == 4)
    f2 = assemble_family({"x": {"PASS": True}}, {"names": {"k": "v"}})
    chk("part 2: with every control passed the names are written", f2["names"] == {"k": "v"} and f2["controls_all_pass"] is True)
    fn = fam_names({"LOGGED-VS-OWN": {"U": "LOWER", "fam": {"LATERAL": "BETTER"}}})
    chk("family names have the form of the card", fn == {"LOGGED-VS-OWN_U": "HANDSRC10K-LOGGED-VS-OWN-U-LOWER", "LOGGED-VS-OWN_LATERAL": "HANDSRC10K-LOGGED-VS-OWN-LATERAL-BETTER"})
    configure(60)
    chk("configure(60): run names, card and prefix of card ADD74", OWN == "R0KP60Hb20" and MODES["SPEED-LOGGED"] == ("R0KP60SlHb20", "speed_logged") and FAM_PAIRS["PATH-LOGGED-VS-OWN"] == ("R0KP60PlHb20", "R0KP60Hb20")
        and PRE == "HANDSRC60" and RUN_ROW[2] == 60 and CARD.endswith("ADD74"))
    pr = {"x": {k: {"D": 1.0} for k in ("POSE", "STATE", "TOTAL")}}
    ad_ok = {"controls_all_pass": True, "parts_by_age": {"P60": {"x": {k: {"D": 1.0} for k in ("POSE", "STATE", "TOTAL")}}}}
    ad_bad = {"controls_all_pass": True, "parts_by_age": {"P60": {"x": {"POSE": {"D": 1.0}, "STATE": {"D": 1.0 + 1e-6}, "TOTAL": {"D": 1.0}}}}}
    chk("P = 60 bank of the own split passes on equal parts and fails on 1e-6", bank_split(pr, ad_ok)["PASS"] and not bank_split(pr, ad_bad)["PASS"] and not bank_split(pr, None)["PASS"])
    hw60 = {c: {"pooled": 0.2, "per_seed": [0.2, 0.2]} for c in ("U", "C", "F")}
    b60 = {"controls_all_pass": True, "beside": {"P60_pairs": {"AGE-K@P60": {"A": "a", "B": "b", "headway": hw60, "cells": {"x": {"D": 2.0, "per_seed": [2.0, 2.0]}}}}}}
    chk("a1 at P = 60 reads the pair AGE-K@P60 of the banked result", a1_check({"A": "a", "B": "b", "headway": hw60, "cells": {"x": {"D": 2.0, "per_seed": [2.0, 2.0]}}}, b60)["PASS"])
    configure(30)
    chk("configure(30) restores the names of card ADD73", OWN == "R0KP30Hb20" and PRE == "HANDSRC10K" and RUN_ROW[2] == 30 and CARD.endswith("ADD73") and MODES["LOGGED"][0] == "R0KP30LgHb20")
    b60s = {"controls_all_pass": True, "beside": {"P60_pairs": {"AGE-K@P60": {"A": "a", "B": "b", "headway": "not reported (a text)", "cells": {"x": {"D": 2.0, "per_seed": [2.0, 2.0]}}}}}}
    configure(60)
    r60 = a1_check({"A": "a", "B": "b", "headway": hw60, "cells": {"x": {"D": 2.0, "per_seed": [2.0, 2.0]}}}, b60s)
    r60b = a1_check({"A": "a", "B": "b", "headway": hw60, "cells": {"x": {"D": 2.0 + 1e-6, "per_seed": [2.0, 2.0]}}}, b60s)
    configure(30)
    chk("a1 with a banked pair that holds no headway share compares the family cells only, and still fails on a cell off by 1e-6", r60["PASS"] and r60["headway"] is not None and not r60b["PASS"])
    chk("no U name is written when the headway read is not reported", fam_names({"LOGGED-VS-OWN": {"U": None, "fam": {"LATERAL": "BETTER"}}}) == {"LOGGED-VS-OWN_LATERAL": "HANDSRC10K-LOGGED-VS-OWN-LATERAL-BETTER"})
    blk = {k: {"D": 1.0} for k in ("state_own", "state_mode", "removed", "Q_removed_minus_half_state")}
    pay = {"names": {"a": "b"}, "modes": {"LOGGED": {"c": blk}}}
    fst = {"controls_all_pass": True, "names": {"a": "b"}, "modes": {"LOGGED": {"c": dict(blk, removed={"D": 1.0 + 1e-6})}}}
    chk("p1 passes on an equal part 1 and fails on a value that differs by 1e-6", p1_check(pay, dict(pay, controls_all_pass=True))["PASS"] and not p1_check(pay, fst)["PASS"])
    hw = {c: {"pooled": 0.1, "per_seed": [0.1, 0.1]} for c in ("U", "C", "F")}
    here = {"A": "a", "B": "b", "headway": hw, "cells": {"x": {"D": 1.0, "per_seed": [1.0, 1.0]}}}
    bk = {"controls_all_pass": True, "pairs": {BANK_NAME: {"A": "a", "B": "b", "headway": hw}}, "cells": {BANK_NAME: {"x": {"D": 1.0, "per_seed": [1.0, 1.0]}}}}
    bk2 = {"controls_all_pass": True, "pairs": {BANK_NAME: {"A": "a", "B": "b", "headway": hw}}, "cells": {BANK_NAME: {"x": {"D": 1.0 + 1e-6, "per_seed": [1.0, 1.0]}}}}
    chk("a1 passes on equal values, fails on 1e-6 and on a result that is not admitted", a1_check(here, bk)["PASS"] and not a1_check(here, bk2)["PASS"] and not a1_check(here, dict(bk, controls_all_pass=False))["PASS"])
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [w for w in words if w in src])
    ok = all(res)
    print("ZZ SELFTEST-HANDSRC-DECOMP ok: %s (%d of %d)" % (ok, sum(res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "add70-json", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--w", default=None)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--p", type=int, default=30)
    ap.add_argument("--agedose-json", default=None)
    ap.add_argument("--first-read-json", default=None)
    ap.add_argument("--first-part1-json", default=None)
    a = ap.parse_args(argv)
    configure(a.p)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
