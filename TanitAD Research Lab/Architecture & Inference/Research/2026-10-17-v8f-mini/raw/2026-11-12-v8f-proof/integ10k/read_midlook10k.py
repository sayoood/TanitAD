#!/usr/bin/env python
"""read_midlook10k.py -- read of card PREREG_V8F_PROOF_ADD76 (MIDLOOK10K): the lead rule with a closing-speed input and a second look in the reader-fed form, at age and fresh.

Two settings, each read completely (grid read, strategic read, window arrays, logs) before the next starts, because the grid read of the note-N4 module clears its start-point registry:
  AGE    keys of grid_p30.json, pre-roll 30. Runs: RRh RQ50hKF5P30Hb20, RRc RQ50cKF5P30Hb20 (new); banked RR RQ50KF5P30Hb20, RLc RLcKF5P30Hb20, no rule R0KP30Hb20; spread R0Hb20; R0.
  FRESH  the pinned grid (--fresh-grid), pre-roll 0. Runs: RRh RQ50hKF5Hb20, RRc RQ50cKF5Hb20 (new); banked RR RQ50KF5Hb20, RLc RLcKF5Hb20, no rule R0KHb20; spread R0Hb20; R0.
Pairs A minus B per setting: READER-CLOSING (RRh, RR), LABEL-CLOSING (RRc, RR), RRH-VS-LABEL-FED (RRh, RLc), RRC-VS-LABEL-FED (RRc, RLc), RRH-VS-NO-RULE (RRh, no rule). Names and verdict names as in
section 3 of the card. The headway side (U, C, F shares) is reported only where the headway instrument is qualified by its own bar.
Engines: the grid read, strategic read and name words of read_longroll10k (under its notes N3 and N4) through read_handsrc_decomp; nothing of them is edited. The runs and
pairs of the two settings are registered in the run table of the engine module in memory (the strategic read looks up the route input of a run without a pre-roll there).
Controls (card section 4): g, s, r1, x1, m1, a1, i1, h1. A control that does not pass: no name and no value is written or printed.
Usage: python read_midlook10k.py --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --agerule-json FILE --w W --tree TREE --out OUT.json [--n-boot 2000]      |      --selftest
Last line: ZZ READ-MIDLOOK10K rc <rc> controls_all_pass <bool>
"""
import argparse
import io
import json
import os
import re
import sys
import types

import read_handsrc_decomp as HS
import read_closrc10k as CR

AD, M, R, N4, np = HS.AD, HS.M, HS.R, HS.N4, HS.np
CARD, PRE = "PREREG_V8F_PROOF_ADD76", "MIDLOOK10K"
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
WRAPPER_SHA = "8732329b4476347d"
TOL_A1, TOL_I1 = 1e-9, 1e-12
FORMS = ("RRh", "RRc", "RR", "RLc", "R0K")
NEW, ZERO = ("RRh", "RRc"), ("RR",)
SETTINGS = {
    "AGE": {"P": 30, "grid": "30", "spread": "R0Hb20", "runs": {"RRh": "RQ50hKF5P30Hb20", "RRc": "RQ50cKF5P30Hb20", "RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20", "R0K": "R0KP30Hb20"}},
    "FRESH": {"P": 0, "grid": "full", "spread": "R0Hb20", "runs": {"RRh": "RQ50hKF5Hb20", "RRc": "RQ50cKF5Hb20", "RR": "RQ50KF5Hb20", "RLc": "RLcKF5Hb20", "R0K": "R0KHb20"}},
}
FORM_OF = {"RRh": "RRh", "RRc": "RRc", "RR": "RR", "RLc": "RLc", "R0K": "R0"}
PAIRS = {"READER-CLOSING": ("RRh", "RR"), "LABEL-CLOSING": ("RRc", "RR"), "RRH-VS-LABEL-FED": ("RRh", "RLc"), "RRC-VS-LABEL-FED": ("RRc", "RLc"), "RRH-VS-NO-RULE": ("RRh", "R0K")}
INTERNAL = {"RR-VS-LABEL-FED": ("RR", "RLc"), "RR-VS-NO-RULE": ("RR", "R0K")}
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NOT_HIGHER = ("LOWER", "NOT-SEPARATED")


# ============================================================================================ names and verdicts
def cname(pair, setting, kind, word):
    return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)


def qualified(cnt, H):
    return bool(cnt["n_lead_windows"] >= H.MIN_WINDOWS and cnt["n_episodes_with_lead"] >= H.MIN_EPISODES)


def no_worse(fam):
    return bool(all(fam[f] != "WORSE" for f in FAMS))


def tri(x):
    """True / False / None (None = the word is not available)"""
    return x


def verdict_names(words):
    """words[setting][pair] = {"C": word or None, "U": word or None, "fam": {family: word}}. -> (names, why). A name that needs a word of a setting that is not qualified is not written; a condition
    that already fails decides its name without the missing word"""
    names, why = {}, {}

    def w(setting, pair, kind):
        return words[setting][pair][kind]

    def nh(word):
        return None if word is None else bool(word in NOT_HIGHER)
    conds = [("AGE C LOWER", None if w("AGE", "READER-CLOSING", "C") is None else w("AGE", "READER-CLOSING", "C") == "LOWER"), ("AGE U not HIGHER", nh(w("AGE", "READER-CLOSING", "U"))),
             ("AGE no family WORSE", no_worse(words["AGE"]["READER-CLOSING"]["fam"])), ("FRESH C not HIGHER", nh(w("FRESH", "READER-CLOSING", "C"))),
             ("FRESH U not HIGHER", nh(w("FRESH", "READER-CLOSING", "U"))), ("FRESH no family WORSE", no_worse(words["FRESH"]["READER-CLOSING"]["fam"]))]
    if any(v is False for _, v in conds):
        names["VERDICT_RRH_CHANGE"] = PRE + "-RRH-CHANGE-NOT-SUPPORTED"
    elif any(v is None for _, v in conds):
        why["VERDICT_RRH_CHANGE"] = "not written: the headway instrument is not qualified in a setting this verdict needs (%s)" % ", ".join(n for n, v in conds if v is None)
    else:
        names["VERDICT_RRH_CHANGE"] = PRE + "-RRH-CHANGE-SUPPORTED"
    c = w("AGE", "LABEL-CLOSING", "C")
    if c is None:
        why["VERDICT_LABEL_CLOSING"] = "not written: the headway instrument is not qualified at AGE (contact share of the pair LABEL-CLOSING not reported)"
    else:
        names["VERDICT_LABEL_CLOSING"] = PRE + ("-LABEL-CLOSING-LOWERS-CONTACT" if c == "LOWER" else "-LABEL-CLOSING-DOES-NOT-LOWER-CONTACT")
    c, u = w("AGE", "RRH-VS-LABEL-FED", "C"), w("AGE", "RRH-VS-LABEL-FED", "U")
    if c is None or u is None:
        why["VERDICT_RRH_VS_LABEL_FED"] = "not written: the headway instrument is not qualified at AGE (C and U of the pair RRH-VS-LABEL-FED not reported)"
    elif c == "HIGHER" or u == "HIGHER":
        names["VERDICT_RRH_VS_LABEL_FED"] = PRE + "-RRH-GAP-TO-LABEL-FED"
    elif c == "NOT-SEPARATED" and u in NOT_HIGHER:
        names["VERDICT_RRH_VS_LABEL_FED"] = PRE + "-RRH-REACHES-LABEL-FED"
    else:
        why["VERDICT_RRH_VS_LABEL_FED"] = "no name: neither condition holds (C %s, U %s)" % (c, u)
    return names, why


def pair_names(setting, pair, wd):
    out = {}
    if wd["C"] is not None:
        out["%s_%s_C" % (pair, setting)] = cname(pair, setting, "C", wd["C"])
    if wd["U"] is not None:
        out["%s_%s_U" % (pair, setting)] = cname(pair, setting, "U", wd["U"])
    for f in FAMS:
        out["%s_%s_%s" % (pair, setting, f)] = cname(pair, setting, f, wd["fam"][f])
    return out


def all_names(words, quals):
    names, why = {}, {}
    for st in SETTINGS:
        if not quals[st]:
            why["HEADWAY_" + st] = "U, C and F of the setting %s are not reported: the headway instrument is not qualified on its grid" % st
        for pair in PAIRS:
            names.update(pair_names(st, pair, words[st][pair]))
    vn, vw = verdict_names(words)
    names.update(vn)
    why.update(vw)
    return names, why


# ============================================================================================ controls
def run_text_ok(p, form, preroll):
    return bool(p and p["form"] == form and p["replan_steps"] == 10 and p["rule_every"] == 5 and p["readout"] == "causal_g" and p["kappa"] == "0.2" and p["preroll"] == preroll
                and p["wrapper"] == WRAPPER_SHA and p["rc0"])


def ctl_r1(texts, n_files, n_clips, preroll):
    out, ok = {}, bool(texts)
    for f in NEW:
        for s in SEEDS:
            p = CR.parse_run_text(texts.get((f, s)))
            good = run_text_ok(p, f, preroll) and n_files.get((f, s)) == n_clips
            out["%s:seed%d" % (f, s)] = {"ok": bool(good), "parsed": p, "dump_files": n_files.get((f, s)), "clips_of_the_grid": n_clips}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def log_fed(z):
    """a rule log or a later-look log -> closing_fed record plus, where the log holds c_read, the largest |c_src - c_read| over the present rows"""
    r = CR.closing_fed(z)
    try:
        cr = np.asarray(z["c_read"], np.float64)
        cs = np.asarray(z["c_src"], np.float64)
        pr = np.asarray(z["present"], bool)
        r["c_read_diff_present"] = float(np.max(np.abs(cs[pr] - cr[pr]))) if pr.any() and np.isfinite(cr[pr]).all() else None
        r["c_read_all_nan"] = bool(np.all(np.isnan(cr)))
    except Exception:                                                                         # noqa: BLE001
        r["c_read_diff_present"], r["c_read_all_nan"] = None, None
    return r


def ctl_x1(rule, mid):
    """rule, mid: {(form, seed): log_fed record or None} for RRh, RRc, RR. RRh and RRc feed a closing that is not 0 on at least one present row in both logs; RRh feeds exactly its logged reader closing
    on the present rows of the later-look log (difference 0); the banked RR run feeds 0 on every row of both logs"""
    out, ok = {}, bool(rule) and bool(mid)
    for f in FORMS[:3]:
        for s in SEEDS:
            res = {}
            good = True
            for tag, lg in (("rule_log", rule), ("mid_log", mid)):
                r = lg.get((f, s))
                if not r:
                    res[tag], good = {"ok": False, "why": "no log"}, False
                    continue
                if f in ZERO:
                    g1 = r["rows"] > 0 and r["present_nonzero"] == 0 and r["max_abs_closing"] == 0.0 and r["closing_finite_where_present"]
                else:
                    g1 = r["present"] > 0 and r["present_nonzero"] > 0 and r["closing_finite_where_present"]
                if tag == "rule_log":
                    g1 = g1 and r["form"] == f
                if f == "RRh" and tag == "mid_log":
                    g1 = g1 and r.get("c_read_diff_present") == 0.0
                res[tag] = dict(r, ok=bool(g1))
                good = good and g1
            out["%s:seed%d" % (f, s)] = dict(res, ok=bool(good))
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_m1(rows):
    """rows: {(form, seed): later-look row count}; the new runs equal the banked RR run of the same seed (same grid, same pre-roll) and have rows"""
    out, ok = {}, bool(rows)
    for s in SEEDS:
        base = rows.get(("RR", s))
        for f in NEW:
            n = rows.get((f, s))
            good = bool(n and base and n == base)
            out["%s:seed%d" % (f, s)] = {"ok": good, "rows": n, "rows_banked_RR": base}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_i1(trips):
    """trips: list of {"ab": [per seed], "ac": [...], "cb": [...]}: ab = ac + cb per seed to 1e-12"""
    w = 0.0
    for t in trips:
        for x, y, z in zip(t["ab"], t["ac"], t["cb"]):
            w = max(w, abs(x - (y + z)))
    return {"PASS": bool(trips and w <= TOL_I1), "largest_abs_dev": w, "n_triples": len(trips), "tolerance": TOL_I1}


def _num(x, y):
    return float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))


def banked_pair(bank, A, B):
    """the pair A minus B (run names) in the banked read of card ADD71: (record, cells or None, where); None when the file does not hold it"""
    for nm, p in (bank.get("pairs") or {}).items():
        if isinstance(p, dict) and p.get("A") == A and p.get("B") == B and isinstance(p.get("headway"), dict):
            return p, (bank.get("cells") or {}).get(nm), "pairs/" + nm
    for sec, blk in (bank.get("beside") or {}).items():
        if isinstance(blk, dict):
            for nm, p in blk.items():
                if isinstance(p, dict) and p.get("A") == A and p.get("B") == B and isinstance(p.get("headway"), dict):
                    return p, p.get("cells"), "beside/%s/%s" % (sec, nm)
    return None


def ctl_a1(here_pair, here_levels, bank, A, B):
    """here_pair: {"headway": ucf, "cells": cells_of} of the pair A minus B read here; here_levels: {run: {U,C,F: per seed}}. The pair is compared with the banked pair where the file holds it (headway and
    family cells, 1e-9); where it does not, the levels of the two runs are compared with the banked levels (beside/levels_P30)"""
    try:
        if not bank or bank.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the banked read of card ADD71 is missing or not admitted"}
        f = banked_pair(bank, A, B)
        d = []
        if f is not None:
            p, cells, where = f
            d += [_num(here_pair["headway"][c]["pooled"], p["headway"][c]["pooled"]) for c in ("U", "C", "F")] + [_num(here_pair["headway"][c]["per_seed"], p["headway"][c]["per_seed"]) for c in ("U", "C", "F")]
            n_cells = 0
            if isinstance(cells, dict):
                same = sorted(here_pair["cells"]) == sorted(cells)
                d += [_num(here_pair["cells"][c]["D"], cells[c]["D"]) for c in cells if c in here_pair["cells"]] + [_num(here_pair["cells"][c]["per_seed"], cells[c]["per_seed"]) for c in cells if c in here_pair["cells"]]
                n_cells = len(cells)
            else:
                same = True
            return {"PASS": bool(same and d and max(d) <= TOL_A1), "route": "pair", "where": where, "largest_difference": max(d) if d else None, "cells_compared": n_cells, "tolerance": TOL_A1}
        lv = (bank.get("beside") or {}).get("levels_P30")
        if not isinstance(lv, dict):
            return {"PASS": False, "why": "the banked file holds neither the pair nor beside/levels_P30"}
        for r in (A, B):
            if r not in lv:
                return {"PASS": False, "why": "run %s not in the banked levels" % r}
            d += [_num(here_levels[r][c], lv[r][c]) for c in ("U", "C", "F")]
        return {"PASS": bool(d and max(d) <= TOL_A1), "route": "levels", "largest_difference": max(d), "runs": [A, B], "tolerance": TOL_A1}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def ctl_h1(RH, n_boot=300, n_ep=40, per=10, seed=11, sd=0.3):
    """planted name rule: a pair with a known sign reads the expected name through the engine and the name builder; a pair with seeds of opposite sign reads NOT-SEPARATED"""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    inv = np.repeat(np.arange(n_ep), per)
    out, ok = {}, True
    for case, eff, want in (("lower", (-1.0, -1.0), "LOWER"), ("higher", (1.0, 1.0), "HIGHER"), ("opposite_seeds", (-1.0, 1.0), "NOT-SEPARATED"), ("no_effect", (0.0, 0.0), "NOT-SEPARATED")):
        ys = [rng.normal(0.0, sd, n) for _ in SEEDS]
        xs = [y + e + rng.normal(0.0, sd, n) for y, e in zip(ys, eff)]
        got = R.u_name(HS.pair(RH, xs, ys, inv, n_ep, n_boot))
        nm = cname("READER-CLOSING", "AGE", "C", got)
        out[case] = {"want": want, "got": got, "name": nm}
        ok = ok and got == want and nm == cname("READER-CLOSING", "AGE", "C", want)
    return dict(out, PASS=bool(ok))


def assemble(controls, payload):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "why_not_written": None, "settings": None}
    if res["controls_all_pass"]:
        res.update(payload)
    return res


# ============================================================================================ one setting
def register_runs(cfg):
    """the run table of the engine module (in memory only): form, readout, P, route input, grid, pre-roll exec, rule interval, kappa"""
    for f, r in cfg["runs"].items():
        R.ALL_RUNS.setdefault(r, (FORM_OF[f], "K", cfg["P"], "deadreckon", cfg["grid"], "plan", None if f == "R0K" else 5, 0.2))
    R.ALL_RUNS.setdefault(cfg["spread"], ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2))


def run_setting(a, E, st, keys, bank, ctx_cache, log):
    """-> (controls with the setting as suffix, part or None). The window arrays are loaded after the grid read, so the registry of the start points is the one of this setting"""
    cfg = SETTINGS[st]
    H, Q3, RH = E
    runs = cfg["runs"]
    J = H.load_join()
    n_clips = int(len(np.unique(keys[:, 0])))
    register_runs(cfg)
    ctl = {}
    run_list = [cfg["spread"]] + [runs[f] for f in FORMS]
    pairs = {nm: (runs[x], runs[y]) for nm, (x, y) in dict(PAIRS, **INTERNAL).items()}
    G = R.read_grid(a, E, cfg["grid"], keys, run_list, cfg["spread"], pairs, {}, log)
    ctl["g_" + st] = {"PASS": bool(G["controls"].get("ALL_PASS") is True), "detail": G["controls"], "missing_or_incomplete": G.get("missing_or_incomplete")}
    if not ctl["g_" + st]["PASS"]:
        return ctl, None
    if st == "AGE":
        rr = G["pairs"]["RR-VS-NO-RULE"]
        ctl["a1_" + st] = ctl_a1({"headway": R.ucf(rr["headway"]), "cells": R.cells_of(rr["family"])}, G["levels"], bank, runs["RR"], runs["R0K"])
    S = cells_s = eid_s = None
    try:
        p_of = {"R0": 0}
        p_of.update({r: cfg["P"] for r in runs.values()})
        SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, ctx_cache)
        ctl["s_" + st] = {"PASS": bool(SR.get("stage_A_pass") and SR.get("stage_B_pass")), "detail": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass"))}
    except SystemExit as ex:
        ctl["s_" + st] = {"PASS": False, "why": "SystemExit: %s" % ex}
    except Exception as ex:                                                                   # noqa: BLE001
        ctl["s_" + st] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
    texts, n_files, rule, mid, rows, Cw, lev = {}, {}, {}, {}, {}, {}, {}
    ref = None
    for f in FORMS:
        for s in SEEDS:
            run_ = runs[f]
            key = R.key_of(run_, s)
            td = R.tick_dir(a.pull, run_, s)
            if f in NEW:
                texts[(f, s)] = R.txt(os.path.join(td, "run_%s.txt" % key))
            if f in FORMS[:3]:
                z = R.npz_load(os.path.join(td, "rule_log_%s.npz" % key))
                rule[(f, s)] = log_fed(z) if z is not None else None
                zm = R.npz_load(os.path.join(td, "rule_log_mid_%s.npz" % key))
                mid[(f, s)] = log_fed(zm) if zm is not None else None
                rows[(f, s)] = int(len(np.asarray(zm["ci"]))) if zm is not None else None
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is None:
                continue
            if f in NEW:
                n_files[(f, s)] = int(raw["n_files"])
            idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
            if rep["missing"]:
                continue
            D = R.cut(raw, idx)
            ref = D if ref is None else ref
            L = H.label_windows(J, D["clip_index"], D["ws"])
            Cw[(f, s)] = np.asarray(H.window_cells(D["g"], D["cl"], L)["C"], np.float64)
            lev[(f, s)] = float(H.level(Cw[(f, s)]))
    ctl["r1_" + st] = ctl_r1(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0)
    ctl["x1_" + st] = ctl_x1(rule, mid)
    ctl["m1_" + st] = ctl_m1(rows)
    have = all((f, s) in Cw for f in FORMS for s in SEEDS)
    dl = max([abs(lev[(f, s)] - float(G["levels"][runs[f]]["C"][i])) for f in FORMS for i, s in enumerate(SEEDS)]) if have else float("inf")
    trips = []
    if have:
        for (a_, b_, c_) in (("RRh", "RLc", "RR"), ("RRc", "RLc", "RR")):                      # (a minus b) = (a minus c) + (c minus b)
            trips.append({"ab": [lev[(a_, s)] - lev[(b_, s)] for s in SEEDS], "ac": [lev[(a_, s)] - lev[(c_, s)] for s in SEEDS], "cb": [lev[(c_, s)] - lev[(b_, s)] for s in SEEDS]})
        pc = lambda p: [float(x) for x in R.ucf(G["pairs"][p]["headway"])["C"]["per_seed"]]    # noqa: E731
        try:                                                                                  # the same identity on the per-seed values of the headway pairs: reported, not judged (the card fixes i1 on the levels)
            pair_route = ctl_i1([{"ab": pc("RRH-VS-LABEL-FED"), "ac": pc("READER-CLOSING"), "cb": pc("RR-VS-LABEL-FED")}])["largest_abs_dev"]
        except Exception as ex:                                                               # noqa: BLE001
            pair_route = "%s: %s" % (type(ex).__name__, str(ex)[:120])
    else:
        pair_route = None
    i1 = ctl_i1(trips)
    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl, pair_route_reported_not_judged=pair_route)
    if S is None or cells_s is None or not all(v.get("PASS") is True for v in ctl.values()):
        return ctl, None
    cnt = G["counts"]
    q = qualified(cnt, H)
    hw_text = "not reported: the headway instrument is not qualified on this grid (%d windows with a lead in %d episodes; its own bar is %d windows and %d episodes)" % (
        cnt["n_lead_windows"], cnt["n_episodes_with_lead"], H.MIN_WINDOWS, H.MIN_EPISODES)
    pairs_out, cells_out, strat, words = {}, {}, {}, {}
    for nm, (x, y) in PAIRS.items():
        A, B = runs[x], runs[y]
        p = G["pairs"][nm]
        hw = R.ucf(p["headway"]) if q else hw_text
        cells_out[nm] = R.cells_of(p["family"])
        fam = {f_: R.reading(cells_out[nm], f_) for f_ in R.NAMED}
        pr = S.pair_read(cells_s, A, B, eid_s, ticks=(2,))
        vs = [c["verdict"] for c in pr["tick2"]["pos"]]
        strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")}
                                                        for arc, b in zip(S.ARCS, pr["tick2"]["pos"])], "reading": R.family_of(vs)}
        fam["STRATEGIC"] = strat[nm]["reading"]
        words[nm] = {"C": R.u_name(hw["C"]) if q else None, "U": R.u_name(hw["U"]) if q else None, "fam": fam}
        pairs_out[nm] = {"A": A, "B": B, "headway": hw, "words": words[nm]}
    part = {"qualified": q, "counts": cnt, "pairs": pairs_out, "cells": cells_out, "strategic": strat, "words": words,
            "levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text)}
    return ctl, part


# ============================================================================================ the read
def run(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    gf = R.jl(a.fresh_grid)
    if g30 is None or gf is None:
        print("STOP: grid_p30.json in %s or the pinned grid %s is missing" % (a.grid_dir, a.fresh_grid))
        return 3
    E = R.engine(a)
    H = E[0]
    J = H.load_join()
    names_ = [str(c) for c in J["clip"]]
    keys = {"AGE": R.grid_keys(g30, names_), "FRESH": R.grid_keys(gf, names_)}
    bank = R.jl(a.agerule_json)
    controls, parts, cache = {}, {}, {}
    for st in SETTINGS:                                                                        # one setting completely, then the next
        try:
            c, p = run_setting(a, E, st, keys[st], bank, cache, log)
        except Exception as ex:                                                               # noqa: BLE001
            c, p = {"setting_ran_" + st: {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:400])}}, None
        controls.update(c)
        parts[st] = p
    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)
    payload = None
    if all(v.get("PASS") is True for v in controls.values()) and all(parts.values()):
        words = {st: parts[st]["words"] for st in SETTINGS}
        names, why = all_names(words, {st: parts[st]["qualified"] for st in SETTINGS})
        payload = {"names": names, "why_not_written": why, "settings": parts}
    res = assemble(controls, payload)
    print(TIER_LINE)
    print("== controls")
    for k, v in controls.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x not in ("PASS", "detail", "runs")}))[:600]))
    if res["controls_all_pass"]:
        print("== names")
        for k, v in res["names"].items():
            print("   %-44s %s" % (k, v))
        for k, v in res["why_not_written"].items():
            print("   %-44s %s" % (k, v))
        h3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["pooled"], c["interval"][0], c["interval"][1])   # noqa: E731
        f3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["D"], c["interval"][0], c["interval"][1])       # noqa: E731
        for st, p in res["settings"].items():
            print("== setting %s: windows %d, with a lead %d in %d episodes, headway qualified %s" % (st, p["counts"]["n_windows"], p["counts"]["n_lead_windows"], p["counts"]["n_episodes_with_lead"], p["qualified"]))
            if p["qualified"]:
                for r, lv in p["levels"].items():
                    print("   %-18s U %s  C %s  F %s" % (r, ["%.4f" % x for x in lv["U"]], ["%.4f" % x for x in lv["C"]], ["%.4f" % x for x in lv["F"]]))
            for nm, po in p["pairs"].items():
                wd = po["words"]
                if p["qualified"]:
                    print("   %-18s U %s %s  C %s %s  F %s" % (nm, h3(po["headway"]["U"]), wd["U"], h3(po["headway"]["C"]), wd["C"], h3(po["headway"]["F"])))
                print("   %-18s families %s" % (nm, json.dumps(wd["fam"])))
                for cell, c in p["cells"][nm].items():
                    print("   %-18s %-30s %s %s" % (nm, cell, f3(c), c["verdict"]))
                for b in p["strategic"][nm]["pos_30_50_80m_tick_t0_plus_20"]:
                    print("   %-18s STRATEGIC pos %sm %+.4f [%+.4f,%+.4f] %s n %s" % (nm, b["arc_m"], b["mean"], b["lo"], b["hi"], b["verdict"], b["n_per_seed"]))
    else:
        print("== a control did not pass: no name and no value is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-MIDLOOK10K rc 0 controls_all_pass %s" % res["controls_all_pass"])
    return 0


# ============================================================================================ selftest
def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-116s %s" % (name, bool(ok)))
    h = ctl_h1(AD.MiniRH, 300)
    chk("h1 planted name rule through the stand-in engine: %s" % [(k, v["got"]) for k, v in h.items() if k != "PASS"], h["PASS"])
    t = ("arm I10RQ50hKF5P30Hb20_s0 form RRh replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 ckpt sha256 ab "
         "wrapper sha256 8732329b4476347d4f2e0a8a6d1c9b77 started 2026-11-20T12:00:00Z\nrc 0 finished 2026-11-20T12:04:00Z\n")
    p = CR.parse_run_text(t)
    chk("run text parsed and accepted: form, cycle 10, rule every 5, readout, bound, pre-roll 30, wrapper, return code 0", run_text_ok(p, "RRh", 30) and not run_text_ok(p, "RRc", 30) and not run_text_ok(p, "RRh", 0))
    chk("run text with the rule every 10, another wrapper hash, a return code 1 or another readout is not accepted",
        not run_text_ok(CR.parse_run_text(t.replace("rule every 5", "rule every 10")), "RRh", 30) and not run_text_ok(CR.parse_run_text(t.replace("8732329b4476347d", "92a8034cc0f7b0d2")), "RRh", 30)
        and not run_text_ok(CR.parse_run_text(t.replace("rc 0 finished", "rc 1 finished")), "RRh", 30) and not run_text_ok(CR.parse_run_text(t.replace("causal_g", "pinned")), "RRh", 30))
    texts = {(f, s): t.replace("form RRh", "form " + f) for f in NEW for s in SEEDS}
    nf = {(f, s): 141 for f in NEW for s in SEEDS}
    t0 = {k: v.replace("preroll 30", "preroll 0") for k, v in texts.items()}
    chk("r1 passes on four good run texts (AGE and, with pre-roll 0, FRESH) and fails on a dump with a clip missing or a FRESH text with pre-roll 30",
        ctl_r1(texts, nf, 141, 30)["PASS"] and ctl_r1(t0, nf, 141, 0)["PASS"] and not ctl_r1(texts, {**nf, ("RRc", 1): 140}, 141, 30)["PASS"] and not ctl_r1(texts, nf, 141, 0)["PASS"])

    def zl(form, cs, cr=None, pres=(True, True, False, True)):
        z = {"present": np.array(pres), "c_src": np.asarray(cs, np.float64), "form": np.asarray(form)}
        if cr is not None:
            z["c_read"] = np.asarray(cr, np.float64)
        return log_fed(z)
    nz, zr = [0.0, -1.5, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0]
    rule = {(f, s): zl(f, zr if f == "RR" else nz) for f in FORMS[:3] for s in SEEDS}
    mid = {(f, s): zl(f, zr if f == "RR" else nz, zr if f == "RR" else (nz if f == "RRh" else [np.nan] * 4)) for f in FORMS[:3] for s in SEEDS}
    chk("x1 passes: RRh and RRc feed a closing in both logs, RRh equals its c_read with difference 0, RR feeds 0 on every row of both logs", ctl_x1(rule, mid)["PASS"])
    bad1 = {**mid, ("RRh", 1): zl("RRh", nz, [0.0, -1.5, 0.0, 0.2999])}
    bad2 = {**rule, ("RR", 0): zl("RR", [0.0, 0.0, 0.0, 0.1])}
    bad3 = {**mid, ("RRc", 0): zl("RRc", zr, [np.nan] * 4)}
    chk("x1 fails when c_src differs from c_read for RRh, when RR was fed a closing in the rule log, when RRc fed none at a later look",
        not ctl_x1(rule, bad1)["PASS"] and not ctl_x1(bad2, mid)["PASS"] and not ctl_x1(rule, bad3)["PASS"])
    chk("x1 fails when RR fed a non-zero closing on an absent row of the later-look log and when a log is missing",
        not ctl_x1(rule, {**mid, ("RR", 1): zl("RR", [0.0, 0.0, 2.0, 0.0], zr)})["PASS"] and not ctl_x1(rule, {**mid, ("RRh", 0): None})["PASS"])
    rw = {(f, s): 900 for f in FORMS[:3] for s in SEEDS}
    chk("m1 passes on equal later-look row counts and fails on a different count or an empty log", ctl_m1(rw)["PASS"] and not ctl_m1({**rw, ("RRh", 0): 899})["PASS"] and not ctl_m1({**rw, ("RR", 1): None})["PASS"])
    tr = {"ab": [0.10, 0.07], "ac": [0.04, 0.05], "cb": [0.06, 0.02]}
    chk("i1 holds on a triple that adds up and fails when off by 1e-6", ctl_i1([tr])["PASS"] and not ctl_i1([dict(tr, ab=[0.10 + 1e-6, 0.07])])["PASS"])

    def wd(c, u, fam=None):
        return {"C": c, "U": u, "fam": dict({f: "NOT-SEPARATED" for f in FAMS}, **(fam or {}))}

    def W(age, fresh, rl_age=None, lab=None):
        a_ = {"READER-CLOSING": age, "LABEL-CLOSING": lab or wd("LOWER", "NOT-SEPARATED"), "RRH-VS-LABEL-FED": rl_age or wd("NOT-SEPARATED", "NOT-SEPARATED")}
        return {"AGE": a_, "FRESH": {"READER-CLOSING": fresh, "LABEL-CLOSING": wd("NOT-SEPARATED", "NOT-SEPARATED"), "RRH-VS-LABEL-FED": wd("NOT-SEPARATED", "NOT-SEPARATED")}}
    n1, y1 = verdict_names(W(wd("LOWER", "LOWER"), wd("NOT-SEPARATED", "NOT-SEPARATED", {"LATERAL": "MIXED"})))
    chk("verdict: RRH-CHANGE-SUPPORTED (AGE C LOWER, FRESH C not HIGHER, MIXED is not WORSE); REACHES-LABEL-FED; LABEL-CLOSING-LOWERS-CONTACT",
        n1["VERDICT_RRH_CHANGE"] == "MIDLOOK10K-RRH-CHANGE-SUPPORTED" and n1["VERDICT_RRH_VS_LABEL_FED"] == "MIDLOOK10K-RRH-REACHES-LABEL-FED" and n1["VERDICT_LABEL_CLOSING"] == "MIDLOOK10K-LABEL-CLOSING-LOWERS-CONTACT")
    chk("verdict: NOT-SUPPORTED when AGE C is not LOWER, when FRESH C is HIGHER, when U is HIGHER, when a family is WORSE at AGE or at FRESH",
        all(verdict_names(w_)[0]["VERDICT_RRH_CHANGE"] == "MIDLOOK10K-RRH-CHANGE-NOT-SUPPORTED" for w_ in (
            W(wd("NOT-SEPARATED", "LOWER"), wd("LOWER", "LOWER")), W(wd("LOWER", "LOWER"), wd("HIGHER", "LOWER")), W(wd("LOWER", "HIGHER"), wd("LOWER", "LOWER")),
            W(wd("LOWER", "LOWER", {"TACTICAL": "WORSE"}), wd("LOWER", "LOWER")), W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER", {"STRATEGIC": "WORSE"})))))
    chk("verdict: LABEL-CLOSING-DOES-NOT-LOWER-CONTACT, RRH-GAP-TO-LABEL-FED on C or U HIGHER, and no third name when neither condition holds",
        verdict_names(W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), lab=wd("NOT-SEPARATED", "LOWER")))[0]["VERDICT_LABEL_CLOSING"] == "MIDLOOK10K-LABEL-CLOSING-DOES-NOT-LOWER-CONTACT"
        and verdict_names(W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), rl_age=wd("HIGHER", "LOWER")))[0]["VERDICT_RRH_VS_LABEL_FED"] == "MIDLOOK10K-RRH-GAP-TO-LABEL-FED"
        and verdict_names(W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), rl_age=wd("NOT-SEPARATED", "HIGHER")))[0]["VERDICT_RRH_VS_LABEL_FED"] == "MIDLOOK10K-RRH-GAP-TO-LABEL-FED"
        and "VERDICT_RRH_VS_LABEL_FED" not in verdict_names(W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), rl_age=wd("LOWER", "NOT-SEPARATED")))[0])
    un = W(wd(None, None), wd("LOWER", "LOWER"), rl_age=wd(None, None), lab=wd(None, None))
    n3, y3 = verdict_names(un)
    chk("verdict: a setting that is not qualified writes no name that needs its C or U and says why; a failing family condition still decides RRH-CHANGE",
        "VERDICT_RRH_CHANGE" not in n3 and "VERDICT_LABEL_CLOSING" not in n3 and "VERDICT_RRH_VS_LABEL_FED" not in n3 and len(y3) == 3
        and verdict_names(W(wd(None, None, {"LATERAL": "WORSE"}), wd("LOWER", "LOWER")))[0]["VERDICT_RRH_CHANGE"] == "MIDLOOK10K-RRH-CHANGE-NOT-SUPPORTED")
    full = {st: {pn: wd("LOWER", "NOT-SEPARATED") for pn in PAIRS} for st in SETTINGS}
    nm, why = all_names(full, {"AGE": True, "FRESH": True})
    chk("names: pattern MIDLOOK10K-<PAIR>-<SETTING>-C-<word>, U and the four families per pair and setting",
        nm["READER-CLOSING_AGE_C"] == "MIDLOOK10K-READER-CLOSING-AGE-C-LOWER" and nm["RRC-VS-LABEL-FED_FRESH_STRATEGIC"] == "MIDLOOK10K-RRC-VS-LABEL-FED-FRESH-STRATEGIC-NOT-SEPARATED"
        and len([k for k in nm if k.endswith("_U")]) == 10 and len([k for k in nm if k.endswith("_LATERAL")]) == 10)
    nm2, why2 = all_names({st: {pn: wd(None, None) for pn in PAIRS} for st in SETTINGS}, {"AGE": False, "FRESH": False})
    chk("names: with no qualified setting no C or U name is written and the output says why", not [k for k in nm2 if k.endswith("_C") or k.endswith("_U")] and "HEADWAY_AGE" in why2 and "HEADWAY_FRESH" in why2)
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}})
    chk("a failed control leaves no name", a_["names"] is None and a_["controls_all_pass"] is False)
    hw = lambda d: {c: {"pooled": d, "per_seed": [d, d]} for c in ("U", "C", "F")}                # noqa: E731
    lv = {"RQ": {c: [0.1, 0.2] for c in ("U", "C", "F")}, "R0": {c: [0.3, 0.4] for c in ("U", "C", "F")}}
    bank = {"controls_all_pass": True, "beside": {"levels_P30": {r: {c: list(v) for c, v in d.items()} for r, d in lv.items()}}}
    chk("a1 by levels where the banked file holds no such pair: equal passes, off by 1e-6 fails, a run missing fails",
        ctl_a1(None, lv, bank, "RQ", "R0")["route"] == "levels" and ctl_a1(None, lv, bank, "RQ", "R0")["PASS"]
        and not ctl_a1(None, {"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2]), "R0": lv["R0"]}, bank, "RQ", "R0")["PASS"] and not ctl_a1(None, lv, bank, "RQ", "RX")["PASS"])
    bank2 = {"controls_all_pass": True, "pairs": {"P": {"A": "RQ", "B": "R0", "headway": hw(0.5)}}, "cells": {"P": {"c": {"D": 1.0, "per_seed": [1.0, 1.0]}}}}
    here = {"headway": hw(0.5), "cells": {"c": {"D": 1.0, "per_seed": [1.0, 1.0]}}}
    chk("a1 by the pair where the banked file holds it: equal passes, a cell off by 1e-6 fails", ctl_a1(here, {}, bank2, "RQ", "R0")["route"] == "pair" and ctl_a1(here, {}, bank2, "RQ", "R0")["PASS"]
        and not ctl_a1({"headway": hw(0.5), "cells": {"c": {"D": 1.000001, "per_seed": [1.0, 1.0]}}}, {}, bank2, "RQ", "R0")["PASS"])
    chk("qualification by the instrument's own bar (200 windows with a lead, 20 episodes)",
        qualified({"n_lead_windows": 325, "n_episodes_with_lead": 46}, types.SimpleNamespace(MIN_WINDOWS=200, MIN_EPISODES=20)) and not qualified({"n_lead_windows": 199, "n_episodes_with_lead": 46}, types.SimpleNamespace(MIN_WINDOWS=200, MIN_EPISODES=20)))
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [x for x in bad if x in src])
    ok = all(res)
    print("ZZ SELFTEST-MIDLOOK10K ok: %s (%d of %d)" % (ok, sum(res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "fresh-grid", "agerule-json", "w", "tree", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
