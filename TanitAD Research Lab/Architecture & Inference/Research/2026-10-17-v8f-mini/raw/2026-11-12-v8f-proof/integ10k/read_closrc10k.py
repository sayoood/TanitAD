#!/usr/bin/env python
"""read_closrc10k.py -- read of card PREREG_V8F_PROOF_ADD75 (CLOSRC10K): which input of the lead rule carries the contact of the reader-fed form at 6 s of own driving.

Runs (P = 30 grid, pre-roll 30, readout K, heading bound 0.2 1/m, the rule at the replan ticks, model seeds 0 and 1): RR RQ50KP30Hb20 (reader presence and gap, closing 0),
RRc RQ50cKP30Hb20 (closing from labels), RRh RQ50hKP30Hb20 (closing from the reader), RLb RLbKP30Hb20 (labels, closing 0), RLc RLcKP30Hb20 (labels with closing); banked R0KP30Hb20
(no rule at age), R0KHb20 (fresh), R0Hb20 (spread), R0 (control of the strategic read).
The grid read and the strategic read of the reader of card ADD71 (read_longroll10k under its notes N3 and N4, not edited) give the headway shares (U, C, F), the family cells and
the route-goal cells of every pair; the name words are those of that reader (u_name for U and for C, reading for a family). The attribution statistic on the contact arrays of the
windows with a lead: (C_RR minus C_X) minus one half of (C_RR minus C_RLc), read as x = 0.5 C_RR + 0.5 C_RLc against y = C_X through the engine of the headway read, on the windows
with a value in all four runs.
Controls (card section 4): g engine controls of the grid read; s stages A and B of the strategic read; a1 the pair no rule at age minus fresh equals the pair AGE-K of the banked
re-read of card ADD70; r1 run texts and dump counts; x1 the closing speed fed to the rule (rule logs); l1 the contact levels of the window arrays equal those of the grid read;
i1 identity of the attribution statistic; h1 planted attribution. A control that does not pass: no name and no value is written.
Usage: python read_closrc10k.py --pull PULL --code CODE --grid-dir DIR --add70-json FILE --w W --tree TREE --out OUT.json [--n-boot 2000]      |      --selftest
Last line: ZZ READ-CLOSRC10K rc <rc> controls_all_pass <bool>
"""
import argparse
import io
import json
import os
import re
import sys

import read_handsrc_decomp as HS

AD, M, R, N4, np = HS.AD, HS.M, HS.R, HS.N4, HS.np
CARD, PRE = "PREREG_V8F_PROOF_ADD75", "CLOSRC10K"
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
RUNS = {"RR": "RQ50KP30Hb20", "RRc": "RQ50cKP30Hb20", "RRh": "RQ50hKP30Hb20", "RLb": "RLbKP30Hb20", "RLc": "RLcKP30Hb20"}
NORULE, FRESH, SPREAD = "R0KP30Hb20", "R0KHb20", "R0Hb20"
PAIRS = {"LABEL-CLOSING": ("RRc", "RR"), "LABEL-SOURCE": ("RLb", "RR"), "READER-CLOSING": ("RRh", "RR"), "LABEL-FED": ("RLc", "RR")}
C_NAMED = ("LABEL-CLOSING", "LABEL-SOURCE", "READER-CLOSING")
RULE_PAIR, GAP_PAIR, BANK_PAIR = "READER-FED-VS-NO-RULE", "GAP", "NO-RULE-AT-AGE-MINUS-FRESH"
ZERO_CLOSING, FED_CLOSING = ("RR", "RLb"), ("RRc", "RRh", "RLc")
WRAPPER_SHA = "92a8034cc0f7b0d2"
TOL_ID = 1e-9
RUN_RE = re.compile(r"form (\S+) replan steps (\d+) rule every (\d+) readout (\S+) kappa (\S+) preroll (\d+)")


def attribution(RH, c_rr, c_rlc, c_x, inv, n_ep, n_boot):
    """c_*: per seed the contact array of the windows (NaN where the window is not scored). On the windows with a value in all three: gap = RR minus RLc, removed = RR minus X,
    Q = removed minus one half of the gap. -> dict with the three cell records, the removed share and the word"""
    rr, rl, xx, n = [], [], [], []
    for a, b, c in zip(c_rr, c_rlc, c_x):
        a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        rr.append(np.where(ok, a, np.nan))
        rl.append(np.where(ok, b, np.nan))
        xx.append(np.where(ok, c, np.nan))
        n.append(int(ok.sum()))
    gap = HS.pair(RH, rr, rl, inv, n_ep, n_boot)
    rem = HS.pair(RH, rr, xx, inv, n_ep, n_boot)
    q = HS.pair(RH, [0.5 * a + 0.5 * b for a, b in zip(rr, rl)], xx, inv, n_ep, n_boot)
    return {"gap": gap, "removed": rem, "Q_removed_minus_half_gap": q, "removed_share": (rem["D"] / gap["D"]) if abs(gap["D"]) > 1e-12 else None, "carries": bool(q["interval"][0] > 0),
            "n_windows_per_seed": n}


def ctl_i1(blocks):
    w = 0.0
    for b in blocks:
        w = max(w, abs(b["Q_removed_minus_half_gap"]["D"] - (b["removed"]["D"] - 0.5 * b["gap"]["D"])))
    return {"PASS": bool(blocks and w <= TOL_ID), "largest_abs_dev": w, "n_blocks": len(blocks), "tolerance": TOL_ID}


def ctl_h1(RH, n_boot=300, n_ep=40, per=10, seed=11, sd=0.3):
    """planted: a gap of 1; an input that removes 0.8 of it is CARRIES, one that removes 0.2 is not"""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    inv = np.repeat(np.arange(n_ep), per)
    out, ok = {}, True
    for case, keep, want in (("removes_0.8", 0.2, True), ("removes_0.2", 0.8, False)):
        rl = [rng.normal(0.0, sd, n) for _ in SEEDS]
        rr = [x + 1.0 + rng.normal(0.0, sd, n) for x in rl]
        xx = [x + keep + rng.normal(0.0, sd, n) for x in rl]
        b = attribution(RH, rr, rl, xx, inv, n_ep, n_boot)
        out[case] = {"want_carries": want, "got": b["carries"], "removed_share": b["removed_share"]}
        ok = ok and b["carries"] is want
    return dict(out, PASS=bool(ok))


def parse_run_text(text):
    """-> (form, replan steps, rule every, readout, kappa, preroll, wrapper sha prefix, rc 0 finished) from the text the queue writes for a rollout; None when a part is missing"""
    m = RUN_RE.search(text or "")
    if not m:
        return None
    w = re.search(r"wrapper sha256 ([0-9a-f]{16})", text)
    return {"form": m.group(1), "replan_steps": int(m.group(2)), "rule_every": int(m.group(3)), "readout": m.group(4), "kappa": m.group(5), "preroll": int(m.group(6)),
            "wrapper": w.group(1) if w else None, "rc0": bool(re.search(r"^rc 0 finished", text, re.M))}


def run_text_ok(p, form):
    return bool(p and p["form"] == form and p["replan_steps"] == 10 and p["rule_every"] == 10 and p["readout"] == "causal_g" and p["kappa"] == "0.2" and p["preroll"] == 30
                and p["wrapper"] == WRAPPER_SHA and p["rc0"])


def ctl_r1(texts, n_files, n_clips):
    """texts: {(form, seed): text or None}; n_files: {(form, seed): dump file count}"""
    out, ok = {}, bool(texts)
    for f in RUNS:
        for s in SEEDS:
            p = parse_run_text(texts.get((f, s)))
            good = run_text_ok(p, f) and n_files.get((f, s)) == n_clips
            out["%s:seed%d" % (f, s)] = {"ok": bool(good), "parsed": p, "dump_files": n_files.get((f, s)), "clips_of_the_grid": n_clips}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def closing_fed(z):
    """rule log arrays -> (rows, rows with a lead present, largest |closing| over all rows, rows present with closing != 0, closing all finite, form in the log)"""
    pres = np.asarray(z["present"], bool)
    c = np.asarray(z["c_src"], np.float64)
    fin = np.isfinite(c)
    try:
        form = str(np.asarray(z["form"]).item())
    except Exception:                                                                         # noqa: BLE001
        form = None
    return {"rows": int(len(c)), "present": int(pres.sum()), "max_abs_closing": float(np.max(np.abs(c[fin]))) if fin.any() else 0.0, "present_nonzero": int((pres & fin & (c != 0.0)).sum()),
            "closing_finite_where_present": bool(fin[pres].all()) if pres.any() else True, "form": form}


def ctl_x1(logs):
    """logs: {(form, seed): closing_fed record or None}. The closing speed fed is exactly 0 on every row with a lead present of RR and RLb and differs from 0 on at least one such row of RRc, RRh, RLc"""
    out, ok = {}, bool(logs)
    for f in RUNS:
        for s in SEEDS:
            r = logs.get((f, s))
            if not r:
                out["%s:seed%d" % (f, s)], ok = {"ok": False, "why": "no rule log"}, False
                continue
            if f in ZERO_CLOSING:
                good = r["form"] == f and r["present"] > 0 and r["present_nonzero"] == 0 and r["closing_finite_where_present"]
            else:
                good = r["form"] == f and r["present_nonzero"] > 0 and r["closing_finite_where_present"]
            out["%s:seed%d" % (f, s)] = dict(r, ok=bool(good))
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def words(rd_c, carries_closing, carries_source):
    """rd_c: {pair name: C word}; -> the names of the card that do not belong to a family"""
    names = {"GAP": "%s-GAP-PRESENT" % PRE if rd_c[GAP_PAIR] == "HIGHER" else "%s-NO-GAP-AT-TICK-LEVEL" % PRE}
    for nm in C_NAMED:
        names[nm + "_C"] = "%s-%s-C-%s" % (PRE, nm, rd_c[nm])
    if rd_c[GAP_PAIR] == "HIGHER":
        if carries_closing:
            names["ATTRIBUTION_CLOSING"] = "%s-CLOSING-CARRIES" % PRE
        if carries_source:
            names["ATTRIBUTION_SOURCE"] = "%s-SOURCE-CARRIES" % PRE
        if not carries_closing and not carries_source:
            names["ATTRIBUTION"] = "%s-NEITHER-INPUT-ALONE" % PRE
    return names


def assemble(controls, payload):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "pairs": None, "cells": None, "strategic": None, "levels": None,
           "attribution": None, "counts": None}
    if res["controls_all_pass"]:
        res.update(payload)
    return res


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
    n_clips = int(len(np.unique(keys[:, 0])))
    bank = R.jl(a.add70_json)
    controls, payload = {}, None
    runs = [SPREAD, FRESH, NORULE] + list(RUNS.values())
    pairs = {nm: (RUNS[x], RUNS[y]) for nm, (x, y) in PAIRS.items()}
    pairs[RULE_PAIR] = (RUNS["RR"], NORULE)
    pairs[GAP_PAIR] = (RUNS["RR"], RUNS["RLc"])
    pairs[BANK_PAIR] = (NORULE, FRESH)
    G = R.read_grid(a, E, "30", keys, runs, SPREAD, pairs, {}, log)
    controls["g_engine_controls_of_the_grid_read"] = {"PASS": bool(G["controls"].get("ALL_PASS") is True), "detail": G["controls"], "missing_or_incomplete": G.get("missing_or_incomplete")}
    if controls["g_engine_controls_of_the_grid_read"]["PASS"]:
        pb = G["pairs"][BANK_PAIR]
        controls["a1_no_rule_at_age_minus_fresh_equals_card_add70"] = HS.a1_check({"A": pb["A"], "B": pb["B"], "headway": R.ucf(pb["headway"]), "cells": R.cells_of(pb["family"])}, bank)
        S = cells_s = eid_s = None
        try:
            p_of = {"R0": 0, NORULE: 30}
            p_of.update({r: 30 for r in RUNS.values()})
            SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": bool(SR.get("stage_A_pass") and SR.get("stage_B_pass")), "detail": dict({k: v for k, v in SR["controls"].items() if k != "s3"},
                                                                                                                                      s3=SR["controls"]["s3"].get("pass"))}
        except SystemExit as ex:
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "SystemExit: %s" % ex}
        except Exception as ex:                                                               # noqa: BLE001
            controls["s_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
        texts, n_files, logs, Cw, lev = {}, {}, {}, {}, {}
        ref = None
        for f, run_ in RUNS.items():
            for s in SEEDS:
                key = R.key_of(run_, s)
                texts[(f, s)] = R.txt(os.path.join(R.tick_dir(a.pull, run_, s), "run_%s.txt" % key))
                z = R.npz_load(os.path.join(R.tick_dir(a.pull, run_, s), "rule_log_%s.npz" % key))
                logs[(f, s)] = closing_fed(z) if z is not None else None
                raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
                if raw is None:
                    continue
                n_files[(f, s)] = int(raw["n_files"])
                idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
                if rep["missing"]:
                    continue
                D = R.cut(raw, idx)
                ref = D if ref is None else ref
                L = H.label_windows(J, D["clip_index"], D["ws"])
                Cw[(f, s)] = np.asarray(H.window_cells(D["g"], D["cl"], L)["C"], np.float64)
                lev[(f, s)] = float(H.level(Cw[(f, s)]))
        controls["r1_run_texts_and_dump_counts"] = ctl_r1(texts, n_files, n_clips)
        controls["x1_closing_speed_fed_to_the_rule"] = ctl_x1(logs)
        have = all((f, s) in Cw for f in RUNS for s in SEEDS)
        dl = max([abs(lev[(f, s)] - float(G["levels"][RUNS[f]]["C"][i])) for f in RUNS for i, s in enumerate(SEEDS)]) if have else float("inf")
        controls["l1_contact_levels_equal_the_grid_read"] = {"PASS": bool(have and dl <= 1e-12), "largest_abs_dev": dl}
        controls["h1_attribution_planted"] = ctl_h1(RH, a.n_boot_control)
        att = {}
        if have:
            uniq, inv = RH.episode_units(ref["eid"])
            n_ep = len(uniq)
            col = lambda f: [Cw[(f, s)] for s in SEEDS]                                         # noqa: E731
            att = {"CLOSING": attribution(RH, col("RR"), col("RLc"), col("RRc"), inv, n_ep, a.n_boot), "SOURCE": attribution(RH, col("RR"), col("RLc"), col("RLb"), inv, n_ep, a.n_boot),
                   "READER-CLOSING": attribution(RH, col("RR"), col("RLc"), col("RRh"), inv, n_ep, a.n_boot)}
            controls["i1_identity_of_the_attribution_statistic"] = ctl_i1(list(att.values()))
        else:
            controls["i1_identity_of_the_attribution_statistic"] = {"PASS": False, "why": "window arrays missing"}
        if all(v.get("PASS") is True for v in controls.values()):
            pairs_out, cells_out, strat, rd, rd_c = {}, {}, {}, {}, {}
            for nm, (A, B) in pairs.items():
                if nm == BANK_PAIR:
                    continue
                p = G["pairs"][nm]
                hw = R.ucf(p["headway"])
                rd_c[nm] = R.u_name(hw["C"])
                pairs_out[nm] = {"A": A, "B": B, "headway": hw, "U_word": R.u_name(hw["U"]), "C_word": rd_c[nm]}
                if nm == GAP_PAIR:
                    continue
                cells_out[nm] = R.cells_of(p["family"])
                fam = {f: R.reading(cells_out[nm], f) for f in R.NAMED}
                pr = S.pair_read(cells_s, A, B, eid_s, ticks=(2,))
                vs = [c["verdict"] for c in pr["tick2"]["pos"]]
                strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")}
                                                                for arc, b in zip(S.ARCS, pr["tick2"]["pos"])], "reading": R.family_of(vs)}
                fam["STRATEGIC"] = strat[nm]["reading"]
                rd[nm] = {"fam": fam, "U": pairs_out[nm]["U_word"]}
            names = words(rd_c, att["CLOSING"]["carries"], att["SOURCE"]["carries"])
            for nm, r_ in rd.items():
                names[nm + "_U"] = "%s-%s-U-%s" % (PRE, nm, r_["U"])
                for f, w in r_["fam"].items():
                    names["%s_%s" % (nm, f)] = "%s-%s-%s-%s" % (PRE, nm, f, w)
            levels = {f: G["levels"][RUNS[f]] for f in RUNS}
            levels["NO-RULE"] = G["levels"][NORULE]
            payload = {"names": names, "pairs": pairs_out, "cells": cells_out, "strategic": strat, "levels": levels, "attribution": att, "counts": G["counts"], "readings": rd}
    res = assemble(controls, payload)
    print(TIER_LINE)
    print("== controls")
    for k, v in controls.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x not in ("PASS", "detail")}))[:600]))
    if res["controls_all_pass"]:
        print("== names")
        for k, v in res["names"].items():
            print("   %-34s %s" % (k, v))
        print("== levels of U, C, F per run and model seed")
        for f, lv in res["levels"].items():
            print("   %-8s U %s  C %s  F %s" % (f, ["%.4f" % x for x in lv["U"]], ["%.4f" % x for x in lv["C"]], ["%.4f" % x for x in lv["F"]]))
        h3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["pooled"], c["interval"][0], c["interval"][1])   # noqa: E731
        f3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["D"], c["interval"][0], c["interval"][1])       # noqa: E731
        print("== pairs: U, C, F")
        for nm, p in res["pairs"].items():
            print("   %-24s U %s %s  C %s %s  F %s" % (nm, h3(p["headway"]["U"]), p["U_word"], h3(p["headway"]["C"]), p["C_word"], h3(p["headway"]["F"])))
        print("== attribution on the contact arrays: gap (RR minus RLc), removed (RR minus X), Q = removed minus half of the gap, removed share")
        for nm, b in res["attribution"].items():
            print("   %-16s gap %s  removed %s  Q %s  share %s  carries %s  n %s" % (nm, f3(b["gap"]), f3(b["removed"]), f3(b["Q_removed_minus_half_gap"]),
                                                                              "n/a" if b["removed_share"] is None else "%.2f" % b["removed_share"], b["carries"], b["n_windows_per_seed"]))
        print("== family cells and route goal per pair")
        for nm in res["cells"]:
            for cell, c in res["cells"][nm].items():
                print("   %-24s %-30s %s %s" % (nm, cell, f3(c), c["verdict"]))
            for b in res["strategic"][nm]["pos_30_50_80m_tick_t0_plus_20"]:
                print("   %-24s STRATEGIC pos %sm %+.4f [%+.4f,%+.4f] %s n %s" % (nm, b["arc_m"], b["mean"], b["lo"], b["hi"], b["verdict"], b["n_per_seed"]))
    else:
        print("== a control did not pass: no name and no value is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-CLOSRC10K rc 0 controls_all_pass %s" % res["controls_all_pass"])
    return 0


def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-116s %s" % (name, bool(ok)))
    h = ctl_h1(AD.MiniRH, 300)
    chk("h1 planted through the stand-in engine: removes 0.8 -> carries %s, removes 0.2 -> carries %s" % (h["removes_0.8"]["got"], h["removes_0.2"]["got"]), h["PASS"])
    rng = np.random.default_rng(2)
    n_ep, per = 30, 8
    inv = np.repeat(np.arange(n_ep), per)
    mk = lambda p: [np.where(rng.random(n_ep * per) < 0.4, np.nan, (rng.random(n_ep * per) < p).astype(float)) for _ in SEEDS]   # noqa: E731
    b = attribution(AD.MiniRH, mk(0.3), mk(0.05), mk(0.15), inv, n_ep, 200)
    chk("i1 holds on 0/1 arrays with different windows without a value (common windows %s)" % b["n_windows_per_seed"], ctl_i1([b])["PASS"] and max(b["n_windows_per_seed"]) < n_ep * per)
    chk("i1 fails when the statistic is off by 1e-6", not ctl_i1([dict(b, Q_removed_minus_half_gap=dict(b["Q_removed_minus_half_gap"], D=b["Q_removed_minus_half_gap"]["D"] + 1e-6))])["PASS"])
    t = ("arm I10RQ50cKP30Hb20_s0 form RRc replan steps 10 rule every 10 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 ckpt sha256 ab "
         "wrapper sha256 92a8034cc0f7b0d232a6f18a5b8d1f03 started 2026-11-20T12:00:00Z\nrc 0 finished 2026-11-20T12:04:00Z\n")
    p = parse_run_text(t)
    chk("run text parsed: form, cycle, rule cadence, readout, bound, pre-roll, wrapper, return code", run_text_ok(p, "RRc") and not run_text_ok(p, "RR"))
    chk("run text with the rule at every 5 steps, another readout or a return code 1 is not accepted", not run_text_ok(parse_run_text(t.replace("rule every 10", "rule every 5")), "RRc")
        and not run_text_ok(parse_run_text(t.replace("causal_g", "pinned")), "RRc") and not run_text_ok(parse_run_text(t.replace("rc 0 finished", "rc 1 finished")), "RRc"))
    texts = {(f, s): t.replace("form RRc", "form " + f) for f in RUNS for s in SEEDS}
    nf = {(f, s): 141 for f in RUNS for s in SEEDS}
    chk("r1 passes on ten good run texts and fails on a dump with a clip missing", ctl_r1(texts, nf, 141)["PASS"] and not ctl_r1(texts, {**nf, ("RR", 0): 140}, 141)["PASS"])

    def zl(form, c_vals):
        pres = np.array([True, True, False, True])
        return closing_fed({"present": pres, "c_src": np.asarray(c_vals, np.float64), "form": np.asarray(form)})
    good = {(f, s): zl(f, [0.0, 0.0, 0.0, 0.0] if f in ZERO_CLOSING else [0.0, -1.5, 0.0, 0.3]) for f in RUNS for s in SEEDS}
    chk("x1 passes: closing 0 for RR and RLb, fed for RRc, RRh, RLc", ctl_x1(good)["PASS"])
    chk("x1 fails when RR was fed a closing speed, when RRh was fed none, and when a log names another form",
        not ctl_x1({**good, ("RR", 0): zl("RR", [0.0, 0.2, 0.0, 0.0])})["PASS"] and not ctl_x1({**good, ("RRh", 1): zl("RRh", [0.0, 0.0, 0.0, 0.0])})["PASS"]
        and not ctl_x1({**good, ("RLc", 0): zl("RLb", [0.0, -1.5, 0.0, 0.3])})["PASS"])
    w = words({GAP_PAIR: "HIGHER", "LABEL-CLOSING": "LOWER", "LABEL-SOURCE": "NOT-SEPARATED", "READER-CLOSING": "LOWER"}, True, False)
    chk("names: gap present, closing carries", w["GAP"] == "CLOSRC10K-GAP-PRESENT" and w["ATTRIBUTION_CLOSING"] == "CLOSRC10K-CLOSING-CARRIES" and "ATTRIBUTION_SOURCE" not in w
        and w["READER-CLOSING_C"] == "CLOSRC10K-READER-CLOSING-C-LOWER")
    w2 = words({GAP_PAIR: "NOT-SEPARATED", "LABEL-CLOSING": "LOWER", "LABEL-SOURCE": "NOT-SEPARATED", "READER-CLOSING": "LOWER"}, True, True)
    chk("names: with no gap no attribution name is written", w2["GAP"] == "CLOSRC10K-NO-GAP-AT-TICK-LEVEL" and not [k for k in w2 if k.startswith("ATTRIBUTION")])
    w3 = words({GAP_PAIR: "HIGHER", "LABEL-CLOSING": "NOT-SEPARATED", "LABEL-SOURCE": "NOT-SEPARATED", "READER-CLOSING": "NOT-SEPARATED"}, False, False)
    chk("names: neither input alone", w3.get("ATTRIBUTION") == "CLOSRC10K-NEITHER-INPUT-ALONE")
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}})
    chk("a failed control leaves no name", a_["names"] is None and a_["controls_all_pass"] is False)
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [x for x in bad if x in src])
    ok = all(res)
    print("ZZ SELFTEST-CLOSRC10K ok: %s (%d of %d)" % (ok, sum(res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "add70-json", "w", "tree", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
