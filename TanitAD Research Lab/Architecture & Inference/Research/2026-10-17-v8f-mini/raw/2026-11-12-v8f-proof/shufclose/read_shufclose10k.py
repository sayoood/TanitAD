#!/usr/bin/env python
"""read_shufclose10k.py -- read of card PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K): is the gain of the estimated closing speed from the scene, or from the spread of the input alone?
Derived from read_estclose10k_n2.py (card ADD79, imported and not edited; under it the engine modules read_handsrc_decomp, read_closrc10k, read_longroll10k). Pattern of constclose/read_constclose10k_n1.py.

Two settings, each read completely before the next starts. New runs (both seeds): AGE RQ50eSL25KF5P30Hb20, RQ50eSN10KF5P30Hb20; FRESH RQ50eSL25KF5Hb20, RQ50eSN10KF5Hb20 (form RRe, closing input = the surrogate table
of the estimate, --closing-table). Banked: the runs of card ADD79 RRe(N10), RRe(L25) (required; their read, --estclose-json, has to say card PREREG_V8F_PROOF_ADD79 and controls_all_pass true) and RRh, RRc, RR, RLc, R0K (row 137).
Pairs per estimator E (N10, L25) and setting, A minus B: EST-VS-SHUF-E (RRe(E), RRe(surrogate of E)), SHUF-VS-RR-E (RRe(surrogate), RR), SHUF-VS-NO-RULE-E (RRe(surrogate), R0K).
Names SHUFCLOSE10K-<PAIR>-<SETTING>-<C|U|F|family>-<word>. Verdicts per E on the AGE setting: SHUFCLOSE10K-<E>-GAIN-FROM-THE-SCENE (EST-VS-SHUF-E at AGE reads C LOWER and U not HIGHER) or
SHUFCLOSE10K-<E>-GAIN-NOT-SHOWN-FROM-THE-SCENE, the name of SHUF-VS-RR-E at AGE (C) written beside; the N10 verdict is marked as resting on a gain that row 141 did not have separated.
Controls (card section 5), printed first: sur (SURROGATE_RECORD.json says pass; its u1 and u4 are carried), u2, u3 (gate per estimator at AGE; FRESH reported, not judged), k3 (levels of the runs of card ADD79 equal
--estclose-json), g, s, a1, r1, x1, m1, i1, h1. A control that does not pass: no name and no value is written or printed. An estimator whose u3 gate fails has no AGE name and no verdict.
Usage: python read_shufclose10k.py --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --shuf-dir DIR --w W --tree TREE --out OUT.json --wrapper-sha HEX16
          [--surrogate-record FILE] [--n-boot 2000]   |   --selftest
Last line: ZZ READ-SHUFCLOSE10K rc <rc> controls_all_pass <bool>
"""
import argparse
import hashlib
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("SC_ESTREAD_DIR"), os.path.join(_HERE, "..", "estclose_read"), _HERE):
    if _p and os.path.isfile(os.path.join(_p, "read_estclose10k_n2.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import read_estclose10k_n2 as EC        # noqa: E402

HS, CR, AD, M, R, np = EC.HS, EC.CR, EC.AD, EC.M, EC.R, EC.np
CARD, PRE, PARENT_CARD, SUR_CARD = "PREREG_V8F_PROOF_ADD83", "SHUFCLOSE10K", "PREREG_V8F_PROOF_ADD79", "PREREG_V8F_PROOF_ADD83"
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
FAMS, NOT_HIGHER = EC.FAMS, EC.NOT_HIGHER
ESTS = ("N10", "L25")
NEWS = tuple("RRe_S" + e for e in ESTS)
EEST = tuple("RRe_" + e for e in ESTS)
BANKED = EC.BANKED
FORMS = NEWS + EEST + BANKED
FORM_OF = dict(EC.FORM_OF, RRe_SN10="RRe", RRe_SL25="RRe")
TOL_U2 = 1e-9
SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90
PAIRS, PAIR_EST = {}, {}
for _e in ESTS:
    for _n, _x, _y in (("EST-VS-SHUF", "RRe_" + _e, "RRe_S" + _e), ("SHUF-VS-RR", "RRe_S" + _e, "RR"), ("SHUF-VS-NO-RULE", "RRe_S" + _e, "R0K")):
        PAIRS["%s-%s" % (_n, _e)] = (_x, _y)
        PAIR_EST["%s-%s" % (_n, _e)] = _e
INTERNAL = EC.INTERNAL
MARK_N10 = PRE + "-N10-VERDICT-RESTS-ON-A-GAIN-THAT-ROW-141-DID-NOT-HAVE-SEPARATED"
MARK_L25 = PRE + "-L25-GAIN-OF-ROW-141-WAS-SEPARATED"


def settings():
    out = {}
    for st, P, grid, suf in (("AGE", 30, "30", "P30Hb20"), ("FRESH", 0, "full", "Hb20")):
        runs = {"RRe_S" + e: "RQ50eS%sKF5%s" % (e, suf) for e in ESTS}
        runs.update({"RRe_" + e: "RQ50e%sKF5%s" % (e, suf) for e in ESTS})
        runs.update({f: EC.SETTINGS[st]["runs"][f] for f in BANKED})
        out[st] = {"P": P, "grid": grid, "spread": "R0Hb20", "runs": runs}
    return out


SET = settings()


# ============================================================================================ names and verdicts
def cname(pair, setting, kind, word):
    return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)


def nh(word):
    return None if word is None else bool(word in NOT_HIGHER)


def verdict_names(words, age_ok):
    """words[setting][pair] = {"C", "U", "F", "fam"}; age_ok[e]: the AGE read of e is admitted by u3. -> (names, why). Per estimator, on AGE: GAIN-FROM-THE-SCENE when EST-VS-SHUF-E reads C LOWER and U not HIGHER;
    GAIN-NOT-SHOWN-FROM-THE-SCENE when one of the two conditions fails (a failing condition decides without the other word); not written when a needed word is not available. The name of SHUF-VS-RR-E (C, AGE)
    is written beside every verdict that is written. The verdict of N10 carries a mark."""
    names, why = {}, {}
    for e in ESTS:
        key = "VERDICT_" + e
        if not age_ok.get(e):
            why[key] = "not written: u3 (the surrogate has the spread of the estimate) did not pass for %s at AGE" % e
            continue
        w = words["AGE"]["EST-VS-SHUF-" + e]
        conds = [("AGE C LOWER", None if w["C"] is None else w["C"] == "LOWER"), ("AGE U not HIGHER", nh(w["U"]))]
        if any(v is False for _, v in conds):
            names[key] = "%s-%s-GAIN-NOT-SHOWN-FROM-THE-SCENE" % (PRE, e)
        elif any(v is None for _, v in conds):
            why[key] = "not written: the headway instrument is not qualified at AGE (%s)" % ", ".join(n for n, v in conds if v is None)
        else:
            names[key] = "%s-%s-GAIN-FROM-THE-SCENE" % (PRE, e)
        if key in names:
            sr = words["AGE"]["SHUF-VS-RR-" + e]["C"]
            if sr is None:
                why[key + "_BESIDE"] = "the name of SHUF-VS-RR-%s is not written: the headway instrument is not qualified at AGE" % e
            else:
                names[key + "_BESIDE_SHUF_VS_RR"] = cname("SHUF-VS-RR-" + e, "AGE", "C", sr)
            names[key + "_MARK"] = MARK_N10 if e == "N10" else MARK_L25
    return names, why


def pair_names(setting, pair, wd):
    out = {}
    for kind in ("C", "U", "F"):
        if wd[kind] is not None:
            out["%s_%s_%s" % (pair, setting, kind)] = cname(pair, setting, kind, wd[kind])
    for f in FAMS:
        out["%s_%s_%s" % (pair, setting, f)] = cname(pair, setting, f, wd["fam"][f])
    return out


def all_names(words, quals, age_ok):
    names, why = {}, {}
    for st in SET:
        if not quals[st]:
            why["HEADWAY_" + st] = "U, C and F of the setting %s are not reported: the headway instrument is not qualified on its grid" % st
        for pair in PAIRS:
            if st == "AGE" and not age_ok.get(PAIR_EST[pair]):
                why["AGE_%s" % pair] = "AGE names of %s are not written: u3 did not pass for %s" % (PAIR_EST[pair], PAIR_EST[pair])
                continue
            names.update(pair_names(st, pair, words[st][pair]))
    vn, vw = verdict_names(words, age_ok)
    names.update(vn)
    why.update(vw)
    return names, why


# ============================================================================================ controls
def ctl_sur(rec, shas):
    """rec: SURROGATE_RECORD.json of make_surrogate.py; shas: {file name: sha256 of the table file read here}. PASS: the record is that of card ADD83, says pass, u1 passed for every table, u4 passed for both
    estimators, and the sha256 of every surrogate table read here equals the one in the record. u1 and u4 are carried."""
    try:
        if not isinstance(rec, dict) or rec.get("card") != SUR_CARD or rec.get("pass") is not True:
            return {"PASS": False, "why": "the surrogate record is missing, is not that of card ADD83 or does not say pass"}
        u1, u4 = rec.get("u1") or {}, rec.get("u4") or {}
        want_u1 = ["%s_s%d" % (e, s) for e in ESTS for s in SEEDS]
        u1_ok = bool(all(k in u1 and u1[k].get("passed") is True for k in want_u1))
        u4_ok = bool(all(e in u4 and u4[e].get("passed") is True for e in ESTS))
        files = rec.get("files") or {}
        bad = [n for n, h in shas.items() if files.get(n) != h]
        ok = bool(u1_ok and u4_ok and shas and not bad)
        return {"PASS": ok, "u1_all_passed": u1_ok, "u4_all_passed": u4_ok, "tables_checked": len(shas), "tables_not_equal_to_the_record": bad, "u1_carried": u1, "u4_carried": u4}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def ctl_r1s(texts, n_files, n_clips, preroll, wrapper, shas):
    """run text of every new run: form RRe, cycle 10, rule every 5, readout K, bound, pre-roll, wrapper, return code 0, one dump file per clip, and the file hash of the est line equals the table's."""
    out, ok = {}, bool(texts)
    for f in NEWS:
        for s in SEEDS:
            p = CR.parse_run_text(texts.get((f, s)))
            pe = EC.parse_est_looks(texts.get((f, s)))
            sha_ok = bool(pe is not None and pe["file_sha256"] == shas.get((f[len("RRe_S"):], s)))
            good = EC.run_text_ok(p, preroll, wrapper) and n_files.get((f, s)) == n_clips and sha_ok
            out["%s:seed%d" % (f, s)] = {"ok": bool(good), "parsed": p, "dump_files": n_files.get((f, s)), "clips_of_the_grid": n_clips, "est_line_file_equals_table": sha_ok}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_x1s(rule, mid):
    """as card ADD79 x1 for the new forms: a closing that is not 0 on at least one present row in both logs, finite where present, exactly 0.0 on present rows without a full history, form RRe in the rule log;
    the banked RR run feeds 0 on every row of both logs"""
    out, ok = {}, bool(rule) and bool(mid)
    for f in NEWS + ("RR",):
        for s in SEEDS:
            res, good = {}, True
            for tag, lg in (("rule_log", rule), ("mid_log", mid)):
                r = lg.get((f, s))
                if not r:
                    res[tag], good = {"ok": False, "why": "no log"}, False
                    continue
                if f == "RR":
                    g1 = r["rows"] > 0 and r["present_nonzero"] == 0 and r["max_abs_closing"] == 0.0 and r["closing_finite_where_present"]
                else:
                    g1 = r["present"] > 0 and r["present_nonzero"] > 0 and r["closing_finite_where_present"] and r.get("est_ok_present") is not None and r["est_ok_present"] > 0 and r.get("fallback_closing_exactly_zero") is True
                    if tag == "rule_log":
                        g1 = g1 and r["form"] == "RRe"
                res[tag] = dict(r, ok=bool(g1))
                good = good and g1
            out["%s:seed%d" % (f, s)] = dict(res, ok=bool(good))
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_m1s(rows):
    """later-look row counts of the new runs equal those of the banked RR run of the same seed, and are not empty"""
    out, ok = {}, bool(rows)
    for s in SEEDS:
        base = rows.get(("RR", s))
        for f in NEWS:
            n = rows.get((f, s))
            good = bool(n and base and n == base)
            out["%s:seed%d" % (f, s)] = {"ok": good, "rows": n, "rows_banked_RR": base}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_u2(tables, logs, tol=TOL_U2):
    """tables: {(E, seed): surrogate table arrays or None}; logs: {(form, seed): {"rule": arrays or None, "mid": arrays or None}} with ci, j, k, c_est, est_ok. In all eight runs and both logs: on the rows with est_ok True
    c_est equals the table at (ci, j) to 1e-9; est_ok is True exactly where the table holds a finite value at (ci, j); c_est is NaN where est_ok is False; rows of the first look (rule log k == 0; mid-look log: its
    smallest tick) are among the compared rows."""
    out, ok, worst = {}, True, 0.0
    for e in ESTS:
        for s in SEEDS:
            tab = tables.get((e, s))
            for tag in ("rule", "mid"):
                z = (logs.get(("RRe_S" + e, s)) or {}).get(tag)
                nm = "%s:seed%d:%s_log" % (e, s, tag)
                if tab is None or z is None:
                    out[nm], ok = {"ok": False, "why": "table or log missing"}, False
                    continue
                look = EC.est_lookup(tab)
                eo = np.asarray(z["est_ok"], bool)
                ce = np.asarray(z["c_est"], np.float64)
                ci, j, kk = np.asarray(z["ci"]), np.asarray(z["j"]), np.asarray(z["k"])
                tv_all, found_all = look(ci, j)
                pattern_bad = int(((found_all & np.isfinite(tv_all)) != eo).sum())
                tv, found = tv_all[eo], found_all[eo]
                d = np.abs(ce[eo] - tv)
                bad = (~found) | ~np.isfinite(tv) | ~np.isfinite(ce[eo]) | ~(d <= tol)
                first = 0 if tag == "rule" else (int(kk.min()) if len(kk) else 0)
                n_first = int(((kk == first) & eo).sum())
                nan_ok = bool(np.all(np.isnan(ce[~eo])))
                mx = float(np.max(d[~bad])) if (~bad).any() else None
                good = bool(eo.any() and not bad.any() and n_first > 0 and nan_ok and pattern_bad == 0)
                out[nm] = {"ok": good, "rows_compared": int(eo.sum()), "rows_first_look_compared": n_first, "rows_failing": int(bad.sum()), "est_ok_pattern_mismatch": pattern_bad, "nan_where_not_ok": nan_ok,
                           "largest_difference": mx, "tolerance": tol}
                ok = ok and good
                worst = max(worst, mx or 0.0)
    return {"PASS": bool(ok), "largest_difference_overall": worst, "runs": out}


def log_input(z):
    """a rule log or a later-look log -> (c_src on the looks with a lead read present, est_ok on them, looks logged, looks with est_ok) or None"""
    try:
        pr = np.asarray(z["present"], bool)
        cs = np.asarray(z["c_src"], np.float64)
        eo = np.asarray(z["est_ok"], bool)
        return cs[pr], eo[pr], int(len(pr)), int(eo.sum())
    except Exception:                                                                         # noqa: BLE001
        return None


def join_inputs(a, b):
    if a is None or b is None:
        return None
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2] + b[2], a[3] + b[3]


def ctl_u3(stats, st):
    """stats: {(form, seed): (c_src present, est_ok present, n looks, n ok) or None} for the new and the banked estimate runs. Per estimator, pooled over the seeds, over the looks with a lead read present
    (both logs): sd of the surrogate input within 0.80 to 1.25 times the sd of the estimate input of the runs of card ADD79, |mean| of the surrogate input <= 0.15 m/s, share of looks marked full >= 0.90.
    PASS = every number is there. gate[e] True when the three hold (judged at AGE; at FRESH the same numbers are reported and not judged)."""
    out, ok, gate = {}, True, {}
    for e in ESTS:
        sf, ef = "RRe_S" + e, "RRe_" + e
        have = all(stats.get((f, s)) is not None for f in (sf, ef) for s in SEEDS)
        if not have:
            out[e], ok, gate[e] = {"complete": False}, False, False
            continue
        cs = np.concatenate([stats[(sf, s)][0] for s in SEEDS])
        ce = np.concatenate([stats[(ef, s)][0] for s in SEEDS])
        okp = np.concatenate([stats[(sf, s)][1] for s in SEEDS])
        sd_s, sd_e = float(np.std(cs)) if len(cs) else None, float(np.std(ce)) if len(ce) else None
        ratio = sd_s / sd_e if sd_s is not None and sd_e not in (None, 0.0) else None
        mean_s = float(np.mean(cs)) if len(cs) else None
        share = float(np.mean(okp)) if len(okp) else None
        n_all = sum(stats[(sf, s)][2] for s in SEEDS)
        share_all = sum(stats[(sf, s)][3] for s in SEEDS) / n_all if n_all else None
        c1, c2, c3 = bool(ratio is not None and SD_LO <= ratio <= SD_HI), bool(mean_s is not None and abs(mean_s) <= MEAN_MAX), bool(share is not None and share >= SHARE_MIN)
        out[e] = {"complete": True, "sd_surrogate": sd_s, "sd_estimate": sd_e, "sd_ratio": ratio, "mean_surrogate": mean_s, "mean_estimate": float(np.mean(ce)), "share_full_present_looks": share,
                  "share_full_all_looks_reported": share_all, "looks_present_surrogate": int(len(cs)), "looks_present_estimate": int(len(ce)), "sd_ratio_within_0.80_1.25": c1, "mean_within_0.15": c2,
                  "share_at_least_0.90": c3, "judged": bool(st == "AGE")}
        gate[e] = bool(c1 and c2 and c3)
        ok = ok and True
    return {"PASS": bool(ok), "per_estimator": out, "gate_u3": gate, "setting": st}


def ctl_h1(RH, n_boot=300, n_ep=40, per=10, seed=11, sd=0.3):
    """planted name rule: a pair with a known sign reads the expected name through the engine and the name builder of this reader; seeds of opposite sign read NOT-SEPARATED"""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    inv = np.repeat(np.arange(n_ep), per)
    out, ok = {}, True
    for case, eff, want in (("lower", (-1.0, -1.0), "LOWER"), ("higher", (1.0, 1.0), "HIGHER"), ("opposite_seeds", (-1.0, 1.0), "NOT-SEPARATED"), ("no_effect", (0.0, 0.0), "NOT-SEPARATED")):
        ys = [rng.normal(0.0, sd, n) for _ in SEEDS]
        xs = [y + e + rng.normal(0.0, sd, n) for y, e in zip(ys, eff)]
        got = R.u_name(HS.pair(RH, xs, ys, inv, n_ep, n_boot))
        nm = cname("EST-VS-SHUF-N10", "AGE", "C", got)
        out[case] = {"want": want, "got": got, "name": nm}
        ok = ok and got == want and nm == cname("EST-VS-SHUF-N10", "AGE", "C", want)
    return dict(out, PASS=bool(ok))


def assemble(controls, payload):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "why_not_written": None, "settings": None}
    if res["controls_all_pass"]:
        res.update(payload)
    return res


def read_e_json(path, jl):
    """-> (ok, reason, json or None). The read of card ADD79 is required: the file has to be there, be the result of that card and say controls_all_pass true."""
    if not path:
        return False, "no --estclose-json was given", None
    ej = jl(path)
    if ej is None:
        return False, "the file %s is not there or not readable" % path, None
    if ej.get("card") != PARENT_CARD:
        return False, "the file is not the result of card ADD79", None
    if ej.get("controls_all_pass") is not True:
        return False, "the read of card ADD79 has not all its controls passed", None
    return True, "", ej


# ============================================================================================ one setting
def register_runs(cfg):
    for f, r in cfg["runs"].items():
        R.ALL_RUNS.setdefault(r, (FORM_OF[f], "K", cfg["P"], "deadreckon", cfg["grid"], "plan", None if f == "R0K" else 5, 0.2))
    R.ALL_RUNS.setdefault(cfg["spread"], ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2))


def log_arrays(z):
    if z is None:
        return None
    try:
        return {k: np.asarray(z[k]) for k in ("ci", "j", "k", "c_est", "est_ok")}
    except Exception:                                                                         # noqa: BLE001
        return None


def run_setting(a, E, st, keys, bank, ej, ej_note, tables, shas, ctx_cache, log):
    cfg = SET[st]
    H, Q3, RH = E
    runs = cfg["runs"]
    J = H.load_join()
    n_clips = int(len(np.unique(keys[:, 0])))
    register_runs(cfg)
    ctl, extra = {}, {}
    run_list = [cfg["spread"]] + [runs[f] for f in FORMS]
    pairs = {nm: (runs[x], runs[y]) for nm, (x, y) in dict(PAIRS, **INTERNAL).items()}
    G = R.read_grid(a, E, cfg["grid"], keys, run_list, cfg["spread"], pairs, {}, log)
    ctl["g_" + st] = {"PASS": bool(G["controls"].get("ALL_PASS") is True), "detail": G["controls"], "missing_or_incomplete": G.get("missing_or_incomplete")}
    if not ctl["g_" + st]["PASS"]:
        return ctl, None, extra
    ctl["a1_" + st] = EC.ctl_a1({runs[f]: G["levels"][runs[f]] for f in BANKED}, bank, st)
    ctl["k3_" + st] = EC.ctl_a1({runs[f]: G["levels"][runs[f]] for f in EEST}, ej, st) if ej is not None else {"PASS": False, "why": ej_note}
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
    texts, n_files, rule, mid, rows, Cw, lev, elogs, stats = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for f in FORMS:
        for s in SEEDS:
            run_ = runs[f]
            key = R.key_of(run_, s)
            td = R.tick_dir(a.pull, run_, s)
            if f in NEWS:
                texts[(f, s)] = R.txt(os.path.join(td, "run_%s.txt" % key))
            if f in NEWS or f in EEST or f == "RR":
                z = R.npz_load(os.path.join(td, "rule_log_%s.npz" % key))
                zm = R.npz_load(os.path.join(td, "rule_log_mid_%s.npz" % key))
                if f in NEWS or f == "RR":
                    rule[(f, s)] = EC.log_fed(z) if z is not None else None
                    mid[(f, s)] = EC.log_fed(zm) if zm is not None else None
                    rows[(f, s)] = int(len(np.asarray(zm["ci"]))) if zm is not None else None
                if f in NEWS:
                    elogs[(f, s)] = {"rule": log_arrays(z), "mid": log_arrays(zm)}
                if f in NEWS or f in EEST:
                    stats[(f, s)] = join_inputs(log_input(z) if z is not None else None, log_input(zm) if zm is not None else None)
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is None:
                continue
            if f in NEWS:
                n_files[(f, s)] = int(raw["n_files"])
            idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
            if rep["missing"]:
                continue
            D = R.cut(raw, idx)
            L = H.label_windows(J, D["clip_index"], D["ws"])
            Cw[(f, s)] = np.asarray(H.window_cells(D["g"], D["cl"], L)["C"], np.float64)
            lev[(f, s)] = float(H.level(Cw[(f, s)]))
    ctl["r1_" + st] = ctl_r1s(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0, a.wrapper_sha, {(e, s): shas.get("surr_%s_s%d.npz" % (e, s)) for e in ESTS for s in SEEDS})
    ctl["x1_" + st] = ctl_x1s(rule, mid)
    ctl["m1_" + st] = ctl_m1s(rows)
    ctl["u2_" + st] = ctl_u2(tables, elogs)
    ctl["u3_" + st] = ctl_u3(stats, st)
    extra["u3_gate"] = ctl["u3_" + st]["gate_u3"]
    have = all((f, s) in Cw for f in FORMS for s in SEEDS)
    dl = max([abs(lev[(f, s)] - float(G["levels"][runs[f]]["C"][i])) for f in FORMS for i, s in enumerate(SEEDS)]) if have else float("inf")
    trips = []
    if have:
        for (a_, b_, c_) in [(n_, "RLc", "RR") for n_ in NEWS + EEST] + [("RRc", "RLc", "RR")]:
            trips.append({"ab": [lev[(a_, s)] - lev[(b_, s)] for s in SEEDS], "ac": [lev[(a_, s)] - lev[(c_, s)] for s in SEEDS], "cb": [lev[(c_, s)] - lev[(b_, s)] for s in SEEDS]})
    i1 = EC.ctl_i1(trips)
    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl)
    if S is None or cells_s is None or not all(v.get("PASS") is True for v in ctl.values()):
        return ctl, None, extra
    cnt = G["counts"]
    q = EC.qualified(cnt, H)
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
        strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pr["tick2"]["pos"])],
                     "reading": R.family_of(vs)}
        fam["STRATEGIC"] = strat[nm]["reading"]
        words[nm] = {"C": R.u_name(hw["C"]) if q else None, "U": R.u_name(hw["U"]) if q else None, "F": R.u_name(hw["F"]) if q else None, "fam": fam}
        pairs_out[nm] = {"A": A, "B": B, "headway": hw, "words": words[nm]}
    part = {"qualified": q, "counts": cnt, "pairs": pairs_out, "cells": cells_out, "strategic": strat, "words": words, "levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text)}
    return ctl, part, extra


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
    bank = R.jl(a.midlook_json)
    e_ok, e_reason, ej = read_e_json(a.estclose_json, R.jl)
    tables, shas = {}, {}
    for e in ESTS:
        for s in SEEDS:
            p = os.path.join(a.shuf_dir, "surr_%s_s%d.npz" % (e, s))
            if os.path.isfile(p):
                tables[(e, s)] = dict(np.load(p, allow_pickle=False))
                shas[os.path.basename(p)] = sha_file(p)
            else:
                tables[(e, s)] = None
    rec = R.jl(a.surrogate_record or os.path.join(a.shuf_dir, "SURROGATE_RECORD.json"))
    controls, parts, cache, gate = {}, {}, {}, {e: False for e in ESTS}
    controls["sur"] = ctl_sur(rec, shas)
    for st in SET:
        try:
            c, p, ex_ = run_setting(a, E, st, keys[st], bank, ej if e_ok else None, e_reason, tables, shas, cache, log)
            if st == "AGE":
                gate = ex_.get("u3_gate", gate)
        except Exception as ex:                                                               # noqa: BLE001
            c, p = {"setting_ran_" + st: {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:400])}}, None
        controls.update(c)
        parts[st] = p
    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)
    payload = None
    if all(v.get("PASS") is True for v in controls.values()) and all(parts.values()):
        words = {st: parts[st]["words"] for st in SET}
        names, why = all_names(words, {st: parts[st]["qualified"] for st in SET}, gate)
        payload = {"names": names, "why_not_written": why, "settings": parts, "u3_gate_at_age": gate}
    res = assemble(controls, payload)
    for line in EC.render(res):
        print(line)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-SHUFCLOSE10K rc 0 controls_all_pass %s" % res["controls_all_pass"])
    return 0


# ============================================================================================ selftest
def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-116s %s" % (name, bool(ok)))
    h = ctl_h1(AD.MiniRH, 300)
    chk("h1 planted name rule (LOWER, HIGHER, opposite seeds, no effect) through the stand-in engine: %s" % [(k, v["got"]) for k, v in h.items() if k != "PASS"], h["PASS"])
    WR = "aaaabbbbccccdddd"
    tabs = {(e, s): {"clip_index": np.repeat(np.arange(5), 6), "frame": np.tile(np.arange(30, 36), 5), "est": np.where(np.arange(30) % 7 == 0, np.nan, np.sin(np.arange(30) + 3 * s) * 2.0)} for e in ESTS for s in SEEDS}
    shas = {"surr_%s_s%d.npz" % (e, s): ("%02x" % (10 * s + len(e))) * 32 for e in ESTS for s in SEEDS}
    tsha = {(e, s): shas["surr_%s_s%d.npz" % (e, s)] for e in ESTS for s in SEEDS}
    t = ("arm I10RQ50eSN10KF5P30Hb20_s0 form RRe replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 ckpt sha256 ab "
         "wrapper sha256 " + WR + "0123456789abcdef0123456789abcdef started 2026-11-21T12:00:00Z\nest looks scored full 900 short 50 preroll full 300 short 200 file %s\nrc 0 finished 2026-11-21T12:04:00Z\n")
    texts = {(f, s): t % tsha[(f[len("RRe_S"):], s)] for f in NEWS for s in SEEDS}
    nf = {(f, s): 141 for f in NEWS for s in SEEDS}
    chk("r1 passes with the table hash in the est line; fails on another hash, a missing dump file, a pre-roll mismatch or another wrapper",
        ctl_r1s(texts, nf, 141, 30, WR, tsha)["PASS"] and not ctl_r1s(texts, nf, 141, 30, WR, {**tsha, ("N10", 0): "00" * 32})["PASS"] and not ctl_r1s(texts, {**nf, ("RRe_SL25", 1): 140}, 141, 30, WR, tsha)["PASS"]
        and not ctl_r1s(texts, nf, 141, 0, WR, tsha)["PASS"] and not ctl_r1s(texts, nf, 141, 30, "0" * 16, tsha)["PASS"])

    def zl(form, cs, ce=None, eo=None, pres=(True, True, False, True)):
        z = {"present": np.array(pres), "c_src": np.asarray(cs, np.float64), "form": np.asarray(form)}
        if eo is not None:
            z["est_ok"] = np.asarray(eo, bool)
            z["c_est"] = np.asarray(ce, np.float64)
        return EC.log_fed(z)
    zr, nz2, eo, ceo = [0.0] * 4, [0.0, -1.5, 0.0, 0.0], [True, True, True, False], [0.0, -1.5, 0.0, np.nan]
    rule = {(f, s): (zl("RR", zr) if f == "RR" else zl("RRe", nz2, ceo, eo)) for f in NEWS + ("RR",) for s in SEEDS}
    chk("x1 passes (new forms feed a closing, 0.0 without a full history, RR feeds 0)", ctl_x1s(rule, dict(rule))["PASS"])
    chk("x1 fails when a new form feeds 0 everywhere, feeds a closing without a full history, names another form, RR was fed, or a log is missing",
        not ctl_x1s({**rule, ("RRe_SN10", 0): zl("RRe", zr, ceo, eo)}, rule)["PASS"] and not ctl_x1s({**rule, ("RRe_SL25", 1): zl("RRe", [0.0, -1.5, 0.0, 0.2], ceo, eo)}, rule)["PASS"]
        and not ctl_x1s({**rule, ("RRe_SN10", 0): zl("RRh", nz2, ceo, eo)}, rule)["PASS"] and not ctl_x1s({**rule, ("RR", 0): zl("RR", [0.0, 0.0, 0.0, 0.1])}, rule)["PASS"]
        and not ctl_x1s(rule, {**rule, ("RRe_SN10", 0): None})["PASS"])
    rw = {(f, s): 900 for f in NEWS + ("RR",) for s in SEEDS}
    chk("m1 passes on equal row counts and fails on a different count or a missing log", ctl_m1s(rw)["PASS"] and not ctl_m1s({**rw, ("RRe_SN10", 0): 899})["PASS"] and not ctl_m1s({**rw, ("RR", 1): None})["PASS"])
    # u2
    ci, jj = np.repeat(np.arange(5), 6), np.tile(np.arange(30, 36), 5)
    kk = np.where(jj % 5 == 0, 0, 5)

    def mklog(s, shift=0.0, ok_all=True, tick=0, nan_to=None, drop_pattern=False):
        tv = tabs[("N10", s)]["est"]
        eo_ = np.isfinite(tv)
        if drop_pattern:
            eo_ = eo_.copy()
            eo_[3] = False
        c = np.where(eo_, tv + shift, np.nan if nan_to is None else nan_to)
        return {"ci": ci, "j": jj, "k": np.where(jj % 5 == 0, tick, tick + 5), "c_est": c, "est_ok": eo_}
    lg = {("RRe_S" + e, s): {"rule": mklog(s), "mid": mklog(s, ok_all=False, tick=2)} for e in ESTS for s in SEEDS}
    tabs2 = {(e, s): dict(tabs[("N10", s)]) for e in ESTS for s in SEEDS}
    chk("u2 passes: c_est equals the surrogate table at (ci, j) on est_ok rows, est_ok follows the table, first look of the mid-look log is its smallest tick (2)", ctl_u2(tabs2, lg)["PASS"])
    chk("u2 fails on a planted difference of 1e-6, passes one of 1e-10", not ctl_u2(tabs2, {**lg, ("RRe_SN10", 1): {"rule": mklog(1, shift=1e-6), "mid": lg[("RRe_SN10", 1)]["mid"]}})["PASS"]
        and ctl_u2(tabs2, {**lg, ("RRe_SN10", 1): {"rule": mklog(1, shift=1e-10), "mid": lg[("RRe_SN10", 1)]["mid"]}})["PASS"])
    chk("u2 fails when est_ok is False where the table has a value, when c_est is not NaN where est_ok is False, when a table or a log is missing, or the table has no such row",
        not ctl_u2(tabs2, {**lg, ("RRe_SL25", 0): {"rule": mklog(0, drop_pattern=True), "mid": lg[("RRe_SL25", 0)]["mid"]}})["PASS"]
        and not ctl_u2(tabs2, {**lg, ("RRe_SL25", 0): {"rule": lg[("RRe_SL25", 0)]["rule"], "mid": mklog(0, ok_all=False, nan_to=0.0, tick=2)}})["PASS"]
        and not ctl_u2({**tabs2, ("N10", 0): None}, lg)["PASS"] and not ctl_u2(tabs2, {**lg, ("RRe_SN10", 0): {"rule": None, "mid": None}})["PASS"]
        and not ctl_u2({**tabs2, ("N10", 0): dict(tabs2[("N10", 0)], frame=tabs2[("N10", 0)]["frame"] + 1000)}, lg)["PASS"])
    nofirst = mklog(0)
    nofirst["k"] = np.full(30, 5)
    chk("u2 fails when no compared row is a first-look row", not ctl_u2(tabs2, {**lg, ("RRe_SN10", 0): {"rule": nofirst, "mid": lg[("RRe_SN10", 0)]["mid"]}})["PASS"])
    # u3
    rg = np.random.default_rng(5)

    def st_of(sd, mean, share, n=400):
        v = rg.normal(mean, sd, n)
        okp = rg.uniform(size=n) < share
        return (v, okp, n + 50, int(okp.sum()) + 20)
    good = {(f, s): st_of(1.3, 0.02, 0.97) for f in NEWS + EEST for s in SEEDS}
    u3 = ctl_u3(good, "AGE")
    chk("u3 passes: sd ratio about 1, mean near 0, share 0.97 -> gate True for both estimators; the report holds the numbers", u3["PASS"] and u3["gate_u3"] == {"N10": True, "L25": True} and abs(u3["per_estimator"]["N10"]["sd_ratio"] - 1) < 0.2)
    chk("u3 gate False at sd ratio 1.5 (L25), at sd ratio 0.6 (N10), at |mean| 0.2, at share 0.85; the other estimator is still read",
        ctl_u3({**good, **{("RRe_SL25", s): st_of(1.3 * 1.5, 0.0, 0.97) for s in SEEDS}}, "AGE")["gate_u3"] == {"N10": True, "L25": False}
        and ctl_u3({**good, **{("RRe_SN10", s): st_of(1.3 * 0.6, 0.0, 0.97) for s in SEEDS}}, "AGE")["gate_u3"] == {"N10": False, "L25": True}
        and ctl_u3({**good, **{("RRe_SN10", s): st_of(1.3, 0.4, 0.97) for s in SEEDS}}, "AGE")["gate_u3"]["N10"] is False
        and ctl_u3({**good, **{("RRe_SL25", s): st_of(1.3, 0.0, 0.85) for s in SEEDS}}, "AGE")["gate_u3"]["L25"] is False)
    chk("u3 does not pass when a run is missing; at FRESH the numbers are reported and marked not judged", not ctl_u3({**good, ("RRe_SN10", 0): None}, "AGE")["PASS"] and ctl_u3(good, "FRESH")["per_estimator"]["N10"]["judged"] is False)
    chk("u3 gate False when the estimate input has no spread (sd 0)", ctl_u3({**good, **{("RRe_N10", s): (np.zeros(50), np.ones(50, bool), 50, 50) for s in SEEDS}}, "AGE")["gate_u3"]["N10"] is False)
    # sur
    rec = {"card": SUR_CARD, "pass": True, "u1": {"%s_s%d" % (e, s): {"passed": True} for e in ESTS for s in SEEDS}, "u4": {e: {"passed": True, "r_surrogate": 0.01} for e in ESTS}, "files": dict(shas)}
    chk("sur passes and carries u1 and u4", ctl_sur(rec, shas)["PASS"] and "u4_carried" in ctl_sur(rec, shas))
    chk("sur fails when the record says fail, u1 of one table failed, u4 of one estimator failed, a table hash differs, the record is missing or of another card",
        not ctl_sur(dict(rec, **{"pass": False}), shas)["PASS"] and not ctl_sur(dict(rec, u1=dict(rec["u1"], L25_s1={"passed": False})), shas)["PASS"]
        and not ctl_sur(dict(rec, u4=dict(rec["u4"], N10={"passed": False})), shas)["PASS"] and not ctl_sur(rec, dict(shas, **{"surr_N10_s0.npz": "00" * 32}))["PASS"]
        and not ctl_sur(None, shas)["PASS"] and not ctl_sur(dict(rec, card="X"), shas)["PASS"] and not ctl_sur(rec, {})["PASS"])
    # e json
    jl = lambda p: {"card": PARENT_CARD, "controls_all_pass": True} if p == "ok" else ({"card": "X", "controls_all_pass": True} if p == "other" else ({"card": PARENT_CARD, "controls_all_pass": False} if p == "bad" else None))   # noqa: E731
    chk("the read of card ADD79 is required: admitted only when present, of that card and passed", read_e_json("ok", jl)[0] and not read_e_json("", jl)[0] and not read_e_json("nope", jl)[0]
        and not read_e_json("other", jl)[0] and not read_e_json("bad", jl)[0])
    lv = {"RQ": {c: [0.1, 0.2] for c in ("U", "C", "F")}}
    bank = {"controls_all_pass": True, "settings": {"AGE": {"levels": {r: {c: list(v) for c, v in d.items()} for r, d in lv.items()}}}}
    chk("k3 (levels of the runs of card ADD79): equal passes, 1e-6 off fails, a bank that is not admitted fails", EC.ctl_a1(lv, bank, "AGE")["PASS"] and not EC.ctl_a1({"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2])}, bank, "AGE")["PASS"]
        and not EC.ctl_a1(lv, dict(bank, controls_all_pass=False), "AGE")["PASS"])

    # verdicts
    def wd(c, u, f="NOT-SEPARATED", fam=None):
        return {"C": c, "U": u, "F": f, "fam": dict({x: "NOT-SEPARATED" for x in FAMS}, **(fam or {}))}

    def W(est_vs_shuf, shuf_rr=None, e="N10"):
        base = wd("NOT-SEPARATED", "NOT-SEPARATED")
        d = {}
        for ee in ESTS:
            d["EST-VS-SHUF-" + ee] = est_vs_shuf if ee == e else base
            d["SHUF-VS-RR-" + ee] = (shuf_rr or base) if ee == e else base
            d["SHUF-VS-NO-RULE-" + ee] = base
        return {"AGE": d, "FRESH": dict(d)}
    both = {"N10": True, "L25": True}
    n1, y1 = verdict_names(W(wd("LOWER", "LOWER")), both)
    chk("verdict: C LOWER and U not HIGHER reads GAIN-FROM-THE-SCENE with the SHUF-VS-RR name beside and the N10 mark",
        n1["VERDICT_N10"] == "SHUFCLOSE10K-N10-GAIN-FROM-THE-SCENE" and n1["VERDICT_N10_BESIDE_SHUF_VS_RR"] == "SHUFCLOSE10K-SHUF-VS-RR-N10-AGE-C-NOT-SEPARATED" and n1["VERDICT_N10_MARK"] == MARK_N10
        and n1["VERDICT_L25"] == "SHUFCLOSE10K-L25-GAIN-NOT-SHOWN-FROM-THE-SCENE" and n1["VERDICT_L25_MARK"] == MARK_L25)
    chk("verdict: U NOT-SEPARATED also counts as not HIGHER", verdict_names(W(wd("LOWER", "NOT-SEPARATED")), both)[0]["VERDICT_N10"].endswith("-GAIN-FROM-THE-SCENE"))
    nn = [verdict_names(W(w_, sr), both)[0] for w_, sr in ((wd("NOT-SEPARATED", "LOWER"), None), (wd("HIGHER", "LOWER"), None), (wd("LOWER", "HIGHER"), None), (wd("NOT-SEPARATED", "NOT-SEPARATED"), wd("LOWER", "LOWER")))]
    chk("verdict: NOT-SHOWN when C is not LOWER or U is HIGHER; the SHUF-VS-RR name beside says whether the spread alone lowers contact (C LOWER)",
        all(x["VERDICT_N10"] == "SHUFCLOSE10K-N10-GAIN-NOT-SHOWN-FROM-THE-SCENE" for x in nn) and nn[3]["VERDICT_N10_BESIDE_SHUF_VS_RR"] == "SHUFCLOSE10K-SHUF-VS-RR-N10-AGE-C-LOWER"
        and nn[0]["VERDICT_N10_BESIDE_SHUF_VS_RR"].endswith("-NOT-SEPARATED"))
    n5, y5 = verdict_names(W(wd("LOWER", "LOWER"), e="L25"), both)
    chk("verdict L25 (the estimator whose gain row 141 separated) carries its own mark; N10 is NOT-SHOWN there", n5["VERDICT_L25"].endswith("-L25-GAIN-FROM-THE-SCENE") and n5["VERDICT_L25_MARK"] == MARK_L25 and n5["VERDICT_N10_MARK"] == MARK_N10)
    n6, y6 = verdict_names(W(wd("LOWER", "LOWER")), {"N10": False, "L25": True})
    chk("verdict: an estimator whose u3 gate fails has no verdict, no beside name, no mark and says why; the other is read", "VERDICT_N10" not in n6 and "VERDICT_N10_MARK" not in n6 and "VERDICT_N10" in y6 and "VERDICT_L25" in n6)
    n7, y7 = verdict_names(W(wd(None, None)), both)
    chk("verdict: not written when C and U are not available (unqualified headway), says why", "VERDICT_N10" not in n7 and "VERDICT_N10" in y7)
    n8, y8 = verdict_names(W(wd("NOT-SEPARATED", None)), both)
    chk("verdict: a failing C condition decides NOT-SHOWN even when U is not available; C LOWER with U not available is not written",
        n8["VERDICT_N10"].endswith("-GAIN-NOT-SHOWN-FROM-THE-SCENE") and "VERDICT_N10" not in verdict_names(W(wd("LOWER", None)), both)[0])
    n9, y9 = verdict_names(W(wd("LOWER", "LOWER"), wd(None, None)), both)
    chk("verdict: with SHUF-VS-RR not available the verdict is written and the beside name is not", "VERDICT_N10" in n9 and "VERDICT_N10_BESIDE_SHUF_VS_RR" not in n9 and "VERDICT_N10_BESIDE" in y9)
    full = {st: {pn: wd("LOWER", "NOT-SEPARATED", "HIGHER") for pn in PAIRS} for st in SET}
    nm, why = all_names(full, {"AGE": True, "FRESH": True}, both)
    chk("names: pattern SHUFCLOSE10K-<PAIR>-<SETTING>-<C|U|F|family>-<word>, C, U, F and four families per pair and setting (6 pairs x 2 settings)",
        nm["EST-VS-SHUF-N10_AGE_C"] == "SHUFCLOSE10K-EST-VS-SHUF-N10-AGE-C-LOWER" and nm["SHUF-VS-NO-RULE-L25_FRESH_F"] == "SHUFCLOSE10K-SHUF-VS-NO-RULE-L25-FRESH-F-HIGHER"
        and nm["SHUF-VS-RR-L25_AGE_STRATEGIC"] == "SHUFCLOSE10K-SHUF-VS-RR-L25-AGE-STRATEGIC-NOT-SEPARATED" and len([k for k in nm if k.endswith("_U")]) == 12 and len([k for k in nm if k.endswith("_F")]) == 12
        and len([k for k in nm if k.endswith("_LATERAL")]) == 12 and len(PAIRS) == 6)
    nm2, why2 = all_names(full, {"AGE": True, "FRESH": True}, {"N10": True, "L25": False})
    chk("names: AGE names of an estimator whose u3 gate fails are not written (and its verdict), its FRESH names are", not [k for k in nm2 if k.startswith("SHUF-VS-RR-L25_AGE") or k.startswith("EST-VS-SHUF-L25_AGE")]
        and "EST-VS-SHUF-L25_FRESH_C" in nm2 and "AGE_EST-VS-SHUF-L25" in why2 and "VERDICT_L25" not in nm2)
    nm3, why3 = all_names({st: {pn: wd(None, None, None) for pn in PAIRS} for st in SET}, {"AGE": False, "FRESH": False}, both)
    chk("names: with no qualified setting no C, U or F name is written and the output says why", not [k for k in nm3 if k.endswith(("_C", "_U", "_F"))] and "HEADWAY_AGE" in why3 and "HEADWAY_FRESH" in why3)
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}})
    chk("a failed control leaves no name and prints none", a_["names"] is None and a_["controls_all_pass"] is False and "SHUFCLOSE10K-" not in "\n".join(EC.render(a_)) and "== names" not in "\n".join(EC.render(a_)))
    chk("pairs: EST-VS-SHUF is RRe(E) minus RRe(surrogate), SHUF-VS-RR is surrogate minus RR, SHUF-VS-NO-RULE is surrogate minus no rule",
        all(PAIRS["EST-VS-SHUF-" + e] == ("RRe_" + e, "RRe_S" + e) and PAIRS["SHUF-VS-RR-" + e] == ("RRe_S" + e, "RR") and PAIRS["SHUF-VS-NO-RULE-" + e] == ("RRe_S" + e, "R0K") for e in ESTS))
    reg = dict(R.ALL_RUNS)
    register_runs(SET["AGE"])
    register_runs(SET["FRESH"])
    chk("register_runs enters the new runs with form RRe, readout K, rule every 5, pre-roll of the setting", R.ALL_RUNS["RQ50eSN10KF5P30Hb20"][:3] == ("RRe", "K", 30) and R.ALL_RUNS["RQ50eSL25KF5Hb20"][2] == 0
        and R.ALL_RUNS["RQ50eSL25KF5Hb20"][6] == 5 and sorted(SET["AGE"]["runs"].values()).count("RQ50eSL25KF5P30Hb20") == 1)
    R.ALL_RUNS.clear()
    R.ALL_RUNS.update(reg)
    chk("the run names of the card: SL25, SN10 and the runs of card ADD79 at both settings", SET["AGE"]["runs"]["RRe_SL25"] == "RQ50eSL25KF5P30Hb20" and SET["FRESH"]["runs"]["RRe_SN10"] == "RQ50eSN10KF5Hb20"
        and SET["AGE"]["runs"]["RRe_N10"] == "RQ50eN10KF5P30Hb20" and SET["FRESH"]["runs"]["RRe_L25"] == "RQ50eL25KF5Hb20")
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [x for x in bad if x in src])
    ok = all(res)
    print("ZZ SELFTEST-SHUFCLOSE10K ok: %s (%d of %d)" % (ok, sum(res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "fresh-grid", "midlook-json", "estclose-json", "shuf-dir", "w", "tree", "out", "wrapper-sha"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--surrogate-record", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
