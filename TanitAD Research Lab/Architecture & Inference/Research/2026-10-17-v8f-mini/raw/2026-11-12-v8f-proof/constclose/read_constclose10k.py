#!/usr/bin/env python
"""read_constclose10k.py -- read of card PREREG_V8F_PROOF_ADD80 (CONSTCLOSE10K): the reader-fed lead rule with a CONSTANT closing input (form RRe, estimator files of history one frame, values C10 = 1.0, C20 = 2.0,
C40 = 4.0 m/s), at age and fresh. Derived from read_estclose10k.py (card ADD79), which is imported and not edited; under it the engine modules (read_handsrc_decomp, read_closrc10k, read_longroll10k).

Two settings, each read completely before the next starts. New runs (both seeds): AGE RQ50eC10KF5P30Hb20, RQ50eC20KF5P30Hb20, RQ50eC40KF5P30Hb20; FRESH RQ50eC10KF5Hb20, RQ50eC20KF5Hb20, RQ50eC40KF5Hb20.
Banked (row 137): RRh, RRc, RR, RLc, R0K (levels only for RRh and RLc: control a1); and, only with --estclose-json that says controls_all_pass true, the runs RRe(N10), RRe(L25) of card ADD79.
Pairs per value c (C10, C20, C40) and setting, A minus B (card section 3): CONST-VS-RR-c (RRe(c), RR), CONST-VS-LABEL-CLOSING-c (RRe(c), RRc), CONST-VS-NO-RULE-c (RRe(c), R0K), and with the admitted
read of card ADD79 CONST-VS-EST-E-c (RRe(c), RRe(E)) for E in N10, L25. Names, F name beside every verdict, the verdicts C<cc> and E (BEYOND / NOT-BEYOND A CONSTANT) as card section 3.
Controls (card section 4), printed first: k0 (files), k2 (the estimate column equals c; coverage of full histories, bar 0.99), k3 (levels of the runs of card ADD79 equal its file; only with the pairs), x1, m1, g, s, a1, r1, i1, h1.
A control that does not pass: no name and no value is written or printed. An absent or not admitted read of card ADD79 is not a failed control: the pairs with E and the verdicts of E are not written, the reason is printed.
Usage: python read_constclose10k.py --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --const-dir DIR --w W --tree TREE --out OUT.json --wrapper-sha HEX16
          [--estclose-json READ_ESTCLOSE10K.json] [--n-boot 2000]   |   --selftest
Last line: ZZ READ-CONSTCLOSE10K rc <rc> controls_all_pass <bool>
"""
import argparse
import hashlib
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("CC_ESTREAD_DIR"), os.path.join(_HERE, "..", "estclose_read"), _HERE):
    if _p and os.path.isfile(os.path.join(_p, "read_estclose10k.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import read_estclose10k as EC           # noqa: E402

HS, CR, AD, M, R, np = EC.HS, EC.CR, EC.AD, EC.M, EC.R, EC.np
CARD, PRE, PARENT_CARD = "PREREG_V8F_PROOF_ADD80", "CONSTCLOSE10K", "PREREG_V8F_PROOF_ADD79"
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
FAMS, NOT_HIGHER = EC.FAMS, EC.NOT_HIGHER
VALUES = ("C10", "C20", "C40")
CVAL = {"C10": 1.0, "C20": 2.0, "C40": 4.0}
ESTS = ("N10", "L25")
BANKED = EC.BANKED
NEWC = tuple("RRe_" + c for c in VALUES)
NEWE = tuple("RRe_" + e for e in ESTS)
TOL_K2 = 1e-6
COV_BAR = 0.99
SETTING_ORDER = ("AGE", "FRESH")
INTERNAL = EC.INTERNAL


def const_file(c):
    return "const_%s_H1.npz" % c


def settings(e_on):
    """run table per setting: key (form) -> run name; the runs of card ADD79 are in it only when its read is admitted"""
    out = {}
    for st, P, grid, suf in (("AGE", 30, "30", "P30Hb20"), ("FRESH", 0, "full", "Hb20")):
        runs = {"RRe_" + c: "RQ50e%sKF5%s" % (c, suf) for c in VALUES}
        if e_on:
            runs.update({"RRe_" + e: "RQ50e%sKF5%s" % (e, suf) for e in ESTS})
        runs.update({f: EC.SETTINGS[st]["runs"][f] for f in BANKED})
        out[st] = {"P": P, "grid": grid, "spread": "R0Hb20", "runs": runs}
    return out


def forms_of(e_on):
    return NEWC + (NEWE if e_on else ()) + BANKED


def pairs_of(e_on):
    p = {}
    for c in VALUES:
        p["CONST-VS-RR-" + c] = ("RRe_" + c, "RR")
        p["CONST-VS-LABEL-CLOSING-" + c] = ("RRe_" + c, "RRc")
        p["CONST-VS-NO-RULE-" + c] = ("RRe_" + c, "R0K")
        if e_on:
            for e in ESTS:
                p["CONST-VS-EST-%s-%s" % (e, c)] = ("RRe_" + c, "RRe_" + e)
    return p


FORM_OF = dict(EC.FORM_OF)
FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES})


# ============================================================================================ names and verdicts
def cname(pair, setting, kind, word):
    return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)


def no_worse(fam):
    return bool(all(fam[f] != "WORSE" for f in FAMS))


def nh(word):
    return None if word is None else bool(word in NOT_HIGHER)


def const_conditions(words, c):
    """the conditions of the verdict of value c (card section 3): True / False / None (None = the word is not available)"""
    pr = "CONST-VS-RR-" + c
    a, f = words["AGE"][pr], words["FRESH"][pr]
    return [("AGE C LOWER", None if a["C"] is None else a["C"] == "LOWER"), ("AGE U not HIGHER", nh(a["U"])), ("AGE no family WORSE", no_worse(a["fam"])),
            ("FRESH C not HIGHER", nh(f["C"])), ("FRESH U not HIGHER", nh(f["U"])), ("FRESH no family WORSE", no_worse(f["fam"]))]


def const_verdicts(words, gate):
    """-> (names, why). A value has a verdict only when both settings are named (coverage gate); a condition that already fails decides its name without a missing word; a missing word otherwise
    writes no name. The F name of the pair (AGE and FRESH) is written beside every verdict that is written. A single SUPPORTED of the three is marked one of three."""
    names, why, supported = {}, {}, []
    for c in VALUES:
        key = "VERDICT_" + c
        notnamed = [st for st in SETTING_ORDER if not gate.get(st)]
        if notnamed:
            why[key] = "not written: the share of looks with a full history over the scored window is below %.2f at %s" % (COV_BAR, ",".join(notnamed))
            continue
        conds = const_conditions(words, c)
        if any(v is False for _, v in conds):
            names[key] = "%s-%s-NOT-SUPPORTED" % (PRE, c)
        elif any(v is None for _, v in conds):
            why[key] = "not written: the headway instrument is not qualified in a setting this verdict needs (%s)" % ", ".join(n for n, v in conds if v is None)
            continue
        else:
            names[key] = "%s-%s-SUPPORTED" % (PRE, c)
            supported.append(c)
        for st in SETTING_ORDER:
            fw = words[st]["CONST-VS-RR-" + c]["F"]
            if fw is not None:
                names["%s_F_%s" % (key, st)] = cname("CONST-VS-RR-" + c, st, "F", fw)
            else:
                why["%s_F_%s" % (key, st)] = "F name not written: the headway instrument is not qualified at %s" % st
    if len(supported) == 1:
        names["VERDICT_ONE_OF_THREE"] = "%s-%s-SUPPORTED-ONE-OF-THREE" % (PRE, supported[0])
    return names, why


def e_matches(words, e):
    """per value c: True when pair CONST-VS-EST-E-c at AGE reads C not HIGHER and F not HIGHER (NOT-SEPARATED counts as a match); None when a word is not available"""
    out = {}
    for c in VALUES:
        wd = words["AGE"]["CONST-VS-EST-%s-%s" % (e, c)]
        a, b = nh(wd["C"]), nh(wd["F"])
        out[c] = None if a is None or b is None else bool(a and b)
    return out


def e_verdicts(words, gate, e_on, e_reason=""):
    names, why = {}, {}
    if not e_on:
        why["E_VERDICTS"] = "the pairs with the estimators and the verdicts of E are not written: %s (this is not a failed control)" % e_reason
        return names, why
    for e in ESTS:
        key = "VERDICT_" + e
        if not gate.get("AGE"):
            why[key] = "not written: the share of looks with a full history over the scored window is below %.2f at AGE" % COV_BAR
            continue
        m = e_matches(words, e)
        if any(v is True for v in m.values()):
            names[key] = "%s-%s-NOT-BEYOND-A-CONSTANT" % (PRE, e)
        elif all(v is False for v in m.values()):
            names[key] = "%s-%s-BEYOND-A-CONSTANT" % (PRE, e)
        else:
            why[key] = "not written: the headway instrument is not qualified at AGE and no value matches %s" % e
    return names, why


def pair_names(setting, pair, wd):
    out = {}
    for kind in ("C", "U", "F"):
        if wd[kind] is not None:
            out["%s_%s_%s" % (pair, setting, kind)] = cname(pair, setting, kind, wd[kind])
    for f in FAMS:
        out["%s_%s_%s" % (pair, setting, f)] = cname(pair, setting, f, wd["fam"][f])
    return out


def all_names(words, quals, gate, e_on, e_reason=""):
    names, why = {}, {}
    for st in SETTING_ORDER:
        if not quals[st]:
            why["HEADWAY_" + st] = "U, C and F of the setting %s are not reported: the headway instrument is not qualified on its grid" % st
        if not gate.get(st):
            why["COVERAGE_" + st] = "names of the setting %s are not written: the share of looks with a full history over the scored window is below %.2f" % (st, COV_BAR)
            continue
        for pair in words[st]:
            names.update(pair_names(st, pair, words[st][pair]))
    vn, vw = const_verdicts(words, gate)
    en, ew = e_verdicts(words, gate, e_on, e_reason)
    names.update(vn)
    names.update(en)
    why.update(vw)
    why.update(ew)
    return names, why


# ============================================================================================ controls
def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ctl_k0(const_dir, texts, CL, seeds=SEEDS, n=32):
    """each constant file loaded by closest.py returns its value for any input of shape [1, 3] and [n, 1, 3], needs 5 frames, has H 1; the sha256 of the file equals the sha256 in the run line of every run of
    the value (both seeds, this setting)"""
    out, ok = {}, bool(texts)
    rng = np.random.default_rng(0)
    for c in VALUES:
        p = os.path.join(const_dir, const_file(c))
        rec = {"ok": False}
        try:
            E = CL.ClosingEstimator.load(p)
            vals = [E.apply(rng.normal(0, 7.0, (1, 3)))[0], *E.apply(rng.normal(0, 7.0, (n, 1, 3)))]
            sh = sha_file(p)
            lines = {s: EC.parse_est_looks(texts.get(("RRe_" + c, s))) for s in seeds}
            good = bool(E.H == 1 and E.need_frames() == 5 and all(v == CVAL[c] for v in vals) and all(t is not None and t["file_sha256"] == sh for t in lines.values()))
            rec = {"ok": good, "H": E.H, "need_frames": E.need_frames(), "all_values_equal_c": bool(all(v == CVAL[c] for v in vals)), "sha256": sh, "run_line_sha256_match": {str(s): bool(t is not None and t["file_sha256"] == sh) for s, t in lines.items()}}
        except Exception as ex:                                                                # noqa: BLE001
            rec = {"ok": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
        out[c] = rec
        ok = ok and rec["ok"]
    return {"PASS": bool(ok), "files": out}


def log_arrays_c(z):
    """arrays of the interface of a rule log of a constant run (None when the log or a key is missing); form is the string of the rule log, '' for a later-look log without it"""
    if z is None:
        return None
    try:
        d = {k: np.asarray(z[k]) for k in ("ci", "j", "k", "c_est", "est_ok", "c_src", "present")}
    except Exception:                                                                          # noqa: BLE001
        return None
    try:
        d["form"] = str(np.asarray(z["form"]).reshape(-1)[0])
    except Exception:                                                                          # noqa: BLE001
        d["form"] = ""
    return d


def ctl_k2(elogs, seeds=SEEDS):
    """elogs: {(RRe_c, seed): {"rule": arrays or None, "mid": arrays or None}}. In all runs and in both logs: on the rows with est_ok True the estimate column c_est equals c to 1e-6 (finite, at least one row,
    rows of the first replan tick -- k == 0 -- among them), it is NaN where est_ok is False. (Replan ticks and the second look: both logs.)"""
    out, ok, worst = {}, bool(elogs), 0.0
    for c in VALUES:
        for s in seeds:
            for tag in ("rule", "mid"):
                z = (elogs.get(("RRe_" + c, s)) or {}).get(tag)
                nm = "%s:seed%d:%s_log" % (c, s, tag)
                if z is None:
                    out[nm], ok = {"ok": False, "why": "log missing"}, False
                    continue
                eo = np.asarray(z["est_ok"], bool)
                ce = np.asarray(z["c_est"], np.float64)
                d = np.abs(ce[eo] - CVAL[c])
                bad = ~np.isfinite(d) | ~(d <= TOL_K2)
                n_first = int(((np.asarray(z["k"]) == 0) & eo).sum())
                nan_ok = bool(np.all(np.isnan(ce[~eo])))
                mx = float(np.max(d[~bad])) if (~bad).any() else None
                good = bool(eo.any() and not bad.any() and n_first > 0 and nan_ok)
                out[nm] = {"ok": good, "rows_compared": int(eo.sum()), "rows_first_tick": n_first, "rows_failing": int(bad.sum()), "nan_where_not_ok": nan_ok, "largest_difference": mx, "tolerance": TOL_K2}
                ok = ok and good
                worst = max(worst, mx or 0.0)
    return {"PASS": bool(ok), "largest_difference_overall": worst, "runs": out}


def ctl_cov(cov, texts, st, seeds=SEEDS):
    """cov: {(RRe_c, seed): {"n": looks logged, "ok": est_ok looks} or None} (both logs together); texts: run texts. Per value and for the setting pooled: share of full histories over the logged looks and over the
    scored window of the run text; the pre-roll counts are reported. PASS = every number is there. gate = the pooled share of the setting is at least 0.99 by the logs and by the run text (the setting is named only then)."""
    per, N = {}, dict(n=0, k=0, sf=0, ss=0, pf=0, ps=0)
    have_all = True
    for c in VALUES:
        n = k = sf = ss = pf = ps = 0
        have = True
        for s in seeds:
            cv = cov.get(("RRe_" + c, s))
            t = EC.parse_est_looks(texts.get(("RRe_" + c, s)))
            if not cv or not cv["n"] or t is None:
                have = False
                continue
            n, k = n + cv["n"], k + cv["ok"]
            sf, ss, pf, ps = sf + t["scored_full"], ss + t["scored_short"], pf + t["preroll_full"], ps + t["preroll_short"]
        per[c] = {"complete": bool(have and n and (sf + ss)), "share_logged_looks": k / n if have and n else None, "share_scored_window_run_text": sf / (sf + ss) if have and (sf + ss) else None,
                  "share_preroll_run_text": pf / (pf + ps) if have and (pf + ps) else None, "looks_logged": n, "scored_full": sf, "scored_short": ss, "preroll_full": pf, "preroll_short": ps}
        have_all = have_all and per[c]["complete"]
        for key, v in zip(("n", "k", "sf", "ss", "pf", "ps"), (n, k, sf, ss, pf, ps)):
            N[key] += v
    sl = N["k"] / N["n"] if have_all and N["n"] else None
    stx = N["sf"] / (N["sf"] + N["ss"]) if have_all and (N["sf"] + N["ss"]) else None
    gate = bool(have_all and sl is not None and stx is not None and sl >= COV_BAR and stx >= COV_BAR)
    return {"PASS": bool(have_all), "per_value": per, "pooled_share_logged_looks": sl, "pooled_share_scored_window_run_text": stx, "gate_pooled_share_at_least_0.99": gate, "setting": st}


def ctl_r1c(texts, n_files, n_clips, preroll, wrapper, seeds=SEEDS):
    out, ok = {}, bool(texts)
    for f in NEWC:
        for s in seeds:
            p = CR.parse_run_text(texts.get((f, s)))
            good = EC.run_text_ok(p, preroll, wrapper) and n_files.get((f, s)) == n_clips
            out["%s:seed%d" % (f, s)] = {"ok": bool(good), "parsed": p, "dump_files": n_files.get((f, s)), "clips_of_the_grid": n_clips}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_x1c(consts, rr_rule, rr_mid, seeds=SEEDS):
    """consts: {(RRe_c, seed): {"rule": arrays, "mid": arrays}} (log_arrays_c); rr_rule, rr_mid: {("RR", seed): log_fed record}. A constant run: on the present rows with est_ok the closing column c_src equals c
    (1e-6), on the present rows without a full history it is exactly 0.0, at least one present row with est_ok in each log, the form of the rule log is RRe. The banked RR run feeds 0 on every row of both logs."""
    out, ok = {}, bool(consts) and bool(rr_rule) and bool(rr_mid)
    for c in VALUES:
        for s in seeds:
            res, good = {}, True
            for tag in ("rule", "mid"):
                z = (consts.get(("RRe_" + c, s)) or {}).get(tag)
                if z is None:
                    res[tag], good = {"ok": False, "why": "no log"}, False
                    continue
                pr, eo = np.asarray(z["present"], bool), np.asarray(z["est_ok"], bool)
                cs = np.asarray(z["c_src"], np.float64)
                full, short = pr & eo, pr & ~eo
                g1 = bool(full.any() and np.all(np.abs(cs[full] - CVAL[c]) <= TOL_K2) and np.all(cs[short] == 0.0))
                if tag == "rule":
                    g1 = g1 and z.get("form") == "RRe"
                res[tag] = {"ok": g1, "present_full": int(full.sum()), "present_short": int(short.sum()), "form": z.get("form")}
                good = good and g1
            out["%s:seed%d" % (c, s)] = dict(res, ok=bool(good))
            ok = ok and good
    for s in seeds:
        res, good = {}, True
        for tag, lg in (("rule_log", rr_rule), ("mid_log", rr_mid)):
            r = lg.get(("RR", s))
            g1 = bool(r and r["rows"] > 0 and r["present_nonzero"] == 0 and r["max_abs_closing"] == 0.0 and r["closing_finite_where_present"])
            res[tag] = {"ok": g1}
            good = good and g1
        out["RR:seed%d" % s] = dict(res, ok=bool(good))
        ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_m1c(rows, seeds=SEEDS):
    """rows: {(form, seed): later-look row count}; each constant run equals the banked RR run of the same seed and has rows"""
    out, ok = {}, bool(rows)
    for s in seeds:
        base = rows.get(("RR", s))
        for f in NEWC:
            n = rows.get((f, s))
            good = bool(n and base and n == base)
            out["%s:seed%d" % (f, s)] = {"ok": good, "rows": n, "rows_banked_RR": base}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def ctl_h1(RH, n_boot=300, n_ep=40, per=10, seed=11, sd=0.3):
    """planted name rule through the engine and the name builder of this reader (C, U and F share the interval rule)"""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    inv = np.repeat(np.arange(n_ep), per)
    out, ok = {}, True
    for case, eff, want in (("lower", (-1.0, -1.0), "LOWER"), ("higher", (1.0, 1.0), "HIGHER"), ("opposite_seeds", (-1.0, 1.0), "NOT-SEPARATED"), ("no_effect", (0.0, 0.0), "NOT-SEPARATED")):
        ys = [rng.normal(0.0, sd, n) for _ in SEEDS]
        xs = [y + e + rng.normal(0.0, sd, n) for y, e in zip(ys, eff)]
        got = R.u_name(HS.pair(RH, xs, ys, inv, n_ep, n_boot))
        nm = cname("CONST-VS-RR-C20", "AGE", "F", got)
        out[case] = {"want": want, "got": got, "name": nm}
        ok = ok and got == want and nm == cname("CONST-VS-RR-C20", "AGE", "F", want)
    return dict(out, PASS=bool(ok))


def assemble(controls, payload):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "why_not_written": None, "settings": None}
    if res["controls_all_pass"]:
        res.update(payload)
    return res


def read_e_json(path, jl):
    """-> (e_on, reason, json or None). The read of card ADD79 is admitted only when the file is there, is the result of that card and says controls_all_pass true."""
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


def run_setting(a, E, st, cfg, keys, bank, ej, e_on, ctx_cache, log, CL):
    """-> (controls with the setting as suffix, part or None, extra)"""
    H, Q3, RH = E
    runs = cfg["runs"]
    FORMS = forms_of(e_on)
    PAIRS = pairs_of(e_on)
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
    if e_on:
        ctl["k3_" + st] = EC.ctl_a1({runs[f]: G["levels"][runs[f]] for f in NEWE}, ej, st)
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
    texts, n_files, rr_rule, rr_mid, consts, rows, Cw, lev, cov = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for f in FORMS:
        for s in SEEDS:
            run_ = runs[f]
            key = R.key_of(run_, s)
            td = R.tick_dir(a.pull, run_, s)
            if f in NEWC:
                texts[(f, s)] = R.txt(os.path.join(td, "run_%s.txt" % key))
            if f in NEWC or f == "RR":
                z = R.npz_load(os.path.join(td, "rule_log_%s.npz" % key))
                zm = R.npz_load(os.path.join(td, "rule_log_mid_%s.npz" % key))
                rows[(f, s)] = int(len(np.asarray(zm["ci"]))) if zm is not None else None
                if f == "RR":
                    rr_rule[(f, s)] = EC.log_fed(z) if z is not None else None
                    rr_mid[(f, s)] = EC.log_fed(zm) if zm is not None else None
                else:
                    la, lm = log_arrays_c(z), log_arrays_c(zm)
                    consts[(f, s)] = {"rule": la, "mid": lm}
                    cov[(f, s)] = {"n": int(len(la["est_ok"]) + len(lm["est_ok"])), "ok": int(np.asarray(la["est_ok"], bool).sum() + np.asarray(lm["est_ok"], bool).sum())} if la is not None and lm is not None else None
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is None:
                continue
            if f in NEWC:
                n_files[(f, s)] = int(raw["n_files"])
            idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
            if rep["missing"]:
                continue
            D = R.cut(raw, idx)
            L = H.label_windows(J, D["clip_index"], D["ws"])
            Cw[(f, s)] = np.asarray(H.window_cells(D["g"], D["cl"], L)["C"], np.float64)
            lev[(f, s)] = float(H.level(Cw[(f, s)]))
    ctl["k0_" + st] = ctl_k0(a.const_dir, texts, CL)
    ctl["r1_" + st] = ctl_r1c(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0, a.wrapper_sha)
    ctl["x1_" + st] = ctl_x1c(consts, rr_rule, rr_mid)
    ctl["m1_" + st] = ctl_m1c(rows)
    ctl["k2_" + st] = ctl_k2(consts)
    ctl["cov_" + st] = ctl_cov(cov, texts, st)
    extra["named"] = bool(ctl["cov_" + st]["gate_pooled_share_at_least_0.99"])
    have = all((f, s) in Cw for f in FORMS for s in SEEDS)
    dl = max([abs(lev[(f, s)] - float(G["levels"][runs[f]]["C"][i])) for f in FORMS for i, s in enumerate(SEEDS)]) if have else float("inf")
    trips = []
    if have:
        for (a_, b_, c_) in [(n_, "RLc", "RR") for n_ in NEWC + (NEWE if e_on else ())] + [("RRc", "RLc", "RR")]:
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


def run(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    gf = R.jl(a.fresh_grid)
    if g30 is None or gf is None:
        print("STOP: grid_p30.json in %s or the pinned grid %s is missing" % (a.grid_dir, a.fresh_grid))
        return 3
    E = R.engine(a)
    H = E[0]
    CL = find_closest(a)
    J = H.load_join()
    names_ = [str(c) for c in J["clip"]]
    keys = {"AGE": R.grid_keys(g30, names_), "FRESH": R.grid_keys(gf, names_)}
    bank = R.jl(a.midlook_json)
    e_on, e_reason, ej = read_e_json(getattr(a, "estclose_json", None), R.jl)
    if not e_on:
        print("NOTE: the pairs with the estimators and the verdicts of E are not written: %s (this is not a failed control)" % e_reason)
    SET = settings(e_on)
    controls, parts, cache, gate = {}, {}, {}, {st: False for st in SET}
    for st in SET:
        try:
            c, p, ex_ = run_setting(a, E, st, SET[st], keys[st], bank, ej, e_on, cache, log, CL)
            gate[st] = bool(ex_.get("named", False))
        except Exception as ex:                                                               # noqa: BLE001
            c, p = {"setting_ran_" + st: {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:400])}}, None
        controls.update(c)
        parts[st] = p
    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)
    payload = None
    if all(v.get("PASS") is True for v in controls.values()) and all(parts.values()):
        words = {st: parts[st]["words"] for st in SET}
        names, why = all_names(words, {st: parts[st]["qualified"] for st in SET}, gate, e_on, e_reason)
        payload = {"names": names, "why_not_written": why, "settings": parts, "coverage_gate": gate, "estclose_pairs_written": e_on}
    res = assemble(controls, payload)
    for line in EC.render(res):
        print(line)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-CONSTCLOSE10K rc 0 controls_all_pass %s" % res["controls_all_pass"])
    return 0


def find_closest(a):
    for d in (os.environ.get("CLOSEST_DIR", ""), getattr(a, "code", ""), os.path.join(_HERE, "..", "closeread"), _HERE):
        if d and os.path.isfile(os.path.join(d, "closest.py")):
            if os.path.abspath(d) not in sys.path:
                sys.path.insert(0, os.path.abspath(d))
            import closest
            return closest
    raise SystemExit("REFUSED: closest.py not found (set CLOSEST_DIR)")


# ============================================================================================ selftest
def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-116s %s" % (name, bool(ok)))
    chk("the values and pre-rolls of the card: 1.0, 2.0, 4.0 m/s; AGE pre-roll 30 grid 30, FRESH pre-roll 0 full grid; coverage bar 0.99; tolerance of k2 1e-6",
        VALUES == ("C10", "C20", "C40") and CVAL == {"C10": 1.0, "C20": 2.0, "C40": 4.0} and settings(False)["AGE"]["P"] == 30 and settings(False)["AGE"]["grid"] == "30" and settings(False)["FRESH"]["P"] == 0
        and settings(False)["FRESH"]["grid"] == "full" and COV_BAR == 0.99 and TOL_K2 == 1e-6)
    h = ctl_h1(AD.MiniRH, 300)
    chk("h1 planted name rule (LOWER, HIGHER, opposite seeds, no effect) through the stand-in engine: %s" % [(k, v["got"]) for k, v in h.items() if k != "PASS"], h["PASS"])

    def wd(c, u, f="NOT-SEPARATED", fam=None):
        return {"C": c, "U": u, "F": f, "fam": dict({x: "NOT-SEPARATED" for x in FAMS}, **(fam or {}))}
    NS = wd("NOT-SEPARATED", "NOT-SEPARATED")

    def W(c, age, fresh, e_pairs=None, extra=None):
        """words of the settings: pair CONST-VS-RR-c has age / fresh, the other pairs of the card are NOT-SEPARATED; e_pairs {(E, c): word at AGE}"""
        w = {st: {} for st in SETTING_ORDER}
        for v in VALUES:
            for st in SETTING_ORDER:
                for fam in ("CONST-VS-RR-", "CONST-VS-LABEL-CLOSING-", "CONST-VS-NO-RULE-"):
                    w[st][fam + v] = NS
                for e in ESTS:
                    w[st]["CONST-VS-EST-%s-%s" % (e, v)] = NS
        w["AGE"]["CONST-VS-RR-" + c], w["FRESH"]["CONST-VS-RR-" + c] = age, fresh
        for (e, v), word in (e_pairs or {}).items():
            w["AGE"]["CONST-VS-EST-%s-%s" % (e, v)] = word
        for (st, pair), word in (extra or {}).items():
            w[st][pair] = word
        return w
    both = {"AGE": True, "FRESH": True}
    # --- verdict C<cc>
    n1, y1 = const_verdicts(W("C20", wd("LOWER", "LOWER", "HIGHER"), wd("NOT-SEPARATED", "NOT-SEPARATED", "LOWER", {"LATERAL": "MIXED"})), both)
    chk("verdict C20 SUPPORTED (AGE C LOWER, FRESH C not HIGHER, MIXED is not WORSE), C10 and C40 NOT-SUPPORTED, the single SUPPORTED is marked one of three, the F names are beside",
        n1["VERDICT_C20"] == "CONSTCLOSE10K-C20-SUPPORTED" and n1["VERDICT_C10"] == "CONSTCLOSE10K-C10-NOT-SUPPORTED" and n1["VERDICT_C40"] == "CONSTCLOSE10K-C40-NOT-SUPPORTED"
        and n1["VERDICT_ONE_OF_THREE"] == "CONSTCLOSE10K-C20-SUPPORTED-ONE-OF-THREE" and n1["VERDICT_C20_F_AGE"] == "CONSTCLOSE10K-CONST-VS-RR-C20-AGE-F-HIGHER"
        and n1["VERDICT_C20_F_FRESH"] == "CONSTCLOSE10K-CONST-VS-RR-C20-FRESH-F-LOWER" and n1["VERDICT_C10_F_AGE"].endswith("-F-NOT-SEPARATED"))
    lo = wd("LOWER", "LOWER")
    bads = {"AGE C not LOWER": (wd("NOT-SEPARATED", "LOWER"), lo), "FRESH C HIGHER": (lo, wd("HIGHER", "LOWER")), "AGE U HIGHER": (wd("LOWER", "HIGHER"), lo), "FRESH U HIGHER": (lo, wd("LOWER", "HIGHER")),
            "AGE family WORSE": (wd("LOWER", "LOWER", fam={"TACTICAL": "WORSE"}), lo), "FRESH family WORSE": (lo, wd("LOWER", "LOWER", fam={"STRATEGIC": "WORSE"}))}
    chk("verdict NOT-SUPPORTED for each of the six failing conditions: %s" % sorted(bads), all(const_verdicts(W("C40", *v), both)[0]["VERDICT_C40"] == "CONSTCLOSE10K-C40-NOT-SUPPORTED" for v in bads.values()))
    nall, _ = const_verdicts(W("C10", lo, lo, extra={(st, "CONST-VS-RR-" + c): lo for st in SETTING_ORDER for c in VALUES}), both)
    chk("verdict: all three SUPPORTED names three and no one-of-three mark", [nall["VERDICT_" + c] for c in VALUES] == ["CONSTCLOSE10K-%s-SUPPORTED" % c for c in VALUES] and "VERDICT_ONE_OF_THREE" not in nall)
    two = {(st, "CONST-VS-RR-" + c): lo for st in SETTING_ORDER for c in ("C10", "C20")}
    ntwo, _ = const_verdicts(W("C40", NS, NS, extra=two), both)
    chk("verdict: two SUPPORTED, no one-of-three mark", ntwo["VERDICT_C10"].endswith("-SUPPORTED") and ntwo["VERDICT_C20"].endswith("-SUPPORTED") and "VERDICT_ONE_OF_THREE" not in ntwo)
    n3, y3 = const_verdicts(W("C20", lo, lo), {"AGE": True, "FRESH": False})
    chk("verdict: a setting below the coverage bar gives no verdict of any value and says why", not [k for k in n3 if k.startswith("VERDICT")] and all("VERDICT_" + c in y3 for c in VALUES))
    n4, y4 = const_verdicts(W("C20", wd(None, None, None), lo), both)
    chk("verdict: an unqualified setting writes no name that needs its C or U", "VERDICT_C20" not in n4 and "VERDICT_C20" in y4)
    n5, y5 = const_verdicts(W("C20", wd(None, None, None, {"LATERAL": "WORSE"}), lo), both)
    chk("verdict: a failing family condition still decides without the missing words, F beside is not written and says why", n5["VERDICT_C20"].endswith("-NOT-SUPPORTED") and "VERDICT_C20_F_AGE" in y5 and "VERDICT_C20_F_AGE" not in n5)
    # --- verdicts of E
    allns = {(e, c): wd("NOT-SEPARATED", "NOT-SEPARATED", "NOT-SEPARATED") for e in ESTS for c in VALUES}
    hi = wd("HIGHER", "NOT-SEPARATED", "NOT-SEPARATED")
    hiF = wd("LOWER", "NOT-SEPARATED", "HIGHER")
    ne, _ = e_verdicts(W("C20", NS, NS, allns), both, True)
    chk("E: NOT-SEPARATED on both measures counts as a match: NOT-BEYOND-A-CONSTANT for both estimators", ne["VERDICT_N10"] == "CONSTCLOSE10K-N10-NOT-BEYOND-A-CONSTANT" and ne["VERDICT_L25"] == "CONSTCLOSE10K-L25-NOT-BEYOND-A-CONSTANT")
    none = {(e, c): (hi if c != "C40" else hiF) for e in ESTS for c in VALUES}
    nb, _ = e_verdicts(W("C20", NS, NS, none), both, True)
    chk("E: no value matches (C HIGHER at two values, F HIGHER at one): BEYOND-A-CONSTANT", nb["VERDICT_N10"] == "CONSTCLOSE10K-N10-BEYOND-A-CONSTANT" and nb["VERDICT_L25"] == "CONSTCLOSE10K-L25-BEYOND-A-CONSTANT")
    one = {**none, ("N10", "C20"): wd("LOWER", "NOT-SEPARATED", "LOWER")}
    n1m, _ = e_verdicts(W("C20", NS, NS, one), both, True)
    chk("E: one matching value is enough: NOT-BEYOND for N10 and BEYOND for L25", n1m["VERDICT_N10"].endswith("-N10-NOT-BEYOND-A-CONSTANT") and n1m["VERDICT_L25"].endswith("-L25-BEYOND-A-CONSTANT"))
    unq = {**none, ("N10", "C20"): wd(None, None, None)}
    nu, yu = e_verdicts(W("C20", NS, NS, unq), both, True)
    chk("E: an unavailable word and no match writes no name and says why; an unavailable word beside a match still names", "VERDICT_N10" not in nu and "VERDICT_N10" in yu
        and e_verdicts(W("C20", NS, NS, {**unq, ("N10", "C10"): wd("LOWER", "LOWER", "LOWER")}), both, True)[0]["VERDICT_N10"].endswith("-NOT-BEYOND-A-CONSTANT"))
    ng, yg = e_verdicts(W("C20", NS, NS, allns), {"AGE": False, "FRESH": True}, True)
    chk("E: AGE below the coverage bar writes no E verdict", not ng and "VERDICT_N10" in yg)
    nf, yf = e_verdicts(W("C20", NS, NS, allns), both, False, "no --estclose-json was given")
    chk("E: pairs not written (read of card ADD79 absent or not admitted): no E verdict, the reason is given", not nf and "no --estclose-json" in yf["E_VERDICTS"] and "not a failed control" in yf["E_VERDICTS"])
    # --- admission of the file of card ADD79
    jl_ = lambda d: (lambda p: d.get(p))                                                       # noqa: E731
    chk("estclose file: absent path / unreadable / other card / not all-pass are not admitted (no failed control); all-pass of card ADD79 is admitted",
        not read_e_json("", jl_({}))[0] and not read_e_json("x", jl_({}))[0] and not read_e_json("x", jl_({"x": {"card": "OTHER", "controls_all_pass": True}}))[0]
        and not read_e_json("x", jl_({"x": {"card": PARENT_CARD, "controls_all_pass": False}}))[0] and read_e_json("x", jl_({"x": {"card": PARENT_CARD, "controls_all_pass": True}}))[0])
    # --- names
    full = {st: {pn: wd("LOWER", "NOT-SEPARATED", "HIGHER") for pn in pairs_of(True)} for st in SETTING_ORDER}
    nm, why = all_names(full, {"AGE": True, "FRESH": True}, both, True)
    chk("names: pattern CONSTCLOSE10K-<PAIR>-<SETTING>-C-<word>, U, F and the four families per pair and setting (15 pairs x 2 settings)",
        nm["CONST-VS-RR-C20_AGE_C"] == "CONSTCLOSE10K-CONST-VS-RR-C20-AGE-C-LOWER" and nm["CONST-VS-EST-L25-C40_FRESH_F"] == "CONSTCLOSE10K-CONST-VS-EST-L25-C40-FRESH-F-HIGHER"
        and nm["CONST-VS-NO-RULE-C10_AGE_STRATEGIC"] == "CONSTCLOSE10K-CONST-VS-NO-RULE-C10-AGE-STRATEGIC-NOT-SEPARATED" and len(pairs_of(True)) == 15 and len(pairs_of(False)) == 9
        and len([k for k in nm if k.endswith("_U")]) == 30 and len([k for k in nm if k.endswith("_F")]) == 30 and len([k for k in nm if k.endswith("_LATERAL")]) == 30)
    chk("names: pair table of the card (A minus B) with and without the estimators",
        pairs_of(False)["CONST-VS-RR-C10"] == ("RRe_C10", "RR") and pairs_of(False)["CONST-VS-LABEL-CLOSING-C20"] == ("RRe_C20", "RRc") and pairs_of(False)["CONST-VS-NO-RULE-C40"] == ("RRe_C40", "R0K")
        and pairs_of(True)["CONST-VS-EST-N10-C20"] == ("RRe_C20", "RRe_N10") and not [k for k in pairs_of(False) if "EST" in k])
    nm2, why2 = all_names({st: {pn: wd("LOWER", "NOT-SEPARATED", "HIGHER") for pn in pairs_of(False)} for st in SETTING_ORDER}, {"AGE": True, "FRESH": True}, both, False, "no --estclose-json was given")
    chk("names: without the pairs of E no EST name and no E verdict is written and the reason is in the output", not [k for k in nm2 if "EST" in k] and "E_VERDICTS" in why2 and "VERDICT_C20" in nm2)
    nm3, why3 = all_names({st: {pn: wd("LOWER", "NOT-SEPARATED", "HIGHER") for pn in pairs_of(True)} for st in SETTING_ORDER}, {"AGE": True, "FRESH": True}, {"AGE": True, "FRESH": False}, True)
    chk("names: the names of a setting below the coverage bar are not written (FRESH), the others are; the output says why", not [k for k in nm3 if "_FRESH_" in k] and any("_AGE_" in k for k in nm3) and "COVERAGE_FRESH" in why3)
    nm4, why4 = all_names({st: {pn: wd(None, None, None) for pn in pairs_of(True)} for st in SETTING_ORDER}, {"AGE": False, "FRESH": False}, both, True)
    chk("names: with no qualified setting no C, U or F name is written and the output says why; the families are named",
        not [k for k in nm4 if k.endswith("_C") or k.endswith("_U") or k.endswith("_F")] and "HEADWAY_AGE" in why4 and "HEADWAY_FRESH" in why4 and any(k.endswith("_TACTICAL") for k in nm4))
    # --- assemble / render
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}})
    shown = "\n".join(EC.render(a_))
    chk("a failed control leaves no name and prints no name and no value", a_["names"] is None and a_["controls_all_pass"] is False and "CONSTCLOSE10K-" not in shown and "== names" not in shown and "no name and no value" in shown)
    chk("assemble with all controls passed carries the payload", assemble({"x": {"PASS": True}}, {"names": {"k": "v"}})["names"] == {"k": "v"} and not assemble({"x": {"PASS": True}}, None)["controls_all_pass"])
    # --- k0
    import tempfile
    CL = find_closest(argparse.Namespace(code=""))
    d = tempfile.mkdtemp(prefix="tmp_cr_", dir=_HERE)
    for c in VALUES:
        CL.save_planted(os.path.join(d, const_file(c)), "const", 1, CVAL[c])
    WR = "aaaabbbbccccdddd"

    def txt(c, sha, kind="est"):
        return ("arm I10RQ50eCCCKF5P30Hb20_s0 form RRe replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 ckpt sha256 ab "
                "wrapper sha256 " + WR + "0123456789abcdef0123456789abcdef started 2026-11-21T12:00:00Z\nest looks scored full 990 short 10 preroll full 300 short 200 file ").replace("CCC", c) + sha + "\nrc 0 finished 2026-11-21T12:04:00Z\n"
    texts = {("RRe_" + c, s): txt(c, sha_file(os.path.join(d, const_file(c)))) for c in VALUES for s in SEEDS}
    chk("k0 passes: files return c on [1,3] and [n,1,3], H 1, 5 frames, the run lines hold the file hashes", ctl_k0(d, texts, CL)["PASS"])
    bad_t = dict(texts)
    bad_t[("RRe_C20", 1)] = txt("C20", "0" * 64)
    chk("k0 fails on a run line with another file hash, a missing line, a missing file", not ctl_k0(d, bad_t, CL)["PASS"] and not ctl_k0(d, {**texts, ("RRe_C10", 0): "x"}, CL)["PASS"] and not ctl_k0(d + "_none", texts, CL)["PASS"])
    CL.save_planted(os.path.join(d, "w.npz"), "const", 1, 2.5)
    d2 = tempfile.mkdtemp(prefix="tmp_cr_", dir=_HERE)
    for c in VALUES:
        CL.save_planted(os.path.join(d2, const_file(c)), "const", 1, CVAL[c] + (0.5 if c == "C40" else 0.0))
    t2 = {("RRe_" + c, s): txt(c, sha_file(os.path.join(d2, const_file(c)))) for c in VALUES for s in SEEDS}
    chk("k0 fails on a file whose value is not c (planted 4.5 for C40)", not ctl_k0(d2, t2, CL)["PASS"])
    # --- k2, cov, x1, m1, r1
    n = 40
    ci = np.repeat(np.arange(4), 10)
    jj = np.tile(np.arange(10, 20), 4)
    kk = np.where(jj % 5 == 0, 0, 5)

    def mk(c, shift=0.0, short=0, nan_bad=False, k=kk, form="RRe", cs_shift=0.0, pres=None):
        eo = np.ones(n, bool)
        eo[:short] = False
        ce = np.where(eo, CVAL[c] + shift, np.nan)
        if nan_bad:
            ce = np.where(eo, ce, 0.0)
        pr = np.ones(n, bool) if pres is None else pres
        return {"ci": ci, "j": jj, "k": k, "c_est": ce, "est_ok": eo, "c_src": np.where(eo, CVAL[c] + cs_shift, 0.0), "present": pr, "form": form}
    el = {("RRe_" + c, s): {"rule": mk(c, short=2), "mid": mk(c)} for c in VALUES for s in SEEDS}
    chk("k2 passes when c_est equals c on the est_ok rows of both logs of all runs and is NaN elsewhere", ctl_k2(el)["PASS"])
    chk("k2 fails on a planted 1e-4 difference, passes at 1e-7", not ctl_k2({**el, ("RRe_C20", 0): {"rule": mk("C20", shift=1e-4), "mid": mk("C20")}})["PASS"] and ctl_k2({**el, ("RRe_C20", 0): {"rule": mk("C20", shift=1e-7), "mid": mk("C20")}})["PASS"])
    chk("k2 fails on the value of another c (2.0 in the C10 run), a non-NaN estimate without a full history, a missing log, no first-tick row",
        not ctl_k2({**el, ("RRe_C10", 1): {"rule": mk("C20"), "mid": mk("C10")}})["PASS"] and not ctl_k2({**el, ("RRe_C40", 0): {"rule": mk("C40", short=2, nan_bad=True), "mid": mk("C40")}})["PASS"]
        and not ctl_k2({**el, ("RRe_C10", 0): {"rule": None, "mid": None}})["PASS"] and not ctl_k2({**el, ("RRe_C20", 0): {"rule": mk("C20", k=np.full(n, 5)), "mid": mk("C20")}})["PASS"])
    cov = {k: {"n": 1000, "ok": 995} for k in el}
    cg = ctl_cov(cov, texts, "AGE")
    chk("cov: gate True at 0.995 (logs) and 0.99 (run text 990 of 1000), the shares and the pre-roll share are reported", cg["PASS"] and cg["gate_pooled_share_at_least_0.99"] and abs(cg["per_value"]["C10"]["share_preroll_run_text"] - 0.6) < 1e-12)
    chk("cov: gate False below 0.99 by the logs (0.98) or by the run text (980 of 1000), and False when a run line or log is missing", not ctl_cov({k: {"n": 1000, "ok": 980} for k in el}, texts, "AGE")["gate_pooled_share_at_least_0.99"]
        and not ctl_cov(cov, {k: v.replace("scored full 990 short 10", "scored full 980 short 20") for k, v in texts.items()}, "AGE")["gate_pooled_share_at_least_0.99"]
        and not ctl_cov(cov, {**texts, ("RRe_C10", 0): "x"}, "AGE")["PASS"] and not ctl_cov({**cov, ("RRe_C40", 1): None}, texts, "AGE")["PASS"])
    zr = {"rows": 10, "present_nonzero": 0, "max_abs_closing": 0.0, "closing_finite_where_present": True}
    rr = {("RR", s): zr for s in SEEDS}
    chk("x1 passes: closing equals c on present rows with a full history, 0.0 without, form RRe; RR feeds 0", ctl_x1c(el, rr, rr)["PASS"])
    chk("x1 fails on a closing of 1.9 for C20, a closing that is not 0 without a full history, another form, RR fed a closing, a missing log",
        not ctl_x1c({**el, ("RRe_C20", 0): {"rule": mk("C20", cs_shift=-0.1), "mid": mk("C20")}}, rr, rr)["PASS"]
        and not ctl_x1c({**el, ("RRe_C10", 0): {"rule": dict(mk("C10", short=2), c_src=np.full(n, 1.0)), "mid": mk("C10")}}, rr, rr)["PASS"]
        and not ctl_x1c({**el, ("RRe_C40", 1): {"rule": mk("C40", form="RRh"), "mid": mk("C40")}}, rr, rr)["PASS"]
        and not ctl_x1c(el, {("RR", 0): dict(zr, present_nonzero=2), ("RR", 1): zr}, rr)["PASS"] and not ctl_x1c({**el, ("RRe_C10", 1): {"rule": None, "mid": None}}, rr, rr)["PASS"])
    rw = {(f, s): 900 for f in NEWC + ("RR",) for s in SEEDS}
    chk("m1 passes on equal later-look row counts and fails on a different count or an empty log", ctl_m1c(rw)["PASS"] and not ctl_m1c({**rw, ("RRe_C10", 0): 899})["PASS"] and not ctl_m1c({**rw, ("RR", 1): None})["PASS"])
    nf = {(f, s): 141 for f in NEWC for s in SEEDS}
    t = {(f, s): txt(f[4:], "ab" * 32) for f in NEWC for s in SEEDS}
    t0 = {k: v.replace("preroll 30", "preroll 0") for k, v in t.items()}
    chk("r1 passes at AGE and (pre-roll 0) FRESH and fails on a missing dump file, a pre-roll mismatch, another form or rule interval",
        ctl_r1c(t, nf, 141, 30, WR)["PASS"] and ctl_r1c(t0, nf, 141, 0, WR)["PASS"] and not ctl_r1c(t, {**nf, ("RRe_C40", 1): 140}, 141, 30, WR)["PASS"] and not ctl_r1c(t, nf, 141, 0, WR)["PASS"]
        and not ctl_r1c({**t, ("RRe_C10", 0): t[("RRe_C10", 0)].replace("rule every 5", "rule every 10")}, nf, 141, 30, WR)["PASS"] and not ctl_r1c({**t, ("RRe_C20", 0): t[("RRe_C20", 0)].replace("form RRe", "form RR")}, nf, 141, 30, WR)["PASS"])
    # --- k3 and a1 (the levels of the runs of card ADD79 and of the banked forms equal the files)
    lv = {"RQ": {c: [0.1, 0.2] for c in ("U", "C", "F")}, "R0": {c: [0.3, 0.4] for c in ("U", "C", "F")}}
    ej = {"controls_all_pass": True, "settings": {"AGE": {"levels": {r: {c: list(v) for c, v in dd.items()} for r, dd in lv.items()}}}}
    chk("k3 (same function as a1): equal levels pass; 1e-6 off, a run missing from the file, an all-pass flag False each fail",
        EC.ctl_a1(lv, ej, "AGE")["PASS"] and not EC.ctl_a1({"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2]), "R0": lv["R0"]}, ej, "AGE")["PASS"] and not EC.ctl_a1({**lv, "RX": lv["RQ"]}, ej, "AGE")["PASS"]
        and not EC.ctl_a1(lv, dict(ej, controls_all_pass=False), "AGE")["PASS"])
    tr = {"ab": [0.10, 0.07], "ac": [0.04, 0.05], "cb": [0.06, 0.02]}
    chk("i1 holds on a triple that adds up and fails when off by 1e-6", EC.ctl_i1([tr])["PASS"] and not EC.ctl_i1([dict(tr, ab=[0.10 + 1e-6, 0.07])])["PASS"])
    # --- run tables
    S_ = settings(True)
    chk("run names of the card: AGE and FRESH, C10 C20 C40 (and N10 L25 when admitted), the banked forms of row 137",
        S_["AGE"]["runs"]["RRe_C20"] == "RQ50eC20KF5P30Hb20" and S_["FRESH"]["runs"]["RRe_C40"] == "RQ50eC40KF5Hb20" and S_["AGE"]["runs"]["RRe_N10"] == "RQ50eN10KF5P30Hb20" and S_["FRESH"]["runs"]["RRe_L25"] == "RQ50eL25KF5Hb20"
        and "RRe_N10" not in settings(False)["AGE"]["runs"] and S_["AGE"]["runs"]["RR"] == "RQ50KF5P30Hb20" and S_["FRESH"]["runs"]["R0K"] == "R0KHb20" and S_["AGE"]["P"] == 30 and S_["FRESH"]["P"] == 0)
    saved = dict(R.ALL_RUNS)
    register_runs(S_["AGE"])
    register_runs(S_["FRESH"])
    chk("register_runs enters the constant runs with form RRe, readout K, rule every 5, pre-roll of the setting", R.ALL_RUNS["RQ50eC10KF5P30Hb20"][:3] == ("RRe", "K", 30) and R.ALL_RUNS["RQ50eC40KF5Hb20"][2] == 0 and R.ALL_RUNS["RQ50eC40KF5Hb20"][6] == 5)
    R.ALL_RUNS.clear()
    R.ALL_RUNS.update(saved)
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [x for x in bad if x in src])
    ok = all(res)
    print("ZZ SELFTEST-CONSTCLOSE10K ok: %s (%d of %d) temporary folders %s %s" % (ok, sum(res), len(res), d, d2))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "fresh-grid", "midlook-json", "const-dir", "w", "tree", "out", "wrapper-sha"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--estclose-json", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
