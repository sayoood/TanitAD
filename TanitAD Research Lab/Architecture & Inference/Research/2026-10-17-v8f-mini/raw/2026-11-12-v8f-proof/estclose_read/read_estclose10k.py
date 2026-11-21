#!/usr/bin/env python
"""read_estclose10k.py -- read of card PREREG_V8F_PROOF_ADD79 (ESTCLOSE10K): the lead rule fed with a closing speed estimated from the reader's own outputs over a history (form RRe), at age and fresh.
Derived from read_midlook10k.py (card ADD76); the engine modules (read_handsrc_decomp, read_closrc10k and under them read_longroll10k with its notes) are imported and not edited.

Two settings, each read completely (grid read, strategic read, window arrays, logs) before the next starts (the grid read of the note-N4 module clears its start-point registry):
  AGE    keys of grid_p30.json, pre-roll 30. New runs RRe(N10) RQ50eN10KF5P30Hb20, RRe(L25) RQ50eL25KF5P30Hb20. Banked: RRh RQ50hKF5P30Hb20, RRc RQ50cKF5P30Hb20, RR RQ50KF5P30Hb20, RLc RLcKF5P30Hb20, no rule R0KP30Hb20; spread R0Hb20.
  FRESH  the pinned grid (--fresh-grid), pre-roll 0. New runs RQ50eN10KF5Hb20, RQ50eL25KF5Hb20. Banked: RQ50hKF5Hb20, RQ50cKF5Hb20, RQ50KF5Hb20, RLcKF5Hb20, R0KHb20; spread R0Hb20.
Pairs per estimator E (N10, L25) and setting, A minus B: EST-CLOSING-E (RRe, RR), EST-VS-LABEL-CLOSING-E (RRe, RRc), EST-VS-LABEL-FED-E (RRe, RLc), EST-VS-NO-RULE-E (RRe, no rule). Names and the two verdicts as card section 4.
Controls (card section 5), printed first: g, s, a1, r1, x1, m1, i1, e3 per setting, e2 at FRESH, h1. A control that does not pass: no name and no value is written or printed.
Interface assumed of the wrapper: rule logs rule_log_<key>.npz and rule_log_mid_<key>.npz hold ci, j, k, c_est (float64, NaN where no full history), est_ok (bool); the run text holds the line
  est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>
Usage: python read_estclose10k.py --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --est-dir DIR --w W --tree TREE --out OUT.json --wrapper-sha HEX16 [--n-boot 2000]   |   --selftest
Last line: ZZ READ-ESTCLOSE10K rc <rc> controls_all_pass <bool>
"""
import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("ECR_INTEG_DIR"), os.path.join(_HERE, "..", "integ10k"), _HERE):
    if _p and os.path.isfile(os.path.join(_p, "read_handsrc_decomp.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import read_handsrc_decomp as HS     # noqa: E402
import read_closrc10k as CR          # noqa: E402

AD, M, R, N4, np = HS.AD, HS.M, HS.R, HS.N4, HS.np
CARD, PRE = "PREREG_V8F_PROOF_ADD79", "ESTCLOSE10K"
SEEDS = R.SEEDS
TIER_LINE = M.TIER_LINE
TOL_A1, TOL_I1, TOL_E2 = 1e-9, 1e-12, 1e-3
COV_BAR = 0.90
ESTS = ("N10", "L25")
NEW = tuple("RRe_" + e for e in ESTS)
FORMS = NEW + ("RRh", "RRc", "RR", "RLc", "R0K")
BANKED = ("RRh", "RRc", "RR", "RLc", "R0K")
SETTINGS = {
    "AGE": {"P": 30, "grid": "30", "spread": "R0Hb20", "runs": {"RRe_N10": "RQ50eN10KF5P30Hb20", "RRe_L25": "RQ50eL25KF5P30Hb20", "RRh": "RQ50hKF5P30Hb20", "RRc": "RQ50cKF5P30Hb20", "RR": "RQ50KF5P30Hb20",
                                                                              "RLc": "RLcKF5P30Hb20", "R0K": "R0KP30Hb20"}},
    "FRESH": {"P": 0, "grid": "full", "spread": "R0Hb20", "runs": {"RRe_N10": "RQ50eN10KF5Hb20", "RRe_L25": "RQ50eL25KF5Hb20", "RRh": "RQ50hKF5Hb20", "RRc": "RQ50cKF5Hb20", "RR": "RQ50KF5Hb20",
                                                                      "RLc": "RLcKF5Hb20", "R0K": "R0KHb20"}},
}
FORM_OF = {"RRe_N10": "RRe", "RRe_L25": "RRe", "RRh": "RRh", "RRc": "RRc", "RR": "RR", "RLc": "RLc", "R0K": "R0"}
PAIRS, PAIR_EST = {}, {}
for _e in ESTS:
    for _n, _b in (("EST-CLOSING", "RR"), ("EST-VS-LABEL-CLOSING", "RRc"), ("EST-VS-LABEL-FED", "RLc"), ("EST-VS-NO-RULE", "R0K")):
        PAIRS["%s-%s" % (_n, _e)] = ("RRe_" + _e, _b)
        PAIR_EST["%s-%s" % (_n, _e)] = _e
INTERNAL = {"RR-VS-LABEL-FED": ("RR", "RLc"), "RR-VS-NO-RULE": ("RR", "R0K"), "LABEL-CLOSING": ("RRc", "RR")}
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NOT_HIGHER = ("LOWER", "NOT-SEPARATED")
EST_RE = re.compile(r"est looks scored full (\d+) short (\d+) preroll full (\d+) short (\d+) file ([0-9a-f]{64})")


# ============================================================================================ names and verdicts
def cname(pair, setting, kind, word):
    return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)


def qualified(cnt, H):
    return bool(cnt["n_lead_windows"] >= H.MIN_WINDOWS and cnt["n_episodes_with_lead"] >= H.MIN_EPISODES)


def no_worse(fam):
    return bool(all(fam[f] != "WORSE" for f in FAMS))


def est_conditions(words, e):
    """the conditions of the verdict of estimator e (card section 4): True / False / None (None = the word is not available)"""
    def w(st, kind):
        return words[st]["EST-CLOSING-" + e][kind]

    def nh(word):
        return None if word is None else bool(word in NOT_HIGHER)
    return [("AGE C LOWER", None if w("AGE", "C") is None else w("AGE", "C") == "LOWER"), ("AGE U not HIGHER", nh(w("AGE", "U"))), ("AGE no family WORSE", no_worse(words["AGE"]["EST-CLOSING-" + e]["fam"])),
            ("FRESH C not HIGHER", nh(w("FRESH", "C"))), ("FRESH U not HIGHER", nh(w("FRESH", "U"))), ("FRESH no family WORSE", no_worse(words["FRESH"]["EST-CLOSING-" + e]["fam"]))]


def verdict_names(words, age_ok):
    """words[setting][pair] = {"C", "U", "fam"}; age_ok[e]: the AGE read of e is admitted by coverage (e3). -> (names, why). An estimator whose AGE coverage is below the bar has no verdict;
    a name that needs a word of a setting that is not qualified is not written; a condition that already fails decides its name without the missing word."""
    names, why, supported = {}, {}, []
    for e in ESTS:
        key = "VERDICT_" + e
        if not age_ok.get(e):
            why[key] = "not written: the share of looks with a full history over the scored window at AGE is below %.2f for %s" % (COV_BAR, e)
            continue
        conds = est_conditions(words, e)
        if any(v is False for _, v in conds):
            names[key] = "%s-%s-NOT-SUPPORTED" % (PRE, e)
        elif any(v is None for _, v in conds):
            why[key] = "not written: the headway instrument is not qualified in a setting this verdict needs (%s)" % ", ".join(n for n, v in conds if v is None)
        else:
            names[key] = "%s-%s-SUPPORTED" % (PRE, e)
            supported.append(e)
    if len(supported) == 1:
        names["VERDICT_ONE_OF_TWO"] = "%s-%s-SUPPORTED-ONE-OF-TWO" % (PRE, supported[0])
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


def all_names(words, quals, age_ok):
    names, why = {}, {}
    for st in SETTINGS:
        if not quals[st]:
            why["HEADWAY_" + st] = "U, C and F of the setting %s are not reported: the headway instrument is not qualified on its grid" % st
        for pair in PAIRS:
            if st == "AGE" and not age_ok.get(PAIR_EST[pair]):
                why["AGE_%s" % pair] = "AGE names of %s are not written: coverage of full histories over the scored window is below %.2f" % (PAIR_EST[pair], COV_BAR)
                continue
            names.update(pair_names(st, pair, words[st][pair]))
    vn, vw = verdict_names(words, age_ok)
    names.update(vn)
    why.update(vw)
    return names, why


def share_of_gap(pooled_est, pooled_label):
    """arithmetic on the pooled pair values of C at AGE: gap = C(RR) - C(RRc) = -pooled(LABEL-CLOSING); removed = C(RR) - C(RRe) = -pooled(EST-CLOSING-E); share = removed / gap. Names nothing."""
    gap, removed = -float(pooled_label), -float(pooled_est)
    if not gap > 0:
        return {"gap": gap, "removed": removed, "share": None, "note": "the label closing speed does not lower contact against RR at AGE: no share"}
    return {"gap": gap, "removed": removed, "share": removed / gap, "note": None}


# ============================================================================================ controls
def run_text_ok(p, preroll, wrapper):
    return bool(p and p["form"] == "RRe" and p["replan_steps"] == 10 and p["rule_every"] == 5 and p["readout"] == "causal_g" and p["kappa"] == "0.2" and p["preroll"] == preroll
                and p["wrapper"] == wrapper and p["rc0"])


def parse_est_looks(text):
    m = EST_RE.search(text or "")
    if not m:
        return None
    a, b, c, d = (int(m.group(i)) for i in (1, 2, 3, 4))
    return {"scored_full": a, "scored_short": b, "preroll_full": c, "preroll_short": d, "file_sha256": m.group(5)}


def ctl_r1(texts, n_files, n_clips, preroll, wrapper):
    out, ok = {}, bool(texts)
    for f in NEW:
        for s in SEEDS:
            p = CR.parse_run_text(texts.get((f, s)))
            good = run_text_ok(p, preroll, wrapper) and n_files.get((f, s)) == n_clips
            out["%s:seed%d" % (f, s)] = {"ok": bool(good), "parsed": p, "dump_files": n_files.get((f, s)), "clips_of_the_grid": n_clips}
            ok = ok and good
    return {"PASS": bool(ok), "runs": out}


def log_fed(z):
    """a rule log or a later-look log -> closing_fed record plus c_read agreement (where the log holds c_read) and, where it holds est_ok, the fallback check: closing exactly 0.0 on present rows without a full history"""
    r = CR.closing_fed(z)
    try:
        cr = np.asarray(z["c_read"], np.float64)
        cs = np.asarray(z["c_src"], np.float64)
        pr = np.asarray(z["present"], bool)
        r["c_read_diff_present"] = float(np.max(np.abs(cs[pr] - cr[pr]))) if pr.any() and np.isfinite(cr[pr]).all() else None
    except Exception:                                                                         # noqa: BLE001
        r["c_read_diff_present"] = None
    try:
        eo = np.asarray(z["est_ok"], bool)
        cs = np.asarray(z["c_src"], np.float64)
        pr = np.asarray(z["present"], bool)
        ce = np.asarray(z["c_est"], np.float64)
        fb = pr & ~eo
        r["est_ok_present"] = int((pr & eo).sum())
        r["fallback_rows"] = int(fb.sum())
        r["fallback_closing_exactly_zero"] = bool(np.all(cs[fb] == 0.0))
        both = pr & eo & np.isfinite(ce)
        r["c_src_vs_c_est_present_reported"] = float(np.max(np.abs(cs[both] - ce[both]))) if both.any() else None
    except Exception:                                                                         # noqa: BLE001
        r["est_ok_present"] = None
    return r


def ctl_x1(rule, mid):
    """rule, mid: {(form, seed): log_fed record or None} for the RRe runs and RR. RRe: a closing that is not 0 on at least one present row in both logs, finite where present, closing exactly 0.0 on the
    present rows without a full history, the form in the rule log is RRe; the banked RR run feeds 0 on every row of both logs"""
    out, ok = {}, bool(rule) and bool(mid)
    for f in NEW + ("RR",):
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


def ctl_m1(rows):
    """rows: {(form, seed): later-look row count}; the RRe runs equal the banked RR run of the same seed and have rows"""
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


def ctl_a1(levels, bank, st):
    """the levels (U, C, F per seed) of the banked forms read here equal those of the banked read of card ADD76 (READ_MIDLOOK10K.json, settings/<st>/levels) to 1e-9"""
    try:
        if not bank or bank.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the banked read of card ADD76 is missing or not admitted"}
        lv = ((bank.get("settings") or {}).get(st) or {}).get("levels")
        if not isinstance(lv, dict):
            return {"PASS": False, "why": "the banked read holds no levels for %s" % st}
        d = []
        for run in sorted(levels):
            if run not in lv:
                return {"PASS": False, "why": "run %s is not in the banked levels" % run}
            d += [_num(levels[run][c], lv[run][c]) for c in ("U", "C", "F")]
        return {"PASS": bool(d and max(d) <= TOL_A1), "largest_difference": max(d) if d else None, "runs_compared": len(levels), "tolerance": TOL_A1}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def est_lookup(tab):
    """est_score table -> function (ci array, j array) -> est (NaN where the key is not in the table), found mask"""
    key = np.asarray(tab["clip_index"], np.int64) * 1000003 + np.asarray(tab["frame"], np.int64)
    o = np.argsort(key, kind="stable")
    ks, ev = key[o], np.asarray(tab["est"], np.float64)[o]

    def f(ci, j):
        q = np.asarray(ci, np.int64) * 1000003 + np.asarray(j, np.int64)
        pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1)
        found = ks[pos] == q
        return np.where(found, ev[pos], np.nan), found
    return f


def ctl_e2(tables, logs):
    """tables: {(E, seed): est_score arrays or None}; logs: {(E, seed): {"rule": arrays or None, "mid": arrays or None}} with ci, j, k, c_est, est_ok. On the rows with est_ok True the c_est of the log equals the
    table at (ci, j) to 1e-3 m/s (both logs, rows of the first replan tick -- k == 0 -- among them); c_est is NaN where est_ok is False."""
    out, ok, worst = {}, True, 0.0
    for e in ESTS:
        for s in SEEDS:
            tab = tables.get((e, s))
            for tag in ("rule", "mid"):
                z = (logs.get(("RRe_" + e, s)) or {}).get(tag)
                nm = "%s:seed%d:%s_log" % (e, s, tag)
                if tab is None or z is None:
                    out[nm], ok = {"ok": False, "why": "table or log missing"}, False
                    continue
                look = est_lookup(tab)
                eo = np.asarray(z["est_ok"], bool)
                ce = np.asarray(z["c_est"], np.float64)
                ci, j, kk = np.asarray(z["ci"]), np.asarray(z["j"]), np.asarray(z["k"])
                tv, found = look(ci[eo], j[eo])
                d = np.abs(ce[eo] - tv)
                bad = (~found) | ~np.isfinite(tv) | ~np.isfinite(ce[eo]) | ~(d <= TOL_E2)
                n_first = int(((kk == 0) & eo).sum())
                bad_first = int((bad & (kk[eo] == 0)).sum())
                nan_ok = bool(np.all(np.isnan(ce[~eo])))
                mx = float(np.max(d[~bad])) if (~bad).any() else None
                good = bool(eo.any() and not bad.any() and n_first > 0 and nan_ok)
                out[nm] = {"ok": good, "rows_compared": int(eo.sum()), "rows_first_tick_compared": n_first, "rows_failing": int(bad.sum()), "rows_failing_first_tick": bad_first, "nan_where_not_ok": nan_ok,
                           "largest_difference": mx, "tolerance": TOL_E2}
                ok = ok and good
                worst = max(worst, mx or 0.0)
    return {"PASS": bool(ok), "largest_difference_overall": worst, "runs": out}


def ctl_e3(cov, texts, st):
    """cov: {(RRe form, seed): {"n": looks logged, "ok": looks with est_ok True} or None} (both logs together); texts: run texts. Per estimator and setting: share of full histories over the logged looks, pre-roll
    counts from the run text. PASS = every number is there; gate[e] (AGE only) = the scored-window share is at least 0.90 by the logs and by the run text (the estimator's AGE read is named only then)."""
    out, ok, gate = {}, True, {}
    for e in ESTS:
        n = k = sf = ss = pf = ps = 0
        have = True
        for s in SEEDS:
            c = cov.get(("RRe_" + e, s))
            t = parse_est_looks(texts.get(("RRe_" + e, s)))
            if not c or not c["n"] or t is None:
                have = False
                continue
            n, k = n + c["n"], k + c["ok"]
            sf, ss, pf, ps = sf + t["scored_full"], ss + t["scored_short"], pf + t["preroll_full"], ps + t["preroll_short"]
        share_log = k / n if have and n else None
        share_text = sf / (sf + ss) if have and (sf + ss) else None
        share_pre = pf / (pf + ps) if have and (pf + ps) else None
        out[e] = {"complete": bool(have and share_log is not None and share_text is not None), "share_full_history_logged_looks": share_log, "share_full_history_scored_window_run_text": share_text,
                  "share_full_history_preroll_run_text": share_pre, "looks_logged": n, "scored_full": sf, "scored_short": ss, "preroll_full": pf, "preroll_short": ps}
        ok = ok and out[e]["complete"]
        gate[e] = bool(out[e]["complete"] and share_log >= COV_BAR and share_text >= COV_BAR)
    return {"PASS": bool(ok), "per_estimator": out, "gate_scored_share_at_least_0.90": gate, "setting": st}


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
        nm = cname("EST-CLOSING-N10", "AGE", "C", got)
        out[case] = {"want": want, "got": got, "name": nm}
        ok = ok and got == want and nm == cname("EST-CLOSING-N10", "AGE", "C", want)
    return dict(out, PASS=bool(ok))


def assemble(controls, payload):
    ok = bool(controls) and all(v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": controls, "controls_all_pass": bool(ok and payload), "names": None, "why_not_written": None, "settings": None}
    if res["controls_all_pass"]:
        res.update(payload)
    return res


# ============================================================================================ one setting
def register_runs(cfg):
    """the run table of the engine module (in memory only): form, readout, P, route input, grid, pre-roll exec, rule interval, kappa. The form RRe is registered like RRc and RRh of card ADD76."""
    for f, r in cfg["runs"].items():
        R.ALL_RUNS.setdefault(r, (FORM_OF[f], "K", cfg["P"], "deadreckon", cfg["grid"], "plan", None if f == "R0K" else 5, 0.2))
    R.ALL_RUNS.setdefault(cfg["spread"], ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2))


def log_arrays(z):
    """the arrays of the interface of a rule log (None when the log or a key is missing)"""
    if z is None:
        return None
    try:
        return {k: np.asarray(z[k]) for k in ("ci", "j", "k", "c_est", "est_ok")}
    except Exception:                                                                         # noqa: BLE001
        return None


def run_setting(a, E, st, keys, bank, ctx_cache, log):
    """-> (controls with the setting as suffix, part or None, extra). The window arrays are loaded after the grid read, so the registry of the start points is the one of this setting"""
    cfg = SETTINGS[st]
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
    ctl["a1_" + st] = ctl_a1({runs[f]: G["levels"][runs[f]] for f in BANKED}, bank, st)
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
    texts, n_files, rule, mid, rows, Cw, lev, cov, elogs = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for f in FORMS:
        for s in SEEDS:
            run_ = runs[f]
            key = R.key_of(run_, s)
            td = R.tick_dir(a.pull, run_, s)
            if f in NEW:
                texts[(f, s)] = R.txt(os.path.join(td, "run_%s.txt" % key))
            if f in NEW or f == "RR":
                z = R.npz_load(os.path.join(td, "rule_log_%s.npz" % key))
                rule[(f, s)] = log_fed(z) if z is not None else None
                zm = R.npz_load(os.path.join(td, "rule_log_mid_%s.npz" % key))
                mid[(f, s)] = log_fed(zm) if zm is not None else None
                rows[(f, s)] = int(len(np.asarray(zm["ci"]))) if zm is not None else None
                if f in NEW:
                    la, lm = log_arrays(z), log_arrays(zm)
                    elogs[(f, s)] = {"rule": la, "mid": lm}
                    if la is not None and lm is not None:
                        cov[(f, s)] = {"n": int(len(la["est_ok"]) + len(lm["est_ok"])), "ok": int(np.asarray(la["est_ok"], bool).sum() + np.asarray(lm["est_ok"], bool).sum())}
                    else:
                        cov[(f, s)] = None
            raw = R.load_dump_dir(R.dump_dir(a.pull, run_, s))
            if raw is None:
                continue
            if f in NEW:
                n_files[(f, s)] = int(raw["n_files"])
            idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
            if rep["missing"]:
                continue
            D = R.cut(raw, idx)
            L = H.label_windows(J, D["clip_index"], D["ws"])
            Cw[(f, s)] = np.asarray(H.window_cells(D["g"], D["cl"], L)["C"], np.float64)
            lev[(f, s)] = float(H.level(Cw[(f, s)]))
    ctl["r1_" + st] = ctl_r1(texts, n_files, n_clips, cfg["P"] if st == "AGE" else 0, a.wrapper_sha)
    ctl["x1_" + st] = ctl_x1(rule, mid)
    ctl["m1_" + st] = ctl_m1(rows)
    ctl["e3_" + st] = ctl_e3(cov, texts, st)
    extra["age_ok"] = ctl["e3_" + st]["gate_scored_share_at_least_0.90"]
    if st == "FRESH":
        tables = {}
        for e in ESTS:
            for s in SEEDS:
                p = os.path.join(a.est_dir, "est_score_%s_s%d.npz" % (e, s))
                tables[(e, s)] = dict(np.load(p, allow_pickle=False)) if os.path.isfile(p) else None
        ctl["e2_" + st] = ctl_e2(tables, elogs)
    have = all((f, s) in Cw for f in FORMS for s in SEEDS)
    dl = max([abs(lev[(f, s)] - float(G["levels"][runs[f]]["C"][i])) for f in FORMS for i, s in enumerate(SEEDS)]) if have else float("inf")
    trips, pair_route = [], None
    if have:
        for (a_, b_, c_) in [(n_, "RLc", "RR") for n_ in NEW] + [("RRc", "RLc", "RR")]:       # (a minus b) = (a minus c) + (c minus b)
            trips.append({"ab": [lev[(a_, s)] - lev[(b_, s)] for s in SEEDS], "ac": [lev[(a_, s)] - lev[(c_, s)] for s in SEEDS], "cb": [lev[(c_, s)] - lev[(b_, s)] for s in SEEDS]})
        pc = lambda p: [float(x) for x in R.ucf(G["pairs"][p]["headway"])["C"]["per_seed"]]    # noqa: E731
        try:                                                                                  # the same identity on the per-seed values of the headway pairs: reported, not judged
            pair_route = {e: ctl_i1([{"ab": pc("EST-VS-LABEL-FED-" + e), "ac": pc("EST-CLOSING-" + e), "cb": pc("RR-VS-LABEL-FED")}])["largest_abs_dev"] for e in ESTS}
        except Exception as ex:                                                               # noqa: BLE001
            pair_route = "%s: %s" % (type(ex).__name__, str(ex)[:120])
    i1 = ctl_i1(trips)
    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl, pair_route_reported_not_judged=pair_route)
    if S is None or cells_s is None or not all(v.get("PASS") is True for v in ctl.values()):
        return ctl, None, extra
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
    share = None
    if q:
        lab = R.ucf(G["pairs"]["LABEL-CLOSING"]["headway"])["C"]["pooled"]
        share = {e: share_of_gap(pairs_out["EST-CLOSING-" + e]["headway"]["C"]["pooled"], lab) for e in ESTS}
    part = {"qualified": q, "counts": cnt, "pairs": pairs_out, "cells": cells_out, "strategic": strat, "words": words, "share_of_gap_between_RR_and_RRc": share,
            "levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text)}
    return ctl, part, extra


# ============================================================================================ the read and its text
def render(res):
    """-> list of lines. A result whose controls did not all pass prints no name and no value."""
    L = [TIER_LINE, "== controls"]
    for k, v in res["controls"].items():
        L.append("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(R.conv({x: y for x, y in v.items() if x not in ("PASS", "detail", "runs")}))[:600]))
    if not res["controls_all_pass"]:
        L.append("== a control did not pass: no name and no value is written")
        return L
    L.append("== names")
    for k, v in res["names"].items():
        L.append("   %-60s %s" % (k, v))
    for k, v in res["why_not_written"].items():
        L.append("   %-60s %s" % (k, v))
    h3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["pooled"], c["interval"][0], c["interval"][1])   # noqa: E731
    f3 = lambda c: "%+.4f [%+.4f,%+.4f]" % (c["D"], c["interval"][0], c["interval"][1])       # noqa: E731
    for st, p in res["settings"].items():
        L.append("== setting %s: windows %d, with a lead %d in %d episodes, headway qualified %s" % (st, p["counts"]["n_windows"], p["counts"]["n_lead_windows"], p["counts"]["n_episodes_with_lead"], p["qualified"]))
        if p["qualified"]:
            for r, lv in p["levels"].items():
                L.append("   %-22s U %s  C %s  F %s" % (r, ["%.4f" % x for x in lv["U"]], ["%.4f" % x for x in lv["C"]], ["%.4f" % x for x in lv["F"]]))
        for nm, po in p["pairs"].items():
            wd = po["words"]
            if p["qualified"]:
                L.append("   %-26s U %s %s  C %s %s  F (falling behind) %s" % (nm, h3(po["headway"]["U"]), wd["U"], h3(po["headway"]["C"]), wd["C"], h3(po["headway"]["F"])))
            L.append("   %-26s families %s" % (nm, json.dumps(wd["fam"])))
            for cell, c in p["cells"][nm].items():
                L.append("   %-26s %-30s %s %s" % (nm, cell, f3(c), c["verdict"]))
            for b in p["strategic"][nm]["pos_30_50_80m_tick_t0_plus_20"]:
                L.append("   %-26s STRATEGIC pos %sm %+.4f [%+.4f,%+.4f] %s n %s" % (nm, b["arc_m"], b["mean"], b["lo"], b["hi"], b["verdict"], b["n_per_seed"]))
        if st == "AGE" and p.get("share_of_gap_between_RR_and_RRc"):
            for e, sg in p["share_of_gap_between_RR_and_RRc"].items():
                L.append("   share of the contact gap RR minus RRc at AGE removed by %s (arithmetic on pooled pair values, names nothing): gap %+.4f removed %+.4f share %s" % (
                    e, sg["gap"], sg["removed"], "n/a (%s)" % sg["note"] if sg["share"] is None else "%.3f" % sg["share"]))
    return L


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
    controls, parts, cache, age_ok = {}, {}, {}, {e: False for e in ESTS}
    for st in SETTINGS:                                                                        # one setting completely, then the next
        try:
            c, p, ex_ = run_setting(a, E, st, keys[st], bank, cache, log)
            if st == "AGE":
                age_ok = ex_.get("age_ok", age_ok)
        except Exception as ex:                                                               # noqa: BLE001
            c, p = {"setting_ran_" + st: {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:400])}}, None
        controls.update(c)
        parts[st] = p
    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)
    payload = None
    if all(v.get("PASS") is True for v in controls.values()) and all(parts.values()):
        words = {st: parts[st]["words"] for st in SETTINGS}
        names, why = all_names(words, {st: parts[st]["qualified"] for st in SETTINGS}, age_ok)
        payload = {"names": names, "why_not_written": why, "settings": parts, "age_coverage_gate": age_ok}
    res = assemble(controls, payload)
    for line in render(res):
        print(line)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-ESTCLOSE10K rc 0 controls_all_pass %s" % res["controls_all_pass"])
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
    t = ("arm I10RQ50eN10KF5P30Hb20_s0 form RRe replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 30 exec plan nav deadreckon grid grid_p30.json model seed 0 reader seed 0 ckpt sha256 ab "
         "wrapper sha256 " + WR + "0123456789abcdef0123456789abcdef started 2026-11-21T12:00:00Z\nest looks scored full 900 short 50 preroll full 300 short 200 file " + "ab" * 32 + "\nrc 0 finished 2026-11-21T12:04:00Z\n")
    p = CR.parse_run_text(t)
    chk("run text accepted: form RRe, cycle 10, rule every 5, readout, bound, pre-roll, wrapper, return code 0", run_text_ok(p, 30, WR) and not run_text_ok(p, 0, WR) and not run_text_ok(p, 30, "0" * 16))
    chk("run text not accepted with the rule every 10, another form, a return code 1 or another readout",
        not run_text_ok(CR.parse_run_text(t.replace("rule every 5", "rule every 10")), 30, WR) and not run_text_ok(CR.parse_run_text(t.replace("form RRe", "form RRh")), 30, WR)
        and not run_text_ok(CR.parse_run_text(t.replace("rc 0 finished", "rc 1 finished")), 30, WR) and not run_text_ok(CR.parse_run_text(t.replace("causal_g", "pinned")), 30, WR))
    texts = {(f, s): t for f in NEW for s in SEEDS}
    nf = {(f, s): 141 for f in NEW for s in SEEDS}
    t0 = {k: v.replace("preroll 30", "preroll 0") for k, v in texts.items()}
    chk("r1 passes at AGE and (pre-roll 0) FRESH and fails on a missing dump file or a pre-roll mismatch",
        ctl_r1(texts, nf, 141, 30, WR)["PASS"] and ctl_r1(t0, nf, 141, 0, WR)["PASS"] and not ctl_r1(texts, {**nf, ("RRe_L25", 1): 140}, 141, 30, WR)["PASS"] and not ctl_r1(texts, nf, 141, 0, WR)["PASS"])
    pe = parse_est_looks(t)
    chk("est looks line parsed (scored 900 / 50, pre-roll 300 / 200, 64 hex) and a text without it gives None", pe["scored_full"] == 900 and pe["preroll_short"] == 200 and len(pe["file_sha256"]) == 64 and parse_est_looks(t.split("est looks")[0]) is None)

    def zl(form, cs, ce=None, eo=None, pres=(True, True, False, True)):
        z = {"present": np.array(pres), "c_src": np.asarray(cs, np.float64), "form": np.asarray(form)}
        if eo is not None:
            z["est_ok"] = np.asarray(eo, bool)
            z["c_est"] = np.asarray(ce, np.float64)
        return log_fed(z)
    nz, zr = [0.0, -1.5, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0]
    eo = [True, True, True, False]
    ceo = [0.0, -1.5, 0.0, np.nan]
    nz2 = [0.0, -1.5, 0.0, 0.0]
    rule = {(f, s): (zl("RR", zr) if f == "RR" else zl("RRe", nz2, ceo, eo)) for f in NEW + ("RR",) for s in SEEDS}
    mid = dict(rule)
    chk("x1 passes: RRe feeds a closing and 0.0 without a full history in both logs, RR feeds 0", ctl_x1(rule, mid)["PASS"])
    chk("x1 fails when RRe feeds 0 everywhere, feeds a non-zero closing without a full history, names another form, or RR was fed a closing; or a log is missing",
        not ctl_x1({**rule, ("RRe_N10", 0): zl("RRe", zr, ceo, eo)}, mid)["PASS"] and not ctl_x1({**rule, ("RRe_L25", 1): zl("RRe", [0.0, -1.5, 0.0, 0.2], ceo, eo)}, mid)["PASS"]
        and not ctl_x1({**rule, ("RRe_N10", 0): zl("RRh", nz2, ceo, eo)}, mid)["PASS"] and not ctl_x1({**rule, ("RR", 0): zl("RR", [0.0, 0.0, 0.0, 0.1])}, mid)["PASS"]
        and not ctl_x1(rule, {**mid, ("RRe_N10", 0): None})["PASS"])
    rw = {(f, s): 900 for f in NEW + ("RR",) for s in SEEDS}
    chk("m1 passes on equal later-look row counts and fails on a different count or an empty log", ctl_m1(rw)["PASS"] and not ctl_m1({**rw, ("RRe_N10", 0): 899})["PASS"] and not ctl_m1({**rw, ("RR", 1): None})["PASS"])
    tr = {"ab": [0.10, 0.07], "ac": [0.04, 0.05], "cb": [0.06, 0.02]}
    chk("i1 holds on a triple that adds up and fails when off by 1e-6", ctl_i1([tr])["PASS"] and not ctl_i1([dict(tr, ab=[0.10 + 1e-6, 0.07])])["PASS"])
    # e2
    ci = np.repeat(np.arange(5), 6)
    jj = np.tile(np.arange(30, 36), 5)
    tab = {"clip_index": ci, "frame": jj, "est": np.sin(np.arange(30)) * 2.0}
    kk = np.where(jj % 5 == 0, 0, 5)

    def mklog(shift=0.0, ok_all=True):
        eo_ = np.ones(30, bool) if ok_all else (np.arange(30) % 7 != 0)
        c = np.where(eo_, tab["est"] + shift, np.nan)
        return {"ci": ci, "j": jj, "k": kk, "c_est": c, "est_ok": eo_}
    tabs = {(e, s): tab for e in ESTS for s in SEEDS}
    lg = {("RRe_" + e, s): {"rule": mklog(), "mid": mklog(ok_all=False)} for e in ESTS for s in SEEDS}
    chk("e2 passes when c_est equals the table at (ci, j) on the est_ok rows (both logs, first-tick rows included)", ctl_e2(tabs, lg)["PASS"])
    lg2 = {**lg, ("RRe_N10", 1): {"rule": mklog(shift=0.01), "mid": mklog()}}
    chk("e2 fails on a planted difference of 0.01 m/s and passes one of 0.0005", not ctl_e2(tabs, lg2)["PASS"] and ctl_e2(tabs, {**lg, ("RRe_N10", 1): {"rule": mklog(shift=0.0005), "mid": mklog()}})["PASS"])
    bad_nan = mklog(ok_all=False)
    bad_nan["c_est"] = np.where(bad_nan["est_ok"], bad_nan["c_est"], 0.0)
    chk("e2 fails when c_est is not NaN where est_ok is False, when the table has no entry, and when a log or table is missing",
        not ctl_e2(tabs, {**lg, ("RRe_L25", 0): {"rule": lg[("RRe_L25", 0)]["rule"], "mid": bad_nan}})["PASS"]
        and not ctl_e2({**tabs, ("N10", 0): dict(tab, frame=tab["frame"] + 1000)}, lg)["PASS"] and not ctl_e2(tabs, {**lg, ("RRe_N10", 0): {"rule": None, "mid": None}})["PASS"] and not ctl_e2({**tabs, ("L25", 1): None}, lg)["PASS"])
    nofirst = mklog()
    nofirst["k"] = np.full(30, 5)
    chk("e2 fails when no compared row is a first-tick row (k == 0)", not ctl_e2(tabs, {**lg, ("RRe_N10", 0): {"rule": nofirst, "mid": lg[("RRe_N10", 0)]["mid"]}})["PASS"])
    # e3
    cov = {(f, s): {"n": 1000, "ok": 950} for f in NEW for s in SEEDS}
    e3 = ctl_e3(cov, texts, "AGE")
    chk("e3 reports the shares and the gate is True at 0.95 / 0.947 (scored 900 of 950 by the text)", e3["PASS"] and e3["gate_scored_share_at_least_0.90"] == {"N10": True, "L25": True}
        and abs(e3["per_estimator"]["N10"]["share_full_history_preroll_run_text"] - 0.6) < 1e-12)
    cov2 = {**cov, ("RRe_L25", 0): {"n": 1000, "ok": 500}, ("RRe_L25", 1): {"n": 1000, "ok": 500}}
    g2 = ctl_e3(cov2, texts, "AGE")["gate_scored_share_at_least_0.90"]
    chk("e3 gate is False for the estimator below 0.90 in the logs (L25 at 0.50) and still True for the other", g2 == {"N10": True, "L25": False})
    t_low = {k: v.replace("scored full 900 short 50", "scored full 500 short 450") for k, v in texts.items()}
    chk("e3 gate is False when the run text scored share is below 0.90 although the logs hold 0.95", ctl_e3(cov, t_low, "AGE")["gate_scored_share_at_least_0.90"] == {"N10": False, "L25": False})
    chk("e3 fails when a run text has no est looks line or a log is missing", not ctl_e3(cov, {**texts, ("RRe_N10", 0): "x"}, "AGE")["PASS"] and not ctl_e3({**cov, ("RRe_N10", 0): None}, texts, "AGE")["PASS"])

    def wd(c, u, fam=None):
        return {"C": c, "U": u, "fam": dict({f: "NOT-SEPARATED" for f in FAMS}, **(fam or {}))}

    def W(age, fresh, e="N10", other=None):
        o = other or wd("NOT-SEPARATED", "NOT-SEPARATED")
        oe = "L25" if e == "N10" else "N10"
        return {"AGE": {"EST-CLOSING-" + e: age, "EST-CLOSING-" + oe: o}, "FRESH": {"EST-CLOSING-" + e: fresh, "EST-CLOSING-" + oe: o}}
    both = {"N10": True, "L25": True}
    n1, y1 = verdict_names(W(wd("LOWER", "LOWER"), wd("NOT-SEPARATED", "NOT-SEPARATED", {"LATERAL": "MIXED"})), both)
    chk("verdict: N10 SUPPORTED (AGE C LOWER, FRESH C not HIGHER, MIXED is not WORSE), L25 NOT-SUPPORTED, and the single SUPPORTED is marked one of two",
        n1["VERDICT_N10"] == "ESTCLOSE10K-N10-SUPPORTED" and n1["VERDICT_L25"] == "ESTCLOSE10K-L25-NOT-SUPPORTED" and n1["VERDICT_ONE_OF_TWO"] == "ESTCLOSE10K-N10-SUPPORTED-ONE-OF-TWO")
    chk("verdict: NOT-SUPPORTED when AGE C is not LOWER, FRESH C HIGHER, AGE U HIGHER, FRESH U HIGHER, a family WORSE at AGE or FRESH",
        all(verdict_names(w_, both)[0]["VERDICT_N10"] == "ESTCLOSE10K-N10-NOT-SUPPORTED" for w_ in (
            W(wd("NOT-SEPARATED", "LOWER"), wd("LOWER", "LOWER")), W(wd("LOWER", "LOWER"), wd("HIGHER", "LOWER")), W(wd("LOWER", "HIGHER"), wd("LOWER", "LOWER")), W(wd("LOWER", "LOWER"), wd("LOWER", "HIGHER")),
            W(wd("LOWER", "LOWER", {"TACTICAL": "WORSE"}), wd("LOWER", "LOWER")), W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER", {"STRATEGIC": "WORSE"})))))
    sup = W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), other=wd("LOWER", "LOWER"))
    ns, _ = verdict_names(sup, both)
    chk("verdict: both SUPPORTED names both and no one-of-two mark; L25 alone SUPPORTED is marked for L25",
        ns["VERDICT_N10"].endswith("-N10-SUPPORTED") and ns["VERDICT_L25"].endswith("-L25-SUPPORTED") and "VERDICT_ONE_OF_TWO" not in ns
        and verdict_names(W(wd("LOWER", "LOWER"), wd("LOWER", "LOWER"), e="L25", other=wd("HIGHER", "LOWER")), both)[0]["VERDICT_ONE_OF_TWO"] == "ESTCLOSE10K-L25-SUPPORTED-ONE-OF-TWO")
    n3, y3 = verdict_names(sup, {"N10": True, "L25": False})
    chk("verdict: an estimator below the AGE coverage bar has no verdict and says why, the other one is still read", "VERDICT_L25" not in n3 and "VERDICT_L25" in y3 and n3["VERDICT_N10"] == "ESTCLOSE10K-N10-SUPPORTED")
    un = W(wd(None, None), wd("LOWER", "LOWER"))
    n4, y4 = verdict_names(un, both)
    chk("verdict: an unqualified setting writes no name that needs its C or U; a failing family condition still decides",
        "VERDICT_N10" not in n4 and "VERDICT_N10" in y4 and verdict_names(W(wd(None, None, {"LATERAL": "WORSE"}), wd("LOWER", "LOWER")), both)[0]["VERDICT_N10"] == "ESTCLOSE10K-N10-NOT-SUPPORTED")
    full = {st: {pn: wd("LOWER", "NOT-SEPARATED") for pn in PAIRS} for st in SETTINGS}
    nm, why = all_names(full, {"AGE": True, "FRESH": True}, both)
    chk("names: pattern ESTCLOSE10K-<PAIR>-<SETTING>-C-<word>, U and the four families per pair and setting (8 pairs x 2 settings)",
        nm["EST-CLOSING-N10_AGE_C"] == "ESTCLOSE10K-EST-CLOSING-N10-AGE-C-LOWER" and nm["EST-VS-LABEL-FED-L25_FRESH_STRATEGIC"] == "ESTCLOSE10K-EST-VS-LABEL-FED-L25-FRESH-STRATEGIC-NOT-SEPARATED"
        and len([k for k in nm if k.endswith("_U")]) == 16 and len([k for k in nm if k.endswith("_LATERAL")]) == 16)
    nm2, why2 = all_names(full, {"AGE": True, "FRESH": True}, {"N10": True, "L25": False})
    chk("names: AGE names of an estimator below the coverage bar are not written, its FRESH names are", not [k for k in nm2 if k.startswith("EST") and k.endswith("L25_AGE_C")] and "EST-CLOSING-L25_FRESH_C" in nm2 and "AGE_EST-CLOSING-L25" in why2)
    nm3, why3 = all_names({st: {pn: wd(None, None) for pn in PAIRS} for st in SETTINGS}, {"AGE": False, "FRESH": False}, both)
    chk("names: with no qualified setting no C or U name is written and the output says why", not [k for k in nm3 if k.endswith("_C") or k.endswith("_U")] and "HEADWAY_AGE" in why3 and "HEADWAY_FRESH" in why3)
    sg = share_of_gap(-0.02, -0.08)
    chk("share of gap: removed 0.02 of a gap of 0.08 is 0.25; no share when the label closing does not lower contact", abs(sg["share"] - 0.25) < 1e-12 and share_of_gap(-0.02, +0.01)["share"] is None)
    a_ = assemble({"x": {"PASS": False}}, {"names": {"k": "v"}})
    chk("a failed control leaves no name", a_["names"] is None and a_["controls_all_pass"] is False)
    shown = "\n".join(render(a_))
    chk("a failed control prints no name and no value", "ESTCLOSE10K-" not in shown and "== names" not in shown and "no name and no value" in shown)
    lv = {"RQ": {c: [0.1, 0.2] for c in ("U", "C", "F")}, "R0": {c: [0.3, 0.4] for c in ("U", "C", "F")}}
    bank = {"controls_all_pass": True, "settings": {"AGE": {"levels": {r: {c: list(v) for c, v in d.items()} for r, d in lv.items()}}, "FRESH": {"levels": "not reported"}}}
    chk("a1: equal levels pass, 1e-6 off fails, a run missing fails, an unqualified banked setting fails, an unadmitted bank fails",
        ctl_a1(lv, bank, "AGE")["PASS"] and not ctl_a1({"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2]), "R0": lv["R0"]}, bank, "AGE")["PASS"] and not ctl_a1({**lv, "RX": lv["RQ"]}, bank, "AGE")["PASS"]
        and not ctl_a1(lv, bank, "FRESH")["PASS"] and not ctl_a1(lv, dict(bank, controls_all_pass=False), "AGE")["PASS"])
    chk("qualification by the instrument's own bar (200 windows with a lead, 20 episodes)",
        qualified({"n_lead_windows": 325, "n_episodes_with_lead": 46}, type("H", (), {"MIN_WINDOWS": 200, "MIN_EPISODES": 20})) and not qualified({"n_lead_windows": 199, "n_episodes_with_lead": 46}, type("H", (), {"MIN_WINDOWS": 200, "MIN_EPISODES": 20})))
    reg = {}
    saved = dict(R.ALL_RUNS)
    register_runs(SETTINGS["AGE"])
    register_runs(SETTINGS["FRESH"])
    chk("register_runs enters the RRe runs with form RRe, readout K, rule every 5, pre-roll of the setting", R.ALL_RUNS["RQ50eN10KF5P30Hb20"][:3] == ("RRe", "K", 30) and R.ALL_RUNS["RQ50eL25KF5Hb20"][2] == 0 and R.ALL_RUNS["RQ50eL25KF5Hb20"][6] == 5)
    R.ALL_RUNS.clear()
    R.ALL_RUNS.update(saved)
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
    chk("text of the tool holds none of the banned words", not [x for x in bad if x in src])
    ok = all(res)
    print("ZZ SELFTEST-ESTCLOSE10K ok: %s (%d of %d)" % (ok, sum(res), len(res)))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("pull", "code", "grid-dir", "fresh-grid", "midlook-json", "est-dir", "w", "tree", "out", "wrapper-sha"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
