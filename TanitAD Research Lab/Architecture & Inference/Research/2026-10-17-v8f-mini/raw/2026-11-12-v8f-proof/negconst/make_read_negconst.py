"""make_read_negconst.py -- writes read_negconst10k.py (reader of card ADD82, NEGCONST10K) from constclose/read_constclose10k_n1.py (sha256 prefix c777b3898a5b6238, the reader of card ADD80 with the
first-tick clause of k2) by counted, asserted text edits. Refuses to overwrite its output (NC_OVERWRITE_OWN=1 rewrites it).
Usage: bash ops/pycap.sh make_read_negconst.py"""
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "constclose", "read_constclose10k_n1.py"))
OUT = os.path.join(HERE, "read_negconst10k.py")
raw = io.open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest()[:16] == "c777b3898a5b6238", "parent changed"
if os.path.exists(OUT) and os.environ.get("NC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists" % OUT)
T = raw.decode("utf-8")
n_edits = 0


def once(a, b, n=1):
    global T, n_edits
    assert T.count(a) == n, (a[:90], T.count(a))
    T = T.replace(a, b)
    n_edits += 1


def glob(a, b):
    global T, n_edits
    n = T.count(a)
    assert n >= 1, a
    T = T.replace(a, b)
    n_edits += 1
    return n


# ------------------------------------------------------------------------------------------------ docstring and head
once('#!/usr/bin/env python\n', '#!/usr/bin/env python\n')
once('CARD, PRE, PARENT_CARD = "PREREG_V8F_PROOF_ADD80", "CONSTCLOSE10K", "PREREG_V8F_PROOF_ADD79"',
     'CARD, PRE, PARENT_CARD, CONST_CARD = "PREREG_V8F_PROOF_ADD82", "NEGCONST10K", "PREREG_V8F_PROOF_ADD79", "PREREG_V8F_PROOF_ADD80"')
once('CVAL = {"C10": 1.0, "C20": 2.0, "C40": 4.0}', 'CVAL = {"M10": -1.0, "M20": -2.0, "M40": -4.0}\nOLDV = ("C10", "C20", "C40")                      # the runs of card ADD80, +1.0 / +2.0 / +4.0 m/s (read for k3, k4 and the dose table)\nOLDVAL = {"C10": 1.0, "C20": 2.0, "C40": 4.0}')
# ------------------------------------------------------------------------------------------------ global renames (the C values of the parent are the M values here)
once("VALUES = (\"C10\", \"C20\", \"C40\")", "VALUES = (\"M10\", \"M20\", \"M40\")")
T, n1_ = re.subn(r"(?<![A-Za-z0-9])C(10|20|40)(?![A-Za-z0-9])", r"M\1", T)
T, n2_ = re.subn(r"RQ50eC(10|20|40)", r"RQ50eM\1", T)
assert n1_ > 50 and n2_ >= 3, (n1_, n2_)
n_edits += 2
# OLDV / OLDVAL must stay as defined (the loop above replaced the quoted strings of their definition)
once('OLDV = ("M10", "M20", "M40")', 'OLDV = ("C10", "C20", "C40")')
once('OLDVAL = {"M10": 1.0, "M20": 2.0, "M40": 4.0}', 'OLDVAL = {"C10": 1.0, "C20": 2.0, "C40": 4.0}')
glob("CONSTCLOSE10K", "NEGCONST10K")
glob("read_constclose10k", "read_negconst10k")
glob("BEYOND-A-CONSTANT", "BEYOND-A-CAUTIOUS-CONSTANT")
# ------------------------------------------------------------------------------------------------ run tables, pairs, forms
once('NEWE = tuple("RRe_" + e for e in ESTS)\n', 'NEWE = tuple("RRe_" + e for e in ESTS)\nOLDC = tuple("RRe_" + c for c in OLDV)\nDOSE = (("RRe_M40", -4.0), ("RRe_M20", -2.0), ("RRe_M10", -1.0), ("RR", 0.0), ("RRe_C10", 1.0), ("RRe_C20", 2.0), ("RRe_C40", 4.0))\n')
once('runs = {"RRe_" + c: "RQ50e%sKF5%s" % (c, suf) for c in VALUES}', 'runs = {"RRe_" + c: "RQ50e%sKF5%s" % (c, suf) for c in VALUES + OLDV}')
once('return NEWC + (NEWE if e_on else ()) + BANKED', 'return NEWC + OLDC + (NEWE if e_on else ()) + BANKED')
once('FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES})', 'FORM_OF.update({"RRe_" + c: "RRe" for c in VALUES + OLDV})')
once('"""run table per setting: key (form) -> run name; the runs of card ADD79 are in it only when its read is admitted"""',
     '"""run table per setting: key (form) -> run name; the runs of card ADD80 are always in it, the runs of card ADD79 when e_on (the reader runs with e_on True: both files are required)"""')

# ------------------------------------------------------------------------------------------------ new functions: admission of the file of card ADD80, k3, k4, the dose table
NEWFN = '''def read_c_json(path, jl):
    """-> (c_on, reason, json or None). The read of card ADD80 is required: admitted only when the file is there, is the result of that card and says controls_all_pass true."""
    if not path:
        return False, "no --constclose-json was given", None
    cj = jl(path)
    if cj is None:
        return False, "the file %s is not there or not readable" % path, None
    if cj.get("card") != CONST_CARD:
        return False, "the file is not the result of card ADD80", None
    if cj.get("controls_all_pass") is not True:
        return False, "the read of card ADD80 has not all its controls passed", None
    return True, "", cj


def ctl_k3(levels, j, st, card):
    """control k3: the levels (U, C, F per seed) of the runs of another card read here equal those of the result file of that card (settings/<st>/levels) to 1e-9; the file has to be the result of the card
    with controls_all_pass true"""
    nm = card.split("_")[-1]
    try:
        if not isinstance(j, dict) or j.get("card") != card or j.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the result file of card %s is missing, is not the result of that card or has not all its controls passed" % nm}
        lv = ((j.get("settings") or {}).get(st) or {}).get("levels")
        if not isinstance(lv, dict):
            return {"PASS": False, "why": "the result file of card %s holds no levels for %s" % (nm, st)}
        d = []
        for run in sorted(levels):
            if run not in lv:
                return {"PASS": False, "why": "run %s is not in the levels of the result file of card %s" % (run, nm)}
            d += [EC._num(levels[run][c], lv[run][c]) for c in ("U", "C", "F")]
        return {"PASS": bool(d and max(d) <= EC.TOL_A1), "largest_difference": max(d) if d else None, "runs_compared": len(levels), "tolerance": EC.TOL_A1, "card": nm}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def ctl_k4(lev, seeds=SEEDS):
    """control k4 (the sign, read from the data and not from a name), AGE only: lev {(form, seed): contact level C of the windows with a lead}; the contact level pooled over the seeds (mean) is not higher at -1 m/s
    than at +1 m/s and not higher at -4 m/s than at +4 m/s. A failure means the rule does not respond to the sign as the code says."""
    out, ok = {}, bool(lev)
    for neg, pos in (("RRe_M10", "RRe_C10"), ("RRe_M40", "RRe_C40")):
        try:
            ma = float(np.mean([lev[(neg, s)] for s in seeds]))
            mb = float(np.mean([lev[(pos, s)] for s in seeds]))
            good = bool(np.isfinite(ma) and np.isfinite(mb) and ma <= mb)
        except KeyError:
            ma = mb = None
            good = False
        out["%s_not_higher_than_%s" % (neg, pos)] = {"ok": good, "pooled_C_negative": ma, "pooled_C_positive": mb}
        ok = ok and good
    return {"PASS": bool(ok), "pairs": out}


def dose_table(levels, runs, seeds=SEEDS):
    """the dose over the constants -4, -2, -1, 0 (form RR), +1, +2, +4 m/s: C, U and F levels per seed and pooled (mean over the seeds). Reported, names nothing."""
    rows = []
    for f, c in DOSE:
        lv = levels[runs[f]]
        row = {"form": f, "run": runs[f], "c_m_per_s": c}
        for k in ("C", "U", "F"):
            per = [float(x) for x in np.asarray(lv[k], np.float64).reshape(-1)[:len(seeds)]]
            row[k] = {"per_seed": per, "pooled": float(np.mean(per))}
        rows.append(row)
    return rows


def render_dose(res):
    """lines of the dose table per setting (only when all controls passed and the headway instrument is qualified there)"""
    L = []
    if not res.get("controls_all_pass"):
        return L
    for st, p in (res.get("settings") or {}).items():
        if not p.get("qualified") or not isinstance(p.get("dose"), list):
            continue
        L.append("== dose table %s (levels of the windows with a lead; per seed and pooled; reported, names nothing)" % st)
        for r in p["dose"]:
            L.append("   c %+.1f m/s %-9s C %s pooled %.4f  U %s pooled %.4f  F %s pooled %.4f" % (r["c_m_per_s"], r["form"], ["%.4f" % x for x in r["C"]["per_seed"]], r["C"]["pooled"],
                                                                                              ["%.4f" % x for x in r["U"]["per_seed"]], r["U"]["pooled"], ["%.4f" % x for x in r["F"]["per_seed"]], r["F"]["pooled"]))
    return L


'''
once("# ============================================================================================ one setting\n", NEWFN + "# ============================================================================================ one setting\n")
once("def run_setting(a, E, st, cfg, keys, bank, ej, e_on, ctx_cache, log, CL):", "def run_setting(a, E, st, cfg, keys, bank, ej, cj, e_on, ctx_cache, log, CL):")
once('''    if e_on:
        ctl["k3_" + st] = EC.ctl_a1({runs[f]: G["levels"][runs[f]] for f in NEWE}, ej, st)
''', '''    if e_on:
        ctl["k3_estclose_" + st] = ctl_k3({runs[f]: G["levels"][runs[f]] for f in NEWE}, ej, st, PARENT_CARD)
    ctl["k3_constclose_" + st] = ctl_k3({runs[f]: G["levels"][runs[f]] for f in OLDC}, cj, st, CONST_CARD)
''')
once('''    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl)
''', '''    ctl["i1_" + st] = dict(i1, PASS=bool(i1["PASS"] and have and dl <= 1e-12), largest_abs_dev_levels_vs_grid_read=dl)
    if st == "AGE":
        ctl["k4_" + st] = ctl_k4(lev)
''')
once('"levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text)}', '"levels": ({r: G["levels"][r] for r in runs.values()} if q else hw_text), "dose": (dose_table(G["levels"], runs) if q else hw_text)}')
# ------------------------------------------------------------------------------------------------ run() and main()
once('''    e_on, e_reason, ej = read_e_json(getattr(a, "estclose_json", None), R.jl)
    if not e_on:
        print("NOTE: the pairs with the estimators and the verdicts of E are not written: %s (this is not a failed control)" % e_reason)
    SET = settings(e_on)
''', '''    e_on, e_reason, ej = read_e_json(getattr(a, "estclose_json", None), R.jl)
    c_on, c_reason, cj = read_c_json(getattr(a, "constclose_json", None), R.jl)
    for nm_, on_, why_ in (("ADD79", e_on, e_reason), ("ADD80", c_on, c_reason)):
        if not on_:
            print("NOTE: control k3 fails, the result file of card %s is required and is not admitted: %s" % (nm_, why_))
    SET = settings(True)
''')
once("c, p, ex_ = run_setting(a, E, st, SET[st], keys[st], bank, ej, e_on, cache, log, CL)", "c, p, ex_ = run_setting(a, E, st, SET[st], keys[st], bank, ej, cj, True, cache, log, CL)")
once('    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)\n',
     '    controls["k3_estclose_json_admitted"] = {"PASS": bool(e_on), "why": e_reason}\n    controls["k3_constclose_json_admitted"] = {"PASS": bool(c_on), "why": c_reason}\n    controls["h1_planted_name_rule"] = ctl_h1(E[2], a.n_boot_control)\n')
once('names, why = all_names(words, {st: parts[st]["qualified"] for st in SET}, gate, e_on, e_reason)', 'names, why = all_names(words, {st: parts[st]["qualified"] for st in SET}, gate, True, "")')
once(', "coverage_gate": gate, "estclose_pairs_written": e_on}', ', "coverage_gate": gate}')
once('''    for line in EC.render(res):
        print(line)
''', '''    for line in EC.render(res) + render_dose(res):
        print(line)
''')
once('    ap.add_argument("--estclose-json", default="")\n', '    ap.add_argument("--estclose-json", required=True)\n    ap.add_argument("--constclose-json", required=True)\n')

# ------------------------------------------------------------------------------------------------ selftest edits
once('CVAL == {"M10": 1.0, "M20": 2.0, "M40": 4.0}', 'CVAL == {"M10": -1.0, "M20": -2.0, "M40": -4.0} and OLDVAL == {"C10": 1.0, "C20": 2.0, "C40": 4.0}')
once("the values and pre-rolls of the card: 1.0, 2.0, 4.0 m/s;", "the values and pre-rolls of the card: -1.0, -2.0, -4.0 m/s (and +1.0, +2.0, +4.0 of card ADD80);")
once("(planted 4.5 for M40)", "(planted -3.5 for M40, a value of the wrong size)")
once("k2 fails on the value of another c (2.0 in the M10 run)", "k2 fails on the value of another c (-2.0 in the M10 run)")
once("cs_shift=-0.1", "cs_shift=0.1")
once("x1 fails on a closing of 1.9 for M20", "x1 fails on a closing of -1.9 for M20")
# k3 / a1 selftest block: the function of the parent is replaced by the two-file k3 of this reader
a0 = T.index('    # --- k3 and a1 (the levels of the runs of card ADD79 and of the banked forms equal the files)')
a1 = T.index('    tr = {"ab": [0.10, 0.07]')
NEWK3 = '''    # --- a1 (banked forms; function of the parent), k3 (two result files), k4, dose table
    lv = {"RQ": {c: [0.1, 0.2] for c in ("U", "C", "F")}, "R0": {c: [0.3, 0.4] for c in ("U", "C", "F")}}
    ej = {"controls_all_pass": True, "settings": {"AGE": {"levels": {r: {c: list(v) for c, v in dd.items()} for r, dd in lv.items()}}}}
    chk("a1 (function of the parent): equal levels pass; 1e-6 off, a run missing from the file, an all-pass flag False each fail",
        EC.ctl_a1(lv, ej, "AGE")["PASS"] and not EC.ctl_a1({"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2]), "R0": lv["R0"]}, ej, "AGE")["PASS"] and not EC.ctl_a1({**lv, "RX": lv["RQ"]}, ej, "AGE")["PASS"]
        and not EC.ctl_a1(lv, dict(ej, controls_all_pass=False), "AGE")["PASS"])
    for card_ in (PARENT_CARD, CONST_CARD):
        jj_ = dict(ej, card=card_)
        res_ = [ctl_k3(lv, jj_, "AGE", card_)["PASS"],
                ctl_k3({"RQ": dict(lv["RQ"], C=[0.1 + 1e-6, 0.2]), "R0": lv["R0"]}, jj_, "AGE", card_)["PASS"],
                ctl_k3({**lv, "RX": lv["RQ"]}, jj_, "AGE", card_)["PASS"],
                ctl_k3(lv, dict(jj_, controls_all_pass=False), "AGE", card_)["PASS"],
                ctl_k3(lv, dict(jj_, card="PREREG_V8F_PROOF_ADD00"), "AGE", card_)["PASS"],
                ctl_k3(lv, None, "AGE", card_)["PASS"],
                ctl_k3(lv, jj_, "FRESH", card_)["PASS"]]
        chk("k3 on the result file of card %s: equal levels pass; 1e-6 off, a run missing, controls_all_pass False, another card, no file, no levels for the setting each fail" % card_[-5:], res_ == [True] + [False] * 6)
    chk("k3 reads each file against its own runs: the levels of the E runs do not pass against the file of the other card",
        not ctl_k3(lv, dict(ej, card=CONST_CARD), "AGE", PARENT_CARD)["PASS"] and not ctl_k3(lv, dict(ej, card=PARENT_CARD), "AGE", CONST_CARD)["PASS"])
    jl2_ = lambda d: (lambda p: d.get(p))                                                      # noqa: E731
    chk("estclose / constclose files: absent path, unreadable, another card, not all-pass are not admitted; all-pass of the right card is admitted (each of the two)",
        not read_c_json("", jl2_({}))[0] and not read_c_json("x", jl2_({}))[0] and not read_c_json("x", jl2_({"x": {"card": PARENT_CARD, "controls_all_pass": True}}))[0]
        and not read_c_json("x", jl2_({"x": {"card": CONST_CARD, "controls_all_pass": False}}))[0] and read_c_json("x", jl2_({"x": {"card": CONST_CARD, "controls_all_pass": True}}))[0]
        and not read_e_json("x", jl2_({"x": {"card": CONST_CARD, "controls_all_pass": True}}))[0])
    lev4 = {(f, s): v for f, v in (("RRe_M10", 0.10), ("RRe_C10", 0.12), ("RRe_M40", 0.05), ("RRe_C40", 0.20)) for s in SEEDS}
    chk("k4 passes when the pooled contact level at -1 is not higher than at +1 and at -4 not higher than at +4; equal levels pass", ctl_k4(lev4)["PASS"] and ctl_k4({**lev4, ("RRe_M10", 0): 0.12, ("RRe_M10", 1): 0.12})["PASS"])
    chk("k4 fails when -1 m/s is higher than +1 m/s (the sign), when -4 m/s is higher than +4 m/s, on a missing level and on no level",
        not ctl_k4({**lev4, ("RRe_M10", 0): 0.20, ("RRe_M10", 1): 0.20})["PASS"] and not ctl_k4({**lev4, ("RRe_M40", 0): 0.30, ("RRe_M40", 1): 0.30})["PASS"]
        and not ctl_k4({k: v for k, v in lev4.items() if k != ("RRe_C40", 1)})["PASS"] and not ctl_k4({})["PASS"])
    chk("k4 pools the two seeds by their mean: a seed that differs in sign does not decide alone",
        ctl_k4({**lev4, ("RRe_M10", 0): 0.20, ("RRe_M10", 1): 0.02})["PASS"] and not ctl_k4({**lev4, ("RRe_M10", 0): 0.20, ("RRe_M10", 1): 0.05})["PASS"])
    srun = settings(True)["AGE"]["runs"]
    lvd = {srun[f]: {k: np.array([base + 0.01, base + 0.03]) for k, base in (("C", 0.1 * (i + 1)), ("U", 0.2), ("F", 0.3))} for i, (f, _) in enumerate(DOSE)}
    dt = dose_table(lvd, srun)
    chk("dose table: seven rows in the order -4, -2, -1, 0 (RR), +1, +2, +4 with C, U, F per seed and pooled (mean)", [r["c_m_per_s"] for r in dt] == [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] and dt[3]["form"] == "RR"
        and abs(dt[0]["C"]["pooled"] - 0.12) < 1e-12 and dt[0]["C"]["per_seed"] == [0.11, 0.13] and abs(dt[6]["C"]["pooled"] - 0.72) < 1e-12)
    dl_ = render_dose({"controls_all_pass": True, "settings": {"AGE": {"qualified": True, "dose": dt}, "FRESH": {"qualified": False, "dose": "x"}}})
    chk("dose table is printed only with all controls passed and only for a qualified setting, and names nothing", len(dl_) == 8 and "NEGCONST10K-" not in "\\n".join(dl_) and not render_dose({"controls_all_pass": False, "settings": {"AGE": {"qualified": True, "dose": dt}}}))
'''
T = T[:a0] + NEWK3 + T[a1:]
n_edits += 1
# the run table check and the register check of the parent
once('and "RRe_N10" not in settings(False)["AGE"]["runs"] and S_["AGE"]["runs"]["RR"]', 'and "RRe_N10" not in settings(False)["AGE"]["runs"] and S_["AGE"]["runs"]["RRe_C20"] == "RQ50eC20KF5P30Hb20" and S_["FRESH"]["runs"]["RRe_C40"] == "RQ50eC40KF5Hb20" and S_["AGE"]["runs"]["RR"]')
once('    saved = dict(R.ALL_RUNS)\n', '''    chk("forms of the card: the new values, the runs of card ADD80 and (admitted) of ADD79 and the banked forms; the pair table is that of the card (15 pairs with the estimators, 9 without)",
        forms_of(True) == NEWC + OLDC + NEWE + BANKED and forms_of(False) == NEWC + OLDC + BANKED and NEWC == ("RRe_M10", "RRe_M20", "RRe_M40") and OLDC == ("RRe_C10", "RRe_C20", "RRe_C40") and FORM_OF["RRe_C20"] == "RRe" and FORM_OF["RRe_M40"] == "RRe")
    saved = dict(R.ALL_RUNS)\n''')
# no trace of the parent card name
for bad in ("CONSTCLOSE", "constclose10k", "BEYOND-A-CONSTANT"):
    assert bad not in T.replace("CAUTIOUS-CONSTANT", ""), bad
end = T.index('Last line: ZZ READ-NEGCONST10K rc <rc> controls_all_pass <bool>\n"""') + len('Last line: ZZ READ-NEGCONST10K rc <rc> controls_all_pass <bool>\n"""')
DOC = '''"""read_negconst10k.py -- read of card PREREG_V8F_PROOF_ADD82 (NEGCONST10K): the reader-fed lead rule with a CONSTANT closing input of the CAUTIOUS sign (form RRe, estimator files of history one frame,
values M10 = -1.0, M20 = -2.0, M40 = -4.0 m/s), at age and fresh. Derived by make_read_negconst.py from the reader of card ADD80 (sha256 prefix c777b3898a5b6238, with the first-tick clause of k2);
under it read_estclose10k.py (card ADD79), which is imported and not edited, and the engine modules (read_handsrc_decomp, read_closrc10k, read_longroll10k).

Two settings, each read completely before the next starts. New runs (both seeds): AGE RQ50eM10KF5P30Hb20, RQ50eM20KF5P30Hb20, RQ50eM40KF5P30Hb20; FRESH RQ50eM10KF5Hb20, RQ50eM20KF5Hb20, RQ50eM40KF5Hb20.
Banked (row 137): RRh, RRc, RR, RLc, R0K (levels only for RRh and RLc: control a1). REQUIRED, from the two result files: the runs RRe(N10), RRe(L25) of card ADD79 (--estclose-json) and the runs RRe(C10), RRe(C20),
RRe(C40) of card ADD80 (--constclose-json, +1.0, +2.0, +4.0 m/s). A file that is absent, not of its card or without controls_all_pass true fails control k3 and no name is written.
Pairs per value c (M10, M20, M40) and setting, A minus B (card section 3): CONST-VS-RR-c (RRe(c), RR), CONST-VS-LABEL-CLOSING-c (RRe(c), RRc), CONST-VS-NO-RULE-c (RRe(c), R0K) and CONST-VS-EST-E-c (RRe(c), RRe(E))
for E in N10, L25. Names NEGCONST10K-<PAIR>-<SETTING>-<C|U|F|family>-<word>, the verdicts NEGCONST10K-<c>-SUPPORTED / -NOT-SUPPORTED with the F name beside and the one-of-three mark, and
NEGCONST10K-<E>-NOT-BEYOND-A-CAUTIOUS-CONSTANT / -BEYOND-A-CAUTIOUS-CONSTANT by the match rule. The dose table over -4, -2, -1, 0 (form RR), +1, +2, +4 m/s (C, U, F per seed and pooled) is reported and names nothing.
Controls (card section 4), printed first: k0 (files), k2 (the estimate column equals c; coverage of full histories, bar 0.99; first-tick clause), k3 (the levels of the runs of cards ADD79 and ADD80 equal their result files;
the admission of both files), k4 (at AGE the pooled contact level is not higher at M10 than at C10 and not higher at M40 than at C40), x1, m1, g, s, a1, r1, i1, h1.
A control that does not pass: no name and no value is written or printed.
Usage: python read_negconst10k.py --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --const-dir DIR --w W --tree TREE --out OUT.json --wrapper-sha HEX16
          --estclose-json READ_ESTCLOSE10K.json --constclose-json READ_CONSTCLOSE10K.json [--n-boot 2000]   |   --selftest
Last line: ZZ READ-NEGCONST10K rc <rc> controls_all_pass <bool>
"""'''
T = DOC + T[end:]
n_edits += 1
io.open(OUT, "w", encoding="utf-8", newline="\n").write(T)
rest = [m.group(0) for m in re.finditer(r".{0,25}\bC(10|20|40)\b.{0,25}", T)]
print("wrote %s sha256 %s edits %d; places that still name C10/C20/C40: %d" % (os.path.basename(OUT), hashlib.sha256(T.encode("utf-8")).hexdigest()[:16], n_edits, len(rest)))
