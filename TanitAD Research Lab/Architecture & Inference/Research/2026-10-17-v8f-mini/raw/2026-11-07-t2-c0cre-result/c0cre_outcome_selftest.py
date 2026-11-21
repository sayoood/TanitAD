"""Self-test of c0cre_outcome.py, including its deliberate-regression checks.  CPU, standard library, seconds.
    python -I c0cre_outcome_selftest.py [--out-dir DIR]        (exit 0 only if every check holds; prints one line per check)

The fixtures are the REAL banked reads of earlier arms used as schema templates (stage5/BPC/score_dom/DOMINANCE_T2.json, stage5/BPC/pair_C0C_BPC/PAIR_T2.json, stage4/score_q_identity/DOMINANCE_T2.json),
re-labelled as C0C-RE and rewritten row by row through ONE helper that keeps `rows`, `family_table`, `read_a` and `decision` consistent, the way the pinned reader writes them.  These are fabricated
inputs for the script's logic; no number in them is a result of C0C-RE.
Deliberate regressions (each must CHANGE the outcome relative to its control):
  R1  one family row of the pair (M1 crawl share) flipped to separated-worse           RE-MOVES -> RE-HARMS
  R2  one family row of C0C-RE against P10 (speed MAE) flipped to separated-worse       RE-MOVES -> RE-HARMS   (a NEW row; the control keeps C0C's own separated-worse rows and stays RE-MOVES)
  R3  one family row flipped to separated-worse inside a DOMINANT-T2 claim              RE-DOMINANT -> INPUT-INVALID (the claim contradicts its own table)
  R4  the same flip written consistently (the reader would have said NOT-DOMINANT)      RE-DOMINANT -> not RE-DOMINANT
  R5  ADE6 of the pair moved from separated-better to unseparated at one seed           RE-MOVES -> RE-NULL;  to separated-worse -> RE-HARMS
  R6  a failed gate in the pinned read                                                  -> NOT-READ
  R7  the pair labelled with the wrong arm; the C0C reference moved; the card prefix edited  -> INPUT-INVALID
"""
import copy
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True                       # the package folder gets no .pyc from this test
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
TPL = {"dom": f"{T}/stage5/BPC/score_dom/DOMINANCE_T2.json", "pair": f"{T}/stage5/BPC/pair_C0C_BPC/PAIR_T2.json", "c0c": f"{T}/stage4/score_q_identity/DOMINANCE_T2.json"}
spec = importlib.util.spec_from_file_location("c0cre_outcome", f"{HERE}/c0cre_outcome.py")
M = importlib.util.module_from_spec(spec)
sys.modules["c0cre_outcome"] = M
spec.loader.exec_module(M)
SB, SW, UN = M.VERDICTS
NUM = {SB: (-0.5, -0.9, -0.1), SW: (0.5, 0.1, 0.9), UN: (0.01, -0.2, 0.2)}
BANNED = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop", "closed-loop")


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def set_verdicts(read, seed, verdicts, pinned=False):
    """Rewrite one seed of a read so rows, family_table (and for a pinned read read_a) all state `verdicts` {row: verdict}; rows not named keep their verdict."""
    bs = read["by_seed"][seed]
    cur = {e["row"]: e["verdict"] for lst in bs["family_table"].values() for e in lst}
    cur.update(verdicts)
    for fam, lst in bs["family_table"].items():
        for e in lst:
            d, lo, hi = NUM[cur[e["row"]]]
            e.update(verdict=cur[e["row"]], delta=d, lo=lo, hi=hi)
    for lab, v in cur.items():
        d, lo, hi = NUM[v]
        bs["rows"][lab].update(delta=d, lo=lo, hi=hi, separated=(v != UN))
    if pinned:
        ra = bs["read_a"]
        ra["separated_worse_rows"] = sorted(r for r, v in cur.items() if v == SW)
        ra["separated_better_rows"] = sorted(r for r, v in cur.items() if v == SB)


def verdicts_of(read, seed):
    return {e["row"]: e["verdict"] for lst in read["by_seed"][seed]["family_table"].values() for e in lst}


def make_dominant(dom, flag):
    for s in M.SEEDS:
        dom["decision"]["per_seed"][s]["DOMINANT"] = flag
        dom["by_seed"][s]["read_a"]["DOMINANT"] = flag
    dom["outcome"] = dom["decision"]["outcome"] = "DOMINANT-T2" if flag else "NOT-DOMINANT"


def base():
    dom, pair, c0c = jload(TPL["dom"]), jload(TPL["pair"]), jload(TPL["c0c"])
    dom["arm"], dom["arm_stage"] = M.ARM, 5
    for s in M.SEEDS:                                   # C0C-RE against P10 = C0C against P10, row for row (so the control has no NEW separated-worse row)
        dom["by_seed"][s]["rows"] = copy.deepcopy(c0c["by_seed"][s]["rows"])
        dom["by_seed"][s]["family_table"] = copy.deepcopy(c0c["by_seed"][s]["family_table"])
        set_verdicts(dom, s, verdicts_of(c0c, s), pinned=True)
    pair["pair"] = f"{M.ARM} - {M.BASE}"
    pair["arm_a"].update(arm=M.ARM, stage=5)
    pair["arm_b"].update(arm=M.BASE, stage=4, root=None)
    pair["arm_vs_P10"] = {M.BASE: {s: copy.deepcopy(c0c["by_seed"][s]["rows"]) for s in M.SEEDS}, M.ARM: {s: copy.deepcopy(dom["by_seed"][s]["rows"]) for s in M.SEEDS}}
    for s in M.SEEDS:                                   # the pair: nothing separated except ADE6 separated-better (RE-MOVES by construction)
        set_verdicts(pair, s, {r: UN for r in M.ROWS} | {"ADE6": SB})
    return dom, pair, c0c


def blind_from(pair):
    b = copy.deepcopy(pair)
    b["pair"] = f"{M.BLIND} - {M.ARM}"
    b["arm_a"].update(arm=M.BLIND)
    b["arm_b"].update(arm=M.ARM, stage=5)
    for s in M.SEEDS:
        set_verdicts(b, s, {"ADE6": SW})
    return b


def run(dom, pair, c0c, blind=None, card=(True, M.CARD_SHA)):
    return M.evaluate(dom, pair, c0c, blind, card, {})


results, fails = [], []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


def main(argv=None):
    out_dir = None
    if argv and "--out-dir" in argv:
        out_dir = argv[argv.index("--out-dir") + 1]
    # control
    dom, pair, c0c = base()
    r0 = run(dom, pair, c0c)
    check("control: RE-MOVES", r0["outcome"] == "RE-MOVES" and r0["exit_code"] == 0, r0["outcome"])
    check("control: C0C's own separated-worse rows are present at both seeds and none is new", all(r0["terms"]["per_seed"][s]["separated_worse_vs_P10"]["C0C"] and r0["terms"]["per_seed"][s]["separated_worse_vs_P10"]["none_new"] for s in M.SEEDS))
    check("control: deterministic (two runs, same JSON)", json.dumps(r0, sort_keys=True) == json.dumps(run(*base()), sort_keys=True))
    # R1
    d, p, c = base()
    set_verdicts(p, "s1", {"M1 crawl share": SW})
    r = run(d, p, c)
    check("R1 pair M1 -> separated-worse at s1 changes RE-MOVES to RE-HARMS", r["outcome"] == "RE-HARMS", r["outcome"])
    # R2
    d, p, c = base()
    set_verdicts(d, "s0", {"speed MAE": SW}, pinned=True)
    p["arm_vs_P10"][M.ARM]["s0"] = copy.deepcopy(d["by_seed"]["s0"]["rows"])
    r = run(d, p, c)
    check("R2 a NEW separated-worse row against P10 (speed MAE, s0) changes RE-MOVES to RE-HARMS", r["outcome"] == "RE-HARMS" and r["terms"]["per_seed"]["s0"]["separated_worse_vs_P10"]["new_in_C0C-RE"] == ["speed MAE"], r["outcome"])
    # R5
    for label, verdicts, want in (("ADE6 unseparated at s1", {"ADE6": UN}, "RE-NULL"), ("ADE6 separated-worse at s0", {"ADE6": SW}, "RE-HARMS")):
        d, p, c = base()
        set_verdicts(p, "s1" if want == "RE-NULL" else "s0", verdicts)
        r = run(d, p, c)
        check(f"R5 {label} -> {want}", r["outcome"] == want, r["outcome"])
    # DOMINANT family
    d, p, c = base()
    for s in M.SEEDS:
        set_verdicts(d, s, {r_: UN for r_ in M.ROWS} | {"ADE6": SB, "speed MAE": SB, "TAC-LON": SB, "S-POS@30": SB}, pinned=True)
        d["by_seed"][s]["read_a"].update(D1=True, D2=True, D3=True, D3_three_families_strategic_paused=True, DOMINANT_three_families_strategic_paused=True)
    make_dominant(d, True)
    p["arm_vs_P10"][M.ARM] = {s: copy.deepcopy(d["by_seed"][s]["rows"]) for s in M.SEEDS}
    d["three_family_form_beside"] = {s: {"D3": True, "DOMINANT": True} for s in M.SEEDS}
    rd = run(d, p, c)
    check("control: pinned DOMINANT-T2 at both seeds -> RE-DOMINANT, replicate required, not counted as the scale signal",
          rd["outcome"] == "RE-DOMINANT" and rd["replicate_required"] is True and rd["counted_as_scale_signal"] is False and rd["also_satisfies_re_moves"] is True, rd["outcome"])
    # R3
    d3 = copy.deepcopy(d)
    for lst in d3["by_seed"]["s1"]["family_table"].values():
        for e in lst:
            if e["row"] == "curvature MAE":
                e["verdict"] = SW
    d3["by_seed"]["s1"]["rows"]["curvature MAE"]["separated"] = True
    d3["by_seed"]["s1"]["read_a"]["separated_worse_rows"] = ["curvature MAE"]
    r = run(d3, p, c)
    check("R3 one family row flipped to separated-worse inside a DOMINANT-T2 claim -> INPUT-INVALID (was RE-DOMINANT)", r["outcome"] == "INPUT-INVALID" and r["exit_code"] == 3, r["outcome"])
    d3b = copy.deepcopy(d)                              # the flip written into rows, family_table AND read_a, with the DOMINANT-T2 claim left standing
    set_verdicts(d3b, "s1", {"curvature MAE": SW}, pinned=True)
    r = run(d3b, p, c)
    check("R3 the flip written consistently everywhere but the DOMINANT-T2 claim left standing (D2 broken) -> INPUT-INVALID", r["outcome"] == "INPUT-INVALID" and any("D2" in x for x in r["integrity"]["problems"]), r["outcome"])
    # R4
    d4 = copy.deepcopy(d)
    set_verdicts(d4, "s1", {"curvature MAE": SW}, pinned=True)
    make_dominant(d4, False)
    p4 = copy.deepcopy(p)
    p4["arm_vs_P10"][M.ARM] = {s: copy.deepcopy(d4["by_seed"][s]["rows"]) for s in M.SEEDS}
    r = run(d4, p4, c)
    check("R4 the same flip written consistently (reader says NOT-DOMINANT) -> not RE-DOMINANT; a new separated-worse row gives RE-HARMS", r["outcome"] == "RE-HARMS", r["outcome"])
    # R6
    d, p, c = base()
    d["outcome"] = d["decision"]["outcome"] = "GATE-FAIL"
    d["gates"]["G3"] = {"PASS": False, "why": ["fixture: a cell is not DONE"]}
    r = run(d, p, c)
    check("R6 a failed gate in the pinned read -> NOT-READ (exit 2), no section-6 outcome", r["outcome"] == "NOT-READ" and r["exit_code"] == 2, r["outcome"])
    d, p, c = base()
    d["outcome"] = d["decision"]["outcome"] = "GATE-FAIL"        # every gate PASS but the reader's own outcome says GATE-FAIL: still not read
    check("R6 the pinned read's own outcome GATE-FAIL with every gate listed PASS -> NOT-READ", run(d, p, c)["outcome"] == "NOT-READ")
    d, p, c = base()
    p["outcome"] = "GATE-FAIL"
    check("R6 the pair read's own outcome GATE-FAIL -> NOT-READ", run(d, p, c)["outcome"] == "NOT-READ")
    # R7
    d, p, c = base()
    p["arm_a"]["arm"] = "BPC"
    check("R7 pair labelled with the wrong arm -> INPUT-INVALID", run(d, p, c)["outcome"] == "INPUT-INVALID")
    d, p, c = base()
    p["arm_vs_P10"][M.BASE]["s0"]["ADE6"]["delta"] += 1e-9
    check("R7 the pair's C0C reference differs from the pin-4 read of C0C by 1e-9 -> INPUT-INVALID", run(d, p, c)["outcome"] == "INPUT-INVALID")
    d, p, c = base()
    check("R7 card prefix edited -> INPUT-INVALID", run(d, p, c, card=(False, "0" * 64))["outcome"] == "INPUT-INVALID")
    d, p, c = base()
    d["script_sha256"] = "0" * 64
    check("R7 the pinned read is not pin 4 -> INPUT-INVALID", run(d, p, c)["outcome"] == "INPUT-INVALID")
    # the real card on disk
    ok, got = M.card_ok(M.CARD)
    check("the card on disk: first 5,544 bytes hash to 425d3496...", ok, (got or "")[:8])
    # BLIND beside
    d, p, c = base()
    rb = run(d, p, c, blind=blind_from(p))
    check("beside: C0C-RE-BLIND - C0C-RE tabulated, outcome unchanged (decides nothing)", rb["outcome"] == "RE-MOVES" and rb["beside_blind"]["decides_nothing"] and not rb["beside_blind"]["problems"] and "ADE6" in rb["beside_blind"]["rows"]["s0"], rb["outcome"])
    bad = blind_from(p)
    bad["pair"] = "BPC - MC"
    check("beside: a BLIND pair with the wrong label is reported as a problem, outcome unchanged", run(d, p, c, blind=bad)["beside_blind"]["problems"] != [] and run(d, p, c, blind=bad)["outcome"] == "RE-MOVES")
    # the .md
    for nm, rr in (("RE-MOVES", r0), ("RE-DOMINANT", rd), ("RE-HARMS", run(*(lambda x: (set_verdicts(x[1], "s1", {"M1 crawl share": SW}) or x))(base()))), ("NOT-READ", r)):
        md = M.render_md(rr)
        low = md.lower()
        check(f"md for {nm}: starts with the outcome, no banned word", md.startswith(f"# RESULT_T2_RENC: {rr['outcome']}") and not any(w in low for w in BANNED), rr["outcome"])
    md = M.render_md(r0)
    check("md: the four families appear for each seed, STRATEGIC marked PAUSED, 19 rows per seed in the pair table",
          all(f in md for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")) and "PAUSED" in md and md.count("| PRIMARY | ADE6 |") >= 3 and md.count("seed s0") == 1 and md.count("seed s1") == 1)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        open(f"{out_dir}/fixture_RE-MOVES.RESULT_T2_RENC.md", "w", encoding="utf-8", newline="\n").write(M.render_md(r0))
        json.dump(r0, open(f"{out_dir}/fixture_RE-MOVES.RESULT_T2_RENC.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"\n{sum(1 for _, ok_, _ in results if ok_)} of {len(results)} checks hold")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
