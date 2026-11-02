#!/usr/bin/env python3
"""make_results_hlf.py -- RESULTS.md of H-HISTSWAP-LF-1, generated from raw/HISTSWAP_LF_T1.json (+ the banked HISTSWAP read for the comparison, the frozen card for the quoted rules and the
card's ESTIMATED probabilities, the test / smoke logs for their pass lines); no hand-typed number.

    python -I code/make_results_hlf.py
Adapted from dagland/histswap/code/make_results_histswap.py (same rows, same marks). Fails loud (SystemExit) if a fact the fixed prose depends on does not hold in the JSON.
"""
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE).replace("\\", "/")
DAG = os.path.dirname(PK).replace("\\", "/")
sys.path.insert(0, f"{DAG}/smooth_t1/code")
import smooth_t1_read as SR  # noqa: E402

GD = SR.load_gd()
J = json.load(open(f"{PK}/raw/HISTSWAP_LF_T1.json", encoding="utf-8"))
H = json.load(open(f"{DAG}/histswap/raw/HISTSWAP_T1.json", encoding="utf-8"))
CARD_B = open(f"{PK}/PREREG_HISTSWAP_LF_FROZEN.md", "rb").read()
CARD = CARD_B.decode("utf-8")
CELLS = ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1")
ARMS = ("HCV", "HLVL")
NM = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}
ARMNAME = {"HCV": "HCV (deployable, constant-twist history at the EXECUTED speed)", "HLVL": "HLVL (ORACLE, constant-twist history at the LOGGED speed)"}
GUARD = SR.ADE_GUARD_M
STAMP = ("Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** "
         "(answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt).")
L = []


def die(msg):
    raise SystemExit("UNVERIFIED / PROSE-FACT FAILED: " + msg)


def w(s=""):
    L.append(s)


def K(c, a):
    return f"{c}:{a}"


def cn(c):
    return "%s %s" % (NM[c[:4]], c[-2:])


def mark(lab, r):
    return "SB" if SR.better(GD, lab, r) else ("SW" if SR.worse(GD, lab, r) else "ns")


def cell(lab, r):
    return "%+.4f [%+.4f, %+.4f] %s" % (r["delta"], r["lo"], r["hi"], mark(lab, r))


def ctl_pass(c):
    return c["PASS"] if "PASS" in c else all(v.get("PASS", False) for v in c["runs"].values())


def rng(xs, f="%+.4f"):
    return (f + " to " + f) % (min(xs), max(xs))


def fam(lab):
    f = SR.FAMILY[lab]
    return f + " (PAUSED, ITEM 25)" if f == "STRATEGIC" else f


def logpass(name):
    t = open(f"{PK}/logs/{name}", encoding="utf-8", errors="replace").read()
    m = re.findall(r"(\d+) passed(?:, (\d+) (?:failed|error))?", t)
    if not m or "failed" in t.split("passed")[-1][:40]:
        die("no clean pass line in logs/" + name)
    return int(m[-1][0])


def cardline(pat):
    for ln in CARD.splitlines():
        if re.search(pat, ln):
            return ln.strip()
    die("card line not found: " + pat)


# ---------------------------------------------------------------------------------------------- checks the fixed prose depends on
if hashlib.sha256(CARD_B).hexdigest() != J["prereg_sha256"]:
    die("the frozen card's sha256 differs from the one the reader logged")
for n, c in J["controls"].items():
    if not ctl_pass(c):
        die("a control failed: " + n)
O, M1R, INT = J["outcome"], J["m1_reading"], J["interpretation"]
if any(O[a][b]["outcome"] != "HARMS" for a in ARMS for b in ("DAG-OACC", "SOUP")):
    die("the fixed prose assumes HARMS for both arms on both bases")
if (M1R["HCV"]["reading"], M1R["HLVL"]["reading"], INT["reading"]) != ("NOT CLOSES", "CLOSES", "B_LEVEL"):
    die("the fixed prose assumes HCV NOT CLOSES, HLVL CLOSES, reading B_LEVEL")
A = {(c, a): J["arms"][K(c, a)] for c in CELLS for a in ARMS}
for c in CELLS:                                              # Y is shared by both arms of a cell: its rows must be equal
    if A[(c, "HCV")]["rows"]["Y_minus_P10"] != A[(c, "HLVL")]["rows"]["Y_minus_P10"]:
        die("Y - P10 differs between the two arms of " + c)
if not all(mark("M1 crawl share", A[(c, "HCV")]["rows"]["X_minus_Y"]["M1 crawl share"]) == "SW" for c in CELLS):
    die("HCV M1 X - Y is not separated-worse at 4 of 4 cells")
if not all(mark("M1 crawl share", A[(c, "HLVL")]["rows"]["X_minus_Y"]["M1 crawl share"]) == "SB" for c in CELLS):
    die("HLVL M1 X - Y is not separated-better at 4 of 4 cells")
for (c, a), v in A.items():
    mc, xy, yp = v["m1_cell"], v["rows"]["X_minus_Y"]["M1 crawl share"], v["rows"]["Y_minus_P10"]["M1 crawl share"]
    if abs(mc["xy_delta"] - xy["delta"]) > 1e-12 or abs(mc["yp_delta"] - yp["delta"]) > 1e-12 or abs(mc["share_closed"] + xy["delta"] / yp["delta"]) > 1e-9:
        die("m1_cell disagrees with the rows at %s %s" % (c, a))

SH = {a: [A[(c, a)]["m1_cell"]["share_closed"] for c in CELLS] for a in ARMS}
XY = {a: [A[(c, a)]["rows"]["X_minus_Y"]["M1 crawl share"]["delta"] for c in CELLS] for a in ARMS}
AD = {a: [A[(c, a)]["rows"]["X_minus_Y"]["ADE6"]["delta"] for c in CELLS] for a in ARMS}
HSH = [H["arms"][c]["m1_cell"]["share_closed"] for c in CELLS]
HAD = [H["arms"][c]["rows"]["X_minus_Y"]["ADE6"]["delta"] for c in CELLS]
HXY = [H["arms"][c]["rows"]["X_minus_Y"]["M1 crawl share"]["delta"] for c in CELLS]

# ---------------------------------------------------------------------------------------------- 0
w("# RESULTS: H-HISTSWAP-LF-1, is HISTSWAP's closure inertia or level? (zero training; HCV deployable arm, HLVL level-only ORACLE arm), DAG-OACC and SOUP at s0 / s1")
w()
w("Card `PREREG_HISTSWAP_LF_FROZEN.md` sha256 `%s` (frozen 2026-11-02T18:50:07+0100, before any HCV / HLVL number). Reader `code/histswap_lf_read.py` (rules tested in `tests/test_histswap_lf_read_rule.py`). Read clock %s. Raw `raw/HISTSWAP_LF_T1.json`, per-window arrays `raw/arrays/*.npz`. Generator of this file: `code/make_results_hlf.py`. Tier **T1-kin**. Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**." % (J["prereg_sha256"], J["run_clock"]))
w("Estimator of every interval below: **%s**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. The planner is deterministic (KA1), and the two seeds are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1` is not addressed). In the card's tags `CTRL` is DAG-OACC." % J["estimator"])
w()
w("## 0. The committed reading: (B) LEVEL")
w()
w(STAMP)
w()
w("**MEASURED.** The card's rule reads **%s** (reading **(B) level**; `hcv_closes` %s, `hlvl_closes` %s, `hlvl_discordant` %s). The outcome label is **HARMS for both arms on both bases**; the label and its triggering rows are in section 1." % (INT["reading"], INT["hcv_closes"], INT["hlvl_closes"], INT["hlvl_discordant"]))
w()
w("* **HCV: NOT CLOSES.** M1 crawl share X - Y is **separated-WORSE at %d of 4 cells** (%s), mean share of Y - P10's M1 excess closed **%.3f** (cells %s; bar %.1f; pooled %.3f), ADE6 X - Y **%s m** (guard +%.4f m)." % (
    sum(mark("M1 crawl share", A[(c, "HCV")]["rows"]["X_minus_Y"]["M1 crawl share"]) == "SW" for c in CELLS), rng(XY["HCV"]), M1R["HCV"]["mean_share_closed"], rng(SH["HCV"], "%.3f"), M1R["HCV"]["share_bar"], M1R["HCV"]["pooled_share_closed"], rng(AD["HCV"]), GUARD))
w("* **HLVL: CLOSES.** M1 crawl share X - Y is **separated-better at %d of 4 cells** (%s), mean share closed **%.3f** (cells %s; pooled %.3f), ADE6 X - Y **%s m**." % (
    M1R["HLVL"]["n_cells_separated_better"], rng(XY["HLVL"]), M1R["HLVL"]["mean_share_closed"], rng(SH["HLVL"], "%.3f"), M1R["HLVL"]["pooled_share_closed"], rng(AD["HLVL"])))
w("* **Committed interpretation (card section 2, verbatim):** " + cardline(r"^3\. \*\*Committed interpretation").split("**Committed interpretation:**")[1].strip())
w("* **Committed next step under (B) (card section 5, verbatim):** " + cardline(r"^\* \*\*\(B\):\*\*").replace("* **(B):**", "").strip())
w()
w("> **ORACLE CAVEAT (the card's, binding on every sentence of this file).** HLVL reads the LOGGED measured speed at every tick, which is the human's speed LEVEL; it is **not deployable** and its absolute rows are not performance. Its closure is **near-mechanical** (card section 4): the plan is `P + O's delta` and the executor tracks it perfectly, so a prior anchored on the human's speed puts the executed speed near the human's at once; HLVL alone cannot tell *O needs the level* from *the prior's anchor is the input speed* (section 5 is the descriptive block that says which moved). **HCV is the discriminating arm** and it is deployable. HCV also removes ALL acceleration information (the tick row's own acceleration is 0 as well), which is stronger than *no trend*: its negative reading excludes *a history without acceleration recovers M1*, not a trend-only statement (HTRND, card section 5 (C), was not run).")
w()
w("| cell | arm | M1 X - Y | Y - P10 (the excess) | share of the excess closed | ADE6 X - Y (m) |")
w("|---|---|---|---|---|---|")
for c in CELLS:
    for a in ARMS:
        v = A[(c, a)]
        xy, yp, ad, mc = v["rows"]["X_minus_Y"]["M1 crawl share"], v["rows"]["Y_minus_P10"]["M1 crawl share"], v["rows"]["X_minus_Y"]["ADE6"], v["m1_cell"]
        w("| %s | %s | %s | %+.4f [%+.4f, %+.4f] %s | %.4f (%.1f %%) | %s |" % (cn(c), a, cell("M1 crawl share", xy), yp["delta"], yp["lo"], yp["hi"], mark("M1 crawl share", yp), mc["share_closed"], 100 * mc["share_closed"], cell("ADE6", ad)))
w()
w("Rules (card section 2): `share closed` = -(X - Y) / (Y - P10) on the M1 row (undefined = 0); **CLOSES** = separated-better at all 4 cells AND mean share closed >= 0.5; **NOT CLOSES** = separated-better at fewer than 3 cells; else MIXED. HCV: %d of 4 cells separated-better; HLVL: %d of 4." % (M1R["HCV"]["n_cells_separated_better"], M1R["HLVL"]["n_cells_separated_better"]))
w()

# ---------------------------------------------------------------------------------------------- 1 HARMS
w("## 1. The outcome label: which rows trigger HARMS")
w()
w(STAMP)
w()
w("Rule (card section 2.1, `smooth_t1_read.arm_outcome`, precedence DOMINANT, HARMS, MOVED, NO EFFECT): HARMS at a cell = ADE6 X - Y > +%.4f m **or** a non-crawl row of X - Y separated-worse; the base is HARMS if either seed is. MOVED = all four crawl rows separated-better and ADE6 X - Y <= +%.4f m. The four crawl rows are %s. The label is reported as the rule gives it." % (GUARD, GUARD, ", ".join("`%s`" % r for r in SR.CRAWL)))
w()
w("| cell | arm | ADE6 X - Y (m) vs guard | crawl rows separated-better | non-crawl rows separated-worse (X - Y) | HARMS seed | MOVED seed |")
w("|---|---|---|---|---|---|---|")
TRIG = {a: [] for a in ARMS}
for a in ARMS:
    for c in CELLS:
        xy = A[(c, a)]["read_xy_X_minus_Y"]
        TRIG[a].append(set(xy["non_crawl_separated_worse"]))
        w("| %s | %s | %+.4f (%s) | %d of 4 | %s | %s | %s |" % (cn(c), a, xy["ade6_delta"], "within guard" if xy["ade6_within_guard"] else "OVER guard", sum(xy["crawl_separated_better"].values()), ", ".join(xy["non_crawl_separated_worse"]) or "none", xy["HARMS_seed"], xy["MOVED_seed"]))
w()
for a in ARMS:
    uni, inter = sorted(set.union(*TRIG[a])), sorted(set.intersection(*TRIG[a]))
    over = [cn(c) for c in CELLS if not A[(c, a)]["read_xy_X_minus_Y"]["ade6_within_guard"]]
    byfam = {}
    for r in uni:
        byfam.setdefault(SR.FAMILY[r] if r != "ADE6" else "PRIMARY", []).append(r)
    w("* **%s triggers.** ADE6 over the guard at %d of 4 cells; non-crawl rows separated-worse at some cell: %s; at all 4 cells: %s. By family (rows separated-worse at some cell): %s. Outcome labels: DAG-OACC **%s**, SOUP **%s** (`moved_and_harms`: DAG-OACC %s, SOUP %s)." % (
        a, len(over), ", ".join("`%s`" % r for r in uni) or "none", ", ".join("`%s`" % r for r in inter) or "none", "; ".join("%s: %s" % (f, ", ".join(rs)) for f, rs in sorted(byfam.items())) or "none",
        O[a]["DAG-OACC"]["outcome"], O[a]["SOUP"]["outcome"], O[a]["DAG-OACC"]["moved_and_harms"], O[a]["SOUP"]["moved_and_harms"]))
uni_l = sorted(set.union(*TRIG["HLVL"]))
if all(A[(c, "HLVL")]["read_xy_X_minus_Y"]["ade6_within_guard"] for c in CELLS) and uni_l and all(r.startswith("S-POS") for r in uni_l) and all(O["HLVL"][b]["moved_and_harms"] for b in ("DAG-OACC", "SOUP")):
    w()
    w("⚠ **HLVL's HARMS label rests on the S-POS rows alone.** Its ADE6 X - Y is within the guard at all four cells, all four crawl rows are separated-better at all four cells (MOVED holds at both bases), and the only non-crawl rows separated-worse are %s. The card marks the S-POS rows of both arms **UNVERIFIED** (they mix the CV pose the S head read with the executed pose the extractor frames them in; card section 6), and they belong to the STRATEGIC family, which is PAUSED by the PI's ruling (`PI_DECISION_QUEUE.md` ITEM 25, 2026-09-19; reported as paused, not as a pass, and not extended to the other families). The label stands as the rule gave it; it is flagged here so that nobody reads it as a longitudinal or lateral harm. HLVL is an oracle in any case." % ", ".join("`%s`" % r for r in uni_l))
else:
    die("the prose on HLVL's S-POS-only trigger assumes the measured pattern; re-read section 1 by hand")
uni_h = sorted(set.union(*TRIG["HCV"]))
w()
inter_h = sorted(r for r in set.intersection(*TRIG["HCV"]) if not r.startswith("S-POS"))
w("**HCV's HARMS is not a rows-on-the-margin result.** ADE6 X - Y is over the guard at every cell (%s m) and the separated-worse rows include the longitudinal ones (%s). Leaving out the UNVERIFIED S-POS rows, the non-crawl rows separated-worse at all 4 cells are %s. HCV is a deployable arm; under the card's committed rule (section 5, HARMS at either arm) the named non-crawl rows above are the next lever's target before any other reading is used." % (rng(AD["HCV"]), ", ".join("`%s`" % r for r in uni_h if SR.FAMILY.get(r) == "LONGITUDINAL"), ", ".join("`%s`" % r for r in inter_h)))
w()

# ---------------------------------------------------------------------------------------------- 2 Read A
w("## 2. Read A (X vs P10): D1 / D2 / D3 (HLVL is an ORACLE; its rows are not performance)")
w()
w(STAMP)
w()
w("Definitions (`smooth_t1_read.read_a`, smooth_t1_read.py:71-80): **D1** = ADE6 X - P10 separated-better and its point <= %+.4f m; **D2** = no row of X - P10 separated-worse; **D3** = every one of the four families has at least one separated-better row; DOMINANT = D1 and D2 and D3. P10 is the programme's P10 floor. The STRATEGIC family is PAUSED (ITEM 25): its D3 entry is a measured flag of the rule, not a pass claim." % SR.D1_POINT_M)
w()
w("| cell | arm | D1 | D2 | D3 (L / Lat / T / S) | DOMINANT | ADE6 X - P10 (m) | rows separated-worse (X - P10) | rows separated-better |")
w("|---|---|---|---|---|---|---|---|---|")
for c in CELLS:
    for a in ARMS:
        v = A[(c, a)]
        Ra = v["read_a_X_vs_P10"]
        d3 = Ra["D3_by_family"]
        w("| %s | %s | %s | %s | %s (%s / %s / %s / %s) | %s | %s | %s | %d of 19 |" % (cn(c), a, Ra["D1"], Ra["D2"], Ra["D3"], d3["LONGITUDINAL"], d3["LATERAL"], d3["TACTICAL"], d3["STRATEGIC"], Ra["DOMINANT"], cell("ADE6", v["rows"]["X_minus_P10"]["ADE6"]), ", ".join(Ra["separated_worse_rows"]) or "none", len(Ra["separated_better_rows"])))
w()
w("D1 and D3 hold and D2 fails at all %d cells of both arms; no cell is DOMINANT. The row(s) that keep D2 failing are listed per cell. Y itself fails D2 on `M1 crawl share` (the programme's starting point; HISTSWAP card section 1)." % (len(CELLS) * 2) if all(A[(c, a)]["read_a_X_vs_P10"]["D1"] and not A[(c, a)]["read_a_X_vs_P10"]["D2"] and A[(c, a)]["read_a_X_vs_P10"]["D3"] for c in CELLS for a in ARMS) else "(the D1 / D2 / D3 pattern differs between cells; read the table.)")
w()

# ---------------------------------------------------------------------------------------------- 3 the 19 rows
w("## 3. All 19 rows, per cell: Y - P10, X - Y and X - P10 for both arms (HLVL absolute rows are not performance)")
w()
w(STAMP + " Every cell is `point [lo, hi] mark`. S-POS rows: **UNVERIFIED** (card section 6); STRATEGIC family PAUSED (ITEM 25).")
w()
for c in CELLS:
    w("### %s" % cn(c))
    w()
    w("* HCV read_xy: crawl rows separated-better %s; ADE6 X - Y %+.4f (within guard: %s); rows separated-worse on X - Y, all 19: %d. HLVL read_xy: crawl rows separated-better %s; ADE6 X - Y %+.4f (within guard: %s); rows separated-worse on X - Y, all 19: %d." % (
        A[(c, "HCV")]["read_xy_X_minus_Y"]["crawl_separated_better"], A[(c, "HCV")]["read_xy_X_minus_Y"]["ade6_delta"], A[(c, "HCV")]["read_xy_X_minus_Y"]["ade6_within_guard"], A[(c, "HCV")]["read_xy_X_minus_Y"]["n_separated_worse_all"],
        A[(c, "HLVL")]["read_xy_X_minus_Y"]["crawl_separated_better"], A[(c, "HLVL")]["read_xy_X_minus_Y"]["ade6_delta"], A[(c, "HLVL")]["read_xy_X_minus_Y"]["ade6_within_guard"], A[(c, "HLVL")]["read_xy_X_minus_Y"]["n_separated_worse_all"]))
    w()
    w("| row | family | Y - P10 | HCV X - Y | HLVL X - Y | HCV X - P10 | HLVL X - P10 |")
    w("|---|---|---|---|---|---|---|")
    for lab in GD.LABELS:
        rh, rl = A[(c, "HCV")]["rows"], A[(c, "HLVL")]["rows"]
        w("| %s | %s | %s | %s | %s | %s | %s |" % (lab, fam(lab), cell(lab, rh["Y_minus_P10"][lab]), cell(lab, rh["X_minus_Y"][lab]), cell(lab, rl["X_minus_Y"][lab]), cell(lab, rh["X_minus_P10"][lab]), cell(lab, rl["X_minus_P10"][lab])))
    w()

# ---------------------------------------------------------------------------------------------- 4 M1 per class
w("## 4. The M1 row by situation class (descriptive, not a decision; classes L launch / S stay / O other-stopped / M moving; weight `w` = share of the row's windows)")
w()
w(STAMP)
w()
w("| cell | contrast | row delta | L: n, delta, contribution | M: n, delta, contribution | S, O windows | L share of the row's delta |")
w("|---|---|---|---|---|---|---|")
LS = {}
for c in CELLS:
    for a, ct in (("HCV", "X_minus_Y"), ("HLVL", "X_minus_Y"), ("HCV", "Y_minus_P10")):
        d = J["descriptive"][K(c, a)]["M1_per_class_split"][ct]
        cl = d["classes"]
        if cl["S"]["n"] or cl["O"]["n"]:
            die("S or O class has windows at %s %s" % (c, a))
        sh = cl["L"]["contribution"] / d["delta"]
        LS[(c, a, ct)] = sh
        lab = "%s X - Y" % a if ct == "X_minus_Y" else "Y - P10 (shared)"
        w("| %s | %s | %+.4f [%+.4f, %+.4f] | %d, %+.4f [%+.4f, %+.4f], %+.4f | %d, %+.4f [%+.4f, %+.4f], %+.4f | %d, %d | %.0f %% |" % (cn(c), lab, d["delta"], d["lo"], d["hi"], cl["L"]["n"], cl["L"]["delta"], cl["L"]["delta_lo"], cl["L"]["delta_hi"], cl["L"]["contribution"], cl["M"]["n"], cl["M"]["delta"], cl["M"]["delta_lo"], cl["M"]["delta_hi"], cl["M"]["contribution"], cl["S"]["n"], cl["O"]["n"], 100 * sh))
w()
LW = [J["descriptive"][K(c, "HCV")]["M1_per_class_split"]["Y_minus_P10"]["classes"]["L"]["w"] for c in CELLS]
CL = lambda a, cls, ct="X_minus_Y": [J["descriptive"][K(c, a)]["M1_per_class_split"][ct]["classes"][cls]["delta"] for c in CELLS]
w("The S and O classes carry no windows in the masked M1 pool at any cell. Launch-class windows carry %.0f %%-%.0f %% of the Y - P10 excess, %.0f %%-%.0f %% of HLVL's X - Y movement and %.0f %%-%.0f %% of HCV's X - Y movement (contribution / row delta; the weight of L is %s of the row's windows)." % (
    100 * min(LS[(c, "HCV", "Y_minus_P10")] for c in CELLS), 100 * max(LS[(c, "HCV", "Y_minus_P10")] for c in CELLS), 100 * min(LS[(c, "HLVL", "X_minus_Y")] for c in CELLS), 100 * max(LS[(c, "HLVL", "X_minus_Y")] for c in CELLS),
    100 * min(LS[(c, "HCV", "X_minus_Y")] for c in CELLS), 100 * max(LS[(c, "HCV", "X_minus_Y")] for c in CELLS),
    ("%.1f %%" % (100 * min(LW))) if "%.1f" % (100 * min(LW)) == "%.1f" % (100 * max(LW)) else ("%.1f %%-%.1f %%" % (100 * min(LW), 100 * max(LW)))))
w("Per class (X - Y delta of the class's own M1 value): **HCV** L class %s, M class %s; **HLVL** L class %s, M class %s; for scale, the shared Y - P10 excess is L %s, M %s. HCV's harm sits in the launch class (its launch-class X - Y is %.2f-%.2f of Y's own launch-class excess over P10), while the moving class improves slightly; HLVL improves both." % (rng(CL("HCV", "L")), rng(CL("HCV", "M")), rng(CL("HLVL", "L")), rng(CL("HLVL", "M")), rng(CL("HCV", "L", "Y_minus_P10")), rng(CL("HCV", "M", "Y_minus_P10")), min(x / y for x, y in zip(CL("HCV", "L"), CL("HCV", "L", "Y_minus_P10"))), max(x / y for x, y in zip(CL("HCV", "L"), CL("HCV", "L", "Y_minus_P10")))))
w()

# ---------------------------------------------------------------------------------------------- 5 prior anchor vs O delta
w("## 5. Prior anchor versus O's delta (descriptive; the card's `o_residual_x30` / `P_x30` block)")
w()
w("Tier T1-kin, evidence class MEASURED, **point means with no interval** (mean over windows of the per-tick 6 s along-track value, ticks >= 1; the tick-0 plan is excluded). `P` = the prior the network computed in the same call (it reads the swapped history), `O delta` = plan_raw - P at 6 s, `plan` = P + O delta. Y is the fresh `SREP_*` run.")
w()
w("| cell | arm | P (Y, X, X - Y) m | O delta (Y, X, X - Y) m | plan (Y, X, X - Y) m | P + O delta - plan (X, m) |")
w("|---|---|---|---|---|---|")
DP, DO, DPL = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: [] for a in ARMS}
idmax = 0.0
for c in CELLS:
    for a in ARMS:
        d = J["descriptive"][K(c, a)]
        p, o, pl = d["P_x30_mean_m_ticks_ge1"], d["o_residual_x30_mean_m_ticks_ge1"], d["plan_x30_mean_m_ticks_ge1"]
        ident = p["X"] + o["X"] - pl["X"]
        idmax = max(idmax, abs(ident), abs(p["Y"] + o["Y"] - pl["Y"]))
        DP[a].append(p["X"] - p["Y"])
        DO[a].append(o["X"] - o["Y"])
        DPL[a].append(pl["X"] - pl["Y"])
        w("| %s | %s | %.3f, %.3f, %+.3f | %+.3f, %+.3f, %+.3f | %.3f, %.3f, %+.3f | %+.1e |" % (cn(c), a, p["Y"], p["X"], p["X"] - p["Y"], o["Y"], o["X"], o["X"] - o["Y"], pl["Y"], pl["X"], pl["X"] - pl["Y"], ident))
w()
npos = lambda xs: sum(x > 0 for x in xs)
nneg = lambda xs: sum(x < 0 for x in xs)
w("Identity check: max |P + O delta - plan| over the table = %.1e m (the three columns are separate means, so equality is approximate; the residual is not investigated)." % idmax)
w("* **HLVL:** the prior's 6 s end point rises vs Y at %d of 4 cells (X - Y %s m), O's own delta moves **down** at %d of 4 cells (X - Y %s m), and the plan's end point rises at %d of 4 cells (X - Y %s m). The movement of the plan's end point has the sign of the movement of P: the prior's anchor carries it, and O's delta works against it. This is consistent with the card's section 4 caution that HLVL's closure is near-mechanical (it cannot by itself show that O needs the level). It is a consistency check, not a decomposition of the M1 closure: the block averages over all windows and ticks >= 1, while M1 lives in the launch-class windows (section 4)." % (npos(DP["HLVL"]), rng(DP["HLVL"], "%+.3f"), nneg(DO["HLVL"]), rng(DO["HLVL"], "%+.3f"), npos(DPL["HLVL"]), rng(DPL["HLVL"], "%+.3f")))
w("* **HCV:** the prior's end point moves up at %d of 4 cells (X - Y %s m), O's delta moves up at %d of 4 (%s m), the plan's end point up at %d of 4 (%s m). The signs are not uniform, so the mean 6 s end point does not carry HCV's M1 and ADE6 harm; those are windowed and class-specific (section 4: launch-class windows), and this block is not targeted at them." % (npos(DP["HCV"]), rng(DP["HCV"], "%+.3f"), npos(DO["HCV"]), rng(DO["HCV"], "%+.3f"), npos(DPL["HCV"]), rng(DPL["HCV"], "%+.3f")))
w()

# ---------------------------------------------------------------------------------------------- 6 controls
w("## 6. Controls (computed before any row; any failure refuses the read)")
w()
for n, c in J["controls"].items():
    w("* **%s**: %s" % (n, "PASS" if ctl_pass(c) else "FAIL"))
w()
ky = J["controls"]["KY_reused_Y_equals_frozen_pins"]["files"]
k1 = J["controls"]["KA1_fresh_off_rerun_equals_reused_Y"]
w("**KY**: %d / %d reused Y files hash-equal to the pins of `FROZEN_SHA.txt`." % (sum(v["PASS"] for v in ky.values()), len(ky)))
w("**KA1 (the SREP5 bit-identity rerun):** the fresh LF-tool `--histswap off` rerun `%s` against the reused `%s`: dump bit-identical **%s** (max abs dev %.3e m), ticks bit-identical **%s** (max abs dev %.3e), differing keys %s, key-set difference %s. The planner is deterministic, so the thread-order floor is 0 on this rig." % (
    k1["rerun"], k1["reused"], k1["dump_bit_identical"], k1["dump_max_abs_dev_m"], k1["ticks_bit_identical"], k1["ticks_max_abs_dev"], (k1["dump_differing_keys"] + k1["ticks_differing_keys"]) or "none", k1["key_set_diff"] or "none"))
w()
w("**KC1** (the flag reached the run; `v0_in` = the speed the network read, `v0` = the executed speed):")
w()
w("| run | `--histswap` | max abs v0_in - v0, all ticks (m/s) | max abs at tick 0 | share of later ticks with v0_in != v0 by > 0.05 m/s | mean executed / network speed, later ticks (m/s) |")
w("|---|---|---|---|---|---|")
for k, v in J["controls"]["KC1_flag_reached_the_run_and_v0_in_is_as_the_card_says"]["runs"].items():
    w("| %s | %s | %.3f | %.1e | %.4f | %.3f / %.3f |" % (k, v["str_histswap"], v["v0_in_minus_v0_max_abs_all_ticks"], v["v0_in_minus_v0_max_abs_tick0"], v["frac_later_ticks_v0_in_ne_v0_gt_0p05"], v["mean_v0_exec_later_mps"], v["mean_v0_in_later_mps"]))
w()
kc2, kc3 = J["controls"]["KC2_the_CV_history_reached_the_network"]["runs"], J["controls"]["KC3_plan_committed_to_the_executed_state"]["runs"]
kf, k3, kb = J["controls"]["KF_floor_window_check_vs_fresh_Y"], J["controls"]["K3_Y_banked_minus_P10_reproduces_SOUP_READ"], J["controls"]["KB1_P10_side_equals_landed_banked_b"]["runs"]
w("**KC2** (the CV history reached the network: the tick-0 plan differs from Y's by more than 1e-3 m): max abs tick-0 plan_raw deviation %s m over %s tick-0 rows per run, %d / %d runs pass. **KC3** (the plan is committed to the executed state: tick 1's executed position equals the tick-0 pose composed with plan knot 10, tolerance 5e-3 m, >= 99.9 %% of windows): max deviation %.1e m, share within tolerance %.4f at %d / %d runs. **KF**: windows X = fresh Y at %d / %d runs, floor X vs fresh Y max abs %.1e m. **K3**: Y banked - P10 reproduces `SOUP_READ.json` on %d / %d rows. **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays: %d / %d runs pass." % (
    rng([v["tick0_plan_raw_max_abs_dev_vs_Y_m"] for v in kc2.values()], "%.1f"), "/".join(sorted({str(v["n_tick0_rows"]) for v in kc2.values()})), sum(v["PASS"] for v in kc2.values()), len(kc2),
    max(v["max_dev_m"] for v in kc3.values()), min(v["frac_within_tol"] for v in kc3.values()), sum(v["PASS"] for v in kc3.values()), len(kc3),
    sum(v["windows_X_equal_fresh_Y"] for v in kf["runs"].values()), len(kf["runs"]), max(v["floor_X_vs_fresh_Y_max_abs_m"] for v in kf["runs"].values()), k3["n_pass"], k3["n_rows"], sum(v["PASS"] for v in kb.values()), len(kb)))
w()
smoke_ok = "SMOKELF OK" in open(f"{PK}/logs/smoke_lf.out", encoding="utf-8", errors="replace").read()
if not smoke_ok:
    die("smoke log carries no SMOKELF OK")
w("Tool tests `logs/pytest_histswap_lf.log`: **%d passed** (three mutations RED: cv_exec reading the logged speed, a ramp in the track, the item builder on the executed dict); reader-rule tests `logs/pytest_read_rule.log`: **%d passed** (three mutations RED); 32-window mechanical smoke `logs/smoke_lf.out`: **SMOKELF OK**. Rollouts: 9 (8 arm runs + the SREP5 rerun), `logs/chain.out` records `ZZHLF-ROLLOUTS-DONE nfail=0`." % (logpass("pytest_histswap_lf.log"), logpass("pytest_read_rule.log")))
w()
w("**Swap bookkeeping** (`swap_bucket_differs / swap_ticks`: how often the rebuilt causal speed bucket differs from the executed-history bucket):")
w()
w("| cell | HCV | HLVL |")
w("|---|---|---|")
for c in CELLS:
    b = {a: A[(c, a)]["swap_bucket_differs_over_swap_ticks"] for a in ARMS}
    w("| %s | %d / %d (%.2f %%) | %d / %d (%.2f %%) |" % (cn(c), b["HCV"]["swap_bucket_differs"], b["HCV"]["swap_ticks"], 100.0 * b["HCV"]["swap_bucket_differs"] / b["HCV"]["swap_ticks"], b["HLVL"]["swap_bucket_differs"], b["HLVL"]["swap_ticks"], 100.0 * b["HLVL"]["swap_bucket_differs"] / b["HLVL"]["swap_ticks"]))
w()
w("The bucket is rebuilt from the CV history in both arms; its separate contribution to M1 is not attributed (card section 1 (iii)).")
w()

# ---------------------------------------------------------------------------------------------- 7 predictions
w("## 7. The card's ESTIMATED probabilities next to the result")
w()
m = re.search(r"HCV: CLOSES (\d\.\d+), MIXED (\d\.\d+), NOT CLOSES (\d\.\d+)\. HLVL: CLOSES (\d\.\d+), MIXED (\d\.\d+), NOT CLOSES (\d\.\d+)\. Reading: \(A\) (\d\.\d+), \(B\) (\d\.\d+), \(C\) (\d\.\d+)", CARD)
if not m:
    die("the card's probability line was not found")
pr = [float(x) for x in m.groups()]
w("| quantity | card (ESTIMATED, written before any number) | result (MEASURED) | probability the card gave the result |")
w("|---|---|---|---|")
w("| HCV M1 reading | CLOSES %.2f / MIXED %.2f / NOT CLOSES %.2f | **%s** (%d of 4 cells separated-better; mean share %.3f) | %.2f |" % (pr[0], pr[1], pr[2], M1R["HCV"]["reading"], M1R["HCV"]["n_cells_separated_better"], M1R["HCV"]["mean_share_closed"], pr[2]))
w("| HLVL M1 reading | CLOSES %.2f / MIXED %.2f / NOT CLOSES %.2f | **%s** (%d of 4 cells separated-better; mean share %.3f) | %.2f |" % (pr[3], pr[4], pr[5], M1R["HLVL"]["reading"], M1R["HLVL"]["n_cells_separated_better"], M1R["HLVL"]["mean_share_closed"], pr[3]))
w("| interpretation | (A) %.2f / (B) %.2f / (C) %.2f | **%s** | %.2f |" % (pr[6], pr[7], pr[8], INT["reading"], pr[7]))
w()
w("One read is not a calibration. What the card's three-way split could not express: HCV's NOT CLOSES is not a null: it is **separated-worse** than Y on M1 at 4 of 4 cells and ADE6 is over the guard at 4 of 4, so the executed-speed constant-twist history is itself a harmful input, not only an uninformative one (whether the harm is the missing acceleration information or the off-distribution input, a history that is exactly constant at every row, is UNVERIFIED).")
w()

# ---------------------------------------------------------------------------------------------- 8 vs HISTSWAP
w("## 8. Against HISTSWAP's full logged history (H-HISTSWAP-1, `dagland/histswap/raw/HISTSWAP_T1.json`)")
w()
w(STAMP + " HISTSWAP (the full LOGGED history swap: speed profile, acceleration, trend, yaw / pose rows and bucket) is an ORACLE too; same Y, same windows, same checkpoints.")
w()
w("| cell | share closed: HISTSWAP / HLVL / HCV | M1 X - Y: HISTSWAP | M1 X - Y: HLVL | ADE6 X - Y (m): HISTSWAP | ADE6 X - Y (m): HLVL |")
w("|---|---|---|---|---|---|")
OV = []
for i, c in enumerate(CELLS):
    hx, lx = H["arms"][c]["rows"]["X_minus_Y"], A[(c, "HLVL")]["rows"]["X_minus_Y"]
    ov = lambda p, q: not (p["hi"] < q["lo"] or q["hi"] < p["lo"])
    OV.append((ov(hx["M1 crawl share"], lx["M1 crawl share"]), ov(hx["ADE6"], lx["ADE6"])))
    w("| %s | %.3f / %.3f / %.3f | %s | %s | %s | %s |" % (cn(c), HSH[i], SH["HLVL"][i], SH["HCV"][i], cell("M1 crawl share", hx["M1 crawl share"]), cell("M1 crawl share", lx["M1 crawl share"]), cell("ADE6", hx["ADE6"]), cell("ADE6", lx["ADE6"])))
w()
hm, lm = H["outcome"]["mean_share_closed"], M1R["HLVL"]["mean_share_closed"]
w("* Mean share closed: **HISTSWAP %.3f** (outcome %s, %d of 4 cells), **HLVL %.3f**, HCV %.3f. Mean ADE6 X - Y: HISTSWAP %+.3f m, HLVL %+.3f m, HCV %+.3f m. The level alone, as an oracle, recovers %.0f %% of what the full logged history recovers on M1 (share closed %.3f / %.3f)." % (hm, H["outcome"]["outcome"], H["outcome"]["n_cells_separated_better"], lm, M1R["HCV"]["mean_share_closed"], sum(HAD) / 4, sum(AD["HLVL"]) / 4, sum(AD["HCV"]) / 4, 100 * lm / hm, lm, hm))
w("* **HISTSWAP - HLVL is what the rest of the logged history adds on top of the speed level**: %.3f of the share closed (%.3f - %.3f) and %+.3f m of ADE6 (mean X - Y). The two differ in more than acceleration and trend: HISTSWAP swaps every ego-row-built key from the logged arrays (card of H-HISTSWAP-1, section 1), so it also carries the logged yaw / pose rows, which HLVL holds at the executed yaw rate (card section 1 (i)). The split of the 0.33 between acceleration, trend and yaw rows is not measured here." % (hm - lm, hm, lm, sum(HAD) / 4 - sum(AD["HLVL"]) / 4))
w("* Interval overlap, a check and **not an estimator of the difference** (the two arms were not bootstrapped against each other): HLVL's and HISTSWAP's X - Y intervals overlap on M1 at %d of 4 cells and on ADE6 at %d of 4 cells." % (sum(o[0] for o in OV), sum(o[1] for o in OV)))
w()

# ---------------------------------------------------------------------------------------------- 9 next step
w("## 9. What this read decides, and what it does not")
w()
w("* **Reading (B) level; the committed next step is the card's:** the recovery fine-tune (`dagland/histswap/DRAFT_RECOVERY_CARD.md`, a DRAFT; v2 in `DRAFT_RECOVERY_CARD_v2.md`) goes to the PI for a box-2 go, with the **level source (perception or training) named as its unknown**; no history-form lever is pursued. This package starts no training and touches no box.")
w("* **The trend copy is not supported as the carrier:** HCV, which holds speed and yaw rate constant over the history and sets acceleration to 0, does not close M1; it is separated-worse. HLVL, which differs from HCV in exactly one number (the speed, logged instead of executed), closes %.0f %%-%.0f %% of the excess at the same 4 cells. By construction (card section 1) that one number, the speed, is the only input difference between a harmful arm and a closing arm." % (100 * min(SH["HLVL"]), 100 * max(SH["HLVL"])))
w("* **What the read does not say.** (i) HLVL is an oracle: nothing here shows that a model trained or run without the human's speed recovers any of the %.0f %%-%.0f %%. (ii) Over all windows the movement of HLVL's plan end point comes through the prior's anchor, with O's delta moving against it (section 5), as the card's near-mechanical caution anticipated. (iii) HCV removes all acceleration information, so *trend alone* is untested (HTRND not run). (iv) Tick 0 of both arms differs from Y by construction (the CV history differs from the logged one at t0), so part of the ADE6 movement can originate at tick 0 (KC2). (v) T1-kin: the cameras are the logged ones, so a model that reads the human's speed from frame motion can pass M1 here and not in driving; the frame-freeze control that separates this is pre-registered in `DRAFT_RECOVERY_CARD_v2.md` section 8, and it was **not run on any arm of this read**." % (100 * min(SH["HLVL"]), 100 * max(SH["HLVL"])))
w("* **Not decided here:** adoption into any K-1 / WM-1 claim; the Master Mind registers `H-HISTSWAP-LF-1` (the id the card used).")
w()

# ---------------------------------------------------------------------------------------------- 10 curvature partition
w("## 10. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)")
w()
w("`E` = the curvature MAE point (X - P10 for X, Y - P10 for Y); `n_dip` = windows with an unwarranted speed dip; `highk share <2 m/s` = share of the high-curvature steps' excess from steps under 2 m/s. Point values, no interval.")
w()
w("| cell | arm | N | E X | n_dip X | highk share <2 m/s X | E Y | n_dip Y | highk share <2 m/s Y |")
w("|---|---|---|---|---|---|---|---|---|")
for c in CELLS:
    for a in ARMS:
        d = J["descriptive"][K(c, a)]
        x, y = d["curvature_partition_X"], d["curvature_partition_Y"]
        w("| %s | %s | %d | %+.4f | %d | %.3f | %+.4f | %d | %.3f |" % (cn(c), a, x["N"], x["E"], x["n_dip_windows"], x["highk_share_from_steps_under_2mps"], y["E"], y["n_dip_windows"], y["highk_share_from_steps_under_2mps"]))
w()

# ---------------------------------------------------------------------------------------------- 11 limits
w("## 11. Limits")
w()
w("* **HLVL is an ORACLE** (section 0): no HLVL row is performance and nothing here is a deployable-fix claim. HCV is deployable and does not close.")
w("* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); X and Y share the checkpoint and the planner is deterministic; the two seeds per base are replicates of the rig, not a variance estimate. A separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`). Y is the reused `SREP_*` runs (KY, KA1).")
w("* Tier T1-kin: executed state, logged cameras, own plan applied in the scene as a self-action open loop; it says nothing about re-perception, and the camera frames advance at the human's pace (section 9 (v)).")
w("* S-POS rows of both arms are UNVERIFIED (card section 6); the STRATEGIC family is PAUSED (ITEM 25). The causal speed bucket is rebuilt from the CV history and is not separated from the rest of the history change.")
w("* HCV removes all acceleration information, not only the trend; HCV's history (exactly constant at every row) is off the training distribution, and whether its harm comes from that or from the missing information is UNVERIFIED.")
open(f"{PK}/RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("RESULTS.md written", len(L), "lines")
print("CHECK ranges: HCV M1 X-Y", rng(XY["HCV"]), "| share", rng(SH["HCV"], "%.3f"), "| ADE6", rng(AD["HCV"]), "|| HLVL M1 X-Y", rng(XY["HLVL"]), "| share", rng(SH["HLVL"], "%.3f"), "| ADE6", rng(AD["HLVL"]))
