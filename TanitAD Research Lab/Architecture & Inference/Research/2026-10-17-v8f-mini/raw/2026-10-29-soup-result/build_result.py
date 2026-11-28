#!/usr/bin/env python
"""build_result.py: renders RESULT_V8F_SOUP.md from the read of record.

Every number in a table or in the prose comes from SOUP_READ.json (or is
arithmetic on it); the prose that quotes the register rows / the PREREG is
checked by check_result.py, which does NOT import this file and re-parses the
finished markdown.

Usage: python -I build_result.py [--json PATH] [--out PATH]
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--json", default="F:/TanitAD-scratch/soup_t1_out/read/SOUP_READ.json")
ap.add_argument("--out", default=os.path.join(HERE, "RESULT_V8F_SOUP.md"))
ARGS = ap.parse_args()

RAWB = open(ARGS.json, "rb").read()
JSON_SHA = hashlib.sha256(RAWB).hexdigest()
assert JSON_SHA.startswith("dca10c07"), JSON_SHA
D = json.loads(RAWB.decode("utf-8"))
T = D["tables"]
DASH = "\u2013"
MINUS = "\u2212"


def n4(x):
    s = "%+.4f" % x
    return "+0.0000" if s == "-0.0000" else s


def u4(x):
    return "%.4f" % x


def flag(c):
    return "B" if c["separated_better"] else ("W" if c["separated_worse"] else DASH)


def cell(c, bold=True):
    s = "%s [%s, %s] %s" % (n4(c["delta"]), n4(c["lo"]), n4(c["hi"]), flag(c))
    return "**" + s + "**" if (c["separated"] and bold) else s


def cs(ct, s, r):
    """a cell as quoted in prose (no bold)"""
    return cell(T[ct][s][r], bold=False)


def nr(n):
    return "unpaired vs CTRV" if n.startswith("ticks>=1") else n


LONG = ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"]
LAT = ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"]
TAC = ["TAC-LAT", "TAC-LON"]
STR = ["S-POS@30", "S-POS@50", "S-POS@80"]
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
ALL19 = ["ADE6"] + LONG + LAT + ["M1 crawl share"] + TAC + STR
assert sorted(ALL19) == sorted(T["SOUP_CTRL"]["s0"].keys())
FAMS = [("LONGITUDINAL", LONG), ("LATERAL", LAT), ("TACTICAL", TAC), ("STRATEGIC", STR)]
BETTER = {r: T["SOUP_CTRL"]["s0"][r]["direction"] for r in ALL19}
BAR = D["floors"]["bar"]
RFB = D["floors"]["RF"]
RF0 = D["floors"]["RF0"]
FVAL = {r: (0.1207 if r == "ADE6" else 0.0) for r in ALL19}
COLS = [("s0", "main"), ("s0", "rep"), ("s1", "main"), ("s1", "rep")]
PR = {
    "main": dict(lab="main (SOUP %s DAG-OACC)" % MINUS, X="SOUP", Y="DAG-OACC", XY="SOUP_CTRL", XP="SOUP_P10", YP="CTRL_P10", fxy="SOUP-CTRL", fxp="SOUP-P10"),
    "rep": dict(lab="replicate (SOUP' %s A)" % MINUS, X="SOUP'", Y="A", XY="SOUPR_A", XP="SOUPR_P10", YP="A_P10", fxy="SOUPR-A", fxp="SOUPR-P10"),
}
HALF = 0.0604

out = []


def w(s=""):
    out.append(s)


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w()


def worsen(r, delta):
    return delta if BETTER[r] == "lower" else -delta


def pair_calc(s, p):
    q = PR[p]
    XY, XP, YP = T[q["XY"]][s], T[q["XP"]][s], T[q["YP"]][s]
    dom = [r for r in ALL19 if not YP[r]["separated_worse"]]
    xyw = [r for r in dom if XY[r]["separated_worse"]]
    xyb = [r for r in xyw if worsen(r, XY[r]["delta"]) >= BAR[r] - 1e-12]
    fire = [r for r in xyb if XP[r]["separated_worse"]]
    return dict(dom=dom, xyw=xyw, xyb=xyb, fire=fire)


def names(l):
    return ", ".join(l) if l else "none"


PS = D["pairs"]
OUT = D["OUTCOME"]
SG = D["sg"]
RA = D["read_a"]
BS = D["beside"]
word = OUT["OUTCOME"]
assert word == "HARMS" and D["route"] == "FLAT" and OUT["controls_pass"] is True and len(D["controls_PASS"]) == 368 and all(D["controls_PASS"].values())

# ---------------------------------------------------------------- header
CRAWL_W = [(s_, p_, r_) for s_ in ("s0", "s1") for p_ in ("main", "rep") for r_ in CRAWL if T[PR[p_]["XY"]][s_][r_]["separated_worse"]]
CRAWL_B = [(s_, p_, r_) for s_ in ("s0", "s1") for p_ in ("main", "rep") for r_ in CRAWL if T[PR[p_]["XY"]][s_][r_]["separated_better"]]
CRAWL_W_S0 = [x for x in CRAWL_W if x[0] == "s0"]
w("# RESULT_V8F_SOUP: a uniform same-init weight average (\"soup\") of two DAG-OACC-recipe O fine-tunes, read at T1-kin. Outcome **HARMS** (not adopted)")
w()
w("Claim row `H-V8F-SOUP-1`. Pre-registration `Project Steering/PREREG_V8F_SOUP.md`, promoted at `e03b348` (2026-10-29 02:00:06; sha256 `830d9c91…`, status line and section 0 item 5 changed at promotion, nothing else). Read 2026-10-29 by the pre-registered reader `soup_read.py` (sha256 `de39aefe…`; it imports the LAUNCH-INTENT reader, sha256 `fb3d2d7d…`, unchanged): all **368 reader controls PASS** (368 of 368), reader wall 634.9 s, rc 0. Output `raw/2026-10-29-soup-read/out/read/SOUP_READ.json` (sha256 `dca10c07…`), banked at `0c7b598` (2026-10-29 03:10:30).")
w()
a0m, a0r = T["SOUP_CTRL"]["s0"]["ADE6"], T["SOUPR_A"]["s0"]["ADE6"]
a1m, a1r = T["SOUP_CTRL"]["s1"]["ADE6"], T["SOUPR_A"]["s1"]["ADE6"]
w("* **The word, as registered.** **HARMS.** The uniform two-draw soup is not adopted. The guard set H fails at s0, in both pairs: R-ADE fails in the main pair (ADE6, in m, SOUP_0 %s DAG-OACC_0: %s) and in the replicate pair (ADE6, in m, SOUP'_0 %s A_0: %s), and R-TLAT fails in the main pair (reader boolean). At s1 every guard holds. The registered rule is that H must hold at both seeds, so the s1 pass does not change the word (PREREG section 10). Nothing below softens that." % (MINUS, cell(a0m, False), MINUS, cell(a0r, False)))
w("* **No crawl row moved by its bar, and no crawl row reaches P10.** R-MOVE is false in all four pairs and R-CRAWL is false in all four pairs. Of the 16 crawl cells X %s Y, %d are separated-better and %d are separated-worse (%d of them at s0, where the soup carries part of the slow member's crawl). Read A reads **%s** for SOUP, SOUP' and SOUP4 at both seeds: D1 (ADE6 better than P10 by at least 2F) and D3 (each family has a separated-better row) hold, D2 fails on the same four crawl rows." % (MINUS, len(CRAWL_B), len(CRAWL_W), len(CRAWL_W_S0), RA["SOUP"]["outcome"]))
w("* **Tier.** T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop'. The R-T0 points and the slow-draw cells are at T0 (teacher-forced own-code, eval-141, ages cycle, dev-box CPU on both arms of a pair). Never 'driving performance'.")
w("* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, same windows and matched cells (JSON `estimator`). Intervals are 95 % (INHERITED: the convention of `RESULT_V8F_LAUNCH_INTENT.md`; the JSON stores the bounds only). It answers *\"would another draw of EPISODES say this?\"* and nothing else. *\"Would another TRAINING RUN say this?\"* is answered only by the replicate pair at each seed and the second seed (`H-ESTIM-SEED-1`). The planner is deterministic and a soup is a fixed set of weights, so no inference-seed question arises.")
w("* **Arms** (zero training; every member banked from the LAUNCH-INTENT and DAG-OACC runs). SOUP_s = DAG-OACC_s + REP_s (donor DAG-OACC_s); SOUP'_s = A_s + A_rep,s (donor A_s; both members carry `--launch-intent-w` 1.58926); SOUP4_s = all four, beside. Weights are averaged in float64 and rounded once to float32; the soups share the init M2b_s with their members. Control of the main pair: DAG-OACC_s. Control of the replicate pair: A_s (PREREG section 0 item 1).")
w("* **Coverage.** The four metric families at T1-kin are in section 4, per seed, per pair, never pooled. **The T0 four families are not in this read** (section 4.7). The R-TLAT / R-TLON macro-F1 values, the T1-kin goal-head F1 and head accuracy, and the anti-echo block are not computed by the rig (sections 2.3 and 4.7). They are work items, and the outcome above does not use them.")
w()
w("**Summary** (each value is in the section named; evidence class MEASURED (ours) unless stated).")
w()
def sgs(key):
    g = SG[key]
    fl = "B" if (g["separated"] and g["delta"] < 0) else ("W" if g["separated"] else DASH)
    return "%s [%s, %s] %s" % (n4(g["delta"]), n4(g["lo"]), n4(g["hi"]), fl)


w("* **HARMS (section 1).** Controls pass. At s0: R-ADE fails in both pairs, R-TLAT fails in the main pair, and the adoption row R-KEEP fails in both pairs (point %s and %s against +0.0604). At s1: every guard, R-KEEP and R-NOBASE hold. R-MOVE and R-CRAWL are false everywhere." % (n4(PS["s0"]["main"]["KEEP_point"]), n4(PS["s0"]["rep"]["KEEP_point"])))
w("* **The soup gain is real and there is no barrier (section 3).** SG on ADE6 (the soup minus the mean of its members' errors, m): SOUP %s (s0), %s (s1); SOUP' %s (s0), %s (s1); SOUP4 %s (s0), %s (s1). Five of six cells are separated-better; SOUP at s1 is not separated. Routing: **FLAT**, so the committed next is SOUP-K." % (sgs("SOUP:s0"), sgs("SOUP:s1"), sgs("SOUPR:s0"), sgs("SOUPR:s1"), sgs("SOUP4:s0"), sgs("SOUP4:s1")))
w("* **Mechanism (section 6, reading, decides nothing).** At s0 the slow draw REP_0 is %s m worse than DAG-OACC_0 on ADE6. The soup recovers most of it (SOUP_0 %s REP_0 %s) and still lands worse than the good draw (SOUP_0 %s DAG-OACC_0 %s)." % (n4(T["REP_CTRL"]["s0"]["ADE6"]["delta"]), MINUS, cs("SOUP_REP", "s0", "ADE6"), MINUS, cs("SOUP_CTRL", "s0", "ADE6")))
w("* **Next (section 7).** Committed, as registered: SOUP-K (FLAT route; 4 box-2 runs; its own PI go). Rule Zero: the crawl diagnostics say the crawl of DAG-OACC s0 sits at the outcome-blind floor (on the point), S_v is the frozen prior's, the M2b checkpoints measured are FIT-LIMITED on TRAIN, the images add little go/stay information beyond ego (none on green-light or lead-departure clips), and the trunk's go/stay edge is the declared speed-ceiling and nav oracles; averaging can shrink the draw-specific part of the crawl and cannot move the part every member shares, so the lever the evidence points to is perception. A finer-token DINOv3 probe is PROPOSED, not registered.")
w()

# ---------------------------------------------------------------- 1
w("## 1. The outcome, as the table decides it")
w()
w("Evidence class MEASURED (ours). Tier and estimator as in the header: T1-kin unless a cell says T0; paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Source: `pairs.<seed>.<main|rep>` and `OUTCOME` in the JSON. X is the soup of the pair (SOUP in the main pair, SOUP' in the replicate pair); Y is the pair's control (DAG-OACC, A). The R-TLAT and R-TLON cells are the reader's guard booleans; the values behind them are not in the JSON (section 2.3).")
w()
hdr = ["predicate (PREREG section 10)", "s0 main (SOUP %s DAG-OACC)" % MINUS, "s0 replicate (SOUP' %s A)" % MINUS, "**s0, both pairs**", "s1 main (SOUP %s DAG-OACC)" % MINUS, "s1 replicate (SOUP' %s A)" % MINUS, "**s1, both pairs**"]


def hf(b):
    return "held" if b else "FAILS"


def tf(b):
    return "true" if b else "false"


def row_pred(name, getter):
    cells = []
    both = {}
    for s in ("s0", "s1"):
        vs = []
        for p in ("main", "rep"):
            b, txt = getter(s, p)
            vs.append(b)
            cells.append(txt)
        both[s] = all(vs)
    return [name, cells[0], cells[1], "**%s**" % ("held" if both["s0"] else "FAILS"), cells[2], cells[3], "**%s**" % ("held" if both["s1"] else "FAILS")]


def row_pred_tf(name, getter):
    r = row_pred(name, getter)
    r[3] = r[3].replace("held", "true").replace("FAILS", "false")
    r[6] = r[6].replace("held", "true").replace("FAILS", "false")
    return r


rows = []
rows.append(["controls (reader, 368)", "PASS", "PASS", "**PASS**", "PASS", "PASS", "**PASS**"])
rows.append(row_pred("R-TLAT (T head lat macro-F1, ticks >= 1, X %s Y not separated-worse)" % MINUS, lambda s, p: (PS[s][p]["guards"]["R-TLAT"], hf(PS[s][p]["guards"]["R-TLAT"]))))
rows.append(row_pred("R-TLON (T head lon macro-F1, ticks >= 1, X %s Y not separated-worse)" % MINUS, lambda s, p: (PS[s][p]["guards"]["R-TLON"], hf(PS[s][p]["guards"]["R-TLON"]))))
rows.append(row_pred("R-ADE (T1-kin ADE6, X %s Y, not separated-worse)" % MINUS, lambda s, p: (PS[s][p]["guards"]["R-ADE"], "%s: %s %s" % (hf(PS[s][p]["guards"]["R-ADE"]), n4(T[PR[p]["XY"]][s]["ADE6"]["delta"]), flag(T[PR[p]["XY"]][s]["ADE6"])))))
rows.append(row_pred("R-T0 (T0 own-code ADE6, X %s Y, point <= +0.0604 m)" % MINUS, lambda s, p: (PS[s][p]["guards"]["R-T0"], "%s: %s" % (hf(PS[s][p]["guards"]["R-T0"]), n4(PS[s][p]["T0_point"])))))
rows.append(row_pred("R-NEW (no row of D fires)", lambda s, p: (PS[s][p]["guards"]["R-NEW"], "%s: %d of %d rows" % (hf(PS[s][p]["guards"]["R-NEW"]), sum(PS[s][p]["R-NEW_fires"].values()), len(PS[s][p]["domain"])))))
rows.append(row_pred_tf("R-MOVE (every crawl row clears its bar)", lambda s, p: (PS[s][p]["flags"]["R-MOVE"], "%s: %d of 4 rows clear" % (tf(PS[s][p]["flags"]["R-MOVE"]), sum(PS[s][p]["R-MOVE_rows"].values())))))
rows.append(row_pred_tf("R-CRAWL (every crawl row of X %s P10 not separated-worse)" % MINUS, lambda s, p: (PS[s][p]["flags"]["R-CRAWL"], "%s: %d of 4 rows" % (tf(PS[s][p]["flags"]["R-CRAWL"]), sum(PS[s][p]["R-CRAWL_rows"].values())))))
rows.append(row_pred_tf("R-KEEP (T1-kin ADE6, X %s Y, point <= +0.0604 m)" % MINUS, lambda s, p: (PS[s][p]["flags"]["R-KEEP"], "%s: %s" % (tf(PS[s][p]["flags"]["R-KEEP"]), n4(PS[s][p]["KEEP_point"])))))
rows.append(row_pred_tf("R-NOBASE", lambda s, p: (PS[s][p]["flags"]["R-NOBASE"], "%s: %d of %d rows violate" % (tf(PS[s][p]["flags"]["R-NOBASE"]), sum(PS[s][p]["R-NOBASE_violations"].values()), len(PS[s][p]["domain"])))))
table(hdr, rows)
w("Per-seed booleans (`OUTCOME.per_seed`; PREREG section 10). Each holds at a seed only when it holds in both pairs of that seed.")
w()
ps = OUT["per_seed"]
table(["seed", "H (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW, both pairs)", "MV (R-MOVE, both pairs)", "AD (R-KEEP and R-NOBASE, both pairs)", "DM (Read A D1, D2, D3 for SOUP and SOUP')"],
      [[s, tf(ps[s]["H"]), tf(ps[s]["MV"]), tf(ps[s]["AD"]), tf(ps[s]["DM"])] for s in ("s0", "s1")])
w("**HARMS.** The table's condition, verbatim from PREREG section 10: *\"controls pass; H fails at ≥ 1 seed\"*. H fails at s0 (R-ADE in both pairs, R-TLAT in the main pair), so the word is HARMS. REFUSED is decided first and HARMS second; the controls pass. Committed next, verbatim from the JSON: *\"the uniform two-draw soup is not adopted. NEXT-SOUP (§11)\"*, with `next_soup` = %s and `route` = %s." % (OUT["next_soup"], D["route"]))
w()
w("⚠ **Reported as written, beside the word.** At s1 alone every guard and both adoption rows hold (H and AD true), and the lever still moves no crawl row (MV false) and does not reach P10 on the crawl rows (DM false). The word is not taken seed by seed: PREREG section 10 requires H at both seeds. The adoption row R-KEEP also fails at s0 in both pairs (section 2.2), so the soup would not have replaced DAG-OACC at s0 even without the guards.")
w()
w("The word is the reader's output (`matching_rows_in_table_order`: one row, HARMS). `check_result.py` recomputes R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0 and R-KEEP per pair from the JSON's cells, floors and points, derives H, MV, AD, DM and the word with the PREREG section 10 precedence, and finds every flag equal to the reader's. R-TLAT and R-TLON are taken from the reader (their statistic is not in the JSON).")
w()

# ---------------------------------------------------------------- 2
w("## 2. The decisive rows")
w()
w("Tier T1-kin unless a heading says T0. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Evidence class MEASURED (ours): every cell is a copy of `tables.<contrast>.<seed>.<row>` in the JSON. Cells read point [95 %% CI] and a flag: B = separated-better, W = separated-worse, %s = not separated. Bold = separated. P10 is the kinematic floor. The `n` of a cell is windows / episodes." % DASH)
w()
w("### 2.1 The floors and the bar (`floors`)")
w()
w("`bar_r = 2 × max(F_r, RF0_r)`, reused unchanged from `PREREG_V8F_LAUNCH_INTENT` section 6 (PREREG_V8F_SOUP section 7), one value per row, applied at both seeds and in both pairs, inclusive. F is 0.1207 m on ADE6 and 0 on every other row (PREREG). RF0 is the frozen same-seed cross-build floor (`P0_RF0.json`, sha256 `1608d70b…`). RF is the larger of two REP %s DAG-OACC draws, reported beside every row; it is not a bar term. Here RF is also the main soup's own member spread." % MINUS)
w()
ORDER = ["ADE6"] + LONG + LAT[:2] + [LAT[2], LAT[3], LAT[4], LAT[5]] + TAC + STR + ["M1 crawl share"]
ORDER = ["ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share"]
assert sorted(ORDER) == sorted(ALL19)
rows = []
for r in ORDER:
    rows.append([r, BETTER[r], u4(FVAL[r]), u4(RF0[r]), "**%s**" % u4(BAR[r]), u4(RFB[r]), "%.2f" % (RFB[r] / BAR[r])])
table(["row", "better", "F", "RF0", "**bar** = 2 × max(F, RF0)", "RF (beside, not a bar term)", "RF / bar"], rows)

w("### 2.2 ADE6: the guard R-ADE and the adoption row R-KEEP (T1-kin)")
w()
rows = []
for s in ("s0", "s1"):
    for p in ("main", "rep"):
        q = PR[p]
        xy, xp, yp = T[q["XY"]][s]["ADE6"], T[q["XP"]][s]["ADE6"], T[q["YP"]][s]["ADE6"]
        rows.append([s, q["lab"], "ADE6", cell(xy), u4(xy["mean_arm"]), u4(xy["mean_ref"]), cell(xp), cell(yp), xy["n"]])
table(["seed", "pair", "row", "X %s Y" % MINUS, "mean X", "mean Y", "X %s P10" % MINUS, "Y %s P10" % MINUS, "n (windows / episodes)"], rows)
w("R-ADE needs X %s Y not separated-worse. It fails in both pairs at s0 and holds in both pairs at s1; at s1 the replicate pair is separated-better (SOUP'_1 %s A_1). R-KEEP reads the same contrast's point against +0.0604 m and fails at s0 in both pairs. R-ADE and R-KEEP read one contrast, as the PREREG discloses." % (MINUS, MINUS))
w()
w("### 2.3 R-TLAT, R-TLON, R-T0 and R-KEEP")
w()
w("R-T0 is at tier T0 (a point, no interval in the JSON; dev-box CPU on both arms of the pair, K-T0DEV). R-KEEP is T1-kin. The TAC-LAT and TAC-LON cells are the rig's **plan-derived agreements** (JSON `four_families` note), shown beside the guard booleans; they are not the statistic the guards test.")
w()
rows = []
for s in ("s0", "s1"):
    for p in ("main", "rep"):
        q = PR[p]
        g = PS[s][p]
        rows.append([s, q["lab"], cell(T[q["XY"]][s]["TAC-LAT"]), hf(g["guards"]["R-TLAT"]), cell(T[q["XY"]][s]["TAC-LON"]), hf(g["guards"]["R-TLON"]), "%s" % n4(g["T0_point"]), hf(g["guards"]["R-T0"]), n4(g["KEEP_point"]), tf(g["flags"]["R-KEEP"])])
table(["seed", "pair", "TAC-LAT X %s Y (plan-derived agreement)" % MINUS, "R-TLAT (reader)", "TAC-LON X %s Y (plan-derived agreement)" % MINUS, "R-TLON (reader)", "T0 own-code ADE6, X %s Y (point)" % MINUS, "R-T0 (<= +0.0604)", "T1-kin ADE6, X %s Y (point)" % MINUS, "R-KEEP (<= +0.0604)"], rows)
t_lat0 = T["SOUP_CTRL"]["s0"]["TAC-LAT"]
t_lon0r = T["SOUPR_A"]["s0"]["TAC-LON"]
w("R-TLAT and R-TLON are booleans in the JSON; the T-head macro-F1 values behind them are not carried (UNVERIFIED as numbers). On TAC-LAT the plan-derived cell and the guard agree in all four pairs: in the s0 main pair both fail (TAC-LAT, SOUP_0 vs DAG-OACC_0: %s). On TAC-LON they do not: in the s0 replicate pair the cell is separated-worse (%s) while the reader's R-TLON holds. The two statistics differ, so a TAC cell is never read as the guard. The word does not depend on it: H already fails at s0 through R-ADE in both pairs. The register row `H-V8F-SOUP-1` labels the %s cell \"TAC-LAT macro-F1\"; in the JSON it is the TAC-LAT row (disclosure 6)." % (cell(t_lat0, False), cell(t_lon0r, False), n4(t_lat0["delta"])))
w()
w("### 2.4 R-MOVE: the crawl rows, X %s Y against the bar (T1-kin)" % MINUS)
w()
w("R-MOVE needs, in each pair, every crawl row separated-better with an improvement of at least the bar. Improvement is %sdelta (all four rows are lower-is-better). The crawl set is yaw-rate MAE, curvature MAE, L2 curvature MAE (GT >= 5.2) and M1 crawl share (`pairs.*.crawl_set`)." % MINUS)
w()
rows = []
nclear = 0
nb = 0
for s in ("s0", "s1"):
    for p in ("main", "rep"):
        q = PR[p]
        for r in CRAWL:
            c = T[q["XY"]][s][r]
            imp = -c["delta"]
            cl = PS[s][p]["R-MOVE_rows"][r]
            nclear += cl
            nb += c["separated_better"]
            rows.append([s, q["lab"], r, cell(c), n4(imp), u4(BAR[r]), "%.1f %%" % (100.0 * imp / BAR[r]), tf(cl)])
table(["seed", "pair", "crawl row", "X %s Y" % MINUS, "improvement", "bar", "improvement / bar", "clears (R-MOVE row)"], rows)
w("%d of the 16 cells are separated-better and %d clear their bar. No pair clears all four rows. %d cells are separated-worse (%d at s0, %d at s1): at s0 the soup is worse than its control on every crawl row in both pairs, the soup carrying part of the slow member's crawl (section 6, Fact 2b)." % (nb, nclear, len(CRAWL_W), len(CRAWL_W_S0), len(CRAWL_W) - len(CRAWL_W_S0)))
w()
w("### 2.5 R-CRAWL: the crawl rows, X %s P10 (T1-kin)" % MINUS)
w()
w("The share closed is (Y %s P10 %s (X %s P10)) / (Y %s P10) from the point deltas; a negative share means the gap to P10 widens. MEASURED (arithmetic on the JSON)." % (MINUS, MINUS, MINUS, MINUS))
w()
rows = []
for s in ("s0", "s1"):
    for p in ("main", "rep"):
        q = PR[p]
        for r in CRAWL:
            xp, yp = T[q["XP"]][s][r], T[q["YP"]][s][r]
            sh = 100.0 * (yp["delta"] - xp["delta"]) / yp["delta"]
            rows.append([s, q["lab"], r, cell(xp), cell(yp), tf(PS[s][p]["R-CRAWL_rows"][r]), "%.1f %%" % sh])
table(["seed", "pair", "crawl row", "X %s P10" % MINUS, "Y %s P10" % MINUS, "X %s P10 not separated-worse (R-CRAWL row)" % MINUS, "share of the Y %s P10 gap closed" % MINUS], rows)
w("All 16 X %s P10 crawl cells are separated-worse. So are the Y %s P10 cells in every pair: that is how the crawl set was fixed." % (MINUS, MINUS))
w()
w("### 2.6 R-NEW and R-NOBASE (T1-kin; domain D is every row not separated-worse for Y %s P10)" % MINUS)
w()
w("R-NEW fires on a row of D when (1) X %s P10 is separated-worse, (2) Y %s P10 is not (the row is in D), and (3) X %s Y is separated-worse with a worsening of at least the bar. R-NOBASE reads (1) and (3) over D. Worsening is delta on lower-is-better rows and %sdelta on higher-is-better rows." % (MINUS, MINUS, MINUS, MINUS))
w()
rows = []
for s in ("s0", "s1"):
    for p in ("main", "rep"):
        q = PR[p]
        pc = pair_calc(s, p)
        rows.append([s, q["lab"], "%s %s P10 not separated-worse on D" % (q["Y"], MINUS), str(len(pc["dom"])), names(pc["xyw"]), names(pc["xyb"]), names(pc["fire"])])
table(["seed", "pair", "control Y", "rows in D", "X %s Y separated-worse (rows of D)" % MINUS, "of those, worsening >= bar", "of those, X %s P10 separated-worse (R-NEW fires)" % MINUS], rows)
w("**R-NEW and R-NOBASE hold at every seed and pair because condition (1) fails, not because condition (3) does.** Rows of D that meet condition (3): s0 main, %s; s0 replicate, %s; s1 main, %s; s1 replicate, %s. None of them has X %s P10 separated-worse. The registered predicate is applied as written." % (names(pair_calc("s0", "main")["xyb"]), names(pair_calc("s0", "rep")["xyb"]), names(pair_calc("s1", "main")["xyb"]), names(pair_calc("s1", "rep")["xyb"]), MINUS))
w()

w("### 2.7 Read A: X %s P10 (`PREREG_V8F_DAG_REREAD` section 2 and 5; T1-kin)" % MINUS)
w()
w("D1: ADE6 separated-better with point <= " + MINUS + "2F = " + MINUS + "0.2415 m (PREREG_V8F_SOUP section 7). D2: no row of the 19 separated-worse. D3: each family has a separated-better row. The label is the reader's `read_a.<soup>.outcome`; its rule is `PREREG_V8F_DAG_REREAD` section 5 (*MARGIN, NOT DOMINANT*: D1 at both seeds, D2 fails at >= 1 seed; if the failing rows are crawl rows, that PREREG names DAgger iteration 2, which this read does not route).")
w()
rows = []
for key, ct, lab in (("SOUP", "SOUP_P10", "SOUP"), ("SOUPR", "SOUPR_P10", "SOUP'"), ("SOUP4", "SOUP4_P10", "SOUP4 (beside)")):
    for s in ("s0", "s1"):
        ra = RA[key]["per_seed"][s]
        c = T[ct][s]["ADE6"]
        fam_ok = ", ".join(f for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "PRIMARY") if ra["D3_families"][f])
        rows.append([lab, s, cell(c), tf(ra["D1"]), "%s (%d rows: %s)" % (tf(ra["D2"]), len(ra["D2_worse_rows"]), ", ".join(ra["D2_worse_rows"])), tf(ra["D3"]), RA[key]["outcome"]])
table(["soup", "seed", "ADE6 X %s P10" % MINUS, "D1", "D2 (rows separated-worse)", "D3", "Read A label (soup level)"], rows)
w("D3 families that hold, from `D3_families` (all five keys true in every row of the table above): LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC and PRIMARY (ADE6). Read A is the same state for all three soups and both seeds, and it is the state of the control: DAG-OACC %s P10 and A %s P10 are separated-worse on the same four crawl rows at both seeds (section 2.5, Y %s P10 column)." % (MINUS, MINUS, MINUS))
w()

# ---------------------------------------------------------------- 3 SG
w("## 3. The soup gain SG and the routing variable (T1-kin)")
w()
w("SG on ADE6 (PREREG section 7): per window, sg_w = ADE6 of the soup %s the mean of its members' ADE6, causal lane; paired episode-cluster bootstrap, B = 2,000. SG < 0 separated: the weight average beats the average of its members' errors. SG > 0 separated: a **barrier**. Source: `sg.<soup>:<seed>` (`SOUPR` is SOUP'). The arithmetic columns recompute the mean difference from the table means (`mean_arm`, `mean_ref`, full precision in the JSON) and agree with SG to the rounding shown (the dropped-window count is 0 in every cell)." % MINUS)
w()


def sgcalc(key, s):
    if key == "SOUP":
        sm = T["SOUP_CTRL"][s]["ADE6"]["mean_arm"]
        mem = [T["SOUP_CTRL"][s]["ADE6"]["mean_ref"], T["SOUP_REP"][s]["ADE6"]["mean_ref"]]
    elif key == "SOUPR":
        sm = T["SOUPR_A"][s]["ADE6"]["mean_arm"]
        mem = [T["SOUPR_A"][s]["ADE6"]["mean_ref"], T["SOUPR_Arep"][s]["ADE6"]["mean_ref"]]
    else:
        sm = T["SOUP4_CTRL"][s]["ADE6"]["mean_arm"]
        mem = [T["SOUP4_CTRL"][s]["ADE6"]["mean_ref"], T["REP_CTRL"][s]["ADE6"]["mean_arm"], T["A_P10"][s]["ADE6"]["mean_arm"], T["SOUPR_Arep"][s]["ADE6"]["mean_ref"]]
    return sm, mem


rows = []
for key, lab, mems in (("SOUP", "SOUP (decisive)", "DAG-OACC, REP"), ("SOUPR", "SOUP' (decisive)", "A, A_rep"), ("SOUP4", "SOUP4 (beside)", "DAG-OACC, REP, A, A_rep")):
    for s in ("s0", "s1"):
        g = SG["%s:%s" % (key, s)]
        sm, mem = sgcalc(key, s)
        mm = sum(mem) / len(mem)
        fl = "B" if (g["separated"] and g["delta"] < 0) else ("W" if g["separated"] else DASH)
        cellsg = "%s [%s, %s] %s" % (n4(g["delta"]), n4(g["lo"]), n4(g["hi"]), fl)
        if g["separated"]:
            cellsg = "**" + cellsg + "**"
        rows.append([lab, s, mems, cellsg, tf(g["barrier"]), u4(sm), ", ".join(u4(m) for m in mem), u4(mm), n4(sm - mm), str(g["n_windows"])])
table(["soup", "seed", "members", "SG (JSON)", "barrier", "soup mean ADE6", "member means (in the order of the members)", "mean of members", "soup %s mean of members (arithmetic)" % MINUS, "n windows"], rows)
w("**Routing (PREREG section 11).** BARRIER = SG separated-worse (> 0) in at least one of the four decisive soup-seed cells (SOUP and SOUP' at both seeds). Here no cell is a barrier (`barrier` false in all six, `separated-worse` in none), so the route is **FLAT** (JSON `route`). FLAT means the weights average at least as well as the metrics; the harm or the shortfall is the members' draws. Committed next: **SOUP-K** (section 7.1). ENS2, the barrier branch, is not triggered.")
w()

# ---------------------------------------------------------------- 4 families
w("## 4. The four families (T1-kin; per family, per seed, per pair, never pooled)")
w()
w("Tier: T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop'. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per pair. Evidence class MEASURED (ours); source `four_families.<pair>:<seed>.<family>.rows`, which equals the `tables` cells cell for cell (asserted by `check_result.py`). **STRATEGIC is not paused for v8F (PI ruling 2026-10-17); the JSON carries `paused: false` and the ruling string in every STRATEGIC block.** Each table shows X %s Y, the one-lever contrast of the pair, then X %s P10 in section 4.6." % (MINUS, MINUS))
w()
w("### 4.1 Separated cells per family, per seed, per pair (B / W / not separated; X %s Y)" % MINUS)
w()
rows = []
for fam, rl in FAMS + [("crawl row outside the family blocks (M1)", ["M1 crawl share"])]:
    cells = []
    for (s, p) in COLS:
        b = sum(T[PR[p]["XY"]][s][r]["separated_better"] for r in rl)
        wv = sum(T[PR[p]["XY"]][s][r]["separated_worse"] for r in rl)
        cells.append("%d / %d / %d" % (b, wv, len(rl) - b - wv))
    rows.append(["%s (%d)" % (fam, len(rl))] + cells)
table(["family (rows)", "s0 main: B / W / not separated", "s0 replicate: B / W / not separated", "s1 main: B / W / not separated", "s1 replicate: B / W / not separated"], rows)


def fam_table(rl, title_note=None):
    rows = []
    for r in rl:
        cl = [cell(T[PR[p]["XY"]][s][r]) for (s, p) in COLS]
        ns = " ; ".join(nr(T[PR[p]["XY"]][s][r]["n"]) for (s, p) in COLS)
        rows.append([r, BETTER[r]] + cl + [ns])
    table(["row", "better", "s0 main (SOUP %s DAG-OACC)" % MINUS, "s0 replicate (SOUP' %s A)" % MINUS, "s1 main (SOUP %s DAG-OACC)" % MINUS, "s1 replicate (SOUP' %s A)" % MINUS, "n, s0 main / s0 replicate / s1 main / s1 replicate"], rows)


w("### 4.2 LONGITUDINAL (speed, along-track, accel, distance keeping)")
w()
fam_table(LONG)
w("Not built for this family: the anti-echo block (JSON note: *\"the anti-echo block is NOT BUILT (the DAG-ORC rig carries none).\"*). M-LAUNCH is a separate artifact and is not part of this read.")
w()
w("### 4.3 LATERAL (read first: the crawl rows are LATERAL plus M1)")
w()
fam_table(LAT)
w("The M1 crawl share is a crawl row outside the LATERAL block of the JSON; its X %s Y cells:" % MINUS)
w()
fam_table(["M1 crawl share"])
w("### 4.4 TACTICAL")
w()
fam_table(TAC)
w("These are the plan-derived agreements. The JSON note: *\"T1-kin goal-head F1 and head accuracy are NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); TAC-LAT / TAC-LON are the plan-derived agreements; R-TLAT / R-TLON are the guards.\"* The guard booleans are in section 2.3.")
w()
w("### 4.5 STRATEGIC (not paused for v8F)")
w()
fam_table(STR)
w("S-POS@30 / @50 / @80 (lower is better). The X %s Y cell is the difference of the two runs' (S %s CTRV, ticks >= 1) contrasts on the same windows (JSON `kind`). Goal and route-setting quality (the strategic decision itself) is not a row of this read, and whether another tool computes it for these arms is UNVERIFIED here." % (MINUS, MINUS))
w()
w("### 4.6 Against P10: X %s P10 and Y %s P10, every row (T1-kin)" % (MINUS, MINUS))
w()
w("X %s P10 for the decisive soups (SOUP in the main pair, SOUP' in the replicate pair). The STRATEGIC X %s P10 cells are unpaired against CTRV (JSON `n`: *ticks>=1, unpaired vs CTRV*); every other cell is paired. Lower is better except TAC-LAT and TAC-LON." % (MINUS, MINUS))
w()
rows = []
for r in ORDER:
    cl = [cell(T[PR[p]["XP"]][s][r]) for (s, p) in COLS]
    ns = " ; ".join(nr(T[PR[p]["XP"]][s][r]["n"]) for (s, p) in COLS)
    rows.append([r, BETTER[r]] + cl + [ns])
table(["row", "better", "SOUP %s P10, s0" % MINUS, "SOUP' %s P10, s0" % MINUS, "SOUP %s P10, s1" % MINUS, "SOUP' %s P10, s1" % MINUS, "n, s0 SOUP / s0 SOUP' / s1 SOUP / s1 SOUP'"], rows)
w("Y %s P10 for the controls (DAG-OACC, A):" % MINUS)
w()
rows = []
for r in ORDER:
    cl = [cell(T["CTRL_P10"]["s0"][r]), cell(T["A_P10"]["s0"][r]), cell(T["CTRL_P10"]["s1"][r]), cell(T["A_P10"]["s1"][r])]
    ns = " ; ".join(nr(x) for x in (T["CTRL_P10"]["s0"][r]["n"], T["A_P10"]["s0"][r]["n"], T["CTRL_P10"]["s1"][r]["n"], T["A_P10"]["s1"][r]["n"]))
    rows.append([r, BETTER[r]] + cl + [ns])
table(["row", "better", "DAG-OACC %s P10, s0" % MINUS, "A %s P10, s0" % MINUS, "DAG-OACC %s P10, s1" % MINUS, "A %s P10, s1" % MINUS, "n, s0 DAG-OACC / s0 A / s1 DAG-OACC / s1 A"], rows)

w("### 4.7 Coverage per family, and what is not computed")
w()
w("`strategic_family` in the JSON reads *\"NOT PAUSED for v8F (PI ruling 2026-10-17)\"*. Per family, whether a row exists in this read, with the reason and the n where it does not:")
w()
table(["family", "rows in this read (T1-kin, X %s Y and X %s P10)" % (MINUS, MINUS), "n (windows / episodes)", "not in this read, with the reason"], [
    ["LONGITUDINAL", "speed MAE, along-track MAE, accel MAE, DK-VIOL, DK-HW, DK-TG", "2399/141 (speed, along-track, accel); 595/70 (DK-VIOL); 709 to 714 / 72 or 73 (DK-HW); 651 to 654 / 72 (DK-TG)", "the anti-echo block: NOT BUILT (the DAG-ORC rig carries none)"],
    ["LATERAL", "heading MAE, yaw-rate MAE, curvature MAE, cross-track MAE, L1 lateral-accel MAE, L2 curvature MAE (GT>=5.2); M1 crawl share beside", "2373/141 (heading); 2365 to 2367 / 141 (yaw-rate, curvature, L1); 2399/141 (cross-track); 2103 to 2106 / 135 (L2, M1)", "none registered for T1-kin"],
    ["TACTICAL", "TAC-LAT, TAC-LON (plan-derived agreements); R-TLAT and R-TLON as reader booleans", "2399/141", "goal-head F1 and head accuracy: NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); the T-head macro-F1 values behind R-TLAT / R-TLON are not carried"],
    ["STRATEGIC", "S-POS@30, S-POS@50, S-POS@80 against CTRV within a run (not paused)", "2321/141 (@30); 2313/141 (@50); 2290/141 (@80); X %s P10 unpaired" % MINUS, "strategic decision and goal / route-setting quality: no row in the JSON; UNVERIFIED whether another tool computes it for these arms"],
    ["all four, at T0", "none: the T0 four families are not in the read", "not applicable", "PREREG section 8 puts them on the dev-box standard eval (`--families`); the reader carries only the R-T0 points (section 2.3) and the slow-draw cells (section 5.3). Nothing is claimed for T0 LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC"],
])

# ---------------------------------------------------------------- 5 beside
w("## 5. Beside, deciding nothing")
w()
w("### 5.1 The members: the slow draw, and the soups against the other contrasts (ADE6 and the crawl rows, T1-kin)")
w()
w("`REP_CTRL` is REP %s DAG-OACC (the registered member spread RF). `SOUP_REP` is SOUP %s REP, `SOUPR_Arep` is SOUP' %s A_rep, `SOUPR_CTRL` is SOUP' %s DAG-OACC, `SOUP4_CTRL` is SOUP4 %s DAG-OACC." % (MINUS, MINUS, MINUS, MINUS, MINUS))
w()
rows = []
BES = [("REP_CTRL", "REP %s DAG-OACC" % MINUS), ("SOUP_REP", "SOUP %s REP" % MINUS), ("SOUPR_Arep", "SOUP' %s A_rep" % MINUS), ("SOUPR_CTRL", "SOUP' %s DAG-OACC" % MINUS), ("SOUP4_CTRL", "SOUP4 %s DAG-OACC" % MINUS)]
for ct, lab in BES:
    for r in ["ADE6"] + CRAWL:
        rows.append([lab, r, cell(T[ct]["s0"][r]), cell(T[ct]["s1"][r])])
table(["contrast", "row", "s0", "s1"], rows)
w("### 5.2 SOUP4 on every row (beside; its members overlap both pairs)")
w()
rows = []
for r in ORDER:
    rows.append([r, BETTER[r], cell(T["SOUP4_CTRL"]["s0"][r]), cell(T["SOUP4_CTRL"]["s1"][r]), cell(T["SOUP4_P10"]["s0"][r]), cell(T["SOUP4_P10"]["s1"][r])])
table(["row", "better", "SOUP4 %s DAG-OACC, s0" % MINUS, "SOUP4 %s DAG-OACC, s1" % MINUS, "SOUP4 %s P10, s0" % MINUS, "SOUP4 %s P10, s1" % MINUS], rows)
w("### 5.3 The slow-draw cells (T0, dev-box standard eval; speed cells, m/s; per-window paired difference, episode-cluster bootstrap)")
w()
w("`beside.slow_draw`: the soup's speed cells against its pair's control, the method of `rep_s0_speedcells.py`. The reader's `threshold_m_s` is 0.5 and its `label` is null in all four entries. The labelling rule lives in the reader and is not re-read here (UNVERIFIED); no label is claimed. Point [95 % CI], bold when the interval excludes 0.")
w()
SDK = ["main:SOUP-CTRL:s0", "main:SOUP-CTRL:s1", "rep:SOUPR-A:s0", "rep:SOUPR-A:s1"]
SLABEL = {"main:SOUP-CTRL:s0": "SOUP_0 %s DAG-OACC_0" % MINUS, "main:SOUP-CTRL:s1": "SOUP_1 %s DAG-OACC_1" % MINUS, "rep:SOUPR-A:s0": "SOUP'_0 %s A_0" % MINUS, "rep:SOUPR-A:s1": "SOUP'_1 %s A_1" % MINUS}
SCELLS = ["lon_v_target_ms", "v_a_ms", "v_end_ms", "v_hi_ms", "v_lo_ms"]


def scell(c):
    s = "%s [%s, %s]" % (n4(c["mean"]), n4(c["lo"]), n4(c["hi"]))
    return "**" + s + "**" if (c["lo"] > 0 or c["hi"] < 0) else s


rows = []
for k in SDK:
    cells = BS["slow_draw"][k]["cells"]
    rows.append([SLABEL[k]] + [scell(cells[x]) for x in SCELLS] + ["null" if BS["slow_draw"][k]["label"] is None else str(BS["slow_draw"][k]["label"]), "%.1f" % BS["slow_draw"][k]["threshold_m_s"]])
table(["contrast (T0)"] + SCELLS + ["reader label", "threshold (m/s)"], rows)
w("At s0 the soup's speed cells are negative with intervals below 0 in both pairs (the soup is slower than its control); at s1 every cell is within 0.15 m/s of 0. This is a T0 reading on one control per pair and decides nothing.")
w()
w("### 5.4 Reader self-checks the JSON carries (controls, by family)")
w()
from collections import Counter
cn = Counter(k.split(":")[0] for k in D["controls_PASS"])
rows = [[k, str(v), "PASS" if all(D["controls_PASS"][x] for x in D["controls_PASS"] if x.split(":")[0] == k) else "FAIL"] for k, v in sorted(cn.items())]
w("%d controls in %d families, all true (`controls_PASS`). Among them K-RC (the reader reproduces the banked LAUNCH-INTENT tables A %s CTRL, Arep %s REP, CTRL %s P10, REP %s P10, REP %s CTRL, A %s P10, Arep %s P10 and RF: 8 controls), K-BASIN and K-MID and K-SOUP-REC (6 each, one per soup), K-XINIT (2, the cross-init soups fail as required), K-ZERO-W (1), K-ZERO (10, (X, X) reads exactly +0 [0, 0]), K-T0DEV-BESIDE (4) and the C3 / C4 window controls (24)." % ((len(D["controls_PASS"]), len(cn)) + (MINUS,) * 7))
w()
table(["control family", "entries", "all true"], rows)

# ---------------------------------------------------------------- 6 mechanism
w("## 6. Reading (decides nothing)")
w()
w("**Label: MEASURED facts from the tables above, arithmetic on them, and one mechanism statement (HYPOTHESIS). The outcome is HARMS, as registered. Nothing in this section changes a flag, a bar or the word.**")
w()
gap0 = T["REP_CTRL"]["s0"]["ADE6"]["delta"]
sm0 = T["SOUP_REP"]["s0"]["ADE6"]["delta"]
sc0 = T["SOUP_CTRL"]["s0"]["ADE6"]["delta"]
gapr0 = T["SOUPR_A"]["s0"]["ADE6"]["delta"] - T["SOUPR_Arep"]["s0"]["ADE6"]["delta"]
scr0 = T["SOUPR_A"]["s0"]["ADE6"]["delta"]
sar0 = T["SOUPR_Arep"]["s0"]["ADE6"]["delta"]
w("**Fact 1. The weight average beats the mean of its members' errors, and no cell is a barrier.** SG on ADE6 is negative at all six soup-seed cells and separated in five (section 3). SOUP at s1 reads %s %s, not separated." % (n4(SG["SOUP:s1"]["delta"]), "[%s, %s]" % (n4(SG["SOUP:s1"]["lo"]), n4(SG["SOUP:s1"]["hi"]))))
w()
w("**Fact 2. At s0 the soup recovers most of the slow draw and still lands worse than the good one.**")
w()
table(["seed 0, ADE6 (m, T1-kin)", "main pair (DAG-OACC_0, REP_0)", "replicate pair (A_0, A_rep,0)"], [
    ["slow member %s good member (the member spread)" % MINUS, "REP_0 %s DAG-OACC_0 %s" % (MINUS, cs("REP_CTRL", "s0", "ADE6")), "A_rep,0 %s A_0 %s (arithmetic: the A %s soup delta minus the A_rep %s soup delta)" % (MINUS, n4(gapr0), MINUS, MINUS)],
    ["soup %s slow member" % MINUS, "SOUP_0 %s REP_0 %s" % (MINUS, cs("SOUP_REP", "s0", "ADE6")), "SOUP'_0 %s A_rep,0 %s" % (MINUS, cs("SOUPR_Arep", "s0", "ADE6"))],
    ["soup %s good member (the control)" % MINUS, "SOUP_0 %s DAG-OACC_0 %s" % (MINUS, cs("SOUP_CTRL", "s0", "ADE6")), "SOUP'_0 %s A_0 %s" % (MINUS, cs("SOUPR_A", "s0", "ADE6"))],
    ["share of the member spread recovered (%sdelta vs slow member / spread)" % MINUS, "%.1f %%" % (100.0 * -sm0 / gap0), "%.1f %%" % (100.0 * -sar0 / gapr0)],
    ["position of the soup between the good and the slow member (delta vs good member / spread)", "%.1f %%" % (100.0 * sc0 / gap0), "%.1f %%" % (100.0 * scr0 / gapr0)],
])
w("MEASURED (arithmetic on the banked point deltas; the member spread of the replicate pair is the difference of two banked deltas). A linear interpolation of the members' errors puts a two-member soup at 50 %% of the way; the PREREG's first-order prediction for SOUP_0 %s DAG-OACC_0 was about +0.20 m (ESTIMATED, section 0 item 3: half of +0.4025). The measured point is %s, so the average does better than linear interpolation of the errors, and it is still separated-worse than the good draw: at s0 the lever replaces the good draw with a worse one. A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base (PREREG section 12)." % (MINUS, n4(sc0)))
w()
s1gap = T["REP_CTRL"]["s1"]["ADE6"]
w("**Fact 2b. The same holds on the four crawl rows at s0: the soup recovers part of the slow member's crawl and stays worse than the good member on every one.**")
w()
rows = []
POS, REC = [], []
for p_, lab in (("main", "main (DAG-OACC_0, REP_0)"), ("rep", "replicate (A_0, A_rep,0)")):
    for r in CRAWL:
        if p_ == "main":
            slow_c, good_c = T["SOUP_REP"]["s0"][r], T["SOUP_CTRL"]["s0"][r]
        else:
            slow_c, good_c = T["SOUPR_Arep"]["s0"][r], T["SOUPR_A"]["s0"][r]
        spread = good_c["delta"] - slow_c["delta"]
        POS.append(100.0 * good_c["delta"] / spread)
        REC.append(100.0 * -slow_c["delta"] / spread)
        rows.append([lab, r, n4(spread), cell(slow_c), "%.1f %%" % (100.0 * -slow_c["delta"] / spread), cell(good_c), "%.1f %%" % (100.0 * good_c["delta"] / spread)])
table(["pair (seed 0, T1-kin)", "crawl row", "member spread (slow %s good)" % MINUS, "soup %s slow member" % MINUS, "share of the spread recovered", "soup %s good member (the control)" % MINUS, "position between good and slow member"], rows)
w("MEASURED (arithmetic on the banked point deltas). The spread is the difference of the two banked deltas, (soup %s good member) %s (soup %s slow member), so the two shares add to 100 %%. The banked `REP_CTRL` delta (the registered RF) is the same quantity read on its own window set: the two agree on ADE6 and M1 to the 4th decimal and differ on the masked rows by up to 3 windows' worth (the n column of the tables), which moves a share by about one percentage point (ESTIMATED). The soup lies between its members on every crawl row, at %.1f to %.1f %% of the way from the good to the slow member against 50 %% for a linear interpolation of the errors, and is separated-worse than the good member on all four rows in both pairs." % (MINUS, MINUS, MINUS, min(POS), max(POS)))
w()
w("**Fact 3. At s1 the members agree, and the soup sits with them.** REP_1 %s DAG-OACC_1 is %s, not separated; SOUP_1 %s DAG-OACC_1 is %s, not separated. In the replicate pair A_rep,1 is %s m better than A_1 by the arithmetic of the two soup deltas, and SOUP'_1 %s A_1 is %s, separated-better, while SOUP'_1 %s A_rep,1 is %s, not separated: the soup sits near the better member, a little beyond it (SG %s). On the four crawl rows at s1, REP_1 %s DAG-OACC_1 is not separated on any (section 5.1), so there is no draw-specific crawl for the average to remove." % (MINUS, cs("REP_CTRL", "s1", "ADE6"), MINUS, cs("SOUP_CTRL", "s1", "ADE6"), u4(abs(T["SOUPR_A"]["s1"]["ADE6"]["delta"] - T["SOUPR_Arep"]["s1"]["ADE6"]["delta"])), MINUS, cs("SOUPR_A", "s1", "ADE6"), MINUS, cs("SOUPR_Arep", "s1", "ADE6"), n4(SG["SOUPR:s1"]["delta"]), MINUS))
w()
w("**Fact 4. What both draws share, the soup keeps.** No crawl row clears its bar in either pair at either seed (R-MOVE false, section 2.4), and every crawl row of every soup stays separated-worse against P10 (R-CRAWL false, section 2.5). Both controls, DAG-OACC and A, are separated-worse than P10 on the four crawl rows at both seeds (section 2.5, Y %s P10 column): the good draw sits above the floor too. Read A is the control's state: D2 fails on the same four rows." % MINUS)
w()
w("**Reading (HYPOTHESIS).** The crawl rows carry two parts. One is draw-specific: at s0 the slow draws REP_0 and A_rep,0 are separated-worse than DAG-OACC_0 and A_0 on all four crawl rows, and averaging recovers %.1f to %.1f %% of that excess (Fact 2b). The PREREG's hypothesis (section 1), *\"part of the crawl-row error of one draw is draw-specific, and an average of two draws keeps the shared part and halves the draw-specific part,\"* is supported in direction at s0. The other part is shared by both draws: both controls are separated-worse than P10 on those rows at both seeds, and on every crawl row at s0 the soup landed between its members (Fact 2b), so it landed above the floor as well. The registered bar is read against the control, which at s0 is the good draw; on the four crawl rows the average with a worse draw did not beat it, and at s1, where the draws agree, there is nothing to average away. So under this design the soup could not have cleared R-MOVE at either seed, which is what the read shows." % (min(REC), max(REC)))
w()
w("## 7. Next (Rule Zero)")
w()
w("### 7.1 Committed next, as registered: SOUP-K")
w()
w("PREREG section 11, FLAT row: **SOUP-K**, K = 4 pure-recipe draws per seed (DAG-OACC_s's argv at trainer seeds 20 + s and 30 + s added to DAG-OACC_s and REP_s; 4 new box-2 runs, about 2.8 h ESTIMATED from the LAUNCH-INTENT runs; its own PI go), read under the same table with SOUP-K_s in place of SOUP_s. **Status: not launched; blocked on the PI go (box-2 time).** The route was decided by the JSON (`route` = FLAT), not by this document.")
w()
w("### 7.2 Rule Zero: what the crawl diagnostics say, and the lever they point to")
w()
w("A refutation is a waypoint. The soup's bar was not cleared, so the question is what the cheapest next lever is and whether the evidence already ranks it. Evidence class as each row states it (MEASURED, PARTIAL or NOT EVALUABLE), quoted from `Project Steering/GOALS_AND_CLAIMS.md` at the tip `0c7b598`; the reading that joins them is a HYPOTHESIS and decides nothing.")
w()
table(["register row", "what it measured (quoted)", "reading (HYPOTHESIS): what it implies for averaging draws"], [
    ["`H-CRAWL-01`", "On the 327 GT-launch windows (49 episodes) an outcome-blind predictor built on TRAIN reads a gap of the same order as the model's: floor median %s [%s, %s], mean %s [%s, %s] m/s; DAG-OACC s0 gap %s; D = +0.159 [-0.136, +0.468] (median), -0.075 [-0.377, +0.232] (mean), at the floor on the point; REP s0 gap %s, D -0.397 [-0.777, +0.010] (median), -0.631 [-1.008, -0.225] (mean). On GT-stay windows (32 windows / 10 episodes) the models move +1.272 [+1.005, +1.624] and +1.215 [+1.037, +1.366] m/s." % ("-1.034", "-1.298", "-0.774", "-0.800", "-1.071", "-0.529", "-0.875 [-1.143, -0.601]", "-1.431"), "the crawl of DAG-OACC s0 sits at the outcome-blind floor on the point only (outcome (b); CI half-width 0.30): the smaller crawl is what a predictor blind to the launch trigger shows. A different training draw moves the model around that floor (REP s0 crawls deeper than it: D -0.631, mean table, separated); a weight average of draws does not move the floor."],
    ["`H-CRAWL-02`", "S_v >= 0.6 on the stopped-start windows (61 windows / 15 episodes): DAG-OACC +0.736 [+0.593, +0.829], REP +0.628 [+0.501, +0.726]. The frozen P10 prior alone gives +0.686 [+0.557, +0.768], identical in both checkpoints; the learned residual adds +0.050 [+0.015, +0.090] (DAG-OACC) and -0.059 [-0.121, +0.012] (REP). The tracking prediction for the run difference is refuted by sign.", "S_v is the frozen prior's. The prior is the same in every member, so an average of members leaves it where it is."],
    ["`H-CRAWL-03`", "NOT EVALUABLE: no intermediate checkpoint exists for either run (the trainer keeps one `ckpt_last.pt` per run). It unblocks with a trainer option that keeps the last k checkpoints, at about zero box-2 cost in the next fine-tune.", "whether the 0.556 m/s run-to-run launch difference is between-run or within-run noise stays open; SOUP-K gives more draws, not intermediate checkpoints."],
    ["`H-ST-REGRET-PRE`", "Crawl states are 5.752 % [5.439, 6.062] (s0) and 2.608 % [2.443, 2.782] (s1) of usable DAgger-bank records; the 50 : 50 deficit-weighted multiplier is 2.68 [2.60, 2.77] (s0) and 3.08 [2.99, 3.18] (s1), below the 5x branch. MIDDLE BAND: weighting buys about 3x.", "a data-weighting lever buys about 3x on a rare state; it is not a draw-variance lever."],
    ["`H-ST-RECOVERYLABEL-1`", "Kept fraction at tick 2: 53.85 % to 59.51 % (s0), 54.69 % to 61.95 % (s1); the 75 to 81 % prediction is NOT met (V2: 58.20 % / 60.07 %).", "the recovery-label route for the off-route states is weaker than predicted."],
    ["`H-DD-FITGAP-M2B`", "On M2b s0 / s1 (2 of the card's 6 checkpoints) the launch-speed deficit on the model's own TRAIN GT-launch windows is as large as on eval-141: r = 1.229 [0.976, 1.680] (s0), 1.612 [1.023, 4.416] (s1), both FIT-LIMITED. PARTIAL: DAG-OACC s0 / s1 and REP s0 / s1 are not measured.", "the model crawls on its own training windows as much as on eval, so the crawl is not a data-coverage gap; by the card's reading no data-amount or data-selection lever can clear a crawl bar. Scope: the two M2b checkpoints, not the soup's members."],
    ["`H-CRAWL-06-PROBE`", "On TRAIN, P-MOVE: AUC E 0.731, V+E 0.784, Z+E 0.880; dAUC(V+E - E) +0.053 [0.041, 0.066] by the frozen rule, but with a nonlinear ego baseline +0.016 [0.007, 0.025]. P-STOP as frozen is NOT EVALUABLE (control C2 missed its committed bound by 7e-5). Image gain over E on P-STOP: +0.099 on red-light clips, -0.058 on green-light clips, -0.019 where the lead vehicle departs.", "the current-frame image features add little go/stay information over ego, and none on green-light or lead-departure clips."],
    ["`H-CRAWL-06-PROBE2`", "Nav is built from turns of the whole recording on the ego's own future path; the larger ego-future channel is the speed-ceiling input (declared oracle, the PI's 2026-09-10 ruling): L+E - E +0.072 [0.057, 0.087] (P-STOP) / +0.053 [0.044, 0.062] (P-MOVE). With nav and the speed ceiling both scrambled, Z_nl+E - V+E +0.006 [-0.016, 0.027] on P-STOP. NO-MOTION: V4+E - V+E -0.005 [-0.011, +0.001] / -0.000 [-0.003, +0.002].", "the trunk's go/stay edge is the declared speed-ceiling and nav oracles; the 4 frames add nothing. The trigger is weakly present in what v8F sees, and absent for green lights and departing leads."],
])
w("**Reading (HYPOTHESIS; decides nothing).** The soup separates two kinds of error. The draw-specific part (REP_0's excess over DAG-OACC_0 on ADE6 and on the crawl rows) is partly recovered by averaging (section 6, Facts 2 and 2b). The part that remains once the draws agree is shared by every member, and the diagnostics place it upstream of anything a weight average acts on: the crawl sits at the outcome-blind floor, S_v is carried by the frozen prior that every member shares, it is fit-limited on the M2b checkpoints that were measured, and the information that would separate go from stay (the trigger) is weakly present in the current-frame features while the trunk's go/stay edge is the declared oracle channels. So averaging, and by the same arithmetic more draws of the same recipe (SOUP-K), can shrink the draw-specific part further and cannot supply a trigger the features barely contain. **The lever the evidence points to is perception.**")
w()
w("**Proposed next experiment, PROPOSED and not registered: a finer-token DINOv3 probe** of the go/stay trigger. The question it would ask is whether DINOv3 tokens finer than the pooled `pool4x10` features of `H-CRAWL-06-PROBE` carry the go/stay trigger beyond ego. Which clips, which probe family and which outcomes is not decided here. It has no card, no committed outcomes and no cost estimate; none is invented. It needs its own PREREG with both outcomes committed before any data (operating standard, rule 5).")
w()
w("### 7.3 What is blocked, and on what (every remaining lever)")
w()
table(["lever", "state", "what would unblock it"], [
    ["SOUP-K (committed next)", "not launched", "the PI go for 4 box-2 runs (about 2.8 h, ESTIMATED in the PREREG)"],
    ["ENS2 (output ensemble; the BARRIER branch)", "not triggered: no cell is a barrier", "a BARRIER cell in a later soup read; it would need its own PREREG with its own K-ZERO"],
    ["LOOP-STATE (zero training; PREREG_V8F_LAUNCH_INTENT section 11) and an architecture change giving O a launch-intent input", "named in PREREG_V8F_SOUP section 11 as the blockers if SOUP-K and ENS2 are exhausted; their status is not re-read here (UNVERIFIED)", "the LOOP-STATE read, or a PI decision on the architecture change"],
    ["`H-DD-FITGAP` on the soup's members", "PARTIAL: 2 of 6 checkpoints measured (register row)", "3 of DAG-OACC s0 / s1 and REP s0 / s1 (about 0.17 h each on a 10 % clip subset, ESTIMATED in the register)"],
    ["`H-CRAWL-03` intermediate checkpoints", "NOT EVALUABLE", "a trainer option that keeps the last k checkpoints, with its G-DVB entry"],
    ["finer-token DINOv3 probe (perception)", "PROPOSED, not registered", "a PREREG with committed outcomes; the cost is not estimated here"],
])

# ---------------------------------------------------------------- 8 cannot claim
w("## 8. What this read cannot claim")
w()
w("* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged. T0 is teacher-forced and is not driving performance.")
w("* **A separated interval is necessary, not sufficient** (`H-ESTIM-SEED-1`). The bootstrap resamples EPISODES with the models held fixed: it answers *\"would another draw of EPISODES say this?\"* and is silent on *\"would another training run say this?\"*. A soup is two training draws inside one number. Here the two pairs of a seed agree on R-ADE (it fails in both at s0 and holds in both at s1) and the two seeds disagree; the seed-level difference is the slow draw (REP_0, A_rep,0) at s0. With two seeds the read cannot say how often a seed like s0 occurs.")
w("* **A soup effect is the effect of averaging these two draws.** N = 2 per decisive soup; nothing here says a soup of K > 2 behaves the same. SOUP4 is beside and its members overlap both pairs.")
w("* **The control is asymmetric by design** (PREREG section 12). Each soup is read against its trainer-seed-s member. A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base.")
w("* **The replicate soup carries the launch-intent lever** on both members; its result is the soup lever on the A recipe and attributes nothing to the launch weight.")
w("* **R-T0 is a dev-box CPU T0 reading** on both arms of the pair; it is not box 2's standard eval.")
w("* **SG is a mean-of-errors statistic.** SG < 0 says the average beats the mean of its members' errors. It does not say the soup beats the better member (Fact 2).")
w("* **The words 'perception is the lever' are a reading of diagnostics** (section 7.2), each measured on other arms and checkpoints (`H-DD-FITGAP-M2B`: the two M2b checkpoints; `H-CRAWL-06-PROBE2`: one checkpoint, DAG-OACC s0). They are not a result of this read.")
w("* ⛔ No bar, row, member set, donor rule, K-BASIN factor, routing variable or outcome changed after any soup row was seen.")
w()

# ---------------------------------------------------------------- 9 disclosures
w("## 9. Disclosures")
w()
w("1. **The reader's provenance label.** The JSON's `provenance.prereg` reads *\"PREREG_V8F_SOUP.md (DRAFT, 2026-10-28)\"*; the sha256 it checked is the PROMOTED file's (`830d9c91…`). The promotion (2026-10-29) changed the status line and section 0 item 5 only, and the reader's PREREG-TEXT check passed on it (INHERITED from the brief). The JSON carries no control named PREREG-TEXT; its static controls are one boolean, `controls_PASS.STATIC`, true, and whether PREREG-TEXT sits inside it is UNVERIFIED here.")
w("2. **The launcher was edited and restored.** `soup_chain.cmd` had its two PREREG lines set (lines 10 and 11). A Git-bash `sed` converted it to LF and the Master Mind restored it from the as-shipped CRLF copy before launch (INHERITED from the brief). MEASURED here from the two banked copies (both under `raw/2026-10-29-soup-read/as_launched/`, both CRLF): the launched file differs from the shipped one in exactly those 2 lines.")
w("3. **Disk.** C: had 307 to 886 MB free at hand-back (INHERITED from the brief); 46 old scratchpad folders (about 32 GB) were MOVED to F: before launch (INHERITED from the brief), and the chain's 2 GB gate was not lowered (the gate line below reads 2048 MB). MEASURED in the chain log: the gate line reads *\"C: gate: 34097 MB free >= 2048 MB\"* at 02:19:45.")
w("4. **A deletion.** The SOUP finishing agent deleted its own test directories in the session scratchpad (INHERITED from the brief). The rule is to move, not delete.")
w("5. **R-T0 used the dev-box T0 of both arms** (PREREG section 0 item 4, K-T0DEV): the soup and the pair's control were both evaluated on the dev-box CPU. Box 2's standard T0 of the members is not the comparison.")
w("6. **A register label.** The register row `H-V8F-SOUP-1` calls the %s cell \"TAC-LAT macro-F1\". In the JSON that cell is the TAC-LAT row, a plan-derived agreement (`four_families` note); the guard R-TLAT is a boolean whose macro-F1 value is not carried. Section 2.3 shows the two agree in direction on TAC-LAT in all four pairs and differ on TAC-LON in the s0 replicate pair." % n4(T["SOUP_CTRL"]["s0"]["TAC-LAT"]["delta"]))
w("7. **Single copies.** The per-window dumps and tick arrays are under `F:/TanitAD-scratch/soup_t1_out/` (`NOT_LANDED.txt` lists them with sha256). The repo holds the rollout logs, the argv, the done records, the chain log and the JSON.")
w("8. **What `check_result.py` is and is not.** It re-finds every table cell, every quoted string and every prose number in the banked JSON, logs, PREREG and register rows, recomputes the decision from the cells, and checks the document for the banned words and LF. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap. A defect inside the estimator or the reader's heavy layer would not show there.")
w("9. **Instrument status (INHERITED).** The reader's tests and mutation pass are as the register row records (127 tests; 286 of 288 mutants killed, 2 named equivalent); the PREREG's package status records that the heavy layer had not run on real dumps before launch precondition 4, and K-RC (8 controls, all true here; PREREG section 9 defines it as 0 differences from the banked LAUNCH-INTENT tables) is that check.")
w()

# ---------------------------------------------------------------- 10 run record
w("## 10. Run record")
w()
w("Evidence class MEASURED (ours): the dev-box chain log as banked (`raw/2026-10-29-soup-read/logs/chain.out`). One torch process at a time, dev-box CPU.")
w()
w("The chain started 02:19:45 and its last step ended 02:56:08. Twenty-eight steps ran, each at attempt 1: for each of the six soups a causal rollout, a logged rollout, a fresh-T0 eval and a standard-T0 eval, then the standard T0 of DAG-OACC_0, DAG-OACC_1, A_0 and A_1. The reader started 02:56:28 and the chain closed at 03:07:03 with *\"ZZSOUP-DONE 28/28 steps, reader rc 0\"*. The 28 steps sum to 2074 s of step wall (derived by `check_result.py` from the log).")
w()
rows = []
STEPS = [
    ("SOUP_0", "SOUP_s0_m10_causal_heads", "SOUP_s0_m10_logged_heads", "M2b-soup_s0-fresh", "M2b-soup_s0"),
    ("SOUP_1", "SOUP_s1_m10_causal_heads", "SOUP_s1_m10_logged_heads", "M2b-soup_s1-fresh", "M2b-soup_s1"),
    ("SOUP'_0", "SOUPR_s0_m10_causal_heads", "SOUPR_s0_m10_logged_heads", "M2b-soupr_s0-fresh", "M2b-soupr_s0"),
    ("SOUP'_1", "SOUPR_s1_m10_causal_heads", "SOUPR_s1_m10_logged_heads", "M2b-soupr_s1-fresh", "M2b-soupr_s1"),
    ("SOUP4_0", "SOUP4_s0_m10_causal_heads", "SOUP4_s0_m10_logged_heads", "M2b-soup4_s0-fresh", "M2b-soup4_s0"),
    ("SOUP4_1", "SOUP4_s1_m10_causal_heads", "SOUP4_s1_m10_logged_heads", "M2b-soup4_s1-fresh", "M2b-soup4_s1"),
]
import re
import subprocess
CH = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", "0c7b598edd16a5cbb422ec050a67760d4ba5b11e:TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-29-soup-read/logs/chain.out"], capture_output=True).stdout.decode("utf-8", "replace")
WALL = {m.group(2): int(m.group(3)) for m in re.finditer(r"(\d\d:\d\d:\d\d) end (\S+) rc=0 wall=(\d+)s", CH)}
assert len(WALL) == 28, len(WALL)
for lab, a, b, c_, d_ in STEPS:
    rows.append([lab, str(WALL[a]), str(WALL[b]), str(WALL[c_]), str(WALL[d_])])
table(["soup", "causal rollout (s)", "logged rollout (s)", "fresh-T0 eval (s)", "standard-T0 eval (s)"], rows)
w("Controls' standard T0 on the dev box (s): DAG-OACC_0 %d, DAG-OACC_1 %d, A_0 %d, A_1 %d." % (WALL["M2b-dagoacc_0-dev"], WALL["M2b-dagoacc_1-dev"], WALL["M2b-a_0-dev"], WALL["M2b-a_1-dev"]))
w()

# ---------------------------------------------------------------- 11 provenance
w("## 11. Provenance (artifact path, sha256)")
w()
w("Evidence class MEASURED (ours). Every sha256 below is of the bytes at the tip `0c7b598edd16a5cbb422ec050a67760d4ba5b11e`, re-measured for this document. Paths are under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/` unless they start with `Project Steering/`.")
w()
PROV = [
    ("read of record (reader output)", "2026-10-29-soup-read/out/read/SOUP_READ.json"),
    ("read of record, console", "2026-10-29-soup-read/logs/soup_read.out"),
    ("dev-box chain log", "2026-10-29-soup-read/logs/chain.out"),
    ("launcher as launched (CRLF; two PREREG lines set)", "2026-10-29-soup-read/as_launched/soup_chain.cmd"),
    ("launcher as shipped (CRLF)", "2026-10-29-soup-read/as_launched/soup_chain.cmd.as_shipped"),
    ("dry run on the promoted PREREG", "2026-10-29-soup-read/as_launched/dry_run_promoted.out"),
    ("files kept on the dev box (listing)", "2026-10-29-soup-read/NOT_LANDED.txt"),
    ("reader of record `soup_read.py`", "2026-10-29-soup/read/soup_read.py"),
    ("LAUNCH-INTENT reader, imported unchanged", "2026-10-29-soup/read/ref/launch_intent_read.py"),
    ("chain pins", "2026-10-29-soup/PINS.txt"),
    ("rendered plan (28 steps)", "2026-10-29-soup/argv/PLAN.txt"),
    ("soup spec", "2026-10-29-soup/soups.json"),
    ("weight-space readings (K-BASIN, K-XINIT)", "2026-10-29-soup/evidence/basin.json"),
    ("PREREG as promoted (and at the tip)", "Project Steering/PREREG_V8F_SOUP.md"),
]
SHAS = {}
import subprocess
RES = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"
TIP = "0c7b598edd16a5cbb422ec050a67760d4ba5b11e"
rows = []
for role, path in PROV:
    full = path if path.startswith("Project Steering/") else RES + path
    b = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", TIP + ":" + full], capture_output=True).stdout
    h = hashlib.sha256(b).hexdigest()
    SHAS[path] = h
    rows.append([role, "`%s`" % path, "`%s`" % h])
table(["role", "repo path", "sha256"], rows)
w("The dev-box manifest `runs_soup.json` (not staged; F: only) is pinned by the JSON as `provenance.manifest_sha256` `35833493…`. The reader's pins in the JSON: `reader_sha256` `de39aefe…`, `li_reader_sha256` `fb3d2d7d…`, `prereg_sha256` `830d9c91…`, `wall_s` 634.9. The soup checkpoints (E: only, 29.0 MB each, `opt` None): SOUP_s0 `238aab03…` (the chain log prints `238aab039dac` for M2b_s0_soup), SOUP_s1 `81924b1a…`, SOUP'_s0 `3db5166a…`, SOUP'_s1 `e0d1b6ec…`, SOUP4_s0 `006b2a0a…`, SOUP4_s1 `8b07ab1a…` (PREREG, weight-space readings).")
w()

txt = "\n".join(out) + "\n"
with open(ARGS.out, "w", encoding="utf-8", newline="\n") as f:
    f.write(txt)
print("wrote", ARGS.out, len(txt), "bytes;", len(out), "lines; json sha256", JSON_SHA)
