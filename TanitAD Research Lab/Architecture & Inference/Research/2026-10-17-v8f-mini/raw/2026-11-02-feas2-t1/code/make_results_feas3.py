#!/usr/bin/env python3
"""make_results_feas3.py -- RESULTS_FEAS3.md of H-FEAS3-T1-1, generated from raw/FEAS3_T1.json (+ raw/FEAS2_T1.json, raw/PRECHECK_FEAS3.json, the reader-fix logs); no hand-typed number.

    python -I code/make_results_feas3.py
Adapted from make_results_feas2.py (same rows, same outcome words); the lever axis is kbox2 / lonfloor / kbox2lf; adds the two pre-registered disclosures of the card.
Fails loud (SystemExit) if a fact the fixed prose depends on does not hold in the JSON.
"""
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE).replace("\\", "/")
SMT = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/code"
sys.path.insert(0, SMT)
import smooth_t1_read as SR  # noqa: E402

GD = SR.load_gd()
J = json.load(open(f"{PK}/raw/FEAS3_T1.json", encoding="utf-8"))
F2 = json.load(open(f"{PK}/raw/FEAS2_T1.json", encoding="utf-8"))
PC = json.load(open(f"{PK}/raw/PRECHECK_FEAS3.json", encoding="utf-8"))
MODES = ("kbox2", "lonfloor", "kbox2lf")
ARMS = (("CTRL", "DAG-OACC"), ("SOUP", "SOUP"))
KEYS = {"CTRL": ("CTRL_s0", "CTRL_s1"), "SOUP": ("SOUP_s0", "SOUP_s1")}
ALLK = ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1")
CRAWL = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
BESIDE = ("ADE6", "heading MAE", "S-POS@30", "S-POS@50", "S-POS@80", "along-track MAE", "cross-track MAE", "speed MAE", "accel MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "yaw-rate MAE", "M1 crawl share")
PRED = {  # the card's ESTIMATED table (PREREG_FEAS3_FROZEN.md)
    "kbox2": {"DOMINANT": 2, "HARMS": 70, "MOVED": 3, "NO EFFECT": 25},
    "lonfloor": {"DOMINANT": 1, "HARMS": 12, "MOVED": 2, "NO EFFECT": 85},
    "kbox2lf": {"DOMINANT": 2, "HARMS": 70, "MOVED": 2, "NO EFFECT": 26},
}
THR = (("thr_5.2_row_definition", "row def (5.2 m/s)"), ("margin_1", "margin 1 m/s"), ("margin_2", "margin 2 m/s"), ("margin_3", "margin 3 m/s"))
HB = ("along-track MAE", "S-POS@30", "S-POS@50", "S-POS@80")
L = []


def die(msg):
    raise SystemExit("UNVERIFIED / PROSE-FACT FAILED: " + msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mark(lab, r):
    if SR.better(GD, lab, r):
        return "SB"
    if SR.worse(GD, lab, r):
        return "SW"
    return "ns"


def mk2(r):  # shares and MAEs: lower is better
    if r["separated"] and r["delta"] < 0:
        return "SB"
    if r["separated"] and r["delta"] > 0:
        return "SW"
    return "ns"


def cell(lab, r):
    return "%+.4f [%+.4f, %+.4f] %s" % (r["delta"], r["lo"], r["hi"], mark(lab, r))


def cell2(r, d=4):
    return "%+.*f [%+.*f, %+.*f] %s" % (d, r["delta"], d, r["lo"], d, r["hi"], mk2(r))


def w(s=""):
    L.append(s)


def trig(a):
    XY, R = a["read_xy"], a["rows"]
    t = []
    if not XY["ade6_within_guard"]:
        t.append("ADE6 X - Y %+.4f > guard %+.4f" % (XY["ade6_delta"], SR.ADE_GUARD_M))
    t += ["%s X - Y %+.4f SW" % (q, R["X_minus_Y"][q]["delta"]) for q in XY["non_crawl_separated_worse"]]
    return t


def A(m, k):
    return J["arms"][f"{k}:{m}"]


def nm_of(k):
    return "DAG-OACC" if k.startswith("CTRL") else "SOUP"


# ---- fixed-prose guards (fail loud) ----
for m in MODES:
    for _, nm in ARMS:
        if J["outcome"][m][nm]["outcome"] != "HARMS":
            die(f"outcome {m}/{nm} is not HARMS; the fixed prose of sections 0, 7 and 9 assumes HARMS at all six")
if not J["outcome"]["lonfloor"]["SOUP"]["moved_and_harms"]:
    die("lonfloor/SOUP moved_and_harms is not True")
if not all(J["outcome"][m][nm]["moved_and_harms"] is (m == "lonfloor" and nm == "SOUP") for m in MODES for _, nm in ARMS):
    die("moved_and_harms is not exactly lonfloor/SOUP")
if not J["disclosures"]["PASS"] or not all(J["disclosures"]["KM1_reader_reproduces_M1_row"].values()):
    die("KM1 not all True")
for c in J["controls"].values():
    if not c["PASS"]:
        die("a control failed")

# reader-fix facts, computed live
RD = f"{HERE}/feas3_t1_read.py"
PRE = "F:/TanitAD-scratch/feas2_t1/feas3_t1_read.pre_pathfix.py"
pin_line = [ln for ln in open(f"{PK}/FROZEN_SHA_FEAS3.txt", encoding="utf-8") if ln.startswith("code/feas3_t1_read.py")][0].split()[-1]
pre_sha = sha(PRE) if os.path.exists(PRE) else None
cur_sha = sha(RD)
cell_re = re.compile(r"^(CTRL|SOUP)_s[01] ")
l1 = [ln for ln in open(f"{PK}/logs/read_feas3.out", encoding="utf-8", errors="replace") if cell_re.match(ln)]
l2 = [ln for ln in open(f"{PK}/logs/read_feas3_rerun.out", encoding="utf-8", errors="replace") if cell_re.match(ln)]
same = (l1 == l2) and len(l1) == 12
first_fail = open(f"{PK}/raw/FEAS3.fail.first_read", encoding="utf-8").read().strip()
done = open(f"{PK}/raw/FEAS3.done", encoding="utf-8").read().strip()

w("# RESULTS_FEAS3: H-FEAS3-T1-1, kbox2 / lonfloor / kbox2lf after S, DAG-OACC and SOUP at s0 / s1")
w()
w("Card `PREREG_FEAS3_FROZEN.md` sha256 `%s` (frozen 2026-11-02T17:43:04+0100). Reader `code/feas3_t1_read.py` (the import-path fix of section 8 applies). Read clock %s (the re-run read). Raw `raw/FEAS3_T1.json`, `raw/PRECHECK_FEAS3.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**." % (J["prereg_sha256"], J["run_clock"]))
w("Estimator of every interval below: **%s**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals answer *another draw of episodes* only, not *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*. X = S + lever (S = `--smooth sg72`); Y = KAPPA-T1's fresh S runs, reused (pins in `FROZEN_SHA_FEAS3.txt`); P10 = the prior floor." % J["estimator"])
w()
w("## 0. Outcome (the card's rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT)")
w()
w("| lever | arm | outcome | seed 0: D1 / D2 / D3 | seed 1: D1 / D2 / D3 | seed 0: MOVED / HARMS | seed 1: MOVED / HARMS |")
w("|---|---|---|---|---|---|---|")
for m in MODES:
    for a, nm in ARMS:
        o = J["outcome"][m][nm]
        c, h = [], []
        for k in KEYS[a]:
            aa = A(m, k)
            c.append("%s / %s / %s" % (aa["read_a"]["D1"], aa["read_a"]["D2"], aa["read_a"]["D3"]))
            h.append("%s / %s" % (aa["read_xy"]["MOVED_seed"], aa["read_xy"]["HARMS_seed"]))
        w("| %s | %s | **%s** | %s | %s | %s | %s |" % (m, nm, o["outcome"], c[0], c[1], h[0], h[1]))
w()
w("Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): " + ", ".join("%s/%s %s" % (m, nm, J["outcome"][m][nm]["moved_and_harms"]) for m in MODES for _, nm in ARMS) + ".")
w()
n_out = sum(1 for m in MODES for _, nm in ARMS if J["outcome"][m][nm]["outcome"] == "HARMS")
cells = [A(m, k) for m in MODES for k in ALLK]
w("**MEASURED headline.** %d of 6 lever x arm outcomes read **HARMS** (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at %d of 12 cells, D3 at %d of 12, and D2 fails at %d of 12 (on M1 crawl share at %d of 12)." % (n_out, sum(c["read_a"]["D1"] for c in cells), sum(c["read_a"]["D3"] for c in cells), sum(not c["read_a"]["D2"] for c in cells), sum("M1 crawl share" in c["read_a"]["separated_worse_rows"] for c in cells)))
w()
lf_s = [A("lonfloor", k) for k in ("SOUP_s0", "SOUP_s1")]
for _k in ("CTRL_s0", "CTRL_s1"):
    if not A("lonfloor", _k)["read_xy"]["crawl_separated_better"]["M1 crawl share"]:
        die("lonfloor M1 X - Y is not separated-better at " + _k)
if not all(all(x["read_xy"]["crawl_separated_better"].values()) and x["read_xy"]["ade6_within_guard"] and x["read_xy"]["MOVED_seed"] for x in lf_s):
    die("lonfloor/SOUP MOVED condition not met at both seeds")
w("**The one MOVED condition hidden by precedence: lonfloor on SOUP.** At both SOUP seeds all four crawl rows (yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) are separated-better on X - Y and ADE6 X - Y is within the guard (%+.4f; %+.4f m), so the MOVED condition holds; HARMS wins because a non-crawl row of 15 is separated-worse on X - Y (%s at s0; %s at s1). The outcome word is HARMS. On DAG-OACC lonfloor's M1 X - Y is separated-better at both seeds but the MOVED condition does not hold (ADE6 X - Y %+.4f / %+.4f m; non-crawl rows separated-worse: %s / %s)." % (
    lf_s[0]["read_xy"]["ade6_delta"], lf_s[1]["read_xy"]["ade6_delta"], ", ".join(lf_s[0]["read_xy"]["non_crawl_separated_worse"]), ", ".join(lf_s[1]["read_xy"]["non_crawl_separated_worse"]),
    A("lonfloor", "CTRL_s0")["read_xy"]["ade6_delta"], A("lonfloor", "CTRL_s1")["read_xy"]["ade6_delta"], ", ".join(A("lonfloor", "CTRL_s0")["read_xy"]["non_crawl_separated_worse"]), ", ".join(A("lonfloor", "CTRL_s1")["read_xy"]["non_crawl_separated_worse"])))
w()
w("## 1. What triggers HARMS and what blocks DOMINANT, per lever, arm and seed (generated from the rows)")
w()
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            aa = A(m, k)
            XY = aa["read_xy"]
            sbc = [q for q, v in XY["crawl_separated_better"].items() if v]
            nsb = [q for q, v in XY["crawl_separated_better"].items() if not v]
            w("* **%s %s %s**: HARMS triggers: %s. D2 (X - P10) failing rows: %s. Crawl rows X - Y separated-better: %s; not: %s." % (m, nm, k[-2:], "; ".join(trig(aa)) or "none", ", ".join(aa["read_a"]["separated_worse_rows"]) or "none", ", ".join(sbc) or "none", ", ".join(nsb) or "none"))
w()
w("### D2 failing rows, count per lever (4 cells: DAG-OACC s0 / s1, SOUP s0 / s1)")
w()
w("| lever | M1 crawl share | curvature MAE | L2 curvature MAE (GT>=5.2) | yaw-rate MAE | any other row |")
w("|---|---|---|---|---|---|")
for m in MODES:
    cnt = {q: 0 for q in CRAWL}
    oth = []
    for k in ALLK:
        for q in A(m, k)["read_a"]["separated_worse_rows"]:
            if q in cnt:
                cnt[q] += 1
            else:
                oth.append(f"{k}:{q}")
    w("| %s | %d / 4 | %d / 4 | %d / 4 | %d / 4 | %s |" % (m, cnt["M1 crawl share"], cnt["curvature MAE"], cnt["L2 curvature MAE (GT>=5.2)"], cnt["yaw-rate MAE"], ", ".join(oth) or "none"))
w()
w("### HARMS triggers, count per lever (cells at which the row is separated-worse on X - Y, or ADE6 beyond the guard)")
w()
rows_seen = sorted({q for m in MODES for k in ALLK for q in A(m, k)["read_xy"]["non_crawl_separated_worse"]})
w("| lever | ADE6 beyond guard | " + " | ".join(rows_seen) + " |")
w("|---|---|" + "---|" * len(rows_seen))
for m in MODES:
    c = [sum(1 for k in ALLK if q in A(m, k)["read_xy"]["non_crawl_separated_worse"]) for q in rows_seen]
    w("| %s | %d / 4 | %s |" % (m, sum(1 for k in ALLK if not A(m, k)["read_xy"]["ade6_within_guard"]), " | ".join("%d / 4" % x for x in c)))
w()
w("## 2. Known answers and controls (all six groups, computed before any row)")
w()
for cname, c in J["controls"].items():
    w("* **%s**: %s" % (cname, "PASS" if c["PASS"] else "FAIL"))
w()
ky = J["controls"]["KY_reused_Y_equals_frozen_pins"]["files"]
w("**KY** (the reused Y files equal the pins of `FROZEN_SHA_FEAS3.txt`): %d / %d files hash-equal." % (sum(v["PASS"] for v in ky.values()), len(ky)))
w()
k1 = J["controls"]["KA1_fresh_rerun_equals_reused_Y"]
w("**KA1** (the fresh rerun `%s` with this tool and `--feasible off` equals the reused `%s`; tolerance 1e-4 m): dump bit-identical %s (max abs dev %.3e m); ticks bit-identical %s (max abs dev %.3e); differing keys %s; key-set difference %s." % (k1["rerun"], k1["reused"], k1["dump_bit_identical"], k1["dump_max_abs_dev_m"], k1["ticks_bit_identical"], k1["ticks_max_abs_dev"], (k1["dump_differing_keys"] + k1["ticks_differing_keys"]) or "none", k1["key_set_diff"] or "none"))
w()
w("**KA2** (X's issued `plan` = lever(P + SG(plan_raw - P), tick v0), the operators re-spelled independently of the tool; tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):")
w()
w("| run | plan_s vs P+SG (m) | plan vs lever(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | lever moved the issued plan by up to (m) |")
w("|---|---|---|---|---|---|")
for k, v in J["controls"]["KA2_issued_plan_is_lever_of_S"]["runs"].items():
    w("| %s | %.3e | %.3e | %.3e | %.3e | %.3f |" % (k, v["plan_s_vs_P_plus_SG_max_abs_m"], v["plan_vs_lever_of_plan_s_max_abs_m"], v["tick0_plan_raw_vs_Y_max_abs_m"], v["floor_max_abs_dev_m"], v["lever_changed_plan_max_abs_m"]))
w()
w("The last KA2 column is the maximum over every tick and knot of |plan - plan_s| (far unexecuted knots included); only executed knots 1-10 reach the rows.")
w()
w("**KF** (floor / window check against the fresh Y; gross guard %.1f m against the banked CTRL):" % J["controls"]["KF_floor_window_check_vs_fresh_Y"]["gross_guard_m"])
w()
w("| run | windows X = Y | windows Y = banked CTRL | floor X vs Y (m) | floor Y vs banked CTRL max / mean (m) |")
w("|---|---|---|---|---|")
for k, v in J["controls"]["KF_floor_window_check_vs_fresh_Y"]["runs"].items():
    w("| %s | %s | %s | %.3e | %.3e / %.3e |" % (k, v["windows_X_equal_fresh_Y"], v["windows_fresh_Y_equal_banked_CTRL"], v["floor_X_vs_fresh_Y_max_abs_m"], v["floor_fresh_Y_vs_banked_CTRL_max_abs_m"], v["floor_fresh_Y_vs_banked_CTRL_mean_abs_m"]))
w()
k3, kb1 = J["controls"]["K3_Y_banked_minus_P10_reproduces_SOUP_READ"], J["controls"]["KB1_P10_side_equals_landed_banked_b"]["runs"]
w("**K3**: Y banked - P10 from the landed arrays reproduces the pinned `SOUP_READ.json` on %d / %d rows (tolerance %.0e). **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays on the non-S-POS rows: %d / %d runs pass." % (k3["n_pass"], k3["n_rows"], SR.TOL_ROW, sum(v["PASS"] for v in kb1.values()), len(kb1)))
w()
km = J["disclosures"]["KM1_reader_reproduces_M1_row"]
w("**KM1** (the disclosure reader reproduces the M1 row's per-window values on Y and on X exactly): %d / %d cells True. **KA3** (`tests/test_v8f_t1_feas3.py`, 16 passed, `logs/pytest_feas3.log`; real-checkpoint smoke `logs/smoke_feas3.out`): run when the tool was built; not re-run in this read." % (sum(km.values()), len(km)))
w()
w("## 3. Read A against P10 (D1 / D2 / D3) and X - Y, per lever, arm and seed")
w()
w("The 19 rows of `extract_smooth.py`; every cell is `point [lo, hi] mark`. D1 bar: ADE6 X - P10 separated-better with point <= %+.4f m. HARMS guard: ADE6 X - Y point <= %+.4f m." % (SR.D1_POINT_M, SR.ADE_GUARD_M))
w()
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            aa = A(m, k)
            Ra, XY, R = aa["read_a"], aa["read_xy"], aa["rows"]
            ap, ax, ay = R["X_minus_P10"]["ADE6"], R["X_minus_Y"]["ADE6"], R["Y_minus_P10"]["ADE6"]
            w("### %s %s %s" % (m, nm, k[-2:]))
            w()
            w("* **PRIMARY ADE6 (m)**: X - P10 %s (D1: **%s**); X - Y %s (guard: **%s**); Y - P10 %s." % (cell("ADE6", ap), Ra["D1"], cell("ADE6", ax), XY["ade6_within_guard"], cell("ADE6", ay)))
            w("* **D2**: **%s**; separated-worse rows (X - P10): %s. **D3**: %s; separated-better rows: %d of 19." % (Ra["D2"], ", ".join(Ra["separated_worse_rows"]) or "none", Ra["D3_by_family"], len(Ra["separated_better_rows"])))
            w("* **Crawl rows X - Y separated-better**: %s. **Non-crawl rows separated-worse on X - Y**: %s. Rows separated-worse on X - Y, all 19: %d." % (XY["crawl_separated_better"], ", ".join(XY["non_crawl_separated_worse"]) or "none", XY["n_separated_worse_all"]))
            w()
            w("| row | family | X - P10 | X - Y | Y - P10 |")
            w("|---|---|---|---|---|")
            for lab in GD.LABELS:
                w("| %s | %s | %s | %s | %s |" % (lab, SR.FAMILY[lab], cell(lab, R["X_minus_P10"][lab]), cell(lab, R["X_minus_Y"][lab]), cell(lab, R["Y_minus_P10"][lab])))
            w()
w("## 4. FEAS2's kbox beside the three FEAS3 levers (X - Y, point and mark; `raw/FEAS2_T1.json`, same Y, same windows)")
w()
w("`kbox` = `project_feasible(a_max=inf, kappa_max=0.2, mu=None, clamp_entry=False)` (module default min_speed 0.5); `kbox2` = the same with min_speed 2.0.")
w()
for a, nm in ARMS:
    for k in KEYS[a]:
        w("### %s %s, X - Y" % (nm, k[-2:]))
        w()
        w("| row | kbox (FEAS2) | kbox2 | lonfloor | kbox2lf |")
        w("|---|---|---|---|---|")
        for lab in BESIDE:
            r0 = F2["arms"][f"{k}:kbox"]["rows"]["X_minus_Y"][lab]
            c = ["%+.4f %s" % (r0["delta"], mark(lab, r0))]
            for m in MODES:
                r = A(m, k)["rows"]["X_minus_Y"][lab]
                c.append("%+.4f %s" % (r["delta"], mark(lab, r)))
            w("| %s | %s |" % (lab, " | ".join(c)))
        w()
g_kbox = sum(1 for k in ALLK if not F2["arms"][f"{k}:kbox"]["read_xy"]["ade6_within_guard"])
g_kbox2 = sum(1 for k in ALLK if not A("kbox2", k)["read_xy"]["ade6_within_guard"])
hm = lambda m, k: A(m, k)["rows"]["X_minus_Y"]["heading MAE"]["delta"] if m != "kbox" else F2["arms"][f"{k}:kbox"]["rows"]["X_minus_Y"]["heading MAE"]["delta"]
w("**MEASURED.** ADE6 X - Y is beyond the guard (%+.4f m) at %d of 4 cells for kbox and at %d of 4 cells for kbox2. Heading MAE X - Y (extractor units), kbox / kbox2 per cell: %s. The card's pre-check (a plan-level counterfactual on issued plans, `raw/PRECHECK_FEAS3.json`) described kbox2 as a small change to kbox: the executed knots differ on %.1f-%.1f %% of ticks, by at most %.2f m; the rollouts do not read it as small on ADE6, and heading MAE X - Y stays separated-worse at %d of 4 kbox2 cells." % (
    SR.ADE_GUARD_M, g_kbox, g_kbox2, "; ".join("%s %+.4f / %+.4f" % (k, hm("kbox", k), hm("kbox2", k)) for k in ALLK),
    100 * min(PC["runs"][r]["kbox2_vs_kbox"]["tick_share_executed_knots_differ_gt_1mm"] for r in PC["runs"]), 100 * max(PC["runs"][r]["kbox2_vs_kbox"]["tick_share_executed_knots_differ_gt_1mm"] for r in PC["runs"]),
    max(PC["runs"][r]["kbox2_vs_kbox"]["max_m"] for r in PC["runs"]), sum(1 for k in ALLK if "heading MAE" in A("kbox2", k)["read_xy"]["non_crawl_separated_worse"])))
w()
w("## 5. The prediction next to the result")
w()
w("Card (ESTIMATED, written before any X number of these arms), per lever: P(outcome) in %; read = the outcome at DAG-OACC / SOUP.")
w()
w("| lever | DOMINANT | HARMS | MOVED | NO EFFECT | read (DAG-OACC / SOUP) | probability the card gave the outcome read |")
w("|---|---|---|---|---|---|---|")
for m in MODES:
    o1, o2 = J["outcome"][m]["DAG-OACC"]["outcome"], J["outcome"][m]["SOUP"]["outcome"]
    p = PRED[m]
    w("| %s | %d %% | %d %% | %d %% | %d %% | **%s / %s** | %d %% / %d %% |" % (m, p["DOMINANT"], p["HARMS"], p["MOVED"], p["NO EFFECT"], o1, o2, p[o1], p[o2]))
w()
w("M1 (crawl share X - Y; share of the Y - P10 excess closed = -(X - Y) / (Y - P10); negative = M1 moved the wrong way). Card: kbox2 unmoved (< 5 % of its excess); lonfloor moves < 5 % of its excess, P(separated-better at both seeds, both arms) 8 %; kbox2lf as lonfloor. P(M1 keeps D2 failing under all three arms) = 92 %. P(any arm DOMINANT) = 4 %.")
w()
w("| lever | cell | M1 X - Y | Y - P10 | share of excess closed | X - Y separated-better | M1 fails D2 (X - P10) |")
w("|---|---|---|---|---|---|---|")
shares = {m: [] for m in MODES}
sbn = {m: 0 for m in MODES}
m1_fail = 0
for m in MODES:
    for k in ALLK:
        aa = A(m, k)
        xy, yp = aa["rows"]["X_minus_Y"]["M1 crawl share"], aa["rows"]["Y_minus_P10"]["M1 crawl share"]
        f = "M1 crawl share" in aa["read_a"]["separated_worse_rows"]
        m1_fail += f
        if yp["delta"] <= 0:
            die("Y - P10 M1 is not an excess at " + k)
        sh = -xy["delta"] / yp["delta"]
        shares[m].append(sh)
        sbn[m] += SR.better(GD, "M1 crawl share", xy)
        w("| %s | %s %s | %+.4f | %+.4f | %.1f %% | %s | %s |" % (m, nm_of(k), k[-2:], xy["delta"], yp["delta"], 100.0 * sh, SR.better(GD, "M1 crawl share", xy), f))
w()
for m in MODES:
    s = shares[m]
    w("* **%s**: M1 X - Y separated-better at %d of 4 cells; share of the excess closed %.1f %% to %.1f %% (mean %.1f %%); below 5 %% at %d of 4 cells." % (m, sbn[m], 100 * min(s), 100 * max(s), 100 * sum(s) / 4, sum(1 for x in s if x < 0.05)))
w()
dom_any = sum(1 for m in MODES for _, nm in ARMS if J["outcome"][m][nm]["DOMINANT"])
w("**MEASURED:** M1 crawl share fails D2 at %d of 12 cells (the card gave 92 %% that it fails under all three arms; held). Any arm DOMINANT: %d of 6 (the card gave 4 %%). kbox2 read HARMS at both arms (card 70 %%) and kbox2lf read HARMS at both arms (card 70 %%). **lonfloor was predicted NO EFFECT at 85 %% and read HARMS at both arms (the card gave HARMS 12 %%)**; the card's 8 %% event (M1 X - Y separated-better at both seeds on both arms) occurred for lonfloor (%d of 4 cells); the card's size estimate for lonfloor (< 5 %% of its excess closed) held at %d of 4 cells (largest %.1f %%). The HARMS triggers of lonfloor are rows of small magnitude (speed MAE, accel MAE, cross-track MAE, along-track MAE, ADE6 within the guard at %d of 4 cells), listed per cell in section 1." % (m1_fail, dom_any, sbn["lonfloor"], sum(1 for x in shares["lonfloor"] if x < 0.05), 100 * max(shares["lonfloor"]), sum(1 for k in ALLK if A("lonfloor", k)["read_xy"]["ade6_within_guard"])))
w()
w("## 6. Disclosure (i): M1 at 1 / 2 / 3 m/s margins and the row definition (pre-registered beside the outcome; changes no outcome word)")
w()
w("`row def (5.2 m/s)` = the M1 row of the 19 (`rr_lib.m1_window`: per window, among GT-fast pairs, GT pair speed >= 5.2 m/s, the share whose own pair speed is < 5.2 m/s). `margin m` = per window, among the GT-fast pairs, the share whose OWN pair speed is < GT pair speed - m. Mean over windows of the share, for X, Y and P10 (P10 = the fresh Y run's own P10 floor run); X - P10, X - Y, Y - P10 with the paired bootstrap. Lower is better.")
w()
D = J["disclosures"]["M1_margins"]
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            d = D[f"{k}:{m}"]
            w("### %s %s %s" % (m, nm, k[-2:]))
            w()
            w("| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |")
            w("|---|---|---|---|---|---|---|")
            for tk, tl in THR:
                z = d[tk]
                w("| %s | %.4f | %.4f | %.4f | %s | %s | %s |" % (tl, z["mean_share"]["X"], z["mean_share"]["Y"], z["mean_share"]["P10"], cell2(z["X_minus_P10"]), cell2(z["X_minus_Y"]), cell2(z["Y_minus_P10"])))
            w()
            w("Windows %d (X - P10, X - Y), %d (Y - P10); episodes in the pool %d." % (d["thr_5.2_row_definition"]["X_minus_Y"]["n_windows"], d["thr_5.2_row_definition"]["Y_minus_P10"]["n_windows"], d["thr_5.2_row_definition"]["X_minus_Y"]["n_episodes_pool"]))
            w()
w("### Where M1 X - Y is separated-better, per cell (SB = separated-better; the card's THRESHOLD-LOCAL reading)")
w()
w("| lever | cell | 5.2 row | 1 m/s | 2 m/s | 3 m/s | THRESHOLD-LOCAL (literal: SB on 5.2 or 2, SB on neither 1 nor 3) | broader reading: SB on 5.2 or 2 but not on both 1 and 3 |")
w("|---|---|---|---|---|---|---|---|")
tl_lit, tl_broad = {m: [] for m in MODES}, {m: [] for m in MODES}
sb_cnt = {m: [0, 0, 0, 0] for m in MODES}
for m in MODES:
    for k in ALLK:
        d = D[f"{k}:{m}"]
        p = [mk2(d[t]["X_minus_Y"]) == "SB" for t, _ in THR]
        for i in range(4):
            sb_cnt[m][i] += p[i]
        lit = (p[0] or p[2]) and not p[1] and not p[3]
        brd = (p[0] or p[2]) and not (p[1] and p[3])
        if lit:
            tl_lit[m].append(k)
        if brd:
            tl_broad[m].append(k)
        w("| %s | %s %s | %s | %s | %s | %s | %s | %s |" % (m, nm_of(k), k[-2:], *["SB" if x else "-" for x in p], "yes" if lit else "no", "yes" if brd else "no"))
w()
w("Cells at which M1 X - Y is separated-better, per lever (5.2 row / 1 m/s / 2 m/s / 3 m/s, of 4): " + "; ".join("%s %s" % (m, " / ".join(str(x) for x in sb_cnt[m])) for m in MODES) + ".")
w()
lit_all = [f"{m}:{k}" for m in MODES for k in tl_lit[m]]
brd_all = [f"{m}:{k}" for m in MODES for k in tl_broad[m]]
w("**Headline reading.** THRESHOLD-LOCAL under the card's wording read literally (separated-better on the 5.2 row or the 2 m/s row, and on neither the 1 m/s nor the 3 m/s row): %s. Under the broader reading (not separated-better on both the 1 m/s and the 3 m/s row): %s. The card's wording admits both; both are printed, the outcome words do not change either way." % (", ".join(lit_all) or "no cell", ", ".join(brd_all) or "no cell"))
w()
sh1 = {m: [-D[f"{k}:{m}"]["margin_1"]["X_minus_Y"]["delta"] / D[f"{k}:{m}"]["margin_1"]["Y_minus_P10"]["delta"] for k in ALLK] for m in MODES}
if not all(D[f"{k}:{m}"]["margin_1"]["Y_minus_P10"]["delta"] > D[f"{k}:{m}"]["thr_5.2_row_definition"]["Y_minus_P10"]["delta"] > 0 for m in MODES for k in ALLK):
    die("margin-1 Y - P10 is not above the row definition's at every cell")
w("**What the margins show.** lonfloor's M1 X - Y is separated-better on the 5.2 row at %d of 4 cells and on the 2 m/s margin at %d of 4, so its small M1 effect is not an artefact of the 5.2 definition; the largest M1 X - Y on the 5.2 row is %+.4f (share of the Y - P10 excess closed %.1f %%-%.1f %%, section 5). The Y - P10 excess at margin 1 is larger than on the 5.2 row at every cell (%.4f-%.4f against %.4f-%.4f); the share of the margin-1 excess closed is %.1f %%-%.1f %% for lonfloor, %.1f %%-%.1f %% for kbox2 and %.1f %%-%.1f %% for kbox2lf." % (
    sb_cnt["lonfloor"][0], sb_cnt["lonfloor"][2], min(D[f"{k}:lonfloor"]["thr_5.2_row_definition"]["X_minus_Y"]["delta"] for k in ALLK), 100 * min(shares["lonfloor"]), 100 * max(shares["lonfloor"]),
    min(D[f"{k}:{m}"]["margin_1"]["Y_minus_P10"]["delta"] for m in MODES for k in ALLK), max(D[f"{k}:{m}"]["margin_1"]["Y_minus_P10"]["delta"] for m in MODES for k in ALLK),
    min(D[f"{k}:{m}"]["thr_5.2_row_definition"]["Y_minus_P10"]["delta"] for m in MODES for k in ALLK), max(D[f"{k}:{m}"]["thr_5.2_row_definition"]["Y_minus_P10"]["delta"] for m in MODES for k in ALLK),
    100 * min(sh1["lonfloor"]), 100 * max(sh1["lonfloor"]), 100 * min(sh1["kbox2"]), 100 * max(sh1["kbox2"]), 100 * min(sh1["kbox2lf"]), 100 * max(sh1["kbox2lf"])))
w()
w("## 7. Disclosure (ii): the hard-braking windows (pre-registered; changes no outcome word)")
w()
hs = J["disclosures"]["hard_braking_set"]
h0 = hs["CTRL_s0"]
w("Windows in the top 5 %% of GT peak deceleration (peak of -(d vG / dt) over the GT pair speeds; fixed by GT alone, one set for every arm): %d windows from %d episodes, threshold %.3f m/s^2 on the %d windows with a finite deceleration; identical at all four cells: %s. X - Y on those windows, **19 episodes make the intervals wide: a null here is not evidence of safety**." % (h0["n_windows"], h0["n_episodes"], h0["threshold_peak_decel_mps2"], h0["n_windows_with_decel"], all(hs[k] == h0 for k in ALLK)))
w()
HBD = J["disclosures"]["hard_braking"]
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            d = HBD[f"{k}:{m}"]
            w("### %s %s %s" % (m, nm, k[-2:]))
            w()
            w("| row | X - Y on the braking set | windows | episodes |")
            w("|---|---|---|---|")
            for q in HB:
                w("| %s | %s | %d | %d |" % (q, cell2(d[q]), d[q]["n_windows"], d[q]["n_episodes_pool"]))
            w()
w("### Separated-worse rows on the braking set, per lever (cells of 4)")
w()
w("| lever | along-track MAE | S-POS@30 | S-POS@50 | S-POS@80 | outcome word | BRAKING-BLOCKED applies (MOVED or DOMINANT read) |")
w("|---|---|---|---|---|---|---|")
hb_cnt = {m: {q: sum(1 for k in ALLK if mk2(HBD[f"{k}:{m}"][q]) == "SW") for q in HB} for m in MODES}
for m in MODES:
    w("| %s | %d / 4 | %d / 4 | %d / 4 | %d / 4 | HARMS / HARMS | no: no arm reads MOVED or DOMINANT |" % (m, *[hb_cnt[m][q] for q in HB]))
w()
hbp = {r: PC["runs"][r]["hard_braking_ticks"] for r in PC["runs"]}
w("**Headline reading.** No arm reads MOVED or DOMINANT, so BRAKING-BLOCKED (the card's label for a MOVED or DOMINANT arm that is separated-worse on any of the four rows on this set) is not applied to any arm. Beside the outcome: along-track MAE X - Y on the braking set is separated-worse at %d of 4 cells for kbox2, %d of 4 for lonfloor and %d of 4 for kbox2lf; S-POS@30/50/80 X - Y is separated-worse at %d of 36 cells (3 rows x 4 cells x 3 levers). The one MOVED condition hidden by precedence (lonfloor on SOUP) has along-track MAE separated-worse on the braking set at both seeds (%+.4f, %+.4f m); had that arm read MOVED the card's label would have been BRAKING-BLOCKED. This is the direction the card's pre-check anticipated for lonfloor (a floor built on a prior that does not predict braking sits above real deceleration; plan-level step-speed MAE against the logged future on the %d top-5 %% ticks of each run, plan_s -> lonfloor in m/s per run: %s); the rollout read is on windows, the pre-check on ticks." % (
    hb_cnt["kbox2"]["along-track MAE"], hb_cnt["lonfloor"]["along-track MAE"], hb_cnt["kbox2lf"]["along-track MAE"], sum(hb_cnt[m][q] for m in MODES for q in HB if q != "along-track MAE"),
    HBD["SOUP_s0:lonfloor"]["along-track MAE"]["delta"], HBD["SOUP_s1:lonfloor"]["along-track MAE"]["delta"], hbp["SREP_DAGOACC_s0"]["n"] if len({hbp[r]["n"] for r in hbp}) == 1 else die("tick counts differ across the four pre-check runs"),
    ", ".join("%.2f -> %.2f" % (hbp[r]["step_speed_mae_mps"]["plan_s"], hbp[r]["step_speed_mae_mps"]["lonfloor"]) for r in PC["runs"])))
w()
w("## 8. The reader fix (an import-path edit after the first read stopped; `READER_FIX_FEAS3.md`)")
w()
w("* **What failed.** The 12 rollouts finished (nfail 0); the first read stopped at rc 1 inside `disclosures()` (the card's beside checks) with `taniteval.stack_guard.StackShadowError`: `lever()` had put the arm tree's stack root (`F:/TanitAD-scratch/feas2_t1/tree_dag/stack`) on `sys.path` to import `tanitad.refs.feasible_decode`, so a second `tanitad` root preceded the pinned one when `rr_lib` imported `taniteval`; the guard refused as designed. Marker kept: `raw/FEAS3.fail.first_read` (`%s`). `logs/read_feas3.out` holds the traceback and the 12 per-cell lines the read had printed before the stop; no `FEAS3_T1.json` existed until the re-run." % first_fail)
w("* **The fix (path only).** `lever()` loads `tree_dag/stack/tanitad/refs/feasible_decode.py` by file path (`importlib.util.spec_from_file_location`) instead of adding the arm tree's stack root to `sys.path`; that file imports only `math` and `torch` and is byte-identical to the pinned tree's copy (`READER_FIX_FEAS3.md`: `diff -q`, no difference). The diff of the reader is that one hunk (3 lines replaced).")
w("* **Reader shas.** `FROZEN_SHA_FEAS3.txt` pins `code/feas3_t1_read.py` at `%s`; the pre-fix file kept at `%s` hashes to `%s` (%s the pin); the reader used for the re-run (current `code/feas3_t1_read.py`) hashes to `%s`." % (pin_line, PRE, pre_sha, "equal to" if pre_sha == pin_line else "NOT equal to", cur_sha))
w("* **Nothing else moved.** No row, threshold, arm, control, outcome rule or bootstrap setting changed and no rollout was re-run. The per-cell outcome lines of the first read and of the re-run are compared line by line here: %d of 12 identical (%s). Marker of the re-run: `raw/FEAS3.done` (`%s`)." % (sum(1 for a, b in zip(l1, l2) if a == b), "identical" if same else "NOT IDENTICAL", done))
w()
w("## 9. The card's committed next arm per outcome (Rule Zero), and what the read selects")
w()
w("The card's lines: **HARMS on heading / S-POS at kbox2 and kbox2lf**: the decode-time box family (box, kbox, abox, kabox, kbox2) is closed as a heading / S-POS lever (the clamp acts on executed steps the logged future truly turns on); remaining levers, each with its blocker: (c) the lateral shrink `P + diag(1, alpha)(S(plan) - P)` with alpha chosen on TRAIN windows (blocked: the tool rolls out eval-141 only; whether DAgger's TRAIN-window rollout inputs exist on disk is **UNVERIFIED**); the clamp as a loss term or output layer at training time (blocked: the PI's go and compute).")
w()
w("* **kbox2 and kbox2lf (HARMS at 4 of 4 cells each)**: the trigger the card names, heading MAE X - Y separated-worse, is present at %d of 4 and %d of 4 cells, S-POS@30/50/80 at %d of 4 and %d of 4 (all three rows); ADE6 X - Y is beyond the guard at %d of 4 and %d of 4. The card's line applies: box family closed as a heading / S-POS lever; remaining levers (c) and the training-time clamp, blockers as above." % (
    sum("heading MAE" in A("kbox2", k)["read_xy"]["non_crawl_separated_worse"] for k in ALLK), sum("heading MAE" in A("kbox2lf", k)["read_xy"]["non_crawl_separated_worse"] for k in ALLK),
    sum(all(q in A("kbox2", k)["read_xy"]["non_crawl_separated_worse"] for q in ("S-POS@30", "S-POS@50", "S-POS@80")) for k in ALLK), sum(all(q in A("kbox2lf", k)["read_xy"]["non_crawl_separated_worse"] for q in ("S-POS@30", "S-POS@50", "S-POS@80")) for k in ALLK),
    g_kbox2, sum(1 for k in ALLK if not A("kbox2lf", k)["read_xy"]["ade6_within_guard"])))
if not all({"M1 crawl share", "curvature MAE", "L2 curvature MAE (GT>=5.2)"} <= set(A("lonfloor", k)["read_a"]["separated_worse_rows"]) for k in ALLK) or "yaw-rate MAE" not in A("lonfloor", "CTRL_s1")["read_a"]["separated_worse_rows"]:
    die("lonfloor D2 failing rows differ from the fixed prose of section 9")
w("* **lonfloor (HARMS at 4 of 4 cells; MOVED condition met on SOUP)**: the card committed lines for DOMINANT, MOVED, HARMS-on-heading (kbox2 and kbox2lf) and NO EFFECT on lonfloor; it committed none for HARMS at lonfloor. The nearest committed lines are NO EFFECT-with-M1-separated-worse (partition M1's executed excess by window position, speed band and whether the preceding executed step was already below the prior) and MOVED (name the rows still separated-worse vs P10 per seed: here M1 crawl share, curvature MAE and L2 curvature MAE at every lonfloor cell, yaw-rate MAE at DAG-OACC s1). No such partition is run in this deliverable; the executed-state question was taken up by the separate package `../histswap` (ORACLE diagnostic, `../histswap/RESULTS.md`).")
w("* **No arm is DOMINANT**, so the card's DOMINANT line (re-read on a fresh seed or window set; the PI's go for the training lever) is not triggered.")
w()
w("## 10. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)")
w()
for m in MODES:
    for k in ALLK:
        d = J["descriptive"][f"{k}:{m}"]
        px, py = d["curvature_partition_X"], d["curvature_partition_Y"]
        f = lambda z: json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in z.items() if not isinstance(v, (list, dict))})
        w("* **%s %s**: X %s | Y %s" % (m, k, f(px), f(py)))
w()
w("Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).")
w()
w("## 11. Limits")
w()
w("* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run). The planner is deterministic, so *another inference run* is not in play for these arms.")
w("* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.")
w("* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception. Inference inputs: vision plus measured ego history.")
w("* The braking-set intervals rest on 19 episodes (section 7). The threshold-local wording is read two ways (section 6).")
w("* The reader was edited after data in one import path (section 8); no number changed.")
w("* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row `H-FEAS3-T1-1` (a proposed id).")
open(f"{PK}/RESULTS_FEAS3.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("RESULTS_FEAS3.md written", len(L), "lines")
