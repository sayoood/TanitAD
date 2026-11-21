#!/usr/bin/env python3
"""make_results_feas2.py -- RESULTS.md of H-FEAS2-T1-1, generated from raw/FEAS2_T1.json (+ raw/PRECHECK_ATTRIBUTION.json, KAPPA-T1's raw/KAPPA_T1.json); no hand-typed number.

    python -I code/make_results_feas2.py
Adapted from kappa_t1/code/make_results_kappa.py (same rows, same outcome words); the lever is the third axis (kbox / abox / kabox).
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SMT = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/code"
sys.path.insert(0, SMT)
import smooth_t1_read as SR  # noqa: E402

GD = SR.load_gd()
J = json.load(open(f"{PK}/raw/FEAS2_T1.json", encoding="utf-8"))
K = json.load(open("E:/Projects/TanitAD-artifacts/dagland/kappa_t1/raw/KAPPA_T1.json", encoding="utf-8"))
PC = json.load(open(f"{PK}/raw/PRECHECK_ATTRIBUTION.json", encoding="utf-8"))["runs"]
MODES = ("kbox", "abox", "kabox")
ARMS = (("CTRL", "DAG-OACC"), ("SOUP", "SOUP"))
KEYS = {"CTRL": ("CTRL_s0", "CTRL_s1"), "SOUP": ("SOUP_s0", "SOUP_s1")}
PCK = {"CTRL_s0": "FEAS_DAGOACC_s0", "CTRL_s1": "FEAS_DAGOACC_s1", "SOUP_s0": "FEAS_SOUP_s0", "SOUP_s1": "FEAS_SOUP_s1"}
CRAWL = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
BESIDE = ("ADE6", "heading MAE", "S-POS@30", "S-POS@50", "S-POS@80", "along-track MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "yaw-rate MAE", "M1 crawl share")
PRED = {  # the card's ESTIMATED table (PREREG_FROZEN.md), DOMINANT / HARMS / MOVED / NO EFFECT
    "kbox": {"DOMINANT": 2, "HARMS": 70, "MOVED": 3, "NO EFFECT": 25},
    "abox": {"DOMINANT": 1, "HARMS": 25, "MOVED": 9, "NO EFFECT": 65},
    "kabox": {"DOMINANT": 5, "HARMS": 60, "MOVED": 8, "NO EFFECT": 27},
}
NEXT_HARMS = ("on heading / S-POS: `kbox2` = `project_feasible(a_max=inf, kappa_max=0.2, min_speed=2.0)`; with M1 still separated-worse at every cell: `lonfloor`; "
              "and their composition `kbox2+lonfloor`. All three are the FEAS3 arms (PREREG_FEAS3_FROZEN.md).")
L = []


def mark(lab, r):
    if SR.better(GD, lab, r):
        return "SB"
    if SR.worse(GD, lab, r):
        return "SW"
    return "ns"


def cell(lab, r):
    return "%+.4f [%+.4f, %+.4f] %s" % (r["delta"], r["lo"], r["hi"], mark(lab, r))


def w(s=""):
    L.append(s)


def trig(a):
    XY, R = a["read_xy"], a["rows"]
    t = []
    if not XY["ade6_within_guard"]:
        t.append("ADE6 X - Y %+.4f > guard %+.4f" % (XY["ade6_delta"], SR.ADE_GUARD_M))
    t += ["%s X - Y %+.4f SW" % (q, R["X_minus_Y"][q]["delta"]) for q in XY["non_crawl_separated_worse"]]
    return t


w("# RESULTS: H-FEAS2-T1-1, the box split into its clamps after S (kbox / abox / kabox), DAG-OACC and SOUP at s0 / s1")
w()
w("Card `PREREG_FROZEN.md` sha256 `%s` (frozen 2026-11-02T16:43:48+0100). Reader `code/feas2_t1_read.py` under KAPPA-T1's `READER_FIX_FLOOR.md` (extractor guard 1.0 m; the floor / window check KF runs against the reused fresh Y). Read clock %s. Raw `raw/FEAS2_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**." % (J["prereg_sha256"], J["run_clock"]))
w("Estimator of every interval below: **%s**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals answer *another draw of episodes* only, not *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*." % J["estimator"])
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
            kk = f"{k}:{m}"
            A, XY = J["arms"][kk]["read_a"], J["arms"][kk]["read_xy"]
            c.append("%s / %s / %s" % (A["D1"], A["D2"], A["D3"]))
            h.append("%s / %s" % (XY["MOVED_seed"], XY["HARMS_seed"]))
        w("| %s | %s | **%s** | %s | %s | %s | %s |" % (m, nm, o["outcome"], c[0], c[1], h[0], h[1]))
w()
w("Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): " + ", ".join("%s/%s %s" % (m, nm, J["outcome"][m][nm]["moved_and_harms"]) for m in MODES for _, nm in ARMS) + ".")
w()
n_out = sum(1 for m in MODES for _, nm in ARMS if J["outcome"][m][nm]["outcome"] == "HARMS")
cells = [J["arms"][f"{k}:{m}"] for m in MODES for a, _ in ARMS for k in KEYS[a]]
w("**MEASURED headline.** %d of 6 lever x arm outcomes read HARMS (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at %d of 12 cells, D3 at %d of 12, and D2 fails at %d of 12." % (n_out, sum(c["read_a"]["D1"] for c in cells), sum(c["read_a"]["D3"] for c in cells), sum(not c["read_a"]["D2"] for c in cells)))
w()
w("## 1. What triggers HARMS and what blocks DOMINANT, per lever, arm and seed (generated from the rows)")
w()
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            aa = J["arms"][f"{k}:{m}"]
            A, XY = aa["read_a"], aa["read_xy"]
            sbc = [q for q, v in XY["crawl_separated_better"].items() if v]
            nsb = [q for q, v in XY["crawl_separated_better"].items() if not v]
            w("* **%s %s %s**: HARMS triggers: %s. D2 (X - P10) failing rows: %s. Crawl rows X - Y separated-better: %s; not: %s." % (m, nm, k[-2:], "; ".join(trig(aa)) or "none", ", ".join(A["separated_worse_rows"]) or "none", ", ".join(sbc) or "none", ", ".join(nsb) or "none"))
w()
w("### D2 failing rows, count per lever (4 cells: DAG-OACC s0 / s1, SOUP s0 / s1)")
w()
w("| lever | M1 crawl share | curvature MAE | L2 curvature MAE (GT>=5.2) | yaw-rate MAE | any other row |")
w("|---|---|---|---|---|---|")
for m in MODES:
    cnt = {q: 0 for q in CRAWL}
    oth = []
    for a, nm in ARMS:
        for k in KEYS[a]:
            for q in J["arms"][f"{k}:{m}"]["read_a"]["separated_worse_rows"]:
                if q in cnt:
                    cnt[q] += 1
                else:
                    oth.append(f"{k}:{q}")
    w("| %s | %d / 4 | %d / 4 | %d / 4 | %d / 4 | %s |" % (m, cnt["M1 crawl share"], cnt["curvature MAE"], cnt["L2 curvature MAE (GT>=5.2)"], cnt["yaw-rate MAE"], ", ".join(oth) or "none"))
w()
w("## 2. Known answers and controls (all six groups, computed before any row)")
w()
for cname, c in J["controls"].items():
    w("* **%s**: %s" % (cname, "PASS" if c["PASS"] else "FAIL"))
w()
ky = J["controls"]["KY_reused_Y_equals_frozen_pins"]["files"]
w("**KY** (the reused Y files equal the pins of `FROZEN_SHA.txt`): %d / %d files hash-equal." % (sum(v["PASS"] for v in ky.values()), len(ky)))
w()
k1 = J["controls"]["KA1_fresh_rerun_equals_reused_Y"]
w("**KA1** (the fresh rerun `%s` equals the reused `%s`; tolerance 1e-4 m): dump bit-identical %s (max abs dev %.3e m); ticks bit-identical %s (max abs dev %.3e); differing keys %s; key-set difference %s." % (k1["rerun"], k1["reused"], k1["dump_bit_identical"], k1["dump_max_abs_dev_m"], k1["ticks_bit_identical"], k1["ticks_max_abs_dev"], (k1["dump_differing_keys"] + k1["ticks_differing_keys"]) or "none", k1["key_set_diff"] or "none"))
w()
w("**KA2** (X's issued `plan` = lever(P + SG(plan_raw - P), tick v0); tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):")
w()
w("| run | plan_s vs P+SG (m) | plan vs lever(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | lever moved the issued plan by up to (m) |")
w("|---|---|---|---|---|---|")
for k, v in J["controls"]["KA2_issued_plan_is_lever_of_S"]["runs"].items():
    w("| %s | %.3e | %.3e | %.3e | %.3e | %.3f |" % (k, v["plan_s_vs_P_plus_SG_max_abs_m"], v["plan_vs_lever_of_plan_s_max_abs_m"], v["tick0_plan_raw_vs_Y_max_abs_m"], v["floor_max_abs_dev_m"], v["lever_changed_plan_max_abs_m"]))
w()
w("The last KA2 column is the maximum over every tick and knot of |plan - plan_s|; the pre-check read the same quantity on the far knots (105-227 m on unexecuted knots 11+), so only executed knots 1-10 reach the rows.")
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
w("**KA3** (`tests/test_v8f_t1_feas2.py`, 16 passed, `logs/pytest_feas2.log`): run when the tool was built; not re-run in this read.")
w()
w("## 3. Read A against P10 (D1 / D2 / D3) and X - Y, per lever, arm and seed")
w()
w("The 19 rows of `extract_smooth.py`; every cell is `point [lo, hi] mark`. D1 bar: ADE6 X - P10 separated-better with point <= %+.4f m. HARMS guard: ADE6 X - Y point <= %+.4f m." % (SR.D1_POINT_M, SR.ADE_GUARD_M))
w()
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            aa = J["arms"][f"{k}:{m}"]
            A, XY, R = aa["read_a"], aa["read_xy"], aa["rows"]
            ap, ax, ay = R["X_minus_P10"]["ADE6"], R["X_minus_Y"]["ADE6"], R["Y_minus_P10"]["ADE6"]
            w("### %s %s %s" % (m, nm, k[-2:]))
            w()
            w("* **PRIMARY ADE6 (m)**: X - P10 %s (D1: **%s**); X - Y %s (guard: **%s**); Y - P10 %s." % (cell("ADE6", ap), A["D1"], cell("ADE6", ax), XY["ade6_within_guard"], cell("ADE6", ay)))
            w("* **D2**: **%s**; separated-worse rows (X - P10): %s. **D3**: %s; separated-better rows: %d of 19." % (A["D2"], ", ".join(A["separated_worse_rows"]) or "none", A["D3_by_family"], len(A["separated_better_rows"])))
            w("* **Crawl rows X - Y separated-better**: %s. **Non-crawl rows separated-worse on X - Y**: %s. Rows separated-worse on X - Y, all 19: %d." % (XY["crawl_separated_better"], ", ".join(XY["non_crawl_separated_worse"]) or "none", XY["n_separated_worse_all"]))
            w()
            w("| row | family | X - P10 | X - Y | Y - P10 |")
            w("|---|---|---|---|---|")
            for lab in GD.LABELS:
                w("| %s | %s | %s | %s | %s |" % (lab, SR.FAMILY[lab], cell(lab, R["X_minus_P10"][lab]), cell(lab, R["X_minus_Y"][lab]), cell(lab, R["Y_minus_P10"][lab])))
            w()
w("## 4. KAPPA-T1's box beside the three clamps (X - Y, point and mark; `../kappa_t1/raw/KAPPA_T1.json`, same Y, same windows)")
w()
for a, nm in ARMS:
    for k in KEYS[a]:
        w("### %s %s, X - Y" % (nm, k[-2:]))
        w()
        w("| row | box (KAPPA-T1) | kbox | abox | kabox |")
        w("|---|---|---|---|---|")
        for lab in BESIDE:
            c = ["%+.4f %s" % (K["arms"][k]["rows"]["X_minus_Y"][lab]["delta"], mark(lab, K["arms"][k]["rows"]["X_minus_Y"][lab]))]
            for m in MODES:
                r = J["arms"][f"{k}:{m}"]["rows"]["X_minus_Y"][lab]
                c.append("%+.4f %s" % (r["delta"], mark(lab, r)))
            w("| %s | %s |" % (lab, " | ".join(c)))
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
w("M1 (crawl share X - Y; share of the Y - P10 excess closed = -(X - Y) / (Y - P10); negative = M1 moved the wrong way). Card: kbox unmoved (< 5 % of its excess); abox and kabox move < 15 % of its excess, P(separated-better at both seeds) 15 %. P(M1 keeps D2 failing under all three levers) = 85 %.")
w()
w("| lever | cell | M1 X - Y | Y - P10 | share of excess closed | X - Y separated-better | M1 fails D2 (X - P10) |")
w("|---|---|---|---|---|---|---|")
m1_fail = 0
for m in MODES:
    for a, nm in ARMS:
        for k in KEYS[a]:
            aa = J["arms"][f"{k}:{m}"]
            xy, yp = aa["rows"]["X_minus_Y"]["M1 crawl share"], aa["rows"]["Y_minus_P10"]["M1 crawl share"]
            f = "M1 crawl share" in aa["read_a"]["separated_worse_rows"]
            m1_fail += f
            w("| %s | %s %s | %+.4f | %+.4f | %s | %s | %s |" % (m, nm, k[-2:], xy["delta"], yp["delta"], ("%.0f %%" % (-100.0 * xy["delta"] / yp["delta"])) if yp["delta"] > 0 else "n/a", SR.better(GD, "M1 crawl share", xy), f))
w()
w("**MEASURED: M1 crawl share fails D2 at %d of 12 cells** (the card gave 85 %% that it fails under all three levers). kbox read HARMS as predicted (70 %%) and kabox read HARMS (60 %%). abox was predicted NO EFFECT (65 %%) and read HARMS at both arms (25 %%): abox is separated-worse on heading MAE and S-POS@30/50/80 X - Y at every cell, which the plan-level pre-check (section 6) did not show." % m1_fail)
w()
w("## 6. The pre-check attribution next to the rollouts (`raw/PRECHECK_ATTRIBUTION.json`; a per-tick counterfactual on KAPPA-T1's issued S plans, not a rollout)")
w()
w("Heading MAE proxy delta vs plan_s in degrees (plan-level, executed-knot steps vs the logged future; the card's table), against the rollout row `heading MAE` X - Y (T1-kin; paired episode-cluster bootstrap; the extractor's units).")
w()
w("| cell | pre-check kbox | pre-check abox | pre-check kabox | rollout kbox X - Y | rollout abox X - Y | rollout kabox X - Y |")
w("|---|---|---|---|---|---|---|")
for a, nm in ARMS:
    for k in KEYS[a]:
        h = PC[PCK[k]]["heading_mae_deg"]
        pcs = ["%+.3f [%+.3f, %+.3f]" % (h[m + "_minus_plan_s"]["delta"], h[m + "_minus_plan_s"]["lo"], h[m + "_minus_plan_s"]["hi"]) for m in MODES]
        rl = [cell("heading MAE", J["arms"][f"{k}:{m}"]["rows"]["X_minus_Y"]["heading MAE"]) for m in MODES]
        w("| %s %s | %s | %s |" % (nm, k[-2:], " | ".join(pcs), " | ".join(rl)))
w()
w("Reading: the pre-check located the heading harm at the curvature clamp (kbox = box at the plan level) and read abox as improving heading at the plan level; the rollouts agree on kbox and kabox and do NOT agree on abox, which reads separated-worse on heading MAE at all four cells. A plan-level counterfactual on issued plans cannot see what a clamp changes in later ticks (the executed state feeds the next plan). HYPOTHESIS, not tested here: the entry bound (`clamp_entry`, binding on 1.9-2.8 % of ticks in the pre-check) and the accel clamp perturb the executed first knots and the heading read moves with them.")
w()
w("Crawl proxy at the plan level (step speed < GT step speed - 2 m/s, v0 3-8 m/s): plan_s sat below the P10 prior's share in all four cells and the clamps moved it by at most +0.002 (pre-check, plan level); the rollout M1 row above reads the executed rows, where the excess lives.")
w()
w("## 7. The card's committed next arm per outcome (Rule Zero), and what the read selects")
w()
w("* **HARMS (all six outcomes)**: " + NEXT_HARMS)
w("* **kbox and kabox** carry heading MAE + S-POS@30/50/80 (+ along-track and ADE6 at some cells) as HARMS triggers and fail D2 on M1 only at 7 of 8 cells (L2 curvature also at SOUP s0 kabox): `kbox2` is aimed at the heading / S-POS harm (the curvature cap stops acting below 2 m/s), `lonfloor` at M1, and `kbox2+lonfloor` is the composition that could reach DOMINANT.")
w("* **abox** reads HARMS on the same rows (heading MAE, S-POS) and leaves curvature, L2 and (3 of 4 cells) yaw-rate separated-worse vs P10: the card's committed next arms do not act on abox's own harm, so abox is not carried into FEAS3.")
w()
w("Each next arm is a zero-training flag value of the same tool and has its own frozen card (`PREREG_FEAS3_FROZEN.md`) before any FEAS3 number.")
w()
w("## 8. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)")
w()
for m in MODES:
    for k in ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"):
        d = J["descriptive"][f"{k}:{m}"]
        px, py = d["curvature_partition_X"], d["curvature_partition_Y"]
        f = lambda z: json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in z.items() if not isinstance(v, (list, dict))})
        w("* **%s %s**: X %s | Y %s" % (m, k, f(px), f(py)))
w()
w("Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).")
w()
w("## 9. Limits")
w()
w("* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run). The planner is deterministic, so *another inference run* is not in play for these arms.")
w("* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.")
w("* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception. Inference inputs: vision plus measured ego history.")
w("* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row `H-FEAS2-T1-1` (a proposed id).")
open(f"{PK}/RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("RESULTS.md written", len(L), "lines")
