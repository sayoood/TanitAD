#!/usr/bin/env python3
"""make_results_kappa.py -- RESULTS.md of H-KAPPA-T1-1 Part 2, generated from raw/KAPPA_T1.json (no hand-typed number).  python code/make_results_kappa.py"""
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
J = json.load(open(f"{PK}/raw/KAPPA_T1.json", encoding="utf-8"))
ARMS = (("CTRL", "DAG-OACC"), ("SOUP", "SOUP"))
KEYS = {"CTRL": ("CTRL_s0", "CTRL_s1"), "SOUP": ("SOUP_s0", "SOUP_s1")}
NEXT = {
    "DOMINANT": "re-read on a fresh seed or window set before adoption; the training lever is the envelope as the decoder's output layer (the PI's go).",
    "MOVED": "name the rows still separated-worse vs P10 per seed; next = the gate's next-ranked class lever composed on top: (b) the accel-bounded longitudinal decode for the DIP class (`project_feasible(kappa_max=inf, a_max=4.0, mu=None, clamp_entry=True, v0 = executed speed at the tick)` after S and the box), then (c) for REST.",
    "NO EFFECT": "(c), the lateral residual shrink `P + diag(1, alpha)(S(plan) - P)` with alpha chosen on TRAIN windows only (or its named blocker: the tool rolls out eval-141 only; a TRAIN grid, join and cache are needed, and whether DAgger's train-window rollout inputs exist on disk is UNVERIFIED).",
    "HARMS": "the kappa-only projection (`a_max` = inf) as a further flag value of the same tool, to attribute the harm to the accel clamp or the curvature clamp.",
}
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


w("# RESULTS: H-KAPPA-T1-1 Part 2, X = S + curvature-bounded decode (box), Y = S rerun, DAG-OACC and SOUP at s0 / s1")
w()
w("Card `PREREG_FROZEN.md` sha256 `%s` (frozen 2026-11-02T15:44:58+0100). Reader `code/kappa_t1_read.py` under `READER_FIX_FLOOR.md` (an extractor-guard fix; not an amendment). Read clock %s. Raw `raw/KAPPA_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours)**." % (J["prereg_sha256"], J["run_clock"]))
w("Estimator of every interval below: **%s**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals do not answer *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*." % J["estimator"])
w()
w("## 0. Outcome (the card's rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT)")
w()
w("| arm | outcome | seed 0: D1 / D2 / D3 | seed 1: D1 / D2 / D3 | seed 0: MOVED / HARMS | seed 1: MOVED / HARMS |")
w("|---|---|---|---|---|---|")
for a, nm in ARMS:
    o = J["outcome"][nm]
    c = []
    for k in KEYS[a]:
        A, XY = J["arms"][k]["read_a"], J["arms"][k]["read_xy"]
        c.append("%s / %s / %s" % (A["D1"], A["D2"], A["D3"]))
    h = ["%s / %s" % (J["arms"][k]["read_xy"]["MOVED_seed"], J["arms"][k]["read_xy"]["HARMS_seed"]) for k in KEYS[a]]
    w("| %s | **%s** | %s | %s | %s | %s |" % (nm, o["outcome"], c[0], c[1], h[0], h[1]))
w()
w("Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): " + ", ".join("%s %s" % (nm, J["outcome"][nm]["moved_and_harms"]) for _, nm in ARMS) + ".")
w()
w("### What triggers HARMS and what blocks DOMINANT, per arm and seed (generated from the rows)")
w()
for a, nm in ARMS:
    for k in KEYS[a]:
        A, XY, R = J["arms"][k]["read_a"], J["arms"][k]["read_xy"], J["arms"][k]["rows"]
        trig = (["ADE6 X - Y %+.4f > guard %+.4f" % (XY["ade6_delta"], SR.ADE_GUARD_M)] if not XY["ade6_within_guard"] else []) + ["%s X - Y %+.4f SW" % (q, R["X_minus_Y"][q]["delta"]) for q in XY["non_crawl_separated_worse"]]
        sbc = [q for q, v in XY["crawl_separated_better"].items() if v]
        nsb = [q for q, v in XY["crawl_separated_better"].items() if not v]
        w("* **%s %s**: HARMS triggers: %s. DOMINANT blocked by: %s (D1 %s, D3 %s). Crawl rows X - Y separated-better: %s; not separated-better: %s." % (nm, k[-2:], "; ".join(trig) or "none",
          "D2 rows " + ", ".join(A["separated_worse_rows"]) if not A["D2"] else "nothing on D1-D3", A["D1"], A["D3"], ", ".join(sbc) or "none", ", ".join(nsb) or "none"))
w()
w("## 1. Known answers and controls (all computed before any row)")
w()
for cname, c in J["controls"].items():
    w("* **%s**: %s" % (cname, "PASS" if c["PASS"] else "FAIL"))
w()
w("**KA1** (Y equals `out_m16/SMOOTH_*`, bit for bit; tolerance 1e-4 m):")
w()
w("| run | dump bit-identical | dump max abs dev (m) | ticks bit-identical | ticks max abs dev |")
w("|---|---|---|---|---|")
for k, v in J["controls"]["KA1_Y_reproduces_out_m16_S"]["runs"].items():
    w("| %s | %s | %.3e | %s | %.3e |" % (k, v["dump_bit_identical"], v["dump_max_abs_dev_m"], v["ticks_bit_identical"], v["ticks_max_abs_dev"]))
w()
w("**KA2** (X's issued `plan` = box(P + SG(plan_raw - P)); tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):")
w()
w("| run | plan_s vs P+SG (m) | plan vs box(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | box moved the issued plan by up to (m) |")
w("|---|---|---|---|---|---|")
for k, v in J["controls"]["KA2_issued_plan_is_box_of_S"]["runs"].items():
    w("| %s | %.3e | %.3e | %.3e | %.3e | %.3f |" % (k, v["plan_s_vs_P_plus_SG_max_abs_m"], v["plan_vs_box_of_plan_s_max_abs_m"], v["tick0_plan_raw_vs_Y_max_abs_m"], v["floor_max_abs_dev_m"], v["box_changed_plan_max_abs_m"]))
w()
w("**KF** (READER_FIX_FLOOR.md: the floor / window check against the fresh Y; gross guard %.1f m against the banked CTRL):" % J["controls"]["KF_floor_window_check_vs_fresh_Y"]["gross_guard_m"])
w()
w("| run | windows X = fresh Y | windows fresh Y = banked CTRL | floor X vs fresh Y (m) | floor fresh Y vs banked CTRL max / mean (m) | floor X vs banked CTRL max (m) |")
w("|---|---|---|---|---|---|")
for k, v in J["controls"]["KF_floor_window_check_vs_fresh_Y"]["runs"].items():
    w("| %s | %s | %s | %.3e | %.3e / %.3e | %.3e |" % (k, v["windows_X_equal_fresh_Y"], v["windows_fresh_Y_equal_banked_CTRL"], v["floor_X_vs_fresh_Y_max_abs_m"], v["floor_fresh_Y_vs_banked_CTRL_max_abs_m"], v["floor_fresh_Y_vs_banked_CTRL_mean_abs_m"], v["floor_X_vs_banked_CTRL_max_abs_m"]))
w()
k3, kb1 = J["controls"]["K3_Y_banked_minus_P10_reproduces_SOUP_READ"], J["controls"]["KB1_P10_side_equals_landed_banked_b"]
w("**K3**: Y banked - P10 from the landed arrays reproduces the pinned `SOUP_READ.json` on %d / %d rows (tolerance %.0e). **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays on the non-S-POS rows: %d / %d runs pass." % (k3["n_pass"], k3["n_rows"], SR.TOL_ROW, sum(v["PASS"] for v in kb1["runs"].values()), len(kb1["runs"])))
w()
w("The last KA2 column is the maximum over every tick and knot of |plan - plan_s| (the box's displacement of the issued plan); the gate's 32-window tool smoke read a maximum of 10.2 m.")
w()
w("**KA3** (pytest on the patched tool, off returns the same object, round trip, planted plan inside the box, deleting the call turns RED): from `logs/pytest_feasible_smooth.log` and `logs/pytest_existing_t1.log`, run when the tool was built (HANDOFF.md); not re-run here.")
w()
w("## 2. Read A against P10 and X - Y, per arm and seed")
w()
for a, nm in ARMS:
    for k in KEYS[a]:
        A, XY, R = J["arms"][k]["read_a"], J["arms"][k]["read_xy"], J["arms"][k]["rows"]
        ap, ax, ay = R["X_minus_P10"]["ADE6"], R["X_minus_Y"]["ADE6"], R["Y_minus_P10"]["ADE6"]
        w("### %s %s" % (nm, k[-2:]))
        w()
        w("* **PRIMARY ADE6 (m)**: X - P10 %s (D1 bar point <= %+.4f and SB: **%s**); X - Y %s (guard point <= %+.4f: **%s**); Y - P10 %s." % (cell("ADE6", ap), SR.D1_POINT_M, A["D1"], cell("ADE6", ax), SR.ADE_GUARD_M, XY["ade6_within_guard"], cell("ADE6", ay)))
        w("* **D2** (no row of 19 separated-worse, X - P10): **%s**; separated-worse rows: %s." % (A["D2"], ", ".join(A["separated_worse_rows"]) or "none"))
        w("* **D3** (a separated-better row per family, X - P10): %s; separated-better rows: %s." % (A["D3_by_family"], ", ".join(A["separated_better_rows"]) or "none"))
        w("* **The four crawl rows, X - Y separated-better**: %s. **Non-crawl rows separated-worse on X - Y**: %s. Rows separated-worse on X - Y, all 19: %d." % (XY["crawl_separated_better"], ", ".join(XY["non_crawl_separated_worse"]) or "none", XY["n_separated_worse_all"]))
        w()
        w("| row | family | X - P10 | X - Y | Y - P10 |")
        w("|---|---|---|---|---|")
        for lab in GD.LABELS:
            w("| %s | %s | %s | %s | %s |" % (lab, SR.FAMILY[lab], cell(lab, R["X_minus_P10"][lab]), cell(lab, R["X_minus_Y"][lab]), cell(lab, R["Y_minus_P10"][lab])))
        w()
w("## 3. The prediction next to the result")
w()
w("Card, written before any number (ESTIMATED): gate lever (a) at 55 %; given (a), **DOMINANT 10 %, MOVED 40 %, NO EFFECT 30 %, HARMS 20 %**, and the stated expectation *the clamp caps the tail, not the ordinary steps: curvature MAE X - Y separated-better at both seeds; M1 and L2 move less*.")
w()
w("| arm | outcome read | curvature MAE X - Y separated-better s0 / s1 | share of the Y - P10 excess closed by X - Y: curvature MAE, L2, M1, yaw-rate (s0 / s1) |")
w("|---|---|---|---|")
for a, nm in ARMS:
    sbs, cl = [], []
    for k in KEYS[a]:
        R = J["arms"][k]["rows"]
        sbs.append(str(SR.better(GD, "curvature MAE", R["X_minus_Y"]["curvature MAE"])))
        parts = []
        for lab in ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE"):
            yp, xy = R["Y_minus_P10"][lab]["delta"], R["X_minus_Y"][lab]["delta"]
            parts.append("%s %s" % (lab.split(" ")[0] if lab != "L2 curvature MAE (GT>=5.2)" else "L2", ("%.0f%%" % (-100.0 * xy / yp)) if yp > 0 else "n/a (Y - P10 <= 0)"))
        cl.append(", ".join(parts))
    w("| %s | **%s** | %s | s0: %s; s1: %s |" % (nm, J["outcome"][nm]["outcome"], " / ".join(sbs), cl[0], cl[1]))
w()
w("## 4. The card's committed next arm for the outcome (Rule Zero)")
w()
for a, nm in ARMS:
    o = J["outcome"][nm]["outcome"]
    w("* **%s, %s**: %s" % (nm, o, NEXT[o]))
w()
w("Each next arm is a zero-training flag value of the same tool and needs its own frozen card before any rollout; none is launched by this read.")
w()
w("## 5. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)")
w()
for k in ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"):
    d = J["descriptive"][k]
    px, py = d["curvature_partition_X"], d["curvature_partition_Y"]
    w("* **%s**: X %s | Y %s" % (k, json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in px.items() if not isinstance(v, (list, dict))}), json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in py.items() if not isinstance(v, (list, dict))})))
w()
w("Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).")
w()
w("## 5b. Reading (HYPOTHESIS, not tested here; the committed next arm discriminates it)")
w()
w("* The box closes the curvature, L2 curvature and yaw-rate excess (section 3) and leaves M1 crawl share separated-worse, moving it the wrong way. The rows it harms are position and heading rows (heading MAE, S-POS, along-track, ADE6 at DAG-OACC s1): the projection re-integrates positions.")
w("* HYPOTHESIS: the acceleration clamp (`a_max` 4.0, which holds a plan's speed down when it launches above the entry speed; GATE_RESULT.md tool smoke) drives M1, along-track and ADE6; the curvature clamp drives curvature, yaw-rate, heading and S-POS. The kappa-only projection (`a_max` = inf) is the arm that separates them.")
w()
w("## 6. Limits")
w()
w("* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run).")
w("* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.")
w("* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception.")
w("* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row.")
open(f"{PK}/RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("RESULTS.md written", len(L), "lines")
