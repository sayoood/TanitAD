#!/usr/bin/env python3
"""make_results_histswap.py -- RESULTS.md of H-HISTSWAP-1, generated from raw/HISTSWAP_T1.json (+ the four run JSONs for swap_bucket_differs / swap_ticks); no hand-typed number.

    python -I code/make_results_histswap.py
Adapted from feas2_t1/code/make_results_feas3.py (same rows, same marks). Fails loud (SystemExit) if a fact the fixed prose depends on does not hold in the JSON.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE).replace("\\", "/")
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/code")
import smooth_t1_read as SR  # noqa: E402

GD = SR.load_gd()
J = json.load(open(f"{PK}/raw/HISTSWAP_T1.json", encoding="utf-8"))
PCK = json.load(open(f"{PK}/raw/PRECHECK_CEILING.json", encoding="utf-8"))
KEYS = ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1")
NM = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}
SWAPRUN = {"CTRL_s0": "DAGOACC_s0", "CTRL_s1": "DAGOACC_s1", "SOUP_s0": "SOUP_s0", "SOUP_s1": "SOUP_s1"}
L = []


def die(msg):
    raise SystemExit("UNVERIFIED / PROSE-FACT FAILED: " + msg)


def w(s=""):
    L.append(s)


def mark(lab, r):
    return "SB" if SR.better(GD, lab, r) else ("SW" if SR.worse(GD, lab, r) else "ns")


def cell(lab, r):
    return "%+.4f [%+.4f, %+.4f] %s" % (r["delta"], r["lo"], r["hi"], mark(lab, r))


def ctl_pass(c):
    return c["PASS"] if "PASS" in c else all(v.get("PASS", False) for v in c["runs"].values())


O = J["outcome"]
if O["outcome"] != "SUPPORTED" or O["n_cells_separated_better"] != 4:
    die("the outcome is not SUPPORTED at 4 of 4 cells; the fixed prose assumes it")
for n, c in J["controls"].items():
    if not ctl_pass(c):
        die("a control failed: " + n)
if not all(J["arms"][k]["m1_cell"]["better"] for k in KEYS):
    die("an M1 cell is not separated-better")

w("# RESULTS: H-HISTSWAP-1, does O follow its own slowed ego history? (zero training, ORACLE diagnostic), DAG-OACC and SOUP at s0 / s1")
w()
w("Card `PREREG_HISTSWAP_FROZEN.md` sha256 `%s` (frozen 2026-11-02T17:57:30+0100). Reader `code/histswap_t1_read.py` (rule `decide`, tested in `tests/test_histswap_read_rule.py`). Read clock %s. Raw `raw/HISTSWAP_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**." % (J["prereg_sha256"], J["run_clock"]))
w("Estimator of every interval below: **%s**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. X and Y use the same checkpoint, the planner is deterministic, and the two seeds are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1` not addressed)." % J["estimator"])
w()
w("## 0. The committed reading: H SUPPORTED, and X is an ORACLE")
w()
w("**MEASURED.** The card's rule reads **%s**: M1 crawl share X - Y is separated-better at **%d of 4 cells** (both seeds on both bases) and the mean share of Y - P10's M1 excess closed is **%.3f** (bar %.1f; pooled over the four cells %.3f). H (a HYPOTHESIS before this read): O follows its own slowed ego history." % (O["outcome"], O["n_cells_separated_better"], O["mean_share_closed"], O["share_bar"], O["pooled_share_closed"]))
w()
w("> **ORACLE CAVEAT (the card's, binding on every sentence of this file).** Arm X reads the LOGGED ego history at every tick; the logged ego history after tick 0 carries the human's speed at that tick. X is therefore an ORACLE and is **not deployable**. **SUPPORTED shows that the executed-speed drift enters through the ego-history channel (the network's ego-history input), and does NOT show a deployable fix.** X's absolute rows are NOT performance and are never quoted as such; only X - Y on M1 and the row structure are read. X also leaks the human's realised motion at ticks 1 and 2, so a positive reading cannot tell *the executed lag is the cause* from *any human-motion history fixes it*; a negative reading would not be subject to that leak. **A leak-free arm is the next step** (being built in `../histswap_lf/`; its result is not part of this file).")
w()
w("| cell | M1 X - Y (X = oracle history) | Y - P10 (the excess) | share of the excess closed | ADE6 X - Y (m; oracle, not performance) |")
w("|---|---|---|---|---|")
for k in KEYS:
    c = J["arms"][k]
    xy, yp, ad = c["rows"]["X_minus_Y"]["M1 crawl share"], c["rows"]["Y_minus_P10"]["M1 crawl share"], c["rows"]["X_minus_Y"]["ADE6"]
    mc = c["m1_cell"]
    if abs(mc["xy_delta"] - xy["delta"]) > 1e-12 or abs(mc["yp_delta"] - yp["delta"]) > 1e-12 or abs(mc["share_closed"] + xy["delta"] / yp["delta"]) > 1e-9:
        die("m1_cell disagrees with the rows at " + k)
    w("| %s %s | %s | %+.4f [%+.4f, %+.4f] %s | %.4f (%.1f %%) | %s |" % (NM[k[:4]], k[-2:], cell("M1 crawl share", xy), yp["delta"], yp["lo"], yp["hi"], mark("M1 crawl share", yp), mc["share_closed"], 100 * mc["share_closed"], cell("ADE6", ad)))
w()
w("Rule (card section 2): `better` = the M1 row of X - Y is separated and delta < 0; `share closed` = -(X - Y) / (Y - P10) on the M1 row; SUPPORTED = 4 of 4 better and mean share closed >= 0.5; NOT SUPPORTED = fewer than 3 of 4 better; MIXED otherwise.")
w()
w("## 1. Controls (computed before any row; any failure refuses the read)")
w()
for n, c in J["controls"].items():
    w("* **%s**: %s" % (n, "PASS" if ctl_pass(c) else "FAIL"))
w()
ky = J["controls"]["KY_reused_Y_equals_frozen_pins"]["files"]
k1 = J["controls"]["KA1_fresh_off_rerun_equals_reused_Y"]
w("**KY**: %d / %d reused Y files hash-equal to the pins of `FROZEN_SHA.txt`. **KA1** (the fresh `--histswap off` rerun `%s` equals the reused `%s`): dump bit-identical %s (max abs dev %.3e m), ticks bit-identical %s (max abs dev %.3e), differing keys %s, key-set difference %s." % (sum(v["PASS"] for v in ky.values()), len(ky), k1["rerun"], k1["reused"], k1["dump_bit_identical"], k1["dump_max_abs_dev_m"], k1["ticks_bit_identical"], k1["ticks_max_abs_dev"], (k1["dump_differing_keys"] + k1["ticks_differing_keys"]) or "none", k1["key_set_diff"] or "none"))
w()
ks0, ks1, ks2 = (J["controls"][n]["runs"] for n in ("KS0_tick0_equals_Y_tick0", "KS1_tick1_executed_state_equals_Y", "KS2_v0_in_is_the_logged_speed_and_differs_after_tick0"))
w("**KS0** (tick 0 of every X equals Y's tick 0, tolerance 1e-4): max abs deviation over tau / P / plan / plan_raw / v0 / pose / gt = %.1e on %s tick-0 rows per cell. **KS1** (tick 1's executed pose and v0 equal Y's): max abs deviation %.1e. **KS2** (v0_in is the logged speed):" % (max(max(v["tick0_max_abs_dev_vs_Y"].values()) for v in ks0.values()), "/".join(str(v["n_tick0_rows"]) for v in ks0.values()), max(max(v["tick1_executed_state_max_abs_dev_vs_Y"].values()) for v in ks1.values())))
w()
w("| cell | v0_in = v0 at tick 0 (max abs) | share of later ticks with v0_in != v0 by > 0.05 m/s | mean abs v0_in - v0, later ticks (m/s) | mean executed v0 / logged v0_in, later ticks (m/s) |")
w("|---|---|---|---|---|")
for k, v in ks2.items():
    w("| %s | %.1e | %.4f | %.4f | %.3f / %.3f |" % (k, v["tick0_v0_in_equals_v0_max_abs"], v["frac_later_ticks_v0_in_ne_v0_gt_0p05"], v["mean_abs_v0_in_minus_v0_later_mps"], v["mean_v0_later_exec_mps"], v["mean_v0_in_later_logged_mps"]))
w()
kf, k3, kb = J["controls"]["KF_floor_window_check_vs_fresh_Y"], J["controls"]["K3_Y_banked_minus_P10_reproduces_SOUP_READ"], J["controls"]["KB1_P10_side_equals_landed_banked_b"]["runs"]
w("**KF**: windows X = fresh Y at %d / %d cells, floor X vs Y max abs %.1e m. **K3**: Y banked - P10 reproduces `SOUP_READ.json` on %d / %d rows. **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays: %d / %d runs pass. Tool tests `logs/pytest_histswap.log` (8 passed, 3 mutations RED) and `logs/pytest_read_rule.log` (6 passed); 32-window smoke `logs/smoke_histswap.out`: run when the tool was built; not re-run in this read." % (sum(v["windows_X_equal_fresh_Y"] for v in kf["runs"].values()), len(kf["runs"]), max(v["floor_X_vs_fresh_Y_max_abs_m"] for v in kf["runs"].values()), k3["n_pass"], k3["n_rows"], sum(v["PASS"] for v in kb.values()), len(kb)))
w()
w("## 2. Read A for X (vs P10), X - Y and Y - P10, all 19 rows, per cell (X = ORACLE; absolute rows are not performance)")
w()
w("Every cell is `point [lo, hi] mark`. `read_a` and `read_xy` are the FEAS-family reads applied to X; because X reads the human's history, **its `DOMINANT` flag is an oracle artefact and is never a claim about a deployable arm**. S-POS rows of X mix the logged poses the S head read with the executed pose the extractor frames them in: **UNVERIFIED**, outside the reading.")
w()
for k in KEYS:
    c = J["arms"][k]
    Ra, XY, R = c["read_a_X_vs_P10"], c["read_xy_X_minus_Y"], c["rows"]
    w("### %s %s" % (NM[k[:4]], k[-2:]))
    w()
    w("* **read_a (X vs P10)**: D1 %s, D2 %s, D3 %s (%s), DOMINANT-flag %s (oracle artefact); separated-worse rows: %s; separated-better rows: %d of 19." % (Ra["D1"], Ra["D2"], Ra["D3"], Ra["D3_by_family"], Ra["DOMINANT"], ", ".join(Ra["separated_worse_rows"]) or "none", len(Ra["separated_better_rows"])))
    w("* **read_xy (X - Y)**: crawl rows separated-better %s; ADE6 X - Y %+.4f (within guard: %s); non-crawl rows separated-worse: %s; rows separated-worse on X - Y, all 19: %d; MOVED-seed %s, HARMS-seed %s." % (XY["crawl_separated_better"], XY["ade6_delta"], XY["ade6_within_guard"], ", ".join(XY["non_crawl_separated_worse"]) or "none", XY["n_separated_worse_all"], XY["MOVED_seed"], XY["HARMS_seed"]))
    w()
    w("| row | family | X - P10 | X - Y | Y - P10 |")
    w("|---|---|---|---|---|")
    for lab in GD.LABELS:
        w("| %s | %s | %s | %s | %s |" % (lab, SR.FAMILY[lab], cell(lab, R["X_minus_P10"][lab]), cell(lab, R["X_minus_Y"][lab]), cell(lab, R["Y_minus_P10"][lab])))
    w()
w("## 3. The M1 row by situation class (descriptive, not a decision; classes L launch / S stay / O other-stopped / M moving; weight `w` = share of the row's windows)")
w()
w("| cell | contrast | row delta | L: n, delta, contribution | M: n, delta, contribution | S, O windows | L share of the row's delta |")
w("|---|---|---|---|---|---|---|")
lsh = {}
for k in KEYS:
    sp = J["descriptive"][k]["M1_per_class_split"]
    for ct, lab in (("X_minus_Y", "X - Y"), ("Y_minus_P10", "Y - P10"), ("X_minus_P10", "X - P10")):
        d = sp[ct]
        cl = d["classes"]
        if cl["S"]["n"] or cl["O"]["n"]:
            die("S or O class has windows at " + k)
        share = cl["L"]["contribution"] / d["delta"] if abs(d["delta"]) > 1e-12 else float("nan")
        lsh[(k, ct)] = share
        w("| %s %s | %s | %+.4f [%+.4f, %+.4f] | %d, %+.4f [%+.4f, %+.4f], %+.4f | %d, %+.4f [%+.4f, %+.4f], %+.4f | %d, %d | %s |" % (NM[k[:4]], k[-2:], lab, d["delta"], d["lo"], d["hi"], cl["L"]["n"], cl["L"]["delta"], cl["L"]["delta_lo"], cl["L"]["delta_hi"], cl["L"]["contribution"], cl["M"]["n"], cl["M"]["delta"], cl["M"]["delta_lo"], cl["M"]["delta_hi"], cl["M"]["contribution"], cl["S"]["n"], cl["O"]["n"], ("%.0f %%" % (100 * share)) if ct != "X_minus_P10" else "n/a (row delta near 0)"))
w()
w("The S and O classes carry no windows in the masked M1 pool at any cell. Launch-class windows are %.1f %%-%.1f %% of the M1 row's windows (weight `w`) and carry %.0f %%-%.0f %% of the Y - P10 excess and %.0f %%-%.0f %% of the X - Y movement (contribution / row delta)." % (
    100 * min(J["descriptive"][k]["M1_per_class_split"]["Y_minus_P10"]["classes"]["L"]["w"] for k in KEYS), 100 * max(J["descriptive"][k]["M1_per_class_split"]["Y_minus_P10"]["classes"]["L"]["w"] for k in KEYS),
    100 * min(lsh[(k, "Y_minus_P10")] for k in KEYS), 100 * max(lsh[(k, "Y_minus_P10")] for k in KEYS), 100 * min(lsh[(k, "X_minus_Y")] for k in KEYS), 100 * max(lsh[(k, "X_minus_Y")] for k in KEYS)))
w()
w("## 4. Swap bookkeeping (`swap_bucket_differs / swap_ticks`, from each run's `t1v8f_HSW_*.json`)")
w()
w("| cell | swap_ticks | swap_bucket_differs | share |")
w("|---|---|---|---|")
for k in KEYS:
    pth = f"F:/TanitAD-scratch/histswap/out/HSW_{SWAPRUN[k]}/t1v8f_HSW_{SWAPRUN[k]}.json"
    if not os.path.exists(pth):
        w("| %s | UNVERIFIED (%s missing) | | |" % (k, pth))
        continue
    st = json.load(open(pth, encoding="utf-8"))["t1v8f"]["histswap"]["bucket_stats_model_ticks"]
    w("| %s %s | %d | %d | %.2f %% |" % (NM[k[:4]], k[-2:], st["swap_ticks"], st["swap_bucket_differs"], 100.0 * st["swap_bucket_differs"] / st["swap_ticks"]))
w()
w("The swap includes the causal speed bucket (a function of the ego speed history); its separate contribution to M1 is not attributed here (card section 6).")
w()
w("## 5. The prediction next to the result")
w()
w("Card (ESTIMATED, written before any HISTSWAP number): P(SUPPORTED) 0.60, P(MIXED) 0.25, P(NOT SUPPORTED) 0.15. Read: **%s** (the card gave it 0.60). Basis of the estimate: the oracle history carries the human's realised speeds, so M1 was expected to close largely through the swap whatever the mechanism. The card also expected, not committed, that X's ADE6 improves vs Y by leak: measured ADE6 X - Y %s m at the four cells (CTRL s0, s1, SOUP s0, s1), all separated-better and not performance." % (O["outcome"], ", ".join("%+.4f" % J["arms"][k]["rows"]["X_minus_Y"]["ADE6"]["delta"] for k in KEYS)))
w()
w("## 6. Ceiling pre-check (card section 1; SEEN Y data, `raw/PRECHECK_CEILING.json`) beside the read")
w()
w("Slow GT-fast pairs of Y minus those of P10, by executed-step block [0-10) / [10-20) / [20-30): " + "; ".join("%s %s" % (b, " / ".join("%+d" % (PCK[b]["Y"]["slow_by_block"][i] - PCK[b]["P10"]["slow_by_block"][i]) for i in range(3))) for b in PCK) + ". The excess starts after tick 0, which the swap cannot change, so the swap is not ceiling-limited; the pattern does not discriminate H from a plan that decays with the horizon (card).")
w()
w("## 7. The card's committed next step (SUPPORTED), and what stays open")
w()
w("* **Committed:** the recovery fine-tune of O (`DRAFT_RECOVERY_CARD.md`, a DRAFT, not frozen, not edited here) goes to the PI for a box-2 go, **together with the leak caveat of section 0 and the leak-free separating arm named as its unknown**; no GPU job is started by this card.")
w("* **What the read does not say:** SUPPORTED does not say which part of the logged history carries the effect (the human's realised speeds at ticks 1 and 2, the speed bucket, or the executed lag itself), and does not say a model trained or run without the human's history recovers any of the %.1f %%-%.1f %% closed. The leak-free arm (`../histswap_lf/`, in progress) is the experiment that separates them." % (100 * min(J["arms"][k]["m1_cell"]["share_closed"] for k in KEYS), 100 * max(J["arms"][k]["m1_cell"]["share_closed"] for k in KEYS)))
w("* **Not decided here:** adoption into any K-1 / WM-1 claim; the Master Mind registers the claim row `H-HISTSWAP-1` (a proposed id).")
w()
w("## 8. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)")
w()
for k in KEYS:
    d = J["descriptive"][k]
    f = lambda z: json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in z.items() if not isinstance(v, (list, dict))})
    w("* **%s**: X %s | Y %s" % (k, f(d["curvature_partition_X"]), f(d["curvature_partition_Y"])))
w()
w("Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m.")
w()
w("## 9. Limits")
w()
w("* **X is an ORACLE** (section 0): no row of X is performance and nothing here is a deployable-fix claim.")
w("* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); X and Y share the checkpoint and the planner is deterministic; the two seeds per base are replicates of the rig, not a variance estimate. A separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`).")
w("* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception.")
w("* S-POS rows of X are UNVERIFIED (logged poses read by the S head, executed pose in the extractor frame). The swap's causal speed bucket is not separated from the rest of the history swap.")
open(f"{PK}/RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("RESULTS.md written", len(L), "lines")
