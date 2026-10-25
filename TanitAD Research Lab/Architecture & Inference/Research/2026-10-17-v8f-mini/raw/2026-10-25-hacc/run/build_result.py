#!/usr/bin/env python3
"""build_result.py -- generates RESULT_V8F_HACC.md, accel_profile.json, weights_HACC_<seed>.json and FILES.txt from the JSONs written by v8f_hacc.py. No number is typed by hand.

  python build_result.py [OUT_DIR]        (v8f_hacc.py auto calls main(out) as its last step)
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SEEDS = ("M2b_s0", "M2b_s1")
NAMES = [("fw_lon_speed_mae", "LONGITUDINAL", "speed MAE (m/s)", 4), ("fw_lon_accel_mae", "LONGITUDINAL", "accel MAE (m/s^2)", 4), ("fw_lon_along_mae", "LONGITUDINAL", "along MAE (m)", 4),
         ("fw_lat_cross_mae", "LATERAL", "cross-track MAE (m)", 4), ("fw_lat_heading_mae_deg", "LATERAL", "heading MAE (deg)", 4),
         ("fw_lat_yaw_rate_mae_degps", "LATERAL", "yaw-rate MAE (deg/s)", 4), ("fw_lat_curvature_mae_1pm", "LATERAL", "curvature MAE (1/m)", 5),
         ("fw_tac_lat_acc", "TACTICAL (plan-derived)", "lateral decision agreement (higher better)", 4),
         ("fw_tac_lon_acc", "TACTICAL (plan-derived)", "longitudinal decision agreement (higher better)", 4)]
CK = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any")
ADE = [f"ade{h}" for h in range(1, 7)]
FDE = [f"fde{h}" for h in range(1, 7)]


def fr(r, nd=4):
    if not r or "mean" not in r:
        return "n/a"
    tag = " *" if r.get("separated") else ""
    return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{tag}"


def fm(r, nd=4):
    if not r or "mean" not in r:
        return "n/a"
    return f"{r['mean']:.{nd}f}"


def tf(x):
    return "PASS" if x else "FAIL"


def blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(out=None):
    OUT = (out or HERE).replace("\\", "/")
    R = json.load(open(f"{OUT}/READ.json"))
    SC = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
    FT = {s: json.load(open(f"{OUT}/fit_{s}.json")) for s in SEEDS}
    DK = json.load(open(f"{OUT}/dk_read.json"))
    X1 = json.load(open(f"{OUT}/control_X1.json"))
    MUT = json.load(open(f"{HERE}/mutation_results.json")) if os.path.exists(f"{HERE}/mutation_results.json") else None
    PYT = open(f"{HERE}/pytest_final.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-1] if os.path.exists(f"{HERE}/pytest_final.log") else "(pytest_final.log absent)"
    F = R["F_exact_m"]
    L = []
    w = L.append

    # ---------------------------------------------------------------------------------------------- weights + accel profile files (banked beside)
    for s in SEEDS:
        ws = FT[s]["w_star"]
        json.dump({"seed": s, "arm": "HACC", "lambda0": FT[s]["lambda0"], "m_star": FT[s]["m_star"], "lam_star": FT[s]["lam_star"], "rule_branch": FT[s]["rule_branch"],
                   "w_along_by_step": [r[0] for r in ws], "w_cross_by_step": [r[1] for r in ws], "tier": SC[s]["tier"], "fit_windows_sha256": FT[s]["fit_windows_sha256"],
                   "note": "H = P + w (.) (tau - P) at inference; w is a fixed function of the step index (30 x 2 per model); NO isotonic step"},
                  open(f"{OUT}/weights_HACC_{s}.json", "w"), indent=1)
    json.dump({"label": "per-step accel-error profile of HACC / H / P10 / B (accel_profile.py's instrument; control 7 ties it to the banked fw_lon_accel_mae)", "tier": SC["M2b_s0"]["tier"],
               "per_seed": {s: SC[s]["accel_profile"] for s in SEEDS}}, open(f"{OUT}/accel_profile.json", "w"), indent=1)

    w("# RESULT_V8F_HACC: PREREG_V8F_HACC executed exactly as written (2026-10-25)")
    w("")
    w(f"Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number is **MEASURED (ours)** from the artifact paths named; tier **{R['tier']}** (the CEILING is T0); "
      f"estimator **{SC['M2b_s0']['estimator']}** Eval-141: 2,399 windows. `OUT` = `{OUT}`.")
    w("")
    # ---------------------------------------------------------------------------------------------- 0 outcome
    w("## 0. Outcome (the PREREG outcome table, applied in its order)")
    w("")
    w("Artifacts: `READ.json` (this table), `score_<seed>.json`, `dk_read.json`, `fit_<seed>.json`, `control_X1.json`; per-window plans in `score_<seed>_perwindow.npz`.")
    w("")
    w(f"**{R['outcome']}**  (PRIMARY-DOM at both seeds: {R['DOM_both']}; PRIMARY-KEEP at both seeds: {R['KEEP_both']}; HACC - H accel AND curvature separated-negative at both seeds: {R['SMOOTHER_both']}; "
      f"HACC - H accel separated-negative at either seed: {R['ACCEL_NEG_either']}; all controls pass: {R['controls_all_pass']}).")
    w("")
    w("| seed | lambda* (m*) | PRIMARY-DOM | rows that fail it | PRIMARY-KEEP (ADE6 HACC - H) | HACC - H accel MAE | HACC - H curvature MAE |")
    w("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        p = R["per_seed"][s]
        fails = [f"{nm} ({r['need']}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}])" for nm, r in p["PRIMARY_DOM"]["rows"].items()
                 if (r["need"] == "separated-better" and not r["better"]) or (r["need"] == "not separated-worse" and r["worse"])]
        k = p["PRIMARY_KEEP"]
        sm = p["HACC_minus_H_accel_curvature"]
        w(f"| {s} | {p['lam_star']:.5g} (m*={p['m_star']}) | {tf(p['PRIMARY_DOM']['pass'])} | {'; '.join(fails) if fails else 'none'} | {tf(k['pass'])}: {k['delta']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}], "
          f"bar hi <= {k['bar_hi_le']:+.5f} | {fr(sm['accel'])} (sep-neg: {sm['accel_sep_neg']}) | {fr(sm['curvature'], 5)} (sep-neg: {sm['curv_sep_neg']}) |")
    w("")
    w(f"**Committed next step (PREREG, verbatim):** *\"{R['committed_next_step_verbatim']}\"*.")
    w("")
    w("**Rule Zero (a refutation is a waypoint).** " + {
        "DOMINATES P10": ("The dominance claim over the programme's kinematic floor is delivered at both seeds (T1-class open loop; not state of the art, and not closed loop). What blocks the committed next step: "
                          "(i) folding 30 x 2 buffers into `forward_infer` is a landing with tests under its own brief (not done here; nothing is committed or pushed); (ii) the T1-kin closed-loop read "
                          "(`D-V8F-T1KIN-1`) needs a PI go for the pooled eval cache."),
        "SMOOTHER, NOT DOMINANT": ("HACC smooths H (accel and curvature separated-negative vs H at both seeds) but does not dominate P10 on the dominance row: the failing rows are named in the table above. "
                                   "The committed lever is OACC on box 2, queued after FLAT; box 2 and its GPU slot are outside this read and the PI's call. Zero-training levers left on this rig: see section 7 (the CEILING) -- "
                                   "if even the eval-GT-fitted w does not separate from P10 on the failing rows, the per-step weighting lever is exhausted rather than under-fitted."),
        "NO SMOOTHING": ("HACC - H accel MAE is not separated-negative at either seed: the roughness is not a weighting artefact. The committed lever is OACC (box 2, queued after FLAT), a GPU/spend decision for the PI."),
        "MIXED": ("Reported per seed; seeds before windows. The next experiment is a replicate seed on this rig (a separated CI from one seed is necessary, not sufficient: H-ESTIM-SEED-1)."),
    }[R["outcome"]])
    w("")
    # ---------------------------------------------------------------------------------------------- 1 what was run
    w("## 1. What was run")
    w("")
    w("* **Route.** HBLEND's / HSPEC's banked machinery, imported read-only (`v8f_hblend.py`: loaders, `apply_w`, `arm_pw`, `rows_for`, `single`, `fit_weights`, `pav_nondecreasing`; `v8f_hspec.py`: `load_world`, "
      "`cons_viol_arrays`, `dk_flags`, `DOM_BETTER`, `DOM_NOT_WORSE`; the eval's own `evaluate()`; the DK reader `v8f_dk_read.read_dk`; `taniteval.v8f_beta_eval.paired_row`). Fit set: HBLEND's 9,582 TRAIN windows "
      "(141 clips, k = 32); eval-141 (2,399 windows) never enters a fit or a selection. B's plan tau = the KIND run's banked `pred`; P = the checkpoint's own frozen P10; H = HBLEND's saved H.")
    w("* **Arm.** HACC = P + w (.) (tau - P), w in [0,1]^(30 x 2) a fixed function of the step index, from a bounded least squares with the position term plus lambda x the second-difference term "
      "(origin prepended, 29 cells), `scipy.optimize.lsq_linear(method='bvls', bounds=(0,1), tol=1e-12)` on the literal stacked system (565,338 x 30 rows at lambda > 0), KKT-checked (relative projected gradient <= 1e-6), "
      "NO isotonic step. Inference stays vision-only: the weight is a function of the step index; P is the deployed v8F prior; no situation-classifier output is in the goal path.")
    w("* **Declared interpretations (fixed in the script's docstring BEFORE any HACC number; none changed after).** I1 axes (0 along / 1 cross); I2 one lambda per seed shared by both axes, lambda0 = S1/S2 pooled over windows, "
      "steps and axes at HBLEND's banked w; I3 the D2 term is D2 e with e_0 = 0 (29 rows per axis); I4 the solver and the KKT definition (HSPEC's); I5 the train metrics of the lambda rule are per-window `arm_pw` means over the "
      "windows where both arms are finite, point comparisons (`<= 0.98 x` for accel and curvature, `<` for ADE2 and ADE6); I6 outcome row 3 reads 'not separated-negative at either seed' as 'at neither'; I7 separated flags = "
      f"`paired_row`'s, distance keeping per HSPEC's `dk_flags`; I8 PRIMARY-KEEP bar = F/2 with the exact F = {F:.5f} m = +{F / 2:.5f} m (the printed +0.0604 variant is reported beside); I9 control thresholds; "
      "I10 the added machinery tie X1 (my dominance reader reproduces HSPEC's banked verdicts and every per-row flag); I11 tactical / strategic by construction.")
    w("")
    # ---------------------------------------------------------------------------------------------- 2 controls
    w("## 2. Controls (each read its known value BEFORE any number was read; a failure is rc 2 and no outcome)")
    w("")
    w("| control | bar | s0 | s1 |")
    w("|---|---|---|---|")
    for lab, bar, fn in (
            ("1. lambda = 0 + HBLEND's isotonic step vs HBLEND's banked w, max abs", "<= 1e-12", lambda s: f"{FT[s]['train_controls']['c1_lam0_plus_isotonic_vs_banked_w_maxabs']:.2e}"),
            ("2. lambda = 0 (no isotonic) vs the clipped closed form, max abs", "<= 1e-12", lambda s: f"{FT[s]['train_controls']['c2_lam0_vs_clipped_closed_form_maxabs']:.2e}"),
            ("3. w = 1: plan vs banked pred (m)", "<= 1e-4", lambda s: f"{SC[s]['controls']['c3_w1_plan_vs_banked_pred_maxabs_m']:.2e}"),
            ("3. w = 1: per-window ADE2 / ADE6 / FDE6 vs banked, max abs (m)", "<= 1e-4", lambda s: f"{SC[s]['controls']['c3_w1_ade2_perwindow_maxabs_vs_banked']:.1e} / {SC[s]['controls']['c3_w1_ade6_perwindow_maxabs_vs_banked']:.1e} / {SC[s]['controls']['c3_w1_fde6_perwindow_maxabs_vs_banked']:.1e}"),
            ("3. w = 0: plan vs the P10 file (m)", "<= 1e-4", lambda s: f"{SC[s]['controls']['c3_w0_plan_vs_P10_file_maxabs_m']:.2e}"),
            ("3. w = 0: ADE6 mean (m)", "2.4870 +- 1e-3", lambda s: f"{SC[s]['controls']['c3_w0_ade6_mean']:.5f}"),
            ("4. KKT residual, max over EVERY train fit (lambda = 0, the 6 grid fits, both regression families at every grid lambda)", "<= 1e-6", lambda s: f"{FT[s]['train_controls']['c4_kkt_max_over_all_train_fits']:.1e}"),
            ("4. KKT residual of the eval-ceiling fit (T0)", "<= 1e-6", lambda s: f"{max(SC[s]['ceiling_T0']['kkt']):.1e}"),
            ("5. regression GT := P10: max w, worst over the grid", "<= 1e-12", lambda s: f"{max(FT[s]['train_controls']['c5_regression_w_maxabs_by_m'].values()):.1e}"),
            ("5. regression: eval ADE6 minus the P10 file's (m)", "<= 1e-4 abs", lambda s: f"{SC[s]['controls']['c5_regression_ade6_minus_P10_file_mean_m']:+.1e}"),
            ("5. r permuted across training windows: mean w at lambda*", "<= 0.10", lambda s: f"{FT[s]['train_controls']['c5_permuted_r_mean_w_at_lam_star']:.3f}"),
            ("6. the lambda selection refuses an eval input (dict world / bare array / eval-split bundle)", "all True", lambda s: str(all(SC[s]["controls"]["c6_eval_input_refused"].values()))),
            ("7. accel instrument vs banked fw_lon_accel_mae, H / P10 (float32)", "<= 2e-6", lambda s: f"{SC[s]['controls']['c7_accel_instrument_vs_banked_maxabs']['H']:.1e} / {SC[s]['controls']['c7_accel_instrument_vs_banked_maxabs']['P10']:.1e}"),
            ("X1 (added, I10). dominance reader reproduces HSPEC's banked verdicts and per-row flags", f"0 of {X1['checked']} rows differ", lambda s: f"{len(X1['mismatches'])} differ"),
            ("all controls pass", "", lambda s: str(SC[s]["controls"]["all_pass"]))):
        w(f"| {lab} | {bar} | {fn('M2b_s0')} | {fn('M2b_s1')} |")
    w("")
    w(f"* **Unit tests** (`test_v8f_hacc.py`, synthetic data plus the two TRAIN-window machinery ties on the real banked arrays): `{PYT}`.")
    if MUT:
        mu = {k: v for k, v in MUT.items() if k.startswith("M")}
        w(f"* **Mutations** (`mutate_hacc.py`, `mutation_results.json`; baseline GREEN: {MUT['BASELINE_unmutated'].get('GREEN')}): {sum(bool(v.get('RED')) for v in mu.values())} of {len(mu)} mutants went RED by an assertion: "
          + "; ".join(f"{k} ({v['n_failed']} failed)" for k, v in mu.items()) + ".")
    w("")
    # ---------------------------------------------------------------------------------------------- 3 the fit
    w("## 3. The fit and the train-only lambda rule")
    w("")
    w("Arm HACC, tier T1-class open loop; every number here is on the 9,582 TRAIN windows (in-sample; the rule's only input); artifact `fit_<seed>.json`, weights in `weights_HACC_<seed>.json`.")
    w("")
    for s in SEEDS:
        f = FT[s]
        w(f"**{s}.**  lambda0 = {f['lambda0']:.5g} (position term / D2 term at HBLEND's banked w, pooled over axes; train).  **lambda\\* = {f['lam_star']:.5g} (m\\* = {f['m_star']})**, rule branch: *{f['rule_branch']}*.")
        w("")
        w("| m | lambda | train accel MAE | P10 | x P10 | train curvature MAE | P10 | x P10 | train ADE2 | P10 | train ADE6 | P10 | qualifies | w mean | KKT |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for g in f["grid"]:
            wm = sum(sum(r) for r in g["w"]) / 60.0
            w(f"| {g['m']} | {g['lam']:.5g} | {g['accel']:.4f} | {g['p_accel']:.4f} | {g['accel'] / g['p_accel']:.3f} | {g['curv']:.5f} | {g['p_curv']:.5f} | {g['curv'] / g['p_curv']:.3f} | "
              f"{g['ade2']:.4f} | {g['p_ade2']:.4f} | {g['ade6']:.4f} | {g['p_ade6']:.4f} | {g['qualifies']} | {wm:.3f} | {max(g['kkt']):.1e} |")
        w("")
        ws = f["w_star"]
        idx = [0, 4, 9, 10, 11, 12, 13, 14, 19, 24, 25, 28, 29]
        w(f"w\\* by step (t = 0.2 (i+1) s), along / cross: " + "; ".join(f"i={i + 1}: {ws[i][0]:.3f} / {ws[i][1]:.3f}" for i in idx)
          + f".  Mean along {sum(r[0] for r in ws) / 30:.3f}, cross {sum(r[1] for r in ws) / 30:.3f}; non-monotone steps (decreases > 1e-9): along {sum(ws[i + 1][0] < ws[i][0] - 1e-9 for i in range(29))}, "
            f"cross {sum(ws[i + 1][1] < ws[i][1] - 1e-9 for i in range(29))}. Banked in `weights_HACC_{s}.json`.")
        w("")
    # ---------------------------------------------------------------------------------------------- 4 dominance rows
    w("## 4. PRIMARY-DOM and PRIMARY-KEEP: the dominance row, HACC - P10 (paired; lower is better except the two tactical agreements; `*` = CI excludes 0)")
    w("")
    w("Artifacts: `score_<seed>.json` (paired.HACC_minus_P10, consistency.paired.HACC_minus_P10.viol_any), `dk_read.json` (pairs HACC_<s>__minus__P10), `READ.json` (per_seed.PRIMARY_DOM).")
    w("")
    w("| metric | requirement | s0: HACC - P10 [95 % CI] | s0 | s1: HACC - P10 [95 % CI] | s1 |")
    w("|---|---|---|---|---|---|")
    rows0, rows1 = R["per_seed"]["M2b_s0"]["PRIMARY_DOM"]["rows"], R["per_seed"]["M2b_s1"]["PRIMARY_DOM"]["rows"]

    def verdict(r):
        if r["need"] == "separated-better":
            return "better / OK" if r["better"] else ("worse / **FAIL**" if r["worse"] else "not sep. / **FAIL**")
        return "worse / **FAIL**" if r["worse"] else ("better / OK" if r["better"] else "not sep. / OK")

    for nm in rows0:
        nd = 5 if "curvature" in nm else 4
        w(f"| {nm} | {rows0[nm]['need']} | {fr(rows0[nm], nd)} | {verdict(rows0[nm])} | {fr(rows1[nm], nd)} | {verdict(rows1[nm])} |")
    w("")
    for s in SEEDS:
        p = R["per_seed"][s]
        k = p["PRIMARY_KEEP"]
        w(f"* PRIMARY-DOM {s}: **{tf(p['PRIMARY_DOM']['pass'])}** (separated-better rows ok: {p['PRIMARY_DOM']['all_better_ok']}, no separated-worse rows: {p['PRIMARY_DOM']['all_not_worse_ok']}). "
          f"PRIMARY-KEEP {s}: ADE6 HACC - H = {k['delta']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}], bar hi <= +{k['bar_hi_le']:.5f} -> **{tf(k['pass'])}** (printed +0.0604: {tf(k['hi_le_prereg_printed_0.0604'])}).")
    w("")
    # ---------------------------------------------------------------------------------------------- 5 the four families
    w("## 5. The four families and ADE: B / H / HACC / P10 and the paired reads (per-family, never pooled)")
    w("")
    w("Tier T1-class open loop; means are unpaired window means, the three right-hand columns are PAIRED episode-cluster bootstrap differences; artifacts `score_<seed>.json` (single, paired, consistency) and `dk_read.json`.")
    w("")
    for s in SEEDS:
        S_ = SC[s]
        sg = S_["single"]
        w(f"**{s}** (lower is better except the tactical rows; `*` = CI excludes 0)")
        w("")
        w("| family | metric | B | H | HACC | P10 | HACC - H | HACC - B | HACC - P10 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for key, fam, lab, nd in NAMES:
            w(f"| {fam} | {lab} | " + " | ".join(fm(sg[a][key], nd) for a in ("B", "H", "HACC", "P10")) + " | "
              + " | ".join(fr(S_["paired"][q].get(key), nd) for q in ("HACC_minus_H", "HACC_minus_B", "HACC_minus_P10")) + " |")
        for k in ADE + FDE[5:]:
            w(f"| ADE/FDE | {k.upper()} (m) | " + " | ".join(fm(sg[a][k], 4) for a in ("B", "H", "HACC", "P10")) + " | "
              + " | ".join(fr(S_["paired"][q].get(k), 4) for q in ("HACC_minus_H", "HACC_minus_B", "HACC_minus_P10")) + " |")
        for k in CK:
            rate = S_["consistency"]["rate"]
            w(f"| CONSISTENCY | {k} (share of windows violating; lower better) | " + " | ".join(f"{rate[a][k]:.4f}" if k in rate.get(a, {}) else "n/a" for a in ("B", "H", "HACC", "P10")) + " | "
              + " | ".join(fr(S_["consistency"]["paired"][q].get(k), 4) for q in ("HACC_minus_H", "HACC_minus_B", "HACC_minus_P10")) + " |")
        w("")
    # DK
    NB = __import__("re").search(r"B=(\d+)", SC["M2b_s0"]["estimator"]).group(1)
    w(f"**LONGITUDINAL distance keeping** (the programme's own reader `v8f_dk_read.read_dk`, K = 30, B = {NB}; headway / time-gap: higher = more margin; LEAD-VIOL: share of at-risk windows, lower better)")
    w("")
    w("| arm | headway min (m) | time-gap min (s) | LEAD-VIOL at-risk |")
    w("|---|---|---|---|")
    for nm, ab in DK["arms"].items():
        m = ab["metrics"]
        w(f"| {nm} | {m['headway_min_m'].get('mean')} | {m['time_gap_min_s'].get('mean')} | {ab['lead_viol'].get('share_of_at_risk')} |")
    w("")
    w("| pair | headway min (m) | time-gap min (s) | LEAD-VIOL at-risk |")
    w("|---|---|---|---|")
    for pn, pv in DK["pairs"].items():
        d = pv["dk"]
        lv = pv["lead_viol"]["at_risk"]

        def dkf(r):
            return "n/a" if not r or r.get("delta") is None else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' *' if r.get('separated') else ''}"
        w(f"| {pn.replace('__minus__', ' - ')} | {dkf(d['headway_min_m'])} | {dkf(d['time_gap_min_s'])} | {dkf(lv)} |")
    w("")
    # ---------------------------------------------------------------------------------------------- 6 accel profile
    w("## 6. The per-step accel profile (beside; `accel_profile.py`'s float32 instrument, tied to the banked metric by control 7)")
    w("")
    w("Tier T1-class open loop, eval-141, all 2,399 windows, unpaired cell means (descriptive, no bar); artifact `accel_profile.json` (= `score_<seed>.json` accel_profile).")
    w("")
    for s in SEEDS:
        ap = SC[s]["accel_profile"]
        w(f"**{s}**: mean |a_plan - a_gt| (m/s^2) per band; the gap columns are minus P10 (negative = better than P10).")
        w("")
        w("| band | HACC | H | P10 | B | HACC - P10 | H - P10 |")
        w("|---|---|---|---|---|---|---|")
        for bn, b in ap["bands"].items():
            w(f"| {bn} | {b['HACC']:.3f} | {b['H']:.3f} | {b['P10']:.3f} | {b['B']:.3f} | {b['HACC_minus_P10']:+.3f} | {b['H_minus_P10']:+.3f} |")
        am = ap["all_cells_mean"]
        w(f"| all 29 cells | {am['HACC']:.3f} | {am['H']:.3f} | {am['P10']:.3f} | {am['B']:.3f} | {am['HACC'] - am['P10']:+.3f} | {am['H'] - am['P10']:+.3f} |")
        w("")
        w("per cell (k: t_s HACC-P10 / H-P10): " + "; ".join(f"{k + 1}: {ap['minus_P10']['HACC'][k]:+.2f} / {ap['minus_P10']['H'][k]:+.2f}" for k in range(29)) + ".  Full arrays: `accel_profile.json`.")
        w("")
    # ---------------------------------------------------------------------------------------------- 7 ceiling
    w("## 7. CEILING (T0 diagnostic, never a result): the same objective at lambda*, fitted on the EVAL GT")
    w("")
    w("Tier **T0 diagnostic** (the fit reads the eval GT); paired episode-cluster bootstrap; artifact `score_<seed>.json` (ceiling_T0).")
    w("")
    w("| seed | metric | CEILING - P10 | CEILING - H | CEILING - HACC |")
    w("|---|---|---|---|---|")
    for s in SEEDS:
        c = SC[s]["ceiling_T0"]
        for key, lab, nd in (("ade2", "ADE2", 4), ("ade6", "ADE6", 4), ("fw_lon_accel_mae", "accel MAE", 4), ("fw_lat_curvature_mae_1pm", "curvature MAE", 5), ("fw_lat_yaw_rate_mae_degps", "yaw-rate MAE", 4)):
            w(f"| {s} | {lab} | {fr(c['vs_P10'][key], nd)} | {fr(c['vs_H'][key], nd)} | {fr(c['vs_HACC'][key], nd)} |")
    w("")
    w("The ceiling reads the eval GT, so it bounds what a per-step weight can do at this lambda; it is not a deployable arm and enters no row above.")
    w("")
    # ---------------------------------------------------------------------------------------------- 8 tactical / strategic
    w("## 8. TACTICAL and STRATEGIC families (identical to H by construction)")
    w("")
    for s in SEEDS:
        si = SC[s]["structural_identity"]
        w(f"* {s}: T's decision heads and S's checkpoint outputs are the banked B arrays; the eval's own `evaluate()` on the HACC plan returns a tactical block equal to the banked B's: **{si['tactical_block_T_heads_HACC_equals_banked_B']}**; "
          f"strategic block equal to the banked B's: **{si['strategic_block_HACC_equals_banked_B']}** (`None` = the evaluate() block carries no strategic entry, in which case S's `ck_pos_err` is carried unchanged from the banked dump).")
    w("* STRATEGIC is **not paused** for v8F (PI-V8F-2026-10-17; R72/R73): HACC - H is exactly 0 on it because the post-map touches the plan only. `v8f_paired_families.py` was not run: it reads eval `run` dumps and HACC produces none through `evaluate()`. "
      "The plan-derived tactical agreements (lateral / longitudinal decision) move with the plan and are in sections 4 and 5.")
    w("")
    w("## 9. Deliverable manifest")
    w("")
    w(f"Everything is under `{OUT}`. Nothing was committed, pushed or staged. The list with sizes and raw-bytes git blob sha1 is `FILES.txt`. The coordinator lands the files.")
    open(f"{OUT}/RESULT_V8F_HACC.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    if OUT == HERE:
        write_files_txt(OUT)
    return 0


def write_files_txt(OUT):
    skip_dirs = {"mutants", "plumb", "__pycache__"}
    skip_names = {"FILES.txt", "PREREG_V8F_HACC.md", "_time_probe.py", "_feas_probe.py", "_patch_tests.py"}
    rows = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for fn in sorted(files):
            if fn in skip_names or fn.endswith(".pyc"):
                continue
            p = os.path.join(root, fn).replace("\\", "/")
            rows.append((p, os.path.getsize(p), blob_sha1(p)))
    with open(f"{HERE}/FILES.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# path<TAB>bytes<TAB>git blob sha1 of the raw bytes (FILES.txt itself excluded; mutants/, plumb/ and the scratch probes excluded)\n")
        for p, n, h in rows:
            fh.write(f"{p}\t{n}\t{h}\n")


if __name__ == "__main__":
    if "--files-only" in sys.argv:
        write_files_txt(HERE)
        sys.exit(0)
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
