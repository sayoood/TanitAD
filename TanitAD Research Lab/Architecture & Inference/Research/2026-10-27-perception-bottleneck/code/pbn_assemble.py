"""pbn_assemble.py -- PERC-BOTTLENECK: RESULT.json + RESULT.md from the three raw JSONs, applying the SEALED prereg's committed
readings mechanically (no number is typed by hand; every table cell is read from raw/*.json).

    python -X utf8 pbn_assemble.py --pkg <package dir>     (reads <pkg>/raw/RESULT_Q{1,2,3}.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, nd=3, sign=False):
    if x is None:
        return "n/a"
    try:
        return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")
    except (TypeError, ValueError):
        return str(x)


def dk(k):
    return str(k).replace("|", " / ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    a = ap.parse_args()
    pkg = Path(a.pkg)
    raw = pkg / "raw"
    q1 = json.loads((raw / "RESULT_Q1.json").read_text(encoding="utf-8")) if (raw / "RESULT_Q1.json").exists() else None
    q2 = json.loads((raw / "RESULT_Q2.json").read_text(encoding="utf-8")) if (raw / "RESULT_Q2.json").exists() else None
    q3 = json.loads((raw / "RESULT_Q3.json").read_text(encoding="utf-8")) if (raw / "RESULT_Q3.json").exists() else None
    L = []
    w = L.append
    w("# PERC-BOTTLENECK — where is refcv8's map / box bottleneck? (RESULT, assembled from raw/*.json)\n")
    w("| | |\n|---|---|")
    w("| **Evidence class** | MEASURED (ours). Every number below is read by `code/pbn_assemble.py` from `raw/RESULT_Q1.json`, "
      "`raw/RESULT_Q2.json`, `raw/RESULT_Q3.json`. |")
    w("| **Tier** | **T0** perception read (frozen forward; no closed loop). Not a driving claim. |")
    w("| **Checkpoint** | refcv8 EMA-48,852 (`E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`; md5 in `raw/capture_*.json`). ONE training seed. |")
    w("| **Prereg** | `PREREG_PERC_BOTTLENECK.md`, sealed sha256 `5489cf56...` BEFORE any probe was fit; `pbn_probe.py` refuses to score on a seal mismatch. |")
    w("| **Estimator** | paired episode-cluster bootstrap over eval clips, 1,000 draws, seed 0, 95 % percentile — answers *another draw of EPISODES*; "
      "probe seeds 0/1 beside it; NOT training-run variance (`H-ESTIM-SEED-1`). |\n")
    verdict_lines = []
    if q1:
        c = q1["controls"]
        w("## Q1 — features or head?\n")
        w(f"n: FIT {q1['n_train_clips']} train clips / {q1['n_train_windows']} windows ({q1['n_val_clips']} clips held out as VAL for every "
          f"hyper-parameter); SCORE {q1['n_eval_clips']} eval clips / {q1['n_eval_windows']} windows. Cells: 0.25 m centres, x 0-40 m, |y| < 3 m.\n")
        w("**Controls.** C1 constant: " + ("PASS" if c["C1_constant"]["pass"] else "**FAIL**") + " (" +
          "; ".join(f"{k}: AP {f(v['ap'], 4)} vs rate {f(v['rate'], 4)}, AUROC {f(v['auroc'], 3)}" for k, v in c["C1_constant"]["read"].items()) + "). "
          "C4 head psw>=0.5 vs MAPSEL bands: " + ("PASS" if c["C4_head_psw_vs_mapsel"]["pass"] else "**FAIL — verdict withheld per prereg**") + " (" +
          "; ".join(f"{k} m: {f(v['read'])} vs {v['mapsel']}" for k, v in c["C4_head_psw_vs_mapsel"]["read"].items()) + ").\n")
        w("**Probe zoo (selection on VAL only; d and n per C3):**\n")
        w("| stage / probe | d | n fit cells | wd | VAL AP sidewalk (A, lift-valid) s0 / s1 |\n|---|---|---|---|---|")
        for k, v in q1["zoo"].items():
            w(f"| {dk(k)} | {v['d']} | {v['n_fit']} | {v['wd']:g} | {f(v['val_ap'][0])} / {f(v['val_ap'][1])} |")
        w("")
        T = q1["table"]
        best = q1["best_probe_by_val"]
        bkey = None
        for kk in T.get("A_0_20|lv", {}):
            if kk.startswith(best + "|s"):
                bkey = kk if bkey is None or T["A_0_20|lv"][kk]["ap_sw"] > T["A_0_20|lv"][bkey]["ap_sw"] else bkey
        w(f"**Sidewalk AP on SCORE cells (head = its calibrated posterior; best probe by VAL = `{dk(best)}`).**\n")
        w("| cells | n | sw rate | head AP | best-probe AP | C2 pos-prior AP | C1 AP | head decision P / R | best-probe P @ head R | C2 P @ head R | head drv AP | probe drv AP |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = ["A_0_20|lv", "B_5_40|lv", "A0_0_5|all", "A0_0_5|lv", "A0_0_5|nlv",
                 "bin_0_5|all", "bin_5_10|lv", "bin_10_20|lv", "bin_20_40|lv", "bin_5_10|all", "bin_10_20|all", "bin_20_40|all"]
        for key in order:
            if key not in T:
                continue
            r = T[key]
            bp = None
            for kk in r:
                if kk.startswith(best + "|s"):
                    bp = kk if bp is None or r[kk]["ap_sw"] > r[bp]["ap_sw"] else bp
            hd = r["head_decision"]
            w(f"| {dk(key)} | {r['n_cells']} | {f(r['sw_rate'])} | {f(r['head']['ap_sw'])} | {f(r[bp]['ap_sw']) if bp else 'n/a'} | "
              f"{f(r['C2_posprior']['ap_sw'])} | {f(r['C1_constant']['ap_sw'])} | {f(hd['precision'])} / {f(hd['recall'])} | "
              f"{f(r[bp]['p_at_head_recall_sw']) if bp else 'n/a'} | {f(r['C2_posprior']['p_at_head_recall_sw'])} | "
              f"{f(r['head']['ap_drv'])} | {f(r[bp]['ap_drv']) if bp else 'n/a'} |")
        w("\n(the best-probe column reports the better of its two probe seeds on that row — descriptive; the committed Δ below uses the "
          "VAL-selected seed.)\n")
        w("**Every probe on the primary cells (A 0-20 m, lift-valid):**\n")
        w("| arm | AP sw | AUROC sw | P @ head R | AP drv | 8-class acc |\n|---|---|---|---|---|---|")
        for kk, v in T["A_0_20|lv"].items():
            if isinstance(v, dict) and "ap_sw" in v:
                w(f"| {dk(kk)} | {f(v['ap_sw'])} | {f(v['auroc_sw'])} | {f(v['p_at_head_recall_sw'])} | {f(v['ap_drv'])} | {f(v['acc8'])} |")
        w("\n**Committed contrasts (paired bootstrap):**\n")
        w("| contrast | cells | point | 95 % CI | separated |\n|---|---|---|---|---|")
        for k, v in q1["bootstrap"].items():
            w(f"| {k.split('|')[0]} (`{dk(v['a'])}` − `{dk(v['b'])}`) | {dk(v['cells'])} | {f(v['point'], 4, True)} | "
              f"[{f(v['ci95'][0], 4, True)}, {f(v['ci95'][1], 4, True)}] | {'yes' if v['separated'] else 'no'} |")
        dlc = q1["data_limited_check"]
        vis = q1["visibility"]
        w(f"\n**Data-limited check** (`enc|mlp` VAL AP, full vs half FIT windows): {f(dlc['enc_mlp_val_ap_full'])} vs "
          f"{f(dlc['enc_mlp_val_ap_half'])}, gain {f(dlc['gain_from_doubling'], 3, True)} -> "
          f"{'DATA-LIMITED: probe Δ is a LOWER bound' if dlc['DATA_LIMITED'] else 'not data-limited at this size'}.\n")
        w(f"**Visibility (A0, 0-5 m):** {vis['A0_seen_cells']} GT-seen cells, {f(100 * vis['A0_share_lift_invalid'], 1)} % lift-invalid; "
          f"{vis['A0_head_sw_flags']} head sidewalk flags, {f(100 * vis['A0_share_of_flags_on_lift_invalid'], 1) if vis['A0_share_of_flags_on_lift_invalid'] is not None else 'n/a'} % "
          f"of them on lift-invalid cells -> {'VISIBILITY-LIMITED' if vis['VISIBILITY_LIMITED'] else 'not visibility-limited'}.\n")
        ver, loc = q1["verdict"], q1["localisation"]
        if not c["C4_head_psw_vs_mapsel"]["pass"] or not c["C1_constant"]["pass"]:
            w("**VERDICT WITHHELD** — a mandatory control failed (prereg).\n")
            verdict_lines.append("Q1 verdict WITHHELD (control failure).")
        else:
            w(f"**VERDICT (committed rule):** band A 0-20 m → **{ver['A_0_20']}**; band B 5-40 m → **{ver['B_5_40']}**. "
              f"Localisation: A → **{loc['A_0_20']}**, B → **{loc['B_5_40']}**. Function classes tested: linear, MLP-256, 3x3-conv "
              f"(a failing probe shows only *not readable by these classes*).\n")
            verdict_lines.append(f"Q1: A {ver['A_0_20']} / B {ver['B_5_40']}; localisation A {loc['A_0_20']} / B {loc['B_5_40']}; "
                                 f"A0 {'VISIBILITY-LIMITED' if vis['VISIBILITY_LIMITED'] else 'not visibility-limited'}.")
    else:
        w("## Q1 — NOT YET SCORED (see HANDOFF)\n")
    if q2:
        w("## Q2 — is the GT wrong where the own map disagrees?\n")
        r = q2["occupied_rate"]
        w(f"Windows read {q2['n_windows_read']} (no bevgt {q2['n_windows_no_bevgt']}, no label {q2['n_windows_no_label']}). "
          f"LiDAR-OCCUPIED rate: FP (own sidewalk on GT-drivable, 0-20 m) **{f(r['FP'])}** (n {q2['counts']['FP']['n']}); "
          f"AGR control (own drivable on GT-drivable) {f(r['AGR'])} (n {q2['counts']['AGR']['n']}); MIS (FP cells, ±20 frames) {f(r['MIS'])}. "
          f"By distance: " + "; ".join(f"{k} m {f(v['rate'])} (n {v['n']})" for k, v in q2["fp_occupied_by_distance"].items()) + ".\n")
        w(f"Committed reading: obstacle/GT-error share {'EXISTS' if q2['OBSTACLE_SHARE_EXISTS'] else 'NOT SHOWN'}; upper bound on the precision "
          f"correction (FP − AGR occupancy) {f(q2['fp_share_explained_upper_bound'])}.\n")
        w(f"Curb test: {q2['curb_height_test']}.\n")
        verdict_lines.append(f"Q2: FP occupancy {f(r['FP'])} vs agreement {f(r['AGR'])} vs mispaired {f(r['MIS'])} -> obstacle share "
                             f"{'EXISTS' if q2['OBSTACLE_SHARE_EXISTS'] else 'NOT SHOWN'}; curb test BLOCKED (no ground-height channel).")
    if q3:
        w("## Q3 — the box head (refcv8 EMA-48,852, eval139 S2 windows, from PERC8's banked dump)\n")
        cr = q3["class_read"]
        ci = q3["ci95"]
        w(f"n: {q3['n_windows']} windows / {q3['n_episodes']} episodes / {q3['n_pos']} VIS-1 positives. Controls: re-derived AP@2 m "
          f"{f(q3['ap2_all'], 4)} vs PERC8 banked 0.328 ({'PASS' if q3['control_rederive_vs_perc8_0.328']['pass'] else 'FAIL'}); vectorised "
          f"bootstrap AP == programme AP ({'PASS' if q3['control_fast_ap_equals_programme']['pass'] else 'FAIL'}).\n")
        w("| AP@2 m | value | 95 % CI | n pos |\n|---|---|---|---|")
        w(f"| all | {f(q3['ap2_all'])} | [{f(ci['all'][0])}, {f(ci['all'][1])}] | {q3['n_pos']} |")
        for b, v in q3["ap2_band"].items():
            w(f"| range {b} m | {f(v)} | [{f(ci['band:' + b][0])}, {f(ci['band:' + b][1])}] | {q3['n_pos_band'][b]} |")
        for g, v in q3["ap2_size_group_proxy"].items():
            w(f"| size group {g} (PROXY by class: {', '.join(v['classes'])}) | {f(v['ap2'])} | — | {v['n_pos']} |")
        for k, v in q3["ap2_class"].items():
            w(f"| class {k} | {f(v)} | — | {q3['n_pos_class'][k]} |")
        w(f"\n**Class collapse?** TP (2 m greedy, all confidences) class accuracy **{f(cr['tp_class_accuracy'])}** "
          f"[{f(ci['tp_class_accuracy'][0])}, {f(ci['tp_class_accuracy'][1])}] vs C-MAJ (always '{cr['C_MAJ_majority_class']}') "
          f"{f(cr['C_MAJ_accuracy'])}; non-automobile TPs predicted automobile {cr['non_automobile_predicted_automobile']}/{cr['non_automobile_tp']} "
          f"({f(100 * cr['non_automobile_share_as_automobile'], 1)} %) -> **{'COLLAPSED' if cr['COLLAPSED'] else 'NOT COLLAPSED'}** (committed rule).\n")
        cl = q3["classes"]
        cm = q3["confusion_tp2m"]["rows_gt_cols_pred"]
        w("Confusion (rows GT, cols predicted argmax) of the 2 m greedy TPs:\n")
        w("| GT \\ pred | " + " | ".join(c[:6] for c in cl) + " |\n|---|" + "---|" * len(cl))
        for i, c in enumerate(cl):
            w(f"| {c} | " + " | ".join(str(x) for x in cm[i]) + " |")
        w("\nSize error |dl|+|dw| (m) on the Hungarian pairs, by GT class (pairs recovered by unique distance; "
          f"{q3['size_err_pairs']['gt_class_unresolved']}/{q3['size_err_pairs']['n_pairs']} unresolved and dropped): " +
          "; ".join(f"{k} median {f(v['median'], 2)} (n {v['n']})" for k, v in q3["size_err_l1_lw_m"].items()) +
          f". C-SIZE: {q3['size_err_pairs']['C_SIZE']}.\n")
        verdict_lines.append(f"Q3: AP@2m {f(q3['ap2_all'])}; 0-20 / 20-40 / 40-60 m {f(q3['ap2_band']['0_20'])} / {f(q3['ap2_band']['20_40'])} / "
                             f"{f(q3['ap2_band']['40_60'])}; class {'COLLAPSED' if cr['COLLAPSED'] else 'NOT collapsed'} "
                             f"(TP acc {f(cr['tp_class_accuracy'])} vs majority {f(cr['C_MAJ_accuracy'])}).")
    w("## Headline\n")
    for v in verdict_lines:
        w(f"- {v}")
    out = {"tool": "pbn_assemble.py", "Q1": q1 and {k: q1[k] for k in ("verdict", "localisation", "visibility", "bootstrap", "controls",
                                                                     "data_limited_check", "best_probe_by_val")},
           "Q2": q2, "Q3": q3 and {k: q3[k] for k in ("ap2_all", "ap2_band", "ap2_class", "ap2_size_group_proxy", "class_read", "ci95",
                                                      "size_err_l1_lw_m", "size_err_pairs")},
           "headline": verdict_lines}
    (pkg / "RESULT.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8", newline="\n")
    (pkg / "RESULT_TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(verdict_lines))


if __name__ == "__main__":
    main()
