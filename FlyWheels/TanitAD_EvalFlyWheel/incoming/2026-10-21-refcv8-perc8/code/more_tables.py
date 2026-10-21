"""more_tables.py -- controls / refit / within-arm / WiSE markdown for RESULT.md.  python more_tables.py --root <perc8 dir>"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import cell, dcell, f


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    R = a.root
    T = f"{R}/raw/tables"
    os.makedirs(T, exist_ok=True)
    # ---------------- controls
    L = ["| control | known value | read | verdict |", "|---|---|---|---|"]
    g = jd(f"{R}/raw/gt_control.json")
    if g:
        L.append(f"| GT as the box prediction (programme `summarise`, {g['box']['n_windows']} windows, {g['box']['n_pos']} positives) | every defined AP cell = 1.0 | "
                 f"{g['box']['n_ap_cells_defined'] - g['box']['n_cells_not_1']}/{g['box']['n_ap_cells_defined']} cells = 1.0 | {'PASS' if g['box']['pass'] else 'FAIL'} |")
        L.append(f"| GT as the map prediction (structured 1000x600 grid) | pooled IoU = 1.0 in every (class, band) with GT | {g['map']['n_cells_iou_1']}/{g['map']['n_cells_with_gt']} cells = 1.0 | {'PASS' if g['map']['pass_gt_as_prediction'] else 'FAIL'} |")
        L.append(f"| all-one-class map prediction (negative control) | IoU of every other class = 0 | max {g['map']['negative_control_const_class1_other_classes_iou_max']} | {'PASS' if g['map']['pass_negative_control'] else 'FAIL'} |")
    r = jd(f"{R}/raw/rescore_refcv7_control.json")
    if r:
        L.append(f"| re-score refcv7's banked dump with THIS instrument (AP@2 m pooled) | banked 0.24906 (within 1e-3) | {r['box_ap2m_mine']:.6f}; max abs diff over {len(r['box_cells'])} AP/AUROC cells = {r['box_max_abs_diff']:.1e} | {'PASS' if r['box_pass_1e-3'] else 'FAIL'} |")
        L.append(f"| re-summarise refcv7's banked map WindowTable | banked IoU cells | {r['map_n_iou_cells']} IoU cells, max abs diff {r['map_max_abs_diff_iou']} | {'PASS' if r['map_pass_1e-3'] else 'FAIL'} |")
    b = jd(f"{R}/raw/box_read_declared.json") or jd(f"{R}/raw/box_read_final.json")
    if b:
        for k, lab in (("C1_weighted_ap_equals_programme_ap", "weighted-AP kernel == `detection_metrics._ap` (4 thr x 2 arms, 1e-9)"),
                       ("C2_draws_equal_ci_draws", "bootstrap draw stream == `ci.paired_episode_cluster_bootstrap` (recall paired CI)"),
                       ("C3_weighted_auroc_equals_programme_auroc", "weighted AUROC == `detection_metrics.auroc` (1e-9)")):
            c = b["controls"].get(k)
            if c:
                L.append(f"| {lab} | equal | see `raw/box_read_declared.json` controls | {'PASS' if c['pass'] else 'FAIL'} |")
    p = jd(f"{R}/raw/A_nosync/perc.json")
    if p:
        lc = p["law_control"]
        L.append(f"| LAW-frame control (perc pass, {len(lc['windows'])} windows: zero LAW frame == real LAW frame for every perception tensor) | equal | all equal, law term differs | {'PASS' if lc['pass'] else 'FAIL'} |")
        L.append(f"| checkpoint md5 | b4364029b68f29a33eeb4aebbbbd002b (battery record `E:/refcv8_eval_kit/ckpt/ckpt_48852.md5`) | {p['ckpt_md5']} | {'PASS' if p['ckpt_md5'] == 'b4364029b68f29a33eeb4aebbbbd002b' else 'FAIL'} |")
    open(f"{T}/controls.md", "w", encoding="utf-8").write("\n".join(L))
    # ---------------- refit
    ng, nn = jd(f"{R}/raw/T_nosync/refcv8_det_presence_gates_train.json"), jd(f"{R}/raw/T_nosync/refcv8_det_nms_train.json")
    L = []
    if nn and ng:
        h = nn["heads"]["box3d"]
        pv = nn["provenance"]
        L.append(f"Fit set (TRAIN, never EVAL): {pv['fit_set']}; {pv['n_windows']} box windows. Tool `stack/scripts/refit_perception_thresholds.py` "
                 f"(sha256 {pv['tool_sha256'][:16]}...). Checkpoint step {pv['checkpoint_step']}, md5 {pv['checkpoint_md5']}.")
        L.append("")
        L.append(f"| head | P=R gate, no NMS | NMS radius (TRAIN-AP@2 m argmax) | gate re-fitted after NMS | train AP@2 m at that radius |")
        L.append("|---|---|---|---|---|")
        for hd in ("box3d", "agent"):
            e = nn["heads"][hd]
            fit = pv["train_grid"][hd]
            L.append(f"| {hd} | {ng['gates'][hd]:.4f} | {e['radius_m']} m | {e['gate']:.4f} | {fit[f'{e['radius_m']:g}']['train_ap2m']:.4f} (no NMS {fit['none']['train_ap2m']:.4f}) |")
        L.append("")
        L.append("box3d TRAIN grid (radius -> TRAIN AP@2 m / TRAIN P=R gate): " + "; ".join(
            f"{k} -> {v['train_ap2m']:.4f} / {v['train_pr_gate']:.4f}" for k, v in pv["train_grid"]["box3d"].items()))
        L.append("")
        L.append("refcv7@50,400's shipped values for reference (fit on ITS TRAIN-DIAG): box3d P=R gate 0.2567, NMS 2.5 m, gate after NMS 0.2145.")
    else:
        L.append("NOT RUN or not finished: see `raw/T_nosync/` and `logs/` (the TRAIN-DIAG pass / refit did not complete in the time budget).")
    open(f"{T}/refit.md", "w", encoding="utf-8").write("\n".join(L))
    # ---------------- within-arm
    bf = jd(f"{R}/raw/box_read_final.json")
    L = []
    if bf:
        for nm, d in bf["within_arm_effects"].items():
            L.append(f"**{nm}** (variant minus declared; paired = the same draws within the arm)")
            L.append("")
            L.append("| effect | delta [95 % CI] |")
            L.append("|---|---|")
            for k, c in d.items():
                L.append(f"| {k} | {dcell(c)} |")
            L.append("")
    else:
        L.append("NOT RUN (box_read_final.json missing).")
    open(f"{T}/within.md", "w", encoding="utf-8").write("\n".join(L))
    # ---------------- WiSE
    w = jd(f"{R}/raw/wise_control.json")
    if w:
        L = [f"DEC = WiSE decoder-only blend (md5 dad4cb5b5fb316922b4e9a7e85cee8f8; build record: outside_set_is_theta8 = true), {w['n_w_windows']} windows (every 16th S2 window), "
             f"unshifted clock, vs the EMA-48,852 pass A on the same (sha12, ws) keys.", "",
             "| head | windows compared | bit-equal | differ | max abs diff (logit / xy / pair_err) |", "|---|---|---|---|---|"]
        for hd, v in w["box"].items():
            mx = v["max_abs_diff_when_differ"]
            L.append(f"| {hd} slots | {v['n_windows_compared']} | {v['n_windows_bit_equal']} | {v['n_windows_differ']} | {mx['logit']:.2e} / {mx['xy']:.2e} / {mx['pair_err']:.2e} |")
        m = w["map"]
        L.append(f"| 10 cm map rows | {m['n_w_rows']} | {m['n_rows_embedded_bit_identical']} (order-preserving embedding into A's table, bit-identical statistics) | {m['n_w_rows'] - m['n_rows_embedded_bit_identical']} | |")
        L.append("")
        L.append(f"Control verdict: **{'PASS' if w['pass'] else 'FAIL'}** (box and map must read identically).")
    else:
        L = ["NOT RUN (the W_dec_sub pass did not complete in the time budget)."]
    open(f"{T}/wise.md", "w", encoding="utf-8").write("\n".join(L))
    print("tables written to", T)


if __name__ == "__main__":
    sys.exit(main())
