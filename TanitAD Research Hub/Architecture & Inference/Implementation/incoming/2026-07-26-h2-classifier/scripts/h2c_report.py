"""H2 classifier — render the report tables STRAIGHT FROM THE ARTIFACT.

Every table in `H2_CLASSIFIER.md` is produced by this script from `artifacts/h2c_results.json` and
`artifacts/cost_model.json`. Nothing is hand-copied: the program's own retraction log records three
errors that entered by transcription from prose into a summary, and a report whose numbers are
typed by hand is one edit away from the same class.

usage:  python h2c_report.py <artifacts dir>
"""
from __future__ import annotations

import json
import os
import sys


def ci(d, p=4):
    if d is None:
        return "—"
    if isinstance(d, dict) and "point" in d:
        return f"{d['point']:.{p}f} [{d['lo']:.{p}f}, {d['hi']:.{p}f}]"
    return f"{d:.{p}f}" if isinstance(d, (int, float)) else str(d)


def main():
    A = sys.argv[1]
    R = json.load(open(os.path.join(A, "h2c_results.json")))
    C = (json.load(open(os.path.join(A, "cost_model.json")))
         if os.path.exists(os.path.join(A, "cost_model.json")) else None)
    U, D, OP = R["universe"], R["discrimination"], R["operating_point"]
    br = U["heldout_base_rate"]
    L = []

    L.append("### Discrimination — held-out, (camera, frame) pairs\n")
    L.append(f"*Base rate (chance AP) = **{br:.5f}** ({U['heldout_positives']} positives in "
             f"{U['heldout_camera_frames']} pairs, {U['heldout_clips']} clips, "
             f"{U['heldout_positive_clips']} of them positive).*\n")
    L.append("| arm | AP (episode-cluster bootstrap CI95) | AP / base rate | AUROC |")
    L.append("|---|---|---|---|")
    for a, d in D.items():
        if not isinstance(d, dict):
            continue
        L.append(f"| `{a}` | {ci(d['AP'], 5)} | **{d['AP']['point']/br:.2f}x** | "
                 f"{d['AUROC']:.4f} |")
    L.append("")

    L.append("### Paired AP deltas vs the primary (`head_img_ego`)\n")
    L.append("| contrast | ΔAP | CI95 | separated? |")
    L.append("|---|---|---|---|")
    for k, d in R["paired_AP_deltas"].items():
        L.append(f"| {k} | {d['delta']:+.5f} | [{d['lo']:+.5f}, {d['hi']:+.5f}] | "
                 f"{'**YES**' if d['separated'] else 'no'} |")
    L.append("")

    L.append("### The operating point (pre-registered `theta*` fixed on TRAIN out-of-fold)\n")
    ct = OP["calibration_transfer"]
    L.append(f"Budget `B* = {OP['B_preregistered']}` extra camera activations/frame ⇒ target "
             f"camera-frame rate {ct['target_camera_frame_rate']:.4f}; **realised "
             f"{ct['realised']:.4f}** (ratio {ct['ratio_realised_over_target']:.2f}x) — "
             f"the calibration-transfer test.\n")
    L.append("| arm | firing rate | extra cams/frame | recall | precision | precision lift | "
             "recall on behavioural slice | missed |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in ("head_img_ego", "head_img", "head_ego", "heur_ego_both",
              "heur_ego_both_rate_matched", "random_at_rate"):
        if a not in OP:
            continue
        d = OP[a]
        L.append(f"| `{a}` | {ci(d['rate'])} | {d['extra_cams_per_frame']:.4f} | "
                 f"{ci(d['recall'])} | {ci(d['precision'], 5)} | "
                 f"{ci(d['precision_lift_over_base'], 2)} | "
                 f"{d['recall_behavioural_slice']:.4f} | "
                 f"{d['missed_positives']}/{int(U['heldout_positives'])} |")
    L.append("")
    L.append("| paired recall delta | Δ | CI95 | separated? |")
    L.append("|---|---|---|---|")
    for k, d in OP["paired_recall_deltas"].items():
        L.append(f"| {k} | {d['delta']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                 f"{'**YES**' if d['separated'] else 'no'} |")
    L.append("")

    L.append("### The efficiency trade-off CURVE (not a point)\n")
    L.append("| B (extra cams/frame) | head realised rate | **head recall** | head precision | "
             "head recall (behavioural) | heuristic recall | random recall | saving vs always-on-3 "
             "| saving vs always-on-7 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for row in R["tradeoff_curve"]:
        h = row["head_img_ego"]
        e = row["heur_ego_both"]
        rnd = row["random_at_rate"]
        s3 = s7 = "—"
        if C:
            ho = C["analytic_macs"]["head_over_encoder"]
            b = h["realised_extra_cams_per_frame"]
            s3 = f"{1-(1+b+ho)/3:.3f}"
            s7 = f"{1-(1+b+ho)/7:.3f}"
        L.append(f"| {row['B_extra_cams_per_frame']:.3f} | "
                 f"{h['realised_extra_cams_per_frame']:.4f} | **{h['recall']:.4f}** | "
                 f"{h['precision']:.5f} | {h['recall_behavioural_slice']:.4f} | "
                 f"{e['recall']:.4f} | {rnd['recall_mean']:.4f} | {s3} | {s7} |")
    L.append("")

    E = OP.get("efficiency")
    if E:
        L.append("### The efficiency ledger — where the saving actually comes from\n")
        L.append("| policy | extra cams/frame | cams/frame | saving vs always-on-3 | "
                 "saving vs always-on-7 | recall of `L2_trigger` |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| **never escalate** (free and useless) | 0 | 1.000 | "
                 f"{E['saving_vs_always_on_3__never_escalate']*100:.1f} % | "
                 f"{E['saving_vs_always_on_7__never_escalate']*100:.1f} % | **0.000** |")
        L.append(f"| **L2 ORACLE** — fires exactly on the label | "
                 f"{E['oracle_extra_cams_per_frame']:.4f} | "
                 f"{1+E['oracle_extra_cams_per_frame']:.4f} | "
                 f"{E['oracle_saving_vs_always_on_3']*100:.1f} % | "
                 f"{E['oracle_saving_vs_always_on_7']*100:.1f} % | **1.000** |")
        s3, s7 = E["saving_vs_always_on_3__macs"], E["saving_vs_always_on_7__macs"]
        ec = E["extra_cams_per_frame"]
        L.append(f"| **`head_img_ego` @ B\\*** | {ec['point']:.4f} [{ec['lo']:.4f}, {ec['hi']:.4f}] "
                 f"| {1+ec['point']:.4f} | {s3['point']*100:.1f} % "
                 f"[{s3['lo']*100:.1f}, {s3['hi']*100:.1f}] | {s7['point']*100:.1f} % "
                 f"[{s7['lo']*100:.1f}, {s7['hi']*100:.1f}] | "
                 f"**{OP['head_img_ego']['recall']['point']:.4f}** "
                 f"[{OP['head_img_ego']['recall']['lo']:.4f}, "
                 f"{OP['head_img_ego']['recall']['hi']:.4f}] |")
        L.append(f"| **always escalate** | 2.000 | 3.000 | "
                 f"{E['saving_vs_always_on_3__always_escalate']*100:.1f} % | "
                 f"{E['saving_vs_always_on_7__always_escalate']*100:.1f} % | **1.000** |")
        L.append("")
        L.append("> The saving is set by the POLICY SHAPE, not by classifier quality: between the "
                 "oracle and our operating point the saving vs always-on-7 moves by well under a "
                 "percentage point. **What the classifier has to earn is RECALL at that budget** — "
                 "which is why recall, not saving, is the axis the verdict is decided on.\n")

    S = R.get("c12_label_structure")
    if S:
        L.append("### C12 — the LABEL's own structure, before any model\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| `T_off` (`a_req_off ≥ τ*`) rate | **{S['T_off_rate']*100:.3f} %** "
                 f"({S['T_off_n']} frames) |")
        L.append(f"| `T_seen` (`a_req_seen < τ*`) rate | **{S['T_seen_rate']*100:.3f} %** "
                 f"({S['T_seen_n']} frames) |")
        L.append(f"| composite `L2_trigger` rate (frame level) | {S['trigger_rate_frame_level']*100:.3f} % "
                 f"({S['trigger_n']} frames) |")
        L.append(f"| **P(trigger \\| `T_off`)** | **{S['P_trigger_given_T_off']:.4f}** |")
        L.append(f"| P(trigger \\| `T_seen`) | {S['P_trigger_given_T_seen']:.5f} |")
        L.append("")

    L.append("### C12 — the composite decomposed into its two conjuncts\n")
    L.append("| conjunct | base rate | AP (CI95) | AP / base | AUROC |")
    L.append("|---|---|---|---|---|")
    for k, d in R["c12_conjuncts"].items():
        L.append(f"| `{k}` | {d['base_rate']:.4f} | {ci(d['AP'], 5)} | "
                 f"**{d['AP_over_base']:.2f}x** | {d['AUROC']:.4f} |")
        if "complement" in d:
            c_ = d["complement"]
            L.append(f"| `{c_['name']}` (the rare, informative side) | {c_['base_rate']:.4f} | "
                     f"{ci(c_['AP'], 5)} | **{c_['AP_over_base']:.2f}x** | "
                     f"{1-d['AUROC']:.4f} |")
    L.append("")

    L.append("### Sensitivities\n")
    L.append("| stratum | n positives | base rate | AP (CI95) | AP / base |")
    L.append("|---|---|---|---|---|")
    for k, d in R["sensitivities"].items():
        key = "AP" if "AP" in d else "AP_of_primary_head"
        a = d[key]
        L.append(f"| `{k}` | {d.get('n_positives', '—')} | {d['base_rate']:.5f} | "
                 f"{ci(a, 5)} | **{a['point']/max(d['base_rate'],1e-12):.2f}x** |")
    L.append("")

    if C:
        L.append("### Measured compute (A40, pod2)\n")
        L.append(f"- one frozen encoder+readout pass, batch 32: **{C['wallclock_s_per_item']['encoder_b32']*1e3:.3f} ms** "
                 f"/ camera-frame · analytic **{C['analytic_macs']['encoder_pass_per_camera_frame']/1e9:.2f} GMAC**")
        L.append(f"- one head forward, batch 32: **{C['wallclock_s_per_item']['head_b32']*1e6:.1f} us** "
                 f"/ frame · analytic **{C['analytic_macs']['head_pass_per_frame']/1e6:.1f} MMAC**")
        L.append(f"- **head / encoder = {C['analytic_macs']['head_over_encoder']*100:.3f} % of one "
                 f"camera pass (MACs), {C['head_over_encoder_wallclock']*100:.2f} % (wall-clock)** — "
                 f"the gate is nearly free, so the saving is set by the firing rate, not by the gate.")
        L.append("")
    ts_path = os.path.join(A, "train_summary.json")
    if os.path.exists(ts_path):
        T = json.load(open(ts_path))
        L.append("### Training-side cross-validation (out-of-fold on TRAIN, grouped by chunk)\n")
        L.append(f"*TRAIN: {T['n_train_windows']:,} windows · CV base rate for reference is the "
                 f"mean of the two per-camera base rates.*\n")
        L.append("| arm \\| target | selected config | selected epoch | **CV-AP** | "
                 "full CV-AP-vs-epoch curve (first 6) | params |")
        L.append("|---|---|---|---|---|---|")
        for k, s in T["selection"].items():
            c = s["cfg"]
            L.append(f"| `{k}` | pos_weight {c['pos_weight']:.0f}, d {c['d']} | {s['epoch']} | "
                     f"**{s['cv_ap']:.4f}** | "
                     f"{', '.join(f'{x:.4f}' for x in s['cv_ap_curve'][:6])} … | "
                     f"{s['n_params']:,} |")
        L.append("")
        L.append("| fold | chunks |")
        L.append("|---|---|")
        for f, cs in T["folds"].items():
            L.append(f"| {f} | {', '.join(cs)} |")
        L.append("")

    txt = "\n".join(L)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf-8").write(txt)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)


if __name__ == "__main__":
    main()
