"""Render raw/x10_measure.json + raw/x10_dataset_e2e.json into markdown tables (no hand transcription).
    python make_tables.py  ->  ../raw/RESULT_tables_generated.md
"""
import json
import os

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(f"{PK}/raw/x10_measure.json", encoding="utf-8"))
e = json.load(open(f"{PK}/raw/x10_dataset_e2e.json", encoding="utf-8"))
out = []


def f(x, d=4):
    return f"{x:.{d}f}"


out.append("## T1  offset distribution, every provider row (MEASURED, raw/x10_measure.json `corpus_delta`)\n")
out.append("| split | clips | rows | mean ms | p05 | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|---|---|")
for s, o in m["corpus_delta"].items():
    d = o["delta_ms"]
    out.append(f"| {s} | {o['n_clips_scored']} | {d['n']} | {f(d['mean'],3)} | {f(d['p05'],2)} | {f(d['p50'],2)} | {f(d['p95'],2)} | {f(d['p99'],2)} | {f(d['max'],2)} |")

for s, o in m["splits"].items():
    out.append(f"\n## T2  {s}: {o['n_clips']} clips, {o['n_windows']} trainer windows -- training-target shift, uncorrected (U) -> window shift (B)\n")
    out.append("| horizon (ticks / s) | n | mean m | 95% CI (clip-cluster) | p50 | p95 | p99 | max | lon signed mean | lon abs mean | lat abs mean | share > 5 cm | share > 10 cm | interp err mean / p95 vs 100 Hz |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h, v in o["by_horizon_ticks"].items():
        u = v["U_to_B_norm_m"]
        ie = v["interp_error_B_vs_100Hz_truth_m"]
        ci = v["U_to_B_norm_mean_ci95_cluster"]
        out.append(f"| {h} / {float(h)*0.1:.1f} | {u['n']} | {f(u['mean'])} | [{f(ci[0])}, {f(ci[1])}] | {f(u['p50'])} | {f(u['p95'])} | {f(u['p99'])} | {f(u['max'],3)} | "
                   f"{f(v['U_to_B_longitudinal_signed_m']['mean'])} | {f(v['U_to_B_longitudinal_signed_m']['abs_mean'])} | {f(v['U_to_B_lateral_abs_mean_m'])} | "
                   f"{f(v['frac_U_to_B_gt_0p05m'],3)} | {f(v['frac_U_to_B_gt_0p10m'],3)} | {f(ie['mean'],4)} / {f(ie['p95'],4)} |")
    for k, name in (("pooled_0_2s", "pooled ticks 1-20 (0-2 s)"), ("pooled_0_6s", "pooled ticks 1-60 (0-6 s)")):
        p = m["splits"][s][k]
        u = p["U_to_B_norm_m"]
        ci = p["U_to_B_norm_mean_ci95_cluster"]
        out.append(f"| {name} | {u['n']} | {f(u['mean'])} | [{f(ci[0])}, {f(ci[1])}] | {f(u['p50'])} | {f(u['p95'])} | {f(u['p99'])} | {f(u['max'],3)} | | | | | | {f(p['interp_error_m']['mean'],4)} / {f(p['interp_error_m']['p95'],4)} |")
    out.append(f"\n**Why not the per-row shift (A):** mean / p95 movement of the target vs U, vs B:\n")
    out.append("| span | A vs U mean | A vs U p95 | A vs B mean | A vs B p95 |\n|---|---|---|---|---|")
    for k, name in (("pooled_0_2s", "0-2 s"), ("pooled_0_6s", "0-6 s")):
        p = m["splits"][s][k]
        out.append(f"| {name} | {f(p['A_to_U_norm_m']['mean'])} | {f(p['A_to_U_norm_m']['p95'])} | {f(p['A_to_B_norm_m']['mean'])} | {f(p['A_to_B_norm_m']['p95'])} |")
    out.append("\n| NOW-state quantity | n | mean | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|")
    for k, name in (("now_pose_shift_m__B_minus_U", "NOW pose position shift B-U (m)"), ("now_pose_shift_m__truth_100Hz", "NOW pose position shift, 100 Hz log (m)"),
                    ("now_speed_shift_ms__B_minus_U", "v0 shift B-U (m/s)"), ("now_speed_shift_ms__truth_100Hz", "v0 shift, 100 Hz log (m/s)"),
                    ("now_yaw_shift_rad__B_minus_U", "NOW yaw shift B-U (rad)"), ("now_speed_ms", "ego speed at NOW (m/s)")):
        q = o[k]
        out.append(f"| {name} | {q['n']} | {f(q['mean'])} | {f(q['p50'])} | {f(q['p95'])} | {f(q['p99'])} | {f(q['max'])} |")

out.append("\n## T3  controls and cross-checks (MEASURED)\n")
out.append("```json\n" + json.dumps({"controls": m["controls"], "reconstruction_of_cached_poses_from_raw_logs": m["reconstruction_of_cached_poses_from_raw_logs"],
                                      "clock_sidecar_crosscheck": {"n": m["clock_sidecar_crosscheck"]["n"],
                                                                   "grid_start_abs_diff_ms_max": m["clock_sidecar_crosscheck"]["grid_start_abs_diff_ms"]["max"],
                                                                   "dt_abs_diff_s_max": m["clock_sidecar_crosscheck"]["dt_abs_diff_s"]["max"]},
                                      "analytic_delta_crosscheck": {k: (v if not isinstance(v, dict) else {a: v[a] for a in ("n", "mean", "p50", "p95", "max") if a in v})
                                                                     for k, v in m["analytic_delta_crosscheck"].items() if k != "what"}}, indent=1) + "\n```")

out.append("\n## T4  the actual dataset hook on real eval139 poses (OFF vs ON through V3Dataset, MEASURED, raw/x10_dataset_e2e.json)\n")
out.append(f"windows evaluated {e['n_windows_evaluated']}; sidecar census {json.dumps({k: e['census'][k] for k in ('n_clips','n_covered','n_uncovered_read_unshifted')})}\n")
out.append(f"keys whose value changed at least once: `{json.dumps(e['keys_that_changed_at_least_once'])}` (every other key of the item is byte-identical)\n")
out.append("| horizon ticks | n | mean m | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|")
for h, v in e["target_shift_by_horizon_ticks__hook_U_to_B_m"].items():
    out.append(f"| {h} | {v['n']} | {f(v['mean'])} | {f(v['p50'])} | {f(v['p95'])} | {f(v['p99'])} | {f(v['max'],3)} |")
ps = e["POSE_SYNC_ms__equivalent_timing_residual_of_pose_last_vs_image_instant_truth"]
out.append(f"\nPOSE-SYNC (SPEC 8.10: mean |t_pose - t_image|, expressed as ||pose_last - pose_100Hz(t_image)|| / speed, windows with v > 2 m/s, n {ps['n_windows_v_gt_2']}):\n")
out.append("| arm | mean ms | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|")
for k, name in (("uncorrected_OFF", "OFF (refcv7 path)"), ("corrected_ON", "ON (pose-sync)")):
    q = ps[k]
    out.append(f"| {name} | {f(q['mean'],3)} | {f(q['p50'],3)} | {f(q['p95'],3)} | {f(q['p99'],3)} | {f(q['max'],3)} |")
out.append("\nOther fields: v0 shift " + json.dumps({k: round(v, 5) for k, v in e["v0_shift_ms"].items()}) +
           "; kinematic factored-label flips " + json.dumps(e["kinematic_factored_label_flips"]) +
           "; ego-history encoder channels (abs shift) " + json.dumps({c: {k: round(v, 5) for k, v in q.items() if k in ("mean", "p95", "max")} for c, q in e["ego_history_encoder_input_channel_shift_abs"].items()}) +
           "; goal_tac xy shift " + json.dumps({k: round(v, 4) for k, v in e["goal_tac_shift"]["goal_tac_xy_shift_m"].items()}))
open(f"{PK}/raw/RESULT_tables_generated.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", f"{PK}/raw/RESULT_tables_generated.md")
