"""Aggregate deliverable 2: V1 (registered) + V2 (post-hoc) results, the real-panel gate numbers, pytest counts, cross-check -> recovery_result.json / RECOVERY_RESULT.md"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import common as C  # noqa: E402
import labeller as L  # noqa: E402

# real-panel gate numbers
z = np.load(RAW / "real_panel.npz")
n_id = n_id_eq = n_off = n_em = viol = dr_n = dr_v = nc_n = nc_v = 0
for i, (kind, c, T, r) in enumerate(z["meta"]):
    s, n = int(z["starts"][i]), int(T - r)
    g, v = z["G"][s:s + n], z["V"][s:s + n]
    ox, oy, oyaw, vown = z["owns"][i]
    own = (ox, oy, oyaw)
    if kind == 0:
        n_id += 1
        res = L.recovery_label(g, v, 0, own, float(vown))
        n_id_eq += int(res["status"] == L.STATUS_IDENTITY and np.array_equal(res["label_own"], L.human_label(g, 0)))
        continue
    n_off += 1
    res = L.recovery_label(g, v, 0, own, float(vown))
    if res["label_own"] is not None:
        n_em += 1
        viol += L.gate_violation_count(res["label_own"], vown, res["k_rej"] + 1)
    dr = L.label_ignore_offset(g, v, 0, own, float(vown))
    if dr["label_own"] is not None:
        dr_n += 1
        dr_v += L.gate_violation_count(dr["label_own"], vown, 6) > 0
    nc = L.recovery_label(g, v, 0, own, float(vown), clip=False, check=False)
    if nc["label_own"] is not None:
        nc_n += 1
        nc_v += L.gate_violation_count(nc["label_own"], vown, nc["k_rej"] + 1) > 0
gate = {"identity_windows": n_id, "identity_bit_equal": n_id_eq, "off_route_records": n_off, "real_labeller_emitted": n_em, "real_labeller_violations": int(viol),
        "regression_arm_ignore_offset": {"labelled": dr_n, "records_failing_gate": int(dr_v), "fraction": dr_v / max(dr_n, 1)},
        "regression_arm_no_clip_no_check": {"labelled": nc_n, "records_failing_gate": int(nc_v), "fraction": nc_v / max(nc_n, 1)},
        "G_REC_real_labeller_PASS": bool(n_id_eq == n_id and viol == 0), "G_REC_ignore_offset_FAILS": bool(dr_v > 0.9 * dr_n)}
out = {"deliverable": "H-ST-RECOVERYLABEL zero-training step", "frozen_cards_sha256": C.FROZEN_SHA, "frozen_amendment_1_sha256": C.AMEND_SHA,
       "frozen_amendment_2_sha256": open(HERE.parent / "FROZEN_AMENDMENT_2.sha256").read().split()[0],
       "tier": "T1-kin rollout states on TRAIN (DAgger bank); labels offline", "evidence_class": "MEASURED", "gate_real_panel": gate,
       "pytest": (RAW / "pytest_counts.txt").read_text(), "crosscheck_vs_stack_recover_controls": json.loads((RAW / "crosscheck_feasible_decode.json").read_text()),
       "V1_registered": {sd: json.loads((RAW / f"recovery_{sd}.json").read_text()) for sd in ("s0", "s1")},
       "V2_posthoc": {sd: json.loads((RAW / f"recoveryV2_{sd}.json").read_text()) for sd in ("s0", "s1")},
       "refusal_breakdown_s0": json.loads((RAW / "refusal_breakdown_s0.json").read_text())}
(HERE / "recovery_result.json").write_text(json.dumps(out, indent=1))

md = ["# H-ST-RECOVERYLABEL zero-training step (deliverable 2)", "",
      f"Frozen: `FROZEN_CARDS.md` `{out['frozen_cards_sha256'][:12]}...`, Amendment 1 `{out['frozen_amendment_1_sha256'][:12]}...` (the labeller construction; written before any real data), Amendment 2 `{out['frozen_amendment_2_sha256'][:12]}...` (POST-HOC V2, written after I saw V1's 3,000-record smoke).",
      "Evidence MEASURED (ours; `recovery_result.json`, `raw/recovery_s{0,1}.json`, `raw/recoveryV2_s{0,1}.json`). Tier: T1-kin rollout states on TRAIN; labels are offline. A census of the full banks (306,566 windows per tick), not a sample, so no sampling interval is quoted; the training-seed variance of the bank itself is the two seeds.", "",
      "Prototype: `recovery/labeller.py` (V1, registered), `recovery/labeller_v2.py` (V2, post-hoc). Not in `stack/`. Envelope constants from `feasible_decode.py` (A_MAX 4.0, KAPPA_MAX 0.2, MU 0.7, 1 % margin); no accel/curvature limit exists in `P10Prior` itself (UNVERIFIED-as-'P10's').", "",
      "## (a) identity, (b) deliberate regression, (d) feasibility", ""]
g = gate
md += [f"* (a) identity: **{g['identity_bit_equal']} / {g['identity_windows']}** real windows reproduce the human label with `np.array_equal` (zero tolerance), also equal to the stack's `gt_waypoints`.",
       f"* (b) gate G-REC on {g['off_route_records']} real off-route records: the real labeller emitted {g['real_labeller_emitted']}, **{g['real_labeller_violations']} checker violations** (PASS); the offset-ignoring arm FAILS the gate on {g['regression_arm_ignore_offset']['records_failing_gate']} / {g['regression_arm_ignore_offset']['labelled']} records ({100 * g['regression_arm_ignore_offset']['fraction']:.1f} %); the no-clip / unchecked arm fails on {g['regression_arm_no_clip_no_check']['records_failing_gate']} / {g['regression_arm_no_clip_no_check']['labelled']} ({100 * g['regression_arm_no_clip_no_check']['fraction']:.1f} %).",
       "* (d) feasibility violations among EMITTED labels, full banks, independent re-check of every label: " + "; ".join(f"V1 {sd}: {out['V1_registered'][sd]['feasibility']['n_emitted_with_checker_violation']} of {out['V1_registered'][sd]['feasibility']['n_emitted']}" for sd in ('s0', 's1')) + "; " + "; ".join(f"V2 {sd}: {out['V2_posthoc'][sd]['feasibility']['n_emitted_with_checker_violation']} of {out['V2_posthoc'][sd]['feasibility']['n_emitted']}" for sd in ('s0', 's1')) + ". Zero by construction (refusal gate) AND measured; a refusal is not a violation. My checker equals the stack's `recover_controls` to 1.8e-15 on 66 real labels.",
       f"* Context: the HUMAN labels themselves violate this envelope at 0.2 s resolution on {100 * out['V1_registered']['s0']['human_label_baseline_beside']['rate']:.1f} % / {100 * out['V1_registered']['s1']['human_label_baseline_beside']['rate']:.1f} % of sampled ok records (s0 / s1), so 'exactly 0' is a property of the recovery phase only.",
       "* pytest: " + out["pytest"].replace("\n", " "), ""]
md += ["## (c) kept fraction (tick 2; card predicts 53.9 % -> ~78 %; frozen band 75-81 %)", "",
       C.md_table([[tag, sd, f"{100 * o[sd]['kept']['tick2']['baseline_kept_fraction']:.2f} %", f"{100 * o[sd]['kept']['tick2']['kept_fraction_with_recovery']:.2f} %", f"{100 * o[sd]['kept']['tick2']['ceiling_if_all_offroute_recovered']:.2f} %",
                    o[sd]["kept"]["tick2"]["recovered_emitted"], f"{100 * o[sd]['kept']['tick1']['baseline_kept_fraction']:.2f} -> {100 * o[sd]['kept']['tick1']['kept_fraction_with_recovery']:.2f} %"]
                   for tag, o in (("V1 registered", out["V1_registered"]), ("V2 post-hoc", out["V2_posthoc"])) for sd in ("s0", "s1")],
                  ["variant", "seed", "tick-2 kept before", "tick-2 kept after", "ceiling (all off-route recovered)", "recovered (tick 2)", "tick 1 before -> after"]), "",
       "**Prediction NOT CONFIRMED** (frozen band 75-81 %): V1 reaches 59.5 % / 62.0 %. V2 (arc lookahead) does not help (58.2 % / 60.1 %). Refusals, V1 over both ticks:", ""]
md.append(C.md_table([[sd] + [out["V1_registered"][sd]["feasibility"]["refusals_by_status"][k] for k in ("emitted", "refused_infeasible", "no_logged_future", "degenerate", "path_exhausted")] for sd in ("s0", "s1")],
                     ["seed", "emitted", "refused_infeasible", "no_logged_future (end-edge hidden by the matcher's cut order)", "degenerate", "path_exhausted"]))
rb = out["refusal_breakdown_s0"]
md += ["", f"Where V1 refuses (s0, tick 2, {rb['n_tick2_cut_records']} off-route records): overall emitted {100 * rb['overall']['emitted']:.1f} %, refused_infeasible {100 * rb['overall']['refused_infeasible']:.1f} %, no_logged_future {100 * rb['overall']['no_logged_future']:.1f} %. "
       "By own speed: " + "; ".join(f"{k} m/s emitted {100 * v['emitted']:.0f} %" for k, v in rb["by_own_speed_mps"].items()) + ". By heading error: " + "; ".join(f"{k} deg emitted {100 * v['emitted']:.1f} %" for k, v in rb["by_heading_error_deg"].items()) + ".", "",
       "## (e) label-vs-human deviation (emitted, world frame, metres)", ""]
rows = []
for tag, o in (("V1", out["V1_registered"]), ("V2", out["V2_posthoc"])):
    for sd in ("s0", "s1"):
        d = o[sd]["deviation_m"]
        rows.append([tag, sd, d["mean_over_k"]["n"]] + [f"{d['mean_over_k'][k]:.2f}" for k in ("p50", "p95", "max")] + [f"{d['end_6s'][k]:.2f}" for k in ("p50", "p95")])
md.append(C.md_table(rows, ["variant", "seed", "n", "mean-over-k p50", "p95", "max", "6 s endpoint p50", "p95"]))
md += ["", "## What it means", "",
       "* The human label at the matched route point stays the only label for ~54 % of tick-2 states; a feasible recovery label is obtainable for about a quarter of the off-route states with this blend, so the card's ~78 % is a ceiling that is out of reach for two measured reasons: **~26 % of tick-2 off-route states have no complete 6 s logged future at r'** (the end-edge condition is hidden behind off_lat / off_head by the matcher's cut order, so the card's 78 % double-counts them) and **~48 % are refused as infeasible**, concentrated at own speed < 3 m/s and heading error > 60 deg (emitted 0.8 % above 90 deg).",
       "* The cheapest remaining levers, named not run: (i) a curvature-bounded planner (Dubins / clothoid with R_min = 5.05 m) in place of the Hermite blend for the low-speed / large-heading states (the Hermite's curvature is the failing check at short rungs); (ii) partial-horizon recovery labels with a loss mask for the no-logged-future states (needs a trainer flag and a G-DVB entry, so box-2 work). Neither can lift tick-2 coverage to 78 %: with no_logged_future removed the labelable ceiling is " + " / ".join(f"{100 * (k['ok'] + k['off_lat']['n'] + k['off_head']['n'] - k['off_lat']['no_logged_future'] - k['off_head']['no_logged_future']) / C.N_WINDOWS:.1f} %" for k in (out['V1_registered'][sd]['kept']['tick2'] for sd in ('s0', 's1'))) + " (s0 / s1)."]
(HERE / "RECOVERY_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md)[:3000])
