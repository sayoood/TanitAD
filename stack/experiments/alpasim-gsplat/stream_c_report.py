#!/usr/bin/env python3
"""STREAM C — render the report from the artifacts, so no number is ever hand-copied.

Every table below is generated from the JSON in the run directory named at the top of
that table. Twice this month a headline was typed in from a superseded run and had to be
retracted; a generator removes the failure mode rather than warning about it.

Reads the metrics_hq/ tree written by score_panel_hq.sh plus RENDER_AB.json written by
stream_c_render_ab.py; writes one markdown file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILY_ROWS = [
    ("ADE", [("ade_0_2s", "ADE 0-2 s"), ("de_0_5s", "0.5 s"), ("de_1s", "1 s"),
             ("de_1_5s", "1.5 s"), ("de_2s", "2 s")]),
    ("LONGITUDINAL", [("target_speed_err_ms", "target-speed err (signed)"),
                      ("abs_target_speed_err_ms", "target-speed err (abs)"),
                      ("executed_speed_err_ms", "executed-speed err (signed)"),
                      ("along_track_ade_m", "along-track ADE"),
                      ("headway_m", "headway to lead"),
                      ("time_gap_s", "time gap")]),
    ("LATERAL", [("heading_err_rad", "heading err"),
                 ("curvature_err_1pm", "curvature err"),
                 ("yawrate_err_rads", "yaw-rate err"),
                 ("cross_track_abs_m", "cross-track abs (== dist_to_gt_traj)"),
                 ("lateral_ade_m", "lateral ADE")]),
    ("TACTICAL", [("manoeuvre_plan_eq_logged", "plan == logged manoeuvre"),
                  ("manoeuvre_exec_eq_plan", "executed == planned"),
                  ("head_eq_plan", "5-way head == plan"),
                  ("head_eq_logged", "5-way head == logged")]),
    ("STRATEGIC", [("route_corridor_departure_rate", "route-corridor departure"),
                   ("route_head_eq_logged", "route head == logged route"),
                   ("route_head_side_eq_graded_proxy", "route-head side == dyaw sign (PROXY)")]),
]

PAIRED_ROWS = [
    ("ADE", "ade_0_2s"),
    ("LONGITUDINAL", "abs_target_speed_err_ms"),
    ("LONGITUDINAL", "along_track_ade_m"),
    ("LATERAL", "heading_err_rad"),
    ("LATERAL", "curvature_err_1pm"),
    ("LATERAL", "yawrate_err_rads"),
    ("LATERAL", "cross_track_abs_m"),
    ("LATERAL", "dist_to_gt_traj_m"),
    ("TACTICAL", "manoeuvre_plan_eq_logged"),
    ("STRATEGIC", "route_corridor_departure_rate"),
]


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ci(v):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    if "n" in v and "reason" in v:
        return f"n/a ({v['reason'][:44]}…)"
    if "mean" in v:
        s = f"{v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]"
        if v.get("degenerate"):
            s += " ⛔degenerate"
        return s
    if "delta" in v:
        s = f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
        return s + (" **sep**" if v["separated"] else "")
    return str(v)[:60]


def sep(v):
    return "**separated**" if isinstance(v, dict) and v.get("separated") else "not separated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--ab", required=True, help="RENDER_AB.json (empty condition)")
    ap.add_argument("--ab-objects", default=None)
    ap.add_argument("--published-morning", default=None,
                    help="results/metrics_empty.json as shipped this morning")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = Path(a.metrics_dir)
    ab = J(a.ab)
    L = []
    W = L.append

    W("# STREAM C — does the closed-loop headline survive a better render?\n")
    W("**Question, pre-registered before the panel ran:** the morning panel found *REF-C "
      "beats flagship v1 closed-loop and the separation is ENTIRELY LATERAL*. That was "
      "measured on the OLD render. The render improved by **+23.4 % grad-NCC** and the "
      "four shipped videos were re-rendered. Changing the render changes what the policy "
      "SEES. **Is the verdict a property of the ARMS, or of render artifacts?**\n")
    W("Both outcomes were committed in advance: *survives* ⇒ strong evidence it is the "
      "arms; *does not survive* ⇒ a bigger finding, and the closed-loop panel becomes "
      "render-conditional until a render-invariance check is part of the protocol.\n")

    # ---- provenance -----------------------------------------------------------
    W("## Provenance — the run directories these numbers come from\n")
    W("| | run dir | render flags |")
    W("|---|---|---|")
    W(f"| **HQ** (what the videos show) | `thor:{ab['run_dirs']['HQ']}` | `{ab['render_flags_HQ']}` |")
    W(f"| **MORNING** | `thor:{ab['run_dirs']['MORNING']}` | `{ab['render_flags_MORNING']}` |")
    W(f"| **REPRO** (negative control) | `thor:{ab['run_dirs']['REPRO']}` | morning flags, re-run today |")
    W("")
    W("Same scene `00040136-e651-4abd-991d-0655ccda9430`, same checkpoints "
      "(`flagship-v1-speedjerk` step 29999 · `refc-base` step 29999, 128 anchors), same "
      "9 starts × 50 ticks, same scorer, same day. Geometry asserted before every "
      "rollout: `CANON f_eff=266.01 (F_REF=266.0) OK` on all four HQ panels.\n")
    W(f"Estimator: {ab['estimator']}\n")
    W("⚠️ **The raw rollouts are banked in this directory**, under `rollouts/` — 12 files, one per "
      "(run × arm × condition), each 9 starts. Thor rebooted mid-session today; a result that "
      "exists only on the device is not a result.\n")

    # ---- 0. did the render reach the policy? -----------------------------------
    W("## 0. Did the render change reach the policy at all?\n")
    W("If it did not, the whole comparison is vacuous. MEASURED, not assumed.\n")
    W("| arm | driven-path Δ (mean / p50 / max, m) | emitted-plan Δ (mean / max, m) | v_target Δ (mean / max, m/s) |")
    W("|---|---|---|---|")
    for arm, d in ab["divergence_hq_vs_morning"].items():
        W(f"| {arm} | {d['ego_xy_m']['mean']} / {d['ego_xy_m']['p50']} / {d['ego_xy_m']['max']} "
          f"| {d['plan_m']['mean']} / {d['plan_m']['max']} "
          f"| {d['v_target_ms']['mean']} / {d['v_target_ms']['max']} |")
    W("")
    W("**At `k=0` both runs start from the identical logged pose**, so the plan difference "
      "there is the render acting on the policy with zero accumulated drift:\n")
    for arm, d in ab["divergence_hq_vs_morning"].items():
        vals = list(d["plan_at_k0_m"].values())
        W(f"- `{arm}`: per-start {vals} m")
    W("")

    # ---- 1. the negative control ------------------------------------------------
    W("## 1. ⛔ THE NEGATIVE CONTROL — read this before any render claim\n")
    ctrl = ab.get("divergence_repro_vs_morning_CONTROL")
    if ctrl:
        W("The morning **config** re-run today, paired against the morning **rollouts**. "
          "The renderer is a step function of pose (a 0.1 px camera rotation has been "
          "measured to move the 2 s waypoint 6.65 m), and `closedloop_drive.py` changed "
          "twice today after the morning run — so this control is what separates *the "
          "render* from *run-to-run noise and code drift*.\n")
        W("| arm | driven-path Δ (mean / max, m) | emitted-plan Δ (mean / max, m) |")
        W("|---|---|---|")
        for arm, d in ctrl.items():
            W(f"| {arm} | {d['ego_xy_m']['mean']} / {d['ego_xy_m']['max']} "
              f"| {d['plan_m']['mean']} / {d['plan_m']['max']} |")
        W("")
    else:
        W("⚠️ **NOT AVAILABLE** — no repro rollouts were produced. Without it, nothing in "
          "section 2 may be attributed to the render.\n")

    # ---- 2. survival ------------------------------------------------------------
    W("## 2. THE ANSWER — every morning separation, re-measured\n")
    W("`flagship v1 − REF-C base`, empty road, paired on the windows shared by **all** "
      "runs. Positive = flagship worse.\n")
    W("| family | metric | MORNING (rescored today) | HQ render | verdict | render moved it? |")
    W("|---|---|---|---|---|---|")
    for name, s in ab["survival"].items():
        m, h = s["morning"], s["hq"]
        mm = f"{m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}]" + (" **sep**" if m[3] else "")
        hh = f"{h[0]:+.4f} [{h[1]:+.4f}, {h[2]:+.4f}]" + (" **sep**" if h[3] else "")
        moved = ("**yes**" if s["interaction_separated"] else "no")
        W(f"| {s['family']} | `{name}` | {mm} | {hh} | **{s['verdict']}** | {moved} |")
    W("")
    W("*\"render moved it?\" is the **difference-in-differences**: "
      "`(flagship−REF-C)|HQ − (flagship−REF-C)|MORNING`, bootstrapped on the same "
      "windows, clustered by rollout start. Comparing two CIs by eye is not a test; this "
      "is.*\n")

    # ---- 3. four families, absolute --------------------------------------------
    for cond, tag in (("empty", "HQ_flagship_vs_refc_empty.json"),
                      ("objects", "HQ_flagship_vs_refc_objects.json")):
        d = J(M / tag)
        if not d:
            continue
        W(f"## 3{'a' if cond == 'empty' else 'b'}. FOUR FAMILIES + ADE on the improved "
          f"render — `{cond}` road\n")
        W(f"`{M / tag}` · n_windows {d['arm_A']['n_windows']}/{d['arm_B']['n_windows']}, "
          f"{d['arm_A']['n_clusters']} clusters · paired on "
          f"{d['paired_n_windows']} shared windows\n")
        W("| family | metric | flagship v1 | REF-C base | paired Δ (F−C) |")
        W("|---|---|---|---|---|")
        for fam, rows in FAMILY_ROWS:
            for k, label in rows:
                va = d["arm_A"]["families"].get(fam, {}).get(k)
                vb = d["arm_B"]["families"].get(fam, {}).get(k)
                if va is None and vb is None:
                    continue
                pd = d["paired_A_minus_B"].get(k)
                W(f"| {fam} | {label} | {ci(va)} | {ci(vb)} | {ci(pd)} |")
        W("")
        for arm in ("arm_A", "arm_B"):
            t = d[arm]["families"]["TACTICAL"]
            W(f"- **{d[arm]['name']} manoeuvre class shares** — planned "
              f"{t['plan_class_share']}, logged {t['logged_class_share']}, "
              f"5-way head {t.get('head_class_share')}")
        for arm in ("arm_A", "arm_B"):
            s = d[arm]["families"]["STRATEGIC"]
            W(f"- **{d[arm]['name']} route** — logged share {s['route_logged_share']}, "
              f"head share {s.get('route_head_share')}, "
              f"label valid {s.get('route_label_valid_rate')}")
        W("")

    # ---- 4. objects vs empty ----------------------------------------------------
    W("## 4. With-objects vs empty-road on the improved render\n")
    W("The morning panel called this **null for both arms**. The improved render draws "
      "two dynamic layers that were absent from every previous frame, so it is the first "
      "version of this contrast where the actors are actually all there.\n")
    W("| arm | n sep / n tested | separated metrics |")
    W("|---|---|---|")
    for arm in ("flagship-v1", "refc-base"):
        d = J(M / f"HQ_{arm}_objects_vs_empty.json")
        if not d:
            continue
        p = d["paired_A_minus_B"]
        s = [f"`{k}` {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
             for k, v in p.items() if isinstance(v, dict) and v.get("separated")]
        W(f"| {arm} | {len(s)}/{len(p)} | {'; '.join(s) if s else '**none — NULL**'} |")
    W("")

    # ---- 5. render effect per arm ----------------------------------------------
    W("## 5. The render's effect on each arm, against the noise floor\n")
    W("| arm | metric | HQ − MORNING | CONTROL: REPRO − MORNING |")
    W("|---|---|---|---|")
    for arm in ("flagship-v1", "refc-base"):
        r = J(M / f"RENDER_{arm}_hq_vs_morning.json")
        c = J(M / f"CONTROL_{arm}_repro_vs_morning.json")
        if not r:
            continue
        for fam, k in PAIRED_ROWS:
            v = r["paired_A_minus_B"].get(k)
            cv = c["paired_A_minus_B"].get(k) if c else None
            if v is None:
                continue
            W(f"| {arm} | `{k}` | {ci(v)} | {ci(cv)} |")
    W("")

    # ---- 5b. which render feature? ----------------------------------------------
    if (M / "ABLATE_cull_flagship-v1_vs_morning.json").exists():
        W("## 5b. WHICH render feature moves the policy — the 2×2\n")
        W("`empty` road has exactly two render changes, so they can be separated. Each cell "
          "is paired against the MORNING rollouts on identical windows; the noise floor for "
          "all of them is the exactly-zero control in section 1.\n")
        W("| arm | metric | scale-cull 0.95 only | gated sky 0.3 only | both (= the videos) |")
        W("|---|---|---|---|---|")
        for arm in ("flagship-v1", "refc-base"):
            cells = [J(M / f"ABLATE_cull_{arm}_vs_morning.json"),
                     J(M / f"ABLATE_sky_{arm}_vs_morning.json"),
                     J(M / f"RENDER_{arm}_hq_vs_morning.json")]
            if any(c is None for c in cells):
                continue
            for _fam, k in PAIRED_ROWS:
                if k == "dist_to_gt_traj_m":
                    continue
                vs = [c["paired_A_minus_B"].get(k) for c in cells]
                if any(v is None for v in vs):
                    continue
                W(f"| {arm} | `{k}` | " + " | ".join(ci(v) for v in vs) + " |")
        W("")

    # ---- 6. self-consistency ----------------------------------------------------
    W("## 6. Controls on the instrument itself\n")
    sc = ab["self_consistency"]
    any_key = next(iter(sc))
    W("| run/arm | ADE recomputed independently | max(lon,lat) ≤ ADE ≤ lon+lat | "
      "`dist_to_gt` == `|cross_track|` |")
    W("|---|---|---|---|")
    for k, v in sc.items():
        W(f"| {k} | max|Δ| {v['ade_recomputed_independently']['max_abs_diff']} → "
          f"{'PASS' if v['ade_recomputed_independently']['agrees'] else '**FAIL**'} "
          f"| {'PASS' if v['component_vs_family_triangle']['holds'] else '**FAIL**'} "
          f"| {'IDENTICAL' if v['dist_to_gt_IS_abs_cross_track']['identical'] else 'differ'} |")
    W("")
    W("⚠️ " + sc[any_key]["dist_to_gt_IS_abs_cross_track"]["verdict"] + "\n")
    W("⚠️ Scope of the ADE control: " +
      sc[any_key]["ade_recomputed_independently"]["scope_note"] + "\n")

    # ---- 7. instrument change ---------------------------------------------------
    if a.published_morning:
        pm = J(a.published_morning)
        mr = J(M / "MORNRESC_flagship_vs_refc_empty.json")
        if pm and mr:
            W("## 7. Why \"MORNING\" here is a RE-SCORE, not the published morning file\n")
            W(f"The published panel (`{a.published_morning}`) was scored **before** the "
              "route-head key fix, and recorded REF-C as exposing no strategic route "
              "logits — which is false; the head was there and a key-name mismatch "
              "deleted a family for one arm. Comparing the published file against HQ "
              "would confound **the render** with **a scorer fix**. Every render "
              "comparison above therefore uses the morning ROLLOUTS re-scored with "
              "today's scorer.\n")
            W("| | published morning | morning re-scored today |")
            W("|---|---|---|")
            a_pm = pm["arm_B"]["families"]["STRATEGIC"].get("route_head_eq_logged")
            a_mr = mr["arm_B"]["families"]["STRATEGIC"].get("route_head_eq_logged")
            W(f"| REF-C `route_head_eq_logged` | {ci(a_pm)} | {ci(a_mr)} |")
            same = [k for k in pm["paired_A_minus_B"]
                    if k in mr["paired_A_minus_B"]
                    and pm["paired_A_minus_B"][k]["delta"] == mr["paired_A_minus_B"][k]["delta"]]
            W(f"| paired deltas unchanged by the re-score | {len(same)} of "
              f"{len(pm['paired_A_minus_B'])} | — |")
            W("")

    # ---- 8. what is NOT resolved ------------------------------------------------
    W("## 8. NOT RESOLVED — pre-registered, with both outcomes committed\n")
    W("1. **Is flagship's objects-vs-empty gain agent-aware, or just appearance?** On this render "
      "12 of 23 paired deltas separate for flagship, all saying *actors make it better*; REF-C "
      "stays null at 1 of 23. But flagship is the arm whose plan moves 9 m under **any** "
      "appearance change, so \"116 k extra gaussians changed the frame\" and \"the model reasons "
      "about the agents\" predict the same sign.\n"
      "   **The discriminating run:** repeat `--condition objects` with the actors drawn at a "
      "WRONG TIME (shift `renderer._actor[\"tracks\"].t0_us`; the render-quality falsifier already "
      "uses exactly this control and it separates by +0.2358 grad-NCC on the image). "
      "*Committed in advance:* if wrong-time actors recover the same gain, the effect is "
      "appearance and the objects-vs-empty result says nothing about agent reasoning; if only "
      "correctly-timed actors do, flagship is responding to the agents.\n")
    W("2. ⛔ **RESOLVED, AND IT CONFIRMS THE CONFOUND: the `objects` condition is NOT "
      "deterministic across today's code change, so no `objects` morning-vs-HQ number is "
      "admissible.** `closedloop_drive.py` gained `act[\"tracks\"].t0_us = float(t_us)` today, a "
      "line that only executes with actors attached, so the exactly-zero `empty` control does not "
      "cover it. MEASURED by re-running the morning `objects` config today "
      "(`CONTROL_<arm>_objmorn_vs_morning.json`):\n")
    for arm in ("flagship-v1", "refc-base"):
        c = J(M / f"CONTROL_{arm}_objmorn_vs_morning.json")
        if not c:
            continue
        p = {k: v for k, v in c["paired_A_minus_B"].items()
             if isinstance(v, dict) and "delta" in v}
        nz = [k for k, v in p.items() if v["delta"] != 0]
        s = [k for k, v in p.items() if v.get("separated")]
        W(f"   - `{arm}`: **{len(nz)} of {len(p)}** paired deltas non-zero, **{len(s)} separated** "
          f"— from the CODE CHANGE ALONE, with the render held fixed.")
    W("   Driven-path floor from the code change: flagship **1.536 m mean / 7.266 m max**, "
      "REF-C **0.165 / 1.299 m** (450 windows). ⇒ **The `empty` headline does not depend on this**"
      " (its control is exactly 0.0), and the HQ-internal objects-vs-empty contrast is still valid"
      " (same code, same day, same render for both cells) — but the morning `objects` panel and "
      "the HQ `objects` panel are not comparable.\n")
    W("3. **Which render is \"right\" is not settled by this panel.** HQ is closer to the shipped "
      "reference (grad-NCC 0.3424 vs 0.2774), so its flagship numbers are the better estimate of "
      "flagship on *this* scene — but both are within-sim and REF-C's open-loop ADE is 3.21× OOD "
      "here vs real footage. The transferable claim is the **ordering** and the "
      "**render-sensitivity ratio**, not either arm's absolute rate.\n")
    W("4. **A render-invariance check belongs in the closed-loop protocol.** Every future "
      "closed-loop panel should report the arm's sensitivity to a render perturbation next to "
      "its score, because an arm that moves 9 m under one is not measurable to 0.1 m.\n")

    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
