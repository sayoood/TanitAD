"""D-APPEAR — emit EVERY table in the deliverable FROM the JSON. Nothing is hand-transcribed.

Reads whichever of the four result files exist and writes ``raw/summary_tables.txt``.
If a number appears in APPEARANCE_SHORTCUT.md and not here, it is a transcription error.

usage:  python summarize.py > raw/summary_tables.txt
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMA_REF = HERE.parent / "2026-08-03-latent-bottleneck" / "results_temporal_falsifier.json"


def load(name):
    p = HERE / name
    return json.load(open(p)) if p.exists() else None


def sepmark(d):
    return "SEP" if d.get("separated") else "---"


def hdr(t):
    print(f"\n{'='*100}\n{t}\n{'='*100}")


# --------------------------------------------------------------------------- #
p1 = load("results_p1_physicalai.json")
if p1:
    m = p1["meta"]
    hdr("P1 — PhysicalAI-AV: speed R2 per arm, with each arm's OWN shuffled control")
    print(f"corpus            : {m['corpus']}")
    print(f"encoder           : {m['encoder']} step {m['encoder_step']}  (SAME encoder as the "
          f"comma2k19 panel)")
    print(f"episodes/windows  : {m['n_episodes']} eps -> {m['n_train_eps']} train "
          f"({m['n_train_windows']} win) / {m['n_holdout_eps']} held out "
          f"({m['n_holdout_windows']} win)")
    print(f"rigs              : {m['rig_counts']}")
    print(f"alignment (A1)    : max |v_latentbank - v_epcache| = "
          f"{m['alignment_checks']['A1_max_abs_speed_delta_mps']:.3e} m/s")
    print(f"estimator         : {m['estimator']}, B={m['n_boot']}")
    print(f"\n{'arm':28s} {'feat':>6s} {'kernel':>7s} {'speed R2':>10s} "
          f"{'d vs shuf':>10s} {'ci95 lo':>9s} {'ci95 hi':>9s}  sep")
    for a, r in p1["arms"].items():
        if a == "NULL_train_mean":
            print(f"{a:28s} {0:6d} {'none':>7s} {r['r2']['speed']['point']:+10.4f} "
                  f"{'-':>10s} {'-':>9s} {'-':>9s}   (the empirical null)")
            continue
        d = r["delta_vs_shuffled"]["speed"]
        print(f"{a:28s} {r['n_features']:6d} {r['kernel']:>7s} "
              f"{r['r2']['speed']['point']:+10.4f} {d['delta']:+10.4f} "
              f"{d['lo']:+9.4f} {d['hi']:+9.4f}  {sepmark(d)}")

    if "primary" in p1:
        pr = p1["primary"]
        hdr("P1 — THE PRE-REGISTERED PRIMARY STATISTIC")
        print(f"definition : {pr['definition']}")
        print(f"numerator   (pix32_centre_rbf, still frame) : "
              f"{pr['numerator_still_frame']:+.5f}   "
              f"separates={pr['still_frame_separates']}")
        print(f"denominator (v1_window, 800 ms latent)      : "
              f"{pr['denominator_latent_window']:+.5f}   "
              f"separates={pr['latent_window_separates']}")
        print(f"RATIO on PhysicalAI-AV : {pr['RATIO']:+.5f}  "
              f"CI95 {pr['ratio_ci95_episode_cluster_bootstrap']}")
        print(f"RATIO on comma2k19     : {pr['comma2k19_reference_ratio']:+.5f}  (0.6642/0.7145)")
        pd_ = pr["paired_delta_still_minus_latent"]
        print(f"paired delta (still - latent) : {pd_['delta']:+.5f} "
              f"[{pd_['lo']:+.5f}, {pd_['hi']:+.5f}] {sepmark(pd_)}  "
              f"n_ep={pd_['n_episodes']}")
        print(f"\n>>> PRE-REGISTERED OUTCOME: {pr['PREREG_OUTCOME']}")

    hdr("P1 — FOUR METRIC FAMILIES (per family, never pooled). ADE is ONE ROW of four.")
    for a in ("v1_window", "pix32_centre_rbf", "mot16_window_rbf", "NULL_train_mean"):
        r = p1["arms"].get(a)
        if not r:
            continue
        ff = r["four_families"]
        L, LA, TA, ST = ff["LONGITUDINAL"], ff["LATERAL"], ff["TACTICAL"], ff["STRATEGIC"]
        ade = r["ade_2s"]
        print(f"\n--- {a} --- (ADE_2s {ade.get('mean', ade.get('point')):.4f} m "
              f"[{ade['lo']:.4f},{ade['hi']:.4f}])")
        print(f"  LONGITUDINAL  target-speed MAE {L['scalar_speed_mae_mps']:.4f} m/s  "
              f"bias {L['scalar_speed_bias_mps']:+.4f}  along-track MAE {L['along_mae_m']:.4f} m "
              f"bias {L['along_bias_m']:+.4f}  accel MAE {L['accel_mae_mps2']:.4f} m/s2")
        dk = L.get("distance_keeping", {})
        print(f"                distance-keeping/TTC: {dk.get('status', dk)}"
              + (f" — {dk.get('reason','')}" if isinstance(dk, dict) and "reason" in dk else ""))
        print(f"  LATERAL       heading MAE {LA['heading_mae_rad']:.5f} rad (n={LA['n_heading']})  "
              f"curvature MAE {LA['curvature_mae_inv_m']:.6f} 1/m (n={LA['n_curvature']})  "
              f"yaw-rate MAE {LA['yaw_rate_mae_rad_s']:.5f} rad/s  "
              f"cross-track MAE {LA['cross_track_mae_m']:.4f} m")
        for sub in ("lateral", "longitudinal", "mixed"):
            s = TA.get(sub, {})
            if isinstance(s, dict) and "balanced_accuracy" in s:
                print(f"  TACTICAL/{sub:12s} bal-acc {s['balanced_accuracy']:.4f} "
                      f"(chance {s.get('chance_balanced_accuracy')})  "
                      f"support {s.get('support_gt')}")
        gs = TA.get("goal_setting", {})
        if isinstance(gs, dict):
            print(f"  TACTICAL/goal_setting  {json.dumps(gs)[:180]}")
        print(f"  STRATEGIC     {ST.get('status')} — {ST.get('reason')} (n={ST.get('n')})")

    if p1.get("strata"):
        hdr("P1 — STRATIFIED speed R2 (pre-registered: pooled numbers are not admissible alone)")
        arms = [a for a in ("v1_window", "pix32_centre_rbf", "mot8_window_rbf")
                if a in p1["arms"]]
        print(f"{'stratum':22s} {'n':>6s} {'n_ep':>5s} {'v_mean':>7s} " +
              " ".join(f"{a[:16]:>17s}" for a in arms) + "   powered")
        for s, r in p1["strata"].items():
            cells = " ".join(
                f"{(r['arms'][a]['r2_speed'] if r['arms'][a]['r2_speed'] is not None else float('nan')):+17.4f}"
                for a in arms)
            print(f"{s:22s} {r['n']:6d} {r['n_episodes']:5d} "
                  f"{(r['gt_speed_mean'] if r['gt_speed_mean'] is not None else float('nan')):7.2f} "
                  f"{cells}   {'yes' if r['powered'] else 'UNPOWERED'}")
        print("\n⚠️ within-stratum R2 uses the STRATUM's own variance as denominator, so a "
              "narrow speed bin has a shrunken denominator by construction and its R2 is not "
              "comparable to the pooled value. Read the ORDERING across arms, not the level.")

# --------------------------------------------------------------------------- #
p1b = load("results_p1b_mechanism.json")
if p1b:
    hdr("P1b — THE LADDER that separates 'the probe is broken' from 'the map does not transfer'")
    print("within_clip = random WINDOW split (LEAKY BY CONSTRUCTION — substrate check only)")
    print("across_clip = episode-disjoint i%3 (the real number)\n")
    print(f"{'corpus':30s} {'arm':18s} {'within_clip':>12s} {'across_clip':>12s} "
          f"{'speed mean':>11s} {'speed cv':>9s}")
    for c, r in p1b["corpora"].items():
        for a, rr in r["arms"].items():
            print(f"{c:30s} {a:18s} {rr['within_clip']['r2_speed']:+12.4f} "
                  f"{rr['across_clip']['r2_speed']:+12.4f} "
                  f"{r['speed_mps']['mean']:11.3f} {r['speed_mps']['cv']:9.3f}")
    hdr("P1b — substrate integrity (a degenerate pixel block would show here)")
    for c, r in p1b["corpora"].items():
        for k, s in r["substrate"].items():
            print(f"{c:30s} {k:6s} n={s['n_windows']:6d} W={s['W']} D={s['D']:5d}  "
                  f"centre mean {s['centre_mean']:+.4f} std {s['centre_std']:.4f} "
                  f"range [{s['centre_min']:.3f},{s['centre_max']:.3f}]  "
                  f"constant feats {s['n_constant_features']}  "
                  f"median feat std {s['median_feature_std']:.5f}")
    hdr("P1b — speed distributions (the corpus property the shortcut depends on)")
    for c, r in p1b["corpora"].items():
        s = r["speed_mps"]
        print(f"{c:30s} mean {s['mean']:6.2f}  std {s['std']:5.2f}  cv {s['cv']:5.3f}  "
              f"p05 {s['p05']:6.2f}  p50 {s['p50']:6.2f}  p95 {s['p95']:6.2f}  "
              f"frac<1m/s {s['frac_below_1mps']:.3f}")

# --------------------------------------------------------------------------- #
p2 = load("results_p2_rig.json")
if p2:
    hdr("P2 — G1: the HORIZON ROW, MEASURED from the pixels of the substrate actually used")
    g1 = p2["G1_horizon_row"]
    print(f"rig A: argmax row {g1['rig_a']['argmax_row']}  centroid {g1['rig_a']['centroid_row']} "
          f"(n_eps {g1['rig_a']['n_episodes']}, cy mean {p2['meta']['rig_A_cy_mean']})")
    print(f"rig B: argmax row {g1['rig_b']['argmax_row']}  centroid {g1['rig_b']['centroid_row']} "
          f"(n_eps {g1['rig_b']['n_episodes']}, cy mean {p2['meta']['rig_B_cy_mean']})")
    print(f"offset: argmax {g1['argmax_offset_rows_256space']} rows, "
          f"centroid {g1['centroid_offset_rows_256space']} rows (256-space)")
    print(f"note  : {g1['interpretation']}")

    hdr("P2 — G2/G4: WITHIN-rig vs CROSS-rig speed R2 (the geometry-vs-appearance discriminator)")
    print(f"{'arm':20s} {'A->A':>9s} {'B->B':>9s} {'B->A':>9s} {'A->B':>9s} "
          f"{'drop on A':>10s} {'drop on B':>10s}")
    for a, r in p2["G2_G4_transfer"].items():
        c = r["cells"]
        dA = r.get("paired_cross_minus_within_on_A", {})
        dB = r.get("paired_cross_minus_within_on_B", {})
        print(f"{a:20s} {c['A->A']['r2_speed']:+9.4f} {c['B->B']['r2_speed']:+9.4f} "
              f"{c['B->A']['r2_speed']:+9.4f} {c['A->B']['r2_speed']:+9.4f} "
              f"{dA.get('delta', float('nan')):+10.4f} {dB.get('delta', float('nan')):+10.4f}")
    print("\npaired intervals (cross-rig minus within-rig, SAME held-out rows):")
    for a, r in p2["G2_G4_transfer"].items():
        for tgt in ("A", "B"):
            d = r.get(f"paired_cross_minus_within_on_{tgt}")
            if d:
                print(f"  {a:20s} on rig {tgt}: {d['delta']:+.4f} "
                      f"[{d['lo']:+.4f},{d['hi']:+.4f}] {sepmark(d)}  n_ep={d['n_episodes']}")

    if "G3_vertical_shift" in p2:
        hdr("P2 — G3: does a PURE GEOMETRIC vertical shift reproduce the collapse?")
        g3 = p2["G3_vertical_shift"]
        print(g3["note"])
        print(g3.get("reference_scale", ""))
        for aname, r in g3.get("arms", {}).items():
            tag = " ⛔ VOID — " + r["void_reason"] if r["VOID"] else ""
            print(f"\n  {aname} (fitted on {r['fitted_on']}), baseline R2 "
                  f"{r['baseline_r2_shift0']:+.4f}{tag}")
            for row in r["sweep"]:
                lost = row["frac_of_baseline_lost"]
                print(f"    shift {row['shift_rows']:3d} rows -> speed R2 "
                      f"{row['r2_speed']:+.5f}"
                      + ("" if lost is None else f"   ({lost*100:5.1f} % of baseline lost)"))

# --------------------------------------------------------------------------- #
p3 = load("results_p3_sitclf.json")
if p3:
    hdr("P3 — the scenario classifier: is 'vision-only' riding APPEARANCE -> SPEED -> the "
        "ego-derived LABEL?")
    sr = p3["speed_readability"]
    print(f"the shortcut's FIRST HOP on this substrate (clip-disjoint held out):")
    print(f"  still 32x32 frame -> speed  R2 {sr['still32_speed_r2_heldout']:+.4f}")
    print(f"  frozen v1 latent  -> speed  R2 {sr['latent_speed_r2_heldout']:+.4f}")
    print(f"\n⚠️ ap_lift = AP / base_rate, so CHANCE = 1.0 (not 0). 'excess' = lift - 1.")
    print(f"⛔ the permuted-feature null is DEGENERATE for 1-feature arms (AP is rank-based) "
          f"and decides nothing.\n")
    for s, r in p3["situations"].items():
        if r.get("status") == "UNPOWERED":
            print(f"{s}: UNPOWERED (n_heldout {r['n_heldout']}, n_pos {r['n_pos_heldout']})")
            continue
        print(f"--- {s} --- n_heldout {r['n_heldout']}, n_pos {r['n_pos_heldout']}, "
              f"base rate {r['base_rate_heldout']:.5f}")
        print(f"  {'arm':24s} {'feat':>6s} {'AP-lift':>9s} {'lo':>8s} {'hi':>8s} {'excess':>8s}")
        for a, rr in r["arms"].items():
            l = rr["ap_lift"]
            print(f"  {a:24s} {rr['n_features']:6d} {l['point']:+9.4f} {l['lo']:+8.4f} "
                  f"{l['hi']:+8.4f} {l['point']-1:+8.4f}")
        v = r["verdict"]
        print(f"  shortcut share (EXCESS lift)  : {v['shortcut_share_EXCESS_lift']:+.4f}")
        print(f"  shortcut share (raw, as prereg): "
              f"{v['shortcut_share_RAW_lift_as_preregistered']:+.4f}")
        print(f"  incremental AP-lift of vision OVER the true speed channel: "
              f"{v['incremental_ap_lift_of_vision_over_speed']:+.4f} "
              f"({v['incremental_share_of_img_excess_lift']:+.4f} of img's excess)")
        d = r.get("paired_img_vs_speed_from_appearance", {})
        print(f"  paired  img_latent - speed_from_appearance : {d.get('delta'):+.4f} "
              f"[{d.get('lo'):+.4f},{d.get('hi'):+.4f}] {sepmark(d)}")
        d = r.get("paired_speedplusimg_vs_speed", {})
        print(f"  paired  speed+img  - ego_speed_true        : {d.get('delta'):+.4f} "
              f"[{d.get('lo'):+.4f},{d.get('hi'):+.4f}] {sepmark(d)}")
        print(f"  >>> {v['PREREG_OUTCOME']}")

# --------------------------------------------------------------------------- #
p4 = load("results_p4_screen.json")
if p4:
    hdr("P4 — the promoted 0-GPU LATENT SCREEN, run on every latent we can reach")
    print(f"gates: {p4['meta']['gates']}")
    print(f"{'latent':46s} {'jitter':>8s} {'dcorr':>8s} {'accelR2':>8s} {'sigma':>7s} "
          f"{'cos100':>7s}  verdict")
    for n, r in p4["screens"].items():
        s = r["screens"]
        print(f"{n:46s} {s['jitter_ratio']['value']:8.2f} "
              f"{s['derivative_corr']['value']:+8.4f} "
              f"{(s['derived_accel_r2']['value'] if s['derived_accel_r2']['value'] is not None else float('nan')):+8.4f} "
              f"{s['speed_sigma_mps']['value']:7.3f} "
              f"{s['cos_adjacent_100ms']['value']:7.4f}  {r['verdict']} "
              f"({','.join(r['failed_screens']) or 'all gates passed'})")
    if "reproduction_check" in p4:
        rc = p4["reproduction_check"]
        print("\nreproduction of the reference measurement through the PROMOTED module:")
        print(f"  reference : {rc['reference']}")
        print(f"  measured  : {rc['measured_now']}")
        print(f"  {rc['note']}")

# --------------------------------------------------------------------------- #
if COMMA_REF.exists() and p1:
    hdr("CROSS-CORPUS CONTRAST — the same arms, the same encoder, two corpora")
    ref = json.load(open(COMMA_REF))
    ra = ref.get("arms", ref)
    print(f"{'arm':24s} {'comma2k19 speed R2':>20s} {'PhysicalAI speed R2':>21s}")
    for a in ("v1_window", "v1_centre", "pix32_centre_rbf", "pix32_centre",
              "stk32_centre", "pix8_tdiff_rbf", "mot8_window_rbf", "mot16_window_rbf"):
        c = ra.get(a, {}).get("r2", {}).get("speed", {}).get("point")
        p = p1["arms"].get(a, {}).get("r2", {}).get("speed", {}).get("point")
        if c is None and p is None:
            continue
        print(f"{a:24s} {(c if c is not None else float('nan')):+20.4f} "
              f"{(p if p is not None else float('nan')):+21.4f}")
