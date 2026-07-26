# -*- coding: utf-8 -*-
"""Re-adjudicate every frozen control row.

Two instruments:
  (1) HIGHER-N SIBLING  -- where a same-probe run at larger n exists, pair them.
  (2) POWER CEILING     -- MDE = the row own 95% half-width; compare it to
                           (a) the max possible effect (bounded metrics), and
                           (b) the reference leak the control exists to catch.
"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
B = r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub"
L = lambda p: json.load(open(B + p, encoding="utf-8"))

S3_NP  = L("/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-s3/s3_blind_baseline_primary.json")
S3_NP8 = L("/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-s3/s3_blind_baseline_sens_h8.json")
S3_P   = L("/Architecture & Inference/Implementation/incoming/2026-07-26-s3-decision-grade/s3_blind_baseline_parity_primary.json")
S3_P8  = L("/Architecture & Inference/Implementation/incoming/2026-07-26-s3-decision-grade/s3_blind_baseline_parity_sens_h8.json")
S3_P2  = L("/Benchmarks & Eval/Implementation/incoming/2026-07-26-pod2-eval-host/artifacts/s3_blind_baseline_pod2_parity_primary.json")
S1     = L("/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-gates/S1_RESULTS.json")

out = {
    "generated": "2026-07-26",
    "estimator": "paired episode-cluster bootstrap (taniteval/ci.py) B=2000, unit = episode cluster",
    "MDE_definition": "the row own 95% half-width -- the smallest effect it could have separated at its own n",
    "families": {},
}

# ---------- S3 LEAK + CLOCK: a higher-n sibling exists (parity, n=520/558) ----------
leak = []
for axis in ("lat", "lon"):
    for node in ("paired_leak_B2_minus_B1", "paired_leak_B3_minus_B1", "paired_clock_B4_minus_B3"):
        for tag, lo_src, hi_src, hi2 in (("H12", S3_NP, S3_P, S3_P2), ("H8", S3_NP8, S3_P8, None)):
            a = lo_src[axis][node]
            b = hi_src[axis][node]
            c = hi2[axis][node] if hi2 else None
            mde_a = (a["hi"] - a["lo"]) / 2
            mde_b = (b["hi"] - b["lo"]) / 2
            rec = {
                "axis": axis, "node": node, "horizon_s": 12.0 if tag == "H12" else 8.0,
                "original": {"delta": a["delta"], "ci95": [a["lo"], a["hi"]], "separated": a["separated"],
                             "n_episodes": a["n_episodes"], "n_windows": a["n_windows"],
                             "MDE": round(mde_a, 4), "corpus": "NON-PARITY dev-box (400 train / 100 val)"},
                "readjudicated": {"delta": b["delta"], "ci95": [b["lo"], b["hi"]], "separated": b["separated"],
                                  "n_episodes": b["n_episodes"], "n_windows": b["n_windows"],
                                  "MDE": round(mde_b, 4),
                                  "corpus": "PARITY e438721ae894 train / 0c5f7dac3b11 val (2376 / 600)"},
                "power_gain_x": round(mde_a / mde_b, 2),
                "verdict_changed": a["separated"] != b["separated"],
                "sign_flipped": (a["delta"] > 0) != (b["delta"] > 0),
                "control_was_VOID_at_original_n": abs(b["delta"]) < mde_a,
            }
            if c:
                rec["independent_rerun_pod2"] = {
                    "delta": c["delta"], "ci95": [c["lo"], c["hi"]], "separated": c["separated"],
                    "agrees_on_verdict": c["separated"] == b["separated"],
                    "point_gap": round(abs(c["delta"] - b["delta"]), 4)}
            leak.append(rec)
out["families"]["S3_LEAK_AND_CLOCK"] = leak

# ---------- S3 BLIND vs CHANCE ----------
bvc = []
for axis in ("lat", "lon"):
    for arm in ("B1_sensor_only", "B2_plus_route", "B3_FULL_CONDITIONING", "B4_plus_clock"):
        a = S3_NP[axis]["arms"][arm]
        b = S3_P[axis]["arms"][arm]
        ka = [k for k in a if k.endswith("qwk_ci")]
        if not ka:
            continue
        ai, bi = a[ka[0]], b[ka[0]]
        aq = a.get("blind_qwk", (ai["lo"] + ai["hi"]) / 2)
        bq = b.get("blind_qwk", (bi["lo"] + bi["hi"]) / 2)
        asep = not (ai["lo"] <= 0 <= ai["hi"])
        bsep = not (bi["lo"] <= 0 <= bi["hi"])
        bvc.append({
            "axis": axis, "arm": arm,
            "original": {"qwk": round(aq, 4), "ci95": [ai["lo"], ai["hi"]], "separated_from_chance": asep,
                         "n_episodes": S3_NP[axis]["n_test_episodes"], "MDE": round((ai["hi"] - ai["lo"]) / 2, 4)},
            "readjudicated": {"qwk": round(bq, 4), "ci95": [bi["lo"], bi["hi"]], "separated_from_chance": bsep,
                              "n_episodes": S3_P[axis]["n_test_episodes"], "MDE": round((bi["hi"] - bi["lo"]) / 2, 4)},
            "verdict_changed": asep != bsep})
out["families"]["S3_BLIND_VS_CHANCE"] = bvc

# ---------- S1: no higher-n sibling exists. POWER-CEILING PROOF instead. ----------
s1 = []
for v in ("E", "H", "NOGOAL"):
    fw = S1["firewall"][v]
    p = fw["blind_vs_majority_paired"]
    mde = (p["hi"] - p["lo"]) / 2
    headroom = fw["acc_ceiling"] - fw["acc_major"]   # max attainable value of (blind - majority)
    s1.append({
        "variant": v, "name": fw["name"], "verdict_as_published": fw["verdict"],
        "delta_blind_minus_majority": p["delta"], "ci95": [p["lo"], p["hi"]], "separated": p["separated"],
        "n_decision_points": p["n_windows"], "n_scene_clusters": p["n_episodes"],
        "MDE": round(mde, 4),
        "max_possible_effect": round(headroom, 4),
        "MDE_as_pct_of_max_possible": round(100 * mde / headroom, 1),
        "CAN_THIS_TEST_EVER_FIRE": bool(mde < headroom),
        "acc_blind": fw["acc_blind"], "acc_major": fw["acc_major"], "acc_chance": fw["acc_chance"],
        "blind_minus_chance": round(fw["acc_blind"] - fw["acc_chance"], 4),
        "R1_threshold_rule": "acc_blind >= 0.98 * ceiling => REFUSE",
        "R1_gap_to_threshold": round(0.98 - fw["acc_blind"], 4)})
out["families"]["S1_BLIND_VS_MAJORITY"] = s1

# ---------- SHUFFLE (lead-state gate): capacity decomposition ----------
lg = L("/Data Engineering/Implementation/incoming/2026-07-21-lead-state-gate/lead_gate_result.json")
sh = []
for cell in ("ridge|canonical", "gbm|canonical"):
    c = lg[cell]
    t = c["paired_mae_A_minus_B"]
    s = c["paired_mae_A_minus_B_shuf"]
    cap = -s["delta"]              # MAE cost of 5 extra PURE-NOISE features
    sig = t["delta"] + cap         # capacity-corrected lead-state signal (ADDITIVITY ASSUMED)
    rel = 100 * sig / c["mae_A"]["mean"]
    sh.append({
        "cell": cell,
        "treatment_A_minus_B": {"delta": t["delta"], "ci95": [t["lo"], t["hi"]],
                                "separated": t["separated"], "MDE": round((t["hi"] - t["lo"]) / 2, 5)},
        "shuffle_control_A_minus_B_shuf": {"delta": s["delta"], "ci95": [s["lo"], s["hi"]],
                                           "separated": s["separated"], "MDE": round((s["hi"] - s["lo"]) / 2, 5)},
        "control_direction_correct_shuffled_features_HURT": bool(s["delta"] < 0),
        "abs_control_over_abs_treatment": round(abs(s["delta"]) / abs(t["delta"]), 3),
        "mae_A": c["mae_A"]["mean"], "mae_B": c["mae_B"]["mean"], "mae_B_shuf": c["mae_B_shuf"]["mean"],
        "capacity_cost_of_5_noise_features": round(cap, 5),
        "capacity_corrected_signal_ESTIMATED": round(sig, 5),
        "capacity_corrected_rel_reduction_pct_ESTIMATED": round(rel, 3),
        "published_rel_reduction_pct": round(100 * c["rel_reduction_B"]["point"], 3),
        "preregistered_FAIL_bar_pct": 5.0,
        "still_FAILS_after_correction": bool(rel < 5.0),
        "n_episodes": c["n_test_episodes"]})
out["families"]["SHUFFLE_lead_state_gate"] = sh

json.dump(out, open("readjudication.json", "w", encoding="utf-8"), indent=1)

print("=== S3 LEAK/CLOCK: rows whose VERDICT CHANGED ===")
for r in leak:
    if r["verdict_changed"]:
        o, n = r["original"], r["readjudicated"]
        print("  %s.%s [H=%.0fs]  n=%d->%d  %+.4f %s -> %+.4f %s   power x%.2f  signflip=%s  VOID_at_orig_n=%s"
              % (r["axis"], r["node"], r["horizon_s"], o["n_episodes"], n["n_episodes"],
                 o["delta"], "SEP" if o["separated"] else "not-sep",
                 n["delta"], "SEP" if n["separated"] else "not-sep",
                 r["power_gain_x"], r["sign_flipped"], r["control_was_VOID_at_original_n"]))
print("\n=== S3 LEAK/CLOCK: rows that HELD ===")
for r in leak:
    if not r["verdict_changed"]:
        o, n = r["original"], r["readjudicated"]
        print("  %s.%s [H=%.0fs]  n=%d->%d  %+.4f -> %+.4f  both %s  power x%.2f"
              % (r["axis"], r["node"], r["horizon_s"], o["n_episodes"], n["n_episodes"],
                 o["delta"], n["delta"], "SEP" if n["separated"] else "not-sep", r["power_gain_x"]))
print("\n=== S3 BLIND vs CHANCE ===")
for r in bvc:
    print("  %s.%-22s %+.4f %-8s -> %+.4f %-8s  changed=%s"
          % (r["axis"], r["arm"], r["original"]["qwk"],
             "SEP" if r["original"]["separated_from_chance"] else "not-sep",
             r["readjudicated"]["qwk"],
             "SEP" if r["readjudicated"]["separated_from_chance"] else "not-sep", r["verdict_changed"]))
print("\n=== S1: can the firewall EVER fire? ===")
for r in s1:
    print("  %-7s n_cl=%2d  MDE=%.4f  max_possible=%.4f  MDE=%.1f%% of max  CAN_FIRE=%s  blind-chance=%+.4f"
          % (r["variant"], r["n_scene_clusters"], r["MDE"], r["max_possible_effect"],
             r["MDE_as_pct_of_max_possible"], r["CAN_THIS_TEST_EVER_FIRE"], r["blind_minus_chance"]))
print("\n=== SHUFFLE: capacity decomposition (lead-state gate) ===")
for r in sh:
    print("  %-16s treat=%+.5f  shuf_ctl=%+.5f  |ctl|/|treat|=%.2f  capacity_cost=%+.5f  corrected=%+.5f (%.2f%%)  still_FAILS=%s"
          % (r["cell"], r["treatment_A_minus_B"]["delta"], r["shuffle_control_A_minus_B_shuf"]["delta"],
             r["abs_control_over_abs_treatment"], r["capacity_cost_of_5_noise_features"],
             r["capacity_corrected_signal_ESTIMATED"], r["capacity_corrected_rel_reduction_pct_ESTIMATED"],
             r["still_FAILS_after_correction"]))
