"""Emit every table in LATENT_BOTTLENECK.md FROM THE ARTIFACTS.

No number in the report is transcribed by hand — this script is what produced
them, and re-running it is how a reader checks the report against the JSON.

usage: python summarize.py > raw/summary_tables.txt
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def s(v, dp=4):
    return "n/a" if v is None else f"{float(v):+.{dp}f}"


def ci(d, dp=4):
    if not d:
        return "n/a"
    if "point" in d:
        return f"{d['point']:+.{dp}f} [{d['lo']:+.{dp}f}, {d['hi']:+.{dp}f}]"
    return (f"{d['delta']:+.{dp}f} [{d['lo']:+.{dp}f}, {d['hi']:+.{dp}f}]"
            + ("*" if d.get("separated") else ""))


def main():
    print("=" * 78)
    print("TABLE 1 — the D probe ladder (results_temporal_falsifier.json)")
    print("=" * 78)
    tf = load("results_temporal_falsifier.json")
    if tf is None:
        print("  [not yet produced]")
    else:
        print(f"{'arm':>20} {'kern':>6} {'F':>7} {'speed R2':>10} "
              f"{'d speed':>12} {'accel R2':>10} {'d accel':>26}")
        for n, a in tf["arms"].items():
            d = tf["paired_vs_control"].get(n, {})
            print(f"{n:>20} {a.get('kernel','-'):>6} "
                  f"{a.get('n_features',0):>7} "
                  f"{s(a['r2']['speed']['point']):>10} "
                  f"{s(d.get('speed',{}).get('delta')):>10}"
                  f"{'*' if d.get('speed',{}).get('separated') else ' '} "
                  f"{s(a['r2']['long_accel']['point']):>10} "
                  f"{ci(d.get('long_accel')):>26}")
        print()
        print("VERDICT:", json.dumps(tf["verdict"]["outcome"]))
        v = tf["verdict"]
        print("  oracle long_accel R2      :", v["oracle_long_accel_r2"])
        print("  temporal arms over floor  :",
              v["temporal_pixel_arms_separated_and_over_floor"])
        print("  temporal arms under floor :",
              v["temporal_pixel_arms_separated_but_under_floor"])
        print("  single-instant controls   :", v.get("single_instant_controls"))
        print("  v1 latent arms separated  :", v["v1_latent_arms_separated"])
        sp = v["speed_positive_control_separated"]
        print("  POSITIVE CONTROL (speed separated) per arm:")
        for k, val in sp.items():
            print(f"      {k:>20}: {val}")
        # the admissibility test the pre-registration fixed in advance
        best_pix = max((a["r2"]["speed"]["point"] for n, a in tf["arms"].items()
                        if a.get("substrate", "").startswith(("pix", "stk",
                                                              "mot"))),
                       default=None)
        best_lat = max((a["r2"]["speed"]["point"] for n, a in tf["arms"].items()
                        if a.get("substrate") == "Z"), default=None)
        print(f"\n  ADMISSIBILITY: best hand-built-substrate speed R2 "
              f"{s(best_pix)} vs best v1-latent speed R2 {s(best_lat)}")
        if best_pix is not None and best_lat is not None:
            print(f"  -> the learned latent is {best_lat / max(best_pix,1e-9):.1f}x "
                  f"the better SPEED reader")

    print()
    print("=" * 78)
    print("TABLE 2 — the mechanism (results_mechanism.json)")
    print("=" * 78)
    me = load("results_mechanism.json")
    if me is None:
        print("  [not yet produced]")
    else:
        lt, oc = me["latent_speed_track"], me["oracle_speed_track"]
        rows = [("speed R2 at centre", lt["speed_r2_centre"],
                 oc["speed_r2_centre"]),
                ("PRED within-window increment std",
                 lt["PRED_within_window_increment_std_mps2"],
                 oc["PRED_within_window_increment_std_mps2"]),
                ("gain (pred/true)", lt["gain_pred_over_true"],
                 oc["gain_pred_over_true"]),
                ("corr(pred incr, true incr)",
                 lt["corr_pred_vs_true_increment"],
                 oc["corr_pred_vs_true_increment"])]
        print(f"{'quantity':>40} {'LATENT':>12} {'ORACLE':>12}")
        for k, a, b in rows:
            print(f"{k:>40} {a:>12} {b:>12}")
        print(f"{'TRUE within-window increment std':>40} "
              f"{lt['TRUE_within_window_increment_std_mps2']:>12}")
        print(f"{'error-increment std':>40} "
              f"{lt['ERROR_increment_std_mps2']:>12}")
        print(f"{'derivative SNR (true/err)':>40} "
              f"{lt['snr_true_over_error_increment']:>12}")
        print(f"{'err autocorr lag1 in-window':>40} "
              f"{lt['err_autocorr_lag1_within_window']:>12}")
        print("\n  LATENT SELF-SIMILARITY (cosine):")
        for k, v in me["latent_self_similarity"].items():
            if k != "note":
                print(f"      {k:>44}: {v}")
        print("\n  DIRECT-INCREMENT REGRESSION (target = the true increments):")
        for tag in ("latent", "oracle"):
            d = me.get(f"direct_increment_{tag}")
            if d:
                print(f"      {tag:>8}: corr {d['corr_pred_vs_true_increment']:>10}"
                      f"   long_accel R2 {ci(d['r2_vs_CAN_long_accel'])}")
        sp = me.get("shared_perposition_readout")
        if sp:
            print("\n  SHARED PER-POSITION READOUT (one linear speed direction):")
            print(f"      centre speed R2  {sp['heldout_speed_r2_centre']}")
            print(f"      pred incr std    {sp['PRED_increment_std_mps2']} "
                  f"(true {sp['TRUE_increment_std_mps2']})")
            print(f"      corr(incr)       {sp['corr_pred_vs_true_increment']}")
            print(f"      long_accel R2    {ci(sp['r2_vs_CAN_long_accel_centred'])}")

    print()
    print("=" * 78)
    print("TABLE 3 — the precision ladder (results_precision_ladder.json)")
    print("=" * 78)
    pl = load("results_precision_ladder.json")
    if pl is None:
        print("  [not yet produced]")
    else:
        print("  NULL   ", ci(pl["target_stats"]["train_mean_null_r2"]))
        print("  ORACLE ", ci(pl["oracle_central_diff_true_speed"]))
        print(f"\n{'sigma':>8} {'white cd':>12} {'white SG':>12} "
              f"{'ar1 cd':>12} {'ar1 SG':>12}")
        wh = {r["sigma_mps"]: r for r in pl["noise_ladders"]["white"]}
        ar = {r["sigma_mps"]: r for r in pl["noise_ladders"]["ar1_matched"]}
        for sig in sorted(wh):
            print(f"{sig:>8} {wh[sig]['central_diff']['point']:>12.4f} "
                  f"{wh[sig]['best_savgol']['point']:>12.4f} "
                  f"{ar[sig]['central_diff']['point']:>12.4f} "
                  f"{ar[sig]['best_savgol']['point']:>12.4f}")
        print("\n  PRECISION TARGET (sigma at which derived accel clears R2):")
        print(json.dumps(pl["precision_target"], indent=6))
        rt = pl["real_latent_track"]
        print(f"\n  REAL latent-read track: centre R2 {rt['heldout_speed_r2_centre']}"
              f"  sigma {rt['heldout_speed_sigma_centre']} m/s"
              f"  err autocorr {rt['err_autocorr_lag1_within_window']}")
        print(f"      derived accel: central diff {ci(rt['central_diff'])}")
        print(f"                     best SG      "
              f"{ci(rt['best_savgol']['r2_ci'])}  "
              f"(stencil {rt['best_savgol']['half_window']}/"
              f"{rt['best_savgol']['poly_order']}, SELECTED ON HELDOUT = "
              f"upper bound)")
        om = pl["one_mechanism_test"]
        print(f"\n  ONE-MECHANISM TEST (matched sigma={om['matched_sigma_mps']} "
              f"rho={om['matched_rho']}):")
        print(f"      simulated {ci(om['simulated_r2'])}")
        print(f"      real      {ci(om['real_r2'])}")
        print(f"      paired d  {ci(om['paired_delta_sim_minus_real'])}")

    print()
    print("=" * 78)
    print("TABLE 4 — approach A cost (raw/temporal_kv_cost.json)")
    print("=" * 78)
    kv = load("raw/temporal_kv_cost.json")
    if kv is None:
        print("  [not yet produced]")
    else:
        for n, v in kv["presets"].items():
            t, a = v["today_KV64"], v["approachA_KVW"]
            print(f"{n:>12}  params {v['params_total']:>12,}  "
                  f"KV {t['kv_len']}->{a['kv_len']}  "
                  f"decoder {t['analytic_macs']['decoder_total']/1e9:>7.3f}->"
                  f"{a['analytic_macs']['decoder_total']/1e9:>7.3f} GMAC "
                  f"(x{v['ratio_decoder_macs']})  "
                  f"peak {t['peak_mib']:>7.1f}->{a['peak_mib']:>7.1f} MiB "
                  f"(x{v['ratio_peak_mib']})  "
                  f"+{v['approachA_added_params']} params "
                  f"({v['approachA_added_params_pct']}%)")
            print(f"{'':>14}attn share of a decoder layer: "
                  f"{t['analytic_macs']['attn_share_of_layer']} -> "
                  f"{a['analytic_macs']['attn_share_of_layer']}")
        print("\n  NOTE: wall-clock ratios in the JSON were taken under GPU "
              "contention and are NOT quotable; MACs / memory / params are.")

    print()
    print("=" * 78)
    print("TABLE 5 — the four families, headline arm (results_temporal_falsifier.json)")
    print("=" * 78)
    if tf is not None:
        pick = max(tf["arms"].items(),
                   key=lambda kv2: kv2[1]["r2"]["speed"]["point"]
                   if kv2[0] != "ORACLE_true_speed_window" else -9)
        name, arm = pick
        print(f"  headline arm = {name}   (best non-oracle speed reader)")
        ff = arm["four_families"]
        print(json.dumps(ff, indent=2)[:4000])


if __name__ == "__main__":
    main()
