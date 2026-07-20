import json
d = json.load(open("/root/taniteval/results/pathspeed_flagship-30k.json"))
print("n_windows", d["n_windows"], "n_eps", d["n_episodes"],
      "ckpt", d.get("ckpt_step"), "wall_s", d.get("wall_s"))
print("strata_meta", d["strata_meta"])
print("\n=== HEADLINE ===")
print(json.dumps(d["headline"], indent=2))
print("\n=== per-horizon DE (all windows), compounding ===")
print(d["per_horizon_de_all_m"], "ratio2s/0.5s", d["compounding_ratio_2s_over_0p5s"])

def row(name, blk):
    if "read" not in blk:
        return f"{name:26s} n={blk.get('n','?')} (skipped/small)"
    r = blk["read"]
    mt = blk["model"]["trajectory"]
    return (f"{name:26s} n={blk['model']['n']:4d} | mADE2s={r['model_ade_2s_m']:.3f} "
            f"ctrvADE2s={r['ctrv_ade_2s_m']:.3f} | long/lat@2s={r['model_long_rmse_2s_m']:.3f}/"
            f"{r['model_lat_rmse_2s_m']:.3f} (ctrv {r['ctrv_long_rmse_2s_m']:.3f}/{r['ctrv_lat_rmse_2s_m']:.3f}) "
            f"| longfrac={r['long_frac_of_2s_sqerr']:.2f} | spdbias={r['model_speed_bias_mps']:+.2f} "
            f"pgeo={mt['path_geometry_crosstrack_rmse_m']:.3f} | {r['dominant_component']} "
            f"| floor={blk['floor_ade_2s_m']:.3f}")

print("\n=== STRATA (long vs lat @2s) ===")
order = ["all", "slow_bottom50pct_speed", "fast_top10pct_speed",
         "sharp_top10pct_curvature", "straight_lt5deg",
         "speed_0-8mps", "speed_8-16mps", "speed_16-24mps", "speed_24-infmps"]
for k in order:
    if k in d["strata"]:
        print(row(k, d["strata"][k]))

print("\n=== FAST stratum per-horizon (compounding of long vs lat) ===")
fh = d["strata"]["fast_top10pct_speed"]["model"]["per_horizon"]
ch = d["strata"]["fast_top10pct_speed"]["ctrv"]["per_horizon"]
for t in ["0.5s", "1s", "1.5s", "2s"]:
    m, c = fh[t], ch[t]
    print(f"  t={t:5s} model de={m['de_at_h_m']:.3f} long={m['long_rmse_m']:.3f} "
          f"lat={m['lat_rmse_m']:.3f} spd_err={m['planned_speed_err_mps']:.2f} "
          f"spd_bias={m['planned_speed_bias_mps']:+.2f} gt_v={m['gt_speed_mps']:.1f} "
          f"pred_v={m['pred_speed_mps']:.1f} | ctrv de={c['de_at_h_m']:.3f} long={c['long_rmse_m']:.3f}")
mt = d["strata"]["fast_top10pct_speed"]["model"]["trajectory"]
print("\n  fast model trajectory:", json.dumps(mt, indent=2))
