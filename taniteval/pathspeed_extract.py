import json


def L(k):
    return json.load(open(f"/root/taniteval/results/pathspeed_{k}.json"))


ra, fl = L("refa-dynin-30k"), L("flagship-30k")


def ph(d, hz, stratum="all"):
    return d["strata"][stratum]["model"]["per_horizon"][hz]


def tr(d, stratum="all"):
    return d["strata"][stratum]["model"]["trajectory"]


print("=== META ===")
for name, d in [("refa-dynin-30k", ra), ("flagship-30k", fl)]:
    print(f"{name}: n_windows={d['n_windows']} n_eps={d['n_episodes']} "
          f"compounding(2s/0.5s)={d['compounding_ratio_2s_over_0p5s']}")

print("\n=== ALL-WINDOWS per-horizon (model): longRMSE / latRMSE / longBias / spdBias / de / longFrac ===")
h = ("hz", "REFA_long", "FL_long", "REFA_lat", "FL_lat", "REFA_longbias",
     "FL_longbias", "REFA_spdbias", "FL_spdbias", "REFA_de", "FL_de", "REFA_lfrac", "FL_lfrac")
fmt = "{:>5} {:>9} {:>8} {:>8} {:>7} {:>13} {:>11} {:>12} {:>10} {:>8} {:>8} {:>10} {:>8}"
print(fmt.format(*h))
for hz in ra["strata"]["all"]["model"]["per_horizon"]:
    r, f = ph(ra, hz), ph(fl, hz)
    print(fmt.format(
        hz, r["long_rmse_m"], f["long_rmse_m"], r["lat_rmse_m"], f["lat_rmse_m"],
        r["long_bias_m"], f["long_bias_m"],
        r["planned_speed_bias_mps"], f["planned_speed_bias_mps"],
        r["de_at_h_m"], f["de_at_h_m"], r["long_frac_of_sqerr"], f["long_frac_of_sqerr"]))

print("\n=== ALL-windows trajectory: speed_bias / along_progress_bias / path_geom_crosstrack / ade2s ===")
for name, d in [("refa-dynin-30k", ra), ("flagship-30k", fl)]:
    t = tr(d)
    print(f"{name}: speed_bias={t['speed_bias_mps']} along_progress_bias(OVERSHOOT)={t['along_track_progress_bias_m']} "
          f"along_progress_err={t['along_track_progress_err_m']} "
          f"path_geom_crosstrack_rmse={t['path_geometry_crosstrack_rmse_m']} "
          f"long_frac_2s={t['long_frac_of_sqerr_2s']} ade2s={t['ade_2s_m']}")

print("\n=== PER-SPEED / CURV strata: 2s longRMSE / latRMSE / spdbias(traj) / overshoot / longfrac ===")
for s in ra["strata"]:
    blk = ra["strata"].get(s, {})
    if "model" not in blk or "trajectory" not in blk["model"]:
        print(f"[{s}] (skipped: {blk.get('note','no data')})")
        continue
    r2, rt = ph(ra, "2s", s), tr(ra, s)
    n = blk["model"]["n"]
    line = (f"[{s:26s} n={n:>4}] REFA long={r2['long_rmse_m']:>6} lat={r2['lat_rmse_m']:>6} "
            f"spdbias={rt['speed_bias_mps']:>6} overshoot={rt['along_track_progress_bias_m']:>6} "
            f"longfrac={r2['long_frac_of_sqerr']:>5} ade2s={rt['ade_2s_m']:>6} pgeom={rt['path_geometry_crosstrack_rmse_m']:>5}")
    if s in fl["strata"] and "trajectory" in fl["strata"][s].get("model", {}):
        f2, ft = ph(fl, "2s", s), tr(fl, s)
        line += (f"  || FLAG long={f2['long_rmse_m']:>6} spdbias={ft['speed_bias_mps']:>6} "
                 f"overshoot={ft['along_track_progress_bias_m']:>6} ade2s={ft['ade_2s_m']:>6}")
    print(line)

print("\n=== HEADLINE verdicts ===")
for name, d in [("refa-dynin-30k", ra), ("flagship-30k", fl)]:
    print(f"[{name}] {d['headline'].get('verdict', d['headline'])}")
