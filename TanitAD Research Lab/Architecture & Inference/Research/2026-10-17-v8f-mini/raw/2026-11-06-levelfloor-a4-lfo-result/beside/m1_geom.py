"""Mechanism instrument for the A4 read (descriptive, after the read; kappa 1 is already the A4 card's committed next arm): inside the M1 ticks of the
LFO075 rollouts, how often can a floor at kappa * v_log put the target above the 5.2 m/s M1 threshold, and how often can the 1.0 m/s^2 ramp reach 5.2 m/s within the executed 2 s."""
import importlib.util, json, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("lf_read", "E:/Projects/TanitAD-artifacts/dagland/motion_c/lf_read/code/lf_read.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
VMIN = lf.VMIN
out = {}
for tag in ("RILFO075_s0", "RILFO075_s1"):
    t = np.load(f"F:/TanitAD-scratch/motion_c/lfo_read/out/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
    v0 = np.asarray(t["v0"], np.float64)
    vl, ok = lf.logged_speed(t["gt"], t["gt_ok"])
    m1 = lf.m1_ticks(v0, vl, ok)
    bind = np.asarray(t["lf_bind"], bool)
    r = {"VMIN": VMIN, "n_m1_ticks": int(m1.sum()),
         "share_m1_with_0.75_vlog_below_VMIN": float(np.mean(0.75 * vl[m1] < VMIN)),
         "share_m1_with_1.00_vlog_below_VMIN": float(np.mean(1.00 * vl[m1] < VMIN)),
         "share_m1_ramp_cannot_reach_VMIN_in_2s": float(np.mean(np.maximum(v0[m1], 0) + 1.0 * 2.0 < VMIN)),
         "v0_m1_quantiles_10_50_90": [float(x) for x in np.quantile(v0[m1], [0.1, 0.5, 0.9])],
         "vlog_m1_quantiles_10_50_90": [float(x) for x in np.quantile(vl[m1], [0.1, 0.5, 0.9])],
         "bind_rate_m1": float(bind[m1].mean()),
         "bind_rate_m1_where_0.75vlog_ge_VMIN": float(bind[m1 & (0.75 * np.nan_to_num(vl) >= VMIN)].mean()) if (m1 & (0.75 * np.nan_to_num(vl) >= VMIN)).any() else None}
    out[tag] = r
print(json.dumps(out, indent=1))
