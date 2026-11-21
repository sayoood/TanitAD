"""E0 — THE REAL SCOPE. Recompute L1_gate against the FULL 120.5 deg front field and split every
positive into RECOVERABLE-BY-CROP vs GENUINE OFF-FRONT RESIDUAL. Then re-derive C-EFF on the
residual, and price the widened crop honestly (it is not free).

Runs only if E1 PASSED. The split is reported as measured, in both directions.

Definitions (d = 3.0 m, the pre-registered conflict radius):
  cropped   L1_gate : trigger agent outside the 51.4 deg ENCODER CROP, inside cross_l/r, in conflict,
                      and no conflicting agent inside the crop            [substrate audit's label]
  full-front L1_gate: trigger agent outside the FULL 120.5 deg FRONT FRAME, inside cross_l/r, in
                      conflict, and no conflicting agent inside the full front frame   [E0's label]
"""
import glob
import json
import math
import sys

import numpy as np
import pandas as pd

S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
DR = r"C:\Users\Admin\tanitad-data\physicalai"
OUT = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Architecture & Inference\Implementation\incoming\2026-07-25-h2-e0-e1")
sys.path.insert(0, S)
from h2e_stats import cluster_rate, cluster_share, paired_cluster_lift   # noqa: E402

CL = "v_camera_cross_left_120fov"
CR = "v_camera_cross_right_120fov"
D = 3.0

A = pd.read_parquet(S + r"\h2e_heldout.parquet").dropna(subset=["dmin_cf", "ego_dv"])
A = A.assign(cf=A.dmin_cf <= D)
res = {"conflict_radius_m": D, "n_clips": int(A.clip_id.nunique()),
       "n_agent_frames": int(len(A))}
print(f"held-out {A.clip_id.nunique():,} clips / {len(A):,} agent-frames, d = {D} m")

A["S_crop"] = (A[CL] | A[CR]) & (~A.in_crop) & A.cf
A["S_full"] = (A[CL] | A[CR]) & (~A.in_front_full) & A.cf
A["F_crop"] = A.in_crop & A.cf
A["F_full"] = A.in_front_full & A.cf
A["ScL"] = A[CL] & (~A.in_crop) & A.cf
A["ScR"] = A[CR] & (~A.in_crop) & A.cf
A["SfL"] = A[CL] & (~A.in_front_full) & A.cf
A["SfR"] = A[CR] & (~A.in_front_full) & A.cf

# ------------------------------------------------------------------ agent-level trigger split
trig = A[A.S_crop]
rec_ag = trig.in_front_full
sh_ag = cluster_share(trig.in_front_full.astype(float).to_numpy(),
                      np.ones(len(trig)), trig.clip_id.to_numpy())
res["agent_level"] = {"n_trigger_agent_frames": int(len(trig)),
                      "recoverable_by_crop": sh_ag,
                      "n_recoverable": int(rec_ag.sum()),
                      "n_residual": int((~rec_ag).sum()),
                      "az_abs_recoverable_p50": round(float(trig.az[rec_ag].abs().median()), 2),
                      "az_abs_residual_p50": round(float(trig.az[~rec_ag].abs().median()), 2),
                      "az_abs_residual_p10": round(float(trig.az[~rec_ag].abs().quantile(.1)), 2)}
print("\n=== AGENT-LEVEL: conflicting agents that trigger L1_gate under the 51.4 deg crop ===")
print(f"  trigger agent-frames                      : {len(trig):,}")
print(f"  RECOVERABLE-BY-CROP (in full front frame) : {int(rec_ag.sum()):,} = "
      f"{sh_ag['share']:.1%}  CI [{sh_ag['lo']:.1%}, {sh_ag['hi']:.1%}]")
print(f"  GENUINE OFF-FRONT RESIDUAL                : {int((~rec_ag).sum()):,} = "
      f"{1-sh_ag['share']:.1%}")
print(f"  |az| recoverable triggers p50 {trig.az[rec_ag].abs().median():.1f} deg | "
      f"|az| residual triggers p50 {trig.az[~rec_ag].abs().median():.1f} deg "
      f"(p10 {trig.az[~rec_ag].abs().quantile(.1):.1f})")

# ------------------------------------------------------------------ frame-level gate recompute
g = A.groupby(["clip_id", "gi"], observed=True).agg(
    Sc=("S_crop", "sum"), Sf=("S_full", "sum"), Fc=("F_crop", "sum"), Ff=("F_full", "sum"),
    ScL=("ScL", "sum"), ScR=("ScR", "sum"), SfL=("SfL", "sum"), SfR=("SfR", "sum"),
    dv=("ego_dv", "first"), junction=("junction", "first"),
    lane_change=("lane_change", "first")).reset_index().sort_values(["clip_id", "gi"])
g["gate_crop"] = (g.Sc > 0) & (g.Fc == 0)
g["gate_full"] = (g.Sf > 0) & (g.Ff == 0)
g["resp"] = g.dv <= -1.0
g["recoverable"] = g.gate_crop & ~g.gate_full
g["residual"] = g.gate_crop & g.gate_full

sh = cluster_share(g.recoverable.astype(float).to_numpy(),
                   g.gate_crop.astype(float).to_numpy(), g.clip_id.to_numpy())
why_trigger = int((g.gate_crop & (g.Sf == 0)).sum())
why_redund = int((g.gate_crop & (g.Sf > 0) & (g.Ff > 0)).sum())
res["frame_level"] = {
    "n_frames": int(len(g)),
    "n_gate_crop": int(g.gate_crop.sum()),
    "gate_crop_rate": cluster_rate(g.gate_crop.astype(float), g.clip_id.to_numpy()),
    "n_gate_full": int(g.gate_full.sum()),
    "gate_full_rate": cluster_rate(g.gate_full.astype(float), g.clip_id.to_numpy()),
    "recoverable_by_crop_share": sh,
    "n_recoverable": int(g.recoverable.sum()),
    "n_residual": int(g.residual.sum()),
    "recoverable_because_trigger_now_visible": why_trigger,
    "recoverable_because_wide_field_shows_a_conflicting_agent": why_redund,
    "n_episodes_gate_crop_positive": int(g.groupby("clip_id").gate_crop.max().sum()),
    "n_episodes_gate_full_positive": int(g.groupby("clip_id").gate_full.max().sum())}
print("\n=== FRAME-LEVEL: L1_gate positives under the 51.4 deg crop, re-scored on 120.5 deg ===")
print(f"  frames total                       : {len(g):,}")
print(f"  L1_gate+ under the 51.4 deg crop   : {int(g.gate_crop.sum()):,} ({g.gate_crop.mean():.3%})")
print(f"  -> RECOVERABLE-BY-CROP             : {int(g.recoverable.sum()):,} = "
      f"{sh['share']:.1%}  CI [{sh['lo']:.1%}, {sh['hi']:.1%}]")
print(f"       trigger now visible in wide front       : {why_trigger:,}")
print(f"       wide front shows a conflicting agent (iv): {why_redund:,}")
print(f"  -> GENUINE OFF-FRONT RESIDUAL      : {int(g.residual.sum()):,} = {1-sh['share']:.1%}")
print(f"  residual gate rate (of all frames) : {g.gate_full.mean():.3%}")
print(f"  episodes gate+ : crop {int(g.groupby('clip_id').gate_crop.max().sum()):,} | "
      f"residual {int(g.groupby('clip_id').gate_full.max().sum()):,} of {g.clip_id.nunique():,}")

# ------------------------------------------------------------------ strata
res["strata"] = {}
print("\n=== STRATA (kinematic detectors, situ_full.py thresholds verbatim) ===")
for name in ("junction", "lane_change"):
    for lab, mask in (("in-stratum", g[name].to_numpy()), ("out-of-stratum", ~g[name].to_numpy())):
        sub = g[mask]
        if not len(sub):
            continue
        entry = {"n_frames": int(len(sub)), "n_clips": int(sub.clip_id.nunique()),
                 "n_gate_crop": int(sub.gate_crop.sum()),
                 "gate_crop_rate": round(float(sub.gate_crop.mean()), 5),
                 "n_gate_full": int(sub.gate_full.sum()),
                 "gate_full_rate": round(float(sub.gate_full.mean()), 5),
                 "n_recoverable": int(sub.recoverable.sum()),
                 "n_residual": int(sub.residual.sum())}
        if sub.gate_crop.sum() >= 10:
            ss = cluster_share(sub.recoverable.astype(float).to_numpy(),
                               sub.gate_crop.astype(float).to_numpy(),
                               sub.clip_id.to_numpy(), n_boot=1000)
            entry["recoverable_by_crop_share"] = ss
            print(f"  {name:12s} {lab:15s} frames {len(sub):8,} clips {sub.clip_id.nunique():5,}  "
                  f"gate+ {int(sub.gate_crop.sum()):5,} ({sub.gate_crop.mean():.2%})  "
                  f"recoverable {ss['share']:.1%} [{ss['lo']:.1%}, {ss['hi']:.1%}]")
        else:
            print(f"  {name:12s} {lab:15s} frames {len(sub):8,}  gate+ "
                  f"{int(sub.gate_crop.sum()):5,}  (too few for a CI)")
        res["strata"][f"{name}__{lab}"] = entry


# ------------------------------------------------------------------ C-EFF re-derivation
def dilate(flag, clip, k):
    """+/- k frames temporal hysteresis, within an episode."""
    o = flag.to_numpy().copy()
    pos = pd.Series(np.arange(len(flag)), index=np.asarray(clip))
    for _, idx in pos.groupby(level=0):
        i = idx.to_numpy()
        v = o[i].copy()
        w = v.copy()
        for sh_ in range(1, k + 1):
            w[:-sh_] |= v[sh_:]
            w[sh_:] |= v[:-sh_]
        o[i] = w
    return pd.Series(o, index=flag.index)


g["Lc"] = (g.ScL > 0) & (g.Fc == 0)
g["Rc"] = (g.ScR > 0) & (g.Fc == 0)
g["Lf"] = (g.SfL > 0) & (g.Ff == 0)
g["Rf"] = (g.SfR > 0) & (g.Ff == 0)
eff = []
print("\n=== C-EFF re-derived: CROPPED definition vs RESIDUAL (full-front) definition ===")
print(f"{'definition':<22}{'policy':<16}{'left':>8}{'right':>8}{'either':>9}"
      f"{'cams/frame':>12}{'vs 7':>8}{'vs 3':>8}")
for defname, (lc, rc) in (("cropped 51.4 deg", ("Lc", "Rc")),
                          ("RESIDUAL 120.5 deg", ("Lf", "Rf"))):
    for pol, k in (("instantaneous", 0), ("+/-1.0 s hyst", 10), ("+/-2.0 s hyst", 20)):
        L = dilate(g[lc], g.clip_id, k) if k else g[lc]
        R = dilate(g[rc], g.clip_id, k) if k else g[rc]
        cams = 1 + L.astype(int) + R.astype(int)
        c = float(cams.mean())
        eff.append(dict(definition=defname, policy=pol, left=round(float(L.mean()), 5),
                        right=round(float(R.mean()), 5), either=round(float((L | R).mean()), 5),
                        cams_per_frame=round(c, 4), saved_vs_7=round(1 - c / 7, 4),
                        saved_vs_3=round(1 - c / 3, 4),
                        cams_per_frame_ci=cluster_rate(cams.astype(float),
                                                       g.clip_id.to_numpy(), n_boot=1000)))
        print(f"{defname:<22}{pol:<16}{L.mean():>8.2%}{R.mean():>8.2%}{(L|R).mean():>9.2%}"
              f"{c:>12.4f}{1-c/7:>8.1%}{1-c/3:>8.1%}")
res["efficiency"] = eff

# --------------------------------------------- what a widened crop actually COSTS (measured)
CIf = pd.concat([pd.read_parquet(f) for f in
                 glob.glob(DR + r"\calibration\camera_intrinsics\*.parquet")]).reset_index()
fw = CIf[CIf.camera_name == "camera_front_wide_120fov"]
F_REF, SIZE = 266.0, 256
CANON_HALF = math.atan((SIZE / 2) / F_REF)


def poly_r(row, th):
    r = 0.0
    for c in reversed([row.fw_poly_0, row.fw_poly_1, row.fw_poly_2, row.fw_poly_3, row.fw_poly_4]):
        r = r * th + c
    return r


ratios = []
for _, row in fw.iterrows():
    c_half = poly_r(row, CANON_HALF)
    rx = min(float(row.cx), float(row.width) - float(row.cx))
    if c_half > 0:
        ratios.append(rx / c_half)
ratios = np.array(ratios)
r50 = float(np.median(ratios))
res["widened_crop_cost"] = {
    "n_clips_measured": int(len(ratios)),
    "px_ratio_full_over_crop_p50": round(r50, 4),
    "px_ratio_p05": round(float(np.percentile(ratios, 5)), 4),
    "px_ratio_p95": round(float(np.percentile(ratios, 95)), 4),
    "option_A_same_input_px": {"input_px": SIZE, "angular_resolution_penalty_x": round(r50, 3)},
    "option_B_same_angular_res_horizontal_only": {"input_px_w": int(round(SIZE * r50)),
                                                  "input_px_h": SIZE,
                                                  "token_cost_x": round(r50, 3)},
    "option_C_same_angular_res_square": {"input_px": int(round(SIZE * r50)),
                                         "token_cost_x": round(r50 ** 2, 3)}}
print("\n=== what widening the crop COSTS (MEASURED from the real f-theta polynomials) ===")
print(f"  native px covering the full front field / px in the 51.4 deg crop: p50 {r50:.2f}x  "
      f"(p05 {np.percentile(ratios,5):.2f}, p95 {np.percentile(ratios,95):.2f}, n={len(ratios)})")
print(f"  A) keep 256 px input   -> angular resolution {r50:.2f}x COARSER, compute unchanged")
print(f"  B) keep angular res, widen horizontally -> {int(round(SIZE*r50))}x256, "
      f"~{r50:.2f}x front-encoder tokens, ALWAYS ON")
print(f"  C) keep angular res, square             -> {int(round(SIZE*r50))}^2, "
      f"~{r50**2:.1f}x front-encoder tokens, ALWAYS ON")

# ------------------------------------------- does the residual gate still carry decision-relevance?
L = paired_cluster_lift(g.resp.to_numpy(), g.gate_full.to_numpy(), g.clip_id.to_numpy())
res["residual_gate_lift"] = L
print("\n=== decision-relevance of the RESIDUAL (full-front) gate, d = 3.0 m ===")
print(f"  lift {L['lift']:.2f}x  CI [{L['lo']:.2f}, {L['hi']:.2f}]  n+={L['n_pos']:,}  "
      f"episodes {L['n_episodes']:,}")

json.dump(res, open(OUT + r"\e0_split.json", "w"), indent=2)
print(f"\nwrote {OUT}\\e0_split.json")
