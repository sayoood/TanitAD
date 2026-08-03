"""Thor P6 — rescore the banked gate windows with the CORRECTED four-family instrument.

0 GPU. Reads `~/thor_c3_windows.pt` (both arms, identical windows) and recomputes the
LONGITUDINAL / LATERAL families with the dt-derived `taniteval.four_families`, so the run reports
BOTH:
  * the pre-fix numbers, which is what every hub document currently quotes, and
  * the corrected numbers, which is what is physically true,
side by side with the exact correction factor. No number is silently replaced.

⛔ The PAIRED PRECISION DELTA is unaffected by the fix: both arms were on the identical grid, so a
common factor cancels in the difference and only the SCALE of the reported delta changes.
"""
import json
import os
import sys

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))

import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval import ci as CI  # noqa: E402

D = torch.load(os.path.expanduser("~/thor_c3_windows.pt"), map_location="cpu",
               weights_only=False)
A, B = D["A"], D["B"]
eid = A["eid"]
OUT = {"purpose": "the same 859 windows, scored on the WRONG grid and the RIGHT grid",
       "n_windows": len(eid), "n_episodes": len(set(eid)),
       "wp_steps": A["wp_steps"],
       "module_has_infer_dt": hasattr(FF, "infer_dt")}

# ⭐ NEGATIVE CONTROL FIRST — the physical quantity the episode already carries.
gtg = FF._seq_geometry(A["gt"].float(), 0.1)
gtc = FF._seq_geometry(A["gt"].float(), 0.5)
true_v = A["speed"].float()
OUT["NEGATIVE_CONTROL"] = {
    "what": "the ego's OWN recorded speed poses[:,3] vs the speed the instrument derives",
    "gt_ego_speed_mps_mean": round(float(true_v.mean()), 4),
    "seq_geometry_dt0.1_mps_mean": round(float(gtg["speed"].mean()), 4),
    "seq_geometry_dt0.5_mps_mean": round(float(gtc["speed"].mean()), 4),
    "ratio_dt0.1_over_truth": round(float(gtg["speed"].mean() / true_v.mean()), 4),
    "ratio_dt0.5_over_truth": round(float(gtc["speed"].mean() / true_v.mean()), 4),
    "verdict": ("the 0.5 s grid reproduces the recorded ego speed; the 0.1 s grid is 5x it. "
                "This is what makes the defect a MEASUREMENT, not an argument.")}

win_kwargs = {"wp_steps": A["wp_steps"], "dt_s": 0.1}
res = {}
for arm, W in (("A_fp32", A), ("B_opt", B)):
    w = {"pred": W["pred"], "gt": W["gt"], **win_kwargs}
    old = {"longitudinal": FF.longitudinal(W["pred"].float(), W["gt"].float(), 0.1),
           "lateral": FF.lateral(W["pred"].float(), W["gt"].float(), 0.1)}
    new = FF.all_families(w, prefer_dense=False)
    res[arm] = {"PRE_FIX_dt0.1_WRONG": {k: old[k] for k in ("longitudinal", "lateral")},
                "CORRECTED_dt0.5": {k: new[k] for k in ("longitudinal", "lateral")},
                "grid": new["_grid"]}
OUT["arms"] = res

# the corrected PAIRED deltas, on the right grid
import math  # noqa: E402


def per_window(W, dt):
    P = FF._seq_geometry(W["pred"].float(), dt)
    G = FF._seq_geometry(W["gt"].float(), dt)
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    both, bp = P["valid"] & G["valid"], P["pair_valid"] & G["pair_valid"]

    def _m(x, m):
        return ((x * m).sum(-1) / m.sum(-1).clamp(min=1)).numpy()
    return {"speed_mae_mps": (P["speed"] - G["speed"]).abs().mean(-1).numpy(),
            "speed_bias_mps": (P["speed"] - G["speed"]).mean(-1).numpy(),
            "along_mae_m": (P["along"] - G["along"]).abs().mean(-1).numpy(),
            "accel_mae_mps2": (P["accel"] - G["accel"]).abs().mean(-1).numpy(),
            "heading_mae_deg": _m(dh.abs().rad2deg(), both),
            "yaw_rate_mae_degps": _m((P["yaw_rate"] - G["yaw_rate"]).abs().rad2deg(), bp),
            "curvature_mae_1pm": _m((P["curvature"] - G["curvature"]).abs(), bp),
            "cross_mae_m": (P["cross"] - G["cross"]).abs().mean(-1).numpy()}


pa, pb = per_window(A, 0.5), per_window(B, 0.5)
OUT["CORRECTED_paired_delta_B_minus_A"] = {
    k: CI.paired_episode_cluster_bootstrap(pb[k], pa[k], eid) for k in pa}
OUT["CORRECTED_arm_levels"] = {
    k: {"A_fp32": CI.episode_cluster_bootstrap(pa[k], eid),
        "B_opt": CI.episode_cluster_bootstrap(pb[k], eid)} for k in pa}

# materiality beside separation — the two are NOT the same test
mat = {}
for k, v in OUT["CORRECTED_paired_delta_B_minus_A"].items():
    lvl = abs(OUT["CORRECTED_arm_levels"][k]["A_fp32"]["mean"]) or 1e-12
    mat[k] = {"delta": v["delta"], "separated": v["separated"],
              "pct_of_fp32_level": round(100.0 * v["delta"] / lvl, 4)}
OUT["MATERIALITY"] = {
    "note": ("a paired episode-cluster bootstrap over 859 windows resolves sub-percent shifts. "
             "SEPARATION is not MATERIALITY — both are reported, neither is used to redefine the "
             "other, and the pre-registered falsifier stands as written."),
    "per_metric": mat}
with open(os.path.expanduser("~/thor_c4_rescore_corrected.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT["NEGATIVE_CONTROL"], indent=1), flush=True)
print(json.dumps(OUT["MATERIALITY"]["per_metric"], indent=1), flush=True)
print("WROTE ~/thor_c4_rescore_corrected.json", flush=True)
