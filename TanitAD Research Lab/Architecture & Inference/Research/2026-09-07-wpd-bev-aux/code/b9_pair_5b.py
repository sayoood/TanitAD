# -*- coding: utf-8 -*-
"""WP-D section 5B -- the PAIRED four-family comparison of D1 (aux ON) against D0 (aux OFF).

⛔ `refcv3_arm.py` pairs arms WITHIN one record (os vs ha0 ...). D0 and D1 are two
CHECKPOINTS, so the paired comparison has to be made across their two dumps -- and
it is a real pairing only if the windows line up. That is ASSERTED here, per
episode and per window start, and the script REFUSES if it does not hold.

Geometry comes from `taniteval.four_families._seq_geometry` -- the programme's own
canonical definitions, imported, not reimplemented -- and every interval is
`taniteval.ci.paired_episode_cluster_bootstrap`.

⭐ It also carries `D-REFAV1-LON-ACTS`'s "IT MUST STILL ACT" control, because a
longitudinal family can be improved by making the planner STOP: mean |a| must hold
or rise (floor +0.044) and the zero-acceleration plan fraction must not rise
(floor 0.05).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SNAP = r"C:\Users\Admin\wpd-probe\snap"
sys.path.insert(0, SNAP)
from taniteval.four_families import _seq_geometry
from taniteval.ci import paired_episode_cluster_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--a", default=r"C:\Users\Admin\wpd-probe\dumps\wpdD1", help="the LEVER (D1)")
ap.add_argument("--b", default=r"C:\Users\Admin\wpd-probe\dumps\wpdD0", help="the CONTROL (D0)")
ap.add_argument("--key", default="os", help="the fed arm in the dump")
ap.add_argument("--dt", type=float, default=0.5)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(d):
    eps = sorted(f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz"))
    P, G, E, W, V = [], [], [], [], []
    for f in eps:
        z = np.load(os.path.join(d, f))
        P.append(z[a.key]); G.append(z["g"]); W.append(z["ws"]); V.append(z["v0"])
        E.append(np.full(len(z["ws"]), int(z["eid"][0]), dtype=np.int64))
    return (np.concatenate(P), np.concatenate(G), np.concatenate(E),
            np.concatenate(W), np.concatenate(V), len(eps))


Pa, Ga, Ea, Wa, Va, na = load(a.a)
Pb, Gb, Eb, Wb, Vb, nb = load(a.b)
print(f"[load] A {a.a}: {Pa.shape[0]} windows / {na} episodes", flush=True)
print(f"[load] B {a.b}: {Pb.shape[0]} windows / {nb} episodes", flush=True)

# ⛔ THE PAIRING IS ASSERTED, NOT ASSUMED. Two arms scored on different windows
# would make every "paired" interval meaningless while looking identical.
align = {"n_windows_equal": Pa.shape[0] == Pb.shape[0],
         "eid_identical": bool(np.array_equal(Ea, Eb)),
         "window_start_identical": bool(np.array_equal(Wa, Wb)),
         "gt_identical": bool(np.array_equal(Ga, Gb)),
         "v0_identical": bool(np.array_equal(Va, Vb))}
print("[align]", json.dumps(align), flush=True)
if not all(align.values()):
    sys.exit("REFUSING: the two dumps are not window-aligned -- a paired interval "
             "across them would be arithmetic, not evidence.")
# ⭐ and the two arms must NOT be identical, or every delta is a tie-break
if np.array_equal(Pa, Pb):
    sys.exit("REFUSING: D1 and D0 emitted BIT-IDENTICAL trajectories")
print(f"[distinct] mean |pred_A - pred_B| = {np.abs(Pa - Pb).mean():.6f} m", flush=True)

TA, TB, TG = (torch.from_numpy(x.astype(np.float32)) for x in (Pa, Pb, Ga))
SA, SB, SG = _seq_geometry(TA, a.dt), _seq_geometry(TB, a.dt), _seq_geometry(TG, a.dt)
EID = Ea


def wmean(x, m=None):
    """per-window mean over the horizon; NaN where a window has no valid step."""
    if m is None:
        return x.mean(dim=1).numpy().astype(np.float64)
    num = (x * m).sum(dim=1)
    den = m.sum(dim=1)
    out = torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, float("nan")))
    return out.numpy().astype(np.float64)


def l2(p, g):
    return torch.linalg.norm(p - g, dim=-1)


FAM = {}
FAM["ADE_m"] = ("ADE", wmean(l2(TA, TG)), wmean(l2(TB, TG)), None)
FAM["FDE_m"] = ("ADE", l2(TA, TG)[:, -1].numpy().astype(np.float64),
                l2(TB, TG)[:, -1].numpy().astype(np.float64), None)
FAM["LON_speed_mae_mps"] = ("LONGITUDINAL", wmean((SA["speed"] - SG["speed"]).abs()),
                            wmean((SB["speed"] - SG["speed"]).abs()), None)
FAM["LON_along_mae_m"] = ("LONGITUDINAL", wmean((SA["along"] - SG["along"]).abs()),
                          wmean((SB["along"] - SG["along"]).abs()), None)
FAM["LON_accel_mae_mps2"] = ("LONGITUDINAL", wmean((SA["accel"] - SG["accel"]).abs()),
                             wmean((SB["accel"] - SG["accel"]).abs()), None)
FAM["LAT_cross_mae_m"] = ("LATERAL", wmean((SA["cross"] - SG["cross"]).abs()),
                          wmean((SB["cross"] - SG["cross"]).abs()), None)
mh = SA["valid"] & SB["valid"] & SG["valid"]
dh_a = ((SA["heading"] - SG["heading"] + np.pi) % (2 * np.pi) - np.pi).abs()
dh_b = ((SB["heading"] - SG["heading"] + np.pi) % (2 * np.pi) - np.pi).abs()
FAM["LAT_heading_mae_deg"] = ("LATERAL", wmean(dh_a * 180 / np.pi, mh),
                              wmean(dh_b * 180 / np.pi, mh), "masked to steps moving in BOTH arms and GT")
mp = SA["pair_valid"] & SB["pair_valid"] & SG["pair_valid"]
FAM["LAT_curvature_mae_1pm"] = ("LATERAL", wmean((SA["curvature"] - SG["curvature"]).abs(), mp),
                                wmean((SB["curvature"] - SG["curvature"]).abs(), mp), "masked, dt-invariant")
FAM["LAT_yawrate_mae_radps"] = ("LATERAL", wmean((SA["yaw_rate"] - SG["yaw_rate"]).abs(), mp),
                                wmean((SB["yaw_rate"] - SG["yaw_rate"]).abs(), mp), "masked")

res = {}
for name, (fam, x, y, note) in FAM.items():
    ok = np.isfinite(x) & np.isfinite(y)
    d = paired_episode_cluster_bootstrap(x[ok], y[ok], list(EID[ok]),
                                         n_boot=a.n_boot, seed=0)
    d["family"] = fam
    d["A_point"] = round(float(np.mean(x[ok])), 6)
    d["B_point"] = round(float(np.mean(y[ok])), 6)
    d["n_windows_used"] = int(ok.sum())
    d["worse_for_A"] = bool(d["separated"] and d["delta"] > 0)   # every metric is an ERROR
    if note:
        d["note"] = note
    res[name] = d
    print(f"[{fam:13s}] {name:24s} A {d['A_point']:.4f}  B {d['B_point']:.4f}  "
          f"delta {d['delta']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}] sep={d['separated']}",
          flush=True)

# ⭐ D-REFAV1-LON-ACTS -- "it must still ACT"
def act(S, P):
    aa = S["accel"].abs()
    return {"mean_abs_a_mps2": round(float(aa.mean()), 6),
            "frac_windows_all_zero_accel": round(float((aa.max(dim=1).values < 1e-6).float().mean()), 6),
            "mean_abs_a_per_window": aa.mean(dim=1).numpy().astype(np.float64)}


AA, AB = act(SA, Pa), act(SB, Pb)
d_act = paired_episode_cluster_bootstrap(AA["mean_abs_a_per_window"],
                                         AB["mean_abs_a_per_window"],
                                         list(EID), n_boot=a.n_boot, seed=0)
ACT = {"A_mean_abs_a": AA["mean_abs_a_mps2"], "B_mean_abs_a": AB["mean_abs_a_mps2"],
       "delta_mean_abs_a": d_act,
       "A_frac_zero_accel": AA["frac_windows_all_zero_accel"],
       "B_frac_zero_accel": AB["frac_windows_all_zero_accel"],
       "floors": {"mean_abs_a": 0.044, "frac_a_eq_0": 0.05},
       "verdict_mean_abs_a": ("HOLDS OR RISES" if AA["mean_abs_a_mps2"] >= AB["mean_abs_a_mps2"] - 0.044
                              else "⛔ FALLS BY MORE THAN THE FLOOR — the arm is ACTING LESS"),
       "verdict_frac_zero": ("does not rise" if AA["frac_windows_all_zero_accel"]
                             <= AB["frac_windows_all_zero_accel"] + 0.05
                             else "⛔ RISES BY MORE THAN THE FLOOR"),
       "rule": ("D-REFAV1-LON-ACTS: a longitudinal family can be improved by making the "
                "planner STOP. mean |a| must hold or rise and the zero-acceleration plan "
                "fraction must not rise.")}
print("[ACT]", json.dumps({k: v for k, v in ACT.items() if k != "delta_mean_abs_a"},
                          ensure_ascii=False), flush=True)

worse = [k for k, v in res.items() if v["worse_for_A"]]
better = [k for k, v in res.items() if v["separated"] and v["delta"] < 0]
out = {"_evidence_class": "MEASURED (ours; artifact = this file + the two dumps)",
       "tier": "T1 (self-action open loop) — the tier refcv3_arm stamps for `os`",
       "A": a.a, "B": a.b, "arm_key": a.key, "dt_s": a.dt,
       "alignment": align, "mean_abs_pred_diff_m": float(np.abs(Pa - Pb).mean()),
       "n_windows": int(Pa.shape[0]), "n_episodes": int(len(set(EID.tolist()))),
       "metrics": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
       "act_control": {k: v for k, v in ACT.items() if k != "delta_mean_abs_a"},
       "act_delta_mean_abs_a": ACT["delta_mean_abs_a"],
       "B1_separably_worse_for_D1": worse,
       "B1_separably_better_for_D1": better,
       "B1_verdict": ("PASS — no family metric is separably worse for D1" if not worse
                      else f"⛔ FAIL — separably WORSE for D1 on: {worse}"),
       "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), n_boot="
                    f"{a.n_boot}, clusters = episodes",
       "what_the_interval_answers": ("would another DRAW OF EPISODES say this? NOT whether "
                                     "another TRAINING RUN would — that is D0b."),
       "strategic_family": ("UNAVAILABLE with its reason and n=0, as refcv3_arm reports it: "
                            "route_pred/route_gt are absent from a trajectory dump and the "
                            "family needs a hierarchy-traversing eval. Stated per family "
                            "rather than silently dropped.")}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
print("B1:", out["B1_verdict"], flush=True)
print("WROTE", a.out)
