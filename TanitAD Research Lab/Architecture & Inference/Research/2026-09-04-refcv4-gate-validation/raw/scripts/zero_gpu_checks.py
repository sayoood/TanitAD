"""Zero-GPU vocabulary checks for refcv4, on the 4,823-window gate surface.

Uses the PROGRAMME'S OWN reach-clamp function (flagship_v15.reachability_mask via
refc_select) rather than a re-implementation, so the kill rate is the one the code
computes, not a look-alike.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
from tanitad.refs.refc_select import anchor_reachability_mask  # noqa: E402
from tanitad.refs import refc as R                             # noqa: E402

DUMP = os.path.join(HERE, "refcv3_40284_dump")
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
HOR_S = max(HORIZONS) * 0.1          # 6.0 — exactly refc.py's derivation
T = np.array(HORIZONS, dtype=np.float64) * 0.1

eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
V0 = np.concatenate([np.load(os.path.join(DUMP, f))["v0"] for f in eps]).astype(np.float64)
print(f"[surface] {len(V0)} windows / {len(eps)} episodes; v0 mean {V0.mean():.3f} m/s")

SETS = {
    "refcv3 SYNTHETIC (incumbent)": os.path.join(HERE, "refcv3_anchors.pt"),
    "refcv4 NEW (shipped)": os.path.join(HERE, "anchors_pod.pt"),
}
report = {}
for name, path in SETS.items():
    A = torch.load(path, map_location="cpu", weights_only=True).double()
    An = A.numpy()
    print(f"\n{'='*74}\n{name}   shape {tuple(A.shape)}\n{'='*74}")

    # ---------------- REACH CLAMP, via the programme's own function ----------
    v0t = torch.from_numpy(V0)
    print("  reach clamp  (v_mean = ||wp_last||/6.0, band v0 +/- a*6.0)")
    # >30 deg turn definitions, computed once
    endb = np.degrees(np.abs(np.arctan2(An[:, -1, 1], An[:, -1, 0])))
    seg = An[:, -1, :] - An[:, -2, :]
    termh = np.degrees(np.abs(np.arctan2(seg[:, 1], seg[:, 0])))
    clamp_rows = []
    for amax in (1.5, 2.0, 2.5):
        keep = anchor_reachability_mask(A[None].expand(len(V0), -1, -1, -1),
                                        v0t, accel_max=amax,
                                        horizon_s=HOR_S).numpy()
        kill = 100.0 * (1.0 - keep.mean())
        empty = 100.0 * (~keep.any(1)).mean()
        surv = keep.sum(1).mean()
        turn_end = (keep & (endb > 30)[None]).sum(1)
        turn_term = (keep & (termh > 30)[None]).sum(1)
        no_turn = 100.0 * (turn_end == 0).mean()
        clamp_rows.append(dict(a_max=amax, band=amax * HOR_S, kill_pct=kill,
                               empty_pct=empty, surv_per_window=float(surv),
                               turns30_endbearing=float(turn_end.mean()),
                               turns30_terminalheading=float(turn_term.mean()),
                               windows_with_no_30deg_turn_pct=float(no_turn)))
        print(f"    a={amax:<4} band=+/-{amax*HOR_S:4.1f} m/s   kill {kill:5.2f}%   "
              f"empty {empty:.2f}%   surv/win {surv:6.1f}   "
              f">30deg turns/win {turn_end.mean():5.1f} (end-bearing) / "
              f"{turn_term.mean():5.1f} (terminal-heading)   "
              f"windows w/ NO >30deg turn {no_turn:5.2f}%")

    # ------------------------------- Kamm circle, per-segment kinematics -----
    P = np.concatenate([np.zeros((An.shape[0], 1, 2)), An], axis=1)
    tt = np.concatenate([[0.0], T])
    dt = np.diff(tt)
    d = np.diff(P, axis=1)
    v = np.linalg.norm(d, axis=-1) / dt[None]
    a_lon = np.diff(np.concatenate([v[:, :1], v], 1), axis=1) / dt[None]
    th = np.arctan2(d[..., 1], d[..., 0])
    dth = np.diff(np.concatenate([th[:, :1], th], 1), axis=1)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    yaw = dth / dt[None]
    a_lat = v * yaw
    a_tot = np.hypot(a_lon, a_lat)
    G = 9.81
    kamm = {}
    for mu in (0.7, 0.8, 0.9, 1.0):
        kamm[f"mu_{mu}"] = int((a_tot > mu * G).any(1).sum())
    stalled = int(((v < 0.5) & (np.abs(yaw) > 0.2)).any(1).sum())
    print(f"  Kamm circle (per 0.5-6 s finite-difference segment):")
    for mu in (0.7, 0.8, 0.9, 1.0):
        print(f"    mu={mu}: {kamm[f'mu_{mu}']:3d} of {An.shape[0]} anchors break the circle")
    print(f"    peak |a| {a_tot.max()/G:.2f} g   max|a_lon| {np.abs(a_lon).max():.2f} "
          f"max|a_lat| {np.abs(a_lat).max():.2f} m/s2   v_max {v.max():.2f} m/s")
    print(f"    turn-while-stalled (v<0.5 m/s & |yaw|>0.2 rad/s): {stalled}")
    print(f"    max |turn| (end-bearing) {endb.max():.1f} deg   "
          f"anchors with >30deg end-bearing: {int((endb>30).sum())}")
    report[name] = {"clamp": clamp_rows, "kamm": kamm, "peak_g": float(a_tot.max() / G),
                    "stalled_turn": stalled, "max_turn_deg": float(endb.max()),
                    "n_turns_gt30": int((endb > 30).sum()),
                    "max_speed_ms": float(v.max()),
                    "max_a_lon": float(np.abs(a_lon).max()),
                    "max_a_lat": float(np.abs(a_lat).max())}

json.dump(report, open(os.path.join(HERE, "ZERO_GPU_CHECKS.json"), "w"), indent=1)
print("\nwrote ZERO_GPU_CHECKS.json")
