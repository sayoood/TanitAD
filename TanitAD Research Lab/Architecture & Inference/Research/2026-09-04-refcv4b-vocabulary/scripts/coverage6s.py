"""Reach clamp + turn coverage + Kamm circle for the v0-CONDITIONED bank.

Same definitions and the same instrument as the 2026-09-04 gate-validation pass
(`zero_gpu_checks.py`), so the incumbent rows reproduce and the new set is
comparable:

  * survivors  = `refc_select.anchor_reachability_mask(bank, v0,
                  accel_max=a, horizon_s=6.0)` -- the programme's own function,
                  which already accepts a PER-WINDOW [B, N, S, 2] bank;
  * a ">30 deg turn" is counted BOTH ways: END-BEARING atan2(y_end, x_end) and
    TERMINAL HEADING (the last segment's direction);
  * "windows with NO >30 deg turn" = the property refcv3 had at 0.00 % and the
    shipped refcv4 fixed-path set lost (4.62 %).

The clamp is evaluated on the SAME 4,823-window surface, at each window's own v0.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

STACK = r"C:\Users\Admin\run_refcv4v\repo\stack"
sys.path.insert(0, STACK)
from tanitad.refs.refa_v1_plan import unicycle_paths            # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS                    # noqa: E402
import tanitad.refs.refc_select as sl                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "..", "vocab", "refcv3_40284_dump")
OUT = os.path.join(HERE, "out")
DT = 0.1
H = max(V3_HORIZONS)
SLOT_ALL = [k - 1 for k in V3_HORIZONS]
HORIZON_S = 6.0

eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
V0 = np.concatenate([np.load(os.path.join(DUMP, f))["v0"] for f in eps]).astype(np.float64)
N = len(V0)
print("[surface] %d windows / %d episodes" % (N, len(eps)))


def grid(na, nk, a_lo, a_hi, k_lim):
    assert na % 2 == 1 and nk % 2 == 1
    a_g = np.linspace(a_lo, a_hi, na)
    if na > 1:
        a_g = np.clip(a_g - a_g[np.abs(a_g).argmin()], a_lo, a_hi)
    k_g = np.linspace(-k_lim, k_lim, nk)
    assert np.any(a_g == 0.0) and np.any(k_g == 0.0)
    aa, kk = np.meshgrid(a_g, k_g, indexing="ij")
    return np.stack([aa.ravel(), kk.ravel()], -1)


def roll(ctrl, v0):
    c = torch.from_numpy(np.repeat(ctrl[:, None, :], H, axis=1)).double()
    p = unicycle_paths(c, torch.tensor(float(v0), dtype=torch.float64), DT,
                       action_units="kappa")
    return p.numpy()[:, SLOT_ALL, :]


def turn_flags(P):
    """[M, S, 2] -> (end-bearing deg, terminal-heading deg), both |.|."""
    eb = np.degrees(np.abs(np.arctan2(P[:, -1, 1], P[:, -1, 0])))
    seg = P[:, -1] - P[:, -2]
    th = np.degrees(np.abs(np.arctan2(seg[:, 1], seg[:, 0])))
    return eb, th


def kamm(P, mu_list=(0.7, 0.8, 0.9, 1.0)):
    """finite differences over the anchor's own slots -> peak |a| in g."""
    t = np.array(V3_HORIZONS, dtype=np.float64) * DT
    v = np.gradient(P, t, axis=1)                       # [M, S, 2] velocity
    a = np.gradient(v, t, axis=1)                       # [M, S, 2] accel
    sp = np.linalg.norm(v, axis=-1)
    lon = np.abs((v * a).sum(-1) / np.maximum(sp, 1e-6))
    lat = np.abs((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0])
                 / np.maximum(sp, 1e-6))
    tot = np.sqrt(lon ** 2 + lat ** 2) / 9.81
    peak = tot.max(1)
    return ({("mu_%.1f" % m): int((peak > m).sum()) for m in mu_list},
            float(peak.max()), float(lon.max()), float(lat.max()),
            float(sp.max()))


FAMS = {
    "13a x 9k = 117 (SHIP CANDIDATE)": grid(13, 9, -4.0, 3.0, 0.06),
    "13a x 11k = 143": grid(13, 11, -4.0, 3.0, 0.06),
    "15a x 9k = 135": grid(15, 9, -5.0, 3.0, 0.06),
}
res = {}
for tag, ctrl in FAMS.items():
    key = np.round(V0, 3)
    cache = {}
    for k in np.unique(key):
        cache[k] = roll(ctrl, k)
    for amax, band_label in ((2.0, "+/-12.0"), (2.5, "+/-15.0"), (1.5, "+/-9.0")):
        killed = np.empty(N); surv = np.empty(N)
        t_eb = np.empty(N); t_th = np.empty(N)
        empty = 0
        for i in range(N):
            P = cache[key[i]]
            m = sl.anchor_reachability_mask(
                torch.from_numpy(P)[None], torch.tensor([V0[i]], dtype=torch.float64),
                accel_max=amax, horizon_s=HORIZON_S)[0].numpy()
            s = int(m.sum())
            surv[i] = s
            killed[i] = len(ctrl) - s
            if s == 0:
                empty += 1
                t_eb[i] = t_th[i] = 0
                continue
            eb, th = turn_flags(P[m])
            t_eb[i] = int((eb > 30).sum()); t_th[i] = int((th > 30).sum())
        no_turn = float((t_eb == 0).mean() * 100.0)
        row = {"accel_max": amax, "band_ms": band_label,
               "killed_pct": float(killed.mean() / len(ctrl) * 100.0),
               "empty_pct": float(empty / N * 100.0),
               "survivors_per_window": float(surv.mean()),
               "turns_gt30_end_bearing_per_window": float(t_eb.mean()),
               "turns_gt30_terminal_heading_per_window": float(t_th.mean()),
               "windows_with_no_gt30_turn_pct": no_turn}
        res["%s @ a_max %.1f" % (tag, amax)] = row
        print("  %-40s a_max %.1f  killed %6.2f%%  empty %.2f%%  surv/win %6.1f  "
              ">30deg %5.1f / %5.1f  NO-turn %.2f%%"
              % (tag, amax, row["killed_pct"], row["empty_pct"],
                 row["survivors_per_window"],
                 row["turns_gt30_end_bearing_per_window"],
                 row["turns_gt30_terminal_heading_per_window"], no_turn))
    # Kamm on a representative mid speed and on the corpus max
    for v in (float(np.median(V0)), float(np.percentile(V0, 95))):
        cnt, peak, mlon, mlat, vmax = kamm(roll(ctrl, v))
        print("     Kamm @ v0=%5.2f m/s : %s  peak %.2f g  max|a_lon| %.2f  "
              "max|a_lat| %.2f  v_max %.2f" % (v, cnt, peak, mlon, mlat, vmax))
        res["%s KAMM v0=%.2f" % (tag, v)] = {
            "over_mu": cnt, "peak_g": peak, "max_a_lon": mlon,
            "max_a_lat": mlat, "v_max_ms": vmax}

res["_incumbents_for_scale"] = {
    "refcv3 synthetic @ a_max 2.5 (shipped)": {
        "killed_pct": 37.10, "survivors_per_window": 80.5,
        "turns_gt30_end_bearing_per_window": 40.4,
        "windows_with_no_gt30_turn_pct": 0.00},
    "refcv4 fixed-path @ a_max 2.0 (shipped, ABORTED)": {
        "killed_pct": 26.60, "survivors_per_window": 94.0,
        "turns_gt30_end_bearing_per_window": 19.0,
        "windows_with_no_gt30_turn_pct": 4.62}}
json.dump(res, open(os.path.join(OUT, "COVERAGE6S.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "COVERAGE6S.json"))
