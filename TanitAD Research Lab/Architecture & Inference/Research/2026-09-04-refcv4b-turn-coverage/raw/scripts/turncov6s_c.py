"""PART 3 -- three checks the verdict must survive.

 C1 KAMM INSTRUMENT. The banked refcv4b Kamm figures (`anchors.build.json`:
    "over_mu_0.7_at_v0_10.09_ms": 12, "peak_g": 1.23) were computed by FINITE
    DIFFERENCING the 8 SLOT waypoints, whose spacing is 0.5-1.0 s. The
    integration step is 0.1 s and the realised lateral load is v(t)^2 * kappa,
    which GROWS QUADRATICALLY along an accelerating candidate. A 0.5-1.0 s
    finite difference smooths that peak. Both are computed here, side by side.

 C2 PHYSICALLY ADMISSIBLE SUPPLY. Part 1's coverage counted every candidate the
    reach clamp keeps. If the >30 deg turns are supplied only by candidates that
    break a mu = 0.7 friction circle, the coverage claim is empty. Recomputed on
    the Kamm-admissible subset.

 C3 DOES THE ORACLE EVER PICK A KAMM-BREAKING CANDIDATE? If it does, the oracle
    ceiling itself is partly unphysical and must be quoted with that caveat.

STRICTLY MODEL-FREE. No checkpoint, no forward pass.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\taniteval")
import tanitad.refs.refc_select as sl                           # noqa: E402

OUT = os.path.join(HERE, "turncov_out")
DUMP = os.path.join(HERE, "refcv3_40284_dump")
DT, H, HORIZON_S = 0.1, 60, 6.0
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [k - 1 for k in HORIZONS]
V_FLOOR, KAPPA_CAP, SEL_ACCEL_MAX, G = 4.0, 0.12, 2.0, 9.81
deg = np.degrees

d = np.load(os.path.join(OUT, "per_window.npz"))
v0 = d["v0"]; eid = d["eid"]; ws = d["ws"]
dem_thx = d["dem_thx"]; dem_eb = d["dem_eb"]
N = len(v0)
rep = {}

CTRL = torch.load(os.path.join(HERE, "pull", "anchors_live.pt"),
                  map_location="cpu",
                  weights_only=False)["controls"].double().numpy()
NA = len(CTRL)

man = torch.load(os.path.join(HERE, "pull", "_v2manifest.pt"),
                 map_location="cpu", weights_only=False)
pbc = {c: man["poses"][i].double().numpy() for i, c in enumerate(man["clip_id"])}
dman = json.load(open(os.path.join(DUMP, "manifest.json")))
clips = []
for e in dman["episodes"]:
    clips += [e["clip_id"]] * e["n_windows"]
gt = np.zeros((N, len(SLOTS), 2)); sv = np.zeros((N, len(SLOTS)))
for i in range(N):
    P = pbc[clips[i]]; T = P.shape[0]; t0 = int(ws[i])
    a = np.array([t0 + k for k in HORIZONS])
    sv[i] = (a <= T - 1).astype(float)
    q = P[np.clip(a, None, T - 1), :2] - P[t0, :2]
    c, s = np.cos(-P[t0, 2]), np.sin(-P[t0, 2])
    gt[i, :, 0] = q[:, 0] * c - q[:, 1] * s
    gt[i, :, 1] = q[:, 0] * s + q[:, 1] * c
svn = np.maximum(sv.sum(1), 1.0)


def roll(ctrl, v0v):
    B, M = len(v0v), len(ctrl)
    vv = np.maximum(v0v, V_FLOOR) ** 2
    kap = np.clip(ctrl[None, :, 1] / vv[:, None], -KAPPA_CAP, KAPPA_CAP)
    a = np.repeat(ctrl[None, :, 0], B, axis=0)
    x = np.zeros((B, M)); y = np.zeros((B, M)); yaw = np.zeros((B, M))
    v = np.repeat(v0v[:, None], M, axis=1).copy()
    wp = np.zeros((B, M, len(SLOTS), 2)); peak = np.zeros((B, M))
    for k in range(H):
        peak = np.maximum(peak, np.hypot(np.where(v > 0, a, np.maximum(a, 0.0)),
                                         v ** 2 * kap))
        x = x + v * np.cos(yaw) * DT
        y = y + v * np.sin(yaw) * DT
        yaw = yaw + v * kap * DT
        v = np.maximum(v + a * DT, 0.0)
        if k in SLOTS:
            wp[:, :, SLOTS.index(k), 0] = x
            wp[:, :, SLOTS.index(k), 1] = y
    seg = wp[:, :, -1] - wp[:, :, -2]
    return wp, yaw, np.arctan2(seg[..., 1], seg[..., 0]), peak


def kamm_slotfd(P):
    """The BANKED instrument: np.gradient over the 8 slot instants (0.5-6.0 s),
    exactly `coverage6s.py::kamm` / `zero_gpu_checks.py`."""
    t = np.array(HORIZONS, dtype=np.float64) * DT
    v = np.gradient(P, t, axis=1)
    a = np.gradient(v, t, axis=1)
    sp = np.linalg.norm(v, axis=-1)
    lon = np.abs((v * a).sum(-1) / np.maximum(sp, 1e-6))
    lat = np.abs((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0])
                 / np.maximum(sp, 1e-6))
    return (np.hypot(lon, lat) / G).max(-1)


# =========================================================================== #
print("=== C1. KAMM: the banked slot-finite-difference vs the per-step exact ===")
print(f"{'v0 (m/s)':<10}{'slot-FD >0.7g':>16}{'slot-FD peak':>14}"
      f"{'per-step >0.7g':>17}{'per-step peak':>15}{'ratio':>8}")
c1 = []
for vq in (10.0, 10.09, 18.0, 27.0, 27.27, 36.0):
    wp, _, _, pk = roll(CTRL, np.array([vq]))
    g_exact = pk[0] / G          # `roll` returns m/s^2; kamm_slotfd returns g
    g_fd = kamm_slotfd(wp[0])
    r = {"v0_ms": vq,
         "slot_fd_n_over_mu070": int((g_fd > 0.7).sum()),
         "slot_fd_peak_g": float(g_fd.max()),
         "per_step_n_over_mu070": int((g_exact > 0.7).sum()),
         "per_step_peak_g": float(g_exact.max()),
         "peak_ratio": float(g_exact.max() / max(g_fd.max(), 1e-9))}
    c1.append(r)
    print(f"{vq:<10.2f}{r['slot_fd_n_over_mu070']:>13}/117"
          f"{r['slot_fd_peak_g']:>13.2f}g{r['per_step_n_over_mu070']:>14}/117"
          f"{r['per_step_peak_g']:>14.2f}g{r['peak_ratio']:>8.2f}x")
rep["C1_kamm_instrument"] = c1
print("  the banked anchors.build.json rows are the SLOT-FD column "
      "(over_mu_0.7 @10.09 = 12, peak 1.23 g; @27.27 = 0, peak 0.68 g)")

# =========================================================================== #
print("\n=== C2. SUPPLY on the PHYSICALLY ADMISSIBLE subset (peak |a| <= mu*g) ===")
CH = 512
sup = {}
for mu in (0.7, 0.9, None):
    sup[mu] = {"thx": np.zeros(N), "eb": np.zeros(N), "ths": np.zeros(N),
               "n": np.zeros(N, dtype=np.int64)}
ade_adm = np.zeros(N)
oracle_peak = np.zeros(N)
ade_full = np.zeros(N)
for s in range(0, N, CH):
    e = min(s + CH, N)
    wp, yaw6, segh, pk = roll(CTRL, v0[s:e])
    keep = sl.anchor_reachability_mask(
        torch.from_numpy(wp), torch.from_numpy(v0[s:e]),
        accel_max=SEL_ACCEL_MAX, horizon_s=HORIZON_S).numpy()
    eb = np.abs(np.arctan2(wp[:, :, -1, 1], wp[:, :, -1, 0]))
    thx = np.abs(yaw6); ths = np.abs(segh)
    dd = wp - gt[s:e, None]
    ade = (np.sqrt((dd ** 2).sum(-1)) * sv[s:e, None]).sum(-1) / svn[s:e, None]
    r = np.arange(e - s)
    bi = np.where(keep, ade, np.inf).argmin(1)
    ade_full[s:e] = ade[r, bi]
    oracle_peak[s:e] = pk[r, bi]
    for mu in (0.7, 0.9, None):
        k = keep if mu is None else (keep & (pk <= mu * G))
        # the straight-ahead control is always admissible, so k is never empty
        sup[mu]["n"][s:e] = k.sum(1)
        sup[mu]["thx"][s:e] = np.where(k, thx, -1).max(1)
        sup[mu]["eb"][s:e] = np.where(k, eb, -1).max(1)
        sup[mu]["ths"][s:e] = np.where(k, ths, -1).max(1)
    k07 = keep & (pk <= 0.7 * G)
    ade_adm[s:e] = np.where(k07, ade, np.inf).min(1)

rows = []
print(f"{'admissibility':<26}{'surv/win':>10}{'no>15':>9}{'no>30':>9}"
      f"{'no>45':>9}{'no>60':>9}{'dem>sup':>9}")
for mu, tag in ((None, "reach clamp only"), (0.9, "+ Kamm mu<=0.9"),
                (0.7, "+ Kamm mu<=0.7")):
    S = sup[mu]
    r = {"admissibility": tag, "survivors_per_window": float(S["n"].mean())}
    for th in (15, 30, 45, 60):
        r[f"no_cand_gt{th}deg_terminal_heading_pct"] = float(
            100.0 * (deg(S["thx"]) <= th).mean())
        r[f"no_cand_gt{th}deg_end_bearing_pct"] = float(
            100.0 * (deg(S["eb"]) <= th).mean())
    r["pct_demand_exceeds_supply_terminal_heading"] = float(
        100.0 * (dem_thx > S["thx"]).mean())
    r["pct_demand_exceeds_supply_end_bearing"] = float(
        100.0 * (dem_eb > S["eb"]).mean())
    rows.append(r)
    print(f"{tag:<26}{r['survivors_per_window']:>10.1f}"
          f"{r['no_cand_gt15deg_terminal_heading_pct']:>8.2f}%"
          f"{r['no_cand_gt30deg_terminal_heading_pct']:>8.2f}%"
          f"{r['no_cand_gt45deg_terminal_heading_pct']:>8.2f}%"
          f"{r['no_cand_gt60deg_terminal_heading_pct']:>8.2f}%"
          f"{r['pct_demand_exceeds_supply_terminal_heading']:>8.2f}%")
rep["C2_supply_by_admissibility"] = rows
print("  (columns are TERMINAL HEADING; the end-bearing twins are in the JSON)")

# =========================================================================== #
print("\n=== C3. does the ORACLE pick Kamm-breaking candidates? ===")
brk = oracle_peak > 0.7 * G
turn = deg(dem_thx) >= 30.0
c3 = {"pct_oracle_pick_over_mu070": float(100.0 * brk.mean()),
      "pct_oracle_pick_over_mu070_on_turning_windows": float(
          100.0 * brk[turn].mean()),
      "n_turning": int(turn.sum()),
      "oracle_peak_g_median": float(np.median(oracle_peak) / G),
      "oracle_peak_g_p95": float(np.percentile(oracle_peak, 95) / G),
      "oracle_peak_g_max": float(oracle_peak.max() / G),
      "ade_unconstrained_m": float(ade_full.mean()),
      "ade_kamm_admissible_m": float(ade_adm.mean()),
      "ade_penalty_of_kamm_filter_m": float(ade_adm.mean() - ade_full.mean())}
print(f"  oracle picks a mu>0.7 candidate on {c3['pct_oracle_pick_over_mu070']:.2f} % "
      f"of windows ({c3['pct_oracle_pick_over_mu070_on_turning_windows']:.2f} % of "
      f"the {c3['n_turning']} turning windows)")
print(f"  oracle peak |a|: median {c3['oracle_peak_g_median']:.2f} g, p95 "
      f"{c3['oracle_peak_g_p95']:.2f} g, max {c3['oracle_peak_g_max']:.2f} g")
print(f"  OIV ADE unconstrained {c3['ade_unconstrained_m']:.4f} m -> Kamm-admissible "
      f"{c3['ade_kamm_admissible_m']:.4f} m  (penalty "
      f"{c3['ade_penalty_of_kamm_filter_m']:+.4f} m)")
rep["C3_oracle_kamm"] = c3

json.dump(rep, open(os.path.join(OUT, "TURNCOV6S_PART3.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "TURNCOV6S_PART3.json"))
