"""PART 2 — where the 8.85 % lives, the Kamm cost of widening, and the AMENDED
vocabulary counterfactual on the SAME windows with a paired episode-cluster
bootstrap.

STRICTLY MODEL-FREE (no checkpoint, no forward pass). Oracle-in-vocabulary
is a CEILING: the live decoder adds a per-anchor offset on top of the bank
(`refc.py::_decode` -> `offset_head`), so these numbers bound what the anchor
PRIOR can express, never what the model achieves.
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
from taniteval.ci import paired_episode_cluster_bootstrap       # noqa: E402

OUT = os.path.join(HERE, "turncov_out")
DUMP = os.path.join(HERE, "refcv3_40284_dump")
DT, H, HORIZON_S = 0.1, 60, 6.0
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [k - 1 for k in HORIZONS]
V_FLOOR, KAPPA_CAP, SEL_ACCEL_MAX = 4.0, 0.12, 2.0
G = 9.81
deg = np.degrees

d = np.load(os.path.join(OUT, "per_window.npz"))
v0 = d["v0"]; eid = d["eid"]; ws = d["ws"]; full6s = d["full6s"]
dem_thx = d["dem_thx"]; dem_eb = d["dem_eb"]; sup_eb = d["sup_eb"]
sup_thx = d["sup_thx"]; ade_ship = d["ade_oiv"]
N = len(v0)
rep = {}

# ---- rebuild the GT slots (needed for the amended-vocabulary ADE) ----------
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


def grid(a_lat_max, n_lat, a_lo=-4.0, a_hi=3.0, n_lon=13, extra=None):
    a_g = np.linspace(a_lo, a_hi, n_lon)
    a_g = np.clip(a_g - a_g[np.abs(a_g).argmin()], a_lo, a_hi)
    c_g = np.linspace(-1.0, 1.0, n_lat) * a_lat_max
    assert np.any(a_g == 0.0) and np.any(c_g == 0.0)
    aa, cc = np.meshgrid(a_g, c_g, indexing="ij")
    ctrl = np.stack([aa.ravel(), cc.ravel()], -1)
    if extra is not None:
        ctrl = np.concatenate([ctrl, extra], 0)
    return ctrl


def roll(ctrl, v0v, want_kamm=False, kappa_cap=KAPPA_CAP):
    B, M = len(v0v), len(ctrl)
    vv = np.maximum(v0v, V_FLOOR) ** 2
    kap = np.clip(ctrl[None, :, 1] / vv[:, None], -kappa_cap, kappa_cap)
    a = np.repeat(ctrl[None, :, 0], B, axis=0)
    x = np.zeros((B, M)); y = np.zeros((B, M)); yaw = np.zeros((B, M))
    v = np.repeat(v0v[:, None], M, axis=1).copy()
    wp = np.zeros((B, M, len(SLOTS), 2))
    peak = np.zeros((B, M))
    for k in range(H):
        if want_kamm:
            # realised accelerations at the START of the step, the state the
            # tyre actually sees: a_lat = v * yaw_rate = v^2 * kappa.
            alat = v ** 2 * kap
            alon = np.where(v > 0, a, np.maximum(a, 0.0))
            peak = np.maximum(peak, np.hypot(alon, alat))
        x = x + v * np.cos(yaw) * DT
        y = y + v * np.sin(yaw) * DT
        yaw = yaw + v * kap * DT
        v = np.maximum(v + a * DT, 0.0)
        if k in SLOTS:
            wp[:, :, SLOTS.index(k), 0] = x
            wp[:, :, SLOTS.index(k), 1] = y
    return wp, yaw, peak


# THE SHIPPED ARM IS THE LIVE ARTIFACT ITSELF, not an analytic rebuild: the pod
# stores `controls` in float32, and rebuilding the same grid in float64 moves the
# oracle ADE by ~4e-7 m. That is a DTYPE difference, and quoting a rebuild as
# "the shipped set" would detach the number from the bytes the A40 imports.
SHIP = torch.load(os.path.join(HERE, "pull", "anchors_live.pt"),
                  map_location="cpu", weights_only=False)["controls"].double().numpy()
assert SHIP.shape == (117, 2)
_rebuild = grid(3.0, 9)
print("[vocab] live artifact controls vs analytic 13x9 rebuild: max abs diff "
      "%.3e (float32 storage)" % float(np.abs(SHIP - _rebuild).max()))

# =========================================================================== #
# A. WHERE THE 8.85 % LIVES — the end-bearing hole, resolved by speed
# =========================================================================== #
print("=== A. the 8.85 % end-bearing hole, resolved by v0 ===")
hole = deg(sup_eb) <= 30.0
VB = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 99)]
rowsA = []
print(f"{'v0 band (m/s)':<16}{'n':>7}{'holE%':>8}{'maxEB':>8}{'maxTHx':>8}"
      f"{'GT|dyaw| p95':>14}{'GT eb p95':>11}{'dem>sup eb':>12}")
for lo, hi in VB:
    m = (v0 >= lo) & (v0 < hi)
    if not m.any():
        continue
    r = {"v0_band_ms": [lo, hi], "n": int(m.sum()),
         "n_episodes": int(len(np.unique(eid[m]))),
         "pct_no_gt30_end_bearing": float(100.0 * hole[m].mean()),
         "median_max_end_bearing_deg": float(np.median(deg(sup_eb[m]))),
         "median_max_terminal_heading_exact_deg": float(np.median(deg(sup_thx[m]))),
         "gt_dyaw_p95_deg": float(np.percentile(deg(dem_thx[m]), 95)),
         "gt_end_bearing_p95_deg": float(np.percentile(deg(dem_eb[m]), 95)),
         "pct_demand_exceeds_supply_end_bearing": float(
             100.0 * (dem_eb[m] > sup_eb[m]).mean()),
         "pct_demand_exceeds_supply_terminal_heading": float(
             100.0 * (dem_thx[m] > sup_thx[m]).mean()),
         "underpowered": bool(m.sum() < 100)}
    rowsA.append(r)
    print(f"[{lo:>2},{hi:>2}){'':<8}{r['n']:>7}{r['pct_no_gt30_end_bearing']:>7.2f}%"
          f"{r['median_max_end_bearing_deg']:>8.1f}"
          f"{r['median_max_terminal_heading_exact_deg']:>8.1f}"
          f"{r['gt_dyaw_p95_deg']:>14.1f}{r['gt_end_bearing_p95_deg']:>11.1f}"
          f"{r['pct_demand_exceeds_supply_end_bearing']:>11.2f}%"
          + ("  UNDERPOWERED" if r["underpowered"] else ""))
rep["A_hole_by_speed"] = rowsA
print(f"  => the hole is {100*hole.mean():.2f} % overall; it sits ENTIRELY at "
      f"v0 >= {v0[hole].min():.2f} m/s (min v0 among hole windows), "
      f"median {np.median(v0[hole]):.2f}")
rep["A_hole_v0"] = {"n": int(hole.sum()), "pct": float(100 * hole.mean()),
                    "v0_min": float(v0[hole].min()),
                    "v0_median": float(np.median(v0[hole])),
                    "v0_max": float(v0[hole].max()),
                    "gt_dyaw_p95_deg": float(np.percentile(deg(dem_thx[hole]), 95)),
                    "gt_end_bearing_p95_deg": float(np.percentile(deg(dem_eb[hole]), 95)),
                    "gt_end_bearing_max_deg": float(deg(dem_eb[hole]).max()),
                    "pct_of_hole_windows_where_demand_exceeds_supply": float(
                        100.0 * (dem_eb[hole] > sup_eb[hole]).mean())}

# the 12 windows where END-BEARING demand really does exceed supply
ex = dem_eb > sup_eb
print(f"\n  the {int(ex.sum())} windows where GT END-BEARING exceeds the best "
      f"candidate: v0 {v0[ex].min():.2f}..{v0[ex].max():.2f} m/s "
      f"(median {np.median(v0[ex]):.2f}), GT |dyaw| median "
      f"{np.median(deg(dem_thx[ex])):.1f} deg, in "
      f"{len(np.unique(eid[ex]))} episodes")
disp6 = np.linalg.norm(gt[:, -1], axis=-1)
print(f"  their GT 6 s DISPLACEMENT is {disp6[ex].min():.3f}..{disp6[ex].max():.3f} m "
      f"(median {np.median(disp6[ex]):.3f}); corpus median {np.median(disp6):.2f} m "
      f"-- a bearing measured on a chord this short is direction NOISE, not a "
      f"manoeuvre the vocabulary failed to supply")
rep["A_end_bearing_exceeding"] = {
    "gt_6s_displacement_m": {"min": float(disp6[ex].min()),
                             "median": float(np.median(disp6[ex])),
                             "max": float(disp6[ex].max()),
                             "corpus_median": float(np.median(disp6))},
    "n": int(ex.sum()), "n_episodes": int(len(np.unique(eid[ex]))),
    "v0_min": float(v0[ex].min()), "v0_max": float(v0[ex].max()),
    "v0_median": float(np.median(v0[ex])),
    "gt_dyaw_median_deg": float(np.median(deg(dem_thx[ex]))),
    "gt_end_bearing_median_deg": float(np.median(deg(dem_eb[ex]))),
    "oiv_ade_m": float(ade_ship[ex].mean())}

# =========================================================================== #
# B. KAMM CIRCLE of the shipped set and of every amendment, at 4 speeds
# =========================================================================== #
print("\n=== B. Kamm circle on the ROLLED path (a_lat = v(t)^2 * kappa, so a "
      "candidate that ACCELERATES raises its own lateral load) ===")
_hi_alat = np.array([[al, ac] for al in (-2.0, 0.0, 1.1667)
                     for ac in (-6.0, -4.5, 4.5, 6.0)])
AMEND = {
    "SHIPPED 13x9 a_lat_max 3.0 (117)": SHIP,
    "13x9 a_lat_max 4.5 (117)": grid(4.5, 9),
    "13x9 a_lat_max 6.0 (117)": grid(6.0, 9),
    "13x11 a_lat_max 3.0 (143)": grid(3.0, 11),
    "13x11 a_lat_max 4.5 (143)": grid(4.5, 11),
    "SHIPPED + 12 high-a_lat (129)": np.concatenate([SHIP, _hi_alat], 0),
    "SHIPPED + kappa_cap 0.25 (117)": SHIP,        # same controls, cap relaxed
}
kamm = {}
print(f"{'family':<36}{'n':>5}" + "".join(f"{f'v0={v}':>22}" for v in (10, 18, 27, 36)))
for nm, ctrl in AMEND.items():
    row = {"n": int(len(ctrl))}
    cells = []
    kc = 0.25 if "kappa_cap" in nm else KAPPA_CAP
    for vq in (10.0, 18.0, 27.0, 36.0):
        _, _, pk = roll(ctrl, np.array([vq]), want_kamm=True, kappa_cap=kc)
        g = pk[0] / G
        row[f"v0_{int(vq)}"] = {"n_over_mu0.7": int((g > 0.7).sum()),
                                "pct_over_mu0.7": float(100 * (g > 0.7).mean()),
                                "peak_g": float(g.max()),
                                "median_g": float(np.median(g))}
        cells.append(f"{int((g>0.7).sum()):>3}/{len(ctrl):<3} pk {g.max():>5.2f}g")
    kamm[nm] = row
    print(f"{nm:<36}{len(ctrl):>5}" + "".join(f"{c:>22}" for c in cells))
rep["B_kamm_mu070"] = kamm

# =========================================================================== #
# C. AMENDED-VOCABULARY COUNTERFACTUAL, PAIRED, SAME WINDOWS
# =========================================================================== #
print("\n=== C. oracle-in-vocabulary ADE of each amendment vs the SHIPPED set, "
      "paired episode-cluster bootstrap (taniteval.ci, n_boot 2000, seed 0) ===")
CH = 512
per_arm = {}
for nm, ctrl in AMEND.items():
    a_all = np.zeros(N)
    kc = 0.25 if "kappa_cap" in nm else KAPPA_CAP
    for s in range(0, N, CH):
        e = min(s + CH, N)
        wp, _, _ = roll(ctrl, v0[s:e], kappa_cap=kc)
        keep = sl.anchor_reachability_mask(
            torch.from_numpy(wp), torch.from_numpy(v0[s:e]),
            accel_max=SEL_ACCEL_MAX, horizon_s=HORIZON_S).numpy()
        dd = wp - gt[s:e, None]
        ade = (np.sqrt((dd ** 2).sum(-1)) * sv[s:e, None]).sum(-1) / svn[s:e, None]
        a_all[s:e] = np.where(keep, ade, np.inf).min(1)
    per_arm[nm] = a_all
    print(f"  {nm:<36} pooled OIV ADE {a_all.mean():.4f} m")

base = per_arm["SHIPPED 13x9 a_lat_max 3.0 (117)"]
chk = float(np.abs(base - ade_ship).max())
print(f"  C-CONTROL  the rebuilt SHIPPED arm reproduces part 1 to "
      f"{chk:.3e} m  {'PASS' if chk < 1e-12 else 'FAIL'}")
assert chk < 1e-12
rep["C_shipped_reproduction_max_abs_err_m"] = chk

TURN = deg(dem_thx) >= 30.0
cmp_rows = []
for nm, arm in per_arm.items():
    if nm == "SHIPPED 13x9 a_lat_max 3.0 (117)":
        continue
    row = {"arm": nm, "n_candidates": int(len(AMEND[nm]))}
    for tag, m in (("ALL", np.ones(N, bool)), ("GT_TURN_ge30deg", TURN)):
        bs = paired_episode_cluster_bootstrap(arm[m], base[m], eid[m],
                                              n_boot=2000, seed=0)
        row[tag] = {"arm_ade_m": float(arm[m].mean()),
                    "shipped_ade_m": float(base[m].mean()),
                    "delta_m": bs["delta"], "lo": bs["lo"], "hi": bs["hi"],
                    "separated": bs["separated"], "n_windows": bs["n_windows"],
                    "n_episodes": bs["n_episodes"],
                    "estimator": bs["estimator"]}
    cmp_rows.append(row)
    for tag in ("ALL", "GT_TURN_ge30deg"):
        r = row[tag]
        print(f"  {nm:<36} {tag:<16} {r['arm_ade_m']:.4f} vs "
              f"{r['shipped_ade_m']:.4f}  delta {r['delta_m']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  n {r['n_windows']}/"
              f"{r['n_episodes']}ep  "
              f"{'SEPARATED' if r['separated'] else 'not separated'}")
rep["C_amendments_vs_shipped"] = cmp_rows

# ---- the residual: what the BEST 117-budget family leaves on the table ----
rep["C_pooled"] = {nm: float(a.mean()) for nm, a in per_arm.items()}
rep["C_turn_bin_pooled"] = {nm: float(a[TURN].mean()) for nm, a in per_arm.items()}
rep["C_n"] = {"n_all": N, "n_turn_ge30": int(TURN.sum()),
              "n_ep_turn_ge30": int(len(np.unique(eid[TURN])))}

json.dump(rep, open(os.path.join(OUT, "TURNCOV6S_PART2.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "TURNCOV6S_PART2.json"))
