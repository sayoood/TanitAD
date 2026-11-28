"""TURN COVERAGE **AND DEMAND** at the 6 s horizon for the LIVE refcv4b vocabulary.

Settles the launch package's own flagged UNVERIFIED: it measured `8.85 %` of
windows with no candidate whose END-BEARING turns > 30 deg and did NOT measure
the TERMINAL-HEADING equivalent, nor the DEMAND the corpus actually places.

⛔ MODEL-FREE. No forward pass, no checkpoint. Every number here is a property of
(the vocabulary, the reach clamp, the corpus) and nothing else, so it is a
CEILING/SUPPLY statement and may never be compared to an achievement
(`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).

DEFINITIONS, DECLARED BEFORE ANY NUMBER
=======================================
For a path P (candidate or GT) over 0..6 s in the ego frame at t0:
  * END-BEARING     eb   = |atan2(y(6s), x(6s))|            -- the CHORD angle
  * TERMINAL-HEAD.  th_s = |atan2(dy, dx)| over the 5->6 s slot segment
                           -- EXACTLY the launch instrument's definition
  * TERMINAL-HEAD.  th_x = |yaw(6s) - yaw(0)|               -- EXACT, integrated
                           (candidates: the unicycle's own yaw state;
                            GT: the corpus pose yaw column, wrapped)
These three are NEVER conflated: all three are reported side by side.

SUPPLY  = max over the RE ACH-CLAMPED survivors at this window's v0.
DEMAND  = the same statistic computed on the GROUND-TRUTH 6 s path.
THE DECISION NUMBER = P(DEMAND > SUPPLY) and the ADE penalty on exactly those
windows.

CONTROLS THAT MUST READ KNOWN VALUES
====================================
 K1 v0 round-trip     : the dump's v0 must equal poses[ws, 3] to 1e-5 on every
                        window, or the window->pose mapping is wrong and every
                        GT number below is void.
 K2 integrator parity : my vectorised roll must reproduce
                        `refa_v1_plan.unicycle_paths` bit-close (< 1e-9).
 K3 straight control  : the {a_lon=0, a_lat=0} candidate must have th_x = eb =
                        th_s = 0 EXACTLY and y == 0 to < 1e-12.
 K4 zero-path control : ADE of the constant-zero predictor must equal the mean
                        ||GT|| recomputed independently in float64.
 K5 n and d printed for every cell.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = r"C:\Users\Admin\run_refcv4v\repo\stack"
CI_PKG = r"C:\Users\Admin\run_refcv4v\repo\taniteval"
sys.path.insert(0, STACK)
sys.path.insert(0, CI_PKG)
from tanitad.refs.refa_v1_plan import unicycle_paths            # noqa: E402
import tanitad.refs.refc_select as sl                           # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap       # noqa: E402

import tanitad                                                  # noqa: E402
print("[env] tanitad imported from", tanitad.__file__)

OUT = os.path.join(HERE, "turncov_out")
os.makedirs(OUT, exist_ok=True)

DUMP = os.path.join(HERE, "refcv3_40284_dump")
MANI = os.path.join(HERE, "pull", "_v2manifest.pt")
ANCH = os.path.join(HERE, "pull", "anchors_live.pt")

# ---- the LIVE run's own constants (config.json of refcv4b-b1-v72-40k) ------
DT = 0.1
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [k - 1 for k in HORIZONS]
H = max(HORIZONS)                       # 60 roll steps == 6.0 s
HORIZON_S = 6.0
V_FLOOR = 4.0
KAPPA_CAP = 0.12
SEL_ACCEL_MAX = 2.0                     # config.selection.sel_accel_max
N_STACK_OFFSET_NOTE = ("manifest poses are ALREADY sliced by n_stack-1, i.e. "
                       "they are PROVIDER rows, and `ws` is a PROVIDER index")

rng_report: dict = {"_definitions": __doc__}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================================================================== #
# 1. the surface: windows from the banked dump, poses from the v2 manifest
# =========================================================================== #
man = torch.load(MANI, map_location="cpu", weights_only=False)
poses_by_clip = {cid: man["poses"][i].double().numpy()
                 for i, cid in enumerate(man["clip_id"])}
print(f"[manifest] {len(poses_by_clip)} clips; "
      f"T_out min/median/max = {min(man['T_out'])}/"
      f"{int(np.median(man['T_out']))}/{max(man['T_out'])}")

dman = json.load(open(os.path.join(DUMP, "manifest.json")))
eps_meta = dman["episodes"]

W_v0, W_ws, W_eid, W_clip = [], [], [], []
for e in eps_meta:
    fi = e["file_index"]
    z = np.load(os.path.join(DUMP, f"ep{fi:03d}.npz"))
    n = int(z["v0"].shape[0])
    assert n == e["n_windows"], (n, e)
    W_v0.append(z["v0"].astype(np.float64))
    W_ws.append(z["ws"].astype(np.int64))
    W_eid.append(np.full(n, fi, dtype=np.int64))
    W_clip += [e["clip_id"]] * n
v0 = np.concatenate(W_v0)
ws = np.concatenate(W_ws)
eid = np.concatenate(W_eid)
N = len(v0)
print(f"[surface] {N} windows / {len(eps_meta)} episodes "
      f"(dump grid '{dman['grid']['name']}', stride {dman['grid']['window_stride']})")
assert N == dman["grid"]["n_windows"] == 4823, N

# ---- K1: the v0 round-trip control ---------------------------------------
gt_full = np.zeros((N, H, 2))       # ego-frame GT xy at 0.1 s .. 6.0 s
gt_yaw = np.zeros((N, H))           # ego-frame GT heading change, wrapped
valid = np.zeros((N, H), dtype=bool)
v0_chk = np.zeros(N)
T_out = np.zeros(N, dtype=np.int64)
for i in range(N):
    P = poses_by_clip[W_clip[i]]
    T = P.shape[0]
    T_out[i] = T
    t0 = int(ws[i])
    v0_chk[i] = P[t0, 3]
    idx = np.clip(np.arange(t0 + 1, t0 + 1 + H), None, T - 1)
    valid[i] = np.arange(t0 + 1, t0 + 1 + H) <= (T - 1)
    d = P[idx, :2] - P[t0, :2]
    c, s = math.cos(-P[t0, 2]), math.sin(-P[t0, 2])
    gt_full[i, :, 0] = d[:, 0] * c - d[:, 1] * s
    gt_full[i, :, 1] = d[:, 0] * s + d[:, 1] * c
    gt_yaw[i] = wrap(P[idx, 2] - P[t0, 2])
k1 = float(np.abs(v0_chk - v0).max())
print(f"K1 CONTROL  v0 round-trip  max|dump.v0 - poses[ws,3]| = {k1:.3e}   "
      f"{'PASS' if k1 < 1e-5 else 'FAIL'}")
assert k1 < 1e-5, k1
rng_report["K1_v0_roundtrip_max_abs_err"] = k1

full6s = valid[:, H - 1]
print(f"[horizon]  windows with a TRUE (unclamped) 6 s future: "
      f"{int(full6s.sum())}/{N} = {100 * full6s.mean():.2f} %   "
      f"(the rest are END-CLAMPED by refc_v3_train.py:366 and are reported "
      f"separately, never silently pooled)")
rng_report["n_windows"] = N
rng_report["n_episodes"] = len(eps_meta)
rng_report["n_windows_true_6s"] = int(full6s.sum())
rng_report["v0_stats_ms"] = {"mean": float(v0.mean()), "p05": float(np.percentile(v0, 5)),
                             "median": float(np.median(v0)),
                             "p95": float(np.percentile(v0, 95)),
                             "max": float(v0.max())}

# =========================================================================== #
# 2. the LIVE vocabulary, rolled at every window's own v0
# =========================================================================== #
art = torch.load(ANCH, map_location="cpu", weights_only=False)
CTRL = art["controls"].double().numpy()             # [117, 2] = (a_lon, a_lat)
NA = CTRL.shape[0]
a_lon_g = np.unique(np.round(CTRL[:, 0], 6))
a_lat_g = np.unique(np.round(CTRL[:, 1], 6))
print(f"[vocab]   {NA} controls = {len(a_lon_g)} a_lon x {len(a_lat_g)} a_lat; "
      f"a_lon {a_lon_g.min():+.4f}..{a_lon_g.max():+.4f}  "
      f"a_lat {a_lat_g.min():+.3f}..{a_lat_g.max():+.3f} m/s^2")
zero_i = int(np.flatnonzero((CTRL[:, 0] == 0) & (CTRL[:, 1] == 0))[0])
rng_report["vocabulary"] = {
    "n": NA, "a_lon_grid": [float(x) for x in a_lon_g],
    "a_lat_grid": [float(x) for x in a_lat_g],
    "straight_index": zero_i,
    "kappa": "clamp(a_lat / max(v0, 4.0)^2, -0.12, 0.12)  [refc.py:1351-1356]",
    "sha256_matches_live_pod_anchors": True}


def roll(ctrl: np.ndarray, v0v: np.ndarray):
    """Vectorised copy of `rollout_unicycle` (SAME update order, refc.py rolls
    it in float32 -- we use float64, which can only be more accurate).

    ctrl [M, 2] = (a_lon, a_lat);  v0v [B]
    -> wp [B, M, S, 2]   xy at the 8 slot horizons
       yaw6 [B, M]       EXACT integrated heading at 6 s
       segh [B, M]       heading of the 5->6 s slot segment
    """
    B, M = len(v0v), len(ctrl)
    vv = np.maximum(v0v, V_FLOOR) ** 2
    kap = np.clip(ctrl[None, :, 1] / vv[:, None], -KAPPA_CAP, KAPPA_CAP)  # [B,M]
    a = np.repeat(ctrl[None, :, 0], B, axis=0)                            # [B,M]
    x = np.zeros((B, M)); y = np.zeros((B, M)); yaw = np.zeros((B, M))
    v = np.repeat(v0v[:, None], M, axis=1).copy()
    wp = np.zeros((B, M, len(SLOTS), 2))
    for k in range(H):
        x = x + v * np.cos(yaw) * DT
        y = y + v * np.sin(yaw) * DT
        yaw = yaw + v * kap * DT
        v = np.maximum(v + a * DT, 0.0)
        if k in SLOTS:
            j = SLOTS.index(k)
            wp[:, :, j, 0] = x
            wp[:, :, j, 1] = y
    seg = wp[:, :, -1] - wp[:, :, -2]
    segh = np.arctan2(seg[..., 1], seg[..., 0])
    return wp, yaw, segh


# ---- K2: integrator parity against the programme's own function ----------
# ⚠️ TWO reference calls, because `unicycle_paths` builds `state0` at the TORCH
# DEFAULT DTYPE (refa_v1_plan.py:319 `torch.zeros(n, 4)` -> float32), so its
# first integration step runs in float32 no matter what the controls are. That
# is a DTYPE difference, not a model difference, and quoting it as a parity
# failure would be the same class of error as reading `df` on a pod. The strict
# parity test therefore calls `rollout_unicycle` -- the same function
# `unicycle_paths` wraps -- with an explicitly float64 `state0`; the float32
# path is reported alongside as the magnitude of the dtype effect (the live
# decoder rolls in float32, refc.py:1344).
from tanitad.models.kinematic import rollout_unicycle              # noqa: E402
_s = np.linspace(0, N - 1, 64).astype(int)
wp_s, yaw_s, _ = roll(CTRL, v0[_s])
err = 0.0
err32 = 0.0
for j, i in enumerate(_s):
    kap = np.clip(CTRL[:, 1] / max(v0[i], V_FLOOR) ** 2, -KAPPA_CAP, KAPPA_CAP)
    c = torch.from_numpy(np.stack([CTRL[:, 0], kap], -1)).double()
    ctl = c[:, None, :].expand(-1, H, -1).contiguous()
    st0 = torch.zeros(NA, 4, dtype=torch.float64)
    st0[:, 3] = float(v0[i])
    p64 = rollout_unicycle(st0, ctl, dt=DT)[..., :2].numpy()[:, SLOTS]
    err = max(err, float(np.abs(p64 - wp_s[j]).max()))
    p32 = unicycle_paths(ctl, torch.tensor(float(v0[i]), dtype=torch.float64),
                         DT, action_units="kappa").numpy()[:, SLOTS]
    err32 = max(err32, float(np.abs(p32 - wp_s[j]).max()))
print(f"K2 CONTROL  integrator parity vs rollout_unicycle(float64 state0): "
      f"max abs err {err:.3e} m   {'PASS' if err < 1e-12 else 'FAIL'}   "
      f"[unicycle_paths' float32-state0 path differs by {err32:.3e} m -- dtype, "
      f"not model]")
assert err < 1e-12, err
rng_report["K2_integrator_parity_max_abs_err_m"] = err
rng_report["K2_float32_state0_path_delta_m"] = err32

# ---- K3: the straight-ahead control -------------------------------------
k3 = {"y_max_abs_m": float(np.abs(wp_s[:, zero_i, :, 1]).max()),
      "yaw6_max_abs_rad": float(np.abs(yaw_s[:, zero_i]).max())}
print(f"K3 CONTROL  straight-ahead anchor idx {zero_i}: max|y| "
      f"{k3['y_max_abs_m']:.3e} m, max|yaw(6s)| {k3['yaw6_max_abs_rad']:.3e} rad  "
      f"{'PASS' if max(k3.values()) < 1e-12 else 'FAIL'}")
assert max(k3.values()) < 1e-12
rng_report["K3_straight_control"] = k3

# =========================================================================== #
# 3. SUPPLY: roll the whole surface, clamp, and take the three turn metrics
# =========================================================================== #
CH = 512
sup_eb = np.zeros(N); sup_ths = np.zeros(N); sup_thx = np.zeros(N)
sup_eb_all = np.zeros(N); sup_thx_all = np.zeros(N)
n_surv = np.zeros(N, dtype=np.int64)
cnt_eb30 = np.zeros(N, dtype=np.int64); cnt_thx30 = np.zeros(N, dtype=np.int64)
ade_oiv = np.zeros(N); along_oiv = np.zeros(N); lat_oiv = np.zeros(N)
ade_oiv_full = np.zeros(N)          # unclamped 117 (no reach filter)
best_idx = np.zeros(N, dtype=np.int64)
ade_ha0 = np.zeros(N)               # the straight-ahead anchor, always present
ade_zero = np.zeros(N)              # the constant-zero predictor (K4)
gt_slots = gt_full[:, SLOTS]                             # [N, 8, 2]
sv = valid[:, SLOTS].astype(np.float64)                  # [N, 8]
sv_n = np.maximum(sv.sum(1), 1.0)

for s in range(0, N, CH):
    e = min(s + CH, N)
    wp, yaw6, segh = roll(CTRL, v0[s:e])                 # [b, M, 8, 2]
    keep = sl.anchor_reachability_mask(
        torch.from_numpy(wp), torch.from_numpy(v0[s:e]),
        accel_max=SEL_ACCEL_MAX, horizon_s=HORIZON_S).numpy()          # [b, M]
    eb = np.abs(np.arctan2(wp[:, :, -1, 1], wp[:, :, -1, 0]))
    thx = np.abs(yaw6)
    ths = np.abs(segh)
    n_surv[s:e] = keep.sum(1)
    neg = -1.0
    sup_eb[s:e] = np.where(keep, eb, neg).max(1)
    sup_thx[s:e] = np.where(keep, thx, neg).max(1)
    sup_ths[s:e] = np.where(keep, ths, neg).max(1)
    sup_eb_all[s:e] = eb.max(1)
    sup_thx_all[s:e] = thx.max(1)
    cnt_eb30[s:e] = (keep & (eb > np.radians(30))).sum(1)
    cnt_thx30[s:e] = (keep & (thx > np.radians(30))).sum(1)
    d = wp - gt_slots[s:e, None]                                       # [b,M,8,2]
    l2 = np.sqrt((d ** 2).sum(-1))                                     # [b,M,8]
    ade = (l2 * sv[s:e, None]).sum(-1) / sv_n[s:e, None]
    ade_oiv_full[s:e] = ade.min(1)
    ade_m = np.where(keep, ade, np.inf)
    bi = ade_m.argmin(1)
    best_idx[s:e] = bi
    r = np.arange(e - s)
    ade_oiv[s:e] = ade_m[r, bi]
    along_oiv[s:e] = (np.abs(d[r, bi, :, 0]) * sv[s:e]).sum(-1) / sv_n[s:e]
    lat_oiv[s:e] = (np.abs(d[r, bi, :, 1]) * sv[s:e]).sum(-1) / sv_n[s:e]
    ade_ha0[s:e] = ade[:, zero_i]
    ade_zero[s:e] = ((np.sqrt((gt_slots[s:e] ** 2).sum(-1)) * sv[s:e]).sum(-1)
                     / sv_n[s:e])

assert (n_surv > 0).all(), "a window with NO surviving candidate"
print(f"[clamp]   a_max {SEL_ACCEL_MAX} (band +/-{SEL_ACCEL_MAX*HORIZON_S:.1f} m/s): "
      f"survivors/window {n_surv.mean():.1f} of {NA} "
      f"({100*(1-n_surv.mean()/NA):.2f} % killed), empty windows 0")

# ---- K4: zero-path control ------------------------------------------------
indep = np.array([float((np.sqrt((gt_slots[i] ** 2).sum(-1)) * sv[i]).sum()
                        / sv_n[i]) for i in range(0, N, 97)])
k4 = float(np.abs(indep - ade_zero[::97]).max())
print(f"K4 CONTROL  constant-zero ADE == mean||GT||: max abs err {k4:.3e} m  "
      f"{'PASS' if k4 < 1e-12 else 'FAIL'}")
assert k4 < 1e-12
rng_report["K4_zero_path_control_max_abs_err_m"] = k4

# =========================================================================== #
# 4. DEMAND: what the corpus actually asks for
# =========================================================================== #
dem_thx = np.abs(gt_yaw[:, H - 1])                       # EXACT, from pose yaw
dem_eb = np.abs(np.arctan2(gt_full[:, H - 1, 1], gt_full[:, H - 1, 0]))
gseg = gt_slots[:, -1] - gt_slots[:, -2]
dem_ths = np.abs(np.arctan2(gseg[:, 1], gseg[:, 0]))

deg = np.degrees


def cov_table(sup, name, mask=None):
    m = np.ones(N, bool) if mask is None else mask
    row = {"metric": name, "n": int(m.sum())}
    for th in (15, 30, 45, 60):
        row[f"no_cand_gt{th}deg_pct"] = float(
            100.0 * (deg(sup[m]) <= th).mean())
    row["max_available_deg_median"] = float(np.median(deg(sup[m])))
    row["max_available_deg_p05"] = float(np.percentile(deg(sup[m]), 5))
    return row


sup_tbl = [cov_table(sup_eb, "END-BEARING (clamped survivors)"),
           cov_table(sup_ths, "TERMINAL-HEADING, 5->6 s slot segment (clamped)"),
           cov_table(sup_thx, "TERMINAL-HEADING, EXACT integrated yaw (clamped)"),
           cov_table(sup_eb_all, "END-BEARING (all 117, no reach clamp)"),
           cov_table(sup_thx_all, "TERMINAL-HEADING EXACT (all 117, no clamp)")]
print("\n=== SUPPLY: windows with NO candidate exceeding theta ===")
print(f"{'metric':<52}{'n':>6}{'>15':>9}{'>30':>9}{'>45':>9}{'>60':>9}"
      f"{'medMax':>9}")
for r in sup_tbl:
    print(f"{r['metric']:<52}{r['n']:>6}"
          f"{r['no_cand_gt15deg_pct']:>8.2f}%{r['no_cand_gt30deg_pct']:>8.2f}%"
          f"{r['no_cand_gt45deg_pct']:>8.2f}%{r['no_cand_gt60deg_pct']:>8.2f}%"
          f"{r['max_available_deg_median']:>9.1f}")
rng_report["supply"] = sup_tbl
rng_report["turns_gt30_per_window"] = {
    "end_bearing": float(cnt_eb30.mean()),
    "terminal_heading_exact": float(cnt_thx30.mean())}
print(f"\n[per-window counts] >30 deg candidates surviving: "
      f"{cnt_eb30.mean():.1f} by end-bearing / {cnt_thx30.mean():.1f} by "
      f"EXACT terminal heading (launch package reported 48.3 for the coarse "
      f"terminal-heading count)")


def dem_table(dem, name, mask):
    d = deg(dem[mask])
    row = {"metric": name, "n": int(mask.sum())}
    for th in (5, 15, 30, 45, 60, 90):
        row[f"gt{th}deg_pct"] = float(100.0 * (d > th).mean())
        row[f"gt{th}deg_n"] = int((d > th).sum())
    row["p50"] = float(np.percentile(d, 50))
    row["p95"] = float(np.percentile(d, 95))
    row["p99"] = float(np.percentile(d, 99))
    row["max"] = float(d.max())
    return row


dem_tbl = [dem_table(dem_thx, "GT TERMINAL-HEADING EXACT (|dyaw| over 6 s)", full6s),
           dem_table(dem_eb, "GT END-BEARING", full6s),
           dem_table(dem_ths, "GT TERMINAL-HEADING, 5->6 s segment", full6s),
           dem_table(dem_thx, "GT TERMINAL-HEADING EXACT (ALL windows, incl. "
                              "end-clamped)", np.ones(N, bool))]
print("\n=== DEMAND: what the corpus asks for over 6 s ===")
print(f"{'metric':<58}{'n':>6}{'>5':>8}{'>15':>8}{'>30':>8}{'>45':>8}{'>60':>8}"
      f"{'p95':>8}{'max':>8}")
for r in dem_tbl:
    print(f"{r['metric']:<58}{r['n']:>6}{r['gt5deg_pct']:>7.2f}%"
          f"{r['gt15deg_pct']:>7.2f}%{r['gt30deg_pct']:>7.2f}%"
          f"{r['gt45deg_pct']:>7.2f}%{r['gt60deg_pct']:>7.2f}%"
          f"{r['p95']:>8.1f}{r['max']:>8.1f}")
rng_report["demand"] = dem_tbl

# =========================================================================== #
# 5. THE DECISION NUMBER: demand > supply, and its ADE penalty
# =========================================================================== #
print("\n=== INTERSECTION: DEMAND vs SUPPLY, per-window and paired ===")
inter = {}
for nm, dm, sp in (("exact_terminal_heading", dem_thx, sup_thx),
                   ("end_bearing", dem_eb, sup_eb),
                   ("segment_terminal_heading", dem_ths, sup_ths)):
    ex = dm > sp                                   # demand exceeds best supply
    for tag, m in (("all_windows", np.ones(N, bool)), ("true_6s", full6s)):
        k = f"{nm}/{tag}"
        sel = ex & m
        inter[k] = {
            "n": int(m.sum()), "n_exceeding": int(sel.sum()),
            "pct_exceeding": float(100.0 * sel.sum() / max(m.sum(), 1)),
            "median_shortfall_deg": (float(np.median(deg(dm[sel] - sp[sel])))
                                     if sel.any() else 0.0),
            "p95_shortfall_deg": (float(np.percentile(deg(dm[sel] - sp[sel]), 95))
                                  if sel.any() else 0.0),
            "max_shortfall_deg": (float(deg(dm[sel] - sp[sel]).max())
                                  if sel.any() else 0.0),
            "oiv_ade_on_exceeding_m": float(ade_oiv[sel].mean()) if sel.any() else 0.0,
            "oiv_ade_on_rest_m": float(ade_oiv[m & ~ex].mean()),
            "n_episodes_exceeding": int(len(np.unique(eid[sel]))) if sel.any() else 0,
        }
        print(f"  {k:<48} n {m.sum():>5}  exceeding {sel.sum():>5} "
              f"({inter[k]['pct_exceeding']:5.2f} %)  med shortfall "
              f"{inter[k]['median_shortfall_deg']:5.1f} deg  OIV-ADE on those "
              f"{inter[k]['oiv_ade_on_exceeding_m']:.4f} m vs rest "
              f"{inter[k]['oiv_ade_on_rest_m']:.4f} m")
rng_report["intersection"] = inter

# =========================================================================== #
# 6. ORACLE-IN-VOCABULARY ADE, SPLIT BY GT TURN MAGNITUDE (ALONG + LAT apart)
# =========================================================================== #
BINS = [(0, 5), (5, 15), (15, 30), (30, 45), (45, 90), (90, 1e9)]
print("\n=== ORACLE-IN-VOCABULARY (MODEL-FREE CEILING) by GT |dyaw| over 6 s ===")
print(f"{'bin (deg)':<12}{'n':>7}{'n_ep':>6}{'ADE':>9}{'ALONG':>9}{'LAT':>9}"
      f"{'ha0 ADE':>10}{'zero ADE':>10}{'supply<dem':>11}")
bin_rows = []
lab = np.full(N, -1)
for bi, (lo, hi) in enumerate(BINS):
    m = (deg(dem_thx) >= lo) & (deg(dem_thx) < hi)
    lab[m] = bi
    if not m.any():
        continue
    row = {"bin_deg": [lo, hi if hi < 1e8 else None], "n": int(m.sum()),
           "n_episodes": int(len(np.unique(eid[m]))),
           "oiv_ade_m": float(ade_oiv[m].mean()),
           "oiv_along_mae_m": float(along_oiv[m].mean()),
           "oiv_lat_mae_m": float(lat_oiv[m].mean()),
           "straight_anchor_ade_m": float(ade_ha0[m].mean()),
           "zero_path_ade_m": float(ade_zero[m].mean()),
           "pct_demand_exceeds_supply": float(
               100.0 * (dem_thx[m] > sup_thx[m]).mean()),
           "oiv_ade_unclamped_117_m": float(ade_oiv_full[m].mean()),
           "underpowered": bool(m.sum() < 100 or len(np.unique(eid[m])) < 10)}
    bin_rows.append(row)
    print(f"{f'[{lo},{hi if hi<1e8 else 999})':<12}{row['n']:>7}"
          f"{row['n_episodes']:>6}{row['oiv_ade_m']:>9.4f}"
          f"{row['oiv_along_mae_m']:>9.4f}{row['oiv_lat_mae_m']:>9.4f}"
          f"{row['straight_anchor_ade_m']:>10.4f}{row['zero_path_ade_m']:>10.4f}"
          f"{row['pct_demand_exceeds_supply']:>10.2f}%"
          + ("   UNDERPOWERED" if row["underpowered"] else ""))
rng_report["oiv_by_turn_bin"] = bin_rows
rng_report["oiv_pooled"] = {
    "ade_m": float(ade_oiv.mean()), "along_mae_m": float(along_oiv.mean()),
    "lat_mae_m": float(lat_oiv.mean()),
    "straight_anchor_ade_m": float(ade_ha0.mean()),
    "zero_path_ade_m": float(ade_zero.mean()),
    "n": N, "n_episodes": len(eps_meta), "d_candidates": NA}
print(f"\n[pooled]  OIV ADE {ade_oiv.mean():.4f} m  (ALONG {along_oiv.mean():.4f} "
      f"/ LAT {lat_oiv.mean():.4f})   straight-anchor {ade_ha0.mean():.4f}   "
      f"zero-path {ade_zero.mean():.4f}   n {N} windows / {len(eps_meta)} eps, "
      f"d = {NA} candidates")

np.savez_compressed(os.path.join(OUT, "per_window.npz"),
                    v0=v0, ws=ws, eid=eid, full6s=full6s,
                    dem_thx=dem_thx, dem_eb=dem_eb, dem_ths=dem_ths,
                    sup_thx=sup_thx, sup_eb=sup_eb, sup_ths=sup_ths,
                    ade_oiv=ade_oiv, along_oiv=along_oiv, lat_oiv=lat_oiv,
                    ade_ha0=ade_ha0, ade_zero=ade_zero, n_surv=n_surv,
                    best_idx=best_idx, turn_bin=lab)
json.dump(rng_report, open(os.path.join(OUT, "TURNCOV6S.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "TURNCOV6S.json"))
