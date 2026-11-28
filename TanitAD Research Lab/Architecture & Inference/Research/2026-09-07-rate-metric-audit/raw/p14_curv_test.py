"""ANALYTIC CIRCLE TEST for the estimator that ACTUALLY produced the 84x:
the LOCAL ``curvature()`` in
  .../Research/2026-09-06-p4-p13-p14-validation/raw/p14_banked_fan.py:38

  kappa = |x' y'' - y' x''| / max((x'^2+y'^2)^{3/2}, 1e-6)
  d1 = np.gradient(p, ts, axis=-2);  d2 = np.gradient(d1, ts, axis=-2)

NOTE it passes the FULL time vector ``ts0``, so np.gradient handles the
non-uniform spacing itself -- it does NOT divide by one dt.

Decided by construction: circular arc, true curvature EXACTLY 1/R.
Null: straight line, true curvature EXACTLY 0.
Then a SPEED SWEEP, because the denominator is floored at 1e-6 (= 0.01 m/s).

ZERO GPU. numpy only.
"""
from __future__ import annotations
import json
import numpy as np

TICK = 0.1
V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)

TS0_V3 = np.concatenate([[0.0], np.array([h * TICK for h in V3_HORIZONS])])
TS0_UNI05 = np.concatenate([[0.0], 0.5 * np.arange(1, 9)])          # uniform 0.5 s
TS0_UNI075 = np.concatenate([[0.0], 0.75 * np.arange(1, 9)])        # uniform, 6 s span
TS0_PREFIX2S = np.concatenate([[0.0], 0.5 * np.arange(1, 5)])       # gate arm's grid
TS0_DENSE = np.concatenate([[0.0], TICK * np.arange(1, 61)])

GRIDS = {"A_v3_NONUNIFORM": TS0_V3, "B_uniform_0.5s": TS0_UNI05,
         "C_uniform_span6s": TS0_UNI075, "D_uniform_2s_prefix": TS0_PREFIX2S,
         "E_dense_0.1s": TS0_DENSE}


def curvature(p, ts):
    """VERBATIM from p14_banked_fan.py:38-47."""
    d1 = np.gradient(p, ts, axis=-2)
    d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)


def arc_with_origin(R, v, ts0):
    """ts0 ALREADY contains t=0, so this is path_with_origin's output."""
    th = v * ts0 / R
    return np.stack([R * np.sin(th), R * (1 - np.cos(th))], -1)[None]


def line_with_origin(v, ts0):
    return np.stack([v * ts0, np.zeros_like(ts0)], -1)[None]


out = {}

# ---- 1. ARC: does it recover 1/R on the REAL non-uniform grid? ------------
arc_rows = []
for R in (20.0, 50.0, 200.0, 1000.0):
    for gname, ts0 in GRIDS.items():
        k = curvature(arc_with_origin(R, 10.0, ts0), ts0)[0]
        arc_rows.append({"grid": gname, "R": R, "v_mps": 10.0,
                         "true_kappa": 1.0 / R,
                         "recovered_mean": float(k.mean()),
                         "ratio": float(k.mean()) / (1.0 / R),
                         "recovered_max": float(k.max())})
out["arc"] = arc_rows

# ---- 2. NULL: straight line, true curvature EXACTLY 0 --------------------
out["null_straight"] = [
    {"grid": g, "max_abs_kappa": float(np.abs(curvature(line_with_origin(10.0, ts0), ts0)).max())}
    for g, ts0 in GRIDS.items()]

# ---- 3. SPEED SWEEP on the REAL v3 grid: the 1e-6 denominator floor ------
# True curvature is 1/R at EVERY speed. A correct estimator is speed-blind.
sweep = []
R = 50.0
for v in (10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0):
    a = curvature(arc_with_origin(R, v, TS0_V3), TS0_V3)[0]
    s = curvature(line_with_origin(v, TS0_V3), TS0_V3)[0]
    sweep.append({"v_mps": v, "true_kappa": 1.0 / R,
                  "arc_kappa_mean": float(a.mean()),
                  "arc_ratio": float(a.mean()) / (1.0 / R),
                  "straight_floor_kappa_mean": float(s.mean())})
out["speed_sweep_R50_v3grid"] = sweep

# ---- 4. THE 84x, RECONSTRUCTED --------------------------------------------
# kmae = mean|k_pred - k_gt|.  A straight-line floor has k_pred == 0 EXACTLY,
# so its kmae is mean|k_gt| and it is IMMUNE to the denominator blow-up.
# A model that predicts a SLOW plan is exposed. Mix slow windows into a
# realistic corpus and read the contrast.
rng = np.random.default_rng(0)
N = 881                       # the p14 corpus size
gt_speeds = np.abs(rng.normal(9.0, 4.0, N)).clip(0.0, 25.0)
gt_R = rng.choice([36.0, 80.0, 400.0, 1e5], N, p=[.25, .25, .25, .25])
recon = []
for frac_slow, label in ((0.0, "no slow windows"), (0.02, "2% slow"),
                         (0.05, "5% slow"), (0.10, "10% slow")):
    sp = gt_speeds.copy()
    n_slow = int(frac_slow * N)
    sp[:n_slow] = rng.uniform(0.0, 0.30, n_slow)      # crawling / stopped
    kt, kg = [], []
    for i in range(N):
        # model predicts the window WELL: same speed, same radius
        kt.append(curvature(arc_with_origin(gt_R[i], sp[i], TS0_V3), TS0_V3)[0])
        kg.append(curvature(arc_with_origin(gt_R[i], sp[i], TS0_V3), TS0_V3)[0])
    kt, kg = np.array(kt), np.array(kg)
    model_kmae = np.abs(kt - kg).mean()               # == 0: a PERFECT model
    # the straight floor: k_pred == 0 exactly
    floor_kmae = np.abs(0.0 - kg).mean()
    recon.append({"case": label, "frac_slow": frac_slow,
                  "perfect_model_kmae": float(model_kmae),
                  "straight_floor_kmae": float(floor_kmae),
                  "mean_abs_k_gt": float(np.abs(kg).mean())})
out["reconstruction_perfect_model"] = recon

# ---- 5. The decisive one: model predicts a SLIGHTLY wrong slow plan -------
# Same corpus, but the model's speed is off by 5% -- a tiny, realistic error.
recon2 = []
for frac_slow in (0.0, 0.01, 0.02, 0.05, 0.10):
    sp = gt_speeds.copy()
    n_slow = int(frac_slow * N)
    sp[:n_slow] = rng.uniform(0.0, 0.30, n_slow)
    kt = np.array([curvature(arc_with_origin(gt_R[i], sp[i] * 1.05, TS0_V3), TS0_V3)[0]
                   for i in range(N)])
    kg = np.array([curvature(arc_with_origin(gt_R[i], sp[i], TS0_V3), TS0_V3)[0]
                   for i in range(N)])
    m = np.abs(kt - kg).mean()
    f = np.abs(0.0 - kg).mean()
    recon2.append({"frac_slow": frac_slow, "model_kmae": float(m),
                   "straight_floor_kmae": float(f),
                   "ratio_model_over_floor": float(m / f)})
out["reconstruction_5pct_speed_error"] = recon2

print(json.dumps(out, indent=1))
