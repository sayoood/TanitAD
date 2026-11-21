"""Does a SPEED-CLAMPED curvature family still clear the gate?

The flat-kappa family shipped at 13a x 9k = 117 reads 0.2610 m
oracle-in-vocabulary and BEATS ha = 0.2996 -- but at v0 = 27.27 m/s
(95th pct) **104 of its 117 anchors break a mu = 0.7 friction circle**,
peak 3.96 g, because a constant kappa at speed gives a_lat = v^2 * kappa
(27^2 x 0.06 = 43.7 m/s^2 = 4.5 g). refcv4's fixed-path set broke NONE.

An unflyable candidate is not WRONG -- the oracle never picks it, because the
GT it is scored against is flyable -- but it is WASTED BUDGET, and at high speed
it wastes almost all of it. A physically-scaled family should therefore be at
least as good AND flyable.

FAMILY: kappa_j(v0) = c_j * min(KAPPA_CAP, A_LAT_MAX / max(v0, V_FLOOR)^2),
c_j in linspace(-1, 1, nk) so c = 0 -- the straight-ahead control -- is EXACT.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "..", "vocab", "refcv3_40284_dump")
OUT = os.path.join(HERE, "out")
DT, H = 0.1, max(V3_HORIZONS)
SLOT_ALL = [k - 1 for k in V3_HORIZONS]
HA_BAR = 0.2996
V_FLOOR, KAPPA_CAP = 4.0, 0.12

eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
G, V0, HA, EPI = [], [], [], []
for i, f in enumerate(eps):
    z = np.load(os.path.join(DUMP, f))
    G.append(z["g"]); V0.append(z["v0"]); HA.append(z["ha"])
    EPI.append(np.full(z["g"].shape[0], i, np.int64))
G = np.concatenate(G).astype(np.float64)
V0 = np.concatenate(V0).astype(np.float64)
HA = np.concatenate(HA).astype(np.float64)
EPI = np.concatenate(EPI)
N = len(G)
HA_PW = np.linalg.norm(HA - G, axis=-1).mean(1)
print("[surface] %d windows / %d episodes   ha = %.6f" % (N, len(eps), HA_PW.mean()))


def accel_grid(na, lo=-4.0, hi=3.0):
    a = np.linspace(lo, hi, na)
    a = np.clip(a - a[np.abs(a).argmin()], lo, hi)
    assert np.any(a == 0.0)
    return a


def roll_ctrl(ctrl, v0):
    c = torch.from_numpy(np.repeat(ctrl[:, None, :], H, axis=1)).double()
    return unicycle_paths(c, torch.tensor(float(v0), dtype=torch.float64), DT,
                          action_units="kappa").numpy()[:, SLOT_ALL, :]


def kamm_peak(P):
    t = np.array(V3_HORIZONS, np.float64) * DT
    v = np.gradient(P, t, axis=1)
    a = np.gradient(v, t, axis=1)
    sp = np.linalg.norm(v, axis=-1)
    lon = np.abs((v * a).sum(-1) / np.maximum(sp, 1e-6))
    lat = np.abs((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0])
                 / np.maximum(sp, 1e-6))
    return (np.sqrt(lon ** 2 + lat ** 2) / 9.81).max(1)


def boot(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(EPI); by = [np.where(EPI == e)[0] for e in uq]
    d = a - b
    o = np.array([d[np.concatenate([by[p] for p in
                                    rng.integers(0, len(uq), len(uq))])].mean()
                  for _ in range(n_boot)])
    return (float(d.mean()), float(np.percentile(o, 2.5)),
            float(np.percentile(o, 97.5)))


def scored(a_g, c_g, alat, tag):
    """kappa scaled per v0. Returns the gate row plus the Kamm census."""
    ade = np.empty(N); al = np.empty(N); la = np.empty(N)
    idx = np.empty(N, np.int64)
    key = np.round(V0, 3)
    cache = {}
    for i in range(N):
        k = key[i]
        if k not in cache:
            km = min(KAPPA_CAP, alat / max(k, V_FLOOR) ** 2)
            aa, cc = np.meshgrid(a_g, c_g * km, indexing="ij")
            ctrl = np.stack([aa.ravel(), cc.ravel()], -1)
            cache[k] = (roll_ctrl(ctrl, k), ctrl)
        P = cache[k][0][:, :4, :]
        d = np.linalg.norm(P - G[i][None], axis=-1).mean(1)
        j = int(d.argmin())
        ade[i] = d[j]; idx[i] = j
        r = P[j] - G[i]
        al[i] = np.abs(r[:, 0]).mean(); la[i] = np.abs(r[:, 1]).mean()
    dd, lo, hi = boot(ade, HA_PW)
    v = "BEATS ha" if hi < 0 else ("loses to ha" if lo > 0 else "tied with ha")
    M = len(a_g) * len(c_g)
    print("  %-46s M=%4d  ADE %.4f  ALONG %.4f  LAT %.4f  used %3d  "
          "vs ha %+.4f [%+.4f, %+.4f] -> %s"
          % (tag, M, ade.mean(), al.mean(), la.mean(), len(np.unique(idx)),
             dd, lo, hi, v))
    kam = {}
    for vq in (float(np.median(V0)), float(np.percentile(V0, 95))):
        km = min(KAPPA_CAP, alat / max(vq, V_FLOOR) ** 2)
        aa, cc = np.meshgrid(a_g, c_g * km, indexing="ij")
        pk = kamm_peak(roll_ctrl(np.stack([aa.ravel(), cc.ravel()], -1), vq))
        kam["v0_%.2f" % vq] = {"over_mu_0.7": int((pk > 0.7).sum()),
                               "peak_g": float(pk.max()),
                               "kappa_max": float(km)}
        print("       Kamm @ v0=%5.2f: kappa_max %.4f  over mu=0.7 %3d/%d  "
              "peak %.2f g" % (vq, km, int((pk > 0.7).sum()), M, pk.max()))
    return {"n_anchors": M, "ade_0_2s_m": float(ade.mean()),
            "along_mae_m": float(al.mean()), "lat_mae_m": float(la.mean()),
            "vs_ha": [dd, lo, hi], "verdict": v, "a_lat_max_ms2": alat,
            "kamm": kam}


print("\n=== SPEED-CLAMPED curvature families (kappa_max = a_lat_max / v0^2, "
      "capped %.2f, v floor %.1f m/s) ===" % (KAPPA_CAP, V_FLOOR))
res = {}
for na, nk, alat in ((13, 9, 4.0), (13, 9, 6.0), (13, 9, 3.0),
                     (11, 11, 4.0), (13, 11, 4.0), (15, 7, 4.0)):
    tag = "%da x %dk = %d, a_lat_max %.1f" % (na, nk, na * nk, alat)
    c_g = np.linspace(-1.0, 1.0, nk)
    assert np.any(c_g == 0.0)
    res[tag] = scored(accel_grid(na), c_g, alat, tag)

json.dump({"_family": "kappa scaled per v0: c_j * min(%.2f, a_lat_max/max(v0, "
                      "%.1f)^2), c in linspace(-1, 1, nk)" % (KAPPA_CAP, V_FLOOR),
           "surface": {"windows": N, "episodes": len(eps)},
           "ha": float(HA_PW.mean()), "bar": HA_BAR,
           "flat_kappa_shipped_for_scale": {
               "n_anchors": 117, "ade_0_2s_m": 0.2610,
               "kamm_over_mu_0.7_at_v0_27.27": 104, "peak_g": 3.96},
           "families": res},
          open(os.path.join(OUT, "KAMMFAM.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "KAMMFAM.json"))
