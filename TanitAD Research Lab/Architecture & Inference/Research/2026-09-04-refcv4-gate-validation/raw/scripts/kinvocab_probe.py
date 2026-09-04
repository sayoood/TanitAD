"""PRE-REGISTRATION PROBE for the NEXT refcv4 arm: is a v0-CONDITIONED
(accel, curvature) vocabulary of the same size enough to clear `ha` = 0.2996 m?

The shipped vocabulary is 128 fixed ego-frame PATHS in absolute metres — it is
NOT conditioned on the window's speed, while `ha` is. This probe prices the
deferred lever ("(accel, curvature) through rollout_unicycle") on the SAME
4,823-window surface, with zero GPU.

⛔ Controls that must read known values:
  * a 1-point grid {a=0, kappa=0} must reproduce `ha0` = 0.6723 EXACTLY;
  * the grid oracle must come in at or below `ha` = 0.2996, because (a0, k0) —
    the control `ha` itself flies — lies inside the grid's range.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
from tanitad.refs.refa_v1_plan import unicycle_paths        # noqa: E402

DUMP = os.path.join(HERE, "refcv3_40284_dump")
eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
G, V0, EPI, HA = [], [], [], []
for i, f in enumerate(eps):
    z = np.load(os.path.join(DUMP, f))
    G.append(z["g"]); V0.append(z["v0"]); HA.append(z["ha"])
    EPI.append(np.full(z["g"].shape[0], i, np.int64))
G = np.concatenate(G).astype(np.float64)
V0 = np.concatenate(V0).astype(np.float64)
HA = np.concatenate(HA).astype(np.float64)
EPI = np.concatenate(EPI)
N = len(G)
SLOT = [4, 9, 14, 19]                       # 0.5/1.0/1.5/2.0 s at dt = 0.1
H, DT = 20, 0.1
print(f"[surface] {N} windows / {len(eps)} episodes")


def roll(a_grid, k_grid, v0_batch):
    """[n_a*n_k, 4, 2] paths for ONE v0."""
    aa, kk = np.meshgrid(a_grid, k_grid, indexing="ij")
    ctrl = np.stack([aa.ravel(), kk.ravel()], -1)                # [M, 2]
    c = torch.from_numpy(np.repeat(ctrl[:, None, :], H, axis=1)).double()
    p = unicycle_paths(c, torch.tensor(float(v0_batch), dtype=torch.float64),
                       DT, action_units="kappa")
    return p.numpy()[:, SLOT, :]


def oracle(a_grid, k_grid, tag):
    ade = np.empty(N); al = np.empty(N); la = np.empty(N)
    # v0 is per-window, so the fan must be rolled per DISTINCT v0. Bucket to
    # 1 mm/s so the roll count is bounded without changing any answer materially.
    key = np.round(V0, 3)
    cache: dict = {}
    for i in range(N):
        k = key[i]
        if k not in cache:
            cache[k] = roll(a_grid, k_grid, k)
        P = cache[k]
        d = np.linalg.norm(P - G[i][None], axis=-1).mean(1)
        j = int(d.argmin())
        ade[i] = d[j]
        r = P[j] - G[i]
        al[i] = np.abs(r[:, 0]).mean(); la[i] = np.abs(r[:, 1]).mean()
    print(f"  {tag:<34s} M={len(a_grid)*len(k_grid):4d}  ADE {ade.mean():.4f}  "
          f"ALONG {al.mean():.4f}  LAT {la.mean():.4f}   "
          f"(distinct v0 rolled: {len(cache)})")
    return ade, al.mean(), la.mean()


def boot(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(EPI); by = [np.where(EPI == e)[0] for e in uq]
    d = a - b
    o = np.array([d[np.concatenate([by[p] for p in
                                    rng.integers(0, len(uq), len(uq))])].mean()
                  for _ in range(n_boot)])
    return float(d.mean()), *(float(x) for x in np.percentile(o, [2.5, 97.5]))


print("\n=== CONTROL — a 1-point grid {a=0, kappa=0} must reproduce ha0 ===")
z, _, _ = oracle(np.array([0.0]), np.array([0.0]), "constant velocity")
print(f"    -> {z.mean():.6f}   published ha0 = 0.6723   "
      f"delta {z.mean() - 0.6723:+.6f}")

print("\n=== THE PROBE — v0-conditioned (accel, curvature) vocabularies ===")
ha_pw = np.linalg.norm(HA - G, axis=-1).mean(1)
res = {}
# ⛔ EVERY grid contains kappa = 0 and accel = 0 EXACTLY (odd counts). A grid
# built with an even count omits the straight-ahead control, and since most
# driving is straight that alone costs ~1 m of lateral error -- an artefact of
# np.linspace parity, not of resolution. Measured: 16x8 (no zero) read 1.2768 m
# against 12x11 (with zero) at 0.2608 m.
for na, nk, tag in ((17, 7, "17 accel x 7 curvature = 119"),
                    (9, 15, "9 accel x 15 curvature = 135"),
                    (11, 11, "11 accel x 11 curvature = 121"),
                    (13, 9, "13 accel x 9 curvature = 117"),
                    (7, 19, "7 accel x 19 curvature = 133")):
    a_g = np.linspace(-4.0, 3.0, na)
    k_g = np.linspace(-0.06, 0.06, nk)
    ade, al, la = oracle(a_g, k_g, tag)
    d, lo, hi = boot(ade, ha_pw)
    v = "BEATS ha" if hi < 0 else ("loses to ha" if lo > 0 else "tied with ha")
    print(f"      vs ha: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {v}")
    res[tag] = {"ade": float(ade.mean()), "along": float(al), "lat": float(la),
                "vs_ha": [d, lo, hi], "verdict": v}

json.dump({"surface": {"windows": N, "episodes": len(eps)},
           "control_constant_velocity_ade": float(z.mean()),
           "control_expected_ha0": 0.6723,
           "ha": float(ha_pw.mean()),
           "shipped_path_vocabulary_ade": 0.3773,
           "grids": res,
           "_reads": "a v0-CONDITIONED kinematic vocabulary of the SAME budget, "
                     "rolled through the programme's own unicycle "
                     "(refa_v1_plan.unicycle_paths, action_units='kappa'), on the "
                     "same 4,823 windows"},
          open(os.path.join(HERE, "KINVOCAB_PROBE.json"), "w"), indent=1)
print("\nwrote KINVOCAB_PROBE.json")
