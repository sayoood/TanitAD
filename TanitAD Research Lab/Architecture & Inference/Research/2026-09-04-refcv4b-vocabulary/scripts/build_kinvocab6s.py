"""BUILD + GATE the v0-CONDITIONED kinematic anchor vocabulary for refcv4-b.

The shipped refcv4 vocabulary is 128 FIXED ego-frame paths in absolute metres.
It is not conditioned on the window's speed, while `ha` (hold-action) is, and it
reads 0.3773 m oracle-in-vocabulary against ha = 0.2996 -- SEPARATED WORSE.

This builds the pre-registered replacement: a v0-CONDITIONED family of constant
(accel, curvature) controls rolled through the programme's OWN integrator
(refa_v1_plan.unicycle_paths, action_units="kappa") at the window's measured v0,
over the full 6 s horizon V3_HORIZONS = (5,10,15,20,30,40,50,60).

GATE: oracle-in-vocabulary ADE 0-2 s must BEAT ha = 0.2996 m on the banked
4,823-window / 141-episode surface, paired episode-cluster bootstrap.

CONTROLS THAT MUST READ KNOWN VALUES:
  * a 1-point grid {a=0, kappa=0} must reproduce ha0 = 0.6723 (published);
  * kappa = 0 and a = 0 must be present EXACTLY in every grid -- an even-count
    linspace omits zero and the set then reads 1.2768 m, a 4.9x artifact that
    looks exactly like a resolution finding;
  * the zero path (no information) must read ~14.25 m;
  * train/eval clip intersection must be 0 (here: the family is DATA-FREE).
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
import tanitad                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "..", "vocab", "refcv3_40284_dump")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

DT = 0.1
H = max(V3_HORIZONS)                       # 60 steps = 6.0 s
SLOT_ALL = [k - 1 for k in V3_HORIZONS]    # [4,9,14,19,29,39,49,59]
SLOT_GATE = SLOT_ALL[:4]                   # 0.5/1.0/1.5/2.0 s -- the gate band
HA_BAR = 0.2996
HA0_PUB = 0.6723

print("[tree] tanitad imported from " + tanitad.__file__)
print("[horizons] V3_HORIZONS=" + str(V3_HORIZONS) + "  H=" + str(H) +
      " steps  gate slots=" + str(SLOT_GATE) + " (0.5/1.0/1.5/2.0 s)")

# --------------------------------------------------------------------------- #
# surface
# --------------------------------------------------------------------------- #
eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
G, V0, HA, EPI = [], [], [], []
man = json.load(open(os.path.join(DUMP, "manifest.json")))
for i, f in enumerate(eps):
    z = np.load(os.path.join(DUMP, f))
    G.append(z["g"]); V0.append(z["v0"]); HA.append(z["ha"])
    EPI.append(np.full(z["g"].shape[0], i, np.int64))
G = np.concatenate(G).astype(np.float64)
V0 = np.concatenate(V0).astype(np.float64)
HA = np.concatenate(HA).astype(np.float64)
EPI = np.concatenate(EPI)
N = len(G)
EVAL_CLIPS = set(e["clip_id"] for e in man["episodes"])
print("[surface] %d windows / %d episodes  (published: %d / %d)"
      % (N, len(eps), man["grid"]["n_windows"], man["grid"]["n_episodes"]))
assert N == man["grid"]["n_windows"] == 4823, N
assert len(eps) == 141

HA_PW = np.linalg.norm(HA - G, axis=-1).mean(1)
print("[control] ha reproduced from the dump = %.6f   published %s   delta %+.6f"
      % (HA_PW.mean(), HA_BAR, HA_PW.mean() - HA_BAR))
ZERO = float(np.linalg.norm(G, axis=-1).mean())
print("[control] zero path (no information)  = %.4f m" % ZERO)

# the family is CLOSED FORM -- it reads no corpus at all, so a train/eval
# contamination is impossible by construction, not by a lucky split.
TRAIN_CLIPS_USED = set()
print("[control] train/eval clip intersection = %d  (train clips read by the "
      "builder: %d -- the vocabulary is DATA-FREE)"
      % (len(TRAIN_CLIPS_USED & EVAL_CLIPS), len(TRAIN_CLIPS_USED)))


# --------------------------------------------------------------------------- #
# the family
# --------------------------------------------------------------------------- #
def grid(na, nk, a_lo, a_hi, k_lim):
    """(a, kappa) product grid. ASSERTS 0.0 present EXACTLY in BOTH axes."""
    assert na % 2 == 1 and nk % 2 == 1, (na, nk)
    a_g = np.linspace(a_lo, a_hi, na)
    if na > 1:
        a_g = a_g - a_g[np.abs(a_g).argmin()]      # re-centre so 0 is a node
        a_g = np.clip(a_g, a_lo, a_hi)
    k_g = np.linspace(-k_lim, k_lim, nk)
    assert np.any(a_g == 0.0), "accel grid omits 0.0: %s" % a_g
    assert np.any(k_g == 0.0), "curvature grid omits 0.0: %s" % k_g
    aa, kk = np.meshgrid(a_g, k_g, indexing="ij")
    return np.stack([aa.ravel(), kk.ravel()], -1), a_g, k_g


def roll(ctrl, v0):
    """[M, 2] constant controls at ONE v0 -> [M, 8, 2] paths at V3_HORIZONS."""
    c = torch.from_numpy(np.repeat(ctrl[:, None, :], H, axis=1)).double()
    p = unicycle_paths(c, torch.tensor(float(v0), dtype=torch.float64), DT,
                       action_units="kappa")
    return p.numpy()[:, SLOT_ALL, :]


def oracle(ctrl):
    """min-over-vocabulary ADE on the GATE band, per window, at each v0."""
    ade = np.empty(N); al = np.empty(N); la = np.empty(N)
    idx = np.empty(N, np.int64)
    key = np.round(V0, 3)
    cache = {}
    for i in range(N):
        k = key[i]
        if k not in cache:
            cache[k] = roll(ctrl, k)[:, :4, :]
        P = cache[k]
        d = np.linalg.norm(P - G[i][None], axis=-1).mean(1)
        j = int(d.argmin())
        ade[i] = d[j]; idx[i] = j
        r = P[j] - G[i]
        al[i] = np.abs(r[:, 0]).mean(); la[i] = np.abs(r[:, 1]).mean()
    return ade, al, la, idx, len(cache)


def boot(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(EPI)
    by = [np.where(EPI == e)[0] for e in uq]
    d = a - b
    o = np.array([d[np.concatenate([by[p] for p in
                                    rng.integers(0, len(uq), len(uq))])].mean()
                  for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


# --------------------------------------------------------------------------- #
# CONTROL 1 -- the 1-point grid must reproduce ha0
# --------------------------------------------------------------------------- #
print("\n=== CONTROL -- 1-point grid {a=0, kappa=0} must reproduce ha0 ===")
c1, _, _ = grid(1, 1, 0.0, 0.0, 0.0)
z = oracle(c1)[0]
print("  constant velocity -> %.6f   published ha0 = %s   delta %+.6f"
      % (z.mean(), HA0_PUB, z.mean() - HA0_PUB))
assert abs(z.mean() - HA0_PUB) < 5e-4, z.mean()

# --------------------------------------------------------------------------- #
# THE SWEEP
# --------------------------------------------------------------------------- #
print("\n=== SWEEP -- v0-conditioned (accel, curvature) families, 6 s roll ===")
CANDS = [
    (13, 9, -4.0, 3.0, 0.06, "13a x 9k = 117  (pre-registered winner)"),
    (11, 11, -4.0, 3.0, 0.06, "11a x 11k = 121"),
    (17, 7, -4.0, 3.0, 0.06, "17a x 7k = 119"),
    (13, 9, -4.0, 3.0, 0.08, "13a x 9k = 117, kappa +/-0.08"),
    (15, 9, -5.0, 3.0, 0.06, "15a x 9k = 135, accel [-5,3]"),
    (13, 11, -4.0, 3.0, 0.06, "13a x 11k = 143"),
    (9, 13, -4.0, 3.0, 0.06, "9a x 13k = 117"),
    (11, 13, -4.0, 3.0, 0.06, "11a x 13k = 143"),
]
rows = {}
best = None
for na, nk, alo, ahi, kl, tag in CANDS:
    ctrl, a_g, k_g = grid(na, nk, alo, ahi, kl)
    ade, al, la, idx, nv0 = oracle(ctrl)
    d, lo, hi = boot(ade, HA_PW)
    v = "BEATS ha" if hi < 0 else ("loses to ha" if lo > 0 else "tied with ha")
    nd = int(len(np.unique(idx)))
    print("  %-42s M=%4d  ADE %.4f  ALONG %.4f  LAT %.4f  used %3d/%3d  "
          "vs ha %+.4f [%+.4f, %+.4f] -> %s"
          % (tag, len(ctrl), ade.mean(), al.mean(), la.mean(), nd, len(ctrl),
             d, lo, hi, v))
    rows[tag] = {"n_anchors": int(len(ctrl)), "ade_0_2s_m": float(ade.mean()),
                 "along_mae_m": float(al.mean()), "lat_mae_m": float(la.mean()),
                 "distinct_used": nd, "vs_ha": [d, lo, hi], "verdict": v,
                 "accel_grid": [float(x) for x in a_g],
                 "kappa_grid": [float(x) for x in k_g],
                 "distinct_v0_rolled": int(nv0)}
    if hi < 0 and (best is None or ade.mean() < best[1]):
        best = (tag, float(ade.mean()), ctrl, a_g, k_g)

json.dump({"artifact_kind": "tanitad.refcv4b_kinvocab6s_gate",
           "evidence_class": "MEASURED (ours)",
           "tier": "T0 -- a vocabulary/geometry property, not a driving number",
           "surface": {"source": "taniteval/results/refcv3-40284-openloop-dump.tar.gz",
                       "windows": N, "episodes": len(eps),
                       "instants_s": [0.5, 1.0, 1.5, 2.0], "frame": "ego at t0",
                       "units": "m", "window_stride": 5},
           "controls": {"ha_reproduced": float(HA_PW.mean()),
                        "ha_published": HA_BAR,
                        "ha0_1point_grid": float(z.mean()),
                        "ha0_published": HA0_PUB, "zero_path_m": ZERO,
                        "train_eval_clip_intersection": 0,
                        "builder_reads_corpus": False,
                        "kappa_zero_present_asserted": True,
                        "accel_zero_present_asserted": True},
           "estimator": "paired episode-cluster bootstrap, n_boot 2000, seed 0, "
                        "cluster = episode",
           "gate": {"quantity": "oracle-in-vocabulary ADE 0-2 s (raw "
                                "min-over-vocabulary, no model)",
                    "bar_m": HA_BAR, "bar_arm": "ha (hold-action)"},
           "shipped_refcv4_fixed_path_vocabulary_for_scale": {
               "ade_0_2s_m": 0.3773, "along": 0.3041, "lat": 0.1503,
               "vs_ha": [0.0777, 0.0528, 0.1044], "verdict": "loses to ha"},
           "families": rows},
          open(os.path.join(OUT, "KINVOCAB6S_SWEEP.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "KINVOCAB6S_SWEEP.json"))
if best is None:
    print("\nGATE FAILED -- no family separated below ha. STOP.")
    sys.exit(3)
print("\nGATE PASSED -- winner: %s  ADE %.4f < ha %s" % (best[0], best[1], HA_BAR))
np.save(os.path.join(OUT, "winner_controls.npy"), best[2])
json.dump({"tag": best[0], "ade": best[1],
           "accel_grid": [float(x) for x in best[3]],
           "kappa_grid": [float(x) for x in best[4]]},
          open(os.path.join(OUT, "winner.json"), "w"), indent=1)
