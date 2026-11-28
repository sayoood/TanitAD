"""ISOLATE THE CAUSE: is the 84x the ESTIMATOR, or the MISSING VALIDITY MASK?

2x2 on the SAME dump, SAME grid ([5,10,15,20] -> 0.5 s uniform), SAME windows:

              | no mask                    | masked (ds > 0.5 m/s * 0.5 s = 0.25 m)
  p14 kappa   | THE PUBLISHED 84x          | ?
  seq_geom    | ?                          | THE CANONICAL METRIC

Plus a paired episode-cluster bootstrap on the contrast that matters, and the
DELIBERATE-REGRESSION control: re-inject the stopped windows and prove the
84x comes back (a guard that cannot fail is not a guard).

ZERO GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import four_families as ff

DT = 0.1
d = torch.load(os.path.join(HERE, "fan_refc-base-30k.pt"),
               map_location="cpu", weights_only=False)
fan, gt, cv = d["fan"].float().numpy(), d["gt"].float().numpy(), d["cv"].float().numpy()
logits = d["logits"].float().numpy()
eid = np.asarray(d["eid"])
ts0 = np.concatenate([[0.0], np.array([w * DT for w in d["wp_steps"]], dtype=np.float64)])
n = gt.shape[0]
GRID_DT = 0.5                    # the UNIFORM spacing of [5,10,15,20] x 0.1 s
MIN_DS = ff.MIN_DS_MPS * GRID_DT   # 0.25 m -- the canonical gate ON THIS GRID


def pwo(p):
    z = np.zeros(p.shape[:-2] + (1, 2), dtype=p.dtype)
    return np.concatenate([z, p], axis=-2)


def kappa_p14(p, ts):
    d1 = np.gradient(p, ts, axis=-2); d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)


def kappa_seq(p):
    g = ff._seq_geometry(torch.as_tensor(p).float(), GRID_DT)
    return g["curvature"].numpy(), g["pair_valid"].numpy()


def step_valid(p):
    """ds > MIN_DS on every step of the ORIGIN-PREPENDED path -> per-window bool."""
    ds = np.linalg.norm(np.diff(pwo(p), axis=-2), axis=-1)      # [n,4]
    return ds > MIN_DS


straight = np.zeros_like(gt); straight[..., 0] = np.linalg.norm(gt, axis=-1)
arms = {"rank_shipped": fan[np.arange(n), logits.argmax(1)],
        "rank_oracle": fan[np.arange(n),
                           np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1).argmin(1)],
        "cv_floor": cv, "straight_floor": straight}

out = {"grid": {"wp_steps": list(d["wp_steps"]), "ts0_s": list(ts0),
                "uniform": True, "grid_dt_s": GRID_DT, "min_ds_gate_m": MIN_DS},
       "n_windows": int(n), "n_episodes": int(len(set(eid.tolist())))}

# --------------------------------------------------------------- the 2x2 --
kg_p14 = kappa_p14(pwo(gt), ts0)
kg_seq, gv_seq = kappa_seq(gt)
gt_ok = step_valid(gt)

cell = {}
for name, traj in arms.items():
    kt_p14 = kappa_p14(pwo(traj), ts0)
    kt_seq, pv_seq = kappa_seq(traj)
    tr_ok = step_valid(traj)
    # per-slot mask: both paths must have a real step on BOTH sides of the pair
    ok = tr_ok & gt_ok                                   # [n,4]
    ok_pair_p14 = ok                                     # p14 kappa is per-POINT (5 pts -> 5)
    # p14's curvature array is [n,5]; align the mask by padding the origin as valid
    m_p14 = np.concatenate([ok[:, :1], ok], axis=1)      # [n,5]
    dk_p14 = np.abs(kt_p14 - kg_p14)
    dk_seq = np.abs(kt_seq - kg_seq)
    m_seq = pv_seq & gv_seq
    cell[name] = {
        "p14_nomask_MEAN": float(dk_p14.mean(-1).mean()),          # PUBLISHED
        "p14_masked_MEAN": float(dk_p14[m_p14].mean()),
        "seq_nomask_MEAN": float(dk_seq.mean()),
        "seq_masked_MEAN": float(dk_seq[m_seq].mean()),
        "p14_frac_masked": float(1 - m_p14.mean()),
        "seq_frac_masked": float(1 - m_seq.mean()),
    }
out["cells"] = cell
sf = cell["straight_floor"]
sh = cell["rank_shipped"]
out["ratios_shipped_over_straight"] = {
    k.replace("_MEAN", ""): sh[k] / sf[k] for k in
    ("p14_nomask_MEAN", "p14_masked_MEAN", "seq_nomask_MEAN", "seq_masked_MEAN")}

# ------------------------------------ paired episode-cluster bootstrap ----
def boot(a, b, eids, B=10000, seed=0):
    """paired: per-window (a-b), resampled by EPISODE cluster."""
    rng = np.random.default_rng(seed)
    ue = np.unique(eids)
    idx = {e: np.where(eids == e)[0] for e in ue}
    dif = a - b
    obs = float(np.nanmean(dif))
    stats = np.empty(B)
    for i in range(B):
        pick = rng.choice(ue, len(ue), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        stats[i] = np.nanmean(dif[sel])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"delta": obs, "lo": float(lo), "hi": float(hi),
            "separated": bool(lo > 0 or hi < 0)}


def per_window(traj, masked):
    kt = kappa_seq(traj)[0]; pv = kappa_seq(traj)[1] & gv_seq
    dk = np.abs(kt - kg_seq)
    if not masked:
        return dk.mean(-1)
    o = np.where(pv, dk, np.nan)
    return np.nanmean(o, axis=-1)


def per_window_p14(traj, masked):
    kt = kappa_p14(pwo(traj), ts0); dk = np.abs(kt - kg_p14)
    if not masked:
        return dk.mean(-1)
    ok = step_valid(traj) & gt_ok
    m = np.concatenate([ok[:, :1], ok], axis=1)
    return np.nanmean(np.where(m, dk, np.nan), axis=-1)


out["contrast_shipped_minus_straight"] = {
    "p14_nomask_PUBLISHED": boot(per_window_p14(arms["rank_shipped"], False),
                                 per_window_p14(arms["straight_floor"], False), eid),
    "p14_masked": boot(per_window_p14(arms["rank_shipped"], True),
                       per_window_p14(arms["straight_floor"], True), eid),
    "seq_masked_CANONICAL": boot(per_window(arms["rank_shipped"], True),
                                 per_window(arms["straight_floor"], True), eid),
}

# ---------------- DELIBERATE REGRESSION: the guard must be able to fail ---
# Drop the stopped windows entirely and the 84x must COLLAPSE; put them back
# and it must RETURN. If both read the same, the mechanism is not what I say.
v0 = d["v0"].numpy()
moving = v0 > 0.5
regress = {}
for label, keep in (("ALL windows (as published)", np.ones(n, bool)),
                    ("MOVING only (v0 > 0.5 m/s)", moving),
                    ("STOPPED only (v0 <= 0.5 m/s)", ~moving)):
    a = np.abs(kappa_p14(pwo(arms["rank_shipped"]), ts0) - kg_p14).mean(-1)[keep]
    b = np.abs(kappa_p14(pwo(arms["straight_floor"]), ts0) - kg_p14).mean(-1)[keep]
    regress[label] = {"n": int(keep.sum()), "shipped": float(a.mean()),
                      "straight_floor": float(b.mean()),
                      "ratio": float(a.mean() / b.mean())}
out["deliberate_regression_v0_split"] = regress
out["n_stopped_windows"] = int((~moving).sum())
out["frac_stopped"] = float((~moving).mean())

print(json.dumps(out, indent=1))
