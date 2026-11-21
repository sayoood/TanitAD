"""Q1 follow-up — coverage under the LIVE O4-WEIGHTED sampler, not the control.

The Q1 package simulated `make_sampler`, which `train_v6_staged.py:5300` itself
calls the `--o4-alpha 0` CONTROL-ARM sampler. Those figures are therefore the
UNIFORM UPPER BOUND. The live arm uses `InteractionSampler` with `--o4-alpha 1.0`
and `--o4-floor 0.25` (argparse defaults, `:7120-7122`), which draws windows
INSIDE an episode by `saliency_weights(s) = (floor + s)**alpha`.

⭐ TWO STRUCTURAL FACTS READ FROM SOURCE BEFORE SIMULATING, because they bound
what weighting can possibly do:
  * `InteractionSampler` draws EPISODES UNIFORMLY (v6.py:786 states it verbatim:
    "Episodes are drawn uniformly so no episode is starved"). ⇒ EPISODE coverage
    is structurally unaffected by O4 and stays 100 %.
  * `floor = 0.25 > 0` keeps every window REACHABLE (v6.py:759 — "NOT cosmetic").
    ⇒ weighting cannot zero any window; it can only slow the tail.

So only WINDOW and FRAME coverage can move, and the question is by how much.

⛔ THE SALIENCY IS MEASURED, NOT MODELLED. A plausible-looking synthetic skew
would decide a GPU-day on an invented distribution. Saliency is computed with the
REAL `kinematic_saliency` from REAL actions derived from the local train cache's
poses, and the resulting weight spread is printed so the draw's skew is stated
rather than assumed.

Evidence class: MEASURED (ours).
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
from tanitad.models.v6 import kinematic_saliency, saliency_weights  # noqa: E402

CACHE = ("C:/Users/Admin/tanitad-data/physicalai/_epcache/"
         "physicalai-train-14231cd29c74")
WINDOW, MAX_H, DT = 6, 20, 0.1
ALPHA, FLOOR = 1.0, 0.25          # argparse defaults :7120-7122
EPS_PER_BATCH, STEPS = 4, 30000
N_PROBE_EPS = 120                 # episodes simulated (each gets its own weights)


def episode_actions(poses: np.ndarray) -> np.ndarray:
    """(steer, accel) per frame from poses [T, >=4] = x, y, yaw, v."""
    yaw, v = poses[:, 2], poses[:, 3]
    steer = np.gradient(np.unwrap(yaw), DT)          # yaw rate as the steer proxy
    accel = np.gradient(v, DT)
    return np.stack([steer, accel], -1)


files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))[:N_PROBE_EPS]
if not files:
    files = sorted(glob.glob(os.path.join(CACHE, "**", "*.pt"), recursive=True))[:N_PROBE_EPS]
print(f"episodes sampled for real saliency: {len(files)}")

all_w, all_n = [], []
for f in files:
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        poses = d["poses"].numpy()
    except Exception:
        continue
    T = poses.shape[0]
    n_win = T - WINDOW - MAX_H
    if n_win <= 0:
        continue
    acts = episode_actions(poses)
    span = WINDOW + MAX_H
    win = np.stack([acts[t:t + span] for t in range(n_win)])      # [N, span, 2]
    s = kinematic_saliency(torch.tensor(win, dtype=torch.float32), dt=DT)
    w = saliency_weights(s, alpha=ALPHA, floor=FLOOR, normalize=True)
    all_w.append(w.numpy())
    all_n.append(n_win)

# ⚠️ episode lengths vary (MEASURED 197-199), so n_win varies. Truncate to the
# common length rather than padding: a padded window would be a window that does
# not exist, and it would dilute the coverage fractions with fictional slots.
n_common = min(all_n)
W = np.stack([w[:n_common] for w in all_w])                      # [E, n_win]
N_FRAMES = n_common + WINDOW + MAX_H
print(f"windows/episode {all_n[0]} · weight spread max/min "
      f"{float(W.max(1).mean() / W.min(1).mean()):.2f}x (mean over episodes)")
q = np.percentile(W[0], [0, 25, 50, 75, 100])
print(f"per-window weight quantiles (ep0): {np.round(q / q.mean(), 3)}")

# ⛔ MATCH THE EPISODE COUNT TO THE CORPUS, or the MIN is not comparable.
# The uniform Q1 run took min over ALL 2,376 / 4,713 episodes; simulating only
# the 120 episodes whose weights I measured would take min over 120 — a far less
# extreme order statistic — and the weighted arm would look like it IMPROVED the
# tail purely because fewer draws were taken from the tail distribution. The 120
# MEASURED weight profiles are TILED to corpus size so both runs take the minimum
# over the same number of episodes.
rng = np.random.default_rng(0)
for batch in (8, 16):
    n_win = W.shape[1]
    # episodes uniform (v6.py:786), windows by weight within the episode
    for corpus, n_real in (("parity 2,376", 2376), ("B1 4,713", 4713)):
        n_eps_sim = n_real
        prof = rng.integers(0, W.shape[0], size=n_eps_sim)   # tile measured profiles
        seen = np.zeros((n_eps_sim, n_win), dtype=bool)
        total_draws = STEPS * batch
        per_ep = total_draws / n_real            # expected draws per episode
        for e in range(n_eps_sim):
            k = rng.poisson(per_ep)              # draws this episode receives
            if k:
                pe = W[prof[e]]
                idx = rng.choice(n_win, size=k, replace=True, p=pe / pe.sum())
                seen[e, idx] = True
        span = WINDOW + MAX_H
        fcov = np.empty(n_eps_sim)
        for e in range(n_eps_sim):
            hit = np.zeros(N_FRAMES, dtype=bool)
            for t in np.flatnonzero(seen[e]):
                hit[t:min(t + span, N_FRAMES)] = True
            fcov[e] = hit.mean()
        print(f"  {corpus:<12} batch {batch:<3} expected draws/ep {per_ep:6.1f} "
              f"| WINDOW {seen.mean()*100:6.2f}%  FRAME mean {fcov.mean()*100:6.2f}% "
              f"p1 {np.percentile(fcov,1)*100:6.2f}% MIN {fcov.min()*100:6.2f}%")
print("\n⚠️ EPISODE coverage is unchanged at 100% by construction — "
      "InteractionSampler draws episodes UNIFORMLY (v6.py:786).")
