#!/usr/bin/env python3
"""Shared loaders, anchor-set machinery and estimators for the FAN CONDITIONING
stream (2026-07-27).

Runs on the DEV BOX, CPU only. No pod is contacted, no checkpoint is loaded, no
episode is ever opened -- every input is an artifact already committed to this
repo. The canonical parity corpus (`physicalai-train-e438721ae894`, skip-hash
`f09e44db`) is therefore never touched, and the dev box's own NON-parity episode
cache (`14231cd29c74`) is never read. Nothing here adds GPU or RAM load to any
pod.

Estimator: `taniteval.ci.episode_cluster_bootstrap` /
`paired_episode_cluster_bootstrap`, B=2000, resampling unit = EPISODE.
`overlapping_holdout_se` is NEVER called (it biases the point estimate as well
as the interval -- CLAUDE.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval.ci import (  # noqa: E402
    episode_cluster_bootstrap,
    paired_episode_cluster_bootstrap,
)

HUB = REPO / "TanitAD Research Hub"
V5 = (HUB / "Architecture & Inference" / "Implementation" / "incoming"
      / "2026-07-26-v5-imagination-selection" / "raw")
REFC_SMALL = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
              / "2026-07-22-refc-small-30k")
RESULTS = REPO / "taniteval" / "results"
OUT = Path(__file__).resolve().parent.parent / "raw"

N_BOOT = 2000
SEED = 20260727
T_HORIZON = 2.0                 # s -- the 2 s planning horizon
WP_STEPS = (5, 10, 15, 20)      # 0.5 / 1.0 / 1.5 / 2.0 s


# --------------------------------------------------------------------------- #
# loaders                                                                      #
# --------------------------------------------------------------------------- #
def load_refc_fan(arm: str = "xl") -> dict:
    """A REF-C anchored-diffusion fan WITH the full per-candidate trajectories.

    `arm` in {'xl' (256 anchors), 'base' (128), 'small' (64)}. These are the only
    artifacts in the repo carrying per-candidate trajectories on the canonical
    881-window / 40-episode val deployment (FAN_CLIP_LOCAL.md S2), which is why
    a counterfactual anchor set can be evaluated against them at all.
    """
    p = (REFC_SMALL / "fan_refc-small-30k.pt" if arm == "small"
         else RESULTS / f"fan_refc-{arm}-30k.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def load_v5(tag: str = "v1") -> dict:
    """The staged v5 REDUCED per-window dump (`tag` in {'v4','v1'} = SCORER WM)."""
    return torch.load(V5 / f"v5_{tag}_windows_reduced.pt", map_location="cpu",
                      weights_only=False)


def eid_str(d: dict) -> np.ndarray:
    return np.asarray([str(x) for x in d["eid"]])


# --------------------------------------------------------------------------- #
# metric primitives -- ade_0_2s is the mean over the 4 waypoints of the L2      #
# displacement error, exactly as every committed bar in this program defines it #
# --------------------------------------------------------------------------- #
def ade(traj: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """`traj` [..., 4, 2] vs `gt` [..., 4, 2] -> [...] mean-over-waypoint L2."""
    return np.linalg.norm(traj - gt, axis=-1).mean(-1)


def ade_set(anchors: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """`anchors` [N,4,2], `gt` [W,4,2] -> [W,N] every anchor against every GT."""
    return np.linalg.norm(anchors[None] - gt[:, None], axis=-1).mean(-1)


def flat(t: np.ndarray) -> np.ndarray:
    """[...,4,2] -> [...,8] the space anchors are fitted / matched in."""
    return t.reshape(*t.shape[:-2], 8)


def mean_speed(traj: np.ndarray) -> np.ndarray:
    """Implied mean along-track speed over the 2 s horizon, m/s."""
    return traj[..., -1, 0] / T_HORIZON


# --------------------------------------------------------------------------- #
# episode-disjoint folds                                                       #
# --------------------------------------------------------------------------- #
def episode_folds(eid: np.ndarray, k: int = 5, seed: int = SEED):
    """k episode-DISJOINT folds. Yields (train_mask, test_mask) over windows."""
    uniq = np.unique(eid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    for f in range(k):
        te_eps = set(uniq[perm[f::k]].tolist())
        te = np.array([e in te_eps for e in eid])
        yield ~te, te


# --------------------------------------------------------------------------- #
# anchor sets                                                                  #
# --------------------------------------------------------------------------- #
def fit_anchors(traj: np.ndarray, n: int, seed: int = SEED) -> tuple:
    """CoverNet-style fixed trajectory set: k-means over training trajectories.

    Returns (anchors [n,4,2], weights [n] = cluster share). If fewer unique
    training trajectories than `n`, every training trajectory becomes an anchor
    and the set is smaller -- reported, never silently padded.
    """
    from sklearn.cluster import KMeans

    x = flat(traj)
    n_eff = int(min(n, len(x)))
    if n_eff < n:
        w = np.ones(n_eff) / n_eff
        return x[:n_eff].reshape(n_eff, 4, 2), w
    km = KMeans(n_clusters=n_eff, n_init=4, random_state=seed).fit(x)
    lab = km.labels_
    w = np.bincount(lab, minlength=n_eff) / len(lab)
    return km.cluster_centers_.reshape(n_eff, 4, 2), w


def bucket_edges(v0_train: np.ndarray, n_buckets: int) -> np.ndarray:
    """Quantile bucket edges fitted on the TRAIN fold only."""
    q = np.linspace(0, 1, n_buckets + 1)[1:-1]
    return np.quantile(v0_train, q)


def assign_bucket(v0: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, v0, side="right")


# --------------------------------------------------------------------------- #
# selection rules that TRANSFER to an arbitrary anchor set                     #
# --------------------------------------------------------------------------- #
def pick_nearest_to(ref: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Nearest anchor to a reference trajectory, under the ADE metric ITSELF.

    `ref` [W,4,2], `anchors` [N,4,2] -> [W] index. This is the transferable
    family: C1 (reference = constant velocity), C2 (reference = one world-model
    roll), and re-quantisation of any deployed planner's own answer are all
    instances. It needs NO training, so it is the only realised-pick family that
    can be evaluated on a counterfactual anchor set without a GPU-week.

    ⚠️ The proximity metric MUST be the same mean-over-waypoint L2 that scores
    the pick. An earlier version matched in flat 8-dim L2 (= sqrt of the SUM of
    squared waypoint errors) while scoring with the mean of waypoint norms;
    those are different metrics with different argmins, and the positive control
    `pick_nearest_to(GT) == oracle_in_fan` FAILED at 0.3655 m. It now holds to
    floating point, which is exactly why that control is run.
    """
    d = np.linalg.norm(anchors[None] - ref[:, None], axis=-1).mean(-1)
    return d.argmin(1)


# --------------------------------------------------------------------------- #
# intervals                                                                    #
# --------------------------------------------------------------------------- #
def ci_single(v, eid, seed: int = SEED) -> dict:
    return episode_cluster_bootstrap(np.asarray(v, dtype=float), list(eid),
                                     n_boot=N_BOOT, seed=seed)


def ci_paired(a, b, eid, seed: int = SEED) -> dict:
    return paired_episode_cluster_bootstrap(np.asarray(a, dtype=float),
                                            np.asarray(b, dtype=float),
                                            list(eid), n_boot=N_BOOT, seed=seed)


def r4(x) -> float:
    return round(float(x), 4)


def sep(ci: dict) -> bool:
    """CI-separated from zero (the program's separation predicate)."""
    lo, hi = ci["lo"], ci["hi"]
    return (lo > 0) or (hi < 0)
