#!/usr/bin/env python3
"""Shared loaders, feature assembly and estimators for the GOAL INPUT stream
(2026-07-27).

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

LEAKAGE AUDIT (pre-registered, PRE_REGISTRATION.md S1.1). These fields are
FUTURE-DERIVED and are inadmissible as head inputs:

    head_deg     |net heading change| over the FUTURE K_MAX steps
                 (stack/scripts/driving_diagnostic.py:139-142)
    a_gt         ground-truth acceleration
    v_target / vt_valid / vt_lookahead
                 lake.vtarget = 85th pct of FUTURE speed, 10-20 s ahead
    route / route_graded
                 route_from_future_v3, <=25 s FORWARD
    gt           the target itself

`head_deg` in particular reads like an ego-state scalar and is not one.
`ADMISSIBLE` below is the whitelist; `assert_no_oracle` enforces it.
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
ARCH = HUB / "Architecture & Inference" / "Implementation" / "incoming"
V5 = ARCH / "2026-07-26-v5-imagination-selection" / "raw"
EH2 = ARCH / "2026-07-27-t3-and-lambda-tau" / "raw" / "eh2_cache.pt"
REFC_SMALL = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
              / "2026-07-22-refc-small-30k")
RESULTS = REPO / "taniteval" / "results"
OUT = Path(__file__).resolve().parent.parent / "raw"

N_BOOT = 2000
SEED = 20260727
T_HORIZON = 2.0                 # s -- the 2 s planning horizon
WP_STEPS = (5, 10, 15, 20)      # 0.5 / 1.0 / 1.5 / 2.0 s
FRAC = np.array([0.25, 0.5, 0.75, 1.0])[None, :, None]   # [1,4,1]

#: committed bars, quoted from V5_PLAN.md / BAR_A.md (primary sources)
BAR_CONFIRM = 0.4907            # in-sample ceiling of ANY re-scoring of the fan
BAR_STRONG = 0.4271             # v1's deployed ade_0_2s (full_set)

#: whitelist of admissible (observation-only) fan-dump fields
ADMISSIBLE = frozenset({"fan", "logits", "sel", "cv", "v0", "speed", "eid",
                        "wp_steps", "n_anchors"})
ORACLE = frozenset({"gt", "a_gt", "head_deg", "v_target", "vt_valid",
                    "vt_lookahead"})


def assert_no_oracle(names) -> None:
    """Abort if a head input names a future-derived field. Called by every
    feature builder; the whole tautology test is void if this ever passes."""
    bad = sorted(set(names) & ORACLE)
    if bad:
        raise RuntimeError(f"ORACLE FIELD IN HEAD INPUT: {bad}")


# --------------------------------------------------------------------------- #
# loaders                                                                      #
# --------------------------------------------------------------------------- #
def load_refc_fan(arm: str = "xl") -> dict:
    """A REF-C anchored-diffusion fan WITH per-candidate trajectories.

    `arm` in {'xl' (256 anchors), 'base' (128), 'small' (64)}. These are the only
    repo artifacts carrying per-candidate trajectories on the canonical
    881-window / 40-episode val deployment.
    """
    p = (REFC_SMALL / "fan_refc-small-30k.pt" if arm == "small"
         else RESULTS / f"fan_refc-{arm}-30k.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def load_eh2() -> dict:
    """v4's cached latent-derived selector ingredients on the SAME 881 windows.

    `lat`(8) / `lon`(7) / `dist`(8) are `head.{lat,lon,dist}_head(state_last)`
    where `state_last = world.encode_window(frames)` -- FRAMES ONLY, no future,
    no label, no batch goal field (bar_a_selector.py build_cache + goal_modes.py).
    `refined_pre`(256) / `prior`(256) are per-candidate confidences.
    """
    return torch.load(EH2, map_location="cpu", weights_only=False)


def eid_str(d: dict) -> np.ndarray:
    return np.asarray([str(x) for x in d["eid"]])


# --------------------------------------------------------------------------- #
# metric primitives -- ade_0_2s is the mean over the 4 waypoints of the L2      #
# displacement error, exactly as every committed bar in this program defines it #
# --------------------------------------------------------------------------- #
def ade(traj: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """`traj` [...,4,2] vs `gt` [...,4,2] -> [...] mean-over-waypoint L2."""
    return np.linalg.norm(traj - gt, axis=-1).mean(-1)


def endpoint_err(p: np.ndarray, gt_end: np.ndarray) -> np.ndarray:
    """2 s ENDPOINT L2, metres. `p` [W,2], `gt_end` [W,2] -> [W].

    This is the axis on which the tautology test compares a goal head against
    the as-trained selector."""
    return np.linalg.norm(p - gt_end, axis=-1)


def goal_reference(goal: np.ndarray) -> np.ndarray:
    """[W,2] goal endpoint -> [W,4,2] constant-velocity path to it.

    Bit-identical to `fanc_goal.py`'s `R_goal2s` construction when
    `goal == gt[:, -1]`; Fidelity-A checks exactly that."""
    return goal[:, None, :] * FRAC


def pick_nearest_to(ref: np.ndarray, fan: np.ndarray) -> np.ndarray:
    """Nearest fan candidate to a reference trajectory, under the ADE metric
    ITSELF. `ref` [W,4,2], `fan` [W,N,4,2] -> [W] index.

    (!) The proximity metric MUST be the same mean-over-waypoint L2 that scores
    the pick. Matching in flat 8-dim L2 (= sqrt of the SUM of squared waypoint
    errors) while scoring with the mean of waypoint norms are DIFFERENT metrics
    with different argmins; that bug produced a stable, plausible, wrong 0.3655 m
    in the parent stream and was caught only by the positive control. Fidelity-C
    re-runs that control here.
    """
    d = np.linalg.norm(fan - ref[:, None], axis=-1).mean(-1)     # [W,N]
    return d.argmin(1)


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
    return (ci["lo"] > 0) or (ci["hi"] < 0)
