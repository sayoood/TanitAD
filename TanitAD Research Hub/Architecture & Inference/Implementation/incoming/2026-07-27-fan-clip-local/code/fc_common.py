#!/usr/bin/env python3
"""Shared loaders + estimators for the LOCAL fan-clip stream (2026-07-27).

Runs on the DEV BOX. No pod is contacted, no checkpoint is loaded, no episode is
re-selected -- every input is an artifact already committed to this repo, so the
canonical parity corpus (`physicalai-train-e438721ae894`, skip-hash `f09e44db`)
is never touched and the dev box's own non-parity episode cache (`14231cd29c74`)
is never read.

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` /
`episode_cluster_bootstrap`, B=2000, resampling unit = EPISODE CLUSTER.
`overlapping_holdout_se` is never called.
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
BARA = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
        / "2026-07-26-bar-a-selector" / "raw")
REFC_SMALL = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
              / "2026-07-22-refc-small-30k")
RESULTS = REPO / "taniteval" / "results"

N_BOOT = 2000
SEED = 20260727
T_HORIZON = 2.0          # s -- the 2 s planning horizon, all fans
WP_STEPS = (5, 10, 15, 20)


# --------------------------------------------------------------------------- #
# loaders                                                                      #
# --------------------------------------------------------------------------- #
def load_v5(tag: str) -> dict:
    """The staged v5 REDUCED per-window dump (`tag` in {'v4','v1'} = the SCORER)."""
    return torch.load(V5 / f"v5_{tag}_windows_reduced.pt", map_location="cpu",
                      weights_only=False)


def load_gt_dense() -> torch.Tensor:
    """[881, 20, 2] dense GT ego waypoints -- from Bar A's staged window dump.

    The v5 REDUCED dump dropped `tgt`; Bar A's kept it, on the identical windows
    (proved in fc_gate.py S0.3).
    """
    d = torch.load(BARA / "bar_a_produced_windows.pt", map_location="cpu",
                   weights_only=False)
    return d["tgt"].float()


def load_refc_fan(arm: str) -> dict:
    """A REF-C anchored-diffusion fan with the FULL candidate trajectories.

    `arm` in {'xl' (256 anchors), 'base' (128), 'small' (64)}. These are the only
    artifacts in the repo that carry per-candidate trajectories on the canonical
    881-window / 40-episode val deployment, which is why the EXACT longitudinal
    band can be evaluated on them and not on v4's fan (see FAN_CLIP_LOCAL.md S2).
    """
    p = (REFC_SMALL / "fan_refc-small-30k.pt" if arm == "small"
         else RESULTS / f"fan_refc-{arm}-30k.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def eid_of(ep) -> list:
    """taniteval.ci wants a per-window episode label list."""
    a = ep.numpy() if isinstance(ep, torch.Tensor) else np.asarray(ep)
    return [str(int(x)) for x in a]


# --------------------------------------------------------------------------- #
# metric + selection primitives                                                #
# --------------------------------------------------------------------------- #
def ade_of_pick(fan_err: torch.Tensor, pick: torch.Tensor) -> np.ndarray:
    """Per-window ade_0_2s of a selection rule. `fan_err` [W, N], `pick` [W]."""
    return fan_err.gather(1, pick.long()[:, None]).squeeze(1).numpy()


def argmin_masked(cost: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """argmin of `cost` [W, N] restricted to `keep` [W, N] (bool).

    Windows with an empty survivor set fall back to the GLOBAL argmin of `cost`
    over all N candidates -- a deterministic, ground-truth-free fallback. The
    number of such windows is reported at every band so the fallback can never
    silently carry a result.
    """
    big = torch.finfo(cost.dtype).max / 4
    masked = torch.where(keep, cost, torch.full_like(cost, big))
    empty = ~keep.any(dim=1)
    out = masked.argmin(dim=1)
    if empty.any():
        out[empty] = cost[empty].argmin(dim=1)
    return out


def min_masked(v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    big = torch.finfo(v.dtype).max / 4
    masked = torch.where(keep, v, torch.full_like(v, big))
    empty = ~keep.any(dim=1)
    out = masked.min(dim=1).values
    if empty.any():
        out[empty] = v[empty].min(dim=1).values
    return out


def random_masked(keep: torch.Tensor, seed: int) -> torch.Tensor:
    """Uniform pick among survivors -- the clip's OWN control.

    Says how much of any gain is the CLIP (fewer, better candidates) rather than
    the ROUTE. A route that only matches this has contributed nothing.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(keep.shape, generator=g)
    u = torch.where(keep, u, torch.full_like(u, -1.0))
    empty = ~keep.any(dim=1)
    out = u.argmax(dim=1)
    if empty.any():
        out[empty] = torch.randint(0, keep.shape[1], (int(empty.sum()),),
                                   generator=g)
    return out


# --------------------------------------------------------------------------- #
# intervals                                                                    #
# --------------------------------------------------------------------------- #
def ci_single(v, eid, seed=SEED) -> dict:
    return episode_cluster_bootstrap(v, eid, n_boot=N_BOOT, seed=seed)


def ci_paired(a, b, eid, seed=SEED) -> dict:
    return paired_episode_cluster_bootstrap(a, b, eid, n_boot=N_BOOT, seed=seed)


def r4(x) -> float:
    return round(float(x), 4)
