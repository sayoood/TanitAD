"""RA-1 data: the S4 pooled DINOv3-L cache -> training segments and eval windows.

Cache contract (``Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py``, read-only here): one
``<cache>/<split>/clips/<clip>.pt`` per clip (``torch.load(weights_only=True)``) with ``pool4x10`` fp8 ``[T, 40, 1024]``
(``pool8x20`` fp8 ``[T, 160, 1024]`` optional), ``v / a / steer`` ``[T]``, ``pose_grid`` ``[T, 4]`` = (x, y, yaw, v), ``kin3``
``[T, 3]`` = (v0, a0, steer0) and ``p10`` ``[T, 8]`` (row 0 NaN), and ``<cache>/<split>/manifest.jsonl``.

NOTHING is re-implemented: the kd_x base (``action_units="steer"`` -- ``actions[:,0]`` is a road-wheel ANGLE, RETRACTION_LOG R46),
the GT waypoints, the clip order and the train/eval disjointness check are ``tanitad.data.v8f_mini_data``'s
(``kdx_base`` / ``gt_future`` / ``sha12`` / ``read_manifest`` / ``assert_disjoint``); the P10 prior is
``tanitad.models.v8f_mini.P10Prior`` (frozen; ``load_p10``).

TRAINING SEGMENTS: a segment is ``T1`` consecutive cache frames ``[s, s+T1)``. The world-model losses use the whole segment; the
trajectory loss is applied at EVERY frame of it whose 6 s future exists (``valid``). A frame is valid iff ``t >= 3`` (the eval grid's
first window), its P10 features are finite and ``t + 30 <= T - 1``. Sampling is a pure function of ``(seed, step)`` so a resumed run
draws exactly what an uninterrupted one would have.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tanitad.data import v8f_mini_data as D

HORIZON = D.HORIZON            # 30
T_FIRST = 3                    # the trunk-probe grid's first window


def grid_key(grid: str) -> str:
    return "pool" + grid


def list_clips(cache_dir, split) -> list:
    """Manifest order (file order, last line per clip wins) -> ``[(clip, n_frames)]``."""
    return D.read_manifest(cache_dir, split)


def pick_train_clips(clips, n: int = 0) -> list:
    """The nested train prefix: ordered by sha12 (the S4 / P10-fit order), first ``n`` (0 = all)."""
    order = sorted(clips, key=D.sha12)
    return order[:n] if n else order


def load_clip(path, prior, grid: str = "4x10") -> dict:
    """One clip -> tensors. ``prior`` is a frozen ``P10Prior`` (its trajectory and its feature standardiser are precomputed per
    frame here, so a batch is a gather)."""
    o = torch.load(path, map_location="cpu", weights_only=True)
    key = grid_key(grid)
    if key not in o:
        raise KeyError(f"{path}: no {key} (has {[k for k in o if k.startswith('pool')]})")
    pg = o["pose_grid"].double().numpy()
    kin3 = o["kin3"].double().numpy()
    T = len(pg)
    pool = o[key]
    assert pool.shape[0] == T, f"{path}: {pool.shape[0]} frames of tokens vs {T} poses"
    base = D.kdx_base(kin3)                                    # [T,30,2] float32, steer unit, NaN row -> 0
    gt, gt_ok = D.gt_future(pg)                                # [T,30,2], t + 30 <= T - 1
    p10f = o["p10"].float()
    fin = torch.isfinite(p10f).all(1)
    f8 = torch.nan_to_num(p10f)
    with torch.no_grad():
        p10_traj = prior(f8, torch.from_numpy(base))           # [T,30,2]
        f8n = prior.standardise(f8)
    a = torch.nan_to_num(o["a"].float())
    act = torch.stack([a, o["steer"].float(), o["v"].float()], -1)           # RAW a, steer, v
    valid = torch.from_numpy(gt_ok) & fin & (torch.arange(T) >= T_FIRST) & torch.isfinite(torch.from_numpy(gt)).all(-1).all(-1)
    return dict(pool=pool, act=act, f8n=f8n, p10_traj=p10_traj, gt=torch.from_numpy(gt), valid=valid,
                pose=torch.from_numpy(pg).float(), kin3=torch.from_numpy(kin3).float(), T=T)


class ClipBank:
    """In-RAM clips (tokens stay fp8: 4x10 is ~4 MB/clip). ``load`` is idempotent and ``keep`` evicts everything else."""

    def __init__(self, cache_dir, split, prior, grid: str = "4x10"):
        self.dir = Path(cache_dir) / split / "clips"
        self.split, self.prior, self.grid = split, prior, grid
        self.clips: dict = {}

    def load(self, names) -> None:
        for c in names:
            if c not in self.clips:
                self.clips[c] = load_clip(self.dir / f"{c}.pt", self.prior, self.grid)

    def keep(self, names) -> None:
        keep = set(names)
        for c in list(self.clips):
            if c not in keep:
                del self.clips[c]

    def token_gb(self) -> float:
        return sum(d["pool"].numel() for d in self.clips.values()) / 2 ** 30

    def act_stats(self, names=None) -> tuple:
        """Per-channel mean / std of the RAW action channels over the given clips' frames (train statistics; saved in the model)."""
        a = torch.cat([self.clips[c]["act"] for c in (names or self.clips)], 0)
        return a.mean(0), a.std(0).clamp_min(1e-6)


def chunk_clips(all_clips, seed: int, step: int, chunk_clips: int, chunk_steps: int) -> list:
    """The clips resident for ``step``: all of them when ``chunk_clips == 0``, else a seeded random subset that changes every
    ``chunk_steps`` steps (a pure function of ``(seed, step // chunk_steps)``)."""
    if not chunk_clips or chunk_clips >= len(all_clips):
        return list(all_clips)
    rng = np.random.default_rng((seed, 7919, step // chunk_steps))
    idx = rng.permutation(len(all_clips))[:chunk_clips]
    return [all_clips[i] for i in sorted(idx)]


def sample_batch(bank: ClipBank, names, seed: int, step: int, batch: int, t1: int) -> dict:
    """``batch`` segments of ``t1`` frames, drawn as a pure function of ``(seed, step)``. Each segment must contain >= 1 valid
    frame (start in ``[0, min(T-31, T-t1)]``)."""
    rng = np.random.default_rng((seed, 104729, step))
    cl = [names[i] for i in rng.integers(0, len(names), size=batch)]
    parts = {k: [] for k in ("tokens", "act", "f8n", "p10_traj", "gt", "valid")}
    for c in cl:
        d = bank.clips[c]
        hi = min(d["T"] - HORIZON - 1, d["T"] - t1)
        if hi < 0:
            raise ValueError(f"clip {c}: T={d['T']} is shorter than a {t1}-frame segment plus a 6 s future")
        s = int(rng.integers(0, hi + 1))
        sl = slice(s, s + t1)
        parts["tokens"].append(d["pool"][sl])
        for k, src in (("act", "act"), ("f8n", "f8n"), ("p10_traj", "p10_traj"), ("gt", "gt"), ("valid", "valid")):
            parts[k].append(d[src][sl])
    out = {k: torch.stack(v) for k, v in parts.items()}
    if not bool(out["valid"].any(1).all()):
        # a segment with no valid frame contributes nothing; with start <= T-31 and t1 >= 4 this cannot happen
        raise AssertionError("a sampled segment has no valid frame")
    return out


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ------------------------------------------------------------------------------------------------ eval windows
def eval_windows(bank: ClipBank, order, grid_json=None, stride: int = 1) -> list:
    """``[(clip, t)]``. With ``grid_json`` (the trunk-probe grid ``{"order": [...], "windows": {clip: [t...]}}``, 141 episodes /
    2,399 windows) exactly those windows (each must be valid); else every valid frame at ``stride``."""
    out = []
    if grid_json:
        g = json.load(open(grid_json, encoding="utf-8"))
        for c in g["order"]:
            for t in g["windows"][c]:
                if not bool(bank.clips[c]["valid"][int(t)]):
                    raise ValueError(f"grid window ({c}, {t}) is not valid in the cache (needs t>={T_FIRST}, finite P10 row, t+30<=T-1)")
                out.append((c, int(t)))
        return out
    for c in order:
        ts = torch.nonzero(bank.clips[c]["valid"]).reshape(-1).tolist()
        out += [(c, t) for t in ts[::stride]]
    return out


def strata_labels(pose: torch.Tensor, t: int) -> dict:
    """EVENT STRATA from GT ego motion only (labels may use anything; thresholds fixed here, BEFORE any RA-1 data):
    ``turn`` |yaw(t+30) - yaw(t)| >= 0.35 rad (20 deg); ``stop`` v(t) > 2 m/s and min v over t+20..t+30 < 0.5 m/s;
    ``brake`` mean decel over the next 3 s >= 1.0 m/s^2. LEAD BRAKING needs a lead track the cache does not carry -> not computable."""
    yaw, v = pose[:, 2], pose[:, 3]
    dyaw = float(torch.atan2(torch.sin(yaw[t + HORIZON] - yaw[t]), torch.cos(yaw[t + HORIZON] - yaw[t])))
    return dict(turn=abs(dyaw) >= 0.35,
                stop=bool(v[t] > 2.0 and float(v[t + 20:t + HORIZON + 1].min()) < 0.5),
                brake=bool((float(v[t]) - float(v[t + 15])) / 3.0 >= 1.0))
