"""RA-1 data: the S4 pooled DINOv3-L cache -> training segments and eval windows.

Cache contract (``Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py``, read-only here): one
``<cache>/<split>/clips/<clip>.pt`` per clip (``torch.load(weights_only=True)``) with ``pool4x10`` fp8 ``[T, 40, 1024]``
(``pool8x20`` fp8 ``[T, 160, 1024]`` optional), ``v / a / steer`` ``[T]``, ``pose_grid`` ``[T, 4]`` = (x, y, yaw, v), ``kin3``
``[T, 3]`` = (v0, a0, steer0) and ``p10`` ``[T, 8]`` (row 0 NaN), and ``<cache>/<split>/manifest.jsonl``.

KINEMATIC MODE (PI ruling D-PI-YAWRATE-1, 2026-10-17: *"the model should get the measured yaw rate"*): ``kin_mode="steer"`` is the legacy
representation above; ``kin_mode="yaw"`` (the trainers' default) builds kin3, the 8 P10 features, the kd_x base (CURVATURE unit) and
action channel 1 from the NATIVE pose track (``native_tracks.pt``) through ``v8f_mini_data.yaw_kin_from_track`` and never reads the
cache's ``steer`` / ``kin3`` / ``p10`` / ``yaw_rate``. The action vector is then ``(a, yaw_rate, v)``.

NOTHING is re-implemented: the kd_x base (``action_units="steer"`` -- ``actions[:,0]`` is a road-wheel ANGLE, RETRACTION_LOG R46),
the GT waypoints, the clip order and the train/eval disjointness check are ``tanitad.data.v8f_mini_data``'s
(``kdx_base`` / ``gt_future`` / ``sha12`` / ``read_manifest`` / ``assert_disjoint``); the P10 prior is
``tanitad.models.v8f_mini.P10Prior`` (frozen; ``load_p10``).

TRAINING SEGMENTS: a segment is ``T1`` consecutive cache frames ``[s, s+T1)``. The world-model losses use the whole segment; the
trajectory loss is applied at EVERY frame of it whose 6 s future exists (``valid``). A frame is valid iff ``t >= 3`` (the eval grid's
first window), its P10 features are finite and ``t + 30 <= T - 1``. Sampling is a pure function of ``(seed, step)`` so a resumed run
draws exactly what an uninterrupted one would have.

HOLDOUT (PREREG_HJ_STAGE0 Amendment 1): the A1 (beta, gamma) cell is selected on windows that do NOT score K. ``holdout_clips`` is a
fixed, episode-disjoint set of ``holdout_n`` (60) of the 600 stage-A TRAIN clips -- the LAST ``holdout_n`` of the first ``pool_n`` (600) in
sha12 order -- and ``split_train_holdout`` removes it from the training clips of EVERY arm. The holdout is a function of the manifest
and ``(holdout_n, pool_n)`` ONLY (never of ``--n-clips``, the arm or the seed), and the 600-clip cache is the nested sha12 prefix of the
4,572, so the same 60 clips are withheld at 600 and at 4,572.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tanitad.data import v8f_mini_data as D

HORIZON = D.HORIZON            # 30
T_FIRST = 3                    # the trunk-probe grid's first window
STAGE_A_CLIPS = 600            # PREREG Amendment 1: the stage-A train pool = the first 600 clips in sha12 order
HOLDOUT_CLIPS = 60             # ... of which the last 60 are the selection holdout (the queue passes ``--holdout-n 60``)


def grid_key(grid: str) -> str:
    return "pool" + grid


def list_clips(cache_dir, split) -> list:
    """Manifest order (file order, last line per clip wins) -> ``[(clip, n_frames)]``."""
    return D.read_manifest(cache_dir, split)


def pick_train_clips(clips, n: int = 0) -> list:
    """The nested train prefix: ordered by sha12 (the S4 / P10-fit order), first ``n`` (0 = all)."""
    order = sorted(clips, key=D.sha12)
    return order[:n] if n else order


def clips_sha256(names) -> str:
    """Order-sensitive digest of a clip-id list: what a run's ``meta.json`` / checkpoint record, and what the evaluator re-derives."""
    return hashlib.sha256(chr(10).join(names).encode("utf-8")).hexdigest()


def holdout_clips(clips, holdout_n: int, pool_n: int = STAGE_A_CLIPS) -> list:
    """The selection holdout: the LAST ``holdout_n`` of the first ``pool_n`` clips in sha12 order. A function of the clip set and
    ``(holdout_n, pool_n)`` only. ``holdout_n`` < 1 is refused (a holdout of zero is a selection on the scoring windows)."""
    if holdout_n < 1:
        raise ValueError(f"holdout_n must be >= 1 (got {holdout_n}): PREREG Amendment 1 withholds a holdout from EVERY arm")
    if holdout_n >= pool_n:
        raise ValueError(f"holdout_n ({holdout_n}) must be smaller than the stage-A pool ({pool_n})")
    order = sorted(clips, key=D.sha12)
    if len(order) < pool_n:
        raise ValueError(f"the train manifest holds {len(order)} clips, fewer than the {pool_n}-clip stage-A pool the holdout is cut from")
    return order[pool_n - holdout_n:pool_n]


def split_train_holdout(clips, n_clips: int, holdout_n: int, pool_n: int = STAGE_A_CLIPS) -> tuple:
    """``(train, holdout)``. ``train`` = the nested prefix ``pick_train_clips(clips, n_clips)`` MINUS the holdout (for every arm, at
    every ``n_clips``); ``holdout`` is independent of ``n_clips``. Disjointness is asserted here, where it is made."""
    hold = holdout_clips(clips, holdout_n, pool_n)
    hs = set(hold)
    train = [c for c in pick_train_clips(clips, n_clips) if c not in hs]
    if hs & set(train):
        raise AssertionError("holdout/train overlap after the split")
    return train, hold


def load_clip(path, prior, grid: str = "4x10", kin_mode: str = "steer", track=None) -> dict:
    """One clip -> tensors. ``prior`` is a frozen ``P10Prior`` (its trajectory and its feature standardiser are precomputed per
    frame here, so a batch is a gather). ``kin_mode="yaw"`` needs the clip's native pose ``track`` (see the module docstring)."""
    D.check_kin_mode(kin_mode)
    o = torch.load(path, map_location="cpu", weights_only=True)
    key = grid_key(grid)
    if key not in o:
        raise KeyError(f"{path}: no {key} (has {[k for k in o if k.startswith('pool')]})")
    pg = o["pose_grid"].double().numpy()
    T = len(pg)
    pool = o[key]
    assert pool.shape[0] == T, f"{path}: {pool.shape[0]} frames of tokens vs {T} poses"
    if kin_mode == "yaw":
        # the cache's kin3 / p10 / steer / yaw_rate are NOT READ (steer-derived, or a forward difference); the native track is
        # checked against the cache's own pose_grid before one number is taken from it.
        if track is None:
            raise ValueError(f"{path}: kin_mode 'yaw' needs this clip's native pose track (native_tracks.pt)")
        D.check_track_matches_grid(track, pg, Path(path).stem)
        kin3, omega = D.yaw_kin_from_track(track)
        p10f = torch.full((T, 8), float("nan"))
        p10f[1:] = torch.from_numpy(D._fk().kin_features(kin3[1:]).astype(np.float32))
        # a TRAIN track starts at raw frame 2 (the v2 manifest), so omega is undefined at cache step 1: the ACTION channel is filled with
        # 0.0 there, exactly as ego_channels_from_poses fills step 0 (a NaN in `act` would NaN the world-model loss of every window that
        # contains it). kin3 / p10f stay NaN on those rows, so they are never `valid`.
        ch1 = torch.from_numpy(np.nan_to_num(omega, nan=0.0).astype(np.float32))
        base = D.kdx_base(kin3, D.KIN_UNITS["yaw"])            # [T,30,2] float32, CURVATURE unit, NaN row -> 0
    else:
        kin3 = o["kin3"].double().numpy()
        base = D.kdx_base(kin3)                                # [T,30,2] float32, steer unit, NaN row -> 0
        p10f = o["p10"].float()
        ch1 = o["steer"].float()
    gt, gt_ok = D.gt_future(pg)                                # [T,30,2], t + 30 <= T - 1
    fin = torch.isfinite(p10f).all(1)
    f8 = torch.nan_to_num(p10f)
    with torch.no_grad():
        p10_traj = prior(f8, torch.from_numpy(base))           # [T,30,2]
        f8n = prior.standardise(f8)
    a = torch.nan_to_num(o["a"].float())
    act = torch.stack([a, ch1, o["v"].float()], -1)                          # RAW a, (steer | yaw_rate), v
    valid = torch.from_numpy(gt_ok) & fin & (torch.arange(T) >= T_FIRST) & torch.isfinite(torch.from_numpy(gt)).all(-1).all(-1)
    return dict(pool=pool, act=act, f8n=f8n, p10_traj=p10_traj, gt=torch.from_numpy(gt), valid=valid,
                pose=torch.from_numpy(pg).float(), kin3=torch.from_numpy(kin3).float(), T=T)


class ClipBank:
    """In-RAM clips (tokens stay fp8: 4x10 is ~4 MB/clip). ``load`` is idempotent and ``keep`` evicts everything else."""

    def __init__(self, cache_dir, split, prior, grid: str = "4x10", kin_mode: str = "steer", tracks: dict | None = None):
        self.dir = Path(cache_dir) / split / "clips"
        self.split, self.prior, self.grid = split, prior, grid
        self.kin_mode, self.tracks = D.check_kin_mode(kin_mode), tracks
        if self.kin_mode == "yaw" and not tracks:
            raise ValueError("kin_mode 'yaw' needs the native pose tracks (native_tracks.pt: --tracks)")
        self.clips: dict = {}

    def load(self, names) -> None:
        for c in names:
            if c not in self.clips:
                if self.kin_mode == "yaw" and c not in self.tracks:
                    raise KeyError(f"{c}: no native pose track in the sidecar")
                self.clips[c] = load_clip(self.dir / f"{c}.pt", self.prior, self.grid, self.kin_mode,
                                          self.tracks[c] if self.kin_mode == "yaw" else None)

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
