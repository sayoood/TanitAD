"""Replay engine: iterate cached episodes as batched windows, run N arms.

The engine owns everything the arms share so the comparison is airtight:
identical windows (same last-frame anchor for every arm, arms with a shorter
causal window slice the TAIL of the engine window), identical ground truth
(waypoints in the repo `_ego` convention, action-at-t semantics), identical
iteration order (deterministic: corpora sorted, episodes sorted, fixed
stride). Per (arm, window) it collects an :class:`ArmOutput` and emits one
:class:`TimestepRecord` per window — the single currency consumed by both
:mod:`tanitad.replay.stats` (test mode) and :mod:`tanitad.replay.rr_log`
(viz mode).

Data contract: episodes are the repo-standard ``ep_*.pt`` cache files
(:func:`tanitad.data.mixing.load_episode`, mmap-backed, uint8 frames). The
corpus tag is the cache-directory name. Everything is fail-loud: missing
caches, too-short episodes, and empty fit splits raise — no silent skips
(REF-B windowing doctrine, 2026-07-10 review).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from tanitad.data._contract import to_float_frames
from tanitad.data.mixing import load_episode
from tanitad.data.toy_driving import ToyEpisode

# The D1 waypoint targets: 0.5/1/1.5/2 s @ 10 Hz. All arms decode/emit
# waypoints at exactly these steps (RefB's tactical horizons are pinned to
# them at construction) so ADE/FDE compare like with like.
WAYPOINT_STEPS: tuple[int, ...] = (5, 10, 15, 20)
DT = 0.1                                    # episode contract: 10 Hz


def ego_frame(dxy: Tensor, yaw: Tensor) -> Tensor:
    """Rotate world displacements [..., 2] into the ego frame at ``yaw``.

    The repo `_ego` convention (scripts/d1_probe_capacity.py): +x = forward,
    +y = left (CCW-positive yaw).
    """
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


# --------------------------------------------------------------------------
# Episode loading / splitting
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ReplayEpisode:
    """One replayable episode plus its corpus tag (cache-dir name)."""
    corpus: str
    episode: ToyEpisode


def load_corpora(data_root: str | Path, episodes: int = 0,
                 pattern: str = "*") -> list[ReplayEpisode]:
    """Load ``ep_*.pt`` episodes (mmap) from ``data_root``.

    Two accepted layouts:
      1. ``data_root`` itself contains ``ep_*.pt`` -> one corpus, tag =
         ``data_root.name`` (point this at a single cache dir).
      2. ``data_root`` contains cache subdirectories matching ``pattern``
         (e.g. ``*val*``), each with ``ep_*.pt`` -> one corpus per subdir,
         tag = subdir name.

    ``episodes`` > 0 bounds the episode count PER corpus (sorted filename
    order — deterministic). Raises FileNotFoundError if nothing matches:
    an empty replay is a configuration error, never a green run.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"replay data root is not a directory: {root}")

    def _load_dir(d: Path) -> list[ReplayEpisode]:
        files = sorted(d.glob("ep_*.pt"))
        if episodes > 0:
            files = files[:episodes]
        return [ReplayEpisode(d.name, load_episode(str(p), mmap=True))
                for p in files]

    if any(root.glob("ep_*.pt")):
        reps = _load_dir(root)
    else:
        subdirs = sorted(d for d in root.glob(pattern)
                         if d.is_dir() and any(d.glob("ep_*.pt")))
        if not subdirs:
            raise FileNotFoundError(
                f"no ep_*.pt files in {root} and no cache subdirectory "
                f"matching {pattern!r} contains any — check --data-root")
        reps = [r for d in subdirs for r in _load_dir(d)]
    if not reps:
        raise FileNotFoundError(f"no episodes loaded from {root}")
    return reps


def split_fit_replay(reps: Sequence[ReplayEpisode], fit_frac: float = 0.5
                     ) -> tuple[list[ReplayEpisode], list[ReplayEpisode]]:
    """Episode-level fit/replay split, per corpus (I3: never split windows
    of one episode across sets).

    Per corpus the LEADING ``round(n * fit_frac)`` episodes (deterministic
    sorted order) go to the probe-fit set, the rest to replay — the same
    leading-half convention as scripts/viz_trajectory_fan.py. Raises if the
    split would leave a corpus without replay episodes.
    """
    if not 0.0 <= fit_frac < 1.0:
        raise ValueError(f"fit_frac must be in [0, 1), got {fit_frac}")
    by_corpus: dict[str, list[ReplayEpisode]] = {}
    for r in reps:
        by_corpus.setdefault(r.corpus, []).append(r)
    fit: list[ReplayEpisode] = []
    replay: list[ReplayEpisode] = []
    for corpus, group in by_corpus.items():
        n_fit = max(1, int(round(len(group) * fit_frac))) if fit_frac > 0 \
            else 0
        if fit_frac > 0 and n_fit >= len(group):
            raise ValueError(
                f"corpus {corpus!r}: fit_frac={fit_frac} leaves no replay "
                f"episodes (n={len(group)}) — load more episodes or lower "
                f"--fit-frac")
        fit.extend(group[:n_fit])
        replay.extend(group[n_fit:])
    return fit, replay


# --------------------------------------------------------------------------
# Window batches
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class WindowRef:
    """Identity of one window: episode handle + frame anchoring."""
    corpus: str
    episode: ToyEpisode = field(repr=False)
    ep_index: int                 # position in the run's episode list
    episode_id: int
    t0: int                       # first frame of the ENGINE window
    last: int                     # t0 + engine_window - 1 (the anchor frame)


@dataclass(eq=False)
class WindowBatch:
    """One batch of aligned windows plus shared ground truth.

    ``frames`` span the ENGINE window (max over arm windows); arms with a
    shorter window slice ``frames[:, -arm.window:]`` so every arm's causal
    history ends on the same anchor frame ``refs[j].last``.
    """
    frames: Tensor                # [B, W, C, H, W'] float32 [0,1], on device
    actions: Tensor               # [B, W, 2], on device
    refs: list[WindowRef]
    gt_waypoints: Tensor          # [B, len(WAYPOINT_STEPS), 2] ego, CPU
    gt_action: Tensor             # [B, 2] action at the anchor frame, CPU
    speed: Tensor                 # [B] v at anchor, CPU
    yaw_rate: Tensor              # [B] rad/s at anchor (finite diff), CPU

    def __len__(self) -> int:
        return len(self.refs)


def gt_disp_at(batch: WindowBatch, k: int) -> Tensor:
    """GT ego displacement at ``anchor + k`` steps for every window: [B, 2].

    Used by arms to fit/score imagination probes at their own predictor
    horizons (which need not coincide with WAYPOINT_STEPS).
    """
    rows = []
    for ref in batch.refs:
        poses = ref.episode.poses
        rows.append(ego_frame(poses[ref.last + k, :2] - poses[ref.last, :2],
                              poses[ref.last, 2]))
    return torch.stack(rows)


def future_frames_at(batch: WindowBatch, k: int) -> Tensor:
    """Frames at ``anchor + k`` for every window: [B, C, H, W'] float [0,1],
    on the batch device. Arms encode these to score imagination against
    reality (imag_rel)."""
    rows = [to_float_frames(ref.episode.frames[ref.last + k])
            for ref in batch.refs]
    return torch.stack(rows).to(batch.frames.device)


def frame_u8(ep: ToyEpisode, t: int) -> np.ndarray:
    """Frame ``t`` as displayable uint8: [H, W, 3] (camera stacks: latest 3
    channels) or [H, W] (single-channel BEV)."""
    f = ep.frames[t]
    if f.dtype != torch.uint8:
        f = (f.clamp(0, 1) * 255).to(torch.uint8)
    if f.shape[0] >= 3:
        return f[-3:].permute(1, 2, 0).contiguous().numpy()
    return f[0].contiguous().numpy()


# --------------------------------------------------------------------------
# Records — the single currency between engine, stats and rr_log
# --------------------------------------------------------------------------

@dataclass(eq=False)
class ArmOutput:
    """Everything one arm produced for one window. ``None`` = the arm does
    not have that head (a structural fact, not a silent skip)."""
    latency_ms: float                                 # control path only
    waypoints: np.ndarray | None = None               # [H, 2] ego metres
    waypoint_steps: tuple[int, ...] = WAYPOINT_STEPS
    action: np.ndarray | None = None                  # [2] (steer, accel)
    action_seq: np.ndarray | None = None              # [K, 2] (REF-B 0.5 s)
    maneuver_probs: np.ndarray | None = None          # [M] softmax (REF-B)
    maneuver_gt: int | None = None                    # pseudo-label (REF-B)
    nav_cmd: int | None = None                        # strategic input (REF-B)
    conf: float | None = None                         # predicted own error
    ood: float | None = None                          # feature-OOD score
    sigma: float | None = None                        # H15 mean belief sigma
    imag_rel: dict[int, float] | None = None          # per predictor horizon
    imag_traj: dict[int, np.ndarray] | None = None    # k -> [2] imag decode


@dataclass(eq=False)
class TimestepRecord:
    """One replayed window: shared ground truth + every arm's output."""
    step: int                     # global timeline (monotonic across episodes)
    corpus: str
    episode_id: int
    ep_index: int
    t: int                        # anchor frame index within the episode
    gt_waypoints: np.ndarray      # [len(WAYPOINT_STEPS), 2]
    gt_action: np.ndarray         # [2]
    speed: float
    yaw_rate: float
    arms: dict[str, ArmOutput]
    frame: np.ndarray | None = None   # uint8 [H,W,3]/[H,W] when emit_frames


# --------------------------------------------------------------------------
# Arm protocol
# --------------------------------------------------------------------------

@runtime_checkable
class ArmAdapter(Protocol):
    """What the engine requires of an architecture arm.

    Implementations: :class:`tanitad.replay.arms.MainArm`,
    :class:`~tanitad.replay.arms.RefAArm`, :class:`~tanitad.replay.arms.RefBArm`.
    """
    name: str
    window: int                   # causal window length the arm consumes
    needs_ahead: int              # future steps required beyond the anchor
    requires_fit: bool            # True -> prepare() must run before replay

    def prepare(self, engine: "ReplayEngine",
                fit_reps: Sequence[ReplayEpisode]) -> None:
        """Fit probes / warm caches on the held-out fit split."""
        ...

    def run_batch(self, batch: WindowBatch) -> list[ArmOutput]:
        """One output per window in the batch, order-aligned with refs."""
        ...


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

class ReplayEngine:
    """Batched open-loop replay of cached episodes through N arms.

    Parameters
    ----------
    arms : arm adapters (unique names). The engine window is the max over
        arm windows; the look-ahead requirement is the max over arm
        ``needs_ahead`` and the D1 waypoint steps.
    device : torch device string for the frame/action tensors.
    batch_size : windows per forward batch (batches never span episodes).
    stride : window start stride during replay (fit uses ``fit_stride``).
    half : autocast fp16 on CUDA (exposed to arms via :meth:`autocast`).
    emit_frames : attach a displayable uint8 anchor frame to every record
        (viz mode; off in test mode to keep memory flat).
    """

    def __init__(self, arms: Sequence[ArmAdapter], device: str = "cpu",
                 batch_size: int = 8, stride: int = 8, half: bool = False,
                 emit_frames: bool = False, fit_stride: int | None = None,
                 min_fit_windows: int = 16, max_fit_windows: int = 4096):
        if not arms:
            raise ValueError("ReplayEngine needs at least one arm")
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")
        if batch_size < 1 or stride < 1:
            raise ValueError(f"batch_size/stride must be >= 1, got "
                             f"{batch_size}/{stride}")
        self.arms = list(arms)
        self.device = device
        self.batch_size = batch_size
        self.stride = stride
        self.half = half
        self.emit_frames = emit_frames
        self.fit_stride = fit_stride if fit_stride is not None else stride
        self.min_fit_windows = min_fit_windows
        self.max_fit_windows = max_fit_windows
        self.window = max(a.window for a in self.arms)
        self.need_ahead = max(max(WAYPOINT_STEPS),
                              *(a.needs_ahead for a in self.arms))
        for a in self.arms:      # bind: arms use engine.autocast()/sync()
            a.engine = self

    # -- device/precision helpers -----------------------------------------
    def autocast(self):
        """fp16 autocast context on CUDA when ``half`` is set (F-5 lever);
        a no-op elsewhere so arms can wrap their control path unconditionally."""
        if self.half and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def sync(self) -> None:
        """CUDA barrier for honest latency timing (no-op on CPU)."""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    # -- window iteration ---------------------------------------------------
    def iter_batches(self, reps: Sequence[ReplayEpisode],
                     stride: int | None = None) -> Iterator[WindowBatch]:
        """Yield deterministic window batches over ``reps``.

        Anchoring is byte-identical to the eval scripts: window frames
        ``t0 .. t0+W-1``, ground truth at ``last = t0+W-1`` (waypoints at
        ``last + k``, action at ``last``). Raises on episodes too short for
        one window + look-ahead — no silent skips.
        """
        stride = self.stride if stride is None else stride
        w, need = self.window, self.need_ahead
        for ep_index, rep in enumerate(reps):
            ep = rep.episode
            T = int(ep.frames.shape[0])
            t_max = T - w - need
            if t_max <= 0:
                raise ValueError(
                    f"episode {rep.corpus}/{ep.episode_id} too short for "
                    f"replay: T={T} <= window({w}) + need_ahead({need}) — "
                    f"no silent skips; fix the cache or the arm horizons")
            starts = list(range(0, t_max, stride))
            for i in range(0, len(starts), self.batch_size):
                chunk = starts[i:i + self.batch_size]
                yield self._build_batch(rep, ep_index, chunk)

    def _build_batch(self, rep: ReplayEpisode, ep_index: int,
                     starts: list[int]) -> WindowBatch:
        ep, w = rep.episode, self.window
        frames = torch.stack(
            [to_float_frames(ep.frames[t:t + w]) for t in starts]
        ).to(self.device)
        actions = torch.stack(
            [ep.actions[t:t + w] for t in starts]).to(self.device)
        refs, wps, acts, speeds, yaw_rates = [], [], [], [], []
        for t in starts:
            last = t + w - 1
            refs.append(WindowRef(rep.corpus, ep, ep_index,
                                  int(ep.episode_id), t, last))
            yaw0, p0 = ep.poses[last, 2], ep.poses[last, :2]
            wps.append(torch.stack(
                [ego_frame(ep.poses[last + k, :2] - p0, yaw0)
                 for k in WAYPOINT_STEPS]))
            acts.append(ep.actions[last])
            speeds.append(ep.poses[last, 3])
            yaw_rates.append((ep.poses[last, 2] - ep.poses[last - 1, 2]) / DT)
        return WindowBatch(frames=frames, actions=actions, refs=refs,
                           gt_waypoints=torch.stack(wps),
                           gt_action=torch.stack(acts),
                           speed=torch.stack(speeds),
                           yaw_rate=torch.stack(yaw_rates))

    # -- probe fitting -------------------------------------------------------
    def prepare(self, fit_reps: Sequence[ReplayEpisode]) -> None:
        """Run every fit-requiring arm's ``prepare`` on the fit split."""
        needy = [a for a in self.arms if a.requires_fit]
        if needy and not fit_reps:
            raise ValueError(
                f"arms {[a.name for a in needy]} require a probe-fit split "
                f"but the fit set is empty — raise --fit-frac or load more "
                f"episodes")
        for arm in needy:
            arm.prepare(self, fit_reps)

    # -- the replay loop -----------------------------------------------------
    def run(self, reps: Sequence[ReplayEpisode]) -> Iterator[TimestepRecord]:
        """Replay ``reps`` and yield one :class:`TimestepRecord` per window.

        Streaming by design: viz mode logs each record and drops the frame,
        so memory stays flat regardless of corpus size.
        """
        if not reps:
            raise ValueError("nothing to replay: episode list is empty")
        step = 0
        for batch in self.iter_batches(reps):
            outs = {arm.name: arm.run_batch(batch) for arm in self.arms}
            for name, arm_outs in outs.items():
                if len(arm_outs) != len(batch):
                    raise RuntimeError(
                        f"arm {name!r} returned {len(arm_outs)} outputs for "
                        f"a batch of {len(batch)} windows")
            for j, ref in enumerate(batch.refs):
                yield TimestepRecord(
                    step=step, corpus=ref.corpus,
                    episode_id=ref.episode_id, ep_index=ref.ep_index,
                    t=ref.last,
                    gt_waypoints=batch.gt_waypoints[j].numpy(),
                    gt_action=batch.gt_action[j].numpy(),
                    speed=float(batch.speed[j]),
                    yaw_rate=float(batch.yaw_rate[j]),
                    arms={name: outs[name][j] for name in outs},
                    frame=(frame_u8(ref.episode, ref.last)
                           if self.emit_frames else None))
                step += 1


class LatencyTimer:
    """Wall-clock timer for one arm's control path over a batch.

    Usage::

        with LatencyTimer(engine) as lt:
            ...forward...
        per_window_ms = lt.ms / len(batch)

    Synchronizes CUDA on entry/exit so fp16/async kernels are billed to the
    arm that launched them.
    """

    def __init__(self, engine: ReplayEngine):
        self.engine = engine
        self.ms = float("nan")

    def __enter__(self) -> "LatencyTimer":
        self.engine.sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.engine.sync()
        self.ms = (time.perf_counter() - self._t0) * 1000.0
