"""comma2k19 -> TanitAD episode contract (D-009: real camera data first).

comma2k19: 33 h of commute driving (comma.ai), 20 fps front camera + CAN
(speed, steering-wheel angle) + global pose — real actions and real
trajectory targets with zero annotation. Source used here: the ungated
HuggingFace mirror ``commaai/comma2k19`` (``raw_data/Chunk_*.zip``), official
layout preserved:

    Chunk_X/<dongle_id|date--time>/<segment>/            (1-minute segments)
        video.hevc                                       20 fps, 1164x874
        processed_log/CAN/speed/{t,value}                m/s
        processed_log/CAN/steering_angle/{t,value}       deg (steering wheel)
        global_pose/frame_times                          s, one per frame
        global_pose/frame_positions                      ECEF meters [T,3]
        global_pose/frame_velocities                     ECEF m/s   [T,3]

Contract produced (camera variant of the toy contract):
    frames  [T, 6, S, S]   2 consecutive RGB frames channel-stacked (t-1, t),
                           float32 in [0,1] — consequence-dominance needs the
                           motion visible inside one input (A8)
    actions [T, 2]         (road-wheel steer rad, longitudinal accel m/s^2)
    poses   [T, 4]         (x_east, y_north, yaw, v) in a segment-local ENU frame
    episode_id             stable int hash of the segment path

Splits are ROUTE-level (I3): split_by_route() groups segments by their
<dongle|date> route folder — never split by frame or by segment of the same
drive. v0 notes: geocentric-latitude ENU approximation (sub-0.3 % tilt over a
1-min segment); constant steering ratio 15.3 (Civic/Corolla-class) for
wheel->road angle; both documented for the D1 gate report.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tanitad.data.toy_driving import ToyEpisode

FPS = 20
STEER_RATIO = 15.3           # steering wheel -> road wheel, v0 constant
WHEEL_TO_RAD = math.pi / 180.0

# --------------------------------------------------------------------------- #
# Heading derivation mode (IDM v3, 2026-07-27)                                 #
# --------------------------------------------------------------------------- #
# comma2k19 heading is `arctan2(enu_v_north, enu_v_east)` — the direction of the
# ENU VELOCITY vector, which is **undefined when the vehicle is stationary**.
# MEASURED on the 64-segment val build: in the v < 0.5 m/s bin, **26.27 % of
# frames carry a physically impossible |yaw_rate|** (up to 15.53 rad/s = 890
# deg/s, at speeds of 0.00-0.01 m/s); in every bin above 0.5 m/s the figure is
# **0.000 %**. The defect is razor-sharp and confined to standstill.
#
# What it cost: the deployed IDM head's pooled `yaw_rate` R2 read **0.1046**
# against these labels and **0.8108** against the repaired ones, with NOTHING
# retrained — the channel was never a model failure.  Per corpus (never quote
# the pooled number): comma2k19 **+0.0114 -> +0.3308**; PhysicalAI **+0.9035
# unchanged, bit-identical** (n_pai_changed = 0).
#   *(CORRECTED 2026-07-27 by the comma-yaw-reissue pass: this comment said
#   "0.83", which is the level a head RETRAINED on repaired labels reaches
#   (R0, pooled 0.8413), not the deployed head's.  The deployed head's repaired
#   pooled R2 is 0.8108 — read from
#   `…/2026-07-27-idm-v3/results/compare_v3.json` ->
#   LABEL_FIX_deployed_head/yaw_rate/repaired/pooled/r2.)*
# HONESTY CONDITION, which must travel with those numbers: the repair fixes the
# TAIL and the mean-square summary statistic, NOT typical accuracy.  On
# comma2k19 alone MAE falls 42.5 %, but medAE moves only -1.1 % and nMedAE gets
# 8.0 % WORSE, with Spearman rho flat (+0.001).
# See `…/incoming/2026-07-27-idm-v3/IDM_V3.md` §4 and the inventory of every
# affected published number in `TanitAD Research Hub/Benchmarks & Eval/
# Implementation/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.
#
# PUBLISHED precedent: comma.ai's own `calib_challenge` **discards every frame
# below 4 m/s** for exactly this reason. Our measured threshold is far less
# aggressive because it is derived from the data rather than assumed.
#
# ⚠️ THE DEFAULT WAS FLIPPED ON 2026-07-27 (`heading-default` pass). It shipped
# opt-in on 2026-07-27 so that no existing cache would move — a sound decision
# whose CONSEQUENCE was that **the defect became the default for all future
# work**: every new comma build got the broken labels unless the caller opted
# in. `DEFAULT_HEADING_MODE` is now `HEADING_MODE_HOLD`.
#
# ⛔ AND `HEADING_MODE_LEGACY` IS NO LONGER REACHABLE SILENTLY. Flipping the
# default alone would have failed in the OTHER direction: a script whose job is
# to reproduce a committed cache would have kept running, silently, on REPAIRED
# labels — and the two results differ ONLY by label protocol, so nothing on the
# page distinguishes them. That is the program's most-logged failure class (C29
# itself; the `heldout`/`full_set` blast radius; the `observed_frac`
# correction). So legacy now needs `allow_legacy=True` AND a written `reason`,
# the `models.vision_rank.resolve_vision_rank` discipline: a boolean can be
# flipped absent-mindedly, a sentence cannot.
#
# ⛔ AND NO EXISTING CACHE CHANGES MEANING. `label_params()` contributes NO key
# for the legacy regime, so every comma cache dir minted before today keeps its
# exact name and contents; the repaired regime contributes
# `{"heading_mode": "enu_velocity_hold_v1"}`, so a repaired build is
# STRUCTURALLY unable to land in a legacy-keyed dir (and vice versa). Same
# construction as `physicalai.label_params`, opposite default — which is
# admissible here and NOT there because comma2k19 is an explicitly NON-PARITY
# corpus (`data.parity`: "unregistered corpus (comma / cosmos / OOD)"). The
# canonical train corpus `physicalai-train-e438721ae894` is untouched by this.
HEADING_MODE_LEGACY = "enu_velocity"        # arctan2 of ENU velocity, as shipped
HEADING_MODE_HOLD = "enu_velocity_hold_v1"  # + hold last observable direction
HEADING_MODES = (HEADING_MODE_LEGACY, HEADING_MODE_HOLD)
DEFAULT_HEADING_MODE = HEADING_MODE_HOLD    # ⭐ FLIPPED 2026-07-27
HEADING_OBSERVABLE_V_MPS = 0.5              # MEASURED: 26.27% impossible below,
#                                             0.000% above (see the table above)

#: The one legitimate reason to ask for the broken labels, shipped so a
#: reproduction does not have to invent wording (cf. `vision_rank`'s
#: `LEGACY_RAW_REASON`). Passing this is still an EXPLICIT act at the call site.
LEGACY_HEADING_REASON = (
    "reproducing a comma2k19 cache/number COMMITTED BEFORE 2026-07-27, which "
    "was measured against the undefined-at-standstill heading. This is a "
    "REPRODUCTION of a published arm, not a label choice for new work.")


class LegacyHeadingRefused(ValueError):
    """The broken (undefined-at-standstill) heading was requested without an
    explicit, written acknowledgement."""


def resolve_heading_mode(heading_mode: str | None = None, *,
                         allow_legacy: bool = False, reason: str = "",
                         _what: str = "heading_mode") -> str:
    """Validate a requested heading mode and return it, or raise.

    ``None`` (a missing config key, an unset default, a dict ``.get`` that found
    nothing) resolves to :data:`DEFAULT_HEADING_MODE` — the REPAIR — so every
    accident mode lands on the correct label. Only
    :data:`HEADING_MODE_LEGACY`, requested by name, can produce the broken one,
    and only with ``allow_legacy=True`` AND a non-empty written ``reason``.
    """
    if heading_mode is None:
        return DEFAULT_HEADING_MODE
    if heading_mode not in HEADING_MODES:
        raise ValueError(f"unknown {_what} {heading_mode!r}; "
                         f"expected one of {HEADING_MODES}")
    if heading_mode == HEADING_MODE_LEGACY:
        if allow_legacy and reason.strip():
            return HEADING_MODE_LEGACY
        raise LegacyHeadingRefused(
            f"{_what}={HEADING_MODE_LEGACY!r} requests the LEGACY comma2k19 "
            f"heading, which is arctan2 of the ENU velocity and is UNDEFINED "
            f"AT STANDSTILL. MEASURED on the 64-segment val build: 26.27 % of "
            f"frames below {HEADING_OBSERVABLE_V_MPS} m/s carry a physically "
            f"impossible |yaw_rate| (up to 15.53 rad/s = 890 deg/s), vs "
            f"0.000 % in every bin above it; PhysicalAI has ZERO in every bin. "
            f"Cost: the DEPLOYED IDM head read comma yaw R2 +0.0114 against "
            f"these labels and +0.3308 against the repaired ones with NOTHING "
            f"retrained. The default is now {DEFAULT_HEADING_MODE!r}. If you "
            f"really are reproducing a pre-2026-07-27 cache, pass "
            f"allow_legacy=True AND a written reason (comma2k19."
            f"LEGACY_HEADING_REASON is provided) — this cannot be reached by a "
            f"default, a missing config key or a None.")
    return heading_mode


def label_params(heading_mode: str = DEFAULT_HEADING_MODE) -> dict:
    """Build-param fragment that separates the two label regimes by cache key.

    ⚠️ The LEGACY regime returns an EMPTY dict on purpose: every comma cache dir
    minted before 2026-07-27 must keep its exact key, or a cache silently stops
    meaning what it meant. Only the repaired regime adds a key, which is what
    makes a repaired build unable to collide with a legacy one. Never make this
    unconditional, and never invert which side is empty.

    This function does NOT refuse legacy — locating or re-keying an existing
    legacy cache is legitimate. The refusal lives on the label-PRODUCING path
    (:func:`resolve_heading_mode`, and therefore :func:`actions_and_poses`,
    :func:`build_episode` and :func:`cache_build_params`).
    """
    if heading_mode not in HEADING_MODES:
        raise ValueError(f"unknown heading_mode {heading_mode!r}; "
                         f"expected one of {HEADING_MODES}")
    if heading_mode == HEADING_MODE_LEGACY:
        return {}
    return {"heading_mode": heading_mode}


def cache_build_params(base: dict, heading_mode: str | None = None, *,
                       allow_legacy: bool = False, reason: str = "") -> dict:
    """``base`` + the label fragment, with the mode RESOLVED (and so refused).

    Cache-building callers must go through this rather than splatting
    :func:`label_params` themselves: forgetting the fragment is the failure that
    makes a repaired build overwrite — or silently load — a legacy-keyed dir,
    which is the one outcome this whole regime exists to prevent.
    """
    mode = resolve_heading_mode(heading_mode, allow_legacy=allow_legacy,
                                reason=reason, _what="comma heading_mode")
    return {**base, **label_params(mode)}


def hold_heading_through_standstill(yaw: np.ndarray, v: np.ndarray,
                                    v_min: float = HEADING_OBSERVABLE_V_MPS
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Replace the heading wherever the ENU velocity is too small to define one,
    by holding the last OBSERVABLE direction (back-filling the leading run).

    Operates on the unit direction vector, not the angle, so no 2*pi wrap can be
    introduced. Returns ``(yaw_fixed, observable_mask)``; callers that need a
    strict admissibility mask should keep the second return value.

    A road vehicle at v ~ 0 has yaw_rate ~ 0 — it cannot rotate in place — so
    holding the heading is the physically correct completion, not a convenience.

    🔴 **THE REPAIR IS A NO-OP ON A WHOLLY-STATIONARY SEGMENT**, and that is
    deliberate — with no observable frame there is no direction to hold, and
    inventing one would be worse. **The consequence is that such a segment has NO
    ADMISSIBLE HEADING ANYWHERE and must be handled by the CALLER**, which is why
    ``observable`` is returned. MEASURED 2026-07-27 (`…/2026-07-27-heading-
    default/raw/idm_head_v1_comma_rescore.json`): one such clip
    (300 frames, **zero** observable, v_max **0.039 m/s**) contributed **84
    physically impossible yaw labels up to 15.28 rad/s** — 2.0 % of a 4,140-window
    val split — and **held the deployed IDM head's comma yaw R² at ~0 even with
    the repair ON**. Raising ``v_min`` cannot help: a stationary clip is
    stationary at every threshold. ⛔ **A repair and an ADMISSIBILITY decision are
    different things.** Requiring ``observable`` at ``t-1/t/t+1`` removed every
    impossible label and collapsed the label's own std 0.938 -> 0.046 rad/s.
    ⚠️ **No caller in this repo consumes the mask yet — see HEADING_DEFAULT.md §7
    escalation 1.**
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    obs = v >= v_min
    if not obs.any():                     # a wholly-stationary segment
        return yaw.copy(), obs
    idx = np.where(obs, np.arange(yaw.size), -1)
    np.maximum.accumulate(idx, out=idx)                     # forward-fill
    idx[idx < 0] = int(np.argmax(obs))                      # back-fill the head
    return np.arctan2(np.sin(yaw)[idx], np.cos(yaw)[idx]), obs

# I7 task-identity fingerprint (D-017): probes fit on this corpus may only be
# consumed by streams with an IDENTICAL fingerprint (i7_task_identity check).
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}


# --------------------------------------------------------------------------- #
# Discovery & splits                                                           #
# --------------------------------------------------------------------------- #
def discover_segments(root: str | Path) -> list[Path]:
    """All segment dirs under root that carry video + pose logs."""
    root = Path(root)
    segs = []
    for video in sorted(root.rglob("video.hevc")):
        seg = video.parent
        if (seg / "global_pose" / "frame_times").exists():
            segs.append(seg)
    return segs


def route_of(segment: Path) -> str:
    """Route id = the <dongle|date--time> folder name above the segment."""
    return segment.parent.name


def sample_segments_across_routes(segments: list[Path], n: int,
                                  seed: int = 0) -> list[Path]:
    """Cap to n segments while spanning as many ROUTES as possible (round-robin).

    Plain `segments[:n]` returns n minutes of the SAME drive (segments of one
    route sort together), which a route-level split then puts wholly into one
    side — the p0-sB00 '0 train segments' failure. Round-robin over routes
    keeps small caps split-able and diverse.
    """
    by_route: dict[str, list[Path]] = {}
    for s in segments:
        by_route.setdefault(route_of(s), []).append(s)
    order = sorted(by_route)
    g = torch.Generator().manual_seed(seed)
    order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
    out: list[Path] = []
    depth = 0
    while len(out) < min(n, len(segments)):
        added = False
        for r in order:
            if depth < len(by_route[r]):
                out.append(by_route[r][depth])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        depth += 1
    return out


def split_by_route(segments: list[Path], val_frac: float = 0.2,
                   seed: int = 0) -> tuple[list[Path], list[Path]]:
    """I3: disjoint ROUTES, not frames, not segments."""
    routes = sorted({route_of(s) for s in segments})
    assert len(routes) >= 2, (
        f"route-level split needs >= 2 routes, got {len(routes)} "
        f"({routes}) — cap segments with sample_segments_across_routes(), "
        "not list slicing")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(routes), generator=g).tolist()
    n_val = max(1, int(len(routes) * val_frac))
    val_routes = {routes[i] for i in perm[:n_val]}
    train = [s for s in segments if route_of(s) not in val_routes]
    val = [s for s in segments if route_of(s) in val_routes]
    assert not {route_of(s) for s in train} & {route_of(s) for s in val}
    return train, val


def episode_id_of(segment: Path) -> int:
    key = f"{route_of(segment)}/{segment.name}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Logs -> actions & poses (pure numpy, unit-tested without real data)          #
# --------------------------------------------------------------------------- #
def _load_tv(seg: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = seg / "processed_log" / "CAN" / name
    return np.load(d / "t"), np.load(d / "value").astype(np.float64).squeeze()


def ecef_to_enu(positions: np.ndarray, velocities: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Segment-local ENU (east, north) positions & velocities.

    Reference = first frame. Geocentric-latitude approximation — fine for a
    1-minute local frame; the derived (x, y) are relative displacements.
    """
    ref = positions[0]
    x, y, z = ref
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    r = np.array([[-so, co, 0.0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    enu_p = (positions - ref) @ r.T
    enu_v = velocities @ r.T
    return enu_p[:, :2], enu_v[:, :2]


def actions_and_poses(frame_times: np.ndarray, positions: np.ndarray,
                      velocities: np.ndarray, can_speed: tuple[np.ndarray, np.ndarray],
                      can_steer_deg: tuple[np.ndarray, np.ndarray],
                      stride: int,
                      heading_mode: str | None = None,
                      *, allow_legacy_heading: bool = False,
                      legacy_heading_reason: str = ""
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Contract actions [T,2] and poses [T,4] at frame_times[::stride].

    ``heading_mode`` defaults to the REPAIR (:data:`DEFAULT_HEADING_MODE`,
    flipped 2026-07-27). To reproduce a pre-2026-07-27 cache pass
    ``heading_mode=HEADING_MODE_LEGACY`` together with
    ``allow_legacy_heading=True`` and a written ``legacy_heading_reason``; that
    path is bit-identical to the old default and is pinned by
    ``tests/test_comma_heading_regime.py``.
    """
    heading_mode = resolve_heading_mode(
        heading_mode, allow_legacy=allow_legacy_heading,
        reason=legacy_heading_reason)
    t = frame_times[::stride]
    speed = np.interp(t, *can_speed)
    steer_wheel_deg = np.interp(t, *can_steer_deg)
    steer_road_rad = steer_wheel_deg * WHEEL_TO_RAD / STEER_RATIO
    accel = np.gradient(speed, t, edge_order=1)
    actions = np.column_stack([steer_road_rad, accel]).astype(np.float32)

    enu_p, enu_v = ecef_to_enu(positions[::stride], velocities[::stride])
    yaw = np.arctan2(enu_v[:, 1], enu_v[:, 0])                 # heading in ENU
    v = np.linalg.norm(enu_v, axis=1)
    if heading_mode == HEADING_MODE_HOLD:
        yaw, _ = hold_heading_through_standstill(yaw, v)
    poses = np.column_stack([enu_p[:, 0], enu_p[:, 1], yaw, v]).astype(np.float32)
    return actions, poses


# --------------------------------------------------------------------------- #
# Video decode (lazy av import) and frame preprocessing                        #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Geometry regimes (wide-FOV enablement, 2026-07-27)                           #
# --------------------------------------------------------------------------- #
# ⚠️ THE HARD FACT THIS CORPUS IMPOSES. comma2k19's ENTIRE horizontal field is
# ``2*atan(582/910) = 65.2027 deg`` (calib.COMMA2K19_MAX_HFOV_DEG). It physically
# CANNOT supply the 100-120 deg the PI asked for, at any resolution or focal.
# Three policies are therefore live, and ALL THREE are expressible here — which
# one the flagship mix uses is the PI's decision, not this module's:
#
#   (a) PER-CORPUS geometry  — comma keeps its own (narrower) frame while
#       PhysicalAI runs wide. `build_episode(..., frame=<comma frame>)`.
#   (b) LETTERBOX            — comma is rendered ON the wide frame with the
#       unobservable periphery EXPLICITLY masked, never silently zoomed:
#       `geometry_mode="rectify"` (calib.pinhole_rectify, which reports
#       `last_observed_frac`). At a 100-deg frame the honest observed fraction
#       is well under 1 and the model sees a black band, which is a real
#       modelling decision and must be visible, not accidental.
#   (c) DROP comma           — a trainer/mix decision, no code here.
#
# ``focal_crop`` is the DEPLOYED regime and the default; it is byte-identical to
# the pre-2026-07-27 path. ⛔ Under `focal_crop` a frame wider than the sensor
# does NOT widen the field — the crop clamps and the image ZOOMS. That is why
# (b) exists and why `focal_crop_resize.last_clamped` is recorded.
GEOMETRY_MODE_CROP = "focal_crop"
GEOMETRY_MODE_RECTIFY = "rectify"
GEOMETRY_MODES = (GEOMETRY_MODE_CROP, GEOMETRY_MODE_RECTIFY)
DEFAULT_GEOMETRY_MODE = GEOMETRY_MODE_CROP


def _decode_video(seg: Path, stride: int, size: int,
                  max_frames: int | None, frame=None,
                  geometry_mode: str = DEFAULT_GEOMETRY_MODE) -> torch.Tensor:
    """video.hevc -> uint8 [T, 3, H, W], every stride-th frame, canonicalized.

    ``frame`` / ``geometry_mode`` default to the deployed square focal-crop, so
    this is byte-identical to its pre-2026-07-27 behaviour for every caller.
    """
    import av                                                   # lazy: .[real]
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"unknown geometry_mode {geometry_mode!r}; "
                         f"expected one of {GEOMETRY_MODES}")
    out = []
    with av.open(str(seg / "video.hevc")) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, f in enumerate(container.decode(stream)):
            if i % stride:
                continue
            rgb = torch.from_numpy(f.to_ndarray(format="rgb24"))  # H W 3
            out.append(rgb.permute(2, 0, 1))
            if max_frames is not None and len(out) >= max_frames:
                break
    vid = torch.stack(out)                                      # T 3 H W uint8
    # D-016: canonical effective focal across corpora (comma is the reference
    # camera — its crop is ~the full frame height, matching prior behavior).
    from tanitad.data.calib import (COMMA2K19_FOCAL_PX, COMMA2K19_INTR,
                                    focal_crop_resize, pinhole_rectify)
    if geometry_mode == GEOMETRY_MODE_RECTIFY:
        return pinhole_rectify(vid, COMMA2K19_INTR, frame=frame)
    return focal_crop_resize(vid, COMMA2K19_FOCAL_PX, size, frame=frame)


def stack_frames(vid_u8: torch.Tensor, n_stack: int = 3) -> torch.Tensor:
    """[T,3,S,S] uint8 -> [T-(n-1), 3n, S, S]: frames t-(n-1)..t channel-stacked.

    D-015: n_stack=3 at 10 Hz -> the encoder sees [t-200ms, t-100ms, t] in one
    9-channel input, making acceleration/curvature observable per input.
    Oldest frame first, current frame in the LAST 3 channels.
    """
    parts = [vid_u8[i:vid_u8.shape[0] - (n_stack - 1) + i] for i in range(n_stack)]
    return torch.cat(parts, dim=1)


def stack_two_frames(vid_u8: torch.Tensor) -> torch.Tensor:
    """Legacy 2-frame stack (pre-D-015). Kept for tests/compat."""
    return stack_frames(vid_u8, n_stack=2)


def build_episode(segment: Path, size: int = 256, stride: int = 2,
                  max_steps: int | None = 300, n_stack: int = 3,
                  decode_fn=_decode_video,
                  heading_mode: str | None = None,
                  frame=None,
                  geometry_mode: str = DEFAULT_GEOMETRY_MODE,
                  *, allow_legacy_heading: bool = False,
                  legacy_heading_reason: str = "") -> ToyEpisode:
    """One comma2k19 segment -> contract episode at FPS/stride Hz.

    D-015: n_stack consecutive strided frames (100 ms apart at stride 2) are
    channel-stacked per step -> [T, 3*n_stack, S, S]; actions/poses aligned to
    the LATEST frame of each stack. max_steps caps memory. decode_fn is
    injectable for tests (no real video needed in CI).

    ``heading_mode`` defaults to the REPAIR — see the module header. Legacy
    needs ``allow_legacy_heading=True`` + ``legacy_heading_reason``.
    """
    segment = Path(segment)
    ft = np.load(segment / "global_pose" / "frame_times")
    pos = np.load(segment / "global_pose" / "frame_positions")
    vel = np.load(segment / "global_pose" / "frame_velocities")
    n_avail = (len(ft) + stride - 1) // stride
    n = n_avail if max_steps is None else min(max_steps + n_stack - 1, n_avail)

    actions, poses = actions_and_poses(
        ft, pos, vel, _load_tv(segment, "speed"),
        _load_tv(segment, "steering_angle"), stride,
        heading_mode=heading_mode,
        allow_legacy_heading=allow_legacy_heading,
        legacy_heading_reason=legacy_heading_reason)
    if frame is None and geometry_mode == DEFAULT_GEOMETRY_MODE:
        vid = decode_fn(segment, stride, size, n)               # [n,3,S,S] u8
    else:                                                       # [n,3,H,W] u8
        vid = decode_fn(segment, stride, size, n, frame=frame,
                        geometry_mode=geometry_mode)
    n = min(n, vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:n], n_stack)                    # [n-k+1,3k,S,S]
    k = n_stack - 1
    # Frames stay uint8 in memory (4x smaller — 584 float32 episodes would
    # need ~400 GB, found on the first pod run); datasets convert per window.
    return ToyEpisode(
        frames=stacked,
        actions=torch.from_numpy(actions[k:n]),
        poses=torch.from_numpy(poses[k:n]),
        episode_id=episode_id_of(segment),
    )


# --------------------------------------------------------------------------- #
# Dataset — window contract identical to the toy/MetaDrive datasets            #
# --------------------------------------------------------------------------- #
class Comma2k19Dataset(torch.utils.data.Dataset):
    """Windows over comma2k19 segments. Build train/val from split_by_route
    (I3) — constructor takes an explicit segment list on purpose."""

    def __init__(self, segments: list[Path], window: int = 8,
                 max_horizon: int = 16, size: int = 256, stride: int = 2,
                 max_steps: int | None = 300, decode_fn=_decode_video,
                 heading_mode: str | None = None, *,
                 allow_legacy_heading: bool = False,
                 legacy_heading_reason: str = ""):
        self.window, self.max_horizon = window, max_horizon
        # resolve ONCE, before any decoding: a refused legacy request must fail
        # in the first millisecond, not after 40 minutes of video decode.
        self.heading_mode = resolve_heading_mode(
            heading_mode, allow_legacy=allow_legacy_heading,
            reason=legacy_heading_reason)
        # carried, not re-invented: the per-episode call re-checks against the
        # SAME acknowledgement the constructor was given.
        self._legacy_ack = (allow_legacy_heading, legacy_heading_reason)
        self.episodes: list[ToyEpisode] = []
        for i, seg in enumerate(segments):
            if i % 20 == 0:
                print(f"[comma2k19] building episodes {i}/{len(segments)} "
                      f"(video decode — this is the slow part)", flush=True)
            try:
                self.episodes.append(build_episode(
                    seg, size=size, stride=stride, max_steps=max_steps,
                    decode_fn=decode_fn, heading_mode=self.heading_mode,
                    allow_legacy_heading=self._legacy_ack[0],
                    legacy_heading_reason=self._legacy_ack[1]))
            except Exception as e:                # corrupt segment: skip, log
                print(f"[comma2k19] skipping {seg}: {type(e).__name__}: {e}")
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(self.episodes):
            t_max = ep.frames.shape[0] - window - max_horizon
            self.index.extend((e_i, t) for t in range(max(0, t_max)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        from tanitad.data._contract import to_float_frames
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        return {
            "frames": to_float_frames(ep.frames[t:t + w]),
            "actions": ep.actions[t:t + w],
            "future_frames": to_float_frames(
                ep.frames[t + w:t + w + self.max_horizon]),
            "future_poses": ep.poses[t + w:t + w + self.max_horizon],
            "pose_last": ep.poses[t + w - 1],
            "episode_id": ep.episode_id,
        }
