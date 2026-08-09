"""trajrecon -> TanitAD episode contract.

Turns a reconstructed smartphone-dashcam session into the same
:class:`~tanitad.data.toy_driving.ToyEpisode` every other adapter emits, so a
model trained on comma2k19 or PhysicalAI consumes these episodes with no code
change::

    frames  [T, 3n, S, S]  n consecutive RGB frames channel-stacked, uint8
    actions [T, 2]         (road-wheel steer rad, longitudinal accel m/s^2)
    poses   [T, 4]         (x_east, y_north, yaw, v) in a SESSION-local ENU frame
    episode_id             stable int hash of the session name

Why this adapter exists at all: the reconstruction produces real trajectory
targets and real actions from a phone on a windscreen, with no annotation and no
instrumented vehicle. That is the same shape of signal comma2k19 gives us, from
hardware we already own.

Frame convention
----------------
``poses`` here are SESSION-local ENU (x east, y north), matching
:mod:`tanitad.data.comma2k19` -- NOT the per-frame FLU ego slices that
:func:`~.trajectory.ego_trajectory` returns. Both exist and they are not
interchangeable: the contract wants one continuous pose track per episode, while
``ego_trajectory`` re-origins at every frame for the planner's target. Use
:func:`ego_windows` when you want the latter.

⚠️ STEERING IS UNOBSERVABLE AT REST -- read before lowering the threshold
--------------------------------------------------------------------------
``steering.estimate_steering`` derives curvature as ``omega / v``, so as ``v``
approaches zero the steer estimate diverges and is meaningless. It reports this
in ``SteeringResult.valid`` (False below ``v_min``, default 1.5 m/s).

This adapter REFUSES to fabricate a steer value there. :func:`steer_channel`
returns the mask, and :func:`assert_steer_admissible` raises rather than let a
caller silently train on invented zeros.

This is not hypothetical caution. The registry documents the identical defect
class on comma2k19, where heading came from the ENU velocity vector and is
undefined at rest: **26.27 %** of frames in the ``v < 0.5 m/s`` bin carried
physically impossible yaw rates (up to 15.53 rad/s at 0.00-0.01 m/s), while every
bin above 0.5 m/s was clean at **0.000 %**. It read as a model failure for
months -- pooled ``yaw_rate`` R2 was **0.1046** against those labels and
**0.8108** against repaired ones with NOTHING retrained. The channel was never
broken; the labels were. Do not re-import that bug through this door.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_V_MIN_MPS = 1.5          # steering.estimate_steering's own default
DEG2RAD = np.pi / 180.0


class InadmissibleSteerLabel(ValueError):
    """Raised when too much of an episode has unobservable steering."""


def episode_id_of(name: str) -> int:
    """Stable int id from a session name (same construction as comma2k19)."""
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def session_poses(traj, times: np.ndarray) -> np.ndarray:
    """Sample a :class:`~.trajectory.TrajectoryResult` at ``times`` -> ``[T, 4]``.

    Columns are ``(x_east, y_north, yaw, v)`` in metres / radians / m/s, with the
    origin at the FIRST sampled instant so an episode is translation-invariant.
    ``yaw`` is unwrapped before interpolation -- interpolating raw heading across
    the +/-pi branch cut would manufacture a spurious ~2*pi step and, with it, an
    enormous fake yaw rate.
    """
    from tanitad.data.trajrecon.trajectory import IE, IN, IPSI, IV

    times = np.asarray(times, dtype=float)
    t = traj.t
    if times.size == 0:
        raise ValueError("times is empty")
    if times.min() < t[0] - 1e-9 or times.max() > t[-1] + 1e-9:
        raise ValueError(
            f"times [{times.min():.3f}, {times.max():.3f}] fall outside the "
            f"reconstructed span [{t[0]:.3f}, {t[-1]:.3f}] -- refusing to extrapolate"
        )

    e = np.interp(times, t, traj.x[:, IE])
    n = np.interp(times, t, traj.x[:, IN])
    v = np.interp(times, t, traj.x[:, IV])
    yaw = np.interp(times, t, np.unwrap(traj.x[:, IPSI]))

    return np.column_stack([e - e[0], n - n[0], yaw, v]).astype(np.float32)


def steer_channel(traj, times: np.ndarray, vehicle: dict | None = None,
                  v_min: float = DEFAULT_V_MIN_MPS,
                  max_wheel_rate_deg_s: float | None = 180.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Road-wheel steer angle in radians at ``times``, plus an admissibility mask.

    Returns ``(steer_rad [T], valid [T] bool)``. ``steer_rad`` is the ROAD-WHEEL
    angle, not the steering-wheel angle -- the contract's action channel, and the
    same convention :mod:`tanitad.data.comma2k19` uses (it divides its CAN wheel
    angle by a steering ratio to get here).

    Where ``valid`` is False the returned value is 0.0, which is a PLACEHOLDER
    and not a measurement. Callers must consult the mask; see
    :func:`assert_steer_admissible`.
    """
    from tanitad.data.trajrecon.steering import estimate_steering

    times = np.asarray(times, dtype=float)
    res = estimate_steering(traj, vehicle=vehicle, v_min=v_min,
                            max_wheel_rate_deg_s=max_wheel_rate_deg_s)

    steer_rad = np.interp(times, res.t, res.road_wheel_deg) * DEG2RAD
    # Nearest-neighbour on the mask: interpolating a boolean would invent
    # half-valid samples at the standstill boundary.
    idx = np.searchsorted(res.t, times).clip(0, len(res.t) - 1)
    valid = np.asarray(res.valid, dtype=bool)[idx]

    steer_rad = np.where(valid, steer_rad, 0.0)
    return steer_rad.astype(np.float32), valid


def assert_steer_admissible(valid: np.ndarray, max_invalid_frac: float = 0.10,
                            *, context: str = "") -> None:
    """Raise if too many frames have unobservable steering.

    A few standstill frames are normal; an episode that is mostly stopped carries
    almost no steering information and would train a model on placeholder zeros.
    Failing loudly forces the caller to drop the episode or widen the threshold
    ON PURPOSE, rather than discovering it as a bias months later.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0:
        raise InadmissibleSteerLabel(f"empty steer mask{' for ' + context if context else ''}")
    frac = 1.0 - float(valid.mean())
    if frac > max_invalid_frac:
        raise InadmissibleSteerLabel(
            f"{frac:.1%} of frames{' in ' + context if context else ''} have "
            f"unobservable steering (v < v_min); limit is {max_invalid_frac:.1%}. "
            f"These carry placeholder zeros, NOT measurements -- drop the episode "
            f"or raise max_invalid_frac deliberately."
        )


def build_episode(frames_u8, traj, times: np.ndarray, session_name: str, *,
                  dt: float, vehicle: dict | None = None,
                  n_stack: int = 3, v_min: float = DEFAULT_V_MIN_MPS,
                  max_invalid_frac: float = 0.10):
    """Assemble a contract-compliant episode from a reconstructed session.

    ``frames_u8`` is ``[T, 3, S, S]`` uint8, one entry per element of ``times``.
    Frames are channel-stacked ``n_stack`` deep (D-015), which consumes the first
    ``n_stack - 1`` samples, so poses and actions are trimmed to match.
    """
    import torch

    from tanitad.data._contract import assert_contract, finite_diff_accel
    from tanitad.data.comma2k19 import stack_frames
    from tanitad.data.toy_driving import ToyEpisode

    times = np.asarray(times, dtype=float)
    if frames_u8.shape[0] != times.shape[0]:
        raise ValueError(f"frames {frames_u8.shape[0]} != times {times.shape[0]}")

    poses = session_poses(traj, times)
    steer_rad, valid = steer_channel(traj, times, vehicle=vehicle, v_min=v_min)

    stacked = stack_frames(frames_u8, n_stack=n_stack)      # drops the first n-1
    cut = n_stack - 1
    poses, steer_rad, valid = poses[cut:], steer_rad[cut:], valid[cut:]

    assert_steer_admissible(valid, max_invalid_frac, context=session_name)

    accel = finite_diff_accel(poses[:, 3], dt)
    actions = np.column_stack([steer_rad, accel]).astype(np.float32)

    ep = ToyEpisode(
        frames=stacked,
        actions=torch.from_numpy(actions),
        poses=torch.from_numpy(poses),
        episode_id=episode_id_of(session_name),
    )
    assert_contract(ep, channels=3 * n_stack)
    return ep


def ego_windows(traj, times: np.ndarray, *, t_past: float = 3.0,
                t_future: float = 5.0, dt_out: float = 0.1) -> list[dict | None]:
    """Per-frame FLU ego slices -- the planner's target, not the pose track.

    Thin pass-through to :func:`~.trajectory.ego_trajectory`, kept here so callers
    do not have to decide between the two pose conventions unaided. Entries are
    ``None`` where the window falls outside the reconstructed span, and carry
    ``standstill=True`` where the future horizon was shortened.
    """
    from tanitad.data.trajrecon.trajectory import ego_trajectory

    return [ego_trajectory(traj, float(t), t_past=t_past, t_future=t_future,
                           dt_out=dt_out) for t in np.asarray(times, dtype=float)]
