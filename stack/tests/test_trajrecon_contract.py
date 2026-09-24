"""trajrecon -> episode contract: shapes, frames, and the standstill refusal.

These run without any recording: the trajectory is synthesised analytically, so
the adapter is testable on a machine with no Sensor Logger data, no video, and no
ffmpeg. That matters -- the 2026-08-08 recording could not be fetched into this
environment at all, and an adapter that can only be tested with data in hand is
an adapter that goes untested.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tanitad.data.trajrecon.contract import (  # noqa: E402
    InadmissibleSteerLabel, assert_steer_admissible, build_episode,
    episode_id_of, session_poses, steer_channel,
)
from tanitad.data.trajrecon.trajectory import TrajectoryResult  # noqa: E402


def make_traj(n: int = 400, dt: float = 0.05, speed: float = 12.0,
              yaw_rate: float = 0.08) -> TrajectoryResult:
    """A constant-speed, constant-yaw-rate arc -- a clean circular turn.

    Constant curvature means kappa = yaw_rate / speed is exactly known, so the
    steer channel has an analytic expectation rather than a regression baseline.
    """
    from tanitad.data.trajrecon.geo import LocalENU

    t = np.arange(n) * dt
    psi = yaw_rate * t
    e = np.cumsum(speed * np.cos(psi) * dt)
    nn = np.cumsum(speed * np.sin(psi) * dt)

    x = np.zeros((n, 6))
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = e, nn, speed, psi
    return TrajectoryResult(t=t, x=x, P=np.zeros((n, 6, 6)),
                            enu=LocalENU(48.8566, 2.3522, 0.0),
                            x_filt=x.copy(), meta={})


def make_traj_stopped(n: int = 200, dt: float = 0.05) -> TrajectoryResult:
    """A stationary vehicle: speed 0, so steering is unobservable throughout."""
    from tanitad.data.trajrecon.geo import LocalENU

    t = np.arange(n) * dt
    x = np.zeros((n, 6))
    return TrajectoryResult(t=t, x=x, P=np.zeros((n, 6, 6)),
                            enu=LocalENU(48.8566, 2.3522, 0.0),
                            x_filt=x.copy(), meta={})


# --------------------------------------------------------------------------- #
# poses                                                                        #
# --------------------------------------------------------------------------- #

def test_session_poses_shape_origin_and_speed() -> None:
    traj = make_traj()
    times = np.arange(0, 10.0, 0.1)
    poses = session_poses(traj, times)

    assert poses.shape == (times.size, 4)
    assert poses.dtype == np.float32
    assert abs(poses[0, 0]) < 1e-5 and abs(poses[0, 1]) < 1e-5, "origin must be the first sample"
    assert np.allclose(poses[:, 3], 12.0, atol=1e-3), "constant-speed arc"


def test_session_poses_refuses_to_extrapolate() -> None:
    """Silently extrapolating past the reconstructed span would invent trajectory."""
    traj = make_traj()
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        session_poses(traj, np.array([traj.t[-1] + 5.0]))


def test_session_poses_unwraps_yaw_across_the_branch_cut() -> None:
    """Interpolating raw heading across +/-pi manufactures a fake ~2pi step.

    With yaw_rate 0.08 rad/s over 400*0.05 s the arc sweeps ~1.6 rad, so force a
    long run that crosses pi and assert the sampled yaw stays monotonic.
    """
    traj = make_traj(n=3000, dt=0.05, yaw_rate=0.5)     # sweeps ~75 rad
    times = np.arange(0, 140.0, 0.5)
    yaw = session_poses(traj, times)[:, 2]

    steps = np.diff(yaw)
    assert (steps > 0).all(), "unwrapped yaw must stay monotonic on a constant turn"
    assert steps.max() < 1.0, f"a ~2pi jump leaked through: max step {steps.max():.3f}"


# --------------------------------------------------------------------------- #
# steering                                                                     #
# --------------------------------------------------------------------------- #

def test_steer_channel_matches_the_bicycle_model_on_a_known_arc() -> None:
    """kappa = omega/v is exact here, so delta ~ atan(L * kappa) is checkable."""
    from tanitad.data.trajrecon.steering import AUDI_A6_ETRON

    speed, yaw_rate = 12.0, 0.08
    traj = make_traj(n=600, dt=0.05, speed=speed, yaw_rate=yaw_rate)
    times = np.arange(2.0, 25.0, 0.1)

    steer_rad, valid = steer_channel(traj, times)

    assert valid.all(), "12 m/s is far above v_min; nothing should be masked"
    kappa = yaw_rate / speed
    expected = np.arctan(AUDI_A6_ETRON["wheelbase_m"] * kappa)      # understeer adds a little
    med = float(np.median(steer_rad))
    assert med > 0, "a positive yaw rate is a left turn -> positive road-wheel angle"
    assert expected * 0.8 < med < expected * 1.6, (
        f"median steer {med:.5f} rad is not near the bicycle-model {expected:.5f}"
    )


def test_steer_is_masked_and_zeroed_at_standstill() -> None:
    """The core refusal: no fabricated steer where v ~ 0."""
    traj = make_traj_stopped()
    times = np.arange(1.0, 8.0, 0.1)

    steer_rad, valid = steer_channel(traj, times)

    assert not valid.any(), "a stationary vehicle has no observable steering"
    assert np.all(steer_rad == 0.0), "masked samples must be placeholder zeros"


def test_assert_steer_admissible_raises_on_a_mostly_stopped_episode() -> None:
    valid = np.array([True] * 50 + [False] * 50)
    with pytest.raises(InadmissibleSteerLabel, match="unobservable steering"):
        assert_steer_admissible(valid, max_invalid_frac=0.10, context="unit-test")

    assert_steer_admissible(np.array([True] * 96 + [False] * 4), max_invalid_frac=0.10)

    with pytest.raises(InadmissibleSteerLabel, match="empty"):
        assert_steer_admissible(np.array([], dtype=bool))


# --------------------------------------------------------------------------- #
# episode assembly                                                             #
# --------------------------------------------------------------------------- #

def test_build_episode_satisfies_the_contract() -> None:
    traj = make_traj(n=600, dt=0.05)
    dt = 0.1
    times = np.arange(2.0, 12.0, dt)
    T, S, n_stack = times.size, 32, 3
    frames = torch.randint(0, 255, (T, 3, S, S), dtype=torch.uint8)

    ep = build_episode(frames, traj, times, "sess-A", dt=dt, n_stack=n_stack)

    kept = T - (n_stack - 1)
    assert ep.frames.shape == (kept, 3 * n_stack, S, S)
    assert ep.frames.dtype == torch.uint8
    assert ep.actions.shape == (kept, 2)
    assert ep.poses.shape == (kept, 4)
    assert ep.episode_id == episode_id_of("sess-A")
    assert torch.isfinite(ep.actions).all() and torch.isfinite(ep.poses).all()


def test_build_episode_stacks_the_current_frame_last() -> None:
    """D-015 ordering: oldest first, current frame in the LAST 3 channels.

    Getting this backwards silently reverses time for the encoder, which is the
    kind of bug that shows up as a mediocre metric rather than a crash.
    """
    traj = make_traj(n=600, dt=0.05)
    dt = 0.1
    times = np.arange(2.0, 8.0, dt)
    T = times.size
    frames = torch.zeros((T, 3, 8, 8), dtype=torch.uint8)
    for i in range(T):
        frames[i] = i                                  # frame i is filled with value i

    ep = build_episode(frames, traj, times, "sess-B", dt=dt, n_stack=3)

    # Episode row 0 stacks source frames 0,1,2 -> last 3 channels are frame 2.
    assert int(ep.frames[0, 0, 0, 0]) == 0
    assert int(ep.frames[0, 3, 0, 0]) == 1
    assert int(ep.frames[0, 6, 0, 0]) == 2, "current frame must occupy the LAST 3 channels"


def test_build_episode_refuses_a_stopped_session() -> None:
    traj = make_traj_stopped(n=400)
    dt = 0.1
    times = np.arange(1.0, 15.0, dt)
    frames = torch.zeros((times.size, 3, 16, 16), dtype=torch.uint8)

    with pytest.raises(InadmissibleSteerLabel):
        build_episode(frames, traj, times, "sess-parked", dt=dt)


def test_build_episode_rejects_mismatched_frame_count() -> None:
    traj = make_traj()
    times = np.arange(2.0, 8.0, 0.1)
    frames = torch.zeros((times.size - 3, 3, 8, 8), dtype=torch.uint8)

    with pytest.raises(ValueError, match="!="):
        build_episode(frames, traj, times, "sess-C", dt=0.1)


def test_episode_id_is_stable_and_distinct() -> None:
    assert episode_id_of("a") == episode_id_of("a")
    assert episode_id_of("a") != episode_id_of("b")
    assert isinstance(episode_id_of("a"), int)
