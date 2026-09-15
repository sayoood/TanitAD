"""`estimate_foe`'s rotation gate must be fed by the gyro, not by image motion.

WHY THIS EXISTS
---------------
`estimate_foe` drops any frame pair whose rotation rate exceeds `max_rot_rate`
(0.035 rad/s = 2.01 deg/s), and upstream read those rates from
`timesync.image_angular_rate`. That function is a **timing** instrument. Its own
docstring (`timesync.py:236-239`) says the yaw amplitude is "only approximate"
because forward translation leaks into `tx`, and that "only the timing of the
signal is used". Thresholding its amplitudes absolutely is a misuse, and the leak
scales with SPEED.

MEASURED 2026-08-08 on the 14-19-54 recording (70-84 km/h), t=5..20 s, n=449:

    camera |yaw|   p50 5.42 deg/s   passes the 2.01 deg/s gate:  4.5%
    gyro   |x|     p50 1.46 deg/s   passes:                     64.6%
    ALL THREE camera axes pass on 0.4% of frames — 2 of 449.

Below `estimate_foe`'s own `len(idxs) < 10` floor, so it returned None, the mount
stayed nominal 0.0, and `lane_calib` then rejected a perfectly good -7.01 deg yaw
for disagreeing with a "FOE" that never existed — 4.9 m of lateral error at 40 m
in the rendered overlay.

`_gyro_rot_on_frame_pairs` supplies the same `(t_mid, omega, comps)` shape from
the gyro instead. These tests pin the three properties that make that swap safe;
each one, if broken, fails silently rather than loudly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# `camera.py` is loaded DIRECTLY from its path rather than as
# `tanitad.data.trajrecon.camera`, because the normal import goes through
# `tanitad/data/__init__.py`, which eagerly imports `toy_driving` -> `torch`
# (`toy_driving.py:21`). Nothing under test here needs torch, and skipping these
# checks wherever torch is absent would mean the gate that caused a 4.9 m error
# is only verified on machines that happen to have a deep-learning stack.
#
# Safe because camera.py's top level is `dataclasses` + `numpy` only; its two
# relative imports (`from .timesync import ...`, lines 195 and 417) sit INSIDE
# functions this file never calls.
_CAMERA_PY = (Path(__file__).resolve().parents[1]
              / "tanitad" / "data" / "trajrecon" / "camera.py")
_spec = importlib.util.spec_from_file_location("_trajrecon_camera_under_test", _CAMERA_PY)
camera = importlib.util.module_from_spec(_spec)
# Register before exec: `@dataclass` resolves annotations via
# `sys.modules[cls.__module__].__dict__`, which is None for an unregistered module.
sys.modules[_spec.name] = camera
_spec.loader.exec_module(camera)


# --------------------------------------------------------------------------
# minimal fakes — numpy only, no pandas
# --------------------------------------------------------------------------

class _Col:
    def __init__(self, a): self._a = np.asarray(a, dtype=float)
    def to_numpy(self, dtype=float): return self._a.astype(dtype)


class _Frame:
    """Just enough of a DataFrame for `gy["seconds_elapsed"]` / `gy[["x","y","z"]]`."""

    def __init__(self, t, w):
        self.t = np.asarray(t, dtype=float)
        self.w = np.asarray(w, dtype=float)

    def __getitem__(self, key):
        if key == "seconds_elapsed":
            return _Col(self.t)
        if list(key) == ["x", "y", "z"]:
            return _Col(self.w)
        raise KeyError(key)


class _Session:
    def __init__(self, frame): self._f = frame
    def has(self, name): return name == "gyro" and self._f is not None
    def __getitem__(self, name): return self._f


class _Video:
    def __init__(self, pts): self.pts = np.asarray(pts, dtype=float)


class _Sync:
    def __init__(self, t0): self.t_video_start = float(t0)


def _make(n_frames=50, fps=30.0, t0=3.3043, rate=0.001, hz=105.0):
    pts = np.arange(n_frames) / fps
    tg = np.arange(0.0, (n_frames / fps) + 1.0, 1.0 / hz) + t0
    w = np.full((len(tg), 3), rate / np.sqrt(3.0))       # |omega| == rate
    return _Video(pts), _Session(_Frame(tg, w)), _Sync(t0)


# --------------------------------------------------------------------------
# the join invariant — the one that would silently select nothing
# --------------------------------------------------------------------------

def test_t_mid_is_bit_identical_to_what_estimate_foe_recomputes():
    """`estimate_foe` joins on `abs(mid - quiet_t) < 1e-6`.

    It rebuilds midpoints as `0.5 * (pts[:-1] + pts[1:])`. If we produced them
    any other way — cumulative sums, linspace, a resampled grid — the join could
    miss and EVERY frame would be dropped, which looks exactly like "no usable
    flow" rather than like a bug. Same arithmetic, same array, bit-identical.
    """
    video, session, sync = _make()
    t_mid, _, _ = camera._gyro_rot_on_frame_pairs(video, session, sync)
    expected = 0.5 * (video.pts[:-1] + video.pts[1:])
    assert t_mid.shape == expected.shape
    assert np.array_equal(t_mid, expected), "not bit-identical — the 1e-6 join may miss"


def test_shape_matches_image_angular_rate_so_estimate_foe_needs_no_change():
    video, session, sync = _make(n_frames=40)
    t_mid, omega, comps = camera._gyro_rot_on_frame_pairs(video, session, sync)
    assert len(t_mid) == len(video.pts) - 1
    assert omega.shape == (len(video.pts) - 1,)
    assert comps.shape == (len(video.pts) - 1, 3)


# --------------------------------------------------------------------------
# gate semantics
# --------------------------------------------------------------------------

def test_comps_is_the_magnitude_so_the_all_axes_test_becomes_a_magnitude_test():
    """`estimate_foe` does `np.all(np.abs(comps) < r, axis=1)`.

    Replicating |omega| across the three columns makes that exactly
    `|omega| < r` — mount-frame independent (no phone-axes-to-camera-axes
    mapping needed) and strictly conservative, since a magnitude under the bound
    puts every component under it.
    """
    video, session, sync = _make(rate=0.02)
    _, omega, comps = camera._gyro_rot_on_frame_pairs(video, session, sync)
    fin = np.isfinite(omega)
    assert np.allclose(comps[fin], omega[fin, None], rtol=1e-12)
    assert np.allclose(omega[fin], 0.02, atol=1e-9)


@pytest.mark.parametrize("rate,should_pass", [(0.010, True), (0.060, False)])
def test_a_quiet_clip_passes_the_gate_and_a_shaky_one_does_not(rate, should_pass):
    video, session, sync = _make(rate=rate)
    _, _, comps = camera._gyro_rot_on_frame_pairs(video, session, sync)
    passed = np.all(np.abs(comps) < 0.035, axis=1)
    assert passed.any() == should_pass


def test_intervals_with_no_gyro_sample_are_inf_and_can_never_read_as_quiet():
    """The dangerous default. A gap filled with 0.0 is indistinguishable from a
    perfectly steady camera, so missing data would be selected *preferentially*."""
    video, session, sync = _make(n_frames=40)
    # gyro covering only the first third of the clip
    tg = session["gyro"].t
    keep = tg < (tg[0] + (video.pts[-1] - video.pts[0]) / 3.0)
    session = _Session(_Frame(tg[keep], session["gyro"].w[keep]))
    _, omega, comps = camera._gyro_rot_on_frame_pairs(video, session, sync)
    assert np.isinf(omega).any(), "uncovered intervals must be inf, not 0.0"
    assert not np.all(np.abs(comps[np.isinf(omega)]) < 0.035, axis=1).any()


def test_the_video_start_offset_is_applied():
    """Gyro time is session time; `video.pts` is video time.

    Drop the offset and the gate reads the wrong stretch of the drive. The
    reassuring part, asserted here: a badly wrong offset does not quietly return
    plausible-looking numbers — every interval lands outside the gyro's span, so
    the whole thing comes back `None` and the caller falls back rather than
    gating on nonsense.
    """
    video, session, sync = _make(n_frames=40, t0=3.3043, rate=0.001)
    _, omega, _ = camera._gyro_rot_on_frame_pairs(video, session, sync)
    assert np.isfinite(omega).all(), "with the offset applied every interval is covered"

    assert camera._gyro_rot_on_frame_pairs(video, session, _Sync(0.0)) is None, (
        "a wholly wrong offset must fail closed (None), never return numbers")


def test_a_partly_wrong_offset_shows_up_as_uncovered_intervals():
    """The subtler case: enough overlap to return something, but not full cover.

    Those intervals must be `inf` so they are excluded, not zero-filled — a gap
    read as 0.0 rad/s would be selected *preferentially* as the quietest data.
    """
    video, session, sync = _make(n_frames=40, t0=3.3043, rate=0.001)
    span = video.pts[-1] - video.pts[0]
    _, omega, _ = camera._gyro_rot_on_frame_pairs(video, session, _Sync(3.3043 - span / 2))
    assert np.isinf(omega).any(), "partial coverage must leave inf intervals"


# --------------------------------------------------------------------------
# fallbacks
# --------------------------------------------------------------------------

def test_no_gyro_stream_returns_none_so_the_caller_falls_back():
    video, _, sync = _make()
    assert camera._gyro_rot_on_frame_pairs(video, _Session(None), sync) is None
    assert camera._gyro_rot_on_frame_pairs(video, None, sync) is None


def test_a_degenerate_clip_returns_none():
    _, session, sync = _make()
    assert camera._gyro_rot_on_frame_pairs(_Video([0.0, 0.033]), session, sync) is None


def test_the_gate_source_is_recorded_in_provenance():
    """Which instrument gated the FOE must be readable off calibration.json —
    the whole defect was invisible because nothing said where a number came from."""
    src = (Path(camera.__file__)).read_text(encoding="utf-8")
    assert 'cam.source["foe_rotation_gate"]' in src
