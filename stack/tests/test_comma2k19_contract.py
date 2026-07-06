"""comma2k19 (D-009) ties into the SHARED episode contract + A8 check.

Complements ``test_comma2k19.py`` (which covers the action/pose math and route
splits) by asserting two things that loader test does not:

  * the D-009/D-015 ``base250cam`` episodes validate against the one shared
    contract home (:func:`tanitad.data._contract.assert_contract`) at
    ``channels=9`` (3 frames @ 100 ms, D-015) -- an *explicit* contract, not
    an untracked drift;
  * consequence-dominance (A8): with real translating motion, the fraction of
    pixels changing per step clears the toy BEV floor (0.03). Measured on one
    real comma2k19 highway segment this metric is ~0.05 at thresh 0.05 -- low
    for raw RGB, which is exactly why the loss is change-weighted (see the
    2026-07-07 data-eng research note).
"""

from pathlib import Path

import numpy as np
import torch

from tanitad.data._contract import assert_contract, frame_change_fraction
from tanitad.data.comma2k19 import build_episode

T_FRAMES = 80


def _make_seg(root: Path) -> Path:
    d = root / "Chunk_1" / "dongleA_2018-01-01--10-00-00" / "5"
    (d / "global_pose").mkdir(parents=True)
    ft = np.arange(T_FRAMES) / 20.0
    ref = np.array([4278000.0, 635000.0, 4672000.0])
    east = np.array([-0.147, 0.989, 0.0])
    np.save(d / "global_pose" / "frame_times", ft)
    np.save(d / "global_pose" / "frame_positions", ref[None] + 10.0 * ft[:, None] * east[None])
    np.save(d / "global_pose" / "frame_velocities", np.tile(10.0 * east, (T_FRAMES, 1)))
    for name, val in [("speed", 10.0), ("steering_angle", 15.3)]:
        c = d / "processed_log" / "CAN" / name
        c.mkdir(parents=True)
        np.save(c / "t", ft)
        np.save(c / "value", np.full(T_FRAMES, val))
    for p in d.rglob("*.npy"):        # real dataset uses bare names
        p.rename(p.with_suffix(""))
    (d / "video.hevc").write_bytes(b"")
    return d


def _moving_decode(seg, stride, size, max_frames):
    """A tall bright bar translating 5 px/frame -> deterministic per-step motion."""
    n = min(max_frames or 60, 60)
    vid = torch.zeros(n, 3, size, size, dtype=torch.uint8)
    for i in range(n):
        x = (5 * i) % (size - 14)
        vid[i, :, 8:56, x:x + 14] = 255
    return vid


def test_comma2k19_satisfies_shared_contract(tmp_path):
    seg = _make_seg(tmp_path)
    ep = build_episode(seg, size=64, stride=2, max_steps=40, decode_fn=_moving_decode)
    # Validates against the single shared contract home at the base250cam width.
    assert_contract(ep, channels=9)                            # D-015
    assert ep.frames.shape[1] == 9


def test_comma2k19_is_consequence_dominant(tmp_path):
    seg = _make_seg(tmp_path)
    ep = build_episode(seg, size=64, stride=2, max_steps=40, decode_fn=_moving_decode)
    # Ego-relevant motion must sit inside the input (A8), above the toy floor.
    assert frame_change_fraction(ep.frames, thresh=0.05) > 0.03
