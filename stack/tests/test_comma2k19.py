"""comma2k19 loader: contract, action/pose math, route splits — all on a
synthetic segment fixture (no real data, no video decode in CI)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.data.comma2k19 import (Comma2k19Dataset, actions_and_poses,
                                    build_episode, discover_segments,
                                    ecef_to_enu, split_by_route,
                                    stack_two_frames)

T_FRAMES = 100  # 20 fps -> 5 s segment


def make_fake_segment(root: Path, route: str, seg: str) -> Path:
    d = root / "Chunk_1" / route / seg
    (d / "global_pose").mkdir(parents=True)
    ft = np.arange(T_FRAMES) / 20.0
    # Drive roughly east at ~10 m/s from an ECEF point (Zurich-ish).
    ref = np.array([4278000.0, 635000.0, 4672000.0])
    east = np.array([-0.147, 0.989, 0.0])           # unit-ish east in ECEF
    pos = ref[None] + 10.0 * ft[:, None] * east[None]
    vel = np.tile(10.0 * east, (T_FRAMES, 1))
    np.save(d / "global_pose" / "frame_times", ft)
    np.save(d / "global_pose" / "frame_positions", pos)
    np.save(d / "global_pose" / "frame_velocities", vel)
    for name, val in [("speed", 10.0), ("steering_angle", 15.3)]:
        c = d / "processed_log" / "CAN" / name
        c.mkdir(parents=True)
        np.save(c / "t", ft)
        np.save(c / "value", np.full(T_FRAMES, val))
    # np.save appends .npy — the real dataset uses bare names; rename.
    for p in d.rglob("*.npy"):
        p.rename(p.with_suffix(""))
    (d / "video.hevc").write_bytes(b"")             # presence only; decode mocked
    return d


def fake_decode(seg, stride, size, max_frames):
    n = min(max_frames or 50, 50)
    return torch.randint(0, 255, (n, 3, size, size), dtype=torch.uint8)


def test_actions_and_poses_math(tmp_path):
    seg = make_fake_segment(tmp_path, "d0_2018-01-01--10-00-00", "3")
    ft = np.load(seg / "global_pose" / "frame_times")
    pos = np.load(seg / "global_pose" / "frame_positions")
    vel = np.load(seg / "global_pose" / "frame_velocities")
    can = (ft, np.full(T_FRAMES, 10.0))
    steer = (ft, np.full(T_FRAMES, 15.3))
    actions, poses = actions_and_poses(ft, pos, vel, can, steer, stride=2)
    assert abs(actions[5, 0] - math.radians(15.3) / 15.3) < 1e-6  # wheel->road
    assert abs(actions[5, 1]) < 1e-6                              # const speed
    assert abs(poses[5, 3] - 10.0) < 0.2                          # v ~ 10 m/s
    d = np.linalg.norm(poses[-1, :2] - poses[0, :2])
    assert 35.0 < d < 60.0                                        # ~10 m/s * ~4.9 s


def test_ecef_to_enu_starts_at_origin():
    pos = np.array([[4278000.0, 635000.0, 4672000.0]] * 3) + np.arange(3)[:, None]
    enu_p, _ = ecef_to_enu(pos, np.zeros((3, 3)))
    assert np.allclose(enu_p[0], 0.0)


def test_build_episode_contract(tmp_path):
    seg = make_fake_segment(tmp_path, "d0_2018-01-01--10-00-00", "3")
    ep = build_episode(seg, size=64, stride=2, max_steps=30,
                       decode_fn=fake_decode)
    t = ep.frames.shape[0]
    assert ep.frames.shape == (t, 9, 64, 64) and t >= 20      # D-015: 3 frames
    assert ep.actions.shape == (t, 2) and ep.poses.shape == (t, 4)
    assert 0.0 <= float(ep.frames.min()) and float(ep.frames.max()) <= 1.0


def test_stack_frames_semantics():
    from tanitad.data.comma2k19 import stack_frames
    vid = torch.arange(5, dtype=torch.uint8).view(5, 1, 1, 1).expand(5, 3, 2, 2)
    s2 = stack_two_frames(vid)
    assert s2.shape == (4, 6, 2, 2)
    assert int(s2[1, 0, 0, 0]) == 1 and int(s2[1, 3, 0, 0]) == 2  # (t-1, t)
    s3 = stack_frames(vid, n_stack=3)                       # D-015
    assert s3.shape == (3, 9, 2, 2)
    # oldest first, current LAST: stack at t=2 must be frames (0, 1, 2)
    assert [int(s3[0, c, 0, 0]) for c in (0, 3, 6)] == [0, 1, 2]


def test_small_cap_spans_routes(tmp_path):
    """Regression p0-sB00: capping segments must span routes, or the
    route-level split starves one side."""
    from tanitad.data.comma2k19 import sample_segments_across_routes
    for r in range(4):
        for s in range(5):
            make_fake_segment(tmp_path, f"d{r}_2018-02-0{r + 1}--09-00-00", str(s))
    segs = discover_segments(tmp_path)
    capped = sample_segments_across_routes(segs, 6, seed=0)
    routes = {s.parent.name for s in capped}
    assert len(capped) == 6 and len(routes) == 4      # spans ALL routes
    train, val = split_by_route(capped, val_frac=0.2, seed=0)
    assert train and val                              # both sides populated


def test_single_route_split_fails_loudly(tmp_path):
    for s in range(3):
        make_fake_segment(tmp_path, "d9_2018-03-01--08-00-00", str(s))
    with pytest.raises(AssertionError, match="needs >= 2 routes"):
        split_by_route(discover_segments(tmp_path))


def test_route_split_and_dataset(tmp_path):
    for r in range(5):
        for s in range(2):
            make_fake_segment(tmp_path, f"d{r}_2018-01-0{r + 1}--10-00-00", str(s))
    segs = discover_segments(tmp_path)
    assert len(segs) == 10
    train, val = split_by_route(segs, val_frac=0.2, seed=0)
    assert len(train) + len(val) == 10 and len(val) >= 2       # whole routes
    ds = Comma2k19Dataset(val[:2], window=4, max_horizon=2, size=64,
                          max_steps=20, decode_fn=fake_decode)
    item = ds[0]
    assert item["frames"].shape == (4, 9, 64, 64)             # D-015
    assert item["future_frames"].shape == (2, 9, 64, 64)
