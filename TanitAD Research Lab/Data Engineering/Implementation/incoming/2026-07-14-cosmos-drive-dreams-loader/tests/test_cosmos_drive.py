"""Standalone tests for the Cosmos-Drive-Dreams loader (D-014 sim arm, CC-BY-4.0).

Zero real bytes and zero `av`: video decode and pose IO are injected. `tanitad`
must be importable (editable stack install, `pip install -e stack`).

    pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-14-cosmos-drive-dreams-loader/tests" -q
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tanitad_cosmos_drive as cd  # noqa: E402
from tanitad.data._contract import assert_contract  # noqa: E402
from tanitad.data.comma2k19 import CORPUS_META as COMMA_META  # noqa: E402
from tanitad.data.mixing import MixedWindowDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic ego trajectory: constant speed, constant yaw-rate arc             #
# --------------------------------------------------------------------------- #
def make_poses(n: int, dt: float, speed: float, yaw_rate: float) -> np.ndarray:
    """[n,4,4] vehicle-to-world for a constant-speed, constant-yaw-rate arc
    (FLU: x forward). Integrated so speed/steer are analytically known."""
    M = np.zeros((n, 4, 4))
    x = y = yaw = 0.0
    for t in range(n):
        c, s = np.cos(yaw), np.sin(yaw)
        M[t] = np.array([[c, -s, 0.0, x],
                         [s,  c, 0.0, y],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
        x += speed * c * dt
        y += speed * s * dt
        yaw += yaw_rate * dt
    return M


def fake_decode(mp4, size):
    return torch.randint(0, 255, (150, 3, size, size), dtype=torch.uint8)


def make_pose_fn(n=150, dt=1 / 30.0, speed=8.0, yaw_rate=0.05):
    return lambda ref: make_poses(n, dt, speed, yaw_rate)


def fake_clip(cid="clip0", weather="rainy"):
    return {"clip_id": cid, "mp4": Path(f"{cid}.mp4"), "pose": Path("pose"),
            "weather": weather}


# --------------------------------------------------------------------------- #
def test_signals_constant_arc():
    dt = 0.1
    M = make_poses(120, dt, speed=8.0, yaw_rate=0.05)
    actions, poses = cd.poses_to_signals(M, dt)
    i = 60                                             # interior, away from edges
    assert abs(poses[i, 3] - 8.0) < 0.1                # v
    assert abs(actions[i, 1]) < 0.05                   # accel ~ 0 (const speed)
    kappa = 0.05 / 8.0
    assert abs(actions[i, 0] - np.arctan(cd.WHEELBASE * kappa)) < 1e-3  # steer


def test_low_speed_steer_guard():
    # near-stationary: kappa = yaw_rate/v must NOT explode into a hard-lock
    M = make_poses(60, 0.1, speed=0.1, yaw_rate=0.3)
    actions, _ = cd.poses_to_signals(M, 0.1)
    assert np.abs(actions[:, 0]).max() <= cd.MAX_STEER_RAD + 1e-6
    assert abs(actions[30, 0]) < 1e-6                  # v<0.5 -> zero curvature


def test_build_episode_contract():
    ep = cd.build_episode(fake_clip(), size=64, decode_fn=fake_decode,
                          pose_fn=make_pose_fn())
    assert_contract(ep, channels=9)                    # D-015 9-ch stack
    # 150 frames @30fps -> stride 3 -> ~50 steps, minus (n_stack-1)
    assert ep.frames.shape == (48, 9, 64, 64)
    assert ep.actions.shape == (48, 2) and ep.poses.shape == (48, 4)
    assert abs(float(ep.poses[:, 3].mean()) - 8.0) < 0.3


def test_i7_fingerprint_matches_comma2k19():
    # D-017: identical CORPUS_META => probes/eval are cross-corpus admissible
    # and MixedWindowDataset (D-014) accepts Cosmos into the real+sim mix.
    assert cd.CORPUS_META == COMMA_META
    assert cd.LICENSE == "CC-BY-4.0"                   # public-claim firewall


def test_split_is_clip_level():
    clips = [fake_clip(f"c{i}") for i in range(5)]
    tr, va = cd.split_clips(clips, val_frac=0.2, seed=0)
    assert len(tr) == 4 and len(va) == 1
    tr_ids = {c["clip_id"] for c in tr}
    assert not (tr_ids & {c["clip_id"] for c in va})   # I3: disjoint clips


def test_episode_id_distinct_per_weather():
    a = cd._episode_id("clipX", "rainy")
    b = cd._episode_id("clipX", "snowy")
    assert a != b                                      # weather variants differ


def test_clip_id_parsing():
    assert cd._clip_id_of(Path("abc123_0_Rainy.mp4")) == "abc123"
    assert cd._clip_id_of(Path("abc123_1.mp4")) == "abc123"
    assert cd._weather_of(Path("abc123_0_Night.mp4")) == "night"


def test_admissible_in_mix():
    # the whole point of the identical contract: this corpus co-trains with the
    # real anchor via MixedWindowDataset without a contract-mismatch error.
    ds = cd.CosmosDriveDataset([fake_clip("a"), fake_clip("b")], window=4,
                               max_horizon=4, size=32, decode_fn=fake_decode,
                               pose_fn=make_pose_fn())
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].shape[1:] == (9, 32, 32)
    assert item["frames"].dtype == torch.float32       # window converts u8->f32
    mix = MixedWindowDataset([(ds, 0.6), (ds, 0.4)], length=8, seed=0)
    assert len(mix) == 8
    assert mix[0]["frames"].shape[1:] == (9, 32, 32)


def test_discover_pairs_video_with_pose(tmp_path):
    vid = tmp_path / "videos" / f"pinhole_{cd.FRONT_CAM}"
    vid.mkdir(parents=True)
    (tmp_path / "vehicle_pose").mkdir()
    for cid in ("clipA", "clipB"):
        (vid / f"{cid}_0_Rainy.mp4").write_bytes(b"")
        pd = tmp_path / "vehicle_pose" / cid
        pd.mkdir()
        for f in range(3):
            np.save(pd / f"{f:06d}.vehicle_pose.npy", np.eye(4))
    clips = cd.discover_clips(tmp_path)
    assert {c["clip_id"] for c in clips} == {"clipA", "clipB"}
    assert all(c["weather"] == "rainy" for c in clips)
    # pose IO round-trips the per-frame dir into [N,4,4]
    poses = cd.load_vehicle_pose(clips[0]["pose"])
    assert poses.shape == (3, 4, 4)
