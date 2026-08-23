"""Standalone tests for the PandaSet loader (OWN_DATASET_PLAN §7 #2, CC-BY-4.0).

Zero real bytes and zero Pillow: image decode and pose IO are injected. `tanitad`
must be importable (editable stack install, `pip install -e stack`); the loader
module lives beside this tests/ dir and is imported by path.

    pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-15-pandaset-loader/tests" -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# import the intake module that sits one level up (not yet in stack/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandaset as ps                                          # noqa: E402

from tanitad.data._contract import assert_contract             # noqa: E402
from tanitad.data.comma2k19 import CORPUS_META as COMMA_META   # noqa: E402
from tanitad.data.mixing import MixedWindowDataset             # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic ego trajectory: constant speed, constant yaw-rate arc (world xy)   #
# --------------------------------------------------------------------------- #
def make_positions(n: int, dt: float, speed: float, yaw_rate: float,
                   theta0: float = 0.0) -> np.ndarray:
    """[n,3] world positions of a constant-speed, constant-yaw-rate arc.

    Because the vehicle is non-holonomic, the MOTION heading equals the integrated
    heading, so the loader's motion-yaw recovers the analytic steer exactly."""
    x = y = 0.0
    yaw = theta0
    out = np.zeros((n, 3))
    for t in range(n):
        out[t] = (x, y, 0.0)
        x += speed * np.cos(yaw) * dt
        y += speed * np.sin(yaw) * dt
        yaw += yaw_rate * dt
    return out


def fake_decode(n=90, size=64):
    """Inject already-canonical frames (bypass real geometry): one frame per image
    so an under-length image list yields a genuinely short clip."""
    def _f(cam_dir, images, intr, s):
        k = min(n, len(images))
        return torch.randint(0, 255, (k, 3, s, s), dtype=torch.uint8)
    return _f


def make_pose_fn(n=90, dt=0.1, speed=8.0, yaw_rate=0.05):
    return lambda ref: make_positions(n, dt, speed, yaw_rate)


def fake_intr_fn(fx=1970.0):
    return lambda ref: {"fx": fx, "fy": fx, "cx": 960.0, "cy": 540.0}


def fake_seq(sid="seq001", n_img=90):
    return {"seq_id": sid, "cam_dir": Path("cam"),
            "images": [Path(f"{i:02d}.jpg") for i in range(n_img)],
            "poses": Path("poses.json"), "intrinsics": Path("intrinsics.json")}


# --------------------------------------------------------------------------- #
def test_signals_constant_arc():
    """Motion-heading steer matches the analytic bicycle value on a clean arc."""
    dt = 0.1
    pos = make_positions(120, dt, speed=8.0, yaw_rate=0.05)
    actions, poses = ps.pandaset_signals(pos, dt)
    i = 60                                              # interior, away from edges
    assert abs(poses[i, 3] - 8.0) < 0.1                # v
    assert abs(actions[i, 1]) < 0.05                   # accel ~ 0 (const speed)
    kappa = 0.05 / 8.0
    assert abs(actions[i, 0] - np.arctan(ps.WHEELBASE * kappa)) < 2e-3  # steer


def test_motion_heading_offset_free():
    """Steer must NOT depend on the absolute world orientation: two arcs that
    differ only by a constant rotation (theta0) give the SAME steer -- the point
    of using motion heading instead of an absolute (camera) heading."""
    dt = 0.1
    a0 = ps.pandaset_signals(make_positions(120, dt, 8.0, 0.05, theta0=0.0), dt)[0]
    a1 = ps.pandaset_signals(make_positions(120, dt, 8.0, 0.05, theta0=1.3), dt)[0]
    assert np.allclose(a0[40:80, 0], a1[40:80, 0], atol=1e-6)


def test_straight_line_zero_steer():
    dt = 0.1
    pos = make_positions(60, dt, speed=10.0, yaw_rate=0.0)
    actions, poses = ps.pandaset_signals(pos, dt)
    assert np.abs(actions[5:-5, 0]).max() < 1e-3       # straight => ~0 steer
    assert abs(poses[30, 3] - 10.0) < 0.1


def test_low_speed_steer_guard():
    dt = 0.1
    pos = make_positions(60, dt, speed=0.1, yaw_rate=0.3)
    actions, _ = ps.pandaset_signals(pos, dt)
    # cosmos poses_to_signals zeroes curvature at v<0.5 -> no spurious hard-lock
    assert np.abs(actions[:, 0]).max() < 1e-3


def test_build_episode_contract():
    ep = ps.build_episode(fake_seq(), size=64, decode_fn=fake_decode(n=90),
                          pose_fn=make_pose_fn(n=90), intr_fn=fake_intr_fn())
    assert_contract(ep, channels=9)                    # D-015 9-ch stack
    # 90 frames @10Hz -> stride 1 -> 90 steps, minus (n_stack-1)=2 -> 88
    assert ep.frames.shape == (88, 9, 64, 64)
    assert ep.frames.dtype == torch.uint8
    assert ep.actions.shape == (88, 2) and ep.poses.shape == (88, 4)
    assert abs(float(ep.poses[:, 3].mean()) - 8.0) < 0.3
    assert bool(torch.isfinite(ep.actions).all())


def test_i7_fingerprint_matches_comma2k19():
    # D-017: identical CORPUS_META => probes/eval cross-corpus admissible and
    # MixedWindowDataset accepts PandaSet into the owned real+sim mix.
    assert ps.CORPUS_META == COMMA_META
    assert ps.LICENSE == "CC-BY-4.0"                   # permissive owned core
    assert ps.DATA_TAG == "data:pandaset"


def test_split_is_sequence_level():
    seqs = [fake_seq(f"s{i}") for i in range(5)]
    tr, va = ps.split_sequences(seqs, val_frac=0.2, seed=0)
    assert len(tr) == 4 and len(va) == 1
    assert not ({s["seq_id"] for s in tr} & {s["seq_id"] for s in va})  # I3


def test_episode_id_distinct_per_sequence():
    assert ps._episode_id("seqA") != ps._episode_id("seqB")
    assert ps._episode_id("seqA", "front_camera") != ps._episode_id("seqA", "back_camera")


def test_load_camera_poses_parses_devkit_schema(tmp_path):
    """poses.json is the devkit list of {position:{x,y,z}, heading:{w,x,y,z}}."""
    recs = [{"position": {"x": 1.0 * i, "y": 2.0 * i, "z": 0.5},
             "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}} for i in range(4)]
    p = tmp_path / "poses.json"
    p.write_text(json.dumps(recs))
    pos = ps.load_camera_poses(p)
    assert pos.shape == (4, 3)
    assert abs(pos[3, 0] - 3.0) < 1e-9 and abs(pos[3, 1] - 6.0) < 1e-9


def test_load_intrinsics_parses_devkit_schema(tmp_path):
    p = tmp_path / "intrinsics.json"
    p.write_text(json.dumps({"fx": 1970.2, "fy": 1969.9, "cx": 960.0, "cy": 540.0,
                             "extra": "ignored"}))
    intr = ps.load_intrinsics(p)
    assert set(intr) == {"fx", "fy", "cx", "cy"} and abs(intr["fx"] - 1970.2) < 1e-6


def test_discover_sequences(tmp_path):
    for sid in ("s001", "s002"):
        cam = tmp_path / sid / "camera" / ps.FRONT_CAM
        cam.mkdir(parents=True)
        for i in range(3):
            (cam / f"{i:02d}.jpg").write_bytes(b"")
        (cam / "poses.json").write_text("[]")
        (cam / "intrinsics.json").write_text("{}")
    # a sequence missing intrinsics must be skipped
    bad = tmp_path / "s003" / "camera" / ps.FRONT_CAM
    bad.mkdir(parents=True)
    (bad / "00.jpg").write_bytes(b"")
    (bad / "poses.json").write_text("[]")
    seqs = ps.discover_sequences(tmp_path)
    assert {s["seq_id"] for s in seqs} == {"s001", "s002"}
    assert len(seqs[0]["images"]) == 3


def test_build_episodes_skips_bad_and_keeps_good():
    good = fake_seq("good", n_img=90)
    short = fake_seq("short", n_img=3)                 # too short -> raises -> skip
    eps = ps.build_episodes([good, short], size=32, decode_fn=fake_decode(n=90),
                            pose_fn=make_pose_fn(n=90), intr_fn=fake_intr_fn())
    # 'short' pose_fn still returns 90, but its image list is 3 -> n=3 < n_stack+2
    assert len(eps) == 1 and int(eps[0].episode_id) == ps._episode_id("good")


def test_admissible_in_mix():
    """Identical contract: PandaSet co-trains with the anchor via MixedWindowDataset
    without a contract-mismatch error (the owned real+sim mix, D-010)."""
    ds = ps.PandaSetDataset([fake_seq("a"), fake_seq("b")], window=4, max_horizon=4,
                            size=32, decode_fn=fake_decode(n=90),
                            pose_fn=make_pose_fn(n=90), intr_fn=fake_intr_fn())
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].shape[1:] == (9, 32, 32)
    assert item["frames"].dtype == torch.float32       # window converts u8->f32
    mix = MixedWindowDataset([(ds, 0.7), (ds, 0.3)], length=8, seed=0)
    assert len(mix) == 8 and mix[0]["frames"].shape[1:] == (9, 32, 32)


def test_front_camera_is_height_bound_not_drop_in():
    """GROUNDED FINDING (real fx=1970.01, 1920x1080): the square crop is bounded
    by the 1080 height, so PandaSet's front camera does NOT reach f_eff=266 --
    it lands ~467px (~1.75x more zoomed). A wider camera (fx<=~1120) WOULD be
    drop-in. This is the D-016 R1 blocker the loader fails loud on."""
    front = ps.front_camera_canonicalization(ps.PANDASET_FRONT["fx"], 1080, 1920, 256)
    assert front["height_clamped"] is True
    assert front["drop_in"] is False
    assert 440 < front["achieved_feff_px"] < 490      # ~467, NOT 266
    # a genuinely wide pinhole DOES canonicalize cleanly
    wide = ps.front_camera_canonicalization(1000.0, 1080, 1920, 256)
    assert wide["drop_in"] is True
    assert abs(wide["achieved_feff_px"] - ps.CANONICAL_FEFF) / ps.CANONICAL_FEFF < 0.05


def test_canonicalize_fails_loud_on_off_scale():
    """_canonicalize must REFUSE to emit silently mis-scaled PandaSet frames under
    strict (default), and emit-with-residual under strict=False."""
    vid = torch.randint(0, 255, (3, 3, 1080, 1920), dtype=torch.uint8)
    with pytest.raises(ps.GeometryError):
        ps._canonicalize(vid, ps.PANDASET_FRONT["fx"], 256, strict=True)
    out = ps._canonicalize(vid, ps.PANDASET_FRONT["fx"], 256, strict=False)
    assert out.shape == (3, 3, 256, 256) and out.dtype == torch.uint8


def test_real_decode_path_blocks_under_strict(tmp_path):
    """The default real decode path (full-size frames + real fx) raises -- so a
    naive PandaSet ingest cannot silently pollute the owned mix with off-scale
    frames. verify_real_clip (strict=False) still measures the residual."""
    seq = fake_seq("s", n_img=6)
    def full_size_decode(cam_dir, images, intr, s):
        vid = torch.randint(0, 255, (len(images), 3, 1080, 1920), dtype=torch.uint8)
        return ps._canonicalize(vid, float(intr["fx"]), s, strict=True)
    with pytest.raises(ps.GeometryError):
        ps.build_episode(seq, size=256, decode_fn=full_size_decode,
                         pose_fn=make_pose_fn(n=6), intr_fn=fake_intr_fn(fx=1970.0))
