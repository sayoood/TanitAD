"""Standalone tests for the MetaDrive front-camera RGB + perturbation package.

Runs with zero simulator dependencies (MetaDrive is lazily imported only inside
the live rollout, which is skipped here). ``tanitad`` must be importable — it is,
via the editable stack install (``pip install -e stack``).

    pytest "TanitAD Research Hub/Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/tests" -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tanitad_metadrive_frontcam as fc  # noqa: E402
from tanitad.data._contract import EpisodeWindowDataset, assert_contract  # noqa: E402
from tanitad.data.mixing import MixedWindowDataset, load_episode, save_episode  # noqa: E402
from tanitad.data.toy_driving import ToyEpisode  # noqa: E402


# --------------------------------------------------------------------------- #
# frontcam_frame: shape / range / geometry                                     #
# --------------------------------------------------------------------------- #
def test_frontcam_frame_uint8_rgb_shape_range():
    rgb = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    f = fc.frontcam_frame(rgb, size=32)
    assert f.shape == (3, 32, 32)
    assert f.dtype == torch.float32
    assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0


def test_frontcam_frame_accepts_stack_gray_and_rgba():
    # (H, W, 3, stack) -> most-recent frame taken
    stacked = (np.random.default_rng(1).random((40, 40, 3, 3)) * 255).astype(np.uint8)
    assert fc.frontcam_frame(stacked, size=16).shape == (3, 16, 16)
    # gray broadcast to 3 channels
    assert fc.frontcam_frame(np.zeros((20, 20), np.uint8), size=8).shape == (3, 8, 8)
    # rgba -> alpha dropped
    assert fc.frontcam_frame(np.ones((20, 20, 4), np.uint8) * 255, size=8).shape == (3, 8, 8)


def test_frontcam_frame_float01_passthrough_and_centercrop():
    # Non-square float frame in [0,1]: the largest centered square is kept.
    arr = np.zeros((10, 30, 3), np.float32)
    arr[:, 10:20] = 1.0                      # center 10-wide column -> full white square
    f = fc.frontcam_frame(arr, size=10)
    assert float(f.mean()) == pytest.approx(1.0, abs=1e-3)
    assert float(f.max()) <= 1.0


def test_frontcam_frame_takes_most_recent_of_stack():
    stack = np.zeros((8, 8, 3, 2), np.float32)
    stack[..., -1] = 1.0                     # newest frame is white
    f = fc.frontcam_frame(stack, size=8)
    assert float(f.mean()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# assemble_frontcam_episode: 6-channel comma2k19-identical contract            #
# --------------------------------------------------------------------------- #
def _fake_frames3(n, s, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(3, s, s, generator=g) for _ in range(n)]


def test_assemble_frontcam_episode_matches_real_contract():
    n, s, dt = 12, 24, 0.1
    frames3 = _fake_frames3(n, s)
    poses = [fc.pose_from_state(float(i), 0.0, 0.02 * i, 8.0 + 0.3 * i)
             for i in range(n)]
    steer = [0.01 * i for i in range(n)]
    ep = fc.assemble_frontcam_episode(frames3, poses, steer, dt=dt, episode_id=42)

    assert isinstance(ep, ToyEpisode)
    assert ep.frames.shape == (n - 1, fc.REAL_CHANNELS, s, s)   # 2-frame RGB stack
    assert ep.actions.shape == (n - 1, 2)
    assert ep.poses.shape == (n - 1, 2 + 2)
    assert ep.episode_id == 42
    assert_contract(ep, channels=fc.REAL_CHANNELS)
    # accel column == finite-diff of pose speed (aligned to t+1, i.e. [1:n])
    from tanitad.data._contract import finite_diff_accel
    full = finite_diff_accel(np.stack(poses)[:, 3].astype(np.float32), dt)
    np.testing.assert_allclose(ep.actions[:, 1].numpy(), full[1:n], rtol=1e-5)


def test_assemble_frontcam_stacks_consecutive_frames():
    # frames6[t] must be [frame t | frame t+1] channel-stacked (comma2k19 order).
    s = 6
    frames3 = [torch.full((3, s, s), v) for v in (0.0, 0.25, 0.5, 0.75)]
    ep = fc.assemble_frontcam_episode(
        frames3, [fc.pose_from_state(0, 0, 0, 5.0)] * 4, [0.0] * 4,
        dt=0.1, episode_id=1)
    f0 = ep.frames[0]
    assert torch.allclose(f0[:3], torch.full((3, s, s), 0.0))    # frame t
    assert torch.allclose(f0[3:], torch.full((3, s, s), 0.25))   # frame t+1


def test_assemble_requires_two_frames():
    with pytest.raises(ValueError):
        fc.assemble_frontcam_episode(
            [torch.rand(3, 8, 8)], [fc.pose_from_state(0, 0, 0, 1.0)],
            [0.0], dt=0.1, episode_id=0)


# --------------------------------------------------------------------------- #
# perturbation policy: off-expert, bounded, reproducible                       #
# --------------------------------------------------------------------------- #
def test_perturb_action_bounded_and_off_expert():
    cfg = fc.PerturbConfig()
    rng = np.random.default_rng(0)
    base = np.array([0.0, 0.5], np.float32)
    diffs = 0
    for t in range(200):
        a = fc.perturb_action(base, t, cfg, rng)
        assert a.shape == (2,)
        assert float(a.min()) >= -1.0 and float(a.max()) <= 1.0
        if not np.allclose(a, base):
            diffs += 1
    assert diffs > 100          # the point of the sim arm: off-expert coverage


def test_perturb_action_reproducible_from_seed():
    cfg = fc.PerturbConfig()
    base = np.array([0.1, 0.4], np.float32)
    a = [fc.perturb_action(base, t, cfg, np.random.default_rng(7)) for t in range(5)]
    b = [fc.perturb_action(base, t, cfg, np.random.default_rng(7)) for t in range(5)]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_perturb_identity_when_disabled():
    cfg = fc.PerturbConfig(steer_amp=0.0, throttle_pulse_prob=0.0, brake_prob=0.0)
    base = np.array([0.2, 0.3], np.float32)
    for t in range(20):
        np.testing.assert_allclose(
            fc.perturb_action(base, t, cfg, np.random.default_rng(t)), base, atol=1e-6)


def test_perturb_brake_precedes_throttle_pulse():
    # brake_prob covers the low end of the unit interval; force rng into it.
    cfg = fc.PerturbConfig(steer_amp=0.0, brake_prob=1.0, throttle_pulse_prob=0.0)
    a = fc.perturb_action(np.array([0.0, 0.9], np.float32), 0, cfg,
                          np.random.default_rng(0))
    assert a[1] == pytest.approx(cfg.brake)


# --------------------------------------------------------------------------- #
# scenario configs -> MetaDrive env kwargs (pure, offline)                     #
# --------------------------------------------------------------------------- #
def test_scenario_env_config_shape():
    sc = fc.cruise_scenario(size=128)
    cfg = sc.env_config(episode_id=3)
    assert cfg["image_observation"] is True
    assert cfg["vehicle_config"]["image_source"] == "rgb_camera"
    name, w, h = cfg["sensors"]["rgb_camera"]
    assert (name, w, h) == ("RGBCamera", 128, 128)
    assert cfg["start_seed"] == 3
    assert cfg["use_render"] is False


def test_scenario_variants_distinct_and_typed():
    occ = fc.scripted_occluder_scenario()
    blk = fc.blocked_route_scenario()
    assert occ.kind == "scripted_occluder" and blk.kind == "blocked_route"
    # occluder uses denser traffic than the blocked-route stall scenario
    assert occ.traffic_density > blk.traffic_density


def test_populate_scene_cruise_is_noop_others_flagged():
    fc.populate_scene(object(), fc.cruise_scenario())        # no-op, no sim touched
    with pytest.raises(NotImplementedError):
        fc.populate_scene(object(), fc.blocked_route_scenario())


# --------------------------------------------------------------------------- #
# THE UNBLOCK: front-cam episodes mix with real (6ch) episodes; 1ch is rejected #
# --------------------------------------------------------------------------- #
def _six_channel_episode(T, s, eid, seed=0):
    """A comma2k19-shaped ToyEpisode built directly (no sim, no assemble)."""
    g = torch.Generator().manual_seed(seed)
    return ToyEpisode(
        frames=torch.rand(T, fc.REAL_CHANNELS, s, s, generator=g),
        actions=torch.zeros(T, 2), poses=torch.zeros(T, 4), episode_id=eid)


def test_frontcam_mixes_with_real_contract():
    s, w, hz = 16, 3, 2
    # "real" arm: comma2k19-shaped 6ch episodes.
    real = EpisodeWindowDataset([_six_channel_episode(20, s, 100)],
                                window=w, max_horizon=hz)
    # "sim" arm: a front-cam episode assembled through the real path.
    frames3 = _fake_frames3(20, s, seed=5)
    sim_ep = fc.assemble_frontcam_episode(
        frames3, [fc.pose_from_state(float(i), 0, 0, 6.0) for i in range(20)],
        [0.0] * 20, dt=0.1, episode_id=200)
    sim = EpisodeWindowDataset([sim_ep], window=w, max_horizon=hz)

    mix = MixedWindowDataset([(real, 1.0), (sim, 1.0)], length=8, seed=0)
    item = mix[0]
    assert item["frames"].shape[1:] == (fc.REAL_CHANNELS, s, s)
    assert item["future_frames"].shape[1:] == (fc.REAL_CHANNELS, s, s)
    assert "domain" in item
    fracs = mix.mix_report()
    assert set(fracs) == {"domain_0_frac", "domain_1_frac"}


def test_single_channel_sim_is_rejected_by_mix():
    """Guard: the OLD 1-channel BEV path is correctly refused by the mix — this
    is exactly why the front-cam path is needed."""
    s, w, hz = 16, 3, 2
    real = EpisodeWindowDataset([_six_channel_episode(20, s, 1)],
                                window=w, max_horizon=hz)
    bev_ep = ToyEpisode(frames=torch.rand(20, 1, s, s),
                        actions=torch.zeros(20, 2), poses=torch.zeros(20, 4),
                        episode_id=2)
    bev = EpisodeWindowDataset([bev_ep], window=w, max_horizon=hz)
    with pytest.raises(AssertionError):
        MixedWindowDataset([(real, 1.0), (bev, 1.0)], length=4, seed=0)


# --------------------------------------------------------------------------- #
# save/load round-trip on a 6-channel episode (backlog #1c persistence)        #
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_six_channel(tmp_path):
    frames3 = _fake_frames3(10, 12, seed=9)
    ep = fc.assemble_frontcam_episode(
        frames3, [fc.pose_from_state(float(i), 0, 0, 7.0) for i in range(10)],
        [0.005 * i for i in range(10)], dt=0.1, episode_id=321)
    p = tmp_path / "ep.pt"
    save_episode(ep, str(p))
    back = load_episode(str(p))
    assert back.frames.shape == ep.frames.shape == (9, fc.REAL_CHANNELS, 12, 12)
    assert back.episode_id == 321
    # uint8 round-trip: within one quantization step
    assert float((back.frames - ep.frames).abs().max()) <= 1.0 / 255 + 1e-6
