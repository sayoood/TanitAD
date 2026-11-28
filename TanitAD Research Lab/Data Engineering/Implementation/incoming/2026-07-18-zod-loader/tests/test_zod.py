"""Standalone tests for the ZOD loader (OWN_DATASET_PLAN §7 #1, CC-BY-SA-4.0).

Zero real bytes and zero Pillow: image decode and OxTS IO are injected. `tanitad`
must be importable (editable stack install, `pip install -e stack`); the loader
module lives beside this tests/ dir and is imported by path.

    pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-18-zod-loader/tests" -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# import the intake module that sits one level up (not yet in stack/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zod as zd                                               # noqa: E402

from tanitad.data._contract import assert_contract             # noqa: E402
from tanitad.data.comma2k19 import CORPUS_META as COMMA_META   # noqa: E402
from tanitad.data.calib import F_REF                           # noqa: E402
from tanitad.data.mixing import MixedWindowDataset             # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic OxTS ego trajectory: constant speed, constant yaw-rate arc         #
# --------------------------------------------------------------------------- #
def make_oxts(n: int, dt: float, speed: float, yaw_rate: float,
              theta0: float = 0.0):
    """(positions_enu[n,3], headings[n]) of a constant-speed constant-yaw arc.

    OxTS heading is the true integrated heading, so `oxts_to_veh4x4` (which uses
    the heading directly) recovers the analytic bicycle steer exactly."""
    x = y = 0.0
    yaw = theta0
    pos = np.zeros((n, 3))
    head = np.zeros(n)
    for t in range(n):
        pos[t] = (x, y, 0.0)
        head[t] = yaw
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        yaw += yaw_rate * dt
    return pos, head


def fake_decode(n=90, size=64):
    """Inject a canonical [n,3,size,size] u8 tensor (skips the real KB crop)."""
    def _decode(frame_paths, intr, s, strict=True):
        return torch.randint(0, 255, (n, 3, s, s), dtype=torch.uint8)
    return _decode


def fake_seq(seq_id="s", n=90, dt=0.1, speed=8.0, yaw_rate=0.05):
    pos, head = make_oxts(n, dt, speed, yaw_rate)
    return {"seq_id": seq_id, "frames": [Path(f"{seq_id}/{i:03d}.jpg")
                                         for i in range(n)],
            "positions_enu": pos, "headings": head}


# --------------------------------------------------------------------------- #
# Kannala-Brandt <-> f-theta identity                                          #
# --------------------------------------------------------------------------- #
def test_kb_to_ftheta_poly_structure():
    intr = zd.kb_to_ftheta(f_px=1800.0, k=(0.1, -0.02, 0.003, -0.0004),
                           cx=1924.0, cy=1084.0, width=3848, height=2168)
    # odd-power KB -> poly = (0, f, 0, f*k1, 0, f*k2, 0, f*k3, 0, f*k4)
    assert intr.poly[0] == 0.0 and abs(intr.poly[1] - 1800.0) < 1e-9
    assert intr.poly[3] == 1800.0 * 0.1 and intr.poly[5] == 1800.0 * -0.02   # k1,k2
    assert intr.poly[2] == 0.0 and intr.poly[4] == 0.0          # even powers zero
    assert abs(intr.paraxial_focal - 1800.0) < 1e-9             # dr/dtheta|0 == f_px
    # r(theta) matches the closed-form KB radius
    th = 0.4
    kb = 1800.0 * (th + 0.1 * th**3 - 0.02 * th**5 + 0.003 * th**7 - 0.0004 * th**9)
    assert abs(float(intr.r_of_theta(th)) - kb) < 1e-6


def test_equidistant_representative_focal():
    # published spec: 3848 wide, HFOV 120 deg -> f = (W/2)/tan-free equidistant
    assert abs(zd.ZOD_FRONT_REPR.paraxial_focal - (3848 / 2) /
               math.radians(60.0)) < 1e-6
    assert zd.ZOD_FRONT_REPR.width == 3848 and zd.ZOD_FRONT_REPR.height == 2168


# --------------------------------------------------------------------------- #
# THE GEOMETRY FALSIFIER (measured, grounded on the published 120-deg spec)     #
# --------------------------------------------------------------------------- #
def test_zod_front_reaches_feff266_fully_observed():
    """GROUNDED FALSIFIER (ZOD front: 120-deg HFOV, 3848x2168). A 120-deg fisheye
    crops INWARD to the canonical ~51.4-deg half-angle -> f_eff=266 with the crop
    box fully inside the native frame -> observed_frac=1.0 -> drop_in=True. This is
    OWN_DATASET_PLAN's #1-unlock falsifier: 'can ZOD reach f_eff=266 at >=50%
    observed?' -> YES, with margin."""
    rep = zd.front_camera_canonicalization(zd.ZOD_FRONT_REPR, 256)
    assert 260.0 < rep["achieved_feff_px"] < 272.0            # ~266, not 434/467
    assert rep["feff_ok"] is True
    assert rep["observed_frac"] >= 0.999                      # crops inward, no pad
    assert rep["observed_ok"] is True
    assert rep["drop_in"] is True


def test_narrow_fov_witness_falsifies_the_gate():
    """The observed_frac gate is NOT vacuous: a NARROW camera (40-deg HFOV) cannot
    fill the canonical canvas -- its periphery would be padded (observed_frac<0.5)
    AND it can't reach the wide canonical angle -> drop_in=False. (Same class as
    Udacity's 0.13 in the D-016 R1 pinhole work.)"""
    f_narrow = (3848 / 2) / math.radians(20.0)               # 40-deg HFOV
    narrow = zd.kb_to_ftheta(f_narrow, (0, 0, 0, 0), 1924.0, 1084.0, 3848, 2168)
    rep = zd.front_camera_canonicalization(narrow, 256)
    assert rep["observed_frac"] < 0.5
    assert rep["drop_in"] is False


def test_canonicalize_fails_loud_on_narrow(monkeypatch):
    """_canonicalize REFUSES (GeometryError) a source that can't reach f_eff=266 at
    >=50% observed, and emits-with-residual under strict=False."""
    f_narrow = (3848 / 2) / math.radians(20.0)
    narrow = zd.kb_to_ftheta(f_narrow, (0, 0, 0, 0), 1924.0, 1084.0, 3848, 2168)
    vid = torch.randint(0, 255, (3, 3, 2168, 3848), dtype=torch.uint8)
    with pytest.raises(zd.GeometryError):
        zd._canonicalize(vid, narrow, 256, strict=True)
    out = zd._canonicalize(vid, narrow, 256, strict=False)
    assert out.shape == (3, 3, 256, 256) and out.dtype == torch.uint8


def test_canonicalize_wide_fisheye_ok():
    """The representative wide ZOD front canonicalizes to [N,3,256,256] u8, no raise."""
    vid = torch.randint(0, 255, (4, 3, 2168, 3848), dtype=torch.uint8)
    out = zd._canonicalize(vid, zd.ZOD_FRONT_REPR, 256, strict=True)
    assert out.shape == (4, 3, 256, 256) and out.dtype == torch.uint8


# --------------------------------------------------------------------------- #
# OxTS -> signals (real heading; shared Cosmos geometry)                        #
# --------------------------------------------------------------------------- #
def test_arc_recovers_analytic_steer():
    dt, speed, yaw_rate = 0.1, 10.0, 0.05
    pos, head = make_oxts(80, dt, speed, yaw_rate)
    actions, poses = zd.zod_signals(pos, head, dt)
    kappa = yaw_rate / speed
    steer_analytic = math.atan(zd.WHEELBASE * kappa)
    mid = slice(5, -5)                                        # drop finite-diff edges
    assert abs(float(np.median(actions[mid, 0])) - steer_analytic) < 5e-3
    assert abs(float(np.median(poses[mid, 3])) - speed) < 0.1  # v recovered


def test_straight_line_zero_steer():
    dt = 0.1
    pos, head = make_oxts(60, dt, speed=12.0, yaw_rate=0.0)
    actions, _ = zd.zod_signals(pos, head, dt)
    assert np.abs(actions[:, 0]).max() < 1e-3


def test_low_speed_steer_guard():
    dt = 0.1
    pos, head = make_oxts(60, dt, speed=0.1, yaw_rate=0.3)
    actions, _ = zd.zod_signals(pos, head, dt)
    assert np.abs(actions[:, 0]).max() < 1e-3                 # curvature zeroed at rest


def test_oxts_heading_is_used_not_motion_heading():
    """Distinguishes ZOD from PandaSet: at v~0 the vehicle heading is still defined
    by OxTS, so yaw follows the OxTS heading rather than an undefined motion dir."""
    pos = np.zeros((10, 3))                                   # stationary
    head = np.full(10, 0.7)
    M = zd.oxts_to_veh4x4(pos, head)
    yaw = np.arctan2(M[:, 1, 0], M[:, 0, 0])
    assert np.allclose(yaw, 0.7, atol=1e-9)


# --------------------------------------------------------------------------- #
# WGS84 -> ENU + CAN steering-ratio recovery                                    #
# --------------------------------------------------------------------------- #
def test_wgs84_to_enu_metres():
    lat0, lon0, alt0 = 57.7, 11.97, 0.0                      # Gothenburg-ish
    lat = np.array([lat0, lat0 + 1e-3])                      # ~111 m north
    lon = np.array([lon0, lon0])
    alt = np.array([0.0, 0.0])
    enu = zd.wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0)
    assert abs(enu[0, 0]) < 1e-6 and abs(enu[0, 1]) < 1e-6    # origin at 0
    assert abs(enu[1, 1] - 111.0) < 2.0                      # ~111 m/mdeg lat


def test_can_steer_ratio_recovers_known_ratio():
    road = np.linspace(-0.3, 0.3, 50)
    can = road * 15.3 + np.random.default_rng(0).normal(0, 1e-3, 50)  # noqa: NPY002
    assert abs(zd.can_steer_ratio(can, road) - 15.3) < 0.05


# --------------------------------------------------------------------------- #
# Contract / I7 / split / mix admissibility                                     #
# --------------------------------------------------------------------------- #
def test_build_episode_contract():
    ep = zd.build_episode(fake_seq(n=90), size=64, decode_fn=fake_decode(90, 64))
    assert_contract(ep, channels=9)                          # D-015 9-ch stack
    assert ep.frames.shape == (88, 9, 64, 64)                # 90 - (n_stack-1)=2
    assert ep.frames.dtype == torch.uint8
    assert ep.actions.shape == (88, 2) and ep.poses.shape == (88, 4)
    assert abs(float(ep.poses[:, 3].mean()) - 8.0) < 0.3     # speed recovered
    assert bool(torch.isfinite(ep.actions).all())


def test_i7_fingerprint_matches_comma2k19():
    assert zd.CORPUS_META == COMMA_META                      # D-017 cross-corpus
    assert zd.LICENSE == "CC-BY-SA-4.0"                      # COPYLEFT -> own shard
    assert zd.DATA_TAG == "data:zod"


def test_split_is_sequence_level():
    seqs = [fake_seq(f"s{i}") for i in range(5)]
    tr, va = zd.split_sequences(seqs, val_frac=0.2, seed=0)
    assert len(tr) == 4 and len(va) == 1
    assert not ({s["seq_id"] for s in tr} & {s["seq_id"] for s in va})  # I3


def test_episode_id_distinct_per_sequence():
    assert zd._episode_id("seqA") != zd._episode_id("seqB")
    assert zd._episode_id("seqA", "front") != zd._episode_id("seqA", "rear")


def test_build_episodes_skips_bad_and_keeps_good():
    good = fake_seq("good", n=90)
    short = fake_seq("short", n=3)                            # too short -> skip
    eps = zd.build_episodes([good, short], size=32, decode_fn=fake_decode(90, 32))
    assert len(eps) == 1 and int(eps[0].episode_id) == zd._episode_id("good")


def test_admissible_in_mix():
    """Identical contract: ZOD co-trains with the anchor via MixedWindowDataset
    without a contract-mismatch error (the owned real+sim mix, D-010)."""
    ds = zd.ZODDataset([fake_seq("a"), fake_seq("b")], window=4, max_horizon=4,
                       size=32, decode_fn=fake_decode(90, 32))
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].shape[1:] == (9, 32, 32)
    assert item["frames"].dtype == torch.float32             # window converts u8->f32
    mix = MixedWindowDataset([(ds, 0.7), (ds, 0.3)], length=8, seed=0)
    assert len(mix) == 8 and mix[0]["frames"].shape[1:] == (9, 32, 32)


def test_f_ref_constant_is_266():
    assert F_REF == 266.0 and zd.CANONICAL_FEFF == 266.0
