"""CPU tests — Stage 2 FILTERING (``lake.filtering``: tier, corrupt-skip, quality
bands, rig, egomotion) + the two-pass DEDUP (``lake.dedup``). Synthetic tensors
only; no cv2, no GPU, no data.
"""

import math

import pytest
import torch

from tanitad.lake import dedup as DD
from tanitad.lake import filtering as FL


def _roll(yaw_rate, speed, T, dt=0.1):
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for t in range(T):
        v = float(speed[t])
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += float(yaw_rate[t]) * dt
    return torch.tensor(rows, dtype=torch.float32)


def _straight(v, T):
    return _roll(torch.zeros(T), torch.full((T,), float(v)), T)


# --------------------------------------------------------------------------- #
# 1. LICENSE -> TIER                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lc,sa,ok,tier", [
    ("owned-safe", False, True, "ship"),
    ("owned-safe", True, False, "ship-sa"),
    ("nc-research", False, False, "nc"),
    ("gated-confidential", False, False, "firewalled"),
])
def test_tier_derivation(lc, sa, ok, tier):
    assert FL.tier_of(lc, sa, ok) == tier


def test_tier_matches_source_registry():
    from tanitad.lake.schema import SOURCE_REGISTRY as R
    assert FL.tier_of(R["comma2k19"].license_class, R["comma2k19"].share_alike,
                      R["comma2k19"].commercial_ok) == "ship"
    assert FL.tier_of(R["zod"].license_class, R["zod"].share_alike,
                      R["zod"].commercial_ok) == "ship-sa"
    assert FL.tier_of(R["nuscenes"].license_class, R["nuscenes"].share_alike,
                      R["nuscenes"].commercial_ok) == "nc"
    assert FL.tier_of(R["physicalai_av"].license_class,
                      R["physicalai_av"].share_alike,
                      R["physicalai_av"].commercial_ok) == "firewalled"


# --------------------------------------------------------------------------- #
# 2. CORRUPT SKIP + the parity skipset                                        #
# --------------------------------------------------------------------------- #
def test_detect_corrupt_variants():
    g = torch.Generator().manual_seed(1)
    good = torch.randint(0, 255, (20, 9, 32, 32), generator=g, dtype=torch.uint8)
    poses = _straight(20.0, 20)
    assert FL.detect_corrupt(good, poses) is None
    assert FL.detect_corrupt(torch.zeros(0, 9, 32, 32), poses) == "zero_length"
    assert FL.detect_corrupt(good, poses[:5]).startswith("frame_count_mismatch")
    assert FL.detect_corrupt(torch.zeros(20, 9, 32, 32, dtype=torch.uint8),
                             poses) == "all_black"
    frozen = good[:1].repeat(20, 1, 1, 1)
    assert FL.detect_corrupt(frozen, poses) == "all_frozen"
    bad_poses = poses.clone()
    bad_poses[3, 0] = float("nan")
    assert FL.detect_corrupt(good, bad_poses) == "nonfinite_poses"


def test_parity_skipset_register_and_query():
    assert "physicalai_av" in FL.CORRUPT_SKIPSET       # seeded with the parity key
    assert FL.is_skipped("physicalai_av", "clip_xyz") is False
    FL.register_corrupt("physicalai_av", "clip_xyz")
    assert FL.is_skipped("physicalai_av", "clip_xyz") is True
    FL.CORRUPT_SKIPSET["physicalai_av"].discard("clip_xyz")   # keep global clean


# --------------------------------------------------------------------------- #
# 3. QUALITY BANDS (banded, not binary)                                       #
# --------------------------------------------------------------------------- #
def test_blur_band_sharp_vs_blurred():
    g = torch.Generator().manual_seed(2)
    sharp = torch.randint(0, 255, (10, 9, 64, 64), generator=g, dtype=torch.uint8)
    assert FL.blur_band(sharp)[0] == "sharp"
    # a smooth gradient has ~0 Laplacian variance -> blurred (but not black/frozen)
    ramp = torch.linspace(0, 255, 64).view(1, 1, 1, 64).expand(10, 9, 64, 64)
    blurred = ramp.round().to(torch.uint8).contiguous()
    assert FL.blur_band(blurred)[0] == "blurred"


def test_exposure_band_dim_and_bright():
    dim = torch.full((8, 9, 32, 32), 2, dtype=torch.uint8)       # nearly all dark
    assert FL.exposure_band(dim)[0] in ("dim", "extreme")
    bright = torch.full((8, 9, 32, 32), 252, dtype=torch.uint8)  # nearly all blown
    assert FL.exposure_band(bright)[0] in ("bright", "extreme")
    g = torch.Generator().manual_seed(4)
    ok = torch.randint(40, 210, (8, 9, 32, 32), generator=g, dtype=torch.uint8)
    assert FL.exposure_band(ok)[0] == "ok"


def test_truncation_static_bottom_band():
    g = torch.Generator().manual_seed(5)
    frames = torch.randint(0, 255, (12, 9, 64, 64), generator=g, dtype=torch.uint8)
    # freeze the bottom quarter across time (a static hood) -> occlusion detected
    frames[:, :, 48:, :] = frames[0:1, :, 48:, :]
    band, frac = FL.truncation_frac(frames)
    assert frac > 0.0 and band in ("partial", "heavy")


# --------------------------------------------------------------------------- #
# 4. EGO-MOTION + RIG SANITY                                                  #
# --------------------------------------------------------------------------- #
def test_assign_rig_two_rig_split():
    assert FL.assign_rig("physicalai_av", 543.0) == ("physicalai_av:rig_a", 543.0)
    assert FL.assign_rig("physicalai_av", 755.0) == ("physicalai_av:rig_b", 755.0)
    assert FL.assign_rig("comma2k19", 437.0) == ("comma2k19:mono", 437.0)
    assert FL.assign_rig("physicalai_av", None) == ("physicalai_av:unknown", None)


def test_egomotion_sane_and_flags():
    p = _straight(30.0, 60)
    sane, reasons, _ = FL.egomotion_sane(p)
    assert sane and reasons == []
    tele = p.clone()
    tele[30, 0] += 500.0                          # a pose teleport
    assert FL.egomotion_sane(tele)[0] is False
    mad = torch.zeros(30, 4)
    mad[:, 3] = torch.arange(30) * 20.0           # +20 m/s per step accel
    ok, reasons, _ = FL.egomotion_sane(mad)
    assert ok is False and any("accel" in r for r in reasons)
    neg = _straight(5.0, 30)
    neg[:, 3] = -3.0                              # negative speed
    assert FL.egomotion_sane(neg)[0] is False


def test_assess_quality_bundle_and_corrupt_shortcircuit():
    g = torch.Generator().manual_seed(6)
    frames = torch.randint(0, 255, (30, 9, 48, 48), generator=g, dtype=torch.uint8)
    poses = _straight(25.0, 30)
    qv = FL.assess_quality(frames, poses, source="comma2k19", cy=437.0)
    assert qv.corrupt is None and qv.egomotion_sane
    assert qv.rig_id == "comma2k19:mono" and qv.blur_band == "sharp"
    d = qv.to_dict()
    assert set(d) >= {"blur_band", "exposure_band", "truncation_band",
                      "egomotion_sane", "rig_id"}
    black = FL.assess_quality(torch.zeros(30, 9, 48, 48, dtype=torch.uint8),
                              poses, source="comma2k19", cy=437.0)
    assert black.corrupt == "all_black"


# =========================================================================== #
# DEDUP — two independent passes                                              #
# =========================================================================== #
def _clip(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, (16, 9, 64, 64), generator=g, dtype=torch.uint8)


def test_phash_identical_and_distinct():
    c = _clip(10)
    assert DD.hamming(DD.clip_phash(c), DD.clip_phash(c.clone())) == 0
    assert DD.hamming(DD.clip_phash(c), DD.clip_phash(_clip(999))) > 10


def test_within_source_collapses_near_dups_keeps_distinct():
    h1 = DD.clip_phash(_clip(10))
    items = [
        {"id": 1, "source": "comma2k19", "phash": h1},
        {"id": 2, "source": "comma2k19", "phash": h1},          # exact dup of 1
        {"id": 3, "source": "comma2k19", "phash": DD.clip_phash(_clip(999))},
    ]
    w = DD.dedup_within_source(items)
    assert w[1]["dedup_cluster_id"] == w[2]["dedup_cluster_id"]  # clustered
    assert w[1]["is_exemplar"] and not w[2]["is_exemplar"]       # lowest id kept
    assert w[3]["dedup_cluster_id"] != w[1]["dedup_cluster_id"]  # distinct kept


def test_within_source_does_not_cross_sources():
    h = DD.clip_phash(_clip(10))
    items = [{"id": 1, "source": "comma2k19", "phash": h},
             {"id": 2, "source": "udacity", "phash": h}]         # same pixels, diff src
    w = DD.dedup_within_source(items)
    assert w[1]["is_exemplar"] and w[2]["is_exemplar"]           # not collapsed


def test_cross_source_gps_time_keeps_multitraversal():
    cell = DD.geo_cell(52.5, 13.4)
    items = [
        {"id": "a", "geo_cell": cell, "t": 1000.0},
        {"id": "b", "geo_cell": cell, "t": 1001.0},   # same place+time -> re-host dup
        {"id": "c", "geo_cell": cell, "t": 9000.0},   # same road, later -> KEEP
        {"id": "d", "geo_cell": "", "t": 0.0},        # no GPS -> keep
    ]
    c = DD.dedup_cross_source(items)
    assert c["a"]["cross_cluster_id"] == c["b"]["cross_cluster_id"]
    assert c["a"]["is_exemplar"] and not c["b"]["is_exemplar"]
    assert c["c"]["is_exemplar"] and c["c"]["multi_traversal"]     # traversal kept+tagged
    assert c["d"]["is_exemplar"] and not c["d"]["multi_traversal"]


def test_two_pass_exemplar_requires_both():
    h = DD.clip_phash(_clip(10))
    cell = DD.geo_cell(1.0, 2.0)
    items = [
        {"id": 1, "source": "comma2k19", "phash": h, "geo_cell": cell, "t": 5.0},
        {"id": 2, "source": "comma2k19", "phash": h, "geo_cell": cell, "t": 6.0},
    ]
    v = DD.two_pass_dedup(items)
    assert v[1].is_exemplar and not v[2].is_exemplar
    assert v[2].within_exemplar is False and v[2].cross_exemplar is False
