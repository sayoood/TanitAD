"""Configurable input geometry — the regression suite (2026-07-27).

The PI asked for a wider field ("at least 100 degree" … "review the 256 px
resolution"). Making that a CONFIG CHANGE rather than a refactor needs three
things to be permanently true, and this file pins all three:

  1. **Nothing moved.** Every default is still 256 / F_REF 266 / square /
     pinhole, and the canonical path is byte-identical. If this file's §1 ever
     fails, a default changed.
  2. **A non-default geometry actually REACHES the encoder** — and a
     HALF-APPLIED one is refused. A stale default in one call site is exactly
     how every committed v4 number became unreproducible this week.
  3. **Selection parity survives a geometry change.** Changing the crop is a
     RE-CACHE, not a re-selection: same episodes, same uids, same skip indices,
     different pixels — and the cache key moves so the two can never collide.

⛔ Nothing here CHOOSES a geometry. That is the FOV audit's deliverable.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tanitad.config import (EncoderConfig, StackConfig, flagship4b_config,
                            flagship4b_smoke_config)
from tanitad.data import calib as C
from tanitad.data.epcache import cache_key
from tanitad.geometry import (GeometryMismatch, apply_frame,
                              assert_geometry_consistent, build_params,
                              frame_of, geometry_report)
from tanitad.models.encoder import ViTEncoder
from tanitad.models.fourbrain import WorldModel
from tanitad.models.imagination import ImaginationField, advect, sector_mask
from tanitad.models.readout import SpatialGridReadout


# =========================================================================== #
# 1. NOTHING MOVED — the defaults are exactly what they were                   #
# =========================================================================== #
def test_the_deployed_constants_are_unchanged():
    assert C.F_REF == 266.0
    assert C.COMMA2K19_FOCAL_PX == 910.0
    assert EncoderConfig().image_size == 64          # toy default
    assert EncoderConfig().image_width is None       # square by default
    assert flagship4b_config().encoder.image_size == 256
    assert flagship4b_config().encoder.image_width is None
    assert flagship4b_config().encoder.patch_size == 16
    assert flagship4b_config().geometry is None
    assert StackConfig().readout.grid_w is None


def test_CANONICAL_256_is_the_deployed_frame():
    f = C.CANONICAL_256
    assert (f.height, f.width, f.f_ref, f.projection) == (256, 256, 266.0,
                                                          "pinhole")
    assert f.is_canonical and f.is_square and f.size == 256
    # the field the encoder has always seen: 51.39 deg of a 120 deg camera
    assert round(f.hfov_deg, 2) == 51.39 == round(
        math.degrees(2 * C.canonical_halfangle_rad()), 2)


def test_flagship_config_still_yields_the_16x16_grid_and_state_dim_2048():
    cfg = flagship4b_config()
    assert cfg.encoder.token_grid() == (16, 16)
    assert cfg.readout.grid * cfg.readout.grid * cfg.readout.d_readout == 2048


def test_comma2k19_field_ceiling_is_a_measured_constant_not_folklore():
    """comma2k19 physically cannot supply 100 deg; the number is quotable."""
    assert round(C.COMMA2K19_MAX_HFOV_DEG, 2) == 65.20
    assert C.COMMA2K19_MAX_HFOV_DEG == math.degrees(
        2 * math.atan(582.0 / C.COMMA2K19_FOCAL_PX))
    assert C.CanonicalFrame.from_hfov(100.0, 256, 640).hfov_deg > \
        C.COMMA2K19_MAX_HFOV_DEG


# =========================================================================== #
# 1b. the rectangular path DEGENERATES to the square one, exactly              #
# =========================================================================== #
_POLY = (0.0, 927.5032, 23.1353, -58.5012, 16.5067)


def _intr(cy=543.0, per_clip=True):
    return C.FThetaIntrinsics(poly=_POLY, cx=958.0, cy=cy, width=1920,
                              height=1080, per_clip=per_clip)


@pytest.mark.parametrize("size", [256, 224, 128, 64])
def test_rect_crop_equals_square_crop_at_a_square_frame(size):
    i = _intr()
    ch, cw = C.ftheta_crop_size_hw(i, C.CanonicalFrame(size, size))
    assert ch == cw == C.ftheta_crop_size(i, size)


@pytest.mark.parametrize("cy", [543.0, 755.0])
def test_ftheta_crop_resize_is_bit_identical_with_and_without_an_explicit_frame(cy):
    i = _intr(cy)
    vid = torch.randint(0, 256, (2, 3, 540, 960),
                        generator=torch.Generator().manual_seed(0),
                        dtype=torch.uint8)
    a = C.ftheta_crop_resize(vid, i, 256)
    b = C.ftheta_crop_resize(vid, i, frame=C.CANONICAL_256)
    assert torch.equal(a, b)
    assert a.shape == (2, 3, 256, 256)


def test_focal_crop_resize_is_bit_identical_with_and_without_an_explicit_frame():
    vid = torch.randint(0, 256, (2, 3, 874, 1164),
                        generator=torch.Generator().manual_seed(1),
                        dtype=torch.uint8)
    a = C.focal_crop_resize(vid, C.COMMA2K19_FOCAL_PX, 256)
    b = C.focal_crop_resize(vid, C.COMMA2K19_FOCAL_PX, 256,
                            frame=C.CANONICAL_256)
    assert torch.equal(a, b)


def test_two_rig_fix_is_not_regressed_by_the_rectangular_path():
    """The whole point of D-016 R1: both rigs put the horizon on the same row.
    It must survive at a WIDE frame too."""
    wide = C.CanonicalFrame.from_hfov(100.0, 256, 640)
    for frame in (C.CANONICAL_256, wide):
        rows = [C.ftheta_horizon_row(_intr(cy), frame=frame, center="principal")
                for cy in (543.0, 755.0)]
        assert abs(rows[0] - rows[1]) < 1e-6
        assert abs(rows[0] - frame.height / 2) < 1.0
        # and the LEGACY geometric centering is still visibly rig-inconsistent
        legacy = [C.ftheta_horizon_row(_intr(cy), frame=frame,
                                       center="geometric")
                  for cy in (543.0, 755.0)]
        assert abs(legacy[0] - legacy[1]) > 10.0


# =========================================================================== #
# 2. ⭐ A NON-DEFAULT GEOMETRY REACHES THE ENCODER — and a partial one fails    #
# =========================================================================== #
def test_nonsquare_geometry_reaches_the_encoder_end_to_end():
    """⭐ THE LOAD-BEARING TEST. It FAILS if any call site keeps 256/square."""
    cfg = flagship4b_smoke_config()
    cfg.encoder = dataclasses.replace(cfg.encoder, in_channels=3, image_size=32,
                                      patch_size=8, d_model=16, depth=1,
                                      n_heads=2)
    cfg.readout = dataclasses.replace(cfg.readout, grid=2, d_readout=4)
    wide = C.CanonicalFrame(height=32, width=80, f_ref=100.0)
    apply_frame(cfg, wide)

    # the config carries it...
    assert cfg.encoder.image_hw() == (32, 80)
    assert cfg.encoder.token_grid() == (4, 10)
    assert frame_of(cfg) == wide
    assert assert_geometry_consistent(cfg) == wide

    # ...and the ENCODER is actually built for it
    enc = ViTEncoder(cfg.encoder)
    assert enc.grid_shape == (4, 10)
    assert enc.n_tokens == 40
    assert enc.pos.shape == (1, 40, 16)
    tok = enc(torch.zeros(2, 3, 32, 80))
    assert tok.shape == (2, 40, 16)

    # ...and a SQUARE input of the old size is now REFUSED, not silently padded
    with pytest.raises(ValueError, match="declares"):
        enc(torch.zeros(2, 3, 32, 32))


def test_the_whole_worldmodel_runs_on_a_nonsquare_input():
    """Readout, imagination and every reshape downstream of the token grid."""
    cfg = flagship4b_smoke_config()
    cfg.encoder = dataclasses.replace(cfg.encoder, in_channels=3, image_size=32,
                                      patch_size=8, d_model=16, depth=1,
                                      n_heads=2)
    cfg.readout = dataclasses.replace(cfg.readout, grid=2, d_readout=4)
    cfg.h15 = dataclasses.replace(cfg.h15, enabled=True, depth=1)
    apply_frame(cfg, C.CanonicalFrame(32, 80, f_ref=100.0))
    m = WorldModel(cfg)
    assert m.encoder.grid_shape == (4, 10)
    assert m.readout.token_h, m.readout.token_w == (4, 10)
    frames = torch.zeros(2, 3, 32, 80)
    tok = m.encode_tokens(frames)
    assert tok.shape == (2, 40, 16)
    assert m.readout(tok).shape == (2, cfg.readout.grid ** 2 * 4)
    masked, vis = sector_mask(frames, m.encoder.grid_shape)
    assert masked.shape == frames.shape and vis.shape == (2, 40)
    out, logvar = m.imagination(tok, vis)
    assert out.shape == tok.shape and logvar.shape == (2, 40)


def test_state_dim_is_INVARIANT_to_the_input_geometry():
    """⭐ Why widening the field is not a whole-model redesign: the readout is
    the firewall. A 4x wider input still hands the predictor state_dim 2048."""
    dims = []
    for w in (256, 512, 640):
        cfg = flagship4b_config()
        apply_frame(cfg, C.CanonicalFrame(256, w, f_ref=300.0))
        enc = ViTEncoder(dataclasses.replace(cfg.encoder, d_model=32, depth=1,
                                             n_heads=2))
        ro = SpatialGridReadout(enc.n_tokens, 32, grid=cfg.readout.grid,
                                d_readout=cfg.readout.d_readout,
                                token_grid=enc.grid_shape,
                                grid_w=cfg.readout.grid_w)
        dims.append(ro.out_dim)
    assert dims == [2048, 2048, 2048]


def test_the_two_pooling_ROUTES_are_the_same_operation():
    """⭐ ROUTE CONFLICT, settled. The FOV-audit stream proposed adaptive-4x4
    pooling as an alternative to this stream's `(grid, grid_w)` readout. They are
    the SAME operation wherever both are defined — bit-identical at the deployed
    16x16 and at 16x40, equal to float32 summation noise elsewhere. Adaptive is
    additionally defined on a grid that does not tile, which is why it is the
    fallback here rather than a competing route."""
    from torch import nn
    g = torch.Generator().manual_seed(7)
    for th, tw in ((16, 16), (16, 40)):
        x = torch.randn(3, 64, th, tw, generator=g)
        assert torch.equal(nn.AdaptiveAvgPool2d((4, 4))(x),
                           nn.AvgPool2d((th // 4, tw // 4))(x))
    for th, tw in ((24, 60), (24, 24)):
        x = torch.randn(3, 64, th, tw, generator=g)
        d = (nn.AdaptiveAvgPool2d((4, 4))(x)
             - nn.AvgPool2d((th // 4, tw // 4))(x)).abs().max()
        assert float(d) < 1e-6, float(d)          # summation order, not semantics
    # the readout picks the exact kernel where it tiles (deployed path unchanged)
    # and adaptive where it does not (instead of the old assert)
    assert SpatialGridReadout(16 * 16, 64, grid=4, d_readout=8).exact_pool
    assert SpatialGridReadout(16 * 40, 64, grid=4, d_readout=8,
                              token_grid=(16, 40)).exact_pool
    ro = SpatialGridReadout(16 * 42, 64, grid=4, d_readout=8,
                            token_grid=(16, 42))
    assert not ro.exact_pool and ro.out_dim == 4 * 4 * 8
    assert ro(torch.randn(2, 16 * 42, 64, generator=g)).shape == (2, 128)


def test_the_multiple_of_64_tiling_rule_and_its_TWO_corrections():
    """⭐ The encoder-tokenization stream reported *"input width must be a
    multiple of 64 or the readout breaks state_dim = 2048"*. The RULE is real
    and is pinned here. TWO parts of the framing are corrected, by measurement:

      1. **448 DOES satisfy it** (448 = 7x64 -> 28 token cols, 28 % 4 == 0).
      2. **state_dim does NOT break when it fails** — since the pooling routes
         were converged the readout falls back to adaptive pooling and still
         yields 2048. The cost is UNEVEN BINS, not a shape failure.

    The rule is also not a constant: it is ``patch_size * readout_grid``."""
    from tanitad.geometry import tiling_report
    cfg = flagship4b_config()
    assert cfg.encoder.patch_size * cfg.readout.grid == 64      # where 64 comes from

    for w, tiles in ((256, True), (384, True), (448, True), (512, True),
                     (640, True), (704, True), (960, True), (672, False)):
        c = flagship4b_config()
        apply_frame(c, C.CanonicalFrame(256, w, f_ref=300.0))
        rep = tiling_report(c)
        assert rep["tiles_exactly"] is tiles, (w, rep)
        assert rep["width_ok"] is (w % 64 == 0)
        assert rep["state_dim"] == 2048                # invariant EITHER WAY
    # 448 specifically — the one reported as failing
    assert 448 % 64 == 0 and (448 // 16) % 4 == 0

    # ...and a non-tiling width really does still produce a 2048-d state
    c = flagship4b_config()
    apply_frame(c, C.CanonicalFrame(256, 672, f_ref=300.0))
    enc = ViTEncoder(dataclasses.replace(c.encoder, d_model=32, depth=1,
                                         n_heads=2))
    ro = SpatialGridReadout(enc.n_tokens, 32, grid=c.readout.grid,
                            d_readout=c.readout.d_readout,
                            token_grid=enc.grid_shape)
    assert enc.grid_shape == (16, 42) and not ro.exact_pool
    assert ro.out_dim == 2048
    assert ro(torch.randn(2, enc.n_tokens, 32)).shape == (2, 2048)


def test_an_asymmetric_readout_grid_is_expressible_and_changes_state_dim():
    cfg = flagship4b_config()
    apply_frame(cfg, C.CanonicalFrame(256, 640, f_ref=300.0))
    cfg.readout = dataclasses.replace(cfg.readout, grid_w=10)
    enc = ViTEncoder(dataclasses.replace(cfg.encoder, d_model=32, depth=1,
                                         n_heads=2))
    ro = SpatialGridReadout(enc.n_tokens, 32, grid=cfg.readout.grid,
                            d_readout=cfg.readout.d_readout,
                            token_grid=enc.grid_shape, grid_w=10)
    assert enc.grid_shape == (16, 40)
    assert ro.out_dim == 4 * 10 * 128


# --- the stale-default guard ------------------------------------------------ #
def test_a_HALF_APPLIED_geometry_is_refused_in_both_directions():
    """⭐ The guard the v4 reproducibility failure earned."""
    cfg = flagship4b_smoke_config()
    cfg.geometry = C.CanonicalFrame(64, 128).to_dict()          # frame moved
    with pytest.raises(GeometryMismatch, match="HALF-APPLIED"):
        assert_geometry_consistent(cfg)

    cfg = flagship4b_smoke_config()
    cfg.encoder = dataclasses.replace(cfg.encoder, image_width=128)  # enc moved
    with pytest.raises(GeometryMismatch, match="HALF-APPLIED"):
        assert_geometry_consistent(cfg)


def test_apply_frame_can_never_half_apply():
    for f in (C.CanonicalFrame(64, 128), C.CanonicalFrame(64, 64),
              C.CanonicalFrame(128, 320, f_ref=200.0, projection="cylindrical")):
        cfg = flagship4b_smoke_config()
        cfg.encoder = dataclasses.replace(cfg.encoder, patch_size=8)
        apply_frame(cfg, f)
        assert assert_geometry_consistent(cfg) == f
        assert cfg.encoder.image_hw() == f.hw


def test_apply_frame_refuses_a_frame_the_patch_size_cannot_tokenize():
    with pytest.raises(GeometryMismatch, match="divisible"):
        apply_frame(flagship4b_smoke_config(), C.CanonicalFrame(60, 60))


def test_grid_hw_raises_on_a_nonsquare_grid_instead_of_lying():
    enc = ViTEncoder(EncoderConfig(in_channels=3, image_size=32, image_width=64,
                                   patch_size=8, d_model=16, depth=1, n_heads=2))
    assert enc.grid_shape == (4, 8)
    with pytest.raises(ValueError, match="non-square"):
        _ = enc.grid_hw
    # a square encoder still exposes it
    assert ViTEncoder(EncoderConfig(in_channels=3, image_size=32, patch_size=8,
                                    d_model=16, depth=1, n_heads=2)).grid_hw == 4


def test_frame_size_scalar_raises_on_a_nonsquare_frame():
    with pytest.raises(ValueError, match="not square"):
        _ = C.CanonicalFrame(256, 640).size


def test_as_frame_refuses_two_sources_of_truth():
    with pytest.raises(ValueError, match="two sources of truth"):
        C.as_frame(C.CanonicalFrame(128, 128), 256, 999.0)
    # frame alone, or scalars alone, are both fine
    assert C.as_frame(None, 128, 300.0) == C.CanonicalFrame(128, 128, 300.0)
    assert C.as_frame(C.CanonicalFrame(128, 128), 256, C.F_REF).height == 128


# =========================================================================== #
# 3. SELECTION PARITY — a geometry change is a RE-CACHE, not a re-selection     #
# =========================================================================== #
_LEGACY_PARAMS = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2"}
_IDS = [{"clip_id": f"c{i:04d}"} for i in range(32)]


def test_the_canonical_geometry_leaves_the_cache_key_BYTE_IDENTICAL():
    """⚠️ PARITY-CRITICAL. If this fails, `physicalai-train-e438721ae894` has
    silently stopped meaning what it means."""
    cfg = flagship4b_config()
    assert build_params(cfg, _LEGACY_PARAMS) == _LEGACY_PARAMS
    assert C.geometry_params() == {}
    assert C.geometry_params(C.CANONICAL_256) == {}
    from tanitad.data.physicalai import geometry_build_params
    assert geometry_build_params() == {}
    assert cache_key(_IDS, build_params(cfg, _LEGACY_PARAMS)) == \
        cache_key(_IDS, _LEGACY_PARAMS)


def test_a_noncanonical_geometry_mints_a_DIFFERENT_key_and_cannot_collide():
    from tanitad.data.physicalai import geometry_build_params
    base = cache_key(_IDS, _LEGACY_PARAMS)
    wide = C.CanonicalFrame.from_hfov(100.0, 256, 640)
    keys = {
        "legacy": base,
        "wide": cache_key(_IDS, {**_LEGACY_PARAMS, **geometry_build_params(wide)}),
        "wide_cyl": cache_key(_IDS, {**_LEGACY_PARAMS,
                                     **geometry_build_params(wide, "cylindrical")}),
        "f_ref_only": cache_key(_IDS, {**_LEGACY_PARAMS, **geometry_build_params(
            C.CanonicalFrame(256, 256, f_ref=400.0))}),
    }
    assert len(set(keys.values())) == 4, keys
    # ...and the key is a pure function of the frame (re-derivable, not random)
    assert cache_key(_IDS, {**_LEGACY_PARAMS,
                            **geometry_build_params(wide)}) == keys["wide"]


def test_changing_only_f_ref_now_moves_the_key():
    """REGRESSION for a PRE-EXISTING hole: the build params carried `size` but
    never `f_ref`, so changing F_REF produced DIFFERENT PIXELS UNDER THE SAME
    CACHE KEY."""
    assert "f_ref" not in _LEGACY_PARAMS
    assert C.geometry_params(C.CanonicalFrame(256, 256, f_ref=400.0)) != {}


def test_the_FOV_audit_streams_exact_reproduction_of_the_key_hole_is_closed():
    """⭐ CONVERGENT DETECTION. The FOV-audit stream found the same defect from
    an independent implementation: *"`F_REF` 266 and 133 give the identical key
    `eafe5e4eb363`; a re-crop keeping `size=256` writes
    `physicalai-train-e438721ae894` with different pixels and passes every
    guard."* This is their exact reproduction, now failing as it should."""
    from tanitad.data.physicalai import geometry_build_params
    legacy = dict(_LEGACY_PARAMS)

    # the OLD behaviour, reconstructed: params carried `size` and nothing else
    # geometric, so both focals hashed identically. This is the bug.
    assert cache_key(_IDS, {**legacy, "size": 256}) == \
        cache_key(_IDS, {**legacy, "size": 256}), "sanity"

    # the FIX: F_REF 266 vs 133 at the SAME size now produce DIFFERENT keys
    f266 = C.CanonicalFrame(256, 256, f_ref=266.0)
    f133 = C.CanonicalFrame(256, 256, f_ref=133.0)
    k266 = cache_key(_IDS, {**legacy, **geometry_build_params(f266)})
    k133 = cache_key(_IDS, {**legacy, **geometry_build_params(f133)})
    assert k266 != k133, "F_REF 266 and 133 must not share a cache key"
    # ...and 266 (the canonical one) is still byte-identical to the legacy key,
    # so `physicalai-train-e438721ae894` keeps its exact meaning
    assert k266 == cache_key(_IDS, legacy)
    assert geometry_build_params(f266) == {}
    assert geometry_build_params(f133) == {"geom": "256x256f133pin"}


def test_a_NON_SQUARE_frame_is_expressible_in_the_build_params():
    """The FOV-audit stream's second point: *"a non-square frame cannot be
    expressed in `params` at all."* It can now — and it lands a distinct key."""
    from tanitad.data.physicalai import geometry_build_params
    wide = C.CanonicalFrame(256, 640, f_ref=268.5119)
    frag = geometry_build_params(wide)
    assert frag == {"geom": "256x640f268.5119pin"}
    assert "640" in frag["geom"] and "256" in frag["geom"]
    # a frame differing ONLY in width must not collide with the square one
    sq = C.CanonicalFrame(256, 256, f_ref=268.5119)
    assert geometry_build_params(sq) != frag
    assert cache_key(_IDS, {**_LEGACY_PARAMS, **frag}) != \
        cache_key(_IDS, {**_LEGACY_PARAMS, **geometry_build_params(sq)})


def test_the_selection_chain_takes_no_geometry_argument():
    """The code-level half of the selection verdict: nothing that decides WHICH
    episodes exist can even see the geometry."""
    import inspect

    from tanitad.data.physicalai import discover_r0_clips, split_clips
    for fn in (discover_r0_clips, split_clips):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"size", "frame", "f_ref", "projection",
                              "projection_mode"}), (fn.__name__, params)


def test_episode_identity_and_skips_are_invariant_to_geometry(tmp_path):
    """⭐ Same sources -> same ep_%05d.pt uids and the SAME skip indices at two
    different geometries. Only the pixels differ."""
    from tanitad.data.epcache import build_episodes_cached
    from tanitad.data.toy_driving import ToyEpisode
    runs = {}
    for label, hw in (("sq", (32, 32)), ("wide", (32, 80))):
        srcs = [{"clip_id": f"c{i:03d}"} for i in range(10)]

        def build_one(s, hw=hw):
            i = int(s["clip_id"][1:])
            if i in (2, 6):
                raise RuntimeError("synthetic corrupt clip")
            return ToyEpisode(frames=torch.zeros(4, 9, *hw, dtype=torch.uint8),
                              actions=torch.zeros(4, 2), poses=torch.zeros(4, 4),
                              episode_id=i)
        eps = build_episodes_cached(srcs, build_one, tmp_path / label, "t",
                                    {"geom": label})
        d = next((tmp_path / label).glob("t-*"))
        runs[label] = (sorted(p.name for p in d.glob("ep_*.pt")),
                       sorted(p.name for p in d.glob("skip_*")),
                       tuple(eps[0].frames.shape))
    assert runs["sq"][0] == runs["wide"][0]          # same episode uids
    assert runs["sq"][1] == runs["wide"][1]          # same skip indices
    assert runs["sq"][0] and len(runs["sq"][1]) == 2
    assert runs["sq"][2] != runs["wide"][2]          # ...only the pixels differ


def test_a_recropped_cache_is_NOT_silently_accepted_as_the_parity_corpus(tmp_path):
    """The operational catch: the guard keys on the DIRECTORY NAME, so a
    re-cropped build reads as NON-PARITY until its key is registered."""
    from tanitad.data import parity
    d = tmp_path / "physicalai-train-deadbeef1234"
    d.mkdir()
    (d / "ep_00000.pt").touch()
    assert parity.corpus_key_of(d) is None
    with pytest.raises(parity.ParityViolation, match="PARITY VIOLATION"):
        parity.assert_parity_corpus(d, label="recrop", require=True)


def test_register_geometry_sibling_accepts_a_true_recache_and_refuses_a_reselection(
        tmp_path):
    from tanitad.data import parity
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    uids = list(ent["episode_uids"])
    good = tmp_path / "good"
    good.mkdir()
    for u in uids:
        (good / u).touch()
    for i in ent["skip_indices"]:
        (good / f"skip_{i:05d}").write_text("x")
    geom = C.CanonicalFrame.from_hfov(100.0, 256, 640).to_dict()
    out = parity.register_geometry_sibling(good, new_key="physicalai-train-aaaa1111",
                                           geometry=geom)
    assert out["episode_count"] == 2376
    assert out["episode_uid_sha256"] == ent["episode_uid_sha256"]
    assert out["provenance"]["derived_from"] == parity.PARITY_TRAIN_KEY
    assert out["provenance"]["geometry"] == geom

    bad = tmp_path / "bad"                       # one episode substituted
    bad.mkdir()
    for u in uids[:-1]:
        (bad / u).touch()
    (bad / "ep_09999.pt").touch()
    for i in ent["skip_indices"]:
        (bad / f"skip_{i:05d}").write_text("x")
    with pytest.raises(parity.ParityViolation, match="never MEMBERSHIP"):
        parity.register_geometry_sibling(bad, new_key="physicalai-train-bbbb2222",
                                         geometry=geom)


# =========================================================================== #
# 4. the CYLINDRICAL projection                                                #
# =========================================================================== #
def test_cylindrical_is_equidistant_in_azimuth_and_pinhole_is_not():
    cyl = C.CanonicalFrame(256, 640, f_ref=366.69, projection="cylindrical")
    pin = C.CanonicalFrame(256, 640, f_ref=268.51, projection="pinhole")
    assert round(cyl.hfov_deg, 1) == round(pin.hfov_deg, 1) == 100.0
    x, _y, z = C.cylindrical_rays(cyl)
    phi = torch.atan2(x, z)[0]                     # azimuth along the top row
    d = torch.diff(phi)
    assert torch.allclose(d, d[:1], atol=1e-6)     # UNIFORM angular step
    xp, _yp, zp = C.cylindrical_rays(pin)
    dp = torch.diff(torch.atan2(xp, zp)[0])
    assert (dp.max() - dp.min()) > 1e-3            # pinhole step is NOT uniform


def test_projection_density_report_reproduces_the_documented_table():
    for hfov, tan_over_t, sec2 in ((100.0, 1.3656, 2.4203), (120.0, 1.6540, 4.0)):
        r = C.projection_density_report(C.CanonicalFrame.from_hfov(hfov, 256, 640))
        assert abs(r["cumulative_radius_vs_equidistant"] - tan_over_t) < 1e-3
        assert abs(r["edge_local_density_vs_center"] - sec2) < 1e-3
    c = C.projection_density_report(
        C.CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical"))
    assert c["edge_local_density_vs_center"] == 1.0


def test_cylindrical_rectify_puts_the_boresight_at_the_output_centre_for_BOTH_rigs():
    """⭐ The rig fix, by construction: ray (0,0,1) -> exactly (cx, cy)."""
    # ODD dims so the frame centre ((W-1)/2) is an exact pixel index — on an
    # even frame the "centre" pixel sits half a pixel off-axis, which shows up
    # as a ~5 px native offset and would make this test measure rounding, not
    # the rig fix.
    frame = C.CanonicalFrame(65, 161, f_ref=90.0, projection="cylindrical")
    for cy in (543.0, 755.0):
        i = _intr(cy)
        grid, mask = C.cylindrical_grid(i, 1080, 1920, frame)
        cr, cc = (frame.height - 1) // 2, (frame.width - 1) // 2
        gx, gy = grid[0, cr, cc].tolist()
        u = (gx + 1) / 2 * (1920 - 1)
        v = (gy + 1) / 2 * (1080 - 1)
        assert abs(u - i.cx) < 1.0 and abs(v - i.cy) < 1.0
        assert bool(mask[cr, cc])


def _padded_row_frac(intr, frame, h=1080, w=1920):
    c_h, c_w, top, left = C.ftheta_crop_box_hw(intr, h, w, center="principal",
                                               frame=frame)
    return (max(0, -top) + max(0, (top + c_h) - h)) / c_h


def test_the_DEPLOYED_crop_is_rig_ASYMMETRIC_and_fabricates_pixels():
    """⛔ A LIVE DEFECT IN WHAT WE TRAIN ON TODAY, not a future concern.

    MEASURED (route_and_rig_2026-07-27.json, and reproduced independently by the
    FOV-audit stream at 11.3 %): the principal-point-centred crop spills past the
    native bottom edge for rig B (cy~755) and is replicate-padded, while rig A
    (cy~543) is padded 0 %. So ~29 % of the corpus carries **fabricated rows that
    correlate perfectly with rig** — precisely the kind of shortcut this model
    eats. This test PINS the defect so it cannot be forgotten or silently
    'fixed' by a geometry change that merely hides it."""
    a = _padded_row_frac(_intr(543.0), C.CANONICAL_256)
    b = _padded_row_frac(_intr(755.0), C.CANONICAL_256)
    assert a == 0.0
    assert 0.10 < b < 0.12, b            # ~10.9 %, matches the 11.3 % report
    # widening with a CROP inherits it, and at 120 deg makes it markedly worse
    b100 = _padded_row_frac(_intr(755.0), C.CanonicalFrame.from_hfov(100.0, 256, 640))
    b120 = _padded_row_frac(_intr(755.0), C.CanonicalFrame.from_hfov(120.0, 256, 640))
    assert b100 > 0.10 and b120 > b100


def test_the_CYLINDRICAL_path_removes_the_FABRICATED_pixel_asymmetry():
    """⭐ The mechanism finding: the crop REPLICATE-PADS (invents rows that look
    like real road), the cylindrical path MASKS (explicitly unobserved, and
    reported via `last_observed_frac`). At 100 deg the geometric asymmetry is
    nearly gone too (rig B 0.69 % vs the crop's 10.5 %).

    ⛔ This does NOT claim cylindrical wins on ADE — that is the FOV audit's
    experiment. It claims the wide rebuild need not bake in a rig-identifiable
    artefact."""
    probe = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8)
    for hfov, tol in ((100.0, 0.02), (120.0, 0.10)):
        frame = C.CanonicalFrame.from_hfov(hfov, 256, 640, "cylindrical")
        fr = {}
        for rig, cy in (("A", 543.0), ("B", 755.0)):
            C.cylindrical_rectify(probe, _intr(cy), frame)
            fr[rig] = 1.0 - C.cylindrical_rectify.last_observed_frac
        assert fr["A"] < 1e-6                      # rig A fully observed
        assert fr["B"] < tol                       # rig B: masked, and small
        # and it is a MASK, never fabricated content
        assert C.cylindrical_rectify.last_mask.dtype == torch.bool


def test_cylindrical_rectify_refuses_the_corpus_median_intrinsic():
    with pytest.raises(ValueError, match="PER-CLIP"):
        C.cylindrical_rectify(torch.zeros(1, 3, 108, 192),
                              C.PHYSICALAI_FRONT_WIDE_FTHETA,
                              C.CanonicalFrame(32, 32))


def test_cylindrical_rectify_shape_dtype_and_observed_mask():
    frame = C.CanonicalFrame(64, 160, f_ref=90.0, projection="cylindrical")
    vid = torch.randint(0, 256, (2, 3, 540, 960),
                        generator=torch.Generator().manual_seed(3),
                        dtype=torch.uint8)
    out = C.cylindrical_rectify(vid, _intr(), frame)
    assert out.shape == (2, 3, 64, 160) and out.dtype == torch.uint8
    assert C.cylindrical_rectify.last_f_eff == 90.0
    assert 0.0 < C.cylindrical_rectify.last_observed_frac <= 1.0
    assert C.cylindrical_rectify.last_frame is frame


# =========================================================================== #
# 5. PER-CORPUS policy — all three live options are EXPRESSIBLE                #
# =========================================================================== #
def test_a_SQUARE_frame_cannot_deliver_100deg_on_this_sensor():
    """⭐ MEASURED (rebuild_cost_2026-07-27.json): asking for 100 deg on a
    256x256 frame delivers **67.1 deg** — the f-theta crop needs a 1595 px side
    and CLAMPS at the sensor's 1080 px height, so it ZOOMS instead of widening.
    Silently, before this. Widening therefore REQUIRES more columns, which is
    the whole reason the non-square encoder path exists."""
    i = _intr()
    sq = C.CanonicalFrame.from_hfov(100.0, 256, 256)
    ch, cw = C.ftheta_crop_size_hw(i, sq)
    assert ch == cw == i.height                       # clamped at the sensor
    vid = torch.zeros(1, 3, i.height, i.width, dtype=torch.uint8)
    C.ftheta_crop_resize(vid, i, frame=sq)
    got = math.degrees(2 * math.atan((sq.width / 2)
                                     / C.ftheta_crop_resize.last_f_eff))
    assert 66.0 < got < 68.0, got                     # NOT the requested 100
    assert sq.hfov_deg - got > 30.0
    # ...whereas the same field on a WIDE frame is delivered in full
    wide = C.CanonicalFrame.from_hfov(100.0, 256, 640)
    ch, cw = C.ftheta_crop_size_hw(i, wide)
    assert cw < i.width and ch < i.height             # fits, no clamp
    C.ftheta_crop_resize(vid, i, frame=wide)
    got_w = math.degrees(2 * math.atan((wide.width / 2)
                                       / C.ftheta_crop_resize.last_f_eff))
    assert abs(got_w - 100.0) < 0.5, got_w


def test_120deg_is_reachable_on_the_physicalai_front_wide():
    """The PI named 100-120 deg. 120 deg is the sensor's physical limit
    (+/-60.3 deg) — check it is actually reachable and not off by rounding."""
    i = _intr()
    wide = C.CanonicalFrame.from_hfov(120.0, 256, 640)
    _ch, cw = C.ftheta_crop_size_hw(i, wide)
    assert cw <= i.width                              # fits inside 1920 px
    vid = torch.zeros(1, 3, i.height, i.width, dtype=torch.uint8)
    C.ftheta_crop_resize(vid, i, frame=wide)
    got = math.degrees(2 * math.atan((wide.width / 2)
                                     / C.ftheta_crop_resize.last_f_eff))
    assert abs(got - 120.0) < 0.5, got


def test_per_corpus_geometry_is_expressible():
    """'PhysicalAI at 100-120 deg, comma2k19 at its own 65.2 deg' — option (a).
    Two different frames, two different cache keys, one program."""
    from tanitad.data.physicalai import geometry_build_params
    pai = C.CanonicalFrame.from_hfov(110.0, 256, 640, "cylindrical")
    comma = C.CanonicalFrame.from_hfov(C.COMMA2K19_MAX_HFOV_DEG, 256, 400)
    assert pai.hfov_deg > comma.hfov_deg
    assert geometry_build_params(pai, "cylindrical") != geometry_build_params(comma)
    assert round(comma.hfov_deg, 2) == round(C.COMMA2K19_MAX_HFOV_DEG, 2)


def test_letterboxing_comma2k19_is_HONEST_not_a_silent_zoom():
    """Option (b): rendered on the wide frame with the unobservable periphery
    explicitly masked. The crop path would instead CLAMP and silently zoom —
    both behaviours are measured here so the difference is not a matter of
    opinion."""
    wide = C.CanonicalFrame.from_hfov(100.0, 256, 640)
    vid = torch.full((1, 3, 874, 1164), 200, dtype=torch.uint8)
    C.pinhole_rectify(vid, C.COMMA2K19_INTR, frame=wide)
    obs = C.pinhole_rectify.last_observed_frac
    assert 0.0 < obs < 1.0                     # an explicit unobserved band
    C.focal_crop_resize(vid, C.COMMA2K19_FOCAL_PX, 256, frame=wide)
    assert C.focal_crop_resize.last_clamped    # the crop path clamps == zooms
    # ...and the achieved field is NOT the requested one under clamping
    assert C.focal_crop_resize.last_f_eff != wide.f_ref


def test_physicalai_projection_modes_are_registered_and_default_to_the_deployed():
    from tanitad.data import physicalai as P
    assert P.DEFAULT_PROJECTION_MODE == P.PROJECTION_MODE_CROP == "ftheta_crop"
    assert set(P.PROJECTION_MODES) == {"ftheta_crop", "cylindrical"}
    with pytest.raises(ValueError, match="unknown projection_mode"):
        P.geometry_build_params(None, "nope")


def test_comma2k19_geometry_modes_are_registered_and_default_to_the_deployed():
    from tanitad.data import comma2k19 as CM
    assert CM.DEFAULT_GEOMETRY_MODE == CM.GEOMETRY_MODE_CROP == "focal_crop"
    assert set(CM.GEOMETRY_MODES) == {"focal_crop", "rectify"}


def test_build_episode_passes_the_frame_through_to_the_decoder():
    """End-to-end: the frame set on the build reaches the decode function. A
    stale default anywhere on this path makes this test fail."""
    import numpy as np
    import pandas as pd
    from tanitad.data import physicalai as P
    seen: dict = {}
    wide = C.CanonicalFrame(32, 80, f_ref=60.0)

    def fake_decode(mp4, size, frame=None, projection_mode="ftheta_crop"):
        seen["size"], seen["frame"] = size, frame
        seen["projection_mode"] = projection_mode
        h, w = (frame.hw if frame is not None else (size, size))
        return torch.zeros(40, 3, h, w, dtype=torch.uint8)

    clip = {"clip_id": "x" * 8, "mp4": "x.mp4", "timestamps": None,
            "ego_zip": None}
    ts = pd.DataFrame({"timestamp": np.linspace(0, 4e9, 40)})
    ego = pd.DataFrame({
        "timestamp": np.linspace(0, 4e9, 40), "vx": np.ones(40),
        "vy": np.zeros(40), "vz": np.zeros(40), "x": np.linspace(0, 4, 40),
        "y": np.zeros(40), "z": np.zeros(40), "qx": np.zeros(40),
        "qy": np.zeros(40), "qz": np.zeros(40), "qw": np.ones(40),
        "curvature": np.zeros(40), "ax": np.zeros(40)})
    import unittest.mock as mock
    with mock.patch.object(P.pd, "read_parquet", return_value=ts), \
         mock.patch.object(P, "load_egomotion", return_value=ego):
        ep = P.build_episode(clip, size=32, decode_fn=fake_decode, frame=wide,
                             projection_mode="cylindrical")
    assert seen["frame"] == wide
    assert seen["projection_mode"] == "cylindrical"
    assert tuple(ep.frames.shape[-2:]) == (32, 80)


# =========================================================================== #
# 5b. warm-starting a checkpoint into a NEW geometry                           #
# =========================================================================== #
def test_pos_embed_resample_lets_a_square_checkpoint_load_into_a_wide_encoder():
    """The ONE checkpoint-shaped tensor a geometry change touches. Without this
    a 16x16-trained checkpoint cannot load into a 16x40 encoder at all."""
    from tanitad.models.encoder import adapt_pos_embed_, resize_pos_embed
    sq = ViTEncoder(EncoderConfig(in_channels=3, image_size=32, patch_size=8,
                                  d_model=16, depth=1, n_heads=2))
    wide = ViTEncoder(EncoderConfig(in_channels=3, image_size=32, image_width=80,
                                    patch_size=8, d_model=16, depth=1, n_heads=2))
    sd = {k: v.clone() for k, v in sq.state_dict().items()}
    with pytest.raises(RuntimeError):                     # shape mismatch today
        wide.load_state_dict(sd, strict=True)
    adapt_pos_embed_(sd, wide)
    wide.load_state_dict(sd, strict=True)                 # ...and now it loads
    assert wide.pos.shape == (1, 40, 16)
    # a same-geometry load is a NO-OP (byte-identical)
    sd2 = {k: v.clone() for k, v in sq.state_dict().items()}
    adapt_pos_embed_(sd2, sq)
    assert torch.equal(sd2["pos"], sq.state_dict()["pos"])
    # and the resampler is shape-correct + refuses a mismatched old grid
    assert resize_pos_embed(torch.zeros(1, 256, 8), (16, 16), (16, 40)).shape \
        == (1, 640, 8)
    with pytest.raises(ValueError, match="does not match old grid"):
        resize_pos_embed(torch.zeros(1, 100, 8), (16, 16), (16, 40))


# =========================================================================== #
# 6. the data card / report                                                    #
# =========================================================================== #
def test_geometry_report_is_complete_and_self_consistent():
    cfg = flagship4b_config()
    r = geometry_report(cfg)
    assert r["is_canonical"] and r["cache_key_fragment"] == {}
    assert r["token_grid"] == [16, 16] and r["n_tokens"] == 256
    assert r["state_dim"] == 2048
    assert r["exceeds_comma2k19_field"] is False

    apply_frame(cfg, C.CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical"))
    r = geometry_report(cfg)
    assert not r["is_canonical"] and r["cache_key_fragment"] != {}
    assert r["token_grid"] == [16, 40] and r["n_tokens"] == 640
    assert r["state_dim"] == 2048                  # unchanged — the firewall
    assert round(r["hfov_deg"]) == 120
    assert r["exceeds_comma2k19_field"] is True


def test_geometry_literals_agree_with_the_canonical_frame_TODAY():
    """⭐ THE DEFAULT-FLIP CHECKLIST, EXECUTED. Four `CORPUS_META` dicts and the
    lake schema hardcode 256 / 266.0 and are reached by no config. This asserts
    they currently agree with `CANONICAL_256`."""
    from tanitad.geometry import assert_geometry_literals_consistent
    rows = assert_geometry_literals_consistent()
    assert rows and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert len(rows) >= 9                      # 4 corpora x 2 + the lake schema


def test_flipping_the_frame_WITHOUT_the_literals_FAILS_and_names_the_files():
    """⭐ The load-bearing half: it must FAIL when the frame moves and the
    literals do not, and the message must name every file to change."""
    from tanitad.geometry import assert_geometry_literals_consistent
    wide = C.CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
    with pytest.raises(GeometryMismatch) as ei:
        assert_geometry_literals_consistent(wide)
    msg = str(ei.value)
    assert "GEOMETRY LITERALS ARE STALE" in msg
    for mod in ("physicalai", "comma2k19", "cosmos_drive", "l2d"):
        assert mod in msg, mod
    assert "lake.schema" in msg
    assert "I7 task-identity" in msg           # says WHY it must be declared


def test_frame_roundtrips_through_json():
    import json
    for f in (C.CANONICAL_256, C.CanonicalFrame.from_hfov(120.0, 384, 960,
                                                          "cylindrical")):
        assert C.CanonicalFrame.from_dict(json.loads(json.dumps(f.to_dict()))) == f
        assert f.tag()


def test_stackconfig_with_a_geometry_still_serializes():
    cfg = flagship4b_config()
    apply_frame(cfg, C.CanonicalFrame.from_hfov(100.0, 256, 640))
    import json
    d = json.loads(cfg.to_json())
    assert d["geometry"]["width"] == 640
    assert d["encoder"]["image_width"] == 640
    assert d["encoder"]["image_size"] == 256
