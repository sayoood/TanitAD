"""refcv6 — nothing in the perception branch may pin a geometry or a trunk.

⛔ **PI, 2026-09-16** (after `SPEC_REFCV6_V2.md`): the input becomes
**256 x 1024** cylindrical at the same 120 deg field, and the trunk becomes
**resnet101** with **resnet34** as a comparison run. So:

* the stride-16 map is **16 x 64 = 1,024** cells (1.875 deg/column) at 1024 and
  16 x 40 = 640 (3.0 deg) at 640 -- nothing may hard-code 40, 640, 160 or 20;
* the stride-16 channel count is **256** (resnet34) or **1024** (resnet101) --
  it comes from `timm.feature_info`, never a literal;
* ⭐ the SAM3 label grid is **UNCHANGED** at 120 x 64 @ 0.5 m, because it is
  METRIC. The lift is simply better sampled.

These are **synthetic-tensor** checks at both geometries: at the time of writing
only the 256 x 640 cache exists on this box (the 256 x 1024 rebuild of the 139
eval clips is another agent's job). The real-data checks in
``test_refcv6_perception_realdata.py`` therefore run at **640**, and this file
is what says the code is ready for **1024**.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import tanitad.models.bev_lift as L
from tanitad.data.semantic_map_gt import CART_SHAPE
from tanitad.models.bev_encoder import BEVEncoderConfig, BEVMapBranch, map_soft_ce
from tanitad.models.bev_lift import HEIGHTS_M, BEVLift
from tanitad.models.box3d_head import Box3DMemory, Box3DSlotDecoder
from tanitad.models.trunk_shapes import (
    FRAME_256x640, FRAME_256x1024, PERCEPTION_STRIDE, PLANNER_STRIDE, TrunkSpec,
    assert_label_grid_unmoved, feature_hw, frame_for_width,
)

TRUNKS = ("resnet34", "resnet101")
GEOMETRIES = ((FRAME_256x640, "256x640"), (FRAME_256x1024, "256x1024"))
STACK = Path(__file__).resolve().parents[1] / "tanitad"


def _timm_or_skip():
    try:
        import timm            # noqa: F401
    except ImportError:
        pytest.skip("timm is not installed")


# =========================================================================== #
# 1. the shapes themselves                                                    #
# =========================================================================== #
def test_the_two_geometries_have_the_field_and_the_columns_the_PI_named():
    import math
    assert (FRAME_256x640.height, FRAME_256x640.width) == (256, 640)
    assert (FRAME_256x1024.height, FRAME_256x1024.width) == (256, 1024)
    # the FIELD is a property of the lens and must be invariant
    for f in (FRAME_256x640, FRAME_256x1024):
        hfov = math.degrees(2.0 * (f.width / 2.0) / f.f_ref)
        assert hfov == pytest.approx(120.0, abs=1e-6), f"{f.width}: {hfov} deg"
    assert feature_hw(FRAME_256x640, PERCEPTION_STRIDE) == (16, 40)
    assert feature_hw(FRAME_256x1024, PERCEPTION_STRIDE) == (16, 64)
    assert feature_hw(FRAME_256x640, PLANNER_STRIDE) == (8, 20)
    assert feature_hw(FRAME_256x1024, PLANNER_STRIDE) == (8, 32)
    # 1.875 deg/column at 1024, 3.000 at 640 -- the quantity the oracle study
    # found load-bearing (cutting azimuth alone costs 57 % of AP)
    assert math.degrees(16.0 / FRAME_256x1024.f_ref) == pytest.approx(1.875, abs=1e-3)
    assert math.degrees(16.0 / FRAME_256x640.f_ref) == pytest.approx(3.000, abs=1e-3)


def test_the_PI_rounded_f_ref_and_the_exact_one_are_reported_not_conflated():
    """⚠️ The PI wrote `f_ref 488.92`; the exact 1024/640 scaling is
    488.92398517830253. The rounding moves the field by 0.0010 deg. The exact
    value is used; the difference is recorded rather than silently adopted."""
    import math
    exact = FRAME_256x1024.f_ref
    assert exact == pytest.approx(488.92398517830253, abs=1e-9)
    rounded_hfov = math.degrees(2.0 * 512.0 / 488.92)
    assert rounded_hfov == pytest.approx(120.0010, abs=1e-4)
    # at the frame edge the disagreement is well under a pixel
    px = 512.0 - 488.92 * (512.0 / exact)
    assert abs(px) < 0.05, f"{px:.4f} px"


def test_trunk_channels_come_from_feature_info_for_both_trunks():
    _timm_or_skip()
    want = {"resnet34": (256, 512), "resnet101": (1024, 2048)}
    for frame, tag in GEOMETRIES:
        for trunk in TRUNKS:
            spec = TrunkSpec.from_timm(trunk, frame)
            assert (spec.perception.channels, spec.planner.channels) == want[trunk], \
                f"{trunk} @ {tag}"
            assert spec.perception.hw == feature_hw(frame, PERCEPTION_STRIDE)
            assert spec.source == f"timm:{trunk}"


def test_a_backbone_without_feature_info_is_refused_not_guessed():
    class Bare(torch.nn.Module):
        pass
    with pytest.raises(ValueError, match="refuses to guess"):
        TrunkSpec.from_timm.__func__(TrunkSpec, Bare(), FRAME_256x1024)


def test_frame_for_width_keeps_the_field():
    import math
    for w in (640, 1024, 1280):
        f = frame_for_width(w)
        assert math.degrees(2.0 * (w / 2.0) / f.f_ref) == pytest.approx(120.0, abs=1e-6)


def test_MUT_the_label_grid_must_not_move_with_the_image():
    """⭐ The SAM3 grid is METRIC. MUTATION: a resolution change that drags the
    label grid with it — then the prediction and the label are on different
    cells and nothing downstream can tell."""
    assert assert_label_grid_unmoved() == (120, 64) == CART_SHAPE
    with pytest.raises(RuntimeError, match="does not move with"):
        assert_label_grid_unmoved((120, 102))          # 64 * 1024/640


# =========================================================================== #
# 2. the lift at both geometries (synthetic feature maps)                     #
# =========================================================================== #
@pytest.mark.parametrize("frame,tag", GEOMETRIES)
@pytest.mark.parametrize("trunk", TRUNKS)
def test_lift_and_map_branch_build_and_run_at_every_combination(frame, tag, trunk):
    _timm_or_skip()
    spec = TrunkSpec.from_timm(trunk, frame)
    p = spec.perception
    extr = {"qx": 0.0, "qy": 0.02, "qz": 0.0, "qw": 0.9998,
            "x": 1.5, "y": 0.0, "z": 1.4}
    geo = L.build_lift_geometry(extr, frame=frame, stride=PERCEPTION_STRIDE)
    assert geo.feat_hw == p.hw, f"{tag}: lift built {geo.feat_hw}, trunk gives {p.hw}"
    assert geo.grid.shape == (len(HEIGHTS_M),) + CART_SHAPE + (2,)
    lift = BEVLift(d_in=p.channels, d_out=32, n_heights=len(HEIGHTS_M),
                   feat_hw=p.hw)
    fmap = torch.randn(2, p.channels, *p.hw)
    bev = lift(fmap, geo.grid.unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous(),
               geo.valid.unsqueeze(0).expand(2, -1, -1, -1).contiguous())
    assert tuple(bev.shape) == (2, 32) + CART_SHAPE, (
        f"{tag}/{trunk}: the BEV output moved with the image resolution")
    br = BEVMapBranch(BEVEncoderConfig(d_in=32, d_model=32, d_out=32,
                                       dilations=(1, 2), norm_groups=8))
    out = br(bev)
    assert tuple(out["map_logits"].shape) == (2, 9) + CART_SHAPE


@pytest.mark.parametrize("frame,tag", GEOMETRIES)
@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_box_head_builds_and_runs_at_every_combination(frame, tag, trunk):
    _timm_or_skip()
    p = TrunkSpec.from_timm(trunk, frame).perception
    mem = Box3DMemory(d_image=p.channels, d_bev=48, d_model=64, image_hw=p.hw)
    assert mem.n_tokens == p.n_tokens + 30 * 16
    dec = Box3DSlotDecoder(d_memory=64, n_memory=mem.n_tokens, n_queries=8,
                           d_model=64, depth=1, n_heads=4, enforce_band=False)
    tok = mem(torch.randn(2, p.channels, *p.hw), torch.randn(2, 48, *CART_SHAPE))
    assert tuple(tok.shape) == (2, mem.n_tokens, 64)
    slots = dec(tok)
    assert tuple(slots["box3d"].shape) == (2, 8, 7)


def test_MUT_a_head_built_for_one_geometry_refuses_the_other():
    """MUTATION: the pinned-geometry defect. A head built on the 640 cache fed a
    1024 map (and the reverse) must REFUSE — the positional table is per token,
    so this is a geometry mismatch, not a resize."""
    _timm_or_skip()
    a = TrunkSpec.from_timm("resnet34", FRAME_256x640).perception
    b = TrunkSpec.from_timm("resnet34", FRAME_256x1024).perception
    mem = Box3DMemory(d_image=a.channels, d_bev=8, d_model=16, image_hw=a.hw)
    bev = torch.randn(1, 8, *CART_SHAPE)
    mem(torch.randn(1, a.channels, *a.hw), bev)                    # green
    with pytest.raises(ValueError, match="geometry mismatch, not a resize"):
        mem(torch.randn(1, b.channels, *b.hw), bev)                # RED
    # and the reverse direction
    mem2 = Box3DMemory(d_image=b.channels, d_bev=8, d_model=16, image_hw=b.hw)
    with pytest.raises(ValueError, match="geometry mismatch, not a resize"):
        mem2(torch.randn(1, b.channels, *a.hw), bev)


def test_MUT_a_head_built_for_one_trunk_refuses_the_other_s_channels():
    """MUTATION: the pinned-channel defect. resnet34 gives 256 channels at
    stride 16 and resnet101 gives 1024; a head that hard-codes either one
    mis-projects the other (or runs through a silently-inserted adapter, and the
    ablation then measures the adapter)."""
    _timm_or_skip()
    r34 = TrunkSpec.from_timm("resnet34", FRAME_256x1024).perception
    r101 = TrunkSpec.from_timm("resnet101", FRAME_256x1024).perception
    assert r34.channels != r101.channels and r34.hw == r101.hw
    mem = Box3DMemory(d_image=r34.channels, d_bev=8, d_model=16, image_hw=r34.hw)
    bev = torch.randn(1, 8, *CART_SHAPE)
    mem(torch.randn(1, r34.channels, *r34.hw), bev)                # green
    with pytest.raises(ValueError, match="channel count is a PARAMETER"):
        mem(torch.randn(1, r101.channels, *r101.hw), bev)          # RED
    # the lift too
    lift = BEVLift(d_in=r34.channels, d_out=16, n_heights=len(HEIGHTS_M),
                   feat_hw=r34.hw)
    with pytest.raises(ValueError, match="channels"):
        geo = L.build_lift_geometry(
            {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 0.0, "y": 0.0,
             "z": 0.0}, frame=FRAME_256x1024, stride=PERCEPTION_STRIDE)
        lift(torch.randn(1, r101.channels, *r101.hw),
             geo.grid.unsqueeze(0), geo.valid.unsqueeze(0))


def test_the_stride_32_map_is_still_named_when_it_is_handed_over():
    """The halved map must still be diagnosed as the STRIDE-32 map, at BOTH
    geometries — the message that stops someone wiring `layer4` by mistake."""
    _timm_or_skip()
    for frame in (FRAME_256x640, FRAME_256x1024):
        spec = TrunkSpec.from_timm("resnet101", frame)
        mem = Box3DMemory(d_image=spec.planner.channels, d_bev=8, d_model=16,
                          image_hw=spec.perception.hw)
        with pytest.raises(ValueError, match="STRIDE-32 map"):
            mem(torch.randn(1, spec.planner.channels, *spec.planner.hw),
                torch.randn(1, 8, *CART_SHAPE))


# =========================================================================== #
# 3. the defect must not be able to come back                                 #
# =========================================================================== #
#: the modules this branch owns; `bev_lift`/`semantic_map_gt` predate the PI's
#: 2026-09-16 geometry ruling and carry 640-era defaults this branch overrides
#: at every call site, so they are listed separately below.
OWNED = ("models/bev_encoder.py", "models/box3d_head.py",
         "models/refc_bev_coupling.py", "models/trunk_shapes.py",
         "models/refcv6_perception_branch.py",     # the trainer's assembly
         "data/agent_cuboid_gt.py", "data/lift_orientation.py",
         "data/perception_targets.py")


#: image-geometry numbers that must never be a literal in executable code.
#: ⚠️ NOT 16 or 64: 16 is the perception STRIDE (a spec constant, not a
#: geometry) and 64 is the SAM3 label grid's own width, which is METRIC and does
#: not move. The banned set is exactly the numbers that change with the image.
BANNED_NUMBERS = {640, 1024, 1504, 40, 20, 160}


#: an exemption must be WRITTEN DOWN on the line it applies to, with a reason.
EXEMPT_MARK = "geometry-exempt:"


def _code_numbers(path: Path, honour_exemptions: bool = True) -> list:
    """Every NUMBER token of a source file, with its line.

    ⭐ Tokenised, not grepped. A line scan cannot tell a docstring from code --
    it flagged the PI's own geometry prose in ``box3d_head.py``'s docstring on
    the first run. ``tokenize`` emits STRING and COMMENT as their own token
    types, so only real literals survive here.

    A line carrying ``# geometry-exempt: <reason>`` is skipped. ⛔ That marker is
    the ONLY way to keep such a literal, and it costs a written reason -- the
    one in ``lift_orientation.py`` is an AUC sample-size floor that happens to
    be 20, which is also the stride-32 width at 640. An exemption that is not
    visible in the source is not an exemption, it is a hole.
    """
    import tokenize
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    with tokenize.open(str(path)) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type != tokenize.NUMBER:
                continue
            ln = tok.start[0]
            if honour_exemptions and EXEMPT_MARK in lines[ln - 1]:
                continue
            try:
                out.append((ln, int(tok.string)))
            except ValueError:
                continue
    return out


def test_no_owned_module_pins_an_image_geometry_in_CODE():
    """⛔ No module this branch owns may contain an image-geometry literal in
    executable code. Prose and docstrings may (and must) name 16x40 and 16x64 --
    the point is that no DEFAULT, comparison or shape depends on one.

    ``trunk_shapes.py`` is the ONE place a geometry may be written down, and it
    is excluded by name so that the exception is visible rather than implicit.
    """
    offenders = []
    for rel in OWNED:
        if rel == "models/trunk_shapes.py":
            continue
        p = STACK / rel
        if not p.is_file():
            continue
        for line, val in _code_numbers(p):
            if val in BANNED_NUMBERS:
                offenders.append(f"{rel}:{line}: literal {val}")
    assert not offenders, (
        "an image geometry is pinned in executable code:\n  "
        + "\n  ".join(offenders))


def test_MUT_the_guard_catches_a_reintroduced_literal(tmp_path):
    """MUTATION: re-introduce the defect — a geometry literal in real code —
    and show the guard fires. An unproven guard is a comment."""
    good = tmp_path / "clean.py"
    good.write_text(
        '"""Doc prose may say 16 x 40 on the 256 x 640 cache and 16 x 64."""\n'
        "# and so may a comment: 1024 tokens\n"
        "HW = spec.perception.hw\n"
        "STRIDE = 16\n"
        "LABEL_W = 64\n", encoding="utf-8")
    assert [v for _, v in _code_numbers(good) if v in BANNED_NUMBERS] == []
    bad = tmp_path / "pinned.py"
    bad.write_text(
        '"""Same prose, 256 x 640."""\n'
        "IMAGE_HW = (16, 40)\n"
        "N_TOKENS = 640\n"
        "def f(x):\n    return x[:, :, :, :1024]\n", encoding="utf-8")
    hits = sorted(v for _, v in _code_numbers(bad) if v in BANNED_NUMBERS)
    assert hits == [40, 640, 1024], hits
    _ = re


def test_MUT_an_exemption_must_be_written_down_to_count():
    """MUTATION: the exemption marker itself. Without it the literal is caught;
    with it, it is skipped — and the SAME file scanned with exemptions off still
    shows the number, so a reader can always see what was excused."""
    p = STACK / "data" / "lift_orientation.py"
    with_ex = [v for _, v in _code_numbers(p) if v in BANNED_NUMBERS]
    without = [v for _, v in _code_numbers(p, honour_exemptions=False)
               if v in BANNED_NUMBERS]
    assert with_ex == [], with_ex
    assert without == [20], without      # the AUC floor, and nothing else
    src = p.read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if EXEMPT_MARK in ln)
    reason = line.split(EXEMPT_MARK, 1)[1].strip()
    assert len(reason) > 10, f"an exemption with no reason: {line!r}"
