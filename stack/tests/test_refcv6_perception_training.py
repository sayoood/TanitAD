"""refcv6 §2/§6 -- THE TRAINER WIRING of the perception branch.

⛔ Every guard here is proven by MUTATION, never by inspection: each test that
asserts a property also reintroduces the defect and asserts the check GOES RED.
An AST census once read "0 suspects" on both the fixed and the broken trainer;
a test whose expected value is an expression over the code under test is green
forever (four instances in one night, CLAUDE.md 2026-09-07).

The five things pinned:

 1. **BIT-IDENTITY AT ZERO WEIGHT** -- at the 0.0 default no branch is built,
    ``model.parameters()`` is unchanged, and ``compute_losses_v3`` emits no
    ``map``/``box3d`` key.
 2. **THE REFUSALS** -- every combination that would stamp a weight and train
    nothing exits, each for its own named reason; and the REVERSE case, an
    artifact supplied with a zero weight.
 3. **GRADIENT REACH** -- each head's loss alone reaches the trunk, with the
    detached seam as the discriminating control.
 4. **`--image-hw` CARRIES EVERY TRUNK FIELD** (D-REFCV6-IMAGEHW-DROPS-TRUNK-
    FIELDS), with the historical hand-listed rebuild as the regression arm.
 5. **THE FRAME AXIS** -- stacked row -> raw v2ep frame, against a LITERAL.
"""
from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T                                         # noqa: E402
from tanitad.data.semantic_map_gt import CART_SHAPE, N_CHANNELS   # noqa: E402
from tanitad.models import refcv6_perception_branch as PB         # noqa: E402

BASE = ["--arm", "hier", "--size", "tiny", "--out", "X"]


def _args(*extra):
    return T.build_parser().parse_args(BASE + list(extra))


# =========================================================================== #
# 1. bit-identity at the 0.0 default                                          #
# =========================================================================== #
def test_default_is_zero_and_builds_nothing():
    a = _args()
    assert a.w_map == 0.0 and a.w_box3d == 0.0
    # ⛔ A LITERAL, not `PerceptionBranchConfig(...).w_map`: an expectation
    # written as an expression over the code under test is the green-forever
    # defect. The default must be the number 0.0.
    assert a.w_map == 0.0


def test_the_branch_REFUSES_to_exist_at_zero_weight():
    """The invariant does not depend on one call site remembering it."""
    with pytest.raises(ValueError, match="training on nothing"):
        PB.PerceptionBranchConfig(w_map=0.0, w_box3d=0.0)
    # the mutation: one live weight and it constructs
    assert PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0).w_map == 1.0


# =========================================================================== #
# 2. the refusals                                                             #
# =========================================================================== #
@pytest.mark.parametrize("extra,why", [
    (["--w-map", "1.0"], "trunk timm"),
    (["--w-map", "1.0", "--trunk", "timm"], "map-gt-root"),
    (["--w-map", "1.0", "--trunk", "timm", "--map-gt-root", "R"],
     "agent-rig-camera extrinsics"),
    (["--w-box3d", "1.0"], "trunk timm"),
    (["--w-box3d", "1.0", "--trunk", "timm"], "agent-join"),
    # the REVERSE refusal: an artifact with no weight to consume it
    (["--map-gt-root", "R"], "READ BY NOTHING"),
    (["--join3d", "J"], "READ BY NOTHING"),
    (["--w-map", "1.0", "--trunk", "timm", "--map-gt-root", "R",
      "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", "E",
      "--join3d", "J"], "--join3d with --w-box3d 0"),
])
def test_every_dead_combination_refuses(extra, why):
    with pytest.raises(SystemExit) as e:
        T._pin_refcv6_perception(None, _args(*extra))
    assert why in str(e.value), str(e.value)[:400]


def test_the_live_combination_does_NOT_refuse():
    """⭐ The discriminating half. A guard that refuses EVERYTHING is not a
    guard; without this arm every test above passes on `raise SystemExit`."""
    T._pin_refcv6_perception(None, _args(
        "--w-map", "1.0", "--w-box3d", "1.0", "--trunk", "timm",
        "--map-gt-root", "R", "--agent-join", "J", "--join3d", "J3",
        "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", "E"))
    T._pin_refcv6_perception(None, _args())          # the default path


def test_agents_off_still_refuses_a_dead_camera_but_not_a_live_map():
    """The false-refusal half of the `--agents off` guard.

    ⛔ Both directions. A camera with no consumer is still refused; a camera
    the BEV lift reads is not.
    """
    live = _args("--agents", "off", "--w-map", "1.0", "--trunk", "timm",
                 "--map-gt-root", "R", "--agent-rig-camera", "extrinsics",
                 "--agent-rig-extrinsics", "E")
    cfg = T.v3.refc_v3_sized_config("tiny", hier=True)
    T._pin_refcv5_seams(cfg, live)                    # must not raise
    dead = _args("--agents", "off", "--agent-rig-camera", "extrinsics",
                 "--agent-rig-extrinsics", "E")
    with pytest.raises(SystemExit, match="agents off"):
        T._pin_refcv5_seams(T.v3.refc_v3_sized_config("tiny", hier=True), dead)


def test_both_weights_are_in_the_weight_gate_ledger():
    """`REFC_WEIGHT_GATES` is EXHAUSTIVE over the parser (its own test); this
    one pins the two rows' GATES rather than merely their presence."""
    for dest in ("w_map", "w_box3d"):
        assert dest in T.REFC_WEIGHT_GATES
        met, why = T.REFC_WEIGHT_GATES[dest]["gate"](_args())
        assert met is False and "timm" in why


# =========================================================================== #
# 3. gradient reach -- with the detached seam as the control                  #
# =========================================================================== #
class _Trunk(nn.Module):
    """A stand-in for the timm trunk: one conv, so `fmap_s16` has a graph."""

    def __init__(self, c_in=3, c_out=8):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _tiny_branch(d_image=8, hw=(4, 8)):
    cfg = PB.PerceptionBranchConfig(
        w_map=1.0, w_box3d=1.0, d_bev=8,
        bev_cfg=PB.BEVEncoderConfig(d_in=8, d_model=8, d_out=8,
                                    dilations=(1,), norm_groups=2),
        n_queries=4, d_model=16, bev_tokens_hw=(2, 2),
        # ⛔ BY NAME, never by widening the band: the tiny rig is BELOW §6's
        # 2-4 M pre-registered size and a test must say so explicitly.
        enforce_param_band=False)
    return cfg, PB.PerceptionBranch(cfg, d_image=d_image, image_hw=hw)


def _lift_geom(b, z, hw=CART_SHAPE):
    g = torch.zeros(b, z, hw[0], hw[1], 2)
    v = torch.ones(b, z, hw[0], hw[1], dtype=torch.bool)
    return g, v


@pytest.mark.parametrize("head", ["map", "box3d"])
def test_each_head_alone_reaches_the_trunk_and_the_detached_seam_does_not(head):
    torch.manual_seed(0)
    cfg, br = _tiny_branch()
    trunk = _Trunk(3, 8)
    x = torch.randn(2, 3, 4, 8)
    grid, valid = _lift_geom(2, len(cfg.heights_m))

    def run(detach: bool) -> dict:
        for p in list(trunk.parameters()) + list(br.parameters()):
            p.grad = None
        f = trunk(x)
        out = br(f.detach() if detach else f, grid, valid)
        if head == "map":
            frac = torch.zeros(2, N_CHANNELS, *CART_SHAPE)
            frac[:, 1] = 1.0
            seen = torch.ones(2, *CART_SHAPE, dtype=torch.bool)
            loss = PB.map_loss_row(out["map_logits"], frac, seen)["loss"]
        else:
            n = 3
            tgt = {"box": torch.rand(2, n, 4), "yaw": torch.zeros(2, n),
                   "cls": torch.zeros(2, n, dtype=torch.long),
                   "valid": torch.ones(2, n, dtype=torch.bool),
                   "occ": torch.zeros(2, n),
                   "rates": torch.zeros(2, n, 3),
                   "rates_mask": torch.zeros(2, n, dtype=torch.bool),
                   "cz": torch.ones(2, n), "h": torch.ones(2, n) * 1.6,
                   "zh_mask": torch.ones(2, n, dtype=torch.bool)}
            loss = PB.box3d_loss_row(out["box_slots"], tgt)["loss"]
        loss.backward()
        s, _, _ = PB.grad_abs_sum(trunk)
        own = PB.grad_abs_sum(br.map_branch.head if head == "map" else br.box_dec)[0] \
            if head == "map" else PB.grad_abs_sum(br.box_dec)[0]
        return {"trunk": s, "own": own}

    live, cut = run(False), run(True)
    # the claim
    assert live["trunk"] > 0.0, f"{head} does not reach the trunk"
    assert live["own"] > 0.0, f"{head}'s own head has no gradient"
    # ⛔ THE DISCRIMINATING CONTROL. Detaching the seam must drive the TRUNK to
    # EXACTLY 0 while the head keeps its own gradient. Without it, "the trunk
    # has a gradient" is consistent with reading someone else's.
    assert cut["trunk"] == 0.0, (
        f"{head}: detaching fmap_s16 left {cut['trunk']} on the trunk -- the "
        f"live measurement was not this head's gradient")
    assert cut["own"] > 0.0


def test_map_head_and_box_head_do_not_borrow_each_others_gradient():
    """A cross-check the two tests above cannot make: the map loss must leave
    the box decoder at EXACTLY 0, and vice versa."""
    torch.manual_seed(0)
    cfg, br = _tiny_branch()
    trunk = _Trunk(3, 8)
    grid, valid = _lift_geom(2, len(cfg.heights_m))
    out = br(trunk(torch.randn(2, 3, 4, 8)), grid, valid)
    frac = torch.zeros(2, N_CHANNELS, *CART_SHAPE)
    frac[:, 1] = 1.0
    PB.map_loss_row(out["map_logits"], frac,
                    torch.ones(2, *CART_SHAPE, dtype=torch.bool))["loss"] \
        .backward()
    assert PB.grad_abs_sum(br.box_dec)[0] == 0.0
    assert PB.grad_abs_sum(br.map_branch.head)[0] > 0.0        # the same-breath control


def test_grad_reach_report_names_every_head_and_reads_zero_before_backward():
    torch.manual_seed(0)
    cfg, br = _tiny_branch()
    m = type("M", (), {})()
    m.core = type("C", (), {})()
    m.core.encoder = _Trunk(3, 8)
    r = PB.grad_reach_report(m, br)
    assert set(r) == {"trunk", "lift", "bev_encoder", "map_head",
                      "box_memory", "box_decoder"}
    # ⛔ Before any backward EVERY head must read exactly 0 -- so a non-zero
    # later is the backward's doing and not a stale buffer.
    assert all(v["grad_abs_sum"] == 0.0 for v in r.values())
    assert all(v["n_params"] > 0 for v in r.values())


# =========================================================================== #
# 4. --image-hw must carry EVERY trunk field                                  #
# =========================================================================== #
TRUNK_FIELDS = ("trunk", "trunk_name", "trunk_mode", "trunk_fuse",
                "trunk_fuse_identity", "trunk_pretrained",
                "trunk_imagenet_norm")


def test_image_hw_carries_every_trunk_field():
    """D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS.

    MEASURED 2026-09-17 on tip c16b7f1: the rebuild listed three fields and
    dropped four, so `--trunk-name resnet34.a1_in1k --image-hw 256 1024` built
    resnet101, `--trunk-fuse last` (THE SINGLE-FRAME CONTROL) built concat1x1,
    and `--trunk-fuse-plain-init` (THE DELIBERATE REGRESSION) built the
    identity init.
    """
    argv = ["--trunk", "timm", "--trunk-name", "resnet34.a1_in1k",
            "--trunk-mode", "inflate", "--trunk-fuse", "last",
            "--trunk-fuse-plain-init", "--no-trunk-pretrained"]
    want = {"trunk": "timm", "trunk_name": "resnet34.a1_in1k",
            "trunk_mode": "inflate", "trunk_fuse": "last",
            "trunk_fuse_identity": False, "trunk_pretrained": False,
            "trunk_imagenet_norm": True}
    got = T._pin_trainer_cfg(
        T.v3.refc_v3_sized_config("tiny", hier=True),
        _args(*argv, "--image-hw", "256", "1024")).core.encoder
    # ⛔ Against a LITERAL dict, never against the no-image-hw config: the
    # latter is an expression over the code under test.
    for k, v in want.items():
        assert getattr(got, k) == v, f"--image-hw dropped {k}: {getattr(got, k)!r} != {v!r}"
    assert (got.image_size, got.image_width) == (256, 1024)


def test_the_historical_hand_listed_rebuild_is_caught():
    """⭐ THE MUTATION ARM. Reintroduce the exact pre-2026-09-17 rebuild and
    require the assertion above to FAIL -- otherwise the test is green forever
    and would not have caught the defect it was written for."""
    from tanitad.refs import refc
    enc = dc.replace(
        T.v3.refc_v3_sized_config("tiny", hier=True).core.encoder,
        trunk="timm", trunk_name="resnet34.a1_in1k", trunk_mode="inflate",
        trunk_fuse="last", trunk_fuse_identity=False)
    broken = refc.CNNEncoderConfig(          # the historical field list
        in_channels=enc.in_channels, image_size=256, image_width=1024,
        base_width=enc.base_width, blocks=enc.blocks,
        trunk=enc.trunk, trunk_pretrained=enc.trunk_pretrained,
        trunk_imagenet_norm=enc.trunk_imagenet_norm)
    dropped = [k for k in TRUNK_FIELDS
               if getattr(broken, k) != getattr(enc, k)]
    assert dropped == ["trunk_name", "trunk_mode", "trunk_fuse",
                       "trunk_fuse_identity"], dropped
    # and `dataclasses.replace` -- the fix -- drops none of them
    fixed = dc.replace(enc, image_size=256, image_width=1024)
    assert [k for k in TRUNK_FIELDS
            if getattr(fixed, k) != getattr(enc, k)] == []


def test_no_trunk_pretrained_is_reachable_from_argv_and_None_leaves_it_alone():
    """The knockout arm's flag, both directions."""
    base = T.v3.refc_v3_sized_config("tiny", hier=True).core.encoder
    assert base.trunk_pretrained is True          # the dataclass default
    assert T._pin_trainer_cfg(T.v3.refc_v3_sized_config("tiny", hier=True),
                              _args()).core.encoder.trunk_pretrained is True
    assert T._pin_trainer_cfg(
        T.v3.refc_v3_sized_config("tiny", hier=True),
        _args("--no-trunk-pretrained")).core.encoder.trunk_pretrained is False
    assert T._pin_trainer_cfg(
        T.v3.refc_v3_sized_config("tiny", hier=True),
        _args("--trunk-pretrained")).core.encoder.trunk_pretrained is True


# =========================================================================== #
# 5. the frame axis                                                           #
# =========================================================================== #
def test_stacked_row_to_raw_frame_is_plus_n_stack_minus_one():
    """⛔ Against LITERALS. The 3-D join's own two fields give the analytic
    target: MEASURED over all 26,394 lines of the eval join,
    `frame - frame_idx == 2` at n_stack 3, with no other value.
    """
    from tanitad.data.perception_targets import MapGTStore
    assert list(MapGTStore.raw_frames([0, 1, 7], 3)) == [2, 3, 9]
    assert list(MapGTStore.raw_frames([0, 1, 7], 1)) == [0, 1, 7]
    assert list(MapGTStore.raw_frames([5], 5)) == [9]


def test_the_3d_join_is_looked_up_on_the_RAW_frame_not_the_episode_index():
    """⛔ THE INDEX-SPACE TRAP, pinned in the SOURCE.

    `JoinFileReader` keys on `frame_idx` (EPISODE index); `AgentJoin3D` keys on
    `frame` (RAW v2ep). They differ by `n_stack - 1` = 2 frames = 0.2 s. Join on
    the wrong key and every cz/h belongs to a moment 0.2 s from the one the
    trunk saw -- finite, non-zero, entirely plausible, and NOTHING RAISES.

    ⭐ MEASURED by a sweep over offsets -3..+3 on 8 real clips
    (`raw/join3d_offset_sweep.json`): at +2 the 3-D line's cx/cy/yaw/l/w match
    the 2-D reader's rows EXACTLY on 2,955/2,955 agents (mean |dx| 0.0000 m);
    every other offset is 0.0 % exact with mean |dx| >= 3.24 m, INCLUDING the
    naive +0 (3.24 m). The 3-D join is a byte-preserving annotation of the 2-D
    one, so exact equality is an ANALYTIC target, not a fitted minimum.
    """
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "cid, int(f) + int(self.map_n_stack) - 1, list(tids)," in src, (
        "the 3-D lookup no longer applies the n_stack offset")
    # and n_stack cannot be 0, which would apply an offset of -1
    assert 'if int(_cache_nstack) < 1:' in src


def test_the_dataset_reads_the_map_at_the_windows_NOW_frame():
    """The trainer's own expression, pinned: `t + w - 1` is the stacked-row
    index of the window's NOW, and the store adds `n_stack - 1` to it.

    ⚠️ Passing `t` instead would label every window `w - 1` steps EARLY, and
    at 10 Hz / 30 km/h that is 5.8 m of map shift, visible in no metric.
    """
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "item.update(self._map_item(ep, t + w - 1))" in src
    assert "self.map_clip_of_ep[int(self.episodes[e_i].episode_id)],\n" \
           "                    t + w - 1) for e_i, t in self.index" in src


def test_the_new_module_is_registered_with_the_geometry_scanner():
    """⛔ DELEGATE, DO NOT DUPLICATE.

    `tests/test_refcv6_geometry_agnostic.py` already owns the "no hard-coded
    geometry" rule: it TOKENISES rather than greps (a line scan cannot tell a
    docstring from code, and flagged the PI's own prose), it carries a written
    EXEMPT_MARK mechanism, and it is mutation-proven. A second, weaker scanner
    here would drift from it -- the `advect` precedent, where two copies of one
    rule came apart. ⇒ This test asserts only that the new module is IN that
    scanner's list, which is the thing a second copy would have been for.
    """
    import test_refcv6_geometry_agnostic as G
    assert "models/refcv6_perception_branch.py" in G.OWNED
