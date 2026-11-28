"""refcv9 C2 -- ``--r9-map-own-enc`` (arm M3): the map's OWN BEV encoder + the shared lift widened (CPU, a tiny REAL RefCV3Model: resnet18 @ 64x64,
the 10 cm branch, the 10 m near lift, the box3d perception branch on the shared 0.25 m BEV; d_lift 16 -> d_map_enc 32 stands for 64 -> 128).

The class this file exists for is "built, tested, and UNREACHABLE FROM ITS CALLER" (CLAUDE.md 2026-09-10/11), so it traces the flag to the
parameters, and it pins the WARM-START RULE: at step 0 the box / planner path equals the un-widened (M2) model.

  1. OFF builds nothing new; the config refuses what it cannot build; the REAL-size parameter counts are MEASURED here;
  2. fresh init: every shared tensor equals the M2 build's (the extra draws are forked), the widened ones agree on the shared slice;
  3. STEP 0: with the pinned warm start the box slots AND the planner outputs equal the M2 model's; the RED arm (non-zero stem input
     channels) FAILS that check, and so does a warm start that skips the zero-extend;
  4. a NON-ZERO gradient on every new parameter, the CONTROL that reads exactly 0.0, the isolation (map loss never reaches ``encoder``, box
     loss never reaches ``map_encoder``) and the RED arm that detaches the own encoder from the banded head;
  5. argv, G-DVB, the registry, the rig warm start (named exemptions, an unnamed missing key still refuses), checkpoint + loader round trip.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_own_enc_flag as MO  # noqa: E402

D0, D1 = 16, 32                       # tiny d_lift -> d_map_enc


def _build(own: bool, banded: bool = True, seed: int = 0, perception: bool = True):
    torch.manual_seed(seed)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False,
                                  in_channels=9, image_size=64, image_width=None, trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m), d_lift=D0,
                            d_model=D0, d_up=16, dilations=(1, 2), grad_ckpt=True, near_lift_x_m=10.0, near_refine_blocks=1,
                            banded=banded, own_enc=own, d_map_enc=D1)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    if banded:
        setattr(model, "_r9_map_banded", True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0
    if perception:        # the box3d branch on the SHARED 0.25 m BEV (A6): it reads `map_hires_bev`, the d_model-channel encoder
        model._perception = PB.build_perception_branch(model, PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, bev_source="map_hires_pool"))
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():            # a TRAINED near lift / refine (zero-initialised at build, so their grads would read 0)
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    return model, cfg


def _forward(model, cfg, seed=5):
    """-> (losses, the model's raw output dict) for one train-mode forward on the real-shaped batch."""
    cap = {}
    h = model.register_forward_hook(lambda mod, a, o: cap.__setitem__("o", o))
    try:
        batch = R._batch(cfg)
        torch.manual_seed(seed)
        model.train()
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    finally:
        h.remove()
    return losses, cap["o"]


def _leaves(x, p=""):
    if torch.is_tensor(x):
        yield p, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, p + "." + str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, p + "[%d]" % i)


def _non_map_leaves(out):
    return {k: v for k, v in _leaves(out) if not k.startswith((".perception.map_banded", ".perception.map_hires"))}


def _state_for_source(ref_model):
    """What refcv8 EMA-48,852 holds for the 10 cm map: the M2 build's state minus the banded head, plus a stand-in 8-way readout."""
    sd = {k: v.detach().clone() for k, v in ref_model.state_dict().items() if not k.startswith("_map_hires.banded.")}
    sd["_map_hires.refine.cls.weight"], sd["_map_hires.refine.cls.bias"] = torch.zeros(8, 16, 1, 1), torch.zeros(8)
    return sd


def _warm(model, sd):
    """The pinned warm start applied to a full (non-rig) model: the explicit zero-extend, then the named-exempt load."""
    loadable = {k: v for k, v in sd.items() if not k.startswith("_map_hires.refine.cls.")}
    rec = MO.apply_warm_start(model, loadable)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    assert unexpected == []
    assert all(k.startswith(("_map_hires.banded.",) + MO.NEW_PREFIXES) for k in missing), missing
    return rec


def _max_diff(a: dict, b: dict) -> float:
    assert set(a) == set(b)
    return max(float((a[k].float() - b[k].float()).abs().max()) if a[k].numel() else 0.0 for k in a)


# =========================================================================== #
# 1. OFF, the config, the REAL-size counts                                     #
# =========================================================================== #
def test_OFF_builds_nothing_new():
    m, _ = _build(own=False)
    br = m._map_hires
    assert br.map_encoder is None and not any("map_encoder" in k for k in m.state_dict())
    d = br.cfg.as_dict()
    assert "own_enc" not in d and "d_map_enc" not in d and "lift_width" not in d and "map_encoder" not in br.param_breakdown()
    assert br.cfg.lift_width == D0 == br.lift.d_out and br.encoder.stem[0].in_channels == D0 and br.refine.inp[0].in_channels == D0
    assert H.MapHiresConfig(w_map_hires=1.0).own_enc is False


def test_the_config_refuses_an_own_encoder_it_cannot_build():
    with pytest.raises(ValueError, match="needs banded"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, own_enc=True)
    with pytest.raises(ValueError, match="exceed d_lift"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, banded=True, own_enc=True, d_map_enc=64)
    c = H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, banded=True, own_enc=True)
    assert c.lift_width == 128 and c.as_dict()["own_enc"] is True and c.as_dict()["lift_width"] == 128


def test_REAL_SIZE_parameter_counts_are_measured_and_match_the_hand_count():
    """The real branch: resnet101 stride-8 map (512 ch, 52 x 128 at 416 x 1024), 100 m x +-30 m, d_lift = d_model = 64, d_up 32."""
    def br(own):
        cfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, near_lift_x_m=10.0, near_refine_blocks=1, banded=True,
                               own_enc=own)
        return H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))
    off, on = br(False), br(True)
    po, pn = off.param_breakdown(), on.param_breakdown()
    n_zc = 4 * 512                                                        # Z = 4 heights x 512 stride-8 channels
    assert po["lift"] == 64 * n_zc + 64 + 64 == 131_200 and po["encoder"] == 406_912            # the design's 538,112 = 131,200 + 406,912
    enc128 = 128 * 128 * 9 + 2 * 128 + 5 * (2 * 128 * 128 * 9 + 4 * 128)
    assert pn["map_encoder"] == enc128 == 1_624_832                       # the design's "+~1.6 M"
    assert pn["lift"] - po["lift"] == 131_200                             # 64 -> 128 output rows
    assert pn["encoder"] - po["encoder"] == 64 * 64 * 9                   # the 64-ch encoder's stem now reads 128 input channels
    assert pn["refine"] - po["refine"] == 64 * 32                         # refine.inp 1x1 reads 128 channels
    assert pn["total"] - po["total"] == 1_624_832 + 131_200 + 36_864 + 2_048 == 1_794_944
    print("[C2 MEASURED] branch params OFF %d -> ON %d (+%d): map_encoder %d, lift +%d, encoder stem +%d, refine.inp +%d"
          % (po["total"], pn["total"], pn["total"] - po["total"], pn["map_encoder"], pn["lift"] - po["lift"],
             pn["encoder"] - po["encoder"], pn["refine"] - po["refine"]))


# =========================================================================== #
# 2. fresh init: the extra draws are forked                                    #
# =========================================================================== #
def test_FRESH_INIT_every_shared_tensor_equals_the_M2_build_and_the_rng_stream_is_not_shifted():
    ref, _ = _build(own=False, seed=3)
    on, _ = _build(own=True, seed=3)
    rs, os_ = ref.state_dict(), on.state_dict()
    widened = set(MO.WIDENED) | {"_map_hires.refine.inp.0.weight"}
    for k, v in rs.items():
        if k in widened:
            continue
        assert torch.equal(v, os_[k]), k                       # incl. everything built AFTER the branch (the perception branch)
    for k, ax in MO.WIDENED.items():
        assert torch.equal(rs[k], os_[k].narrow(ax, 0, D0)), k
    assert float(os_["_map_hires.encoder.stem.0.weight"][:, D0:].abs().sum()) == 0.0        # the stem's extra input channels: ZERO
    assert float(os_["_map_hires.lift.proj.weight"][D0:].abs().sum()) > 0.0                 # the extra lift rows: fresh, non-zero
    assert set(os_) - set(rs) == {k for k in os_ if k.startswith("_map_hires.map_encoder.")}


# =========================================================================== #
# 3. STEP 0                                                                    #
# =========================================================================== #
def _step0(red_stem=False, skip_zero_extend=False):
    ref, cfg = _build(own=False, seed=3)
    on, _ = _build(own=True, seed=9)                              # a DIFFERENT seed: the warm start alone must make them equal
    sd = _state_for_source(ref)
    if skip_zero_extend:                                          # RED: a warm start that leaves the stem's extra channels at a fresh init
        real = MO.ZERO_EXTRA
        MO.ZERO_EXTRA = ()
        try:
            with torch.no_grad():
                on._map_hires.encoder.stem[0].weight[:, D0:] = 0.1 * torch.randn_like(on._map_hires.encoder.stem[0].weight[:, D0:])
            _warm(on, sd)
        finally:
            MO.ZERO_EXTRA = real
    else:
        _warm(on, sd)
    if red_stem:                                                  # RED: non-zero init of the stem's extra input channels
        with torch.no_grad():
            w = on._map_hires.encoder.stem[0].weight
            w[:, D0:] = 0.1 * torch.randn_like(w[:, D0:])
    with torch.no_grad():
        _, o_ref = _forward(ref, cfg)
        _, o_on = _forward(on, cfg)
    a, b = _non_map_leaves(o_ref), _non_map_leaves(o_on)
    assert any(k.startswith(".perception.box_slots") for k in a) and any(k.startswith(".waypoints") for k in a)
    return a, b


def test_STEP0_box_and_planner_outputs_equal_the_M2_model_with_the_pinned_warm_start():
    a, b = _step0()
    d = _max_diff(a, b)
    exact = sum(1 for k in a if torch.equal(a[k], b[k]))
    print("[C2 MEASURED] step-0 box3d slots + planner outputs, ON vs M2: %d tensors, %d bit-identical, max |diff| %.3g" % (len(a), exact, d))
    assert d <= 1e-5, d                                           # stated tolerance: float32 re-association in the 128-channel GEMM


def test_STEP0_RED_nonzero_stem_input_channels_FAIL_the_equality():
    a, b = _step0(red_stem=True)
    assert _max_diff(a, b) > 1e-3                                  # the equality check is not vacuous: the RED arm is far outside it
    with pytest.raises(AssertionError):
        assert _max_diff(a, b) <= 1e-5


def test_STEP0_RED_a_warm_start_without_the_zero_extend_FAILS_the_equality():
    a, b = _step0(skip_zero_extend=True)
    # the untouched fresh (non-zero) stem channels survive apply_warm_start only if it does not zero them: the equality must break
    assert _max_diff(a, b) > 1e-3


# =========================================================================== #
# 4. gradients                                                                 #
# =========================================================================== #
def _grads(model, cfg, which):
    """which: 'map' (the banded loss), 'box' (a surrogate on the box3d slots -- the box loss is B3's file, not touched here)."""
    losses, out = _forward(model, cfg)
    model.zero_grad(set_to_none=True)
    if which == "map":
        losses["loss"].backward()
    else:
        s = sum(v.sum() for v in out["perception"]["box_slots"].values() if torch.is_tensor(v) and v.dtype.is_floating_point)
        s.backward()
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}


def _new_names(model):
    return [n for n, _ in model.named_parameters() if n.startswith("_map_hires.map_encoder.")]


def test_GRAD_every_new_parameter_gets_a_nonzero_gradient_and_the_control_reads_exactly_zero(monkeypatch):
    on, cfg = _build(own=True)
    names = _new_names(on)
    assert len(names) == 3 + 2 * 6                                 # stem conv + GN (w, b); 2 dilated blocks x (2 convs + 2 GN x (w, b))
    g = _grads(on, cfg, "map")
    dead = [n for n in names + ["_map_hires.refine.inp.0.weight"] if g[n] is None or float(g[n].abs().sum()) == 0.0]
    assert dead == [], dead                                                            # NON-ZERO on every own-encoder parameter
    # the widened lift: the EXTRA rows reach the optimiser through the own encoder (they are the only consumer of those rows at step 0)
    for n in ("_map_hires.lift.proj.weight", "_map_hires.lift.proj.bias", "_map_hires.lift.unobserved"):
        assert float(g[n][D0:].abs().sum()) > 0.0, n
    # ISOLATION: the map loss never reaches the d_model-channel encoder that feeds the box / planner path
    enc = [n for n in g if n.startswith("_map_hires.encoder.")]
    assert enc and all(g[n] is None or float(g[n].abs().sum()) == 0.0 for n in enc), "the banded loss reached `encoder`"
    # CONTROL: every cell unobserved, max-entropy off -> the SAME parameters read EXACTLY 0.0 (the probe works at all)
    from tanitad.models import map_banded as MB
    monkeypatch.setattr(MB, "MAXENT_W", 0.0)
    real = MB.unobs_masks
    monkeypatch.setattr(MB, "unobs_masks", lambda v, gm: [torch.ones_like(m) for m in real(v, gm)])
    g0 = _grads(on, cfg, "map")
    nonzero = [n for n in names + ["_map_hires.refine.inp.0.weight"] if g0[n] is not None and float(g0[n].abs().sum()) != 0.0]
    assert nonzero == [], nonzero
    assert float(g0["_map_hires.lift.proj.weight"][D0:].abs().sum()) == 0.0


def test_GRAD_the_box_path_reaches_the_zero_initialised_stem_channels_and_never_the_own_encoder():
    on, cfg = _build(own=True)
    g = _grads(on, cfg, "box")
    stem = g["_map_hires.encoder.stem.0.weight"]
    assert float(stem[:, D0:].abs().sum()) > 0.0               # the zero-initialised input channels TRAIN (they carry a gradient at step 0)
    assert float(stem[:, :D0].abs().sum()) > 0.0
    # analytic identity at step 0: those stem channels are zero, so the box loss's gradient into the EXTRA lift rows is exactly 0.0
    assert float(g["_map_hires.lift.proj.weight"][D0:].abs().sum()) == 0.0
    assert float(g["_map_hires.lift.proj.weight"][:D0].abs().sum()) > 0.0
    own = _new_names(on) + ["_map_hires.refine.inp.0.weight"]
    assert all(g[n] is None or float(g[n].abs().sum()) == 0.0 for n in own), "the box loss reached the map's own encoder"


def test_GRAD_RED_an_own_encoder_detached_from_the_banded_head_turns_the_nonzero_check_red(monkeypatch):
    real = H.MapHiresBranch._forward_banded

    def detach(self, feats, fmap_s8, grid, valid, near_source, near_block_zeros, map_feats=None):
        return real(self, feats, fmap_s8, grid, valid, near_source, near_block_zeros, None if map_feats is None else map_feats.detach())
    monkeypatch.setattr(H.MapHiresBranch, "_forward_banded", detach)
    on, cfg = _build(own=True)
    g = _grads(on, cfg, "map")
    dead = [n for n in _new_names(on) if g[n] is None or float(g[n].abs().sum()) == 0.0]
    assert dead, "the RED arm should have left the own encoder with no gradient"


# =========================================================================== #
# 5. argv, G-DVB, the rig warm start, checkpoint, loader                       #
# =========================================================================== #
def _a(**over):
    a = dict(r9_map_own_enc=True, r9_map_banded=True, init_from=None, r9_perc_rig=False)
    a.update(over)
    return argparse.Namespace(**a)


def test_ARGV_off_is_a_noop_and_the_refusals_are_loud():
    MO.pin_args(_a(r9_map_own_enc=False, r9_map_banded=False))
    MO.pin_args(_a())
    MO.pin_args(_a(init_from="x.pt", r9_perc_rig=True))
    with pytest.raises(SystemExit, match="--r9-map-banded is absent"):
        MO.pin_args(_a(r9_map_banded=False))
    MO.pin_args(_a(init_from="x.pt"))                          # FULL-WARM: the full model's `--init-from` is ADMITTED (it applies the same pinned widening
    #                                                            as the rig's: refcv9_perc_rig.apply_refcv9_exemptions; tests/test_full_model_warm_start_r9.py)
    with pytest.raises(SystemExit, match="freeze trains"):
        MO.pin_args(_a(r8_train_only_map=True))                # the map-only freeze trains refine / near, not the own encoder or the lift


def test_the_trainer_parser_owns_the_flag_and_the_registry_has_its_entry():
    parser = T.build_parser()
    a = parser.parse_args(["--arm", "hier", "--out", "z", "--r9-map-banded", "--r9-map-own-enc"])
    assert a.r9_map_own_enc is True and parser.parse_args(["--arm", "hier", "--out", "z"]).r9_map_own_enc is False
    assert D.coverage(parser) == [] and D.REGISTRY["r9_map_own_enc"].kind == "built"


def test_GDVB_the_lever_reads_what_the_model_holds_both_ways():
    on, _ = _build(own=True, perception=False)
    off, _ = _build(own=False, perception=False)
    chk = D.REGISTRY["r9_map_own_enc"].check
    assert chk(on, _a()) == [] and chk(off, _a(r9_map_own_enc=False)) == []
    assert "model._map_hires.map_encoder" in [m.read_from for m in chk(off, _a())]                       # declared, not built
    assert "model._map_hires.map_encoder" in [m.read_from for m in chk(on, _a(r9_map_own_enc=False))]    # built, not declared
    on._map_hires.refine.inp[0] = torch.nn.Conv2d(D0, 16, 1, bias=False)                                  # a readout still reading the 64-ch features
    assert any("refine.inp" in m.read_from for m in chk(on, _a()))


def _rig_pair(monkeypatch, tmp_path):
    orig = v3.RefCV3Model
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: orig(cfg, perc_rig=True))
    ref, _ = _build(own=False, seed=3)
    on, _ = _build(own=True, seed=9)
    src = tmp_path / "ema.pt"
    sd = _state_for_source(ref)
    torch.save({"model": sd, "step": 48852}, src)
    return ref, on, sd, src


def test_RIG_WARM_START_zero_extends_the_widened_keys_names_the_new_ones_and_refuses_an_unnamed_missing_key(monkeypatch, tmp_path):
    ref, on, sd, src = _rig_pair(monkeypatch, tmp_path)
    assert rig.own_enc_exemptions(on) == MO.NEW_PREFIXES and rig.own_enc_exemptions(ref) == ()
    rec = rig.warm_start_perc_rig(on, str(src))
    assert sorted(rec["own_enc"]["widened_keys"]) == sorted(MO.WIDENED) and rec["own_enc"]["dropped_source_keys"] == [
        "_map_hires.refine.inp.0.weight", "_map_hires.refine.inp.1.bias", "_map_hires.refine.inp.1.weight"]
    s = on.state_dict()
    assert torch.equal(s["_map_hires.encoder.stem.0.weight"][:, :D0], sd["_map_hires.encoder.stem.0.weight"])
    assert float(s["_map_hires.encoder.stem.0.weight"][:, D0:].abs().sum()) == 0.0
    assert torch.equal(s["_map_hires.lift.proj.weight"][:D0], sd["_map_hires.lift.proj.weight"])
    assert float(s["_map_hires.lift.proj.weight"][D0:].abs().sum()) > 0.0                                  # fresh rows kept
    assert torch.equal(s["_map_hires.encoder.blocks.0.conv1.weight"], sd["_map_hires.encoder.blocks.0.conv1.weight"])
    # RED: an UNNAMED missing key (a 64-ch encoder block the own-encoder exemption does not cover) still refuses
    bad = {k: v for k, v in sd.items() if k != "_map_hires.encoder.blocks.0.conv1.weight"}
    torch.save({"model": bad, "step": 1}, src)
    with pytest.raises(SystemExit, match="source lacks"):
        rig.warm_start_perc_rig(on, str(src))
    # RED: a source whose widened tensor is ALREADY the wrong width is refused by name, not loaded
    wrong = dict(sd)
    wrong["_map_hires.lift.proj.weight"] = torch.zeros_like(on.state_dict()["_map_hires.lift.proj.weight"])
    torch.save({"model": wrong, "step": 1}, src)
    with pytest.raises(SystemExit, match="widening rule"):
        rig.warm_start_perc_rig(on, str(src))
    # an unknown extra source key is still refused
    ex = dict(sd)
    ex["_map_hires.encoder.mystery.weight"] = torch.zeros(1)
    torch.save({"model": ex, "step": 1}, src)
    with pytest.raises(SystemExit, match="neither built nor in RIG_SKIPPED"):
        rig.warm_start_perc_rig(on, str(src))


def test_CKPT_round_trip_with_the_flag_on_is_strict_and_the_forward_is_reproduced():
    on, cfg = _build(own=True, seed=1)
    sd = {k: v.detach().clone() for k, v in on.state_dict().items()}
    twin, _ = _build(own=True, seed=2)
    r = twin.load_state_dict(sd, strict=True)
    assert list(r.missing_keys) == [] and list(r.unexpected_keys) == []
    with torch.no_grad():
        _, o1 = _forward(on, cfg)
        _, o2 = _forward(twin, cfg)
    assert _max_diff(dict(_leaves(o1)), dict(_leaves(o2))) == 0.0


def test_LOADER_builds_the_same_branch_from_the_runs_config():
    """G-EVAL's builder (`refcv7_loader.map_hires_config`) reads `args.r9_map_own_enc` and rebuilds the identical branch: strict 0 / 0 keys."""
    from tanitad.eval import refcv7_loader as L
    args = argparse.Namespace(w_map_hires=1.0, map_hires_x_max_m=100.0, map_hires_y_half_m=30.0, map_hires_grad_ckpt="on",
                              map_hires_near_lift_m=10.0, map_hires_near_refine_blocks=1, map_hires_decision_rule="prior_corrected",
                              r9_map_banded=True, r9_map_own_enc=True)
    cfg = L.map_hires_config(T, args, "")
    assert cfg.banded and cfg.own_enc and cfg.lift_width == 128
    cfg_off = L.map_hires_config(T, argparse.Namespace(**{**vars(args), "r9_map_own_enc": False}), "")
    assert cfg_off.own_enc is False
    a = H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))
    b = H.MapHiresBranch(dc.replace(cfg), d_image=512, image_hw=(52, 128))
    r = b.load_state_dict(a.state_dict(), strict=True)
    assert list(r.missing_keys) == [] and list(r.unexpected_keys) == []
    assert not L.map_hires_config(T, argparse.Namespace(**{**vars(args), "r9_map_own_enc": False}), "").as_dict().get("own_enc")
