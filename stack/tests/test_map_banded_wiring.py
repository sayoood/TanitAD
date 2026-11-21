"""refcv9 C1 -- ``--r9-map-banded`` WIRED INTO THE TRAINER (CPU, a tiny REAL RefCV3Model: resnet18 @ 64x64, the 10 cm branch, the 10 m
near lift). The class this file exists for is "built, tested, and UNREACHABLE FROM ITS CALLER" (CLAUDE.md 2026-09-10/11): a head that
works in a unit test and never reaches the optimiser. So it traces the flag to the loss and to the parameters:

  1. OFF: no banded module, no ``refine.cls`` change, no state_dict key, no config key, no attribute (the bit-identity against the
     pristine tip is the evidence script ``off_digest.py`` run on both trees -- init state_dict, loss scalars, gradients);
  2. ON: the trainer's own ``compute_losses_v3`` emits ONE telemetry key per channel x band (30 + maxent + counts), the 8-way term is
     gone, ``map_hires_logits`` is None BY DESIGN;
  3. ⭐ a NON-ZERO gradient on EVERY banded parameter, and the CONTROL: with every cell unobserved and the max-entropy weight 0 the
     very same parameters read EXACTLY 0.0 (a zero reading is otherwise indistinguishable from a broken probe);
  4. the refusals (the hook that does not pass the channels through; argv), G-DVB both ways, the registry entry;
  5. the rig warm start names the new banded keys and the dropped ``refine.cls`` -- never a blanket ``strict=False``.
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
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_banded_flag as MF  # noqa: E402

#: by hand (d_up = 16, 8 norm groups): head_near 16*10+10; each decoder 2 * (16*16*9) + 2 * 32 (GN) + (16*10+10)
HEAD_NEAR, DECODER = 170, 2 * 2304 + 2 * 32 + 170
BANDED_PARAMS = HEAD_NEAR + 2 * DECODER                      # 9,854
CLS_PARAMS = 8 * 16 + 8                                      # the 8-way readout that is NOT built under the flag


def _build(banded: bool, near_m: float = 10.0):
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False,
                                  in_channels=9, image_size=64, image_width=None, trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m), d_lift=16,
                            d_model=16, d_up=16, dilations=(1, 2), grad_ckpt=True, near_lift_x_m=near_m, near_refine_blocks=1,
                            banded=banded)
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = 1.0
    if banded:
        setattr(model, MF.MODEL_ATTR, True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():            # a TRAINED near lift / refine (they are zero-initialised at build, so their grads would read 0)
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    return model, cfg


def _losses(model, cfg, seed=0):
    torch.manual_seed(seed)
    batch = R._batch(cfg)
    model.train()
    return T.compute_losses_v3(model, batch, "cpu", mode="diffusion")


# =========================================================================== #
# 1. OFF                                                                       #
# =========================================================================== #
def test_OFF_builds_nothing_new():
    model, cfg = _build(banded=False, near_m=20.0)
    br = model._map_hires
    assert br.banded is None and br.refine.cls is not None and not hasattr(model, MF.MODEL_ATTR)
    assert not any("banded" in k for k in model.state_dict())
    assert "banded" not in br.cfg.as_dict() and "banded" not in br.param_breakdown()
    assert H.MapHiresConfig(w_map_hires=1.0).banded is False
    assert H.is_exact_log_key("map_hires_n_x") and H.is_exact_log_key("map_banded_loss_drv_near") and not H.is_exact_log_key("map_fp")
    losses = _losses(model, cfg)
    assert not any(k.startswith("map_banded_") for k in losses)


def test_the_banded_config_refuses_what_it_cannot_tile_or_a_wrong_near_lift():
    with pytest.raises(ValueError, match="near_lift_x_m 10"):
        H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=20.0, banded=True)
    c = H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, banded=True)
    assert c.as_dict()["banded"] is True


# =========================================================================== #
# 2 + 3. ON: the loss, the keys, the gradients                                 #
# =========================================================================== #
def test_ON_param_counts_and_the_absent_8way_readout():
    off, _ = _build(banded=False)
    on, _ = _build(banded=True)
    assert on._map_hires.refine.cls is None
    pb = on._map_hires.param_breakdown()
    assert pb["banded"] == BANDED_PARAMS == 9_854
    n_off, n_on = (sum(p.numel() for p in m._map_hires.parameters()) for m in (off, on))
    assert n_on - n_off == BANDED_PARAMS - CLS_PARAMS
    assert not any(k.startswith("_map_hires.refine.cls") for k in on.state_dict())


def test_ON_the_trainers_loss_emits_one_key_per_channel_and_band_and_no_8way_term():
    model, cfg = _build(banded=True)
    losses = _losses(model, cfg)
    keys = [k for k in losses if k.startswith("map_banded_loss_")]
    assert len(keys) == 30 and {"map_banded_loss_%s_%s" % (c, b) for c in MB.CHANNELS for b in MB.BAND_NAMES} == set(keys)
    for k in MB.telemetry_keys(with_metrics=True):
        assert k in losses, k
    assert all(torch.isfinite(torch.as_tensor(float(losses[k]))) for k in keys)
    assert not any(k.startswith("map_hires_iou") or k.startswith("map_hires_lc_") for k in losses)       # no 8-way per-class rows
    assert float(losses["map_hires"]) > 0 and float(losses["n_map_hires_cells"]) > 0
    # `unobs` reaches the loss end to end: the tiny camera sees part of the map, so the near / mid bands have observed AND unobserved cells
    fr = {b: float(losses["map_banded_unobs_frac_%s" % b]) for b in MB.BAND_NAMES}
    print("[wiring] unobs fractions on the tiny rig:", fr)
    assert 0.0 < fr["near"] < 1.0 and 0.0 < fr["mid"] < 1.0 and all(0.0 <= v < 1.0 for v in fr.values()), fr


def _banded_names(model):
    return [n for n, _ in model.named_parameters() if n.startswith("_map_hires.banded.")]


def test_GRAD_every_banded_parameter_gets_a_nonzero_gradient_and_the_control_reads_exactly_zero(monkeypatch):
    model, cfg = _build(banded=True)
    names = _banded_names(model)
    assert len(names) == 2 + 2 * 8
    losses = _losses(model, cfg)
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    P = dict(model.named_parameters())
    dead = [n for n in names if P[n].grad is None or float(P[n].grad.abs().sum()) == 0.0]
    assert dead == [], dead                                              # NON-ZERO on every new parameter
    # the shared trunk of the near band reaches the optimiser through the banded loss too
    for n in ("_map_hires.refine.conv1.weight", "_map_hires.refine.inp.0.weight", "_map_hires.encoder.stem.0.weight"):
        assert float(P[n].grad.abs().sum()) > 0.0, n
    # CONTROL: every cell unobserved, max-entropy off -> the SAME parameters read EXACTLY 0.0 (the probe works at all)
    monkeypatch.setattr(MB, "MAXENT_W", 0.0)
    real = MB.unobs_masks
    monkeypatch.setattr(MB, "unobs_masks", lambda v, g: [torch.ones_like(m) for m in real(v, g)])
    model.zero_grad(set_to_none=True)
    losses = _losses(model, cfg)
    losses["loss"].backward()
    P = dict(model.named_parameters())
    assert float(losses["map_hires"]) == 0.0
    nonzero = [n for n in names if P[n].grad is not None and float(P[n].grad.abs().sum()) != 0.0]
    assert nonzero == [], nonzero


def test_GRAD_a_banded_head_that_never_reaches_the_loss_turns_the_nonzero_check_red(monkeypatch):
    """The unreachable-from-its-caller class, simulated: the trainer's banded term replaced by a zero. Every banded parameter then reads
    exactly 0.0 -- the check that every one is non-zero must FAIL."""
    monkeypatch.setattr(T, "_map_banded_loss", lambda model, out, batch, device, extra, with_metrics=True:
                        sum((z.sum() * 0.0 for z in out["perception"]["map_banded"]["logits"]), torch.zeros(())))
    with pytest.raises(AssertionError):
        test_GRAD_every_banded_parameter_gets_a_nonzero_gradient_and_the_control_reads_exactly_zero(monkeypatch)


def test_a_forward_that_does_not_pass_the_channels_through_is_refused_not_skipped(monkeypatch):
    model, cfg = _build(banded=True)
    real = H.MapHiresBranch._forward_banded

    def strip(self, *a, **k):
        o = real(self, *a, **k)
        o.pop("map_banded")
        return o
    monkeypatch.setattr(H.MapHiresBranch, "_forward_banded", strip)
    with pytest.raises(SystemExit, match="carries no `map_banded`"):
        _losses(model, cfg)


# =========================================================================== #
# 4. argv, G-DVB                                                               #
# =========================================================================== #
def _a(**over):
    a = dict(r9_map_banded=True, map_hires="on", w_map_hires=1.0, map_hires_near_lift_m=10.0, w_map_fp=0.0,
             map_lift_valid_mask=True, map_hires_class_thresholds=None, r8_critic_drivable=False, w_r8_drivable=0.0)
    a.update(over)
    return argparse.Namespace(**a)


def test_ARGV_off_is_a_noop_and_the_refusals_are_loud():
    MF.pin_args(_a(r9_map_banded=False, map_hires="off"))                       # OFF: nothing is checked
    MF.pin_args(_a())                                                           # the legal combination
    for over, why in ((dict(map_hires="off"), "map-hires is not"), (dict(w_map_hires=0.0), "w-map-hires is 0"),
                      (dict(map_hires_near_lift_m=20.0), "near-lift-m"), (dict(map_hires_near_lift_m=None), "near-lift-m"),
                      (dict(w_map_fp=0.1), "RETIRED"), (dict(map_lift_valid_mask=False), "unobs"),
                      (dict(map_hires_class_thresholds="x.json"), "class-thresholds")):
        with pytest.raises(SystemExit, match=why):
            MF.pin_args(_a(**over))
    MF.pin_args(_a(r8_critic_drivable=True, w_r8_drivable=0.1))                 # refcv9 C8 W-MAP-1: the critic now reads the banded channels


def test_the_trainer_parser_owns_the_flag_and_the_registry_has_its_entry():
    parser = T.build_parser()
    a = parser.parse_args(["--arm", "hier", "--out", "z", "--r9-map-banded"])
    assert a.r9_map_banded is True and parser.parse_args(["--arm", "hier", "--out", "z"]).r9_map_banded is False
    assert D.coverage(parser) == [] and D.REGISTRY["r9_map_banded"].kind == "built"
    assert "r9_map_banded" in {x.dest for x in parser._actions}


def test_GDVB_the_lever_reads_what_the_model_holds_both_ways():
    on, _ = _build(banded=True)
    off, _ = _build(banded=False)
    chk = D.REGISTRY["r9_map_banded"].check
    assert chk(on, _a()) == [] and chk(off, _a(r9_map_banded=False)) == []
    assert [m.read_from for m in chk(off, _a())] == ["model._map_hires.banded", "model._r9_map_banded"]       # declared, not built
    assert "model._map_hires.banded" in [m.read_from for m in chk(on, _a(r9_map_banded=False))]                # built, not declared
    on._map_hires.refine.cls = torch.nn.Conv2d(16, 8, 1)                                                       # an untrained readout
    assert any("refine.cls" in m.read_from for m in chk(on, _a()))


# =========================================================================== #
# 5. the rig warm start                                                        #
# =========================================================================== #
def _rig_banded(monkeypatch):
    orig = v3.RefCV3Model
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: orig(cfg, perc_rig=True))
    return _build(banded=True)


def test_WARM_START_names_the_new_banded_keys_and_the_dropped_readout(monkeypatch, tmp_path):
    model, _ = _rig_banded(monkeypatch)
    assert rig.banded_exemptions(model) == (("_map_hires.banded.",), ("_map_hires.refine.cls.",))
    sd = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("_map_hires.banded.")}
    sd["_map_hires.refine.cls.weight"], sd["_map_hires.refine.cls.bias"] = torch.zeros(8, 16, 1, 1), torch.zeros(8)
    src = tmp_path / "ema.pt"
    torch.save({"model": sd, "step": 48852}, src)
    rec = rig.warm_start_perc_rig(model, str(src))
    assert rec["banded"]["n_new_keys"] == 2 + 2 * 8 and rec["banded"]["dropped_source_keys"] == [
        "_map_hires.refine.cls.bias", "_map_hires.refine.cls.weight"]
    # any OTHER missing key is still refused (the exemption is the named prefix, not a blanket strict=False)
    bad = {k: v for k, v in sd.items() if k != "_map_hires.refine.conv1.weight"}
    torch.save({"model": bad, "step": 1}, src)
    with pytest.raises(SystemExit, match="source lacks"):
        rig.warm_start_perc_rig(model, str(src))
    # and an UNKNOWN extra source key is refused too
    ex = dict(sd)
    ex["_map_hires.refine.mystery.weight"] = torch.zeros(1)
    torch.save({"model": ex, "step": 1}, src)
    with pytest.raises(SystemExit, match="neither built nor in RIG_SKIPPED"):
        rig.warm_start_perc_rig(model, str(src))
    # OFF model: no exemption at all
    off, _ = _build(banded=False, near_m=20.0)
    assert rig.banded_exemptions(off) == ((), ())
