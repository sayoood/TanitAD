"""refcv7 NEW-2 -- the EVAL loader rebuilds the IDENTICAL 10 cm branch (G-EVAL).

``taniteval/tools/refcv3_arm.rebuild_map_hires_branch`` reads ``config.json
["map_hires"]``, switches the trunk's stride-8 tap on and attaches the branch before
the strict load. Pinned: strict 0 missing / 0 unexpected, and BIT-IDENTICAL 10 cm
logits on a fixed batch vs the trained model; a stamp whose parameter count
disagrees is refused. Deliberate-regression arm: a loader that skips the rebuild
must fail the strict load (RED).

⭐ SPEC_REFCV7 A6/A7: the branch runs INSIDE the forward (its logits ride out on
``out["perception"]``), and an A6 model -- the 10 cm branch at the A7 extent plus the
perception branch whose planner pool reads it -- rebuilds strictly, with bit-identical
10 cm logits AND pooled planner BEV. A stamp whose extent does not produce its stamped
grid is refused.

⛔ Lands WITH the shared-file edits (the trunk tap, the ``fmap_s8`` pass-through and
the ``refcv3_arm`` rebuild).
"""
from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "stack"), str(ROOT / "stack" / "scripts"),
           str(ROOT / "taniteval" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refcv3_arm as A                                            # noqa: E402
from tanitad.data.rig_projection import RigCamera                 # noqa: E402
from tanitad.models import bev_lift as L                          # noqa: E402
from tanitad.models import map_head_hires as H                    # noqa: E402
from tanitad.models import refcv6_perception_branch as PB         # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width           # noqa: E402
from tanitad.refs import refc_v3 as v3                            # noqa: E402


def _model():
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(
        cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
        trunk_pretrained=False, in_channels=9, image_size=64, image_width=None,
        trunk_frozen_bn=True)
    return v3.RefCV3Model(cfg)


def _trained(*, x_max_m=60.0, y_half_m=16.0, pool=False):
    torch.manual_seed(0)
    m = _model()
    tap = m.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=0.7, x_max_m=x_max_m, y_half_m=y_half_m,
                            d_lift=16, d_model=16, d_up=16, dilations=(1, 2))
    m._map_hires = H.build_map_hires_branch(m, hcfg)
    config = {}
    if pool:                                                  # A6: the planner pool
        m._perception = PB.build_perception_branch(m, PB.PerceptionBranchConfig(
            w_map=0.0, w_box3d=0.0, bev_source="map_hires_pool"))
    with torch.no_grad():                                     # "trained" weights
        for p in list(m._map_hires.parameters()) + (
                list(m._perception.parameters()) if pool else []):
            p.add_(torch.randn_like(p) * 0.05)
    stamp = {**hcfg.as_dict(), "trunk_tap": tap,
             "branch_params": m._map_hires.param_breakdown(),
             "class_weights": {"sha256": "ab" * 32,
                               "weights": [0.16, 0.07, 0.97, 2.6, 13.6, 4.6, 1.0, 0.06]}}
    config["map_hires"] = stamp
    if pool:
        config["refcv6_perception"] = {**m._perception.cfg.as_dict(),
                                       "branch_params": m._perception.param_breakdown()}
    return m, config


def _out(model, frames):
    """The forward, with THE perception geometry (the 0.25 m one, A6)."""
    h, w = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(w), int(h))
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5),
                              frame=frame, stride=8,
                              grid=model._map_hires.cfg.lift_grid)
    b = frames.shape[0]
    grid = g.grid.unsqueeze(0).expand(b, *g.grid.shape)
    valid = g.valid.unsqueeze(0).expand(b, *g.valid.shape)
    return model(frames, nav_cmd=torch.zeros(b, dtype=torch.long),
                 v0=torch.tensor([4.0] * b), perception_grid=grid,
                 perception_valid=valid)


def _logits(model, frames):
    return _out(model, frames)["perception"]["map_hires_logits"]


def test_the_eval_loader_rebuilds_the_identical_branch():
    m, config = _trained()
    sd = m.state_dict()
    r = _model()
    rec = A.rebuild_map_hires_branch(r, config, "cpu", targs=None)
    assert rec is not None and rec["_tap"]["s8_dim"] == 128
    # the DECLARED rule and the frozen weights the decision needs come back too
    assert r._map_hires.cfg.decision_rule == "prior_corrected"
    assert r._map_hires_class_weight.tolist() == pytest.approx(
        [0.16, 0.07, 0.97, 2.6, 13.6, 4.6, 1.0, 0.06])
    res = r.load_state_dict(sd, strict=False)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    m.eval()
    r.eval()
    fr = torch.rand(2, int(m.cfg.core.window), 9, 64, 64,
                    generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(_logits(m, fr), _logits(r, fr))


def test_DELIBERATE_REGRESSION_a_loader_that_skips_the_rebuild_goes_RED():
    m, _config = _trained()
    r = _model()                                              # no rebuild
    with pytest.raises(RuntimeError, match="_map_hires"):
        r.load_state_dict(m.state_dict(), strict=True)


def test_a_stamp_that_disagrees_is_refused():
    _m, config = _trained()
    config["map_hires"]["branch_params"] = {"total": 1}
    with pytest.raises(SystemExit, match="branch params"):
        A.rebuild_map_hires_branch(_model(), config, "cpu")
    assert A.rebuild_map_hires_branch(_model(), {}, "cpu") is None
    _m, config = _trained()
    config["map_hires"]["out_hw"] = [1000, 600]         # the grid of ANOTHER extent
    with pytest.raises(SystemExit, match="out_hw"):
        A.rebuild_map_hires_branch(_model(), config, "cpu")


def test_the_eval_loader_rebuilds_an_A6_model_at_the_A7_extent_identically():
    """The 10 cm branch at 100 m x +-30 m and the perception branch whose planner pool
    reads it: strict 0/0, bit-identical 10 cm logits AND pooled 120 x 64 planner BEV."""
    m, config = _trained(x_max_m=100.0, y_half_m=30.0, pool=True)
    sd = m.state_dict()
    r = _model()
    A.rebuild_map_hires_branch(r, config, "cpu", targs=None)       # FIRST (the pool
    A.rebuild_perception_branch(r, config, "cpu", targs=None)      # reads it)
    assert r._map_hires.cfg.out_hw == (1000, 600)
    assert r._perception.lift is None and r._perception.bev_pool is not None
    res = r.load_state_dict(sd, strict=False)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    m.eval()
    r.eval()
    fr = torch.rand(1, int(m.cfg.core.window), 9, 64, 64,
                    generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a, b = _out(m, fr)["perception"], _out(r, fr)["perception"]
    assert tuple(a["map_hires_logits"].shape) == (1, 8, 1000, 600)
    assert tuple(a["bev_feats"].shape) == (1, 96, 120, 64)
    assert torch.equal(a["map_hires_logits"], b["map_hires_logits"])
    assert torch.equal(a["bev_feats"], b["bev_feats"])
