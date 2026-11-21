"""refcv7 NEW-2 -- the 10 cm map branch (``tanitad.models.map_head_hires``).

Independent of the shared-file edits (the trunk tap and the ``fmap_s8``
pass-through are pinned in ``test_map_hires_wiring.py``, which lands WITH them).
Pinned here, each with a literal or a discriminating control:

1. the lift is ``BEVLift`` with the projection moved first -- equal outputs AND
   equal gradients to the original on random inputs, same state_dict keys;
2. the geometry is the refcv6 bank's own at stride 8 / 0.25 m, cache bounded;
3. SHAPES at the REAL feature geometry (resnet101 stride 8 at 416 x 1024 = 512 x
   52 x 128) -> ``[B, 8, 600, 320]``; the parameter count, printed, < 2 M;
4. GRADIENT REACH to a stand-in stride-8 stage and to EVERY new parameter, with a
   detached seam as the control (trunk gradient exactly 0);
5. the loss: a not-seen cell contributes EXACTLY 0 to the value and the gradient,
   with a deliberate-regression arm that counts them and must go RED;
6. the class-weights loader's refusals (incl. the DRY RUN marker, and weights counted
   on another EXTENT) and its sha256;
7. ⭐ SPEC_REFCV7 A6/A7: the extent is a declared parameter -- 100 m x +-30 m gives a
   400 x 240 lift and 1000 x 600 logits, five 20 m bands (a partial last band keeps
   its edge); the encoder output is SHARED (``map_hires_bev``); a zero-weight class is
   NEVER decided by the prior-corrected rule (red arm: the naive ``z - log 0``).
"""
from __future__ import annotations

import json
import math
import sys

import pytest
import torch
from torch import nn

from tanitad.data.bev_raster import BEVGrid
from tanitad.data.rig_projection import RigCamera
from tanitad.models import bev_lift as L
from tanitad.models import map_head_hires as H
from tanitad.models.trunk_shapes import FRAME_416x1024, frame_for_width

FINE = (600, 320)


def _geom(frame=FRAME_416x1024, stride=8, cfg=None, b=1):
    cfg = cfg or H.MapHiresConfig(w_map_hires=1.0)
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5),
                              frame=frame, stride=stride, grid=cfg.lift_grid)
    return (g.grid.unsqueeze(0).expand(b, *g.grid.shape).contiguous(),
            g.valid.unsqueeze(0).expand(b, *g.valid.shape).contiguous())


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w", [0.0, -1.0, float("nan")])
def test_a_weightless_branch_refuses_to_exist(w):
    with pytest.raises(ValueError, match="no live weight"):
        H.MapHiresConfig(w_map_hires=w)


def test_config_pins_the_spec_grids():
    c = H.MapHiresConfig(w_map_hires=1.0)
    assert c.lift_grid.shape == (240, 128) and c.out_hw == (600, 320)
    assert c.stride == 8 and c.receptive_field_m == 15.75
    assert c.band_keys == ("0_20", "20_40", "40_60")
    with pytest.raises(ValueError):
        H.MapHiresConfig(w_map_hires=1.0, lift_cell_m=0.5)
    with pytest.raises(ValueError):
        H.MapHiresConfig(w_map_hires=1.0, stride=16)
    with pytest.raises(TypeError):
        H.MapHiresConfig(w_map_hires=1.0, out_hw=(120, 64))  # derived, never declared


def test_the_extent_is_a_declared_parameter_A7():
    """SPEC_REFCV7 §12: 100 m x +-30 m -> 400 x 240 at 0.25 m, 1000 x 600 at 0.1 m,
    bands every 20 m. Literal numbers."""
    c = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
    assert c.lift_grid.shape == (400, 240) and c.out_hw == (1000, 600)
    assert c.band_keys == ("0_20", "20_40", "40_60", "60_80", "80_100")
    d = c.as_dict()
    assert (d["x_max_m"], d["y_half_m"], d["out_hw"], d["lift_grid_hw"]) == (
        100.0, 30.0, [1000, 600], [400, 240])
    assert H.MapHiresConfig(w_map_hires=1.0, x_max_m=70.0).band_keys[-1] == "60_70"
    for bad in ({"x_max_m": 55.0}, {"y_half_m": 12.0}, {"x_max_m": 100.25}):
        with pytest.raises(ValueError):
            H.MapHiresConfig(w_map_hires=1.0, **bad)
    assert H.band_keys_for_rows(1000) == ("0_20", "20_40", "40_60", "60_80", "80_100")
    assert H.band_keys_for_rows(700)[-1] == "60_70"


def test_A7_extent_forward_shapes_bands_and_lift_valid():
    torch.manual_seed(0)
    cfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0,
                           d_lift=8, d_model=8, d_up=8, dilations=(1,))
    br = H.MapHiresBranch(cfg, d_image=16, image_hw=(52, 128))
    grid, valid = _geom(cfg=cfg)
    with torch.no_grad():
        out = br(torch.randn(1, 16, 52, 128), grid, valid)
    assert tuple(out["map_hires_logits"].shape) == (1, 8, 1000, 600)
    assert tuple(out["map_hires_lift_valid"].shape) == (1, 400, 240)
    assert tuple(out["map_hires_bev"].shape) == (1, 8, 400, 240)     # shared features
    f = H.lift_valid_to_fine(out["map_hires_lift_valid"])
    assert tuple(f.shape) == (1, 1000, 600)
    codes = torch.randint(0, 8, (1, 1000, 600), dtype=torch.uint8)
    r = H.map_hires_loss_row(out["map_hires_logits"], codes,
                             lift_valid_025=out["map_hires_lift_valid"])
    assert "map_hires_n_lane_80_100" in r and "map_hires_n_lane_60_80" in r
    assert H.missing_per_class_keys(H.derived_per_class(r, band_keys=cfg.band_keys),
                                    band_keys=cfg.band_keys) == []
    with pytest.raises(ValueError):          # a 60 m target under a 100 m head
        H.hires_map_ce(out["map_hires_logits"],
                       torch.zeros(1, 600, 320, dtype=torch.uint8))


# --------------------------------------------------------------------------- #
# 1. the lift is BEVLift, projection first                                     #
# --------------------------------------------------------------------------- #
def test_project_first_lift_equals_bevlift_values_and_gradients():
    torch.manual_seed(0)
    B, C, h, w, Z, X, Y, D = 2, 12, 6, 16, 4, 10, 7, 8
    ref = L.BEVLift(d_in=C, d_out=D, n_heights=Z, feat_hw=(h, w))
    new = H.BEVLiftProjectFirst(d_in=C, d_out=D, n_heights=Z, feat_hw=(h, w))
    assert set(new.state_dict()) == set(ref.state_dict())
    new.load_state_dict(ref.state_dict())
    f1 = torch.randn(B, C, h, w, requires_grad=True)
    f2 = f1.detach().clone().requires_grad_(True)
    grid = torch.rand(B, Z, X, Y, 2) * 2.4 - 1.2          # some samples off the map
    valid = torch.rand(B, Z, X, Y) > 0.3
    valid[:, :, 0, 0] = False                              # a fully unobserved cell
    a, b = ref(f1, grid, valid), new(f2, grid, valid)
    torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
    (a * torch.arange(a.numel()).reshape(a.shape).float().sin()).sum().backward()
    (b * torch.arange(b.numel()).reshape(b.shape).float().sin()).sum().backward()
    torch.testing.assert_close(f2.grad, f1.grad, rtol=1e-4, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), new.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. the geometry is the refcv6 bank's, at stride 8 / 0.25 m                   #
# --------------------------------------------------------------------------- #
def test_hires_bank_is_the_refcv6_bank_at_stride_8_with_a_bounded_cache():
    from tanitad.data.v2_dataset import stable_episode_id
    cfg = H.MapHiresConfig(w_map_hires=1.0)
    cam = RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5)
    table = {f"clip-{i}": cam for i in range(4)}
    bank = H.HiresLiftGeometryBank(table, frame=FRAME_416x1024, cfg=cfg, max_cache=2)
    ids = [int(stable_episode_id(f"clip-{i}")) for i in range(4)]
    g, v = bank.for_episodes(ids)
    assert tuple(g.shape) == (4, 4, 240, 128, 2) and tuple(v.shape) == (4, 4, 240, 128)
    assert len(bank._cache) == 2                          # bounded
    ref = L.build_lift_geometry(cam, frame=FRAME_416x1024, stride=8,
                                grid=BEVGrid(60.0, 16.0, 0.25))
    torch.testing.assert_close(g[0], ref.grid)
    assert torch.equal(v[0], ref.valid)
    # a nominal level camera sees ~89 % of the 0.25 m cells in at least one height
    assert 0.8 < float(v[0].any(0).float().mean()) < 0.95


# --------------------------------------------------------------------------- #
# 3. shapes at the real feature geometry, and the parameter budget             #
# --------------------------------------------------------------------------- #
def test_real_geometry_shapes_and_param_budget():
    torch.manual_seed(0)
    cfg = H.MapHiresConfig(w_map_hires=1.0)
    br = H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))     # r101 s8 @ 416x1024
    grid, valid = _geom()
    with torch.no_grad():
        out = br(torch.randn(1, 512, 52, 128), grid, valid)
    assert tuple(out["map_hires_logits"].shape) == (1, 8, 600, 320)
    assert tuple(out["map_hires_lift_valid"].shape) == (1, 240, 128)
    assert tuple(out["map_hires_bev"].shape) == (1, 64, 240, 128)
    pb = br.param_breakdown()
    print("map-hires params", pb)
    assert pb["total"] == pb["lift"] + pb["encoder"] + pb["refine"]
    assert pb["lift"] == 512 * 4 * 64 + 64 + 64            # proj W + b + unobserved
    assert pb["total"] == 559_048 and pb["total"] < 2_000_000


def test_missing_stride8_map_is_refused_not_skipped():
    cfg = H.MapHiresConfig(w_map_hires=1.0)
    br = H.MapHiresBranch(cfg, d_image=16, image_hw=(52, 128))
    grid, valid = _geom()
    with pytest.raises(H.MapHiresMissingInput, match="fmap_s8 is None"):
        br(None, grid, valid)
    with pytest.raises(H.MapHiresMissingInput, match="lift geometry"):
        br(torch.zeros(1, 16, 52, 128), None, None)


# --------------------------------------------------------------------------- #
# 4. gradient reach -- with the detached seam as the control                   #
# --------------------------------------------------------------------------- #
class _S8Stage(nn.Module):
    """A stand-in stride-8 stage: 8x downsampling, so `fmap_s8` has a graph."""

    def __init__(self, c=16):
        super().__init__()
        self.conv = nn.Conv2d(3, c, 8, stride=8)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_gradient_reaches_the_stride8_stage_and_every_new_parameter(grad_ckpt):
    torch.manual_seed(0)
    frame = frame_for_width(256, 64)                      # s8 map 8 x 32
    cfg = H.MapHiresConfig(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16,
                           dilations=(1, 2), grad_ckpt=grad_ckpt)
    trunk = _S8Stage(16)
    br = H.MapHiresBranch(cfg, d_image=16, image_hw=(8, 32))
    grid, valid = _geom(frame=frame, cfg=cfg, b=2)
    x = torch.randn(2, 3, 64, 256)
    codes = torch.randint(0, 8, (2, 600, 320), dtype=torch.uint8)
    codes[:, :50] = 255

    def run(detach):
        for p in list(trunk.parameters()) + list(br.parameters()):
            p.grad = None
        f = trunk(x)
        out = br(f.detach() if detach else f, grid, valid)
        H.hires_map_ce(out["map_hires_logits"], codes)["loss"].backward()
        return H.grad_abs_sum(trunk)[0]

    live = run(False)
    assert live > 0.0, "the 10 cm loss does not reach the stride-8 stage"
    dead = [n for n, p in br.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"new parameters with NO gradient: {dead}"
    assert run(True) == 0.0                              # the control: exactly 0
    rep = H.grad_reach_report_hires(None, br)            # the instrument reads it too
    assert set(rep) == {"lift", "encoder", "refine"}
    assert all(v["grad_abs_sum"] > 0.0 for v in rep.values())


# --------------------------------------------------------------------------- #
# 5. the loss                                                                  #
# --------------------------------------------------------------------------- #
def _assert_not_seen_cells_contribute_zero(loss_fn) -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 600, 320, requires_grad=True)
    codes = torch.randint(0, 8, (2, 600, 320), dtype=torch.uint8)
    codes[0, :100] = 255
    codes[1, :, 200:] = 255
    w = torch.tensor([0.5, 1.0, 3.0, 5.0, 20.0, 4.0, 25.0, 0.4])
    r1 = loss_fn(logits, codes, class_weight=w)
    r1["loss"].backward()
    g = logits.grad.clone()
    unseen = (codes == 255).unsqueeze(1).expand_as(g)
    assert float(g[unseen].abs().max()) == 0.0, "a not-seen cell got a gradient"
    # the value is invariant to ANY change of the logits on not-seen cells
    l2 = logits.detach().clone()
    l2[unseen] = torch.randn(int(unseen.sum())) * 50.0
    r2 = loss_fn(l2, codes, class_weight=w)
    assert float(r2["loss"].detach()) == float(r1["loss"].detach())
    assert r1["n_cells"] == int((codes != 255).sum())


def test_not_seen_cells_contribute_exactly_zero():
    _assert_not_seen_cells_contribute_zero(H.hires_map_ce)


def test_DELIBERATE_REGRESSION_counting_not_seen_cells_goes_RED(monkeypatch):
    """255 folded into a class (here: sidewalk) instead of ignored -- the loss then
    trains on cells nobody labelled. The invariance above must FAIL."""
    monkeypatch.setattr(H, "_ce_targets", lambda codes: codes.long().clamp(max=7))
    with pytest.raises(AssertionError):
        _assert_not_seen_cells_contribute_zero(H.hires_map_ce)


def test_loss_edge_cases_and_the_weight_denominator():
    logits = torch.zeros(1, 8, 600, 320, requires_grad=True)
    allunseen = torch.full((1, 600, 320), 255, dtype=torch.uint8)
    r = H.hires_map_ce(logits, allunseen)
    assert float(r["loss"]) == 0.0 and r["n_cells"] == 0
    r["loss"].backward()                                   # attached, finite
    codes = torch.randint(0, 8, (1, 600, 320), dtype=torch.uint8)
    lg = torch.randn(1, 8, 600, 320)
    a = float(H.hires_map_ce(lg, codes)["loss"])
    b = float(H.hires_map_ce(lg, codes, class_weight=torch.full((8,), 7.0))["loss"])
    assert a == pytest.approx(b, rel=1e-6)                 # uniform weights == none
    # uniform logits: CE = ln 8 exactly, weighted or not
    u = H.hires_map_ce(torch.zeros(1, 8, 600, 320), codes,
                       class_weight=torch.arange(1.0, 9.0))
    assert float(u["loss"]) == pytest.approx(math.log(8.0), abs=1e-6)
    with pytest.raises(ValueError):
        H.hires_map_ce(lg, codes.to(torch.int64))


def test_lift_valid_narrows_the_supervised_cells_on_the_nearest_exact_rows():
    lv = torch.zeros(1, 240, 128, dtype=torch.bool)
    lv[0, 1, :] = True                                     # 0.25 m row 1 = x 0.25..0.5 m
    f = H.lift_valid_to_fine(lv)
    rows = torch.nonzero(f[0].any(1)).flatten().tolist()
    # fine rows centred at 0.25, 0.35, 0.45 m fall in [0.25, 0.5); 0.55 m does not.
    # (plain `nearest` would give floor(0.4 j) -> rows 3, 4 -- off by half a cell)
    assert rows == [2, 3, 4]
    codes = torch.randint(0, 8, (1, 600, 320), dtype=torch.uint8)
    r = H.hires_map_ce(torch.randn(1, 8, 600, 320), codes, lift_valid=f)
    assert r["n_cells"] == 3 * 320 and r["n_cells_seen"] == 600 * 320


# --------------------------------------------------------------------------- #
# 6. the class-weights file                                                    #
# --------------------------------------------------------------------------- #
def _weights_file(tmp_path, **over):
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    d = {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
         "weights": [0.04, 0.03, 0.7, 2.1, 16.8, 1.9, 17.9, 0.03],
         "split": "train", "inputs_sha256": "0" * 64,
         "definition": "median-frequency (Eigen & Fergus 2015)", "dry_run": False}
    d.update(over)
    d = {k: v for k, v in d.items() if v is not None}
    p = tmp_path / "w.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_class_weights_load_and_stamp_their_sha256(tmp_path):
    import hashlib
    p = _weights_file(tmp_path)
    w, st = H.load_class_weights(p)
    assert w.dtype == torch.float32 and tuple(w.shape) == (8,)
    assert st["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert st["extent"] == {"x_max_m": 60.0, "y_half_m": 16.0}   # no key = the /2 grid


def test_class_weights_counted_on_another_extent_are_refused(tmp_path):
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7, EXTENT_V2
    legacy = _weights_file(tmp_path)                               # no extent key
    H.load_class_weights(legacy, extent=EXTENT_V2)
    with pytest.raises(ValueError, match="another extent|Recompute"):
        H.load_class_weights(legacy, extent=EXTENT_REFCV7)
    a7 = _weights_file(tmp_path, extent={"x_max_m": 100.0, "y_half_m": 30.0})
    w, st = H.load_class_weights(a7, extent=EXTENT_REFCV7)
    assert st["extent"] == {"x_max_m": 100.0, "y_half_m": 30.0}
    with pytest.raises(ValueError, match="Recompute"):
        H.load_class_weights(a7, extent=EXTENT_V2)


@pytest.mark.parametrize("over,msg", [
    ({"schema": "x/1"}, "schema"),
    ({"weights": [1.0] * 7}, "7 weights"),
    ({"weights": [1.0] * 7 + [26.0]}, "clip"),
    ({"weights": [1.0] * 7 + [0.0]}, "finite and > 0"),
    ({"classes": ["a"] * 8}, "class order"),
    ({"dry_run": True, "dry_run_note": "DRY RUN"}, "DRY RUN"),
])
def test_class_weights_refusals(tmp_path, over, msg):
    with pytest.raises(ValueError, match=msg):
        H.load_class_weights(_weights_file(tmp_path, **over))


# --------------------------------------------------------------------------- #
# 7. per-class instrumentation (Master Mind addition 1): analytic control      #
# --------------------------------------------------------------------------- #
def _const_scene():
    """band 0 (rows 0-199): drivable, a 2-column lane at cols 150-151;
    band 1 (rows 200-399): sidewalk; band 2 (rows 400-599): NOT SEEN."""
    codes = torch.full((1, 600, 320), 1, dtype=torch.uint8)
    codes[0, :200, 150:152] = 2
    codes[0, 200:400] = 7
    codes[0, 400:] = 255
    return codes


#: float32 sums over ~1e5 cells: 1e-5 relative bounds both x86 and aarch64 (see below)
F32_SUM_REL = 1e-5


def _assert_constant_logit_control(fn) -> None:
    """Constant logits => p = 1/8 everywhere, CE = ln 8 on every supervised cell,
    N = 63,600 + 400 + 64,000 = 128,000 supervised cells, and the logit gradient
    is (1/8 - [c = y]) / N. Every expected value below is hand-derived.

    ⚠️ The code sums in float32, so the loss / ``lc`` comparisons are held to
    :data:`F32_SUM_REL`, not 1e-6. MEASURED on Thor (aarch64, the gate's venv): ``lc[1, 0]``
    read 1.0332245826721191 against 1.0332225160221682 -- 2.0e-6 relative, the float32
    pairwise-summation scale at N = 128,000 (log2 N x eps = 17 x 1.19e-7); the dev box
    (x86) stays under 1e-6. Not a defect: the hand-derived value is exact arithmetic."""
    s = fn(torch.zeros(1, 8, 600, 320), _const_scene())
    ln8, N = math.log(8.0), 128_000.0
    assert s["n_supervised"] == 128_000 and s["W"] == 128_000.0
    assert s["loss"] == pytest.approx(ln8, rel=F32_SUM_REL)
    n = s["n"].tolist()
    assert n[1] == [63_600.0, 0.0, 0.0] and n[2] == [400.0, 0.0, 0.0]
    assert n[7] == [0.0, 64_000.0, 0.0] and float(s["n"].sum()) == N
    lc = s["lc"]
    assert float(lc[1, 0]) == pytest.approx(63_600 * ln8 / N, rel=F32_SUM_REL)
    assert float(lc[2, 0]) == pytest.approx(400 * ln8 / N, rel=F32_SUM_REL)
    assert float(lc[7, 1]) == pytest.approx(0.5 * ln8, rel=F32_SUM_REL)
    assert float(lc.sum()) == pytest.approx(ln8, rel=F32_SUM_REL)
    gn = s["gn"]
    # lane channel, band 0: 400 cells at (1/8 - 1)/N and 63,600 at (1/8)/N
    assert float(gn[2, 0]) == pytest.approx(math.sqrt(1300.0) / N, rel=1e-5)
    # edge channel (no edge cells), band 0: 64,000 cells at (1/8)/N
    assert float(gn[5, 0]) == pytest.approx(math.sqrt(1000.0) / N, rel=1e-5)
    # sidewalk channel, band 1: every cell is sidewalk, at (1/8 - 1)/N
    assert float(gn[7, 1]) == pytest.approx(7 / 8 * math.sqrt(64_000.0) / N, rel=1e-5)
    assert float(s["gno"][2, 0]) == pytest.approx(7 / 8 * 20.0 / N, rel=1e-5)
    assert float(gn[:, 2].abs().max()) == 0.0          # the unseen band: nothing
    # argmax of equal logits is class 0 everywhere: no class-0 GT -> inter 0
    assert float(s["inter"].sum()) == 0.0
    assert s["union"][0].tolist() == [64_000.0, 64_000.0, 0.0]


def test_per_class_signal_constant_logit_control():
    _assert_constant_logit_control(H.per_class_signal)


def test_DELIBERATE_REGRESSION_per_class_signal_counting_unseen_goes_RED(monkeypatch):
    monkeypatch.setattr(H, "_ce_targets", lambda codes: codes.long().clamp(max=7))
    with pytest.raises(AssertionError):
        _assert_constant_logit_control(H.per_class_signal)


def test_per_class_signal_matches_autograd_and_the_loss():
    torch.manual_seed(4)
    codes = torch.randint(0, 8, (2, 600, 320), dtype=torch.uint8)
    codes[:, :30] = 255
    lv = torch.rand(2, 600, 320) > 0.2
    w = torch.tensor([0.2, 0.1, 1.0, 2.6, 13.6, 4.6, 1.0, 0.06])
    logits = (torch.randn(2, 8, 600, 320) * 2.0).requires_grad_(True)
    ref = H.hires_map_ce(logits, codes, class_weight=w, lift_valid=lv)
    ref["loss"].backward()
    g = logits.grad.double()
    s = H.per_class_signal(logits, codes, class_weight=w, lift_valid=lv)
    assert s["loss"] == pytest.approx(float(ref["loss"].detach()), rel=1e-5)
    assert float(s["lc"].sum()) == pytest.approx(float(ref["loss"].detach()), rel=1e-5)
    want = (g * g).reshape(2, 8, 3, 200, 320).sum(dim=(0, 3, 4)).sqrt()
    torch.testing.assert_close(s["gn"], want, rtol=1e-4, atol=1e-9)
    assert s["n_supervised"] == ref["n_cells"]


def test_derived_per_class_pools_a_mean_of_rows():
    lg = torch.zeros(1, 8, 600, 320)
    lg[0, 1] = 1.0                                      # predict drivable everywhere
    codes = _const_scene()
    r = H.per_class_log_values(H.per_class_signal(lg, codes))
    d = H.derived_per_class(r)
    # drivable: GT 63,600 cells of band 0, predicted on all 64,000 supervised
    assert d["map_hires_iou_drivable_0_20"] == pytest.approx(63_600 / 64_000)
    assert d["map_hires_iou_sidewalk_20_40"] == 0.0     # union 64,000, inter 0
    assert d["map_hires_iou_lane_40_60"] is None        # union 0: undefined -> None
    assert r["map_hires_union_lane_40_60"] == 0.0       # ... and the raw 0 is logged
    assert len([k for k in d if "_iou_" in k]) == 24    # ALL 8 x 3, always (A3 G-DVB)
    assert len([k for k in d if "_lshare_" in k]) == 24
    shares = [v for k, v in d.items() if "_lshare_" in k and v is not None]
    assert sum(shares) == pytest.approx(1.0)
    # an eval row = the MEAN of two batch rows -> the IoU is the POOLED one
    r2 = H.per_class_log_values(H.per_class_signal(torch.zeros(1, 8, 600, 320), codes))
    mean = {k: (r[k] + r2[k]) / 2.0 for k in r}
    dm = H.derived_per_class({"eval_" + k: v for k, v in mean.items()}, "eval_")
    assert dm["eval_map_hires_iou_drivable_0_20"] == pytest.approx(
        (63_600 + 0) / (64_000 + 63_600))


# --------------------------------------------------------------------------- #
# 8. G-DVB: the four NEW-2 levers, declared vs BUILT (fakes; the real model   #
#    is pinned in test_map_hires_wiring.py, which lands with the trainer)     #
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402


def _dvb_model(*, on=True, tap=True, passthrough=("layer_logits", "fmap_s8"), w=1.0,
               cw_file=None, cfg_sha=None, tensor=None, bank_rows=43,
               grad_ckpt=False):
    core_cls = type("FakeCore", (), {"DECODER_PASSTHROUGH": tuple(passthrough)})
    core = core_cls()
    core.encoder = SimpleNamespace(s8_tap=tap, s8_dim=16)
    m = SimpleNamespace(core=core, _w_map_hires=(w if on else 0.0),
                        _map_hires=None, _lift_bank_hires=None,
                        _map_hires_class_weight=tensor, _perception=None)
    if on:
        sha = cfg_sha
        if sha is None and cw_file is not None:
            import hashlib as _h
            sha = _h.sha256(cw_file.read_bytes()).hexdigest()
        cfg = H.MapHiresConfig(w_map_hires=w, d_lift=16, d_model=16, d_up=16,
                               dilations=(1,), class_weights_sha256=sha or "",
                               grad_ckpt=grad_ckpt)
        m._map_hires = H.MapHiresBranch(cfg, d_image=16, image_hw=(8, 16))
        frame = frame_for_width(128, 64)
        m._lift_bank_hires = H.HiresLiftGeometryBank(
            {"c": RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=cfg,
            equalize_bottom_rows=bank_rows)
    return m


def _dvb_args(**over):
    a = dict(map_hires="on", w_map_hires=1.0, map_hires_class_weights=None,
             equalize_bottom_rows=43, map_hires_x_max_m=60.0, map_hires_y_half_m=16.0,
             map_hires_grad_ckpt="off")
    a.update(over)
    return SimpleNamespace(**a)


def _all_checks(m, a):
    return (H.dvb_check_map_hires(m, a) + H.dvb_check_w_map_hires(m, a)
            + H.dvb_check_class_weights(m, a)
            + H.dvb_check_extent("map_hires_x_max_m")(m, a)
            + H.dvb_check_extent("map_hires_y_half_m")(m, a)
            + H.dvb_check_grad_ckpt(m, a))


def test_dvb_GREEN_a_consistent_build_and_the_default_both_pass(tmp_path):
    p = _weights_file(tmp_path)
    w = [0.04, 0.03, 0.7, 2.1, 16.8, 1.9, 17.9, 0.03]
    m = _dvb_model(cw_file=p, tensor=torch.tensor(w))
    assert _all_checks(m, _dvb_args(map_hires_class_weights=str(p))) == []
    off = _dvb_model(on=False, tap=False)
    assert _all_checks(off, _dvb_args(map_hires="off", w_map_hires=0.0,
                                      map_hires_x_max_m=None, map_hires_y_half_m=None,
                                      map_hires_grad_ckpt=None)) == []


@pytest.mark.parametrize("model_kw,args_kw,lever,read_from", [
    ({"passthrough": ("layer_logits",)}, {}, "--map-hires", "DECODER_PASSTHROUGH"),
    ({"tap": False}, {}, "--map-hires", "core.encoder.s8_tap"),
    ({"bank_rows": 0}, {}, "--map-hires", "_lift_bank_hires.equalize_bottom_rows"),
    ({"on": False, "tap": False}, {}, "--map-hires", "model._map_hires is not None"),
    ({}, {"map_hires": "off", "w_map_hires": 0.0}, "--map-hires",
     "model._map_hires is not None"),
    ({"w": 0.5}, {}, "--w-map-hires", "model._w_map_hires (read by the loss)"),
    # the declared EXTENT vs the built branch (SPEC_REFCV7 §11.2 item 4)
    ({}, {"map_hires_x_max_m": 100.0}, "--map-hires", "model._map_hires.out_hw"),
    ({}, {"map_hires_x_max_m": 100.0}, "--map-hires-x-max-m",
     "model._map_hires.cfg.x_max_m"),
    ({}, {"map_hires_y_half_m": 30.0}, "--map-hires-y-half-m",
     "model._lift_bank_hires.grid_spec.y_half_m"),
    ({}, {"map_hires_x_max_m": None, "map_hires_y_half_m": None}, "--map-hires",
     "model._map_hires.lift_grid"),                 # unset = the A7 extent (100 x 30)
    # the declared checkpointing vs the built config (SPEC_REFCV7 §12 item 4)
    ({}, {"map_hires_grad_ckpt": "on"}, "--map-hires-grad-ckpt",
     "model._map_hires.cfg.grad_ckpt"),
    ({}, {"map_hires_grad_ckpt": None}, "--map-hires-grad-ckpt",
     "model._map_hires.cfg.grad_ckpt"),             # unset = ON; built off
])
def test_dvb_RED_every_disagreement_is_named(model_kw, args_kw, lever, read_from):
    got = _all_checks(_dvb_model(**model_kw), _dvb_args(**args_kw))
    assert any(x.lever == lever and read_from in x.read_from for x in got), \
        [str(x) for x in got]


def test_dvb_RED_the_class_weights_file_is_the_one_the_loss_reads(tmp_path):
    p = _weights_file(tmp_path)
    w = [0.04, 0.03, 0.7, 2.1, 16.8, 1.9, 17.9, 0.03]
    a = _dvb_args(map_hires_class_weights=str(p))
    wrong_sha = _dvb_model(cw_file=p, cfg_sha="0" * 64, tensor=torch.tensor(w))
    assert [x.read_from for x in H.dvb_check_class_weights(wrong_sha, a)] == [
        "model._map_hires.cfg.class_weights_sha256 vs sha256(argv file)"]
    wrong_vals = _dvb_model(cw_file=p, tensor=torch.ones(8))
    assert [x.read_from for x in H.dvb_check_class_weights(wrong_vals, a)] == [
        "model._map_hires_class_weight (read by the loss)"]
    no_tensor = _dvb_model(cw_file=p, tensor=None)
    assert [x.read_from for x in H.dvb_check_class_weights(no_tensor, a)] == [
        "model._map_hires_class_weight"]


def test_dvb_registration_is_EXPLICIT_and_names_the_eleven_kinds():
    got = []
    kinds = H.register_dvb_levers(lambda d, k, c: got.append((d, k, callable(c))))
    assert kinds == {"map_hires": "built", "w_map_hires": "loss",
                     "map_hires_class_weights": "built",
                     "map_hires_decision_rule": "built",
                     "map_hires_x_max_m": "built", "map_hires_y_half_m": "built",
                     "map_hires_grad_ckpt": "built",
                     "bev_source": "built", "bev_planner_crop_m": "built",
                     "map_hires_near_lift_m": "built",          # NEW-2 R2 (A12)
                     "map_hires_near_refine_blocks": "built"}   # NEW-2 R3 (A15)
    assert got == [("map_hires", "built", True), ("w_map_hires", "loss", True),
                   ("map_hires_class_weights", "built", True),
                   ("map_hires_decision_rule", "built", True),
                   ("map_hires_x_max_m", "built", True),
                   ("map_hires_y_half_m", "built", True),
                   ("map_hires_grad_ckpt", "built", True),
                   ("bev_source", "built", True),
                   ("bev_planner_crop_m", "built", True),
                   ("map_hires_near_lift_m", "built", True),
                   ("map_hires_near_refine_blocks", "built", True)]
    from tanitad.train import declared_vs_built as dvb
    # importing this module registered NOTHING (the trainer registers, at its import)
    assert not (set(H.DVB_KINDS) & set(dvb.REGISTRY)) or "refc_v3_train" in sys.modules


def test_MapHiresConfig_is_strict_for_G_HYG():
    from tanitad.train import config_hygiene as hyg
    c = H.MapHiresConfig(w_map_hires=1.0, class_weights_sha256="ab" * 32)
    assert hyg.is_strict(c) and hyg.undeclared_attributes(c) == []
    assert c.as_dict()["class_weights_sha256"] == "ab" * 32
    with pytest.raises(Exception):
        c.some_lever = 1                              # frozen: refused at assignment


# --------------------------------------------------------------------------- #
# 9. the DECLARED decision rule (map-signal audit D1): prior-corrected argmax  #
# --------------------------------------------------------------------------- #
W_MF = torch.tensor([1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def _calibrated(p_lane: float) -> torch.Tensor:
    """z_c = log(w_c P(c|x)): what a weighted CE converges to. Truth: drivable
    everywhere with P(lane) = p_lane on every cell."""
    P = torch.full((1, 8, 600, 320), 1e-9)
    P[:, 1], P[:, 2] = 1.0 - p_lane, p_lane
    return torch.log(W_MF.view(1, 8, 1, 1) * P)


def _assert_rule_recovers_drivable() -> None:
    codes = torch.full((1, 600, 320), 1, dtype=torch.uint8)       # all drivable
    s = H.per_class_signal(_calibrated(0.10), codes, class_weight=W_MF,
                           decision_rule="prior_corrected")
    # the RAW diagnostic misfires: lane on every cell at a 10 % posterior
    assert float(s["interraw"][1].sum()) == 0.0
    assert float(s["unionraw"][2].sum()) == 192_000.0
    # the DECLARED prior-corrected rule decides drivable on every cell
    assert float(s["inter"][1].sum()) == 192_000.0 == float(s["union"][1].sum())
    assert float(s["union"][2].sum()) == 0.0


def test_prior_corrected_rule_fixes_the_raw_misfire():
    _assert_rule_recovers_drivable()


def test_DELIBERATE_REGRESSION_a_rule_that_ignores_the_weights_goes_RED(monkeypatch):
    monkeypatch.setattr(H, "decide", lambda z, rule, w=None: z.argmax(dim=1))
    with pytest.raises(AssertionError):
        _assert_rule_recovers_drivable()


def _assert_zero_weight_class_is_never_decided(decide_fn) -> None:
    """A class with loss weight 0 was never supervised: the prior-corrected rule must
    never decide it -- not even where its logit is the largest."""
    z = torch.zeros(1, 8, 4, 4)
    z[:, 2] = 5.0                                   # lane's logit wins everywhere ...
    z[:, 1] = 1.0
    w = torch.ones(8)
    w[2] = 0.0                                      # ... but lane was never supervised
    got = decide_fn(z, "prior_corrected", w)
    assert int((got == 2).sum()) == 0 and int((got == 1).sum()) == 16


def test_a_zero_weight_class_is_never_decided():
    _assert_zero_weight_class_is_never_decided(H.decide)


def test_DELIBERATE_REGRESSION_the_naive_z_minus_log_0_goes_RED():
    """``z - log w`` with ``w = 0`` is ``+inf``: the naive formula decides the
    never-supervised class on EVERY cell."""
    def naive(z, rule, w):
        return (z.float() - torch.log(w).view(1, -1, 1, 1)).argmax(dim=1)
    with pytest.raises(AssertionError):
        _assert_zero_weight_class_is_never_decided(naive)


def test_the_rule_refuses_no_weights_and_is_the_identity_at_uniform_weights():
    z = torch.randn(2, 8, 600, 320)
    with pytest.raises(ValueError, match="class weights"):
        H.decide(z, "prior_corrected", None)
    with pytest.raises(ValueError):
        H.decide(z, "posterior", W_MF)
    assert torch.equal(H.decide(z, "prior_corrected", torch.ones(8)),
                       H.decide(z, "raw"))                       # control C3
    assert H.MapHiresConfig(w_map_hires=1.0).decision_rule == "prior_corrected"
    with pytest.raises(ValueError, match="decision_rule"):
        H.MapHiresConfig(w_map_hires=1.0, decision_rule="posterior")


def test_the_loss_row_uses_the_declared_rule_and_logs_the_raw_one_beside_it():
    codes = torch.full((1, 600, 320), 1, dtype=torch.uint8)
    row = H.map_hires_loss_row(_calibrated(0.10), codes, class_weight=W_MF,
                               decision_rule="prior_corrected")
    d = H.derived_per_class(row)
    assert d["map_hires_iou_drivable_0_20"] == 1.0
    assert row["map_hires_interraw_drivable_0_20"] == 0.0     # the diagnostic
    assert row["map_hires_unionraw_lane_0_20"] == 64_000.0


def test_dvb_the_decision_rule_is_read_off_the_built_config():
    m = _dvb_model()
    assert H.dvb_check_decision_rule(m, _dvb_args()) == []
    got = H.dvb_check_decision_rule(m, _dvb_args(map_hires_decision_rule="raw"))
    assert [x.read_from for x in got] == ["model._map_hires.cfg.decision_rule"]
    off = _dvb_model(on=False, tap=False)
    assert H.dvb_check_decision_rule(off, _dvb_args(map_hires="off")) == []
    assert len(H.dvb_check_decision_rule(
        off, _dvb_args(map_hires="off", map_hires_decision_rule="raw"))) == 1



# =========================================================================== #
# NEW-2 R2 (SPEC_REFCV7 A12): THE 0.1 m NEAR-RANGE LIFT                         #
# =========================================================================== #
def _a7_near(near=20.0):
    return H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0,
                            near_lift_x_m=near)


@pytest.mark.parametrize("near,msg", [
    (0.3, "multiple of 0.5"), (0.25, "multiple of 0.5"), (100.5, "past the map extent"),
    (-0.5, ">= 0"), (float("nan"), ">= 0"), (float("inf"), ">= 0")])
def test_R2_the_near_lift_range_is_validated(near, msg):
    with pytest.raises(ValueError, match=msg):
        _a7_near(near)


def test_R2_the_near_lift_is_declared_and_stamped():
    off = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
    assert off.near_lift_x_m == 0.0 and off.near_rows == 0
    d = _a7_near(20.0).as_dict()
    assert d["near_lift_x_m"] == 20.0 and d["near_lift_rows"] == 200    # literals
    assert off.as_dict()["near_lift_x_m"] == 0.0 and off.as_dict()["near_lift_rows"] == 0
    assert _a7_near(100.0).near_rows == 1000                            # the whole extent


def _near_geometry_errors(eq_rows: int) -> dict:
    """The DERIVED 0.1 m geometry vs the EXACT one (``build_lift_geometry`` on a 0.1 m grid),
    in IMAGE pixels, on the A7 lift at 20 m, nominal camera."""
    cfg = _a7_near(20.0)
    cam = RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5)
    obs = None
    if eq_rows:
        obs = torch.ones(416, 1024, dtype=torch.bool)
        obs[-eq_rows:] = False
    g25 = L.build_lift_geometry(cam, frame=FRAME_416x1024, stride=8, grid=cfg.lift_grid,
                                observed=obs)
    g10 = L.build_lift_geometry(cam, frame=FRAME_416x1024, stride=8,
                                grid=BEVGrid(x_fwd_m=20.0, y_half_m=30.0, cell_m=0.1),
                                observed=obs)
    gd, vd = H.derive_near_geometry(g25.grid.unsqueeze(0), g25.valid.unsqueeze(0),
                                    near_rows=200, out_w=600)
    gd, vd = gd[0], vd[0]
    assert tuple(gd.shape) == (4, 200, 600, 2) and tuple(vd.shape) == (4, 200, 600)
    both = vd & g10.valid
    du = (gd[..., 0] - g10.grid[..., 0]).abs() * 128 / 2 * 8      # feature -> image px
    dv = (gd[..., 1] - g10.grid[..., 1]).abs() * 52 / 2 * 8
    e = torch.maximum(du, dv)
    far = both.clone()
    far[:, :30] = False                                            # x >= 3 m
    return {"derived_only": int((vd & ~g10.valid).sum()),
            "kept": float(both.sum()) / float(g10.valid.sum()),
            "p99": float(e[both].quantile(0.99)), "max_x_ge_3m": float(e[far].max()),
            "zeros_where_invalid": bool((gd[~vd] == 0).all())}


def _assert_near_geometry_is_exact_enough(eq_rows: int) -> None:
    r = _near_geometry_errors(eq_rows)
    # MEASURED 2026-09-27 (nominal camera): p99 0.098 / 0.085 px, max over x >= 3 m 0.59 px,
    # 0 cells valid only in the derived geometry, 98.0 % of the exact valid cells kept.
    assert r["derived_only"] == 0, r                       # CONSERVATIVE validity
    assert r["kept"] >= 0.975, r
    assert r["p99"] < 0.25, r
    assert r["max_x_ge_3m"] < 1.0, r
    assert r["zeros_where_invalid"], r


@pytest.mark.parametrize("eq_rows", [0, 43])
def test_R2_the_derived_near_geometry_matches_the_exact_one(eq_rows):
    """The one design risk of deriving instead of rebuilding: pinned against the EXACT
    0.1 m geometry of the same camera. (Inside 3 m the camera-height sample sits in the
    lens's own horizontal plane -- up to ~13 px there, MEASURED -- which is why the max is
    pinned from 3 m out; no road-plane sample is involved.)"""
    _assert_near_geometry_is_exact_enough(eq_rows)


def test_R2_DELIBERATE_REGRESSION_the_naive_fine_to_coarse_index_goes_RED(monkeypatch):
    """``a * 0.4`` instead of ``(a + 0.5) * 0.4 - 0.5``: every sample 0.3 coarse cells
    (7.5 cm) off -- several pixels at 5-10 m. The accuracy pin must fail."""
    monkeypatch.setattr(H, "_fine_to_coarse_index", lambda n, r, device: torch.arange(
        int(n), device=device, dtype=torch.float32) * float(r))
    with pytest.raises(AssertionError):
        _assert_near_geometry_is_exact_enough(0)


def test_R2_DELIBERATE_REGRESSION_a_majority_validity_rule_goes_RED(monkeypatch):
    """A fine cell 'valid' when most of its weight is valid samples a sanitised-zero
    coordinate: cells valid in the derived geometry and NOT in the exact one appear."""
    monkeypatch.setattr(H, "_near_valid", lambda bad: bad < 0.5)
    with pytest.raises(AssertionError):
        _assert_near_geometry_is_exact_enough(43)


def _small_near_pair(seed=0, near=10.0, grad_ckpt=False):
    """Two branches from ONE seed on the /2 extent: NEW-2 as landed, and with the near lift."""
    kw = dict(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1, 2),
              grad_ckpt=grad_ckpt)
    torch.manual_seed(seed)
    a = H.MapHiresBranch(H.MapHiresConfig(**kw), d_image=16, image_hw=(8, 16))
    torch.manual_seed(seed)
    b = H.MapHiresBranch(H.MapHiresConfig(near_lift_x_m=near, **kw), d_image=16,
                         image_hw=(8, 16))
    frame = frame_for_width(128, 64)
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5), frame=frame,
                              stride=8, grid=a.cfg.lift_grid)
    geo = (g.grid.unsqueeze(0).expand(2, *g.grid.shape).contiguous(),
           g.valid.unsqueeze(0).expand(2, *g.valid.shape).contiguous())
    torch.manual_seed(seed + 1)
    return a, b, torch.randn(2, 16, 8, 16), geo


def _assert_near_branch_is_new2_at_init(a, b, f, geo) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(set(sb) - set(sa)) == ["near.lift.proj.bias", "near.lift.proj.weight",
                                         "near.lift.unobserved"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)             # built LAST
    with torch.no_grad():
        la = a(f, *geo)["map_hires_logits"]
        lb = b(f, *geo)["map_hires_logits"]
    assert torch.equal(la, lb)                                     # zero-initialised


def test_R2_the_near_branch_IS_the_NEW2_branch_at_step_0():
    a, b, f, geo = _small_near_pair()
    _assert_near_branch_is_new2_at_init(a, b, f, geo)
    # d_image 16 x 4 heights x d_up 16 + bias 16 + unobserved 16 -- a literal
    assert b.param_breakdown() == {"lift": a.param_breakdown()["lift"],
                                   "encoder": a.param_breakdown()["encoder"],
                                   "refine": a.param_breakdown()["refine"],
                                   "near": 1056, "total": a.param_breakdown()["total"] + 1056}
    assert "near" not in a.param_breakdown()


def test_R2_DELIBERATE_REGRESSION_a_non_zero_skip_init_goes_RED():
    a, b, f, geo = _small_near_pair()
    with torch.no_grad():
        b.near.lift.proj.weight.normal_(std=0.02)
    with pytest.raises(AssertionError):
        _assert_near_branch_is_new2_at_init(a, b, f, geo)


@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_R2_the_skip_is_MAP_ONLY_and_trains(grad_ckpt):
    """Perturbing the near lift moves the 10 cm logits and leaves the SHARED 0.25 m encoder
    output (what the planner pool reads) byte-identical; one backward reaches the skip's
    projection even from its zero init, and with the regression arm's zero source the
    projection's gradient is EXACTLY 0 while its bias still trains."""
    _, b, f, geo = _small_near_pair(grad_ckpt=grad_ckpt)
    with torch.no_grad():
        o0 = b(f, *geo)
        b.near.lift.proj.weight.fill_(0.01)
        o1 = b(f, *geo)
    assert torch.equal(o0["map_hires_bev"], o1["map_hires_bev"])   # map-only
    assert not torch.equal(o0["map_hires_logits"], o1["map_hires_logits"])
    _, b, f, geo = _small_near_pair(grad_ckpt=grad_ckpt)
    b.train()
    fr = f.clone().requires_grad_(True)
    b(fr, *geo)["map_hires_logits"].square().mean().backward()
    assert float(b.near.lift.proj.weight.grad.abs().sum()) > 0.0
    assert float(fr.grad.abs().sum()) > 0.0
    rep = H.grad_reach_report_hires(None, branch=b)
    assert sorted(rep) == ["encoder", "lift", "near", "refine"]
    assert rep["near"]["grad_abs_sum"] > 0.0 and rep["near"]["n_params"] == 1056
    b.zero_grad(set_to_none=True)
    b(f, *geo, near_source=torch.zeros_like(f))["map_hires_logits"].square().mean().backward()
    assert float(b.near.lift.proj.weight.grad.abs().sum()) == 0.0   # literal
    assert float(b.near.lift.proj.bias.grad.abs().sum()) > 0.0


def test_R2_the_near_source_seam_is_refused_where_it_cannot_apply():
    a, b, f, geo = _small_near_pair()
    with pytest.raises(ValueError, match="no near lift"):
        a(f, *geo, near_source=torch.zeros_like(f))
    with pytest.raises(ValueError, match="near_source"):
        b(f, *geo, near_source=torch.zeros(2, 16, 8, 8))
    assert sorted(H.grad_reach_report_hires(None, branch=a)) == ["encoder", "lift", "refine"]


def _dvb_near_model(near: float):
    m = _dvb_model()
    cfg = H.MapHiresConfig(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1,),
                           near_lift_x_m=near)
    m._map_hires = H.MapHiresBranch(cfg, d_image=16, image_hw=(8, 16))
    return m


def test_R2_dvb_the_near_lift_is_built_iff_declared_GREEN_and_RED():
    ok = _dvb_near_model(10.0)
    assert H.dvb_check_near_lift(ok, _dvb_args(map_hires_near_lift_m=10.0)) == []
    assert H.dvb_check_near_lift(_dvb_model(), _dvb_args()) == []          # unset = 0
    assert H.dvb_check_near_lift(_dvb_model(), _dvb_args(map_hires_near_lift_m=0.0)) == []
    off = _dvb_model(on=False, tap=False)
    assert H.dvb_check_near_lift(off, _dvb_args(map_hires="off")) == []
    # RED: declared, not built / built, not declared / built at another range / no branch
    for m, a, where in (
            (_dvb_model(), _dvb_args(map_hires_near_lift_m=10.0), "model._map_hires.near"),
            (ok, _dvb_args(), "model._map_hires.near"),
            (_dvb_near_model(5.0), _dvb_args(map_hires_near_lift_m=10.0),
             "model._map_hires.near.rows/out_w"),
            (off, _dvb_args(map_hires="off", map_hires_near_lift_m=10.0),
             "model._map_hires")):
        got = H.dvb_check_near_lift(m, a)
        assert got and all(x.lever == "--map-hires-near-lift-m" for x in got), got
        assert any(x.read_from == where for x in got), [x.read_from for x in got]
    st = H.built_state(ok)
    assert st["near_lift_x_m"] == 10.0 and st["near_lift_built"] is True
    assert H.built_state(_dvb_model())["near_lift_built"] is False



# =========================================================================== #
# NEW-2 R3 (SPEC_REFCV7 §20, A15): THE NEAR REFINE BLOCK (the decoder lever)   #
# =========================================================================== #
@pytest.mark.parametrize("kw,msg", [
    ({"near_refine_blocks": -1}, "integer in"), ({"near_refine_blocks": 5}, "integer in"),
    ({"near_refine_blocks": 1.5}, "integer in"), ({"near_refine_blocks": True}, "integer in"),
    ({"near_refine_blocks": 1, "near_lift_x_m": 0.0}, "without the near lift")])
def test_R3_the_block_count_is_validated(kw, msg):
    base = {"near_lift_x_m": 10.0}
    with pytest.raises(ValueError, match=msg):
        H.MapHiresConfig(w_map_hires=1.0, **{**base, **kw})


def test_R3_the_block_is_declared_and_stamped():
    c = H.MapHiresConfig(w_map_hires=1.0, near_lift_x_m=10.0, near_refine_blocks=1)
    assert c.as_dict()["near_refine_blocks"] == 1                        # literal
    assert H.MapHiresConfig(w_map_hires=1.0).as_dict()["near_refine_blocks"] == 0
    assert H.NearRefineBlock.DILATIONS == (2, 4)


def _r2_r3_pair(seed=0, grad_ckpt=False):
    """R2 (the near lift) and R3 (+ one near refine block) from ONE seed."""
    kw = dict(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1, 2),
              grad_ckpt=grad_ckpt, near_lift_x_m=10.0)
    torch.manual_seed(seed)
    a = H.MapHiresBranch(H.MapHiresConfig(**kw), d_image=16, image_hw=(8, 16))
    torch.manual_seed(seed)
    b = H.MapHiresBranch(H.MapHiresConfig(near_refine_blocks=1, **kw), d_image=16,
                         image_hw=(8, 16))
    frame = frame_for_width(128, 64)
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5), frame=frame,
                              stride=8, grid=a.cfg.lift_grid)
    geo = (g.grid.unsqueeze(0).expand(2, *g.grid.shape).contiguous(),
           g.valid.unsqueeze(0).expand(2, *g.valid.shape).contiguous())
    torch.manual_seed(seed + 1)
    return a, b, torch.randn(2, 16, 8, 16), geo


def _assert_r3_is_r2_at_init(a, b, f, geo) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(set(sb) - set(sa)) == ["near_refine.0.c1.weight", "near_refine.0.c2.weight",
                                         "near_refine.0.n1.bias", "near_refine.0.n1.weight"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)             # built LAST
    with torch.no_grad():
        la = a(f, *geo)["map_hires_logits"]
        lb = b(f, *geo)["map_hires_logits"]
    assert torch.equal(la, lb)                                     # zero-init last conv


def test_R3_the_decoder_arm_IS_R2_at_step_0():
    a, b, f, geo = _r2_r3_pair()
    _assert_r3_is_r2_at_init(a, b, f, geo)
    # 2 x (3x3 x 16 x 16) + GN 2 x 16 -- a literal
    assert b.param_breakdown()["near_refine"] == 4640
    assert b.param_breakdown()["total"] == a.param_breakdown()["total"] + 4640


def test_R3_DELIBERATE_REGRESSION_a_non_zero_last_conv_goes_RED():
    a, b, f, geo = _r2_r3_pair()
    with torch.no_grad():
        b.near_refine[0].c2.weight.normal_(std=0.02)
    with pytest.raises(AssertionError):
        _assert_r3_is_r2_at_init(a, b, f, geo)


@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_R3_the_block_is_MAP_ONLY_reaches_only_the_near_rows_and_trains(grad_ckpt):
    """Perturbing the block moves the logits and leaves map_hires_bev byte-identical; one
    backward reaches the block from its zero init; with the regression arm's zero input, the
    block takes EXACTLY zero gradient (it is the A12 function, and stays it)."""
    _, b, f, geo = _r2_r3_pair(grad_ckpt=grad_ckpt)
    with torch.no_grad():
        o0 = b(f, *geo)
        b.near_refine[0].c2.weight.fill_(0.01)
        o1 = b(f, *geo)
    assert torch.equal(o0["map_hires_bev"], o1["map_hires_bev"])   # map-only
    assert not torch.equal(o0["map_hires_logits"], o1["map_hires_logits"])
    _, b, f, geo = _r2_r3_pair(grad_ckpt=grad_ckpt)
    b.train()
    b(f, *geo)["map_hires_logits"].square().mean().backward()
    assert float(b.near_refine[0].c2.weight.grad.abs().sum()) > 0.0
    rep = H.grad_reach_report_hires(None, branch=b)
    assert sorted(rep) == ["encoder", "lift", "near", "near_refine", "refine"]
    assert rep["near_refine"]["n_params"] == 4640
    b.zero_grad(set_to_none=True)
    b(f, *geo, near_block_zeros=True)["map_hires_logits"].square().mean().backward()
    for n_, p_ in b.near_refine.named_parameters():
        assert p_.grad is None or float(p_.grad.abs().sum()) == 0.0, n_   # literal
    assert float(b.near.lift.proj.weight.grad.abs().sum()) > 0.0        # the lift still trains


def test_R3_the_block_seam_is_refused_where_it_cannot_apply():
    a, b, f, geo = _r2_r3_pair()
    with pytest.raises(ValueError, match="no near refine block"):
        a(f, *geo, near_block_zeros=True)


def _dvb_r3_model(blocks: int, near: float = 10.0):
    m = _dvb_model()
    cfg = H.MapHiresConfig(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1,),
                           near_lift_x_m=near, near_refine_blocks=blocks)
    m._map_hires = H.MapHiresBranch(cfg, d_image=16, image_hw=(8, 16))
    return m


def test_R3_dvb_the_block_is_built_iff_declared_GREEN_and_RED():
    ok = _dvb_r3_model(1)
    a1 = _dvb_args(map_hires_near_lift_m=10.0, map_hires_near_refine_blocks=1)
    assert H.dvb_check_near_refine(ok, a1) == []
    assert H.dvb_check_near_refine(_dvb_model(), _dvb_args()) == []        # unset = 0
    off = _dvb_model(on=False, tap=False)
    assert H.dvb_check_near_refine(off, _dvb_args(map_hires="off")) == []
    for m, a, where in (
            (_dvb_r3_model(0), a1, "model._map_hires.near_refine"),     # declared, not built
            (ok, _dvb_args(map_hires_near_lift_m=10.0), "model._map_hires.near_refine"),
            (_dvb_r3_model(2), a1, "model._map_hires.near_refine"),     # another count
            (_dvb_model(), a1, "model._map_hires.near"),                # no near lift built
            (off, _dvb_args(map_hires="off", map_hires_near_refine_blocks=1),
             "model._map_hires")):
        got = H.dvb_check_near_refine(m, a)
        assert got and all(x.lever == "--map-hires-near-refine-blocks" for x in got), got
        assert any(x.read_from == where for x in got), [x.read_from for x in got]
    # RED: a block of another design (dilations) on the SAME declared count
    bad = _dvb_r3_model(1)
    blk = bad._map_hires.near_refine[0]
    blk.c2 = nn.Conv2d(16, 16, 3, padding=1, dilation=1, bias=False)
    assert any(x.read_from == "model._map_hires.near_refine[0]"
               for x in H.dvb_check_near_refine(bad, a1))
    st = H.built_state(ok)
    assert st["near_refine_blocks"] == 1 and st["near_refine_built"] == 1
