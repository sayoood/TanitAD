"""refcv7 NEW-2 -- the WIRING: the trunk's stride-8 tap, the ``fmap_s8`` pass-through
in ``RefCModel.forward``, and the trainer's ``--map-hires`` pins and loss block.

⛔ This file lands TOGETHER with the shared-file edits it pins
(``tanitad/models/timm_trunk.py``, ``tanitad/refs/refc.py``,
``scripts/refc_v3_train.py``); on a tree without them it fails at the first tap call.

What is pinned, each with its discriminating control or deliberate-regression arm:

1. the tap is REFUSED without frozen BN, reads its stage from ``feature_info``, and is
   the SAME function as the main pass's stride-8 stage on the newest frame -- values
   AND trunk gradients -- also under chunked checkpointing and frame dedup;
2. tap OFF: ``out["fmap_s8"] is None`` and every other output is bit-identical to
   tap ON; the state_dict keys never change;
3. ⛔⛔ THE F3-WHITELIST CLASS: with ``--map-hires on`` the trainer's loss block
   REFUSES when ``fmap_s8`` does not arrive. The deliberate-regression arm is the
   historical defect verbatim -- the production forward's output WITHOUT the key --
   and the liveness check must go RED;
4. every ``--map-hires`` refusal, each with its GREEN control, and the reverse
   refusal (a weight / a weights file supplied with the branch off);
5. ⭐ SPEC_REFCV7 A6/A7: ONE lift in the forward feeds the 10 cm map AND the planner's
   pooled 120 x 64 BEV (a consumer's gradient reaches the shared lift/encoder and the
   stride-8 stage, not the map-only decoder); a second (stride-16) lift is refused;
   G-DVB's red arms -- a consumer still wired to a stride-16 lift, an uncropped
   200 x 120 planner grid -- go RED; the eval loader rebuilds an A6 model strictly;
   the Watch contract's 40 literal keys at the 100 m x +-30 m extent.
"""
from __future__ import annotations

import dataclasses as dc
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T                                         # noqa: E402
from tanitad.data.rig_projection import RigCamera                 # noqa: E402
from tanitad.data.semantic_map_gt_fine import FINE_CLASSES        # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id             # noqa: E402
from tanitad.models import map_head_hires as H                    # noqa: E402
from tanitad.models import timm_trunk as TT                       # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width           # noqa: E402
from tanitad.refs import refc                                     # noqa: E402
from tanitad.refs import refc_v3 as v3                            # noqa: E402


def _trunk(**kw):
    base = dict(model_name="resnet18.a1_in1k", pretrained=False,
                verify_imagenet_stats=False, frames=3, image_hw=(64, 128),
                frozen_bn=True)
    base.update(kw)
    return TT.TimmResNetTrunk(TT.TimmTrunkConfig(**base))


# =========================================================================== #
# 1. the tap                                                                  #
# =========================================================================== #
def test_tap_refuses_without_frozen_bn():
    t = _trunk(frozen_bn=False)
    with pytest.raises(ValueError, match="frozen BatchNorm"):
        t.enable_s8_tap()
    assert t.s8_tap is False


def test_tap_reads_its_stage_from_feature_info():
    t = _trunk()
    keys = set(t.state_dict())
    st = t.enable_s8_tap()
    assert st["s8_module"] == "layer2" and st["s8_dim"] == 128      # resnet18
    assert st["s8_hw"] == [8, 16] and st["s8_frame"] == "newest"
    assert set(t.state_dict()) == keys                              # no new tensor
    assert "s8" in t.provenance() and "s8" not in _trunk().provenance()


def _hooked_main_s8(t, x):
    """The main pass's stride-8 stage output, captured by a hook, per FRAME SLOT."""
    got = []
    h = t.net[t.s8_stage()["module"]].register_forward_hook(
        lambda m, i, o: got.append(o))
    try:
        s16, s32, pooled = t.forward_features(x)
    finally:
        h.remove()
    return torch.cat(got, dim=0), (s16, s32, pooled)


@pytest.mark.parametrize("kw", [{}, {"chunk_ckpt": 2}, {"dedup_frames": True}])
def test_tap_is_the_main_pass_stride8_stage_on_the_newest_frame(kw):
    torch.manual_seed(0)
    t = _trunk(**kw)
    t.enable_s8_tap()
    x = torch.rand(4, 9, 64, 128)
    x[1] = x[0]                                                     # dedup has work
    rows = torch.tensor([3, 1])
    main, _ = _hooked_main_s8(t, x)                 # [slots, C8, 8, 16]
    if kw.get("dedup_frames"):
        # dedup computes each DISTINCT frame once; recompute the plain reference
        ref_t = _trunk()
        ref_t.load_state_dict(t.state_dict())
        ref_t.enable_s8_tap()
        main, _ = _hooked_main_s8(ref_t, x)
    want = main.reshape(4, 3, *main.shape[1:])[rows, -1]           # newest frame
    s8, s16, s32, pooled = t.forward_features_s8(x, rows)
    torch.testing.assert_close(s8, want, rtol=1e-5, atol=1e-5)
    # ... and the main outputs are forward_features' own, bit for bit
    a16, a32, ap = t.forward_features(x)
    assert torch.equal(a16, s16) and torch.equal(a32, s32) and torch.equal(ap, pooled)


def test_tap_gradient_equals_the_main_pass_gradient_on_that_frame():
    """Same function => same trunk gradient: d(sum(R * s8_tap))/dW equals
    d(sum(R * main_s8[newest]))/dW on every stem..stride-8 parameter."""
    torch.manual_seed(0)
    t = _trunk()
    t.enable_s8_tap()
    x = torch.rand(2, 9, 64, 128)
    R = torch.randn(2, 128, 8, 16)
    params = [p for n, p in t.net.named_parameters()
              if n.split(".")[0] in ("conv1", "bn1", "layer1", "layer2")]
    s8 = t.forward_features_s8(x, torch.tensor([0, 1]))[0]
    g_tap = torch.autograd.grad((s8 * R).sum(), params)
    main, _ = _hooked_main_s8(t, x)
    newest = main.reshape(2, 3, *main.shape[1:])[:, -1]
    g_main = torch.autograd.grad((newest * R).sum(), params)
    for a, b in zip(g_tap, g_main):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert sum(float(g.abs().sum()) for g in g_tap) > 0.0


# =========================================================================== #
# 2. the pass-through in RefCModel.forward                                    #
# =========================================================================== #
def _model(frozen_bn=True):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(
        cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
        trunk_pretrained=False, in_channels=9, image_size=64, image_width=None,
        trunk_frozen_bn=frozen_bn)
    return v3.RefCV3Model(cfg), cfg


def _frames(model, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    enc = model.cfg.core.encoder
    h, w = enc.image_hw()
    return torch.rand(b, int(model.cfg.core.window), int(enc.in_channels), h, w,
                      generator=g)


def _fwd(model, frames):
    b = frames.shape[0]
    return model(frames, nav_cmd=torch.zeros(b, dtype=torch.long),
                 v0=torch.tensor([4.0] * b))


def test_tap_off_emits_None_and_tap_on_changes_nothing_else():
    torch.manual_seed(0)
    model, _ = _model()
    model.eval()
    fr = _frames(model)
    with torch.no_grad():
        off = _fwd(model, fr)
    assert "fmap_s8" in off and off["fmap_s8"] is None
    keys = set(model.state_dict())
    model.core.encoder.enable_s8_tap()
    with torch.no_grad():
        on = _fwd(model, fr)
    assert set(model.state_dict()) == keys
    f8 = on["fmap_s8"]
    assert tuple(f8.shape) == (2, 128, 8, 8)
    # the newest frame of the LAST window row, through the tap directly
    enc = model.core.encoder
    with torch.no_grad():
        want = enc.s8_from_normalised(enc.normalise(fr[:, -1]), torch.arange(2))
    torch.testing.assert_close(f8, want, rtol=1e-5, atol=1e-5)
    diff = [k for k, v in off.items()
            if torch.is_tensor(v) and not torch.equal(v, on[k])]
    assert not diff, f"tap ON changed outputs {diff}"


# =========================================================================== #
# 3. A6: the ONE lift runs in the forward and feeds the map AND the planner    #
# =========================================================================== #
CLIP = "synthetic-map-hires-clip"
from tanitad.models import refcv6_perception_branch as PB           # noqa: E402
from tanitad.data.semantic_map_gt_fine import (                     # noqa: E402
    EXTENT_REFCV7, EXTENT_V2)


def _attach(model, tmp_path=None, w=1.0, *, pool=False, extent=EXTENT_V2,
            grad_ckpt=False, near=0.0):
    """The trainer's build order under --map-hires on: the 10 cm branch first, then
    (with a BEV consumer) the perception branch whose planner pool reads it.
    ``near`` (NEW-2 R2, A12): ``MapHiresConfig.near_lift_x_m``."""
    model.core.encoder.enable_s8_tap()
    hcfg = H.MapHiresConfig(w_map_hires=w, x_max_m=float(extent.x_max_m),
                            y_half_m=float(extent.y_half_m), d_lift=16, d_model=16,
                            d_up=16, dilations=(1, 2), grad_ckpt=grad_ckpt,
                            near_lift_x_m=float(near))
    model._map_hires = H.build_map_hires_branch(model, hcfg)
    model._w_map_hires = float(w)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank(
        {CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=hcfg)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0              # as the trainer sets them (A6)
    if pool:
        model._perception = PB.build_perception_branch(
            model, PB.PerceptionBranchConfig(w_map=0.0, w_box3d=0.0,
                                             bev_source="map_hires_pool"))
    return hcfg


def _geom(model, b=2):
    g, v = model._lift_bank_hires.for_episodes([int(stable_episode_id(CLIP))] * b)
    return {"perception_grid": g, "perception_valid": v}


def _fwd_hi(model, frames):
    b = frames.shape[0]
    return model(frames, nav_cmd=torch.zeros(b, dtype=torch.long),
                 v0=torch.tensor([4.0] * b), **_geom(model, b))


def _batch(b=2, labelled=True, shape=(600, 320)):
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (b,) + tuple(shape), generator=g, dtype=torch.uint8)
    codes[:, :40] = 255
    return {"map_fine": codes,
            "map_fine_label": torch.tensor([labelled] * b),
            "map_ep": torch.tensor([int(stable_episode_id(CLIP))] * b)}


def test_the_one_lift_feeds_the_10cm_map_and_the_planner_pool():
    """A6 (SPEC_REFCV7 §11.1): one forward, one lift. The 10 cm logits AND the pooled
    120 x 64 BEV ride out on out["perception"]; a planner-side loss reaches the shared
    lift + encoder + the trunk's stride-8 stage and NOT the map-only decoder; the map
    loss reaches the decoder and the SAME shared encoder."""
    torch.manual_seed(0)
    model, _ = _model()
    _attach(model, pool=True)
    model.train()
    out = _fwd_hi(model, _frames(model))
    po = out["perception"]
    assert tuple(po["map_hires_logits"].shape) == (2, 8, 600, 320)
    assert tuple(po["bev_feats"].shape) == (2, 96, 120, 64)        # refcv6's planner grid
    assert out["fmap_s8"] is not None
    model.zero_grad(set_to_none=True)
    po["bev_feats"].square().mean().backward()                      # a consumer's loss
    rep = H.grad_reach_report_hires(model)
    assert rep["lift"]["grad_abs_sum"] > 0 and rep["encoder"]["grad_abs_sum"] > 0
    assert rep["trunk_s8_stage"]["grad_abs_sum"] > 0
    assert rep["refine"]["grad_abs_sum"] == 0.0                      # map-only
    pr = PB.grad_reach_report(model)
    assert pr["bev_pool"]["grad_abs_sum"] > 0 and "lift" not in pr
    model.zero_grad(set_to_none=True)
    out = _fwd_hi(model, _frames(model))
    out["perception"]["map_hires_logits"].square().mean().backward()
    rep = H.grad_reach_report_hires(model)
    assert rep["refine"]["grad_abs_sum"] > 0 and rep["encoder"]["grad_abs_sum"] > 0
    assert PB.grad_reach_report(model)["bev_pool"]["grad_abs_sum"] == 0.0


def test_the_hook_refuses_a_missing_geometry_and_a_second_lift():
    model, _ = _model()
    _attach(model, pool=True)
    fr = _frames(model)
    with pytest.raises(ValueError, match="0.25 m lift"):
        _fwd(model, fr)                                    # no perception geometry
    two, _ = _model()
    _attach(two)
    two._perception = PB.build_perception_branch(
        two, PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0))   # a stride-16 lift
    with pytest.raises(ValueError, match="ONE lift"):
        _fwd_hi(two, _frames(two))


def _assert_hires_live() -> None:
    torch.manual_seed(0)
    model, _ = _model()
    _attach(model)
    model.train()
    out = _fwd_hi(model, _frames(model))
    extra = {}
    term = T._map_hires_loss(model, out, _batch(), "cpu", extra)
    assert term is not None and torch.isfinite(term), term
    assert extra["n_map_hires_cells"] > 0 and "map_hires" in extra
    model.zero_grad(set_to_none=True)
    term.backward()
    rep = H.grad_reach_report_hires(model)
    for part in ("lift", "encoder", "refine", "trunk_s8_stage"):
        assert rep[part]["grad_abs_sum"] > 0.0, (part, rep[part])


def test_GREEN_the_10cm_term_is_live_and_reaches_the_stride8_stage():
    _assert_hires_live()


def test_DELIBERATE_REGRESSION_the_forward_drops_fmap_s8_and_goes_RED(monkeypatch):
    """The historical F3 defect, re-introduced for this key: the production forward's
    output WITHOUT `fmap_s8`. The block must REFUSE (SystemExit); if it skipped
    silently the liveness assertions would fail instead. Either way: RED."""
    orig = refc.RefCModel.forward

    def as_shipped(self, *a, **k):
        o = orig(self, *a, **k)
        o.pop("fmap_s8", None)
        return o
    monkeypatch.setattr(refc.RefCModel, "forward", as_shipped)
    with pytest.raises(SystemExit, match="fmap_s8"):
        _assert_hires_live()


def test_the_block_refuses_a_batch_without_the_10cm_target_or_the_logits():
    model, _ = _model()
    _attach(model)
    out = _fwd_hi(model, _frames(model))
    b = _batch()
    b.pop("map_fine")
    with pytest.raises(SystemExit, match="map_fine"):
        T._map_hires_loss(model, out, b, "cpu", {})
    no_logits = dict(out, perception={})
    with pytest.raises(SystemExit, match="10 cm logits"):
        T._map_hires_loss(model, no_logits, _batch(), "cpu", {})


def test_off_is_absent_from_the_graph_and_the_log():
    model, _ = _model()
    out = _fwd(model, _frames(model))
    extra = {}
    assert T._map_hires_loss(model, out, {}, "cpu", extra) is None and extra == {}


def test_unlabelled_windows_are_a_counted_zero():
    model, _ = _model()
    _attach(model)
    out = _fwd_hi(model, _frames(model))
    extra = {}
    term = T._map_hires_loss(model, out, _batch(labelled=False), "cpu", extra)
    assert float(term.detach()) == 0.0 and extra["n_map_hires_cells"] == 0.0
    assert extra["map_hires_n_labelled"] == 0.0 and extra["map_hires_n_windows"] == 2.0


# =========================================================================== #
# 4. the pins                                                                 #
# =========================================================================== #
ON = ["--arm", "hier", "--size", "tiny", "--out", "X", "--trunk", "timm",
      "--trunk-frozen-bn", "--map-gt-root", "R",
      "--agent-rig-camera", "extrinsics", "--agent-rig-extrinsics", "E",
      "--map-hires", "on", "--w-map-hires", "1.0",
      "--map-hires-class-weights", "W.json"]
POOL = ON + ["--w-box3d", "1.0", "--agent-join", "J", "--bev-source", "map_hires_pool"]


def _pin(argv):
    return T._pin_map_hires(None, T.build_parser().parse_args(argv))


def _without(argv, flag, n=1):
    i = argv.index(flag)
    return argv[:i] + argv[i + 1 + n:]


def test_GREEN_the_on_combinations_and_the_default_do_not_refuse():
    _pin(ON)                                     # a 10 cm map with no BEV consumer
    _pin(POOL)                                   # A6: box3d reads the pooled BEV
    _pin(ON + ["--map-hires-x-max-m", "60", "--map-hires-y-half-m", "16",
               "--map-hires-grad-ckpt", "off", "--bev-planner-crop-m", "60", "16"])
    _pin(["--arm", "hier", "--out", "X"])
    a = T.build_parser().parse_args(ON)
    assert H.declared_extent(a) == EXTENT_REFCV7          # unset = the A7 extent
    assert H.declared_grad_ckpt(a) is True                # unset = ON (A7 item 4)
    # refcv6's pin must not call the SAM3 root dead when the 10 cm map reads it
    T._pin_refcv6_perception(None, a)


@pytest.mark.parametrize("argv,needle", [
    (_without(ON, "--w-map-hires"), "--w-map-hires 0"),
    (_without(ON, "--trunk"), "--trunk timm"),
    (_without(ON, "--trunk-frozen-bn", 0), "--trunk-frozen-bn"),
    (ON + ["--w-map", "1.0"], "REMOVED"),
    (_without(ON, "--map-gt-root"), "NO LABELS"),
    (_without(ON, "--agent-rig-camera"), "--agent-rig-camera extrinsics"),
    (_without(ON, "--map-hires-class-weights"), "--map-hires-class-weights"),
    (ON + ["--w-box3d", "1.0", "--agent-join", "J"], "--bev-source map_hires_pool"),
    (ON + ["--bev-source", "map_hires_pool"], "NO BEV consumer"),
    (POOL + ["--bev-planner-crop-m", "100", "30"], "PI decision"),
    (ON + ["--map-hires-x-max-m", "55"], "smaller than"),
    (["--arm", "hier", "--out", "X", "--w-map-hires", "1.0"], "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--map-hires-class-weights", "W"],
     "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--map-hires-x-max-m", "100"], "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--map-hires-grad-ckpt", "on"], "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--bev-source", "map_hires_pool"],
     "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--bev-planner-crop-m", "60", "16"],
     "READ BY NOTHING"),
])
def test_every_dead_combination_refuses(argv, needle):
    with pytest.raises(SystemExit) as e:
        _pin(argv)
    assert needle in str(e.value), str(e.value)[:300]


def test_the_weight_gate_row():
    met, why = T.REFC_WEIGHT_GATES["w_map_hires"]["gate"](
        T.build_parser().parse_args(ON))
    assert met is True
    met, why = T.REFC_WEIGHT_GATES["w_map_hires"]["gate"](
        T.build_parser().parse_args(["--arm", "hier", "--out", "X"]))
    assert met is False and "--map-hires on" in why


def test_the_trainer_loads_a_real_weights_file_and_refuses_a_dry_run(tmp_path):
    good = {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
            "weights": [1.0] * 8, "dry_run": False}
    p = tmp_path / "w.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    H.load_class_weights(p)
    p.write_text(json.dumps(dict(good, dry_run=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="DRY RUN"):
        H.load_class_weights(p)


# =========================================================================== #
# 5. the eval loader rebuilds an A6 model strictly                             #
# =========================================================================== #
def _arm_module():
    import importlib.util
    src = ROOT.parent / "taniteval" / "tools" / "refcv3_arm.py"
    spec = importlib.util.spec_from_file_location("refcv3_arm_a6", src)
    A = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_a6"] = A
    spec.loader.exec_module(A)
    return A


def test_the_eval_loader_rebuilds_an_A6_model_strictly():
    """The map-hires branch FIRST, then the pool reading it; strict 0/0."""
    A = _arm_module()
    model, _ = _model()
    hcfg = _attach(model, pool=True)
    cw = {"weights": [1.0] * 8, "sha256": "ab" * 32}
    tap = model.core.encoder.enable_s8_tap()
    stamp = {"map_hires": {**hcfg.as_dict(), "class_weights": cw,
                           "trunk_tap": {k: tap[k] for k in ("s8_module", "s8_dim",
                                                             "s8_hw")},
                           "branch_params": model._map_hires.param_breakdown()},
             "refcv6_perception": {**model._perception.cfg.as_dict(),
                                   "branch_params": model._perception.param_breakdown()}}
    fresh, _ = _model()
    A.rebuild_map_hires_branch(fresh, stamp, "cpu")
    A.rebuild_perception_branch(fresh, stamp, "cpu")
    res = fresh.load_state_dict(model.state_dict(), strict=False)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    assert fresh._perception.cfg.bev_source == "map_hires_pool"
    assert fresh._perception.lift is None and fresh._perception.bev_pool is not None


# =========================================================================== #
# 6. SPEC_REFCV7 A3/A7: the in-run per-class logging (G-DVB) and the per-class #
#    signal (G-LIVE), on the trainer's own row builders                        #
# =========================================================================== #
def _extras(model, seed):
    out = _fwd_hi(model, _frames(model, seed=seed))
    hw = tuple(model._map_hires.cfg.out_hw)
    b = _batch(shape=hw)
    g = torch.Generator().manual_seed(seed)
    b["map_fine"] = torch.randint(0, 8, (2,) + hw, generator=g, dtype=torch.uint8)
    b["map_fine"][:, :40] = 255
    extra = {}
    with torch.no_grad():
        T._map_hires_loss(model, out, b, "cpu", extra)
    return extra


def _eval_row(model):
    """Two real eval batches through the trainer's accumulation and row builder."""
    acc = {}
    for el in (_extras(model, 1), _extras(model, 2)):
        for k, v in el.items():
            if torch.is_tensor(v) and v.ndim == 0:
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            elif isinstance(v, (int, float, bool)):
                acc[k] = acc.get(k, 0.0) + float(v)
    return T._eval_row_from_acc(acc, 2, model), acc


def test_the_eval_row_carries_all_8x3_classes_pooled_and_drivable_only_FAILS():
    model, _ = _model()
    _attach(model)
    model.eval()
    e1, e2 = _extras(model, 1), _extras(model, 2)
    acc = {}
    for el in (e1, e2):                                # the eval loop's accumulation
        for k, v in el.items():
            if torch.is_tensor(v) and v.ndim == 0:
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            elif isinstance(v, (int, float, bool)):
                acc[k] = acc.get(k, 0.0) + float(v)
    erow = T._eval_row_from_acc(acc, 2, model)
    assert H.missing_per_class_keys(erow, "eval_") == []
    k_i = "map_hires_inter_lane_0_20"
    k_u = "map_hires_union_lane_0_20"
    want = (e1[k_i] + e2[k_i]) / (e1[k_u] + e2[k_u])
    assert erow["eval_map_hires_iou_lane_0_20"] == pytest.approx(want, rel=1e-12)
    # the raw per-class floats are NOT rounded to 5 dp
    k_g = "map_hires_gn_lane_0_20"
    assert erow["eval_" + k_g] == (e1[k_g] + e2[k_g]) / 2
    # ⛔ the refcv6 shape -- drivable only -- is what the check exists to FAIL
    refcv6_like = {"eval_map": 1.0, "eval_map_iou_drivable": 0.6}
    assert len(H.missing_per_class_keys(refcv6_like, "eval_")) == 48


def test_a_default_run_eval_row_is_unchanged():
    model, _ = _model()
    acc = {"loss": 3.0000049, "traj": 1.25, "n_x": 4.0}
    assert T._eval_row_from_acc(acc, 2, model) == {
        "eval_loss": 1.5, "eval_traj": 0.625, "eval_n_x": 2.0}


def test_a_default_run_train_row_is_unchanged():
    """`--map-hires off`: the factored train-row builder is the tip's row, key for key
    (5 dp everywhere, plain scalars pass through, non-scalar tensors dropped)."""
    from types import SimpleNamespace as _NS
    losses = {"loss": torch.tensor(1.234567, dtype=torch.float64), "traj": 0.1234567,
              "flag": True, "vec": torch.ones(3), "f32": torch.tensor(3.0000049)}
    row = T._train_row_scalars(losses, _NS(_map_hires=None))
    # float32(3.0000049) is 3.0000050068 -> 3.00001 at 5 dp (the tip rounds the same)
    assert row == {"loss": 1.23457, "traj": 0.12346, "flag": 1.0, "f32": 3.00001}
    # ... and it is the tip's own comprehension (b3f7ea6 refc_v3_train.py), verbatim
    tip = {k: (round(float(v.detach()), 5) if torch.is_tensor(v)
               else round(float(v), 5))
           for k, v in losses.items()
           if (torch.is_tensor(v) and v.ndim == 0)
           or isinstance(v, (int, float, bool))}
    assert row == tip and list(row) == list(tip)


# ⛔ The launch gate's Watch contract (the Master Mind, 2026-09-27, SPEC_REFCV7 §12):
# at the A7 extent (100 m x +-30 m) every class in EVERY 20 m band -- these 40 names,
# written out LITERALLY -- never generated from CLASS_KEYS / band keys -- so that a
# rename inside the module goes RED here instead of silently renaming the Watch's input.
WATCH_CONTRACT_KEYS = (
    "eval_map_hires_iou_nocls_0_20", "eval_map_hires_iou_nocls_20_40",
    "eval_map_hires_iou_nocls_40_60", "eval_map_hires_iou_nocls_60_80",
    "eval_map_hires_iou_nocls_80_100",
    "eval_map_hires_iou_drivable_0_20", "eval_map_hires_iou_drivable_20_40",
    "eval_map_hires_iou_drivable_40_60", "eval_map_hires_iou_drivable_60_80",
    "eval_map_hires_iou_drivable_80_100",
    "eval_map_hires_iou_lane_0_20", "eval_map_hires_iou_lane_20_40",
    "eval_map_hires_iou_lane_40_60", "eval_map_hires_iou_lane_60_80",
    "eval_map_hires_iou_lane_80_100",
    "eval_map_hires_iou_crosswalk_0_20", "eval_map_hires_iou_crosswalk_20_40",
    "eval_map_hires_iou_crosswalk_40_60", "eval_map_hires_iou_crosswalk_60_80",
    "eval_map_hires_iou_crosswalk_80_100",
    "eval_map_hires_iou_arrow_0_20", "eval_map_hires_iou_arrow_20_40",
    "eval_map_hires_iou_arrow_40_60", "eval_map_hires_iou_arrow_60_80",
    "eval_map_hires_iou_arrow_80_100",
    "eval_map_hires_iou_edge_0_20", "eval_map_hires_iou_edge_20_40",
    "eval_map_hires_iou_edge_40_60", "eval_map_hires_iou_edge_60_80",
    "eval_map_hires_iou_edge_80_100",
    "eval_map_hires_iou_hatched_0_20", "eval_map_hires_iou_hatched_20_40",
    "eval_map_hires_iou_hatched_40_60", "eval_map_hires_iou_hatched_60_80",
    "eval_map_hires_iou_hatched_80_100",
    "eval_map_hires_iou_sidewalk_0_20", "eval_map_hires_iou_sidewalk_20_40",
    "eval_map_hires_iou_sidewalk_40_60", "eval_map_hires_iou_sidewalk_60_80",
    "eval_map_hires_iou_sidewalk_80_100",
)


def _a7_eval_row(extent=EXTENT_REFCV7):
    """The trainer's eval-row builder on a model whose 10 cm branch is at ``extent``,
    fed two batches of the per-class signal the loss block logs (a random head on a
    random scene at 0.1 m over the extent). No forward: the row builder is the unit."""
    from types import SimpleNamespace as _NS
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=float(extent.x_max_m),
                            y_half_m=float(extent.y_half_m))
    acc = {}
    for seed in (1, 2):
        g = torch.Generator().manual_seed(seed)
        codes = torch.randint(0, 8, (1,) + tuple(hcfg.out_hw), generator=g,
                              dtype=torch.uint8)
        lg = torch.randn((1, 8) + tuple(hcfg.out_hw), generator=g)
        row = H.per_class_log_values(H.per_class_signal(
            lg, codes, class_weight=torch.ones(8), decision_rule="prior_corrected"))
        for k, v in row.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    return T._eval_row_from_acc(acc, 2, _NS(_map_hires=_NS(cfg=hcfg)))


def _watch_contract_violations(erow) -> list:
    """Missing contract keys, contract keys that are not an IoU in [0, 1], and any
    OTHER `eval_map_hires_iou_*` key (a renamed class would log beside the contract)."""
    bad = [k for k in WATCH_CONTRACT_KEYS if k not in erow]
    bad += [k for k in WATCH_CONTRACT_KEYS if k in erow and not (
        isinstance(erow[k], float) and 0.0 <= erow[k] <= 1.0)]
    bad += sorted(k for k in erow if k.startswith("eval_map_hires_iou_")
                  and k not in WATCH_CONTRACT_KEYS)
    return bad


def test_WATCH_CONTRACT_the_40_literal_eval_iou_keys_are_logged():
    assert len(set(WATCH_CONTRACT_KEYS)) == 40
    erow = _a7_eval_row()
    assert _watch_contract_violations(erow) == []
    back = json.loads(json.dumps(erow))                  # the log line is json.dumps
    assert [k for k in WATCH_CONTRACT_KEYS if k not in back] == []


def test_DELIBERATE_REGRESSION_a_60m_branch_or_a_renamed_class_breaks_the_contract(
        monkeypatch):
    short = _watch_contract_violations(_a7_eval_row(EXTENT_V2))
    assert {f"eval_map_hires_iou_{c}_{b}" for c in ("lane", "edge")
            for b in ("60_80", "80_100")} <= set(short), short
    monkeypatch.setattr(H, "CLASS_KEYS", tuple("lane_line" if k == "lane" else k
                                               for k in H.CLASS_KEYS))
    got = _watch_contract_violations(_a7_eval_row())
    assert {"eval_map_hires_iou_lane_0_20", "eval_map_hires_iou_lane_80_100"} <= set(got)


def _lc_sum_breaks(row, prefix="", band_keys=H.BAND_KEYS) -> bool:
    """LOGGING_SPEC_MAP10 §6 check 2: |sum(map_hires_lc_*) - map_hires| <= 1e-5 |map_hires|."""
    tot = sum(row[H.per_class_key("lc", c, b, prefix)] for c in range(8)
              for b in band_keys)
    loss = row[prefix + "map_hires"]
    return abs(tot - loss) > 1e-5 * abs(loss)


def _small_loss_rows():
    """A small 10 cm loss (0.0123456789) split evenly over the 24 class x band cells:
    where a 5 dp rounded loss (0.01235) breaks the sum check by 3.5e-4 relative."""
    from types import SimpleNamespace as _NS
    lval = 0.0123456789
    lcs = {H.per_class_key("lc", c, b): lval / 24 for c in range(8) for b in range(3)}
    train = T._train_row_scalars(
        {"map_hires": torch.tensor(lval, dtype=torch.float64), **lcs},
        _NS(_map_hires=None))
    acc = {"map_hires": 2 * lval, **{k: 2 * v for k, v in lcs.items()}}
    ev = T._eval_row_from_acc(acc, 2, _NS(_map_hires=None))
    return train, ev


def test_GREEN_the_10cm_loss_is_logged_exact_and_the_lc_sum_check_holds():
    train, ev = _small_loss_rows()
    assert train["map_hires"] == 0.0123456789
    assert not _lc_sum_breaks(train) and not _lc_sum_breaks(ev, "eval_")
    # and on the real chain: the forward -> the trainer's block -> the train-row builder
    model, _ = _model()
    _attach(model)
    model.eval()
    row = T._train_row_scalars(_extras(model, 1), model)
    assert not _lc_sum_breaks(row)
    assert H.missing_per_class_keys(row) == []


def test_DELIBERATE_REGRESSION_the_loss_rounded_to_5dp_breaks_the_lc_sum_check(
        monkeypatch):
    """The pre-fix rule: only `map_hires_*` exact, the loss itself rounded."""
    monkeypatch.setattr(H, "is_exact_log_key", lambda k: k.startswith(H.LOG_PREFIX))
    train, ev = _small_loss_rows()
    assert train["map_hires"] == 0.01235
    assert _lc_sum_breaks(train) and _lc_sum_breaks(ev, "eval_")


def test_per_class_liveness_passes_healthy_and_catches_a_zero_weight_class():
    torch.manual_seed(0)
    codes = torch.randint(0, 8, (1, 600, 320), dtype=torch.uint8)
    lg = torch.randn(1, 8, 600, 320)
    ok = H.per_class_log_values(H.per_class_signal(lg, codes))
    r = H.per_class_liveness(ok, min_cells=100)
    assert r["checked"] == 24 and r["violations"] == []
    w = torch.ones(8)
    w[2] = 0.0                                          # lane weight 0
    bad = H.per_class_log_values(H.per_class_signal(lg, codes, class_weight=w))
    v = H.per_class_liveness(bad, min_cells=100)["violations"]
    assert {(x["class"], x["stat"]) for x in v} == {("lane", "lc"), ("lane", "gno")}
    # the lane CHANNEL still has a gradient (every other cell pushes it down): gn
    # alone could not have caught the zero weight -- gno does
    assert bad["map_hires_gn_lane_0_20"] > 0.0


# =========================================================================== #
# 7. G-DVB on the REAL model: the pass-through, the NEW-2 levers, the A6 seam  #
# =========================================================================== #
import argparse as _argparse  # noqa: E402


def _args(**over):
    a = dict(map_hires="on", w_map_hires=1.0, map_hires_class_weights=None,
             map_hires_decision_rule="prior_corrected", equalize_bottom_rows=0,
             map_hires_x_max_m=60.0, map_hires_y_half_m=16.0,
             map_hires_grad_ckpt="off", bev_source="map_hires_pool",
             bev_planner_crop_m=None, w_map=0.0, w_box3d=0.0)
    a.update(over)
    return _argparse.Namespace(**a)


def test_DELIBERATE_REGRESSION_fmap_s8_dropped_from_DECODER_PASSTHROUGH_goes_RED(
        monkeypatch):
    """The Master Mind's red arm: `fmap_s8` removed from the ONE pass-through
    declaration. The forward then does not emit it, the trainer's block REFUSES,
    and G-DVB names the tuple statically -- before any forward."""
    assert "fmap_s8" in refc.RefCModel.DECODER_PASSTHROUGH
    monkeypatch.setattr(refc.RefCModel, "DECODER_PASSTHROUGH", tuple(
        k for k in refc.RefCModel.DECODER_PASSTHROUGH if k != "fmap_s8"))
    with pytest.raises(SystemExit, match="fmap_s8"):
        _assert_hires_live()
    model, _ = _model()
    _attach(model)
    got = H.dvb_check_map_hires(model, _args())
    assert [x.read_from for x in got] == ["RefCModel.DECODER_PASSTHROUGH"]


def test_GDVB_the_NEW2_levers_are_registered_and_clean_on_a_built_A6_model(tmp_path):
    from tanitad.train import declared_vs_built as dvb
    model, _ = _model()
    _attach(model, pool=True)
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
                             "weights": [1.0] * 8, "dry_run": False}), encoding="utf-8")
    t, st = H.load_class_weights(p)
    model._map_hires_class_weight = t
    import dataclasses as _dc
    model._map_hires.cfg = _dc.replace(model._map_hires.cfg,
                                       class_weights_sha256=st["sha256"])
    args = _args(map_hires_class_weights=str(p))
    assert set(H.DVB_KINDS) == {"map_hires", "w_map_hires", "map_hires_class_weights",
                                "map_hires_decision_rule", "map_hires_x_max_m",
                                "map_hires_y_half_m", "map_hires_grad_ckpt",
                                "bev_source", "bev_planner_crop_m",
                                "map_hires_near_lift_m"}         # NEW-2 R2 (A12)
    for d in H.DVB_KINDS:
        assert dvb.REGISTRY[d].kind == H.DVB_KINDS[d]
        assert dvb.REGISTRY[d].check(model, args) == [], d
    # the refcv6 perception lever still reads clean on an A6 model (w_map 0, pool built)
    assert dvb.REGISTRY["w_map"].check(model, args) == []
    # one RED per lever family, on the SAME built model
    for over, dest in (({"w_map_hires": 0.5}, "w_map_hires"),
                       ({"map_hires_decision_rule": "raw"}, "map_hires_decision_rule"),
                       ({"map_hires_x_max_m": 100.0}, "map_hires_x_max_m"),
                       ({"map_hires_y_half_m": 30.0}, "map_hires_y_half_m"),
                       ({"map_hires_grad_ckpt": "on"}, "map_hires_grad_ckpt"),
                       ({"bev_source": "s16_lift"}, "bev_source"),
                       # NEW-2 R2: a near lift declared on a model built without one
                       ({"map_hires_near_lift_m": 20.0}, "map_hires_near_lift_m")):
        assert dvb.REGISTRY[dest].check(model, _argparse.Namespace(
            **{**vars(args), **over})), dest


def test_DELIBERATE_REGRESSION_a_consumer_still_wired_to_a_stride16_lift_goes_RED():
    """SPEC_REFCV7 §11.1's regression arm: argv declares the pooled BEV, the built
    perception branch still lifts the stride-16 map for its consumers."""
    model, _ = _model()
    _attach(model)
    model._perception = PB.build_perception_branch(
        model, PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0))   # the refcv6 lift
    got = [x.read_from for x in H.dvb_check_bev_source(model, _args())]
    assert "model._perception.lift is None" in got
    assert "model._perception.cfg.bev_source" in got
    good, _ = _model()
    _attach(good, pool=True)
    assert H.dvb_check_bev_source(good, _args()) == []


def test_DELIBERATE_REGRESSION_an_uncropped_200x120_planner_grid_goes_RED():
    """SPEC_REFCV7 §12 item 2's regression arm: the pool over the WHOLE A7 extent
    emits 200 x 120 -- a grid every consumer would still accept silently."""
    from types import SimpleNamespace as _NS
    hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
    cfg_ok = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=0.0, bev_source="map_hires_pool")
    ok = _NS(_perception=_NS(cfg=cfg_ok, bev_pool=PB.PlannerBEVPool(
        64, 96, hcfg.lift_grid, dst_grid=cfg_ok.planner_grid)))
    assert tuple(ok._perception.bev_pool.dst_hw) == (120, 64)
    assert H.dvb_check_planner_crop(ok, _args()) == []
    full = PB.BEVGrid(x_fwd_m=100.0, y_half_m=30.0, cell_m=0.5)
    bad = _NS(_perception=_NS(cfg=cfg_ok, bev_pool=PB.PlannerBEVPool(
        64, 96, hcfg.lift_grid, dst_grid=full)))
    assert tuple(bad._perception.bev_pool.dst_hw) == (200, 120)
    got = [x.read_from for x in H.dvb_check_planner_crop(bad, _args())]
    assert got == ["model._perception.bev_pool.dst_grid",
                   "model._perception.bev_pool.dst_hw"], got


def test_the_conflict_registry_under_A6():
    class _M:
        _w_map, _w_box3d, _w_bev_aux, _w_tac_v6 = 1.0, 0.0, 0.0, 0.0
    assert T._conflict_aux_weights(_M()) == {"map": 1.0}              # refcv6
    a6 = _M()
    a6._w_map, a6._w_box3d, a6._w_map_hires = 0.0, 1.0, 0.7          # --w-map 0 pinned
    assert T._conflict_aux_weights(a6) == {"box3d": 1.0, "map_hires": 0.7}


def test_the_trainers_compute_losses_runs_the_one_lift_end_to_end():
    """The trainer's OWN `compute_losses_v3` on a real (synthetic) batch: it picks the
    0.25 m bank as THE perception geometry, the forward's hook runs the one lift, the
    10 cm term enters the total, no 0.5 m map key exists, and one backward of the
    trainer's total reaches the lift, the shared encoder, the decoder and the trunk's
    stride-8 stage."""
    torch.manual_seed(0)
    model, cfg = _model()
    _attach(model)
    model.train()
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[CLIP, CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    # the smoke dataset carries no v7.2 join: the labels the loss needs, injected the
    # way test_built_heads_receive_gradient does
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID],
                                   dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    batch["tac_goal_y"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch.update(_batch())                    # map_fine / map_fine_label / map_ep
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    assert "map_hires" in losses and torch.isfinite(losses["loss"])
    assert losses["n_map_hires_cells"] > 0
    assert not [k for k in losses if k == "map" or k.startswith("aux05")]
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    rep = H.grad_reach_report_hires(model)
    for part in ("lift", "encoder", "refine", "trunk_s8_stage"):
        assert rep[part]["grad_abs_sum"] > 0.0, (part, rep[part])
    row = T._train_row_scalars(losses, model)
    assert row["map_hires"] == float(losses["map_hires"].detach())     # exact, not 5 dp
    assert H.missing_per_class_keys(row) == []


# =========================================================================== #
# D3 (fixes batch 2, b4a59b9): the 10 cm reach keys are DECLARED and HELD      #
# =========================================================================== #
#: what `grad_reach_report_hires` names on a built branch with the stride-8 tap, as the
#: loop writes them (`ga_mh_<part>`, `ga_mh_<part>_n`) -- a LITERAL
MH_GA_KEYS = ["ga_mh_encoder", "ga_mh_encoder_n", "ga_mh_lift", "ga_mh_lift_n",
              "ga_mh_refine", "ga_mh_refine_n", "ga_mh_trunk_s8_stage",
              "ga_mh_trunk_s8_stage_n"]
#: the tip's parts on the smoke rig (trunk + planner), unchanged by NEW-2 -- a LITERAL
TIP_GA_KEYS = ["ga_planner", "ga_planner_n", "ga_trunk", "ga_trunk_n"]


def _d3_config(model):
    return {"grad_reach_logging": T._grad_reach_declaration(
        model, _argparse.Namespace(log_every=50))}


def _d3_row(keys, drop=()):
    return {"loss": 1.0, "step": 50, **{k: 0.5 for k in keys if k not in drop}}


def test_D3_GREEN_a_map_hires_arm_declares_its_10cm_reach_keys():
    """A map-hires-only arm (no perception branch, no tac decoder) declares -- and so the
    loop writes, through the same `_grad_reach_declared` -- the tip's parts AND the 10 cm
    branch's; a row carrying them is clean."""
    from tanitad.train import declared_vs_built as dvb
    model, _ = _model()
    _attach(model)
    assert getattr(model, "_perception", None) is None
    assert getattr(model, "tac_decoder_v6", None) is None
    assert T._grad_reach_declared(model)
    cfg = _d3_config(model)
    dec = cfg["grad_reach_logging"]
    assert dec["declared"] is True
    assert dec["keys"] == sorted(TIP_GA_KEYS + MH_GA_KEYS)
    assert "grad_reach_report_hires" in dec["source"]
    assert dvb.check_logged_rows(cfg, [_d3_row(dec["keys"])]) == []


def test_D3_DELIBERATE_REGRESSION_a_dead_10cm_reach_row_is_caught_at_the_first_row():
    """RED: the loop stops writing one 10 cm part -> the first-row check names it. And the
    blind spot this closes: under a declaration of the perception parts only (the tip's),
    a row with EVERY 10 cm key missing passes."""
    from tanitad.train import declared_vs_built as dvb
    model, _ = _model()
    _attach(model, pool=True)
    cfg = _d3_config(model)
    keys = cfg["grad_reach_logging"]["keys"]
    assert set(MH_GA_KEYS) <= set(keys) and "ga_bev_pool" in keys
    bad = dvb.check_logged_rows(cfg, [_d3_row(keys, drop=("ga_mh_refine",))])
    assert len(bad) == 1 and "ga_mh_refine" in str(bad[0])
    tip_only = {"grad_reach_logging": {**cfg["grad_reach_logging"],
                                       "keys": [k for k in keys
                                                if not k.startswith("ga_mh_")]}}
    assert dvb.check_logged_rows(tip_only, [_d3_row(keys, drop=MH_GA_KEYS)]) == []


def test_D3_an_arm_without_the_10cm_branch_declares_exactly_what_the_tip_did():
    model, _ = _model()
    assert not T._grad_reach_declared(model)
    dec = _d3_config(model)["grad_reach_logging"]
    assert dec["declared"] is False and dec["keys"] == []
    assert dec["source"] == ("refcv6_perception_branch.grad_reach_report, read off the "
                             "BUILT model")



# =========================================================================== #
# NEW-2 R2 (SPEC_REFCV7 A12): the 0.1 m near lift through the trainer          #
# =========================================================================== #
def test_R2_the_near_lift_pin_GREEN_and_every_dead_or_illegal_value_refuses():
    _pin(ON + ["--map-hires-near-lift-m", "20"])
    _pin(ON + ["--map-hires-near-lift-m", "0"])                   # 0 = no near lift
    _pin(["--arm", "hier", "--out", "X", "--map-hires-near-lift-m", "0"])
    a = T.build_parser().parse_args(ON)
    assert H.declared_near_lift_m(a) == 0.0                       # unset = none
    assert H.declared_near_lift_m(T.build_parser().parse_args(
        ON + ["--map-hires-near-lift-m", "20"])) == 20.0
    for argv, needle in (
            (["--arm", "hier", "--out", "X", "--map-hires-near-lift-m", "20"],
             "READ BY NOTHING"),
            (ON + ["--map-hires-near-lift-m", "0.3"], "--map-hires-near-lift-m"),
            (ON + ["--map-hires-near-lift-m", "-0.5"], "--map-hires-near-lift-m"),
            (ON + ["--map-hires-near-lift-m", "100.5"], "--map-hires-near-lift-m"),
            (ON + ["--map-hires-x-max-m", "60", "--map-hires-y-half-m", "16",
                   "--map-hires-near-lift-m", "61"], "--map-hires-near-lift-m")):
        with pytest.raises(SystemExit) as e:
            _pin(argv)
        assert needle in str(e.value), (argv, str(e.value)[:300])


def test_R2_the_eval_loader_rebuilds_a_near_lift_model_strictly():
    """The stamp's ``near_lift_x_m`` rebuilds the near lift; strict 0/0. The RED arm: a
    loader that drops the field builds a branch short by exactly the near lift."""
    A = _arm_module()
    model, _ = _model()
    hcfg = _attach(model, pool=True, near=10.0)
    assert model._map_hires.near is not None
    cw = {"weights": [1.0] * 8, "sha256": "ab" * 32}
    tap = model.core.encoder.enable_s8_tap()
    stamp = {"map_hires": {**hcfg.as_dict(), "class_weights": cw,
                           "trunk_tap": {k: tap[k] for k in ("s8_module", "s8_dim",
                                                             "s8_hw")},
                           "branch_params": model._map_hires.param_breakdown()},
             "refcv6_perception": {**model._perception.cfg.as_dict(),
                                   "branch_params": model._perception.param_breakdown()}}
    assert stamp["map_hires"]["near_lift_x_m"] == 10.0
    fresh, _ = _model()
    A.rebuild_map_hires_branch(fresh, stamp, "cpu")
    A.rebuild_perception_branch(fresh, stamp, "cpu")
    res = fresh.load_state_dict(model.state_dict(), strict=False)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    assert fresh._map_hires.cfg.near_lift_x_m == 10.0 and fresh._map_hires.near is not None
    # RED: the field dropped from the stamp -> the rebuilt branch is refused (its parameter
    # count no longer matches the stamp's branch_params), never loaded half-empty
    dropped = {**stamp, "map_hires": {k: v for k, v in stamp["map_hires"].items()
                                      if k != "near_lift_x_m"}}
    other, _ = _model()
    with pytest.raises(SystemExit):
        A.rebuild_map_hires_branch(other, dropped, "cpu")


def test_R2_the_trainers_compute_losses_trains_the_near_lift_end_to_end():
    """The trainer's OWN `compute_losses_v3` on a near-lift model: the 10 cm term is live, one
    backward reaches the near lift (from its zero init), and D3 DECLARES its reach keys."""
    torch.manual_seed(0)
    model, cfg = _model()
    _attach(model, near=10.0)
    model.train()
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[CLIP, CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID],
                                   dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    batch["tac_goal_y"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch.update(_batch())
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    assert "map_hires" in losses and torch.isfinite(losses["loss"])
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    rep = H.grad_reach_report_hires(model)
    for part in ("lift", "encoder", "refine", "near", "trunk_s8_stage"):
        assert rep[part]["grad_abs_sum"] > 0.0, (part, rep[part])
    keys = _d3_config(model)["grad_reach_logging"]["keys"]
    assert "ga_mh_near" in keys and "ga_mh_near_n" in keys
