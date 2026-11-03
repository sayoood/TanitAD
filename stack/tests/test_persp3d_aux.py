"""refcv9 C4 (arms B2 and B2c): the training-only perspective 3D head and its parameter-matched BEV control -- analytic target controls (the
heat peak on the predicted column / row, the depth bin, 0 boxes -> all-zero), visibility, loss forms, REACHABILITY (argv -> model -> forward ->
loss -> backward with an exact-0.0 control), eval-forward bit-identity, the warm start's NAMED exemption, G-DVB, the stamp, the worker hook.
CPU, the tiny REAL RefCV3Model rig of test_map_only_ft.  Every expectation is a LITERAL or an analytic identity; each guard has a RED arm
(a mutant that must make the check go red)."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_box_depth_aux as C3T  # noqa: E402   (the C3 helpers: the agent block, the unzero-BN, the grad probe, the tree equality)
import test_map_only_ft as R  # noqa: E402
from tanitad.data import box_depth_targets as BD  # noqa: E402
from tanitad.data.bev_raster import GRID_DEFAULT  # noqa: E402
from tanitad.models import box_depth_aux as BDA  # noqa: E402
from tanitad.models import persp3d_aux as P  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import persp3d_aux_flag as F  # noqa: E402

H, W = 416, 1024


# ===================================================================================================================== #
# 1. analytic target controls (the cylindrical canonical frame the trainer uses)                                          #
# ===================================================================================================================== #
def _cam():
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    return BDA.camera_of(RigCamera.nominal(CanonicalFrame.from_hfov(120.0, H, W, "cylindrical"), height_m=1.5, x_m=0.0))


def _box_at(az_deg, rng, l=4.5, w=1.9, h=1.5, yaw=0.0):
    """A cuboid whose CENTRE is at azimuth ``az_deg`` (left positive) and horizontal ground distance ``rng`` from the camera, centre z = camera z."""
    a = math.radians(az_deg)
    return [rng * math.cos(a), rng * math.sin(a), 1.5, l, w, h, yaw]


def _pred_cell(cam, az_deg, row_pix, s):
    """The cell the cylinder puts the centre in: column = (W-1)/2 + f*phi (phi = -azimuth: +y is LEFT, columns grow to the RIGHT), cell = floor((px+0.5)/s)."""
    col = (W - 1) / 2.0 + cam.f_ref * (-math.radians(az_deg))
    return int(math.floor((row_pix + 0.5) / s)), int(math.floor((col + 0.5) / s)), col


@pytest.mark.parametrize("az,rng", [(20.0, 20.0), (-35.0, 12.0), (45.0, 30.0), (0.0, 25.0)])
def test_heat_peak_lands_on_the_predicted_column_row_and_the_depth_bin(az, rng):
    """TOLERANCE: 0 cells (the integer cell of the exact projected centre), both strides; the depth bin equals the LID bin of the centre RANGE."""
    cam = _cam()
    box = _box_at(az, rng)
    r = P.persp_targets(np.array([box]), np.array([5]), 8, cam)
    row_pix = (H - 1) / 2.0 + cam.f_ref * (1.5 - 1.5) / rng                       # centre z == camera z -> row (H-1)/2
    for s in (8, 16):
        ci, cj, _ = _pred_cell(cam, az, row_pix, s)
        hm = r[f"heat{s}"]
        assert hm[ci, cj] == 1.0 and int((hm >= 1.0).sum()) == 1, (s, ci, cj)
        assert tuple(np.unravel_index(hm.argmax(), hm.shape)) == (ci, cj)
        assert r[f"own{s}"][ci, cj] == 5 and set(np.unique(r[f"own{s}"])) <= {-1, 5}
        assert 0.0 < r[f"ctr{s}"][ci, cj] <= 1.0
    d = math.hypot(math.hypot(box[0], box[1]), 0.0)                              # centre z == camera z -> range == ground distance
    i0, _i1, _w0, w1 = BD.soft_two_hot(np.array([d]))
    assert r["i0"][5] == int(i0[0]) and r["w1"][5] == pytest.approx(float(w1[0]), abs=1e-6) and (r["i0"] == -1).sum() == 7


def test_zero_boxes_give_an_all_zero_target():
    r = P.persp_targets(np.zeros((0, 7)), np.zeros(0, np.int64), 4, _cam(), equalize_bottom_rows=43)
    for s in (8, 16):
        assert r[f"heat{s}"].max() == 0.0 and (r[f"own{s}"] == -1).all() and r[f"ctr{s}"].max() == 0.0
    assert (r["i0"] == -1).all() and r["w1"].max() == 0.0
    b = P.bev_targets(np.zeros((0, 7)), np.zeros(0, np.int64), GRID_DEFAULT)
    assert b["heat"].max() == 0.0 and (b["own"] == -1).all() and b["ctr"].max() == 0.0


def test_red_arm_a_pinhole_projection_moves_the_peak_off_the_predicted_column():
    """THE DELIBERATE REGRESSION: the pinhole formula in place of the cylindrical one (same grid, same f_ref) puts the peak on the WRONG column."""
    cam = _cam()
    pin = dataclasses.replace(cam, projection="pinhole")
    box = _box_at(45.0, 30.0)
    ok = P.persp_targets(np.array([box]), np.array([0]), 4, cam)
    bad = P.persp_targets(np.array([box]), np.array([0]), 4, pin)
    ci, cj, _ = _pred_cell(cam, 45.0, (H - 1) / 2.0, 8)
    assert ok["heat8"][ci, cj] == 1.0
    assert bad["heat8"][ci, cj] != 1.0                                          # the control goes RED under the wrong model


def test_an_occluded_object_is_not_a_target_and_the_visibility_rule_has_a_red_arm(monkeypatch):
    cam = _cam()
    near = _box_at(0.0, 8.0, l=4.0, w=12.0, h=4.0)                              # a wide near wall at azimuth 0
    far = _box_at(0.0, 30.0)                                                    # directly behind it
    boxes, slots = np.array([near, far]), np.array([0, 1])
    r = P.persp_targets(boxes, slots, 4, cam)
    assert r["census"]["n_visible"] == 1 and r["i0"][1] == -1 and 1 not in np.unique(r["own8"]) and 0 in np.unique(r["own8"])
    monkeypatch.setattr(P, "VIS_MIN_PIX", 0)                                    # MUTANT: no visibility rule -> the occluded object becomes a target
    r2 = P.persp_targets(boxes, slots, 4, cam)
    assert r2["census"]["n_visible"] == 2 and r2["i0"][1] != -1


def test_equalized_bottom_rows_are_ignored_not_targets():
    cam = _cam()
    box = _box_at(0.0, 3.0, h=1.0)
    box[2] = 0.3                                                                # centre low in the frame: row ~ (H-1)/2 + f*1.2/3 = 403 > 416-43
    r = P.persp_targets(np.array([box]), np.array([0]), 2, cam, equalize_bottom_rows=43)
    assert r["ign8"][-6:].all() and not r["ign8"][:40].any()
    assert r["census"]["n_heat_pos"] == 0 and r["census"]["n_centre_unobserved"] == 1 and r["heat8"].max() == 0.0


def test_bev_targets_known_cell_and_footprint():
    b = P.bev_targets(np.array([[10.2, -3.1, 0.8, 4.0, 2.0, 1.6, 0.0]]), np.array([2]), GRID_DEFAULT)
    i, j = int(10.2 / 0.5), int((-3.1 + 16.0) / 0.5)
    assert b["heat"][i, j] == 1.0 and int((b["heat"] >= 1.0).sum()) == 1 and b["own"][i, j] == 2
    assert int((b["own"] == 2).sum()) >= 4 * 8                                  # the 4 x 2 m footprint at 0.5 m is ~32 cells
    off = P.bev_targets(np.array([[70.0, 0.0, 0.8, 4.0, 2.0, 1.6, 0.0]]), np.array([0]), GRID_DEFAULT)
    assert off["census"]["n_off_grid"] == 1 and (off["own"] == -1).all()


# ===================================================================================================================== #
# 2. loss forms (hand-computed)                                                                                         #
# ===================================================================================================================== #
def _toy(B=1, Hs=4, Ws=6, A=3):
    own = torch.full((B, Hs, Ws), -1, dtype=torch.int16)
    own[0, 1, 2] = 1
    heat = torch.zeros(B, Hs, Ws)
    heat[0, 1, 2] = 1.0
    heat[0, 1, 3] = 0.5
    tg = {"heat": heat, "own": own, "ctr": torch.zeros(B, Hs, Ws), "ign": torch.zeros(B, Hs, Ws, dtype=torch.bool)}
    i0 = torch.full((B, A), -1, dtype=torch.long)
    i0[0, 1] = 10
    w1 = torch.zeros(B, A)
    w1[0, 1] = 0.25
    return tg, i0, w1


def test_focal_heat_equals_the_hand_computed_centernet_loss():
    tg, _, _ = _toy()
    x = torch.full((1, 1, 4, 6), -30.0)
    x[0, 0, 1, 2] = 0.7
    x[0, 0, 1, 3] = 0.2
    loss, n = P.focal_heat(x, tg["heat"])
    p = torch.sigmoid(torch.tensor(0.7)).item()
    q = torch.sigmoid(torch.tensor(0.2)).item()
    want = -((1 - p) ** 2 * math.log(p)) - ((1 - 0.5) ** 4 * q ** 2 * math.log(1 - q))
    assert int(n) == 1 and float(loss) == pytest.approx(want, rel=1e-5)
    ign = tg["ign"].clone()
    ign[0, 1, 3] = True                                                         # an IGNORE cell is out of the negatives
    l2, _ = P.focal_heat(x, tg["heat"], ign)
    assert float(l2) == pytest.approx(-((1 - p) ** 2 * math.log(p)), rel=1e-5)


def test_persp_parts_equal_hand_computed_values_and_vanish_when_perfect():
    tg, i0, w1 = _toy()
    size_t = torch.zeros(1, 3, 3)
    size_t[0, 1] = torch.tensor([1.5, 0.6, 0.4])
    yaw_t = torch.zeros(1, 3, 2)
    yaw_t[0, 1] = torch.tensor([0.0, 1.0])
    slots = {"size": size_t, "yaw": yaw_t, "xy": torch.zeros(1, 3, 2), "cz": torch.zeros(1, 3)}
    T_ = {8: tg, "i0": i0, "w1": w1}
    pred = {8: {"heat": torch.zeros(1, 1, 4, 6), "depth": torch.zeros(1, 80, 4, 6), "size": torch.zeros(1, 3, 4, 6),
                "yaw": torch.zeros(1, 2, 4, 6), "ctr": torch.zeros(1, 1, 4, 6)}}
    _, parts, _ = P.persp_loss(pred, T_, slots)
    assert float(parts["depth"]) == pytest.approx(math.log(80.0), rel=1e-6)           # a uniform depth distribution: CE = ln 80 against any two-hot
    sl1 = sum((a - 0.05 if a > 0.1 else 0.5 * a * a / 0.1) for a in (1.5, 0.6, 0.4))  # smooth-L1, beta 0.1
    assert float(parts["size"]) == pytest.approx(sl1, rel=1e-5)
    assert float(parts["yaw"]) == pytest.approx(1.0, rel=1e-6)                         # |0-0| + |0-1|
    assert float(parts["ctr"]) == pytest.approx(math.log(2.0), rel=1e-6)               # BCE(0 logit, target 0)
    pred2 = copy.deepcopy(pred)
    pred2[8]["size"][0, :, 1, 2] = torch.tensor([1.5, 0.6, 0.4])
    pred2[8]["yaw"][0, :, 1, 2] = torch.tensor([0.0, 1.0])
    d = torch.full((80,), -50.0)
    d[10] = math.log(0.75)
    d[11] = math.log(0.25)
    pred2[8]["depth"][0, :, 1, 2] = d
    pred2[8]["ctr"][0, 0, 1, 2] = -50.0
    _, perfect, _ = P.persp_loss(pred2, T_, slots)
    assert float(perfect["size"]) < 1e-6 and float(perfect["yaw"]) < 1e-6 and float(perfect["ctr"]) < 1e-6
    assert float(perfect["depth"]) == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)), rel=1e-5)   # the entropy of the two-hot


def test_a_frame_with_no_positive_gives_exact_zero_regression_parts_with_the_graph_kept():
    tg, i0, w1 = _toy()
    tg["own"][:] = -1
    tg["heat"][:] = 0.0
    slots = {"size": torch.zeros(1, 3, 3), "yaw": torch.zeros(1, 3, 2), "xy": torch.zeros(1, 3, 2), "cz": torch.zeros(1, 3)}
    pred = {8: {k: torch.randn(1, c, 4, 6, requires_grad=True) for k, c in P.PERSP_CH.items()}}
    loss, parts, _ = P.persp_loss(pred, {8: tg, "i0": i0, "w1": w1}, slots)
    assert all(float(parts[k]) == 0.0 for k in ("depth", "size", "yaw", "ctr")) and float(parts["heat"]) > 0.0
    loss.backward()
    assert all(float(pred[8][k].grad.abs().max()) == 0.0 for k in ("depth", "size", "yaw", "ctr")) and float(pred[8]["heat"].grad.abs().max()) > 0.0


# ===================================================================================================================== #
# 3. the real tiny rig: reachability, eval identity, OFF == tree                                                        #
# ===================================================================================================================== #
def _build(monkeypatch, persp=False, bev2=False, w=0.1):
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: C3T._ORIG_MODEL(cfg, perc_rig=True))
    model, cfg = R._build()
    model._rig_camera = R.RigCamera.nominal(model._lift_bank_hires.frame, height_m=1.5, x_m=1.5)
    args = argparse.Namespace(r9_box_persp_aux=persp, w_box_persp=(w if persp else 0.0), r9_box_bev2_aux=bev2, w_box_bev2=(w if bev2 else 0.0))
    F.attach(model, args)
    return model, cfg, args


def _src(model, kind):
    eq = int(getattr(getattr(model.core.encoder, "cfg", None), "equalize_bottom_rows", 0) or 0)
    return F.TargetSource("persp", cam=model._rig_camera, eq=eq) if kind == "persp" else F.TargetSource("bev2", grid=GRID_DEFAULT)


def _with_targets(batch, model, kind):
    src = _src(model, kind)
    keys = P.SLOT_KEYS + ("agent_label", "agent_ep")
    items = [F.item_targets({k: batch[k][i] for k in keys if k in batch}, src) for i in range(int(batch["agent_box"].shape[0]))]
    batch.update(torch.utils.data.default_collate(items))
    return batch


def _batch(cfg, model, kind, rows=None):
    batch = R._batch(cfg)
    rows = rows if rows is not None else [[(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)], [(6.0, 0.5, 1.2, 4.0, 1.8, 1.6, -0.4), (12.0, -3.0, 1.0, 4.5, 2.0, 1.5, 1.2)]]
    batch.update(C3T._agent_block(rows))
    batch["agent_ep"] = batch["map_ep"].clone()
    return _with_targets(batch, model, kind)


def _reach(monkeypatch, kind, term_fn=None):
    """Run the loss on the tiny rig and report (new params with non-zero grad, new params, controls exactly zero, term finite & > 0)."""
    model, cfg, args = _build(monkeypatch, persp=(kind == "persp"), bev2=(kind == "bev2"))
    C3T._unzero_bn(model)
    key, pfx = ("box_persp", F.KEY_PREFIX_P) if kind == "persp" else ("box_bev2", F.KEY_PREFIX_B)
    batch = _batch(cfg, model, kind)
    model.train()
    torch.manual_seed(5)
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    if key not in out:
        return {"present": False}
    w = float(getattr(model, F.WEIGHT_ATTR_P if kind == "persp" else F.WEIGHT_ATTR_B))
    gd = C3T._named_grads(w * out[key], model)
    new = {n: g for n, g in gd.items() if n.startswith(pfx)}
    bev_path = ("_perception.bev_pool.", "_map_hires.")                      # the shared BEV's own producers: B2c's term LEGITIMATELY reaches them
    if kind == "persp":      # the term never touches the 10 cm branch, the BEV pool or the agent head
        ctrl = [n for n in gd if n.startswith(("core.agent_head.", "_map_hires.")) or (n.startswith("_perception.") and not n.startswith(pfx))]
    else:                    # B2c reads the pooled BEV <- the 10 cm ENCODER <- the lift <- the trunk's s8 stage; it must NOT reach s16 / s32, the refine / near heads
        ctrl = [n for n in gd if n.startswith(("core.encoder.net.layer3.", "core.encoder.net.layer4.", "core.encoder.fuse", "_map_hires.refine.",
                                                "_map_hires.near", "core.agent_head."))]
    gm = C3T._named_grads(out["map_hires"], model)
    return {"present": True, "term": float(out[key]), "n_new": len(new), "n_nonzero": sum(not C3T._is_zero(g) for g in new.values()),
            "zero_new": [n for n, g in new.items() if C3T._is_zero(g)], "n_ctrl": len(ctrl), "ctrl_zero": all(C3T._is_zero(gd[n]) for n in ctrl),
            "bev_reached": any(not C3T._is_zero(g) for n, g in gd.items() if n.startswith(bev_path) and g is not None),
            "poscontrol": any(not C3T._is_zero(g) for n, g in gm.items() if n.startswith("_map_hires.")),
            "map_leaves_head_zero": all(C3T._is_zero(g) for n, g in gm.items() if n.startswith(pfx)),
            "trunk_s8": any(not C3T._is_zero(g) for n, g in gd.items() if n.startswith("core.encoder.net.layer2.") and g is not None and g.ndim == 4),
            "trunk_s16": any(not C3T._is_zero(g) for n, g in gd.items() if n.startswith("core.encoder.net.layer3.") and g is not None and g.ndim == 4),
            "out": out}


@pytest.mark.parametrize("kind,n_tensors", [("persp", 2 * (6 + 10)), ("bev2", 6 + 12)])
def test_reachability_every_hop_gradient_nonzero_and_the_zero_control(monkeypatch, kind, n_tensors):
    r = _reach(monkeypatch, kind)
    assert r["present"] and math.isfinite(r["term"]) and r["term"] > 0.0
    assert r["n_new"] == n_tensors and r["n_nonzero"] == r["n_new"], r["zero_new"]          # EVERY new parameter tensor gets a gradient
    assert r["n_ctrl"] > 5 and r["ctrl_zero"]                                              # EXACTLY 0.0 on the box decoder / memory / agent head
    assert r["poscontrol"] and r["map_leaves_head_zero"]                                  # the probe can see non-zero (map term) and the map term misses the new head
    if kind == "persp":
        assert r["trunk_s8"] and r["trunk_s16"]                                           # the s8 and s16 stages are trained BY the term
    else:
        assert r["bev_reached"] and r["trunk_s8"]                                         # B2c's gradient flows through the shared BEV into the 10 cm branch / trunk
    assert r["out"][f"box_{'persp' if kind == 'persp' else 'bev2'}_n_heat_pos"] >= 1.0


@pytest.mark.parametrize("kind", ["persp", "bev2"])
def test_red_arm_a_deleted_loss_line_makes_the_reachability_check_red(monkeypatch, kind):
    assert _reach(monkeypatch, kind)["present"]
    monkeypatch.setattr(F, "loss_term", lambda *a, **k: None)                            # MUTANT: the trainer's loss line is gone
    assert not _reach(monkeypatch, kind)["present"]


@pytest.mark.parametrize("kind", ["persp", "bev2"])
def test_frame_without_boxes_trains_only_the_heat_branch(monkeypatch, kind):
    """No cuboid in the batch: the heat branch gets a (negatives-only) gradient and every regression branch EXACTLY 0.0 -- the probe discriminates."""
    model, cfg, _ = _build(monkeypatch, persp=(kind == "persp"), bev2=(kind == "bev2"))
    batch = _batch(cfg, model, kind, rows=[[], []])
    model.train()
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    key = "box_persp" if kind == "persp" else "box_bev2"
    head = F._head_p(model) if kind == "persp" else F._head_b(model)
    pfx = F.KEY_PREFIX_P if kind == "persp" else F.KEY_PREFIX_B
    g = C3T._named_grads(out[key], model)
    heat = [n for n in g if n.startswith(pfx) and ".heat." in n or (n.startswith(pfx) and ".tower." in n)]
    reg = [n for n in g if n.startswith(pfx) and any(f".{k}." in n for k in ("depth", "size", "yaw", "ctr", "off", "cz"))]
    assert head is not None and reg and float(out[key]) > 0.0
    assert all(C3T._is_zero(g[n]) for n in reg)
    assert any(not C3T._is_zero(g[n]) for n in heat)


def _fwd(m, batch):
    fr = T.frames_to_device(batch["frames"][:, -1:], "cpu")
    gg, vv = m._lift_bank_hires.for_episodes(batch["map_ep"], device="cpu")
    with torch.no_grad():
        return m(fr, perception_grid=gg, perception_valid=vv)


@pytest.mark.parametrize("kind", ["persp", "bev2"])
def test_eval_forward_is_bit_identical_with_the_flag_on_and_off(monkeypatch, kind):
    on, cfg, _ = _build(monkeypatch, persp=(kind == "persp"), bev2=(kind == "bev2"))
    off, _, _ = _build(monkeypatch)
    sd_on, sd_off = on.state_dict(), off.state_dict()
    pfx = F.KEY_PREFIX_P if kind == "persp" else F.KEY_PREFIX_B
    extra = sorted(set(sd_on) - set(sd_off))
    assert extra and all(k.startswith(pfx) for k in extra) and set(sd_off) <= set(sd_on)
    assert all(torch.equal(sd_on[k], sd_off[k]) for k in sd_off)                         # OFF/ON share EVERY init draw (fork_rng)
    batch = _batch(cfg, on, kind)
    on.eval(), off.eval()
    assert C3T._tree_equal(_fwd(on, batch), _fwd(off, batch))
    lo = T.compute_losses_v3(on, batch, "cpu", log_metrics=False)
    assert not any(k.startswith(("box_persp", "box_bev2")) for k in lo)                  # eval mode: the head is not even called
    # RED: a head left in the eval forward
    real = on.forward

    def leaky(*a, **k):
        out = real(*a, **k)
        out["leak"] = F._head_p(on)(out["fmap_s8"], out["fmap_s16"]) if kind == "persp" else F._head_b(on)(out["perception"]["bev_feats"])
        return out
    on.forward = leaky
    assert not C3T._tree_equal(_fwd(on, batch), _fwd(off, batch))


@pytest.mark.parametrize("kind", ["persp", "bev2"])
def test_train_loss_with_the_flag_on_is_the_off_loss_plus_the_weighted_term(monkeypatch, kind):
    on, cfg, _ = _build(monkeypatch, persp=(kind == "persp"), bev2=(kind == "bev2"))
    off, _, _ = _build(monkeypatch)
    batch = _batch(cfg, on, kind)
    on.train(), off.train()
    torch.manual_seed(5)
    lo = T.compute_losses_v3(on, batch, "cpu", log_metrics=False)
    torch.manual_seed(5)
    lf = T.compute_losses_v3(off, batch, "cpu", log_metrics=False)
    key = "box_persp" if kind == "persp" else "box_bev2"
    assert float(lo["loss"]) == pytest.approx(float(lf["loss"]) + 0.1 * float(lo[key]), rel=1e-6, abs=1e-7)
    for k, v in lf.items():
        if k != "loss" and torch.is_tensor(v) and v.numel() == 1:
            assert float(lo[k]) == float(v), k                                          # every other loss scalar is bit-identical


def test_flag_off_builds_nothing_and_adds_no_key(monkeypatch):
    model, cfg, args = _build(monkeypatch)
    assert F._head_p(model) is None and F._head_b(model) is None and not any(".box_persp_aux." in k or ".box_bev2_aux." in k for k in model.state_dict())
    assert F.stamp(model, args) is None and F.dvb_check_persp(model, args) == [] and F.dvb_check_bev2(model, args) == []
    assert F.loss_term(model, {}, {}, "cpu", {}) is None and rig.persp_exemptions(model) == ()


# ===================================================================================================================== #
# 4. parameter counts, the B2c match and its RED arm; refusals                                                          #
# ===================================================================================================================== #
def test_head_sizes_are_in_the_design_band_and_b2c_is_matched():
    b2 = P.n_params(P.Persp3DAuxHead(512, 1024))                                         # the resnet101 widths of the deployed trunk (s8 512, s16 1024)
    assert 1_400_000 <= b2 <= 1_650_000, b2                                              # design "~ 1.5 M"; box3d cap <= 12 M (PI D2) has >10 M of headroom
    mid = P.matched_mid(96, b2)
    b2c = P.n_params(P.Bev2AuxHead(96, mid=mid))
    assert P.count_matched(b2c, b2) and abs(b2c - b2) / b2 < 0.02, (mid, b2c, b2)
    assert not P.count_matched(P.n_params(P.Bev2AuxHead(96, mid=64)), b2)                # RED: an unmatched control fails the band
    assert not P.count_matched(int(b2 * 1.11), b2) and P.count_matched(int(b2 * 1.09), b2)


def test_b2c_unmatched_head_is_caught_by_the_dvb_check_and_stamp_reports_both_counts(monkeypatch):
    model, _, args = _build(monkeypatch, bev2=True)
    assert F.dvb_check_bev2(model, args) == [] and F.dvb_check_weight_bev2(model, args) == []
    st = F.stamp(model, args)
    assert st["n_params"] == P.n_params(F._head_b(model)) and P.count_matched(st["n_params"], st["ref_params_b2"])
    bad = copy.deepcopy(model)
    br = bad._perception
    br.box_bev2_aux = P.Bev2AuxHead(96, mid=48)                                          # MUTANT: a control that is not capacity-matched
    br.box_bev2_aux.ref_params = st["ref_params_b2"]
    assert any("capacity-matched" in str(m) for m in F.dvb_check_bev2(bad, args))


def test_defaults_are_off_and_named_flags_are_required():
    a = T.build_parser().parse_args(["--arm", "flat", "--out", "x"])
    assert a.r9_box_persp_aux is False and a.w_box_persp == 0.0 and a.r9_box_bev2_aux is False and a.w_box_bev2 == 0.0
    F.pin_args(a)                                                                        # the default argv passes untouched


def test_pin_args_refusals():
    base = dict(r9_box_persp_aux=True, w_box_persp=0.1, r9_box_bev2_aux=False, w_box_bev2=0.0, r9_perc_rig=True, w_box3d=1.0, map_hires="on",
                join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    F.pin_args(argparse.Namespace(**base))                                               # the complete B2 argv is admitted
    b2c = {**base, "r9_box_persp_aux": False, "w_box_persp": 0.0, "r9_box_bev2_aux": True, "w_box_bev2": 0.1, "agent_rig_camera": "off"}
    F.pin_args(argparse.Namespace(**b2c))                                                # B2c needs no camera
    for k, v, frag in (("w_box_persp", 0.0, "positive weight"), ("r9_perc_rig", False, "r9-perc-rig"), ("w_box3d", 0.0, "w-box3d"),
                       ("map_hires", "off", "map-hires"), ("join3d", "", "join3d"), ("agent_join", "", "agent-join"),
                       ("agent_rig_camera", "off", "agent-rig-camera")):
        with pytest.raises(SystemExit, match=frag):
            F.pin_args(argparse.Namespace(**{**base, k: v}))
    for k, v, frag in (("w_box_bev2", 0.0, "positive weight"), ("w_box3d", 0.0, "w-box3d"), ("r9_perc_rig", False, "r9-perc-rig")):
        with pytest.raises(SystemExit, match=frag):
            F.pin_args(argparse.Namespace(**{**b2c, k: v}))
    with pytest.raises(SystemExit, match="MUTUALLY EXCLUSIVE"):                          # both flags: refused
        F.pin_args(argparse.Namespace(**{**base, "r9_box_bev2_aux": True, "w_box_bev2": 0.1}))
    for kw in ({"w_box_persp": 0.2, "r9_box_persp_aux": False}, {"w_box_bev2": 0.2}):    # a weight with no flag: a dead flag
        with pytest.raises(SystemExit, match="dead flag"):
            F.pin_args(argparse.Namespace(**{**dict(r9_box_persp_aux=False, w_box_persp=0.0, r9_box_bev2_aux=False, w_box_bev2=0.0), **kw}))


# ===================================================================================================================== #
# 5. warm-start exemption, G-DVB, stamp, hops                                                                           #
# ===================================================================================================================== #
class _Per(torch.nn.Module):
    def __init__(self, attr=None, extra=False):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        if attr:
            setattr(self, attr, torch.nn.Linear(4, 2))
        if extra:
            self.other = torch.nn.Linear(2, 2)


class _Model(torch.nn.Module):
    def __init__(self, attr=None, extra=False):
        super().__init__()
        self._perception = _Per(attr, extra)


def _srcfile(tmp_path):
    src = _Model()
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "step": 7}, str(p))
    return src, str(p)


@pytest.mark.parametrize("attr,pfx", [("box_persp_aux", "_perception.box_persp_aux."), ("box_bev2_aux", "_perception.box_bev2_aux.")])
def test_head_keys_are_named_and_everything_else_still_loads(tmp_path, attr, pfx):
    src, p = _srcfile(tmp_path)
    m = _Model(attr)
    assert rig.persp_exemptions(_Model()) == () and rig.persp_exemptions(m) == (pfx,)
    rec = rig.warm_start_perc_rig(m, p)
    assert rec["n_loaded"] == 2 and torch.equal(m._perception.proj.weight, src._perception.proj.weight)
    with pytest.raises(SystemExit):                                                      # an unnamed missing key beyond the head still refuses
        rig.warm_start_perc_rig(_Model(attr, extra=True), p)


def test_red_head_without_the_exemption_refuses(tmp_path, monkeypatch):
    _, p = _srcfile(tmp_path)
    monkeypatch.setattr(rig, "persp_exemptions", lambda model: ())
    with pytest.raises(SystemExit, match=r"rig key\(s\) the source lacks"):
        rig.warm_start_perc_rig(_Model("box_persp_aux"), p)


def test_dvb_checks_agree_and_disagree(monkeypatch):
    model, _, args = _build(monkeypatch, persp=True)
    assert F.dvb_check_persp(model, args) == [] and F.dvb_check_weight_persp(model, args) == []
    assert F.dvb_check_persp(model, argparse.Namespace(r9_box_persp_aux=False, w_box_persp=0.1))               # declared OFF, head built
    assert F.dvb_check_weight_persp(model, argparse.Namespace(r9_box_persp_aux=True, w_box_persp=0.2))         # weight disagrees
    off, _, oargs = _build(monkeypatch)
    assert F.dvb_check_persp(off, oargs) == [] and F.dvb_check_weight_persp(off, oargs) == []
    assert F.dvb_check_persp(off, argparse.Namespace(r9_box_persp_aux=True, w_box_persp=0.1))                  # declared ON, nothing built
    assert F.dvb_check_weight_bev2(off, argparse.Namespace(r9_box_bev2_aux=True, w_box_bev2=0.1))
    bad = copy.deepcopy(model)
    bad._perception.box_persp_aux.s8.depth = torch.nn.Conv2d(bad._perception.box_persp_aux.mid, 64, 1)
    assert any("depth bins" in str(m) for m in F.dvb_check_persp(bad, args))
    assert {"r9_box_persp_aux", "w_box_persp", "r9_box_bev2_aux", "w_box_bev2"} <= set(T._dvb.REGISTRY)


def test_stamp_states_the_unpinned_definitions(monkeypatch):
    model, _, args = _build(monkeypatch, persp=True)
    st = F.stamp(model, args)
    assert st["n_bins"] == 80 and st["strides"] == [8, 16] and st["n_params"] == P.n_params(F._head_p(model)) and "HYPOTHESIS" in st["sigma"]
    assert st["outputs"] == {"heat": 1, "depth": 80, "size": 3, "yaw": 2, "ctr": 1} and "never by RefCV3Model.forward" in st["training_only"]
    off, _, oargs = _build(monkeypatch)
    assert F.stamp(off, oargs) is None


def test_trainer_hops_are_present_in_the_source():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    for needle in ("from tanitad.train import persp3d_aux_flag as _p3f", "_p3f.register_dvb_levers(_dvb.register)", "_p3f.pin_args(args)",
                   "_p3f.attach(model, args)", "_p3_term = _p3f.loss_term(model, out, batch, device, extra)", "loss = loss + _p3_term",
                   '"r9_box_persp_aux": _p3f.stamp(model, args)', '"r9_box_bev2_aux": _p3f.stamp(model, args)', '"--r9-box-persp-aux"',
                   'g8.add_argument("--w-box-persp"', '"--r9-box-bev2-aux"', 'g8.add_argument("--w-box-bev2"', '"w_box_persp": {', '"w_box_bev2": {', "_p3f.attach_dataset(ds, model, args)",
                   "item.update(_p3f.item_targets(item, self._r9_p3))"):
        assert src.count(needle) == 1, needle
    h = src.index("item.update(_p3f.item_targets(item, self._r9_p3))")
    assert src.index("item.update(_bda.item_targets(item, self._r9_bd))") < h < src.index("def _nullctx(")
    a = src.index("_p3f.attach_dataset(ds, model, args)")
    assert a < src.index("ds, batch_size=args.batch, sampler=_train_sampler, num_workers=args.workers,")
    assert src.index("_p3f.attach(model, args)") < a
    i, j = src.index("def compute_losses_perc_rig("), src.index("def _lan_arm_preflight(")
    assert i < src.index("_p3f.loss_term(model, out, batch, device, extra)") < j
    assert (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_text(encoding="utf-8").count("tr._p3f.attach(model, args)") == 1


# ===================================================================================================================== #
# 6. the worker hook                                                                                                    #
# ===================================================================================================================== #
def test_target_source_survives_a_pickle_round_trip_as_a_spawned_worker_needs(monkeypatch):
    model, _, _ = _build(monkeypatch)
    for kind in ("persp", "bev2"):
        s2 = pickle.loads(pickle.dumps(_src(model, kind)))
        assert s2.kind == kind and s2.eq == _src(model, kind).eq


def test_a_batch_without_the_precomputed_targets_refuses_and_the_step_builds_nothing(monkeypatch):
    model, cfg, _ = _build(monkeypatch, persp=True)
    batch = _batch(cfg, model, "persp")
    model.train()
    monkeypatch.setattr(P, "persp_targets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the step must not build targets")))
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)                  # the step moved the worker's tensors, built nothing
    assert math.isfinite(float(out["box_persp"]))
    for k in F.BATCH_KEYS_P:
        stripped = {a: b for a, b in batch.items() if a != k}
        with pytest.raises(SystemExit, match="precomputed"):
            T.compute_losses_v3(model, stripped, "cpu", log_metrics=False)


def test_a_no_label_row_teaches_nothing_empty_and_an_unlabelled_item_ignores_every_heat_cell(monkeypatch):
    model, cfg, _ = _build(monkeypatch, persp=True)
    batch = R._batch(cfg)
    batch.update(C3T._agent_block([[(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)], [(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)]]))
    batch["agent_label"] = torch.tensor([True, False])                                  # row 1 is NO_LABEL (the join says nothing about it)
    batch["agent_ep"] = batch["map_ep"].clone()
    it0 = F.item_targets({k: batch[k][0] for k in P.SLOT_KEYS + ("agent_label", "agent_ep")}, _src(model, "persp"))
    it1 = F.item_targets({k: batch[k][1] for k in P.SLOT_KEYS + ("agent_label", "agent_ep")}, _src(model, "persp"))
    assert not it0["p3_ign8"].all() and it1["p3_ign8"].all() and it1["p3_ign16"].all() and it1["p3_heat8"].max() == 0.0
    b2 = F.item_targets({k: batch[k][1] for k in P.SLOT_KEYS + ("agent_label",)}, _src(model, "bev2"))
    assert b2["b2_ign"].all() and b2["b2_heat"].max() == 0.0
    x = torch.zeros(1, 1, *it1["p3_heat8"].shape)
    l_ign, _ = P.focal_heat(x, it1["p3_heat8"][None], it1["p3_ign8"][None])
    assert float(l_ign) == 0.0                                                          # RED if the ignore map were dropped: the focal would be > 0
    assert float(P.focal_heat(x, it1["p3_heat8"][None])[0]) > 0.0


def _fake_agent_item(ds, ep, f):
    blk = C3T._agent_block([[(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)]])
    return {k: blk[k][0] for k in P.SLOT_KEYS + ("agent_label",)} | {"agent_ep": torch.tensor(int(ep.episode_id), dtype=torch.long)}


def test_the_trainers_own_dataset_hook_adds_the_targets_only_when_attached(monkeypatch):
    model, cfg, _ = _build(monkeypatch, persp=True)
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    monkeypatch.setattr(T.V3Dataset, "_agent_item", _fake_agent_item)
    ds.agent_join = object()
    assert not any(k.startswith(("p3_", "b2_")) for k in ds[0])                          # OFF: the item key set is unchanged
    ds._r9_p3 = _src(model, "persp")
    it = ds[0]
    assert all(k in it for k in F.BATCH_KEYS_P + F.CENSUS_KEYS_P) and float(it["p3_heat8"].max()) == 1.0
    ds._r9_p3 = _src(model, "bev2")
    assert all(k in ds[0] for k in F.BATCH_KEYS_B + F.CENSUS_KEYS_B)


def test_attach_dataset_off_on_and_its_startup_refusals(monkeypatch, capsys):
    model, _, _ = _build(monkeypatch)
    on_p = argparse.Namespace(r9_box_persp_aux=True, w_box_persp=0.1, r9_box_bev2_aux=False, w_box_bev2=0.0)
    on_b = argparse.Namespace(r9_box_persp_aux=False, w_box_persp=0.0, r9_box_bev2_aux=True, w_box_bev2=0.1, workers=3)
    off = argparse.Namespace(r9_box_persp_aux=False, w_box_persp=0.0, r9_box_bev2_aux=False, w_box_bev2=0.0)
    mk = lambda **k: types.SimpleNamespace(**{"agent_join": object(), "join3d": object(), "episodes": [types.SimpleNamespace(episode_id=7)], **k})  # noqa: E731
    ds = mk()
    assert F.attach_dataset(ds, model, off) == {} and not hasattr(ds, "_r9_p3")                  # OFF: touches nothing
    assert F.attach_dataset(ds, model, on_p)["kind"] == "persp" and ds._r9_p3.kind == "persp"
    assert "WARNING --workers 0" in capsys.readouterr().out
    assert F.attach_dataset(mk(), model, on_b)["kind"] == "bev2" and "WARNING" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="agent_join / join3d"):
        F.attach_dataset(mk(join3d=None), model, on_p)
    model._rig_camera = None
    with pytest.raises(SystemExit, match="_rig_camera is None"):
        F.attach_dataset(mk(), model, on_p)


def test_slot_tables_are_finite_even_if_an_invalid_slot_holds_nan():
    """A NaN in a padded / invalid slot must not poison the loss through `NaN * 0` at a masked cell (the owner maps never gather it into a positive)."""
    batch = C3T._agent_block([[(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)]])
    for k in ("agent_box", "agent_yaw", "agent_cz", "agent_h", "agent_valid", "agent_zh_mask"):
        batch[k][0, 1] = batch[k][0, 0]                                                   # the real cuboid moves to slot 1
    batch["agent_valid"][0, 0] = False
    batch["agent_h"][0, 0] = float("nan")
    batch["agent_box"][0, 0, 2] = float("nan")
    slots = P.slot_tables(batch, "cpu")
    assert all(bool(torch.isfinite(v).all()) for v in slots.values())
    tg, i0, w1 = _toy(A=4)                                                                # the positive cell is owned by slot 1
    pred = {8: {k: torch.randn(1, c, 4, 6, requires_grad=True) for k, c in P.PERSP_CH.items()}}
    loss, parts, _ = P.persp_loss(pred, {8: tg, "i0": i0, "w1": w1}, slots)
    loss.backward()
    assert math.isfinite(float(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in pred[8].values())
