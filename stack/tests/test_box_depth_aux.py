"""refcv9 C3 (arm B1): the training-only box-derived depth head -- builder pin, analytic controls + RED arms, loss form, reachability
(argv -> model -> forward -> loss -> backward, with a ZERO control), eval-forward bit-identity, the rig warm start's named exemption,
G-DVB. CPU, the tiny REAL RefCV3Model rig of test_map_only_ft (resnet18 @ 64 x 64, the 10 cm branch). Every expectation is a LITERAL
or an analytic identity; each guard has a RED arm."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import math
import sys
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
import test_map_only_ft as R  # noqa: E402
from tanitad.data import box_depth_targets as BD  # noqa: E402
from tanitad.models import box_depth_aux as BDA  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import box_depth_aux_flag as F  # noqa: E402

BD_PATH = Path(BD.__file__)
E0_BLOB_SHA1 = "6fff1d3a15d9e66fa8ef82256654a30ca26f21d6"                       # git blob of the e0 copy (landed 96d8beb)
E0_BODY_SHA256 = "eccc37a78f667df4ac78ea422ed99eeb17d74b57e45a4f7f35be3d1eb2b0f553"
E0_REL = ("TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-27-perception-redesign", "e0", "e0g_depth_targets.py")


def _body(raw: bytes) -> bytes:
    first, rest = raw.split(b"\n", 1)
    assert first.startswith(b"# PROVENANCE (refcv9 C3)"), first[:60]
    return rest


def _blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# ===================================================================================================================== #
# 0. the tree under test + the builder pin                                                                              #
# ===================================================================================================================== #
def test_tanitad_is_from_the_tree_under_test():
    import tanitad
    p = Path(tanitad.__file__).resolve()
    assert ROOT.resolve() in p.parents, (p, ROOT)                      # never the G: / E: editable install


def test_builder_copy_is_byte_equal_to_the_e0_blob():
    body = _body(BD_PATH.read_bytes())
    assert hashlib.sha256(body).hexdigest() == E0_BODY_SHA256
    assert _blob_sha1(body) == E0_BLOB_SHA1
    e0 = ROOT.parent.joinpath(*E0_REL)                                  # present in a full checkout (not in the stack-only export)
    if e0.is_file():
        assert body == e0.read_bytes()


def test_red_a_one_byte_edit_of_the_copy_turns_the_pin_red():
    body = bytearray(_body(BD_PATH.read_bytes()))
    i = body.index(b"ALPHA_FG, ALPHA_BG = 3.25, 0.25")
    body[i + len(b"ALPHA_FG, ALPHA_BG = 3.") ] ^= 0x01                  # 3.25 -> 3.34
    assert hashlib.sha256(bytes(body)).hexdigest() != E0_BODY_SHA256
    assert _blob_sha1(bytes(body)) != E0_BLOB_SHA1


def test_builder_is_torch_free_at_import():
    import subprocess
    # the module by FILE (its package __init__ may import torch; the property is the builder's own, as in the e0 test)
    code = ("import sys, importlib.util as u; s = u.spec_from_file_location('bdt', %r); m = u.module_from_spec(s); "
            "sys.modules['bdt'] = m; s.loader.exec_module(m); assert 'torch' not in sys.modules; print('ok')" % str(BD_PATH))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ===================================================================================================================== #
# 1. the analytic control through the PIPELINE'S camera + the pinhole RED arm                                           #
# ===================================================================================================================== #
H, W = BD.FRAME_H, BD.FRAME_W


def _box(cx, cy, cz, l, w, h, yaw=0.0):
    return [cx, cy, cz, l, w, h, yaw]


def _known_range_ok(cam, boxes, row, col, expected, tol, kind="range"):
    got = float(BD.raycast(boxes, cam, depth_kind=kind).depth[row, col])
    return (np.isfinite(got) and abs(got - expected) <= tol), got


def _face_box_at_column(cam, col, D, l=4.0, w=6.0, h=4.0):
    phi = (col - (W - 1) / 2.0) / cam.f_ref                              # CYLINDRICAL azimuth (column linear in azimuth)
    d = np.array([math.cos(phi), -math.sin(phi)])
    c = (D + l / 2.0) * d
    return _box(c[0], c[1], float(cam.t_cam_in_rig[2]), l, w, h, math.atan2(d[1], d[0]))


def _pipeline_camera():
    """The camera the TRAINER builds: a RigCamera on the 416 x 1024 cylindrical canonical frame -> `camera_of`."""
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    fr = CanonicalFrame.from_hfov(120.0, H, W, "cylindrical")
    return BDA.camera_of(RigCamera.nominal(fr, height_m=1.5, x_m=0.0)), fr


def test_camera_of_keeps_the_cylindrical_frame():
    cam, fr = _pipeline_camera()
    assert (cam.height, cam.width, cam.projection) == (416, 1024, "cylindrical")
    assert cam.f_ref == pytest.approx(488.92398517830253, abs=1e-9) == pytest.approx(fr.f_ref, abs=1e-12)


@pytest.mark.parametrize("col,D", [(895, 20.0), (130, 35.0), (700, 12.0)])
def test_known_range_off_centre_through_the_pipeline_camera(col, D):
    cam, _ = _pipeline_camera()
    b = _face_box_at_column(cam, col, D)
    row = 207
    v = (row - (H - 1) / 2.0) / cam.f_ref
    assert _known_range_ok(cam, [b], row, col, D, 1e-9, "rho")[0]
    assert _known_range_ok(cam, [b], row, col, D * math.sqrt(1 + v * v), 1e-9, "range")[0]      # the `range` depth the head learns


@pytest.mark.parametrize("D", [5.0, 20.0, 99.0])
def test_known_range_at_the_centre_pixel(D):
    cam, _ = _pipeline_camera()
    b = [_box(D + 2.0, 0.0, 1.5, 4.0, 6.0, 4.0)]
    for r, c in ((207, 511), (208, 512)):
        ok, got = _known_range_ok(cam, b, r, c, D, 1e-3)
        assert ok and abs(got - D) < 2e-6 * D + 1e-9, (D, r, c, got)


def test_red_arm_pinhole_targets_fail_the_off_centre_control():
    """THE DELIBERATE REGRESSION: the pinhole formula in place of the cylindrical one (same grid, same f_ref)."""
    cam, _ = _pipeline_camera()
    pin = dataclasses.replace(cam, projection="pinhole")
    assert pin.f_ref == cam.f_ref
    assert _known_range_ok(pin, [_box(22.0, 0.0, 1.5, 4.0, 6.0, 4.0)], 207, 511, 20.0, 1e-3)[0]     # at the centre they agree
    fails = 0
    for col, D in ((895, 20.0), (130, 35.0), (700, 12.0)):
        b = _face_box_at_column(cam, col, D)
        ok_cyl = _known_range_ok(cam, [b], 207, col, D, 1e-9, "rho")[0]
        ok_pin = _known_range_ok(pin, [b], 207, col, D, 1e-9, "rho")[0]
        assert ok_cyl
        fails += int(not ok_pin)
    assert fails == 3                                                       # the control FAILS on every off-centre case


def test_red_arm_a_camera_converter_that_drops_the_projection_fails_the_control(monkeypatch):
    """The pipeline's own converter, mutated to hand the builder a pinhole camera: the SAME control must go red."""
    real = BDA.camera_of

    def bad(rc, **kw):
        return dataclasses.replace(real(rc, **kw), projection="pinhole")
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    fr = CanonicalFrame.from_hfov(120.0, H, W, "cylindrical")
    rc = RigCamera.nominal(fr, height_m=1.5, x_m=0.0)
    b = _face_box_at_column(real(rc), 895, 20.0)
    assert _known_range_ok(real(rc), [b], 207, 895, 20.0, 1e-9, "rho")[0]
    monkeypatch.setattr(BDA, "camera_of", bad)
    assert not _known_range_ok(BDA.camera_of(rc), [b], 207, 895, 20.0, 1e-9, "rho")[0]


# ===================================================================================================================== #
# 2. targets from the batch                                                                                             #
# ===================================================================================================================== #
def _agent_block(boxes_rows, pad=4):
    """[B] lists of (cx, cy, cz, l, w, h, yaw) -> the batch's agent_* tensors (the dataset's layout: box = cx, cy, l, w)."""
    B = len(boxes_rows)
    bx = torch.zeros(B, pad, 4)
    yaw, cz, h = torch.zeros(B, pad), torch.zeros(B, pad), torch.zeros(B, pad)
    valid, zh = torch.zeros(B, pad, dtype=torch.bool), torch.zeros(B, pad, dtype=torch.bool)
    for b, rows in enumerate(boxes_rows):
        for a, (cx, cy, z, l, w, hh, yw) in enumerate(rows):
            bx[b, a] = torch.tensor([cx, cy, l, w])
            yaw[b, a], cz[b, a], h[b, a] = yw, z, hh
            valid[b, a] = True
            zh[b, a] = True
    return {"agent_box": bx, "agent_yaw": yaw, "agent_valid": valid, "agent_cz": cz, "agent_h": h, "agent_zh_mask": zh,
            "agent_ep": torch.tensor([7] * B), "agent_label": torch.tensor([True] * B)}


def test_boxes_from_batch_layout_is_cx_cy_cz_l_w_h_yaw():
    blk = _agent_block([[(10.0, -2.0, 0.9, 4.5, 1.9, 1.6, 0.3)], []])
    boxes, st = BDA.boxes_from_batch(blk)
    np.testing.assert_allclose(boxes[0], [[10.0, -2.0, 0.9, 4.5, 1.9, 1.6, 0.3]], atol=1e-6)
    assert boxes[1].shape == (0, 7) and st == {"n_rows": 2, "n_rows_labelled": 2, "n_boxes": 1, "n_boxes_no_zh": 0}


def test_no_label_row_is_empty_and_missing_zh_is_counted_not_dropped_silently():
    blk = _agent_block([[(10.0, 0.0, 0.9, 4.5, 1.9, 1.6, 0.0), (20.0, 0.0, 0.9, 4.5, 1.9, 1.6, 0.0)], [(8.0, 0.0, 0.9, 4.0, 2.0, 1.5, 0.0)]])
    blk["agent_zh_mask"][0, 1] = False                                   # a cuboid without a z/h label cannot be ray-cast: counted
    blk["agent_label"] = torch.tensor([True, False])                     # NO_LABEL is not 'labelled clear'
    boxes, st = BDA.boxes_from_batch(blk)
    assert boxes[0].shape == (1, 7) and boxes[1].shape == (0, 7)
    assert st["n_boxes_no_zh"] == 1 and st["n_rows_labelled"] == 1


def test_a_batch_without_the_cuboids_refuses_not_skips():
    blk = _agent_block([[]])
    del blk["agent_cz"]
    with pytest.raises(SystemExit, match="agent_cz"):
        BDA.boxes_from_batch(blk)


def test_build_targets_known_cell_depth_and_zero_rows():
    cam, _ = _pipeline_camera()
    big = _box(40.0, 0.0, 1.5, 4.0, 40.0, 30.0)                          # a wall 38 m ahead filling the centre of the frame
    out = BDA.build_targets([[big], np.zeros((0, 7))], [cam, cam])
    t8, fg8, w8 = out[8]
    assert t8.shape == (2, 80, 52, 128) and fg8.shape == (2, 52, 128) and out[16][0].shape == (2, 80, 26, 64)
    assert fg8[0].any() and not fg8[1].any() and t8[1].sum() == 0.0      # a row with no cuboid: zero fg, exactly-zero target mass
    assert np.allclose(t8[0].sum(axis=0)[fg8[0]], 1.0, atol=1e-6)        # a FG cell's target sums to 1
    assert np.all(t8[0].sum(axis=0)[~fg8[0]] == 0.0)                     # a non-FG cell carries NO label (alpha_bg is inert)
    cells = np.argwhere(fg8[0])
    r, c = cells[np.argmin(np.abs(cells[:, 0] - 25.5) + np.abs(cells[:, 1] - 63.5))]        # the FG cell nearest the frame centre
    phi, v = (c * 8 + 3.5 - 511.5) / cam.f_ref, (r * 8 + 3.5 - 207.5) / cam.f_ref
    want = 38.0 * math.sqrt(1 + v * v) / math.cos(phi)                   # the wall's near face (x = 38 m), range along the pixel ray
    assert float(BD.expected_depth(t8[0][:, r, c])) == pytest.approx(want, rel=0.02)
    assert set(np.unique(w8[0])) <= {np.float32(0.0), np.float32(3.25), np.float32(0.25)}


def test_equalized_bottom_rows_are_ignored_not_targets():
    cam, _ = _pipeline_camera()
    big = _box(3.0, 0.0, 0.2, 4.0, 40.0, 4.0)                            # a box reaching the bottom of the frame
    a = BDA.build_targets([[big]], [cam], equalize_bottom_rows=0)
    b = BDA.build_targets([[big]], [cam], equalize_bottom_rows=43)
    assert a[8][1][0, -5:].any() and not b[8][1][0, -5:].any()          # the last 5 s8 rows lie wholly in the zeroed 43 px: no FG there
    assert (b[8][2][0, -5:] == 0.0).all()                                # and no weight (IGNORE), so no gradient either


# ===================================================================================================================== #
# 3. the loss form                                                                                                       #
# ===================================================================================================================== #
def _toy_targets(B=2, D=80, H8=4, W8=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    fg = torch.zeros(B, H8, W8, dtype=torch.bool)
    fg[:, 1, 2:4] = True
    tgt = torch.zeros(B, D, H8, W8)
    for b in range(B):
        for (r, c) in torch.nonzero(fg[b]).tolist():
            i = int(torch.randint(0, D - 1, (1,), generator=g))
            tgt[b, i, r, c], tgt[b, i + 1, r, c] = 0.3, 0.7
    wt = torch.full((B, H8, W8), 0.25)
    wt[~torch.ones_like(fg)] = 0.0
    wt = torch.where(fg, torch.tensor(3.25), wt)                         # alpha_fg on FG, alpha_bg on the rest (an inert label-free BG)
    return {8: (tgt, fg, wt)}


def test_loss_equals_alpha_fg_times_the_hand_computed_soft_ce():
    tg = _toy_targets()
    lg = torch.randn(2, 80, 4, 6, generator=torch.Generator().manual_seed(1))
    loss, met = BDA.depth_loss({8: lg}, tg)
    lp = torch.log_softmax(lg, 1)
    ce = [-(tg[8][0][b, :, 1, c] * lp[b, :, 1, c]).sum() for b in range(2) for c in (2, 3)]
    assert float(loss) == pytest.approx(3.25 * float(sum(ce)) / 4.0, rel=1e-6)
    assert met["n_fg_cells_s8"] == 4.0


def test_loss_is_on_object_cells_only_and_alpha_bg_is_inert():
    """Perturbing the logits (and alpha_bg) on every NON-FG cell changes neither the loss nor any gradient there (exactly 0)."""
    tg = _toy_targets()
    fg = tg[8][1]
    lg = torch.randn(2, 80, 4, 6, generator=torch.Generator().manual_seed(2), requires_grad=True)
    l0, _ = BDA.depth_loss({8: lg}, tg)
    (g0,) = torch.autograd.grad(l0, lg)
    lg2 = lg.detach().clone()
    lg2[:, :, ~fg[0]] += 5.0 * torch.randn_like(lg2[:, :, ~fg[0]])      # the same rows (fg identical across the batch here)
    lg2.requires_grad_(True)
    tg2 = {8: (tg[8][0], fg, torch.where(fg, tg[8][2], torch.full_like(tg[8][2], 99.0)))}   # alpha_bg 0.25 -> 99
    l1, _ = BDA.depth_loss({8: lg2}, tg2)
    (g1,) = torch.autograd.grad(l1, lg2)
    assert float(l1) == pytest.approx(float(l0), rel=1e-6)
    assert float(g0[:, :, ~fg[0]].abs().max()) == 0.0 and float(g1[:, :, ~fg[0]].abs().max()) == 0.0
    assert torch.allclose(g0[:, :, fg[0]], g1[:, :, fg[0]], atol=1e-7)


def _object_pixels_only_check(loss_fn) -> bool:
    """True iff `loss_fn(logits, targets)` is invariant to logits on non-FG cells (the property under test)."""
    tg = _toy_targets()
    fg = tg[8][1]
    lg = torch.randn(2, 80, 4, 6, generator=torch.Generator().manual_seed(2))
    lg2 = lg.clone()
    lg2[:, :, ~fg[0]] += 5.0 * torch.randn(lg2[:, :, ~fg[0]].shape, generator=torch.Generator().manual_seed(9))   # (a constant shift is a softmax no-op)
    return abs(float(loss_fn({8: lg}, tg)[0]) - float(loss_fn({8: lg2}, tg)[0])) < 1e-6


def test_red_arm_a_loss_on_background_pixels_turns_the_object_only_check_red():
    assert _object_pixels_only_check(BDA.depth_loss)

    def bg_loss(logits, targets):                                         # MUTANT: a uniform label on BG cells (what an invented BG label does)
        tgt, fg, wt = targets[8]
        tgt = torch.where(fg[:, None], tgt, torch.full_like(tgt, 1.0 / tgt.shape[1]))
        return BDA.depth_loss(logits, {8: (tgt, fg, wt)})
    assert not _object_pixels_only_check(bg_loss)
    # (the second spelling of the same mutation: the CE summed over every cell with weight > 0, not only the FG ones)

    def all_cells(logits, targets):
        tgt, fg, wt = targets[8]
        tgt = torch.where(fg[:, None], tgt, torch.full_like(tgt, 1.0 / tgt.shape[1]))
        ce = -(tgt * torch.log_softmax(logits[8], 1)).sum(1)
        return (wt * ce).sum() / fg.sum(), {}
    assert not _object_pixels_only_check(all_cells)


def test_a_batch_with_no_object_cell_gives_exactly_zero_with_the_graph_kept():
    tg = _toy_targets()
    tg = {8: (tg[8][0] * 0.0, torch.zeros_like(tg[8][1]), tg[8][2])}
    lg = torch.randn(2, 80, 4, 6, requires_grad=True)
    loss, met = BDA.depth_loss({8: lg}, tg)
    assert float(loss) == 0.0 and loss.requires_grad and met["n_fg_cells_s8"] == 0.0
    (g,) = torch.autograd.grad(loss, lg)
    assert float(g.abs().max()) == 0.0


# ===================================================================================================================== #
# 4. the tiny rig: reachability, zero control, eval-forward bit-identity                                                #
# ===================================================================================================================== #
_ORIG_MODEL = v3.RefCV3Model                                                 # captured once: a second monkeypatch must not wrap the first


def _build_rig(monkeypatch, flag=True, w=0.1):
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: _ORIG_MODEL(cfg, perc_rig=True))
    model, cfg = R._build()
    frame = model._lift_bank_hires.frame
    model._rig_camera = R.RigCamera.nominal(frame, height_m=1.5, x_m=1.5)
    args = argparse.Namespace(r9_box_depth_aux=flag, w_box_depth=(w if flag else 0.0))
    F.attach(model, args)
    return model, cfg, args


def _unzero_bn(model):
    """timm zero-inits every residual block's last BN gamma (zero_init_last) and this rig is NOT pretrained, so layer2/3's residual convs
    would read exactly-zero gradient from ANY loss: a rig artefact, not a wiring one. Set gamma = 1 so the probe sees the whole stage."""
    for m in model.core.encoder.net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)


def _src_of(model):
    """The worker's TargetSource, built the way ``attach_dataset`` builds it (camera object + the trunk's zeroed rows)."""
    eq = int(getattr(getattr(model.core.encoder, "cfg", None), "equalize_bottom_rows", 0) or 0)
    return F.TargetSource(model._rig_camera, eq)


def _with_targets(batch, model, src=None):
    """C3B: B1's targets are built per ITEM in the DataLoader worker; the step sees the default-collated ``bd_*`` entries."""
    src = src or _src_of(model)
    B = int(batch["agent_box"].shape[0])
    keys = BDA.KEYS_NEEDED + ("agent_label",)
    items = [F.item_targets({k: batch[k][i] for k in keys if k in batch}, src) for i in range(B)]
    batch.update(torch.utils.data.default_collate(items))
    return batch


def _batch_with_agents(cfg, model):
    batch = R._batch(cfg)
    rows = [[(7.0, 0.0, 1.5, 6.0, 7.0, 4.0, 0.0)], [(6.0, 0.5, 1.5, 6.0, 7.0, 4.0, 0.0)]]
    batch.update(_agent_block(rows))
    batch["agent_ep"] = batch["map_ep"].clone()
    return _with_targets(batch, model)


def _named_grads(loss, model, retain=True):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    gs = torch.autograd.grad(loss, [dict(model.named_parameters())[n] for n in names], retain_graph=retain, allow_unused=True)
    return {n: (None if g is None else g) for n, g in zip(names, gs)}


def _is_zero(g):
    return g is None or float(g.abs().max()) == 0.0


def test_flag_off_builds_nothing_and_adds_no_key(monkeypatch):
    model, cfg, _ = _build_rig(monkeypatch, flag=False)
    assert F._head(model) is None and not any(".box_depth_aux." in k for k in model.state_dict())
    batch = _batch_with_agents(cfg, model)
    model.train()
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert not any(k.startswith("box_depth") for k in out)


def test_reachability_every_hop_gradient_nonzero_and_the_zero_control(monkeypatch):
    model, cfg, args = _build_rig(monkeypatch)
    head = F._head(model)
    _unzero_bn(model)
    assert head is not None and getattr(model, "_w_box_depth") == 0.1
    batch = _batch_with_agents(cfg, model)
    model.train()
    torch.manual_seed(5)
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert float(out["box_depth"]) > 0.0 and math.isfinite(float(out["box_depth"]))
    assert out["box_depth_n_fg_cells_s8"] >= 1.0 and out["box_depth_n_fg_cells_s16"] >= 1.0
    term = model._w_box_depth * out["box_depth"]
    gd = _named_grads(term, model)
    # (a) EVERY new parameter gets a non-zero gradient
    new = {n: g for n, g in gd.items() if n.startswith(F.KEY_PREFIX)}
    assert len(new) == 12 and all(not _is_zero(g) for g in new.values()), [n for n, g in new.items() if _is_zero(g)]
    # (b) the trunk's s8 and s16 stages get gradient FROM THE DEPTH TERM (conv weights)
    st = model.core.encoder.s8_stage()["module"]
    s8w = [g for n, g in gd.items() if n.startswith(f"core.encoder.net.{st}.") and g is not None and g.ndim == 4]
    s16w = [g for n, g in gd.items() if n.startswith("core.encoder.net.layer3.") and g is not None and g.ndim == 4]
    assert st == "layer2" and s8w and all(not _is_zero(g) for g in s8w)
    assert s16w and all(not _is_zero(g) for g in s16w)
    # (c) THE ZERO CONTROL: the depth term's gradient on the BEV lift / encoder, the 10 cm map head, the box decoder and the agent head is EXACTLY 0.0
    ctrl = [n for n in gd if n.startswith(("_map_hires.", "core.agent_head.")) or (n.startswith("_perception.") and not n.startswith(F.KEY_PREFIX))]
    assert len(ctrl) > 20
    assert all(_is_zero(gd[n]) for n in ctrl), [n for n in ctrl if not _is_zero(gd[n])][:3]
    # (d) the probe can see non-zero where there should be: the map term DOES reach the 10 cm map head (positive control for (c))
    gm = _named_grads(out["map_hires"], model)
    assert any(not _is_zero(g) for n, g in gm.items() if n.startswith("_map_hires."))
    assert all(_is_zero(g) for n, g in gm.items() if n.startswith(F.KEY_PREFIX))      # and the map term does not touch the depth head


@pytest.mark.parametrize("stride,must,must_not", [(8, "layer2", "layer3"), (16, "layer3", "layer4")])
def test_each_stride_reaches_its_own_trunk_stage_and_not_the_deeper_one(monkeypatch, stride, must, must_not):
    model, cfg, _ = _build_rig(monkeypatch)
    _unzero_bn(model)
    batch = _batch_with_agents(cfg, model)
    model.train()
    real = BDA.depth_loss
    monkeypatch.setattr(BDA, "depth_loss", lambda lg, tg: real({stride: lg[stride]}, {stride: tg[stride]}))
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    gd = _named_grads(out["box_depth"], model)
    reach = lambda stage: any(not _is_zero(g) for n, g in gd.items() if n.startswith(f"core.encoder.net.{stage}.") and g is not None and g.ndim == 4)  # noqa: E731
    assert reach(must) and not reach(must_not)                              # s8 stops at layer2; s16 stops at layer3 (layer4 is the s32 head)


def test_red_arm_a_dropped_call_in_the_loss_makes_the_reachability_check_red(monkeypatch):
    model, cfg, _ = _build_rig(monkeypatch)
    batch = _batch_with_agents(cfg, model)
    model.train()
    monkeypatch.setattr(F, "loss_term", lambda *a, **k: None)               # the mutation: the trainer never adds the term
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert "box_depth" not in out                                           # the reachability assertion above would fail on this line
    g = torch.autograd.grad(out["loss"], list(F._head(model).parameters()), allow_unused=True)
    assert all(_is_zero(x) for x in g)                                      # every head parameter reads exactly 0: the probe sees the break


def test_trainer_hops_are_present_in_the_source():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    for needle in ("from tanitad.train import box_depth_aux_flag as _bda", "_bda.register_dvb_levers(_dvb.register)", "_bda.pin_args(args)",
                   "_bda.attach(model, args)", "_bd_term = _bda.loss_term(model, out, batch, device, extra)", "loss = loss + _bd_term",
                   '"r9_box_depth_aux": _bda.stamp(model, args)', '"--r9-box-depth-aux"', 'g8.add_argument("--w-box-depth"', '"w_box_depth": {',
                   "_bda.attach_dataset(ds, model, args)", "item.update(_bda.item_targets(item, self._r9_bd))"):
        assert src.count(needle) == 1, needle
    # C3B: the targets are built in the DataLoader worker -- the dataset hook sits at the end of V3Dataset.__getitem__ (after the agent block
    # that makes the cuboids), and the dataset is handed the camera BEFORE the training DataLoader is built (workers pickle/fork it at the first iter)
    h = src.index("item.update(_bda.item_targets(item, self._r9_bd))")
    assert src.index("item.update(self._agent_item(ep, t + w - 1))") < h < src.index("def _nullctx(")
    a = src.index("_bda.attach_dataset(ds, model, args)")
    assert a < src.index("ds, batch_size=args.batch, sampler=_train_sampler, num_workers=args.workers,")
    assert src.index("_bda.attach(model, args)") < a
    # the loss call sits inside compute_losses_perc_rig (between its def and the next top-level def)
    i, j = src.index("def compute_losses_perc_rig("), src.index("def _lan_arm_preflight(")
    assert i < src.index("_bda.loss_term(model, out, batch, device, extra)") < j
    ld = (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_text(encoding="utf-8")
    assert ld.count("tr._bda.attach(model, args)") == 1


def test_defaults_are_off_and_named_flags_are_required():
    a = T.build_parser().parse_args(["--arm", "flat", "--out", "x"])
    assert a.r9_box_depth_aux is False and a.w_box_depth == 0.0
    F.pin_args(a)                                                           # the default argv passes untouched


def test_pin_args_refusals():
    base = dict(r9_box_depth_aux=True, w_box_depth=0.1, r9_perc_rig=True, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    F.pin_args(argparse.Namespace(**base))                                  # the complete argv is admitted
    for k, v, frag in (("w_box_depth", 0.0, "positive weight"), ("r9_perc_rig", False, "r9-perc-rig"), ("map_hires", "off", "map-hires"),
                       ("join3d", "", "join3d"), ("agent_join", "", "agent-join"), ("agent_rig_camera", "off", "agent-rig-camera")):
        with pytest.raises(SystemExit, match=frag):
            F.pin_args(argparse.Namespace(**{**base, k: v}))
    with pytest.raises(SystemExit, match="dead flag"):                      # the weight alone multiplies nothing
        F.pin_args(argparse.Namespace(r9_box_depth_aux=False, w_box_depth=0.2))


def test_eval_forward_is_bit_identical_with_the_flag_on_and_off(monkeypatch):
    on, cfg, _ = _build_rig(monkeypatch, flag=True)
    off, _, _ = _build_rig(monkeypatch, flag=False)
    sd_on, sd_off = on.state_dict(), off.state_dict()
    extra = sorted(set(sd_on) - set(sd_off))
    assert extra and all(k.startswith(F.KEY_PREFIX) for k in extra) and set(sd_off) <= set(sd_on)
    assert all(torch.equal(sd_on[k], sd_off[k]) for k in sd_off)           # OFF/ON share EVERY init draw (the head is built inside fork_rng)
    batch = _batch_with_agents(cfg, on)
    on.eval(), off.eval()
    fr = T.frames_to_device(batch["frames"][:, -1:], "cpu")

    def fwd(m):
        g, v = m._lift_bank_hires.for_episodes(batch["map_ep"], device="cpu")
        with torch.no_grad():
            return m(fr, perception_grid=g, perception_valid=v)
    assert _tree_equal(fwd(on), fwd(off))
    lo = T.compute_losses_v3(on, batch, "cpu", log_metrics=False)
    assert not any(k.startswith("box_depth") for k in lo)                   # eval mode: the head is not even called


def _tree_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    return a == b


def test_red_arm_the_head_left_in_the_eval_forward_turns_the_bit_identity_check_red(monkeypatch):
    on, cfg, _ = _build_rig(monkeypatch, flag=True)
    off, _, _ = _build_rig(monkeypatch, flag=False)
    batch = _batch_with_agents(cfg, on)
    on.eval(), off.eval()
    fr = T.frames_to_device(batch["frames"][:, -1:], "cpu")
    real_forward = on.forward

    def leaky(*a, **k):                                                     # MUTANT: the depth head runs inside the eval forward (ON model only)
        out = real_forward(*a, **k)
        out["box_depth_logits"] = F._head(on)(out["fmap_s8"], out["fmap_s16"])
        return out
    on.forward = leaky

    def fwd(m):
        g, v = m._lift_bank_hires.for_episodes(batch["map_ep"], device="cpu")
        with torch.no_grad():
            return m(fr, perception_grid=g, perception_valid=v)
    assert not _tree_equal(fwd(on), fwd(off))


def test_train_loss_with_the_flag_on_is_the_off_loss_plus_the_weighted_term(monkeypatch):
    on, cfg, _ = _build_rig(monkeypatch, flag=True)
    off, _, _ = _build_rig(monkeypatch, flag=False)
    batch = _batch_with_agents(cfg, on)
    on.train(), off.train()
    torch.manual_seed(5)
    lo = T.compute_losses_v3(on, batch, "cpu", log_metrics=False)
    torch.manual_seed(5)
    lf = T.compute_losses_v3(off, batch, "cpu", log_metrics=False)
    assert float(lo["loss"]) == pytest.approx(float(lf["loss"]) + 0.1 * float(lo["box_depth"]), rel=1e-6, abs=1e-7)
    for k, v in lf.items():
        if k != "loss" and torch.is_tensor(v) and v.numel() == 1:
            assert float(lo[k]) == float(v), k                              # every other loss scalar is bit-identical


# ===================================================================================================================== #
# 5. the rig warm start's NAMED exemption, G-DVB, the stamp                                                              #
# ===================================================================================================================== #
class _Per(torch.nn.Module):
    def __init__(self, head: bool, extra: bool = False):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        if head:
            self.box_depth_aux = torch.nn.Linear(4, 2)
        if extra:
            self.other = torch.nn.Linear(2, 2)


class _Model(torch.nn.Module):
    def __init__(self, head: bool, extra: bool = False):
        super().__init__()
        self._perception = _Per(head, extra)


def _src(tmp_path):
    src = _Model(False)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "step": 7}, str(p))
    return src, str(p)


def test_off_model_has_no_depth_exemption():
    assert rig.box_depth_exemptions(_Model(False)) == ()


def test_head_keys_are_named_and_everything_else_still_loads(tmp_path):
    src, p = _src(tmp_path)
    m = _Model(True)
    assert rig.box_depth_exemptions(m) == ("_perception.box_depth_aux.",)
    rec = rig.warm_start_perc_rig(m, p)
    assert rec["n_loaded"] == 2 and torch.equal(m._perception.proj.weight, src._perception.proj.weight)


def test_red_head_without_the_exemption_refuses(tmp_path, monkeypatch):
    _, p = _src(tmp_path)
    monkeypatch.setattr(rig, "box_depth_exemptions", lambda model: ())
    with pytest.raises(SystemExit, match=r"rig key\(s\) the source lacks"):
        rig.warm_start_perc_rig(_Model(True), p)


def test_an_unnamed_missing_key_beyond_the_head_still_refuses(tmp_path):
    _, p = _src(tmp_path)
    with pytest.raises(SystemExit):
        rig.warm_start_perc_rig(_Model(True, extra=True), p)


def test_dvb_checks_agree_and_disagree(monkeypatch):
    model, _, args = _build_rig(monkeypatch)
    assert F.dvb_check(model, args) == [] and F.dvb_check_weight(model, args) == []
    assert F.dvb_check(model, argparse.Namespace(r9_box_depth_aux=False, w_box_depth=0.1))                 # declared OFF, head built
    assert F.dvb_check(model, argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.1)) == []
    assert F.dvb_check_weight(model, argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.2))            # weight disagrees
    off, _, oargs = _build_rig(monkeypatch, flag=False)
    assert F.dvb_check(off, oargs) == [] and F.dvb_check_weight(off, oargs) == []
    assert F.dvb_check(off, argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.1))                    # declared ON, nothing built
    assert F.dvb_check_weight(off, argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.1))
    bad = copy.deepcopy(model)
    bad._perception.box_depth_aux.n_bins = 64
    assert any("LID bins" in str(m) for m in F.dvb_check(bad, args))
    assert {"r9_box_depth_aux", "w_box_depth"} <= set(T._dvb.REGISTRY)


def test_stamp_states_alpha_bg_is_inert_and_is_absent_when_off(monkeypatch):
    model, _, args = _build_rig(monkeypatch)
    st = F.stamp(model, args)
    assert st["alpha_fg"] == 3.25 and st["alpha_bg"] == 0.25 and "INERT" in st["alpha_bg_note"] and st["depth_kind"] == "range"
    assert st["n_bins"] == 80 and st["strides"] == [8, 16] and st["head"]["n_params"] == sum(p.numel() for p in F._head(model).parameters())
    off, _, oargs = _build_rig(monkeypatch, flag=False)
    assert F.stamp(off, oargs) is None


# ===================================================================================================================== #
# 6. C3B: the targets are built OFF the step, in the DataLoader workers                                                  #
# ===================================================================================================================== #
def _ka_block():
    """A fixed batch of 4 rows: 3 boxes / labelled-clear / NO_LABEL (a cuboid that must NOT be cast) / a box without z,h (counted, not cast)."""
    rows = [[(7.0, 0.0, 1.5, 6.0, 7.0, 4.0, 0.0), (12.0, -3.0, 1.0, 4.5, 1.9, 1.6, 0.4), (25.0, 4.0, 0.8, 4.5, 1.9, 1.6, -0.2)],
            [],
            [(9.0, 2.0, 1.2, 4.5, 1.9, 1.6, 1.0)],
            [(6.0, 0.5, 1.5, 6.0, 7.0, 4.0, 0.0)]]
    blk = _agent_block(rows)
    blk["agent_label"] = torch.tensor([True, True, False, True])
    blk["agent_zh_mask"][3, 0] = False
    blk["agent_ep"] = torch.tensor([7, 8, 7, 8])
    return blk


def _bank(monkeypatch):
    from tanitad.refs.refc_agents import RigCameraBank
    model, _, _ = _build_rig(monkeypatch)
    fr = model._lift_bank_hires.frame
    return RigCameraBank(by_episode={7: R.RigCamera.nominal(fr, height_m=1.5, x_m=1.5), 8: R.RigCamera.nominal(fr, height_m=1.2, x_m=2.0)},
                         default=None), model


class _ConstBank:
    """The single-camera case as a bank: every row the same RigCamera (the tiny rig's ``model._rig_camera``)."""

    def __init__(self, cam):
        self.cam = cam

    def for_episodes(self, ep):
        return [self.cam] * int(torch.as_tensor(ep).numel())


def _onstep_reference(batch, bank, eq):
    """C3's ON-STEP build, line for line as ``loss_term`` ran it before the move: boxes_from_batch -> per-row cameras -> build_targets(threads=4) -> to_tensors."""
    boxes, st = BDA.boxes_from_batch(batch)
    cams = [BDA.camera_of(c) for c in bank.for_episodes(batch["agent_ep"])]
    return BDA.to_tensors(BDA.build_targets(boxes, cams, equalize_bottom_rows=eq, threads=4), "cpu"), st


def _worker_batch(blk, src):
    keys = BDA.KEYS_NEEDED + ("agent_label",)
    items = [F.item_targets({k: blk[k][i] for k in keys}, src) for i in range(int(blk["agent_box"].shape[0]))]
    return torch.utils.data.default_collate(items)


def _bytes_equal(ref, wb):
    for s_ in F.STRIDES:
        for name, a in zip(("t", "fg", "w"), ref[s_]):
            b = wb[f"bd_{name}{s_}"]
            if a.dtype != b.dtype or a.shape != b.shape or a.numpy().tobytes() != b.numpy().tobytes():
                return False
    return True


@pytest.mark.parametrize("eq", [0, 16])
def test_ka_worker_targets_are_byte_identical_to_the_on_step_targets(monkeypatch, eq):
    """THE KA that the move changed nothing: per-item targets built by ``item_targets`` and default-collated are BYTE-identical (dtype, shape,
    bytes) to C3's on-step batch build, over a fixed batch with two different clip cameras, a clear row, a NO_LABEL row and a z-less box."""
    bank, _ = _bank(monkeypatch)
    blk = _ka_block()
    ref, st = _onstep_reference(blk, bank, eq)
    wb = _worker_batch(blk, F.TargetSource(bank, eq))
    assert _bytes_equal(ref, wb)
    assert int(wb["bd_n_boxes"].sum()) == st["n_boxes"] == 3 and int(wb["bd_n_boxes_no_zh"].sum()) == st["n_boxes_no_zh"] == 1
    # non-vacuous: row 0 has object cells at both strides, the clear row and the NO_LABEL row have none
    assert wb["bd_fg8"][0].any() and wb["bd_fg16"][0].any() and not wb["bd_fg8"][1].any() and not wb["bd_fg8"][2].any()
    assert float(wb["bd_t8"][2].abs().sum()) == 0.0


def test_red_arm_the_ka_sees_a_wrong_camera_in_the_worker(monkeypatch):
    """The KA can fail: the same worker with the two clips' cameras swapped is NOT byte-identical to the on-step build."""
    from tanitad.refs.refc_agents import RigCameraBank
    bank, _ = _bank(monkeypatch)
    swapped = RigCameraBank(by_episode={7: bank.by_episode[8], 8: bank.by_episode[7]}, default=None)
    blk = _ka_block()
    ref, _ = _onstep_reference(blk, bank, 0)
    assert _bytes_equal(ref, _worker_batch(blk, F.TargetSource(bank, 0)))
    assert not _bytes_equal(ref, _worker_batch(blk, F.TargetSource(swapped, 0)))


def test_target_source_survives_a_pickle_round_trip_as_a_spawned_worker_needs(monkeypatch):
    import pickle
    bank, _ = _bank(monkeypatch)
    blk = _ka_block()
    src = pickle.loads(pickle.dumps(F.TargetSource(bank, 16)))
    assert _bytes_equal(_onstep_reference(blk, bank, 16)[0], _worker_batch(blk, src))


def test_item_with_no_camera_in_the_bank_refuses_loudly_in_the_worker(monkeypatch):
    from tanitad.refs.refc_agents import RigCameraBank
    bank, _ = _bank(monkeypatch)
    blk = _ka_block()
    blk["agent_ep"] = torch.tensor([7, 99, 7, 8])                             # clip 99 is not in the bank
    with pytest.raises(RuntimeError, match="not in the camera bank"):
        _worker_batch(blk, F.TargetSource(RigCameraBank(by_episode=bank.by_episode, default=None), 0))
    item = {k: v[0] for k, v in _ka_block().items() if k != "agent_cz"}       # an item without the cuboid z
    with pytest.raises(RuntimeError, match="agent_cz"):
        F.item_targets(item, F.TargetSource(bank, 0))


def test_a_batch_without_the_precomputed_targets_refuses_not_skips_not_rebuilds(monkeypatch):
    """Head built + training + the dataset hook absent => the loss REFUSES (it does not skip the term and does not rebuild on the step)."""
    model, cfg, _ = _build_rig(monkeypatch)
    batch = _batch_with_agents(cfg, model)
    model.train()
    ok = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert "box_depth" in ok
    stripped = {k: v for k, v in batch.items() if not k.startswith("bd_")}
    with pytest.raises(SystemExit, match="precomputed"):
        T.compute_losses_v3(model, stripped, "cpu", log_metrics=False)
    one = {k: v for k, v in batch.items() if k != "bd_t16"}                    # a partial set refuses too
    with pytest.raises(SystemExit, match="bd_t16"):
        T.compute_losses_v3(model, one, "cpu", log_metrics=False)
    model.eval()                                                                # eval never reads them
    assert "box_depth" not in T.compute_losses_v3(model, stripped, "cpu", log_metrics=False)


def test_the_step_builds_nothing_the_cylindrical_builder_is_not_called_in_loss_term(monkeypatch):
    """RED-style: make the builder raise; the training loss still runs because it only reads the collated targets."""
    model, cfg, _ = _build_rig(monkeypatch)
    batch = _batch_with_agents(cfg, model)
    model.train()
    def boom(*a, **k):
        raise AssertionError("the step called the target builder")
    monkeypatch.setattr(BDA, "build_targets", boom)
    monkeypatch.setattr(BD, "build_depth_targets", boom)
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert float(out["box_depth"]) > 0.0 and "box_depth_main_s" in out and out["box_depth_build_s"] >= 0.0


def test_the_loss_from_worker_targets_equals_the_loss_from_the_on_step_targets(monkeypatch):
    """The soft-CE fed the on-step tensors and fed the collated worker tensors (through ``targets_from_batch``) is the same number, bit for bit."""
    model, cfg, _ = _build_rig(monkeypatch)
    batch = _batch_with_agents(cfg, model)
    ref, _ = _onstep_reference(batch, _ConstBank(model._rig_camera), _src_of(model).eq)
    wt = F.targets_from_batch(batch, "cpu")[0]
    torch.manual_seed(7)
    lg = {s_: torch.randn(2, 80, *ref[s_][1].shape[-2:]) for s_ in F.STRIDES}
    l1, m1 = BDA.depth_loss(lg, ref)
    l2, m2 = BDA.depth_loss(lg, wt)
    assert float(l1) == float(l2) and m1 == m2 and float(l1) > 0.0


def _fake_agent_item(ds, ep, f):
    blk = _ka_block()
    return {k: blk[k][0] for k in BDA.KEYS_NEEDED + ("agent_label",)} | {"agent_ep": torch.tensor(int(ep.episode_id), dtype=torch.long)}


def test_the_trainers_own_dataset_hook_adds_the_targets_only_when_attached(monkeypatch):
    """The REAL ``V3Dataset.__getitem__`` (synthetic episodes; ``_agent_item`` stubbed with a fixed cuboid block): no ``bd_*`` key unless
    ``ds._r9_bd`` is set; set, the item carries them and they equal the on-step build of the same cuboids."""
    model, cfg, _ = _build_rig(monkeypatch)
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    monkeypatch.setattr(T.V3Dataset, "_agent_item", _fake_agent_item)
    ds.agent_join = object()
    assert not any(k.startswith("bd_") for k in ds[0])                          # OFF: the item key set is unchanged
    ds._r9_bd = _src_of(model)
    it = ds[0]
    assert all(k in it for k in F.BATCH_KEYS + F.CENSUS_KEYS)
    blk = _ka_block()
    one = {k: blk[k][:1].clone() for k in blk}
    one["agent_ep"] = it["agent_ep"].reshape(1)
    ref, _ = _onstep_reference(one, _ConstBank(model._rig_camera), _src_of(model).eq)
    got = torch.utils.data.default_collate([it])
    assert _bytes_equal(ref, got)


def test_attach_dataset_off_on_and_its_startup_refusals(monkeypatch, capsys):
    import types
    bank, model = _bank(monkeypatch)
    on = argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.1)
    off = argparse.Namespace(r9_box_depth_aux=False, w_box_depth=0.0)
    mk = lambda **k: types.SimpleNamespace(**{"agent_join": object(), "join3d": object(), "episodes": [types.SimpleNamespace(episode_id=7), types.SimpleNamespace(episode_id=8)], **k})  # noqa: E731
    ds = mk()
    assert F.attach_dataset(ds, model, off) == {} and not hasattr(ds, "_r9_bd")           # OFF: touches nothing
    model._rig_camera = bank
    rec = F.attach_dataset(ds, model, on)
    assert isinstance(ds._r9_bd, F.TargetSource) and rec["n_episodes_checked"] == 2 and rec["workers"] == 0
    assert "WARNING --workers 0" in capsys.readouterr().out                                   # the main-process cost is printed, not hidden
    assert F.attach_dataset(mk(), model, argparse.Namespace(r9_box_depth_aux=True, w_box_depth=0.1, workers=6))["workers"] == 6
    assert "WARNING" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="agent_join / join3d"):
        F.attach_dataset(mk(join3d=None), model, on)
    with pytest.raises(SystemExit, match="covers no camera for 1"):                        # episode 9 has no camera: refused at STARTUP
        F.attach_dataset(mk(episodes=[types.SimpleNamespace(episode_id=7), types.SimpleNamespace(episode_id=9)]), model, on)
    model._rig_camera = None
    with pytest.raises(SystemExit, match="_rig_camera is None"):
        F.attach_dataset(mk(), model, on)
