"""PREREG_V8F_STOPPERC arm A: ``--stop-aux {off,on}`` + ``--w-stop-aux`` + ``--stop-aux-extr`` in ``stack/tanitad/models/v8f_mini.py``, ``stack/tanitad/data/v8f_stopaux.py`` and ``stack/scripts/v8f_mini_train.py``.

A vision-only head on the NOW frame's adapted 4x10 tokens: per-token BCE on the cell the ego's rest point (``stop_x_m``, ``stop_y_m``, NOW frame) projects into (column from the CYLINDRICAL bearing, row from the clip's
``sensor_extrinsics``) + a presence BCE. Nothing reads the head at inference. THIS file proves the lever is REACHABLE from the real caller (CLAUDE.md 2026-09-10/11), hop by hop:

  argv --stop-aux/--w-stop-aux/--stop-aux-extr -> make_config -> cfg -> the BUILT model (stop_head + stop_pres_head) -> declared_vs_built (bad_stop_aux, FLAG_CHECKS) -> make_stop_aux (camera arrays)
  -> stop_aux_probe (G-LIVE on the loop's own label call stop_aux_stage) -> ``stop_lab = stop_aux_stage(b, SA)`` BEFORE the ego stages, ``b["stop_cell"], b["stop_pos"], b["stop_has"]`` AFTER them
  -> V8FMini.loss (per-token BCE + presence BCE + ONE total line) -> train_log.jsonl / selfcheck.json / the checkpoint.

  1. PROJECTION: hand-computed cylindrical cases (straight ahead, +-bearing, a near row, behind the camera, out of the field of view, below the frame, NaN), the pinhole trap, the camera height, parity with the
     single-camera ``RigCamera`` and with ``FrontWideExtrinsics``.
  2. LABELS: the masks (no label, no camera row, no stop = a negative, out of view = a negative).
  3. OFF == BASE (715ef8fc): parameter list, state dict, every loss term, every gradient, forward_infer -- bit for bit; the checkpoint's cfg keeps the base's key set.
  4. ON adds EXACTLY the declared keys and leaves every other parameter's init the base's (the head is built LAST).
  5. G-LIVE (b): finite term, NON-ZERO gradient on both heads AND the adapter, EXACTLY 0.0 on the head at w = 0 / off; (c) a mutation that deletes the total line goes RED.
  6. ADMISSIBILITY (d): the head reads vision only -- static, key-guard and perturbation guards, and the red arms.
  7. The launch gate refuses; ``--init-ckpt`` leaves fresh ONLY the declared keys; the W script.
  8. End to end on the trainer's synthetic world.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_stopaux as S  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
BASE_COMMIT = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
BASE_MODEL_SHA = "41cadb4025cf45a8c65b9eff4a86ea6708fedc109dbecdbe875112cc8b0a21f1"       # sha256 of the base's stack/tanitad/models/v8f_mini.py (blob 1cb245e4)
MIRROR = "C:/Users/Admin/tanitad-push"
STOP_KEYS = {"stop_head.0.weight", "stop_head.0.bias", "stop_head.2.weight", "stop_head.2.bias", "stop_pres_head.0.weight", "stop_pres_head.0.bias", "stop_pres_head.2.weight", "stop_pres_head.2.bias"}
NOMINAL_Q = dict(qx=-0.5, qy=0.5, qz=-0.5, qw=0.5)          # camera -> vehicle for the nominal mount (boresight = vehicle forward, up = up): R = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
CELL = lambda r, c: r * 10 + c                                # noqa: E731


# ------------------------------------------------------------------------------------------------ fixtures / helpers
def _stop_cons(n, seed=0):
    """Controlled rest points for ``n`` windows against the NOMINAL camera at (2, 0, 1.5): every case of the hand table, cycled."""
    pts = [(12.0, 0.0), (2 + 17.3205, 10.0), (2 + 17.3205, -10.0), (-3.0, 0.0), (8.0, 18.79), (float("nan"), float("nan"))]
    xy = torch.tensor([pts[(i + seed) % len(pts)] for i in range(n)], dtype=torch.float32)
    return xy


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("stopaux_world")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def make_batch(world, n=16, seed=0, hmax=10):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx]), J


def table(names, z=1.5, x=2.0):
    return {c: {**NOMINAL_Q, "x": x, "y": 0.0, "z": z} for c in names}


def with_stop(b, J, tab=None, seed=0):
    """The batch with controlled rest points in lab['cons'] and the loop's own label call applied."""
    SA = S.make_stop_aux_arrays(J, tab if tab is not None else table(J["clip"]))
    cons = b["lab"]["cons"].clone()
    xy = _stop_cons(cons.shape[0], seed)
    cons[:, M.CI["stop_x_m"]], cons[:, M.CI["stop_y_m"]] = xy[:, 0], xy[:, 1]
    lon = torch.where(torch.isfinite(xy).all(-1), 2, 6).to(b["lab"]["lon_cls"].dtype)       # as in v9: a rest point exists only on STOP (2); the NaN rows ride a MOVING class (6)
    b2 = {**b, "lab": {**b["lab"], "cons": cons, "lon_cls": lon}}
    cell, pos, has = TR.stop_aux_stage(b2, SA)
    return {**b2, "stop_cell": cell, "stop_pos": pos, "stop_has": has}, SA


def build(J, mod=M, seed=0, wake=True, cls=None, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = (cls or mod.V8FMini)(cfg, mod.P10Prior.synthetic(0), mu, sd)
    if wake:                                                   # the residual head is zero-init: wake it so a change at O's input can be seen at its output
        with torch.no_grad():
            w = m.head_o[2].weight
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return m


def run_loss(m, b, seed=3):
    torch.manual_seed(seed)
    return m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(seed))


def base_module():
    """The BASE model module (715ef8fc), read from the local git mirror and pinned by sha256; skip when the mirror is absent."""
    name = "v8f_mini_base_stopaux"
    if name in sys.modules:
        return sys.modules[name]
    try:
        src = subprocess.run(["git", "-C", MIRROR, "show", f"{BASE_COMMIT}:stack/tanitad/models/v8f_mini.py"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the base commit is not readable from the local git mirror")
    assert hashlib.sha256(src).hexdigest() == BASE_MODEL_SHA, "the base blob is not the pinned one"
    mod = types.ModuleType(name)
    mod.__file__ = name + ".py"
    sys.modules[name] = mod
    exec(compile(src.decode("utf-8"), name + ".py", "exec"), mod.__dict__)
    return mod


def mutated_module(drop_line: str, name: str):
    """A copy of the CURRENT model module with the single source line containing ``drop_line`` replaced by an empty statement (the 'delete the term' mutation)."""
    path = Path(M.__file__)
    lines = path.read_text(encoding="utf-8").split("\n")
    hits = [i for i, ln in enumerate(lines) if drop_line in ln]
    assert len(hits) == 1, f"{drop_line!r} must name exactly one line, found {len(hits)}"
    ln = lines[hits[0]]
    lines[hits[0]] = ln[:len(ln) - len(ln.lstrip())] + "pass"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile("\n".join(lines), str(path), "exec"), mod.__dict__)
    return mod


def check_b(model, b, w=0.5):
    """The card's G-LIVE (b) on a model: a finite stop-aux term, a NON-ZERO gradient on the head AND on the adapter that comes from that term, and the total moved by w x term. Raises AssertionError if any fails."""
    model.cfg.w_stop_aux = w
    model.zero_grad(set_to_none=True)
    t_w = run_loss(model, b)["terms"]
    t_w["total"].backward()
    g_w = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.cfg.w_stop_aux = 0.0
    model.zero_grad(set_to_none=True)
    t_0 = run_loss(model, b)["terms"]
    t_0["total"].backward()
    g_0 = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    term = float(t_w["stop_aux"])
    assert math.isfinite(term) and term > 0.0, f"term {term}"
    assert abs(float(t_w["total"] - t_0["total"]) - w * term) <= 1e-4 * w * term + 1e-5, "the total does not move by w x term"
    d = lambda pre: sum(float((g_w[n] - g_0.get(n, torch.zeros_like(g_w[n]))).abs().sum()) for n in g_w if n.startswith(pre))      # noqa: E731
    assert d("stop_head.") > 0.0 and d("stop_pres_head.") > 0.0, "a head receives no gradient from the term"
    assert d("adapter.") > 0.0, "the adapter receives no gradient from the term"
    zero = sum(float(g_0[n].abs().sum()) for n in g_0 if n.startswith(("stop_head.", "stop_pres_head.")))
    return {"term": term, "head_grad": d("stop_head."), "pres_grad": d("stop_pres_head."), "adapter_grad": d("adapter."), "zero_control": zero}


# ------------------------------------------------------------------------------------------------ 1. the projection
def cell_of(p_rig_xy, z=1.5, cam_x=2.0, q=None):
    cam = torch.tensor([[*(q or NOMINAL_Q).values(), cam_x, 0.0, z]], dtype=torch.float64)
    c, v = S.stop_cell(torch.tensor([p_rig_xy], dtype=torch.float64), cam)
    return int(c[0]), bool(v[0])


@pytest.mark.parametrize("xy,expect", [
    ((12.0, 0.0), CELL(2, 4)),                           # straight ahead 10 m from the camera: column (W-1)/2 = 319.5 -> c 4; row 127.5 + 305.58 x 1.5 / 10 = 173.3 -> r 2
    ((2 + 17.3205, 10.0), CELL(2, 2)),                   # 30 deg LEFT at 20 m: column 319.5 - 305.58 x pi/6 = 159.5 -> c 2 (linear in azimuth); row 150.4 -> r 2
    ((2 + 17.3205, -10.0), CELL(2, 7)),                  # 30 deg RIGHT: column 479.5 -> c 7
    ((6.0, 0.0), CELL(3, 4)),                            # 4 m ahead of the camera: row 127.5 + 305.58 x 1.5 / 4 = 242.1 -> r 3 (the last row of cells)
    ((2 + 20 * math.cos(math.radians(55)), 20 * math.sin(math.radians(55))), CELL(2, 0)),   # 55 deg left: column 319.5 - 305.58 x 0.9599 = 26.2 -> c 0 (a PINHOLE column would fall outside the frame)
    ((2 + 20 * math.cos(math.radians(55)), -20 * math.sin(math.radians(55))), CELL(2, 9)),  # 55 deg right: column 612.8 -> c 9
])
def test_projection_cylindrical_hand_cases(xy, expect):
    assert cell_of(xy) == (expect, True)


@pytest.mark.parametrize("xy", [
    (-3.0, 0.0),                                          # behind the camera (and behind the rear axle)
    (-1.0, 5.0),                                          # behind the camera, to the side
    (2 + 20 * math.cos(math.radians(70)), 20 * math.sin(math.radians(70))),     # 70 deg: outside the 120 deg field (column -53.8)
    (3.0, 0.0),                                           # 1 m ahead of the camera: row 127.5 + 305.58 x 1.5 = 585.9, below the frame
    (float("nan"), 3.0),                                  # no stop
])
def test_projection_invalid_is_presence_zero(xy):
    assert cell_of(xy) == (-1, False)


def test_pinhole_formula_is_not_what_is_used():
    """TRAPS.md: the pinhole 2 atan((W/2)/f) reads 92.6 deg on this frame. A point at 55 deg left lies INSIDE the real 120 deg field (cylindrical c 0); under the pinhole it would land at column
    319.5 - 305.58 x tan(55 deg) = -117 (outside). The cell the module returns is the cylindrical one."""
    f, cx = S.STOPAUX_FRAME.f_ref, (S.STOPAUX_FRAME.width - 1) / 2
    assert cx - f * math.tan(math.radians(55)) < 0.0 < cx - f * math.radians(55)
    assert S.STOPAUX_FRAME.projection == "cylindrical" and abs(2 * (S.STOPAUX_FRAME.width / 2) / f - 2 * math.pi / 3) < 1e-9
    assert cell_of((2 + 20 * math.cos(math.radians(55)), 20 * math.sin(math.radians(55)))) == (CELL(2, 0), True)


def test_camera_height_moves_the_row():
    """The row comes from the clip's extrinsics: a stop 7 m ahead of the camera is in row 2 for a 1.2 m mount (row 127.5 + 305.58 x 1.2 / 7 = 179.9) and row 3 for 1.67 m (row 200.4)."""
    assert cell_of((9.0, 0.0), z=1.2) == (CELL(2, 4), True)
    assert cell_of((9.0, 0.0), z=1.67) == (CELL(3, 4), True)


def test_projection_matches_the_single_camera_implementation_and_the_dataset_rotation():
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera
    g = torch.Generator().manual_seed(4)
    for _ in range(12):
        q = torch.randn(4, generator=g, dtype=torch.float64)
        q = q / q.norm()
        e = FrontWideExtrinsics(qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]), x=1.8, y=0.02, z=1.35)
        R = S._rot_cam_to_vehicle(q[None])[0]
        assert float((R - torch.as_tensor(e.rotation_cam_to_vehicle())).abs().max()) < 1e-12
    q = torch.tensor([-0.4997, 0.5044, -0.4918, 0.5037], dtype=torch.float64)       # a real clip's mount (rig_extrinsics_train2400.json)
    e = FrontWideExtrinsics(qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]), x=1.70, y=0.018, z=1.43)
    cam = RigCamera.from_extrinsics(e, S.STOPAUX_FRAME)
    pts = torch.tensor([[x, y, 0.0] for x in (6.0, 9.0, 15.0, 30.0, 60.0) for y in (-9.0, -2.0, 0.0, 3.0, 11.0)], dtype=torch.float64)
    col, row, valid = cam.project(pts)
    cell, v = S.stop_cell(pts[:, :2], torch.tensor([[float(q[0]), float(q[1]), float(q[2]), float(q[3]), 1.70, 0.018, 1.43]], dtype=torch.float64).expand(len(pts), -1))
    assert torch.equal(v, valid)
    want = (row.clamp(0, 255) / 64).floor().long().clamp(0, 3) * 10 + (col.clamp(0, 639) / 64).floor().long().clamp(0, 9)
    assert torch.equal(cell[valid], want[valid]) and bool((cell[~valid] == -1).all())


# ------------------------------------------------------------------------------------------------ 2. the label masks
def test_label_masks(world):
    b, J = make_batch(world)
    b2, SA = with_stop(b, J)
    cell, pos, has = b2["stop_cell"], b2["stop_pos"], b2["stop_has"]
    assert bool(b2["lab"]["has"].all()) and int(pos.sum()) > 0 and bool((cell[pos] >= 0).all()) and bool((cell[~pos] == -1).all())
    xy = b2["lab"]["cons"][:, [M.CI["stop_x_m"], M.CI["stop_y_m"]]]
    nan = ~torch.isfinite(xy).all(-1)
    _, valid = S.stop_cell(xy.double(), torch.as_tensor(SA["cam"])[b["ci"].long()])
    assert torch.equal(pos, valid & ~nan)
    assert int(nan.sum()) > 0 and torch.equal(has, pos | nan)       # no rest point + a MOVING lon class = a negative: has, not pos
    out = ~nan & ~valid
    assert int(out.sum()) > 0 and not bool(has[out].any())          # MM reading (AMENDMENT_3): a rest point outside the frame is LEFT OUT of the presence BCE, not a negative
    for lc in (0, 1, D.IGNORE):                                      # ... and so is a window with no rest point at HOLD / CREEP / an unknown class
        lab = {**b2["lab"], "lon_cls": torch.full_like(b2["lab"]["lon_cls"], lc)}
        c_, p_, h_ = TR.stop_aux_stage({**b2, "lab": lab}, SA)
        assert torch.equal(p_, pos) and torch.equal(h_, pos) and torch.equal(c_, cell)
    tab = table(J["clip"])
    gone = str(J["clip"][int(b["ci"][0])])
    del tab[gone]                                                    # a clip without an extrinsics row: neither has nor pos
    _, pos3, has3 = TR.stop_aux_stage({**b2}, S.make_stop_aux_arrays(J, tab))
    on_gone = b["ci"] == int(b["ci"][0])
    assert bool((~has3[on_gone]).all()) and bool((~pos3[on_gone]).all()) and torch.equal(has3[~on_gone], has[~on_gone])
    nolab = {**b2, "lab": {**b2["lab"], "has": torch.zeros_like(b2["lab"]["has"])}}
    assert not bool(TR.stop_aux_stage(nolab, SA)[2].any())          # no v9 label row: not in the presence BCE either


# ------------------------------------------------------------------------------------------------ 3. OFF == BASE
def _grads(m, b):
    m.zero_grad(set_to_none=True)
    out = run_loss(m, b)
    out["terms"]["total"].backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in m.named_parameters()}, out


def test_off_equals_base_bit_for_bit(world):
    b, J = make_batch(world)
    B = base_module()
    m_base, m_off = build(J, mod=B), build(J, mod=M)
    assert [n for n, _ in m_base.named_parameters()] == [n for n, _ in m_off.named_parameters()]
    assert list(m_base.state_dict()) == list(m_off.state_dict())
    assert all(torch.equal(v, m_off.state_dict()[k]) for k, v in m_base.state_dict().items())
    assert m_off.stop_head is None and m_off.stop_pres_head is None
    g_base, o_base = _grads(m_base, b)
    g_off, o_off = _grads(m_off, b)
    assert set(o_base["terms"]) == set(o_off["terms"]) and not any(k.startswith("stop") for k in o_off["terms"])
    assert all(torch.equal(o_base["terms"][k].detach(), o_off["terms"][k].detach()) for k in o_base["terms"])
    assert all((g_base[n] is None and g_off[n] is None) or torch.equal(g_base[n], g_off[n]) for n in g_base)
    m_base.eval(), m_off.eval()
    with torch.no_grad():
        i0, i1 = m_base.forward_infer(b), m_off.forward_infer(b)
    assert set(i0) == set(i1) and all(torch.equal(i0[k], i1[k]) for k in i0 if torch.is_tensor(i0[k]))
    assert not any(k.startswith("stop") for k in m_off.built_levers()) and set(m_off.built_levers()) == set(m_base.built_levers())
    assert [g for g in m_off.trainable_groups()] == [g for g in m_base.trainable_groups()]


def test_off_checkpoint_cfg_keeps_the_base_key_set(tmp_path):
    """The NULL reuse rule compares the checkpoint's sha256 with BAL-NULL's: the pickled cfg must not gain keys when the lever is off."""
    base_fields = {f for f in base_module().V8FMiniConfig.__dataclass_fields__}
    cfg = TR.make_config(TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2"]))
    kept = {k for k in cfg.__dict__ if cfg.stop_aux == "on" or k not in M.STOPAUX_CFG_FIELDS}
    assert kept == base_fields and set(cfg.__dict__) - base_fields == set(M.STOPAUX_CFG_FIELDS)
    on = TR.make_config(TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--stop-aux", "on", "--w-stop-aux", "1"]))
    assert {k for k in on.__dict__ if on.stop_aux == "on" or k not in M.STOPAUX_CFG_FIELDS} == base_fields | set(M.STOPAUX_CFG_FIELDS)


# ------------------------------------------------------------------------------------------------ 4. ON adds exactly the declared keys
def test_on_adds_exactly_the_declared_keys_and_leaves_every_other_init_the_bases(world):
    _, J = make_batch(world)
    B = base_module()
    m_base, m_on = build(J, mod=B), build(J, mod=M, stop_aux="on", w_stop_aux=0.5)
    new = set(m_on.state_dict()) - set(m_base.state_dict())
    assert new == STOP_KEYS and set(m_base.state_dict()) <= set(m_on.state_dict())
    assert all(torch.equal(v, m_on.state_dict()[k]) for k, v in m_base.state_dict().items())          # built LAST: the RNG stream of every module above is the base's
    assert set(m_on.trainable_groups()) - set(m_base.trainable_groups()) == {"STOPAUX"}
    lv = m_on.built_levers()
    assert lv["stop_aux"] == "on" and lv["w_stop_aux"] == 0.5 and lv["stop_vision_only_source"] is True
    assert lv["params"]["STOPAUX"] == sum(p.numel() for n, p in m_on.named_parameters() if n.startswith(("stop_head.", "stop_pres_head.")))
    assert M.NEW_LEVER_KEY_PREFIXES[-2:] == M.STOPAUX_KEY_PREFIXES
    with pytest.raises(ValueError, match="4x10"):
        M.V8FMiniConfig(**{**TINY, "n_tok": 160, "stop_aux": "on", "w_stop_aux": 1.0})


def test_inference_never_reads_the_head(world):
    """Nothing consumes the head: forward_infer of an ON model equals the OFF model's with the head weights changed (a PURE auxiliary, Q4)."""
    b, J = make_batch(world)
    m = build(J, stop_aux="on", w_stop_aux=0.5).eval()
    with torch.no_grad():
        a = m.forward_infer(b)
        for p in list(m.stop_head.parameters()) + list(m.stop_pres_head.parameters()):
            p.add_(torch.randn_like(p))
        c = m.forward_infer(b)
    assert all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k]))
    assert not any(k.startswith("stop") for k in a)


# ------------------------------------------------------------------------------------------------ 5. G-LIVE (b) and the mutation (c)
def test_b_term_gradients_and_zero_controls(world):
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    m = build(J, stop_aux="on", w_stop_aux=0.5)
    r = check_b(m, b2)
    assert r["term"] > 0 and r["head_grad"] > 0 and r["pres_grad"] > 0 and r["adapter_grad"] > 0
    assert r["zero_control"] == 0.0                                     # at w = 0 NOTHING else reaches the head: EXACTLY 0.0
    t = run_loss(m, b2)["terms"]
    assert {"stop_cell", "stop_pres", "stop_aux"} <= set(t) and all(math.isfinite(float(t[k])) for k in ("stop_cell", "stop_pres", "stop_aux"))
    # the hand computation: stop_aux = mean_tokens BCE(cell) over the positive windows + BCE(presence) over the label-bearing windows
    out = m.forward(b2, p_gt=0.5, gen=torch.Generator().manual_seed(3))
    cl, pl = out["stop_cell_logits"].detach(), out["stop_pres_logit"].detach()
    pos, has = b2["stop_pos"], b2["stop_has"]
    cell_ref = np.mean([float(torch.nn.functional.binary_cross_entropy_with_logits(cl[i], torch.nn.functional.one_hot(b2["stop_cell"][i], 40).float())) for i in range(len(pos)) if pos[i]])
    pres_ref = np.mean([float(torch.nn.functional.binary_cross_entropy_with_logits(pl[i], torch.tensor(float(pos[i])))) for i in range(len(pos)) if has[i]])
    torch.manual_seed(3)
    t2 = m.loss(b2, p_gt=0.5, gen=torch.Generator().manual_seed(3))["terms"]
    assert abs(float(t2["stop_cell"]) - cell_ref) < 1e-5 and abs(float(t2["stop_pres"]) - pres_ref) < 1e-5


def test_masks_select_the_rows(world):
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    m = build(J, stop_aux="on", w_stop_aux=0.5)
    none = run_loss(m, {**b2, "stop_has": torch.zeros_like(b2["stop_has"])})["terms"]
    assert float(none["stop_aux"]) == 0.0 and float(none["total"]) == float(run_loss(m, b2)["terms"]["total"] - 0.5 * run_loss(m, b2)["terms"]["stop_aux"])
    nofr = run_loss(m, {**b2, "frame_ok": torch.zeros_like(b2["frame_ok"])})["terms"]
    assert float(nofr["stop_aux"]) == 0.0                                 # a NOW frame that is not a real frame carries no stop term
    with pytest.raises(ValueError, match="stop_cell"):
        m.loss({k: v for k, v in b.items()}, p_gt=0.5)                    # the loss refuses a batch with no stop labels (a deleted loop line is loud, not silent)


def test_c_mutation_that_deletes_the_term_from_the_total_goes_red(world):
    """(c) the single line that adds w x term to the total is replaced by `pass`: (b) must fail, and so must the trainer's own G-LIVE probe."""
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    check_b(build(J, stop_aux="on", w_stop_aux=0.5), b2)                  # GREEN on the real module
    Mm = mutated_module('total = total + cfg.w_stop_aux * T["stop_aux"]', "v8f_mini_mut_total")
    mm = build(J, mod=Mm, stop_aux="on", w_stop_aux=0.5)
    assert "stop_aux" in run_loss(mm, b2)["terms"]                        # the term is still computed and logged ...
    with pytest.raises(AssertionError, match="total does not move|no gradient"):
        check_b(mm, b2)                                                   # ... but it is not in the total: RED
    mm.cfg.w_stop_aux = 0.5                                               # check_b left the weight at 0
    with pytest.raises(AssertionError, match="does not reach the loss|receives no gradient"):
        TR.stop_aux_probe(mm, b2, mm.cfg)


def test_c_mutation_that_cuts_the_head_from_the_adapter_goes_red(world):
    """A head that reads a DETACHED token has a gradient of its own but shapes nothing: the adapter check goes RED."""
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    path = Path(M.__file__)
    src = path.read_text(encoding="utf-8")
    old = "        return self.stop_head(tok).squeeze(-1).float(), self.stop_pres_head(tok.mean(1)).squeeze(-1).float()"
    assert src.count(old) == 1
    mod = types.ModuleType("v8f_mini_mut_detach")
    mod.__file__ = str(path)
    sys.modules["v8f_mini_mut_detach"] = mod
    exec(compile(src.replace(old, old.replace("self.stop_head(tok)", "self.stop_head(tok.detach())").replace("tok.mean(1)", "tok.detach().mean(1)")), str(path), "exec"), mod.__dict__)
    with pytest.raises(AssertionError, match="adapter"):
        check_b(build(J, mod=mod, stop_aux="on", w_stop_aux=0.5), b2)


# ------------------------------------------------------------------------------------------------ 6. admissibility (d)
def test_d_vision_only_guard_passes_on_the_real_head(world):
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    m = build(J, stop_aux="on", w_stop_aux=0.5)
    assert M.stop_head_source_problems() == []
    rec = M.assert_stop_head_vision_only(m, b2)
    assert rec["keys_read"] == [] and rec["vision_sensitivity"] > 0.0 and rec["source_problems"] == []


def test_d_red_arms(world):
    b, J = make_batch(world)
    b2, _ = with_stop(b, J)
    with pytest.raises(AssertionError, match="no stop head"):
        M.assert_stop_head_vision_only(build(J), b2)

    class ReadsNav(M.V8FMini):                                              # static: the source names an ego / nav input
        def _stop(self, b, adapt, last):
            cl, pl = super()._stop(b, adapt, last)
            return cl + b["nav"].sum(), pl

    class ReadsViaKey(M.V8FMini):                                           # evades the static scan (a computed key); the RUNTIME key guard must catch it
        def _stop(self, b, adapt, last):
            cl, pl = super()._stop(b, adapt, last)
            k = "p10" + "f"
            return cl + 0.0 * b[k].sum(), pl

    class IgnoresVision(M.V8FMini):                                         # reads nothing, depends on nothing: the head must MOVE with the features
        def _stop(self, b, adapt, last):
            return torch.zeros(adapt.shape[0], adapt.shape[2]), torch.zeros(adapt.shape[0])

    class ShiftsOnEgo(M.V8FMini):                                           # reads only frame_ok by the guard's account, but moves when a non-vision key changes (via a stashed attribute)
        def _stop(self, b, adapt, last):
            cl, pl = super()._stop(b, adapt, last)
            self._seen = getattr(self, "_seen", 0) + 1
            return cl + 1e-3 * self._seen, pl

    with pytest.raises(AssertionError, match="non-vision inputs"):
        M.assert_stop_head_vision_only(build(J, cls=ReadsNav, stop_aux="on", w_stop_aux=0.5), b2)
    with pytest.raises(AssertionError, match="READS batch keys"):
        M.assert_stop_head_vision_only(build(J, cls=ReadsViaKey, stop_aux="on", w_stop_aux=0.5), b2)
    with pytest.raises(AssertionError, match="did not move"):
        M.assert_stop_head_vision_only(build(J, cls=IgnoresVision, stop_aux="on", w_stop_aux=0.5), b2)
    with pytest.raises(AssertionError, match="moved by"):
        M.assert_stop_head_vision_only(build(J, cls=ShiftsOnEgo, stop_aux="on", w_stop_aux=0.5), b2)


# ------------------------------------------------------------------------------------------------ 7. the launch gate, --init-ckpt, the W script
def parse(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


def gate(world, extra, info_stop=None, **over):
    _, J = make_batch(world)
    a = parse(extra)
    cfg = TR.make_config(a)
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **{k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    info = {"stop_aux": info_stop} if info_stop is not None else {}
    return TR.bad_stop_aux(a, model, info), a


def good_info(path, cov=1.0):
    return {"path": str(Path(path).resolve()), "sha256": "0" * 64, "n_train_clips": 100, "n_train_clips_with_extrinsics": int(100 * cov), "coverage_train": cov}


def test_gate_passes_the_declared_arm_and_the_default(world, tmp_path):
    ext = tmp_path / "ext.json"
    ext.write_text("{}")
    assert gate(world, [])[0] == []                                           # the default argv: nothing declared, nothing built
    bad, _ = gate(world, ["--stop-aux", "on", "--w-stop-aux", "1.0", "--stop-aux-extr", str(ext)], good_info(ext))
    assert bad == []


@pytest.mark.parametrize("extra,match", [
    (["--w-stop-aux", "1.0"], "a loss weight with no head"),
    (["--stop-aux-extr", "x.json"], "an input with no head"),
    (["--stop-aux", "on", "--stop-aux-extr", "x.json"], "a head with no loss"),
    (["--stop-aux", "on", "--w-stop-aux", "1.0"], "no extrinsics arrays"),
])
def test_gate_refuses(world, extra, match):
    bad, _ = gate(world, extra)
    assert any(match in x for x in bad), bad


def test_gate_refuses_thin_extrinsics_coverage_and_a_wrong_file(world, tmp_path):
    ext = tmp_path / "ext.json"
    ext.write_text("{}")
    argv = ["--stop-aux", "on", "--w-stop-aux", "1.0", "--stop-aux-extr", str(ext)]
    bad, _ = gate(world, argv, good_info(ext, cov=0.5))
    assert any("TRAIN clips have a sensor_extrinsics row" in x for x in bad), bad
    bad, _ = gate(world, argv, {**good_info(ext), "path": "/elsewhere.json"})
    assert any("declared" in x and "built from" in x for x in bad), bad
    bad, _ = gate(world, argv, {**good_info(ext), "sha256": None})
    assert any("sha256" in x for x in bad), bad
    with pytest.raises(ValueError, match="4x10"):
        TR.make_config(parse([*argv, "--grid", "pool8x20"]))


def test_every_new_flag_has_its_gdvb_entry():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert {"stop_aux", "w_stop_aux", "stop_aux_extr"} <= dests <= set(TR.FLAG_CHECKS)
    assert TR.FLAG_CHECKS["stop_aux"] == "lever:stop_aux" and TR.FLAG_CHECKS["w_stop_aux"] == "cfg:w_stop_aux"
    d = TR.build_parser()
    assert d.get_default("stop_aux") == "off" and d.get_default("w_stop_aux") == 0.0 and d.get_default("stop_aux_extr") is None


def test_init_ckpt_leaves_fresh_only_the_declared_keys(world, tmp_path):
    _, J = make_batch(world)
    off = build(J, wake=False)
    ck = tmp_path / "off.pt"
    torch.save({"model": off.state_dict(), "cfg": {k: v for k, v in off.cfg.__dict__.items() if k not in M.STOPAUX_CFG_FIELDS}}, ck)
    on = build(J, seed=7, wake=False, stop_aux="on", w_stop_aux=0.5)
    rec = TR.load_init_ckpt(str(ck), on, on.cfg)
    assert rec["n_mismatch_after_load"] == 0 and set(rec["fresh_keys"]) == STOP_KEYS == set(rec["declared_new_keys"]) & STOP_KEYS
    a = parse(["--stop-aux", "on", "--w-stop-aux", "1.0", "--init-ckpt", str(ck)])
    assert TR.bad_init_ckpt(a, {"init_ckpt": {**rec, "path": str(ck.resolve()), "sha256": "0" * 64}}) == []
    off_on = build(J, seed=7, wake=False)                                    # an OFF model loads the same checkpoint with the strict load, verbatim
    rec0 = TR.load_init_ckpt(str(ck), off_on, off_on.cfg)
    assert rec0["n_mismatch_after_load"] == 0 and "fresh_keys" not in rec0


def test_w_script_reads_the_smoke_log(tmp_path):
    import stop_aux_w_from_smoke as WS
    sc = {"stop_aux_probe": {"step0_goal_term": 0.693, "step0_stop_aux_term": 1.386, "w_stop_aux_for_goal_equal": 0.5, "pass": True}}
    assert WS.w_from_selfcheck(sc) == pytest.approx(0.5)
    scp = tmp_path / "selfcheck.json"
    scp.write_text(json.dumps(sc))
    logp = tmp_path / "train_log.jsonl"
    logp.write_text(json.dumps({"step": 1, "goal": 0.70, "stop_aux": 1.40}) + "\n")
    assert WS.main(["--selfcheck", str(scp), "--train-log", str(logp)]) == 0
    with pytest.raises(SystemExit):
        WS.w_from_selfcheck({"stop_aux_probe": {"step0_goal_term": 0.7, "step0_stop_aux_term": 0.0}})
    with pytest.raises(SystemExit):
        WS.w_from_selfcheck({})


# ------------------------------------------------------------------------------------------------ 8. end to end on the trainer's synthetic world
def _patched_world(monkeypatch, tmp_path, positives=True, cover=True):
    """make_synthetic_world + rest points in the join (the synthetic drive has none) + a nominal-camera extrinsics table for its clips."""
    orig = D.make_synthetic_world
    ext = tmp_path / "ext.json"

    def wrapped(root, **kw):
        w = orig(root, **kw)
        J = D.load_join(Path(root) / "join_train.npz")
        n = len(J["has"])
        if positives:
            xy = _stop_cons(n)
            J["cons"][:, M.CI["stop_x_m"]], J["cons"][:, M.CI["stop_y_m"]] = xy[:, 0].numpy(), xy[:, 1].numpy()
        D.save_join(J, str(Path(root) / "join_train.npz"))
        names = w["names"]["train"]
        ext.write_text(json.dumps(table(names if cover else names[:2])))
        return w

    monkeypatch.setattr(D, "make_synthetic_world", wrapped)
    return ext


def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def test_end_to_end_on_trains_with_the_term_and_writes_the_record(monkeypatch, tmp_path):
    ext = _patched_world(monkeypatch, tmp_path)
    out = tmp_path / "on"
    assert TR.main(["--out", str(out), *RUN, "--steps", "3", "--stop-aux", "on", "--w-stop-aux", "0.5", "--stop-aux-extr", str(ext)]) in (0, None)
    sc = _sc(out)
    assert sc["declared_vs_built"] == "PASS" and sc["levers"]["stop_aux"] == "on" and sc["levers"]["w_stop_aux"] == 0.5
    p = sc["stop_aux_probe"]
    assert p["pass"] and p["term"] > 0 and p["head_grad_diff"] > 0 and p["pres_head_grad_diff"] > 0 and p["adapter_grad_diff"] > 0 and p["head_grad_at_w0_zero_control"] == 0.0
    assert p["w_stop_aux_for_goal_equal"] == pytest.approx(p["step0_goal_term"] / p["step0_stop_aux_term"])
    assert sc["stop_aux_extrinsics"]["coverage_train"] == 1.0 and len(sc["stop_aux_extrinsics"]["sha256"]) == 64
    assert any(k.startswith("stop_head.") for k in sc["live"]["groups_grad"]) or sc["live"]["pass"]
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and all(math.isfinite(r["stop_aux"]) and r["stop_aux"] > 0 for r in log)                  # the term reaches the loop's loss at EVERY step
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert STOP_KEYS <= set(ck["model"]) and ck["cfg"]["stop_aux"] == "on" and ck["cfg"]["w_stop_aux"] == 0.5
    import stop_aux_w_from_smoke as WS
    assert WS.main(["--selfcheck", str(out / "selfcheck.json"), "--train-log", str(out / "train_log.jsonl")]) == 0      # the W script reads the real smoke record
    assert float(WS.w_from_selfcheck(sc)) == pytest.approx(p["w_stop_aux_for_goal_equal"])


def test_end_to_end_off_is_the_default_run(monkeypatch, tmp_path):
    _patched_world(monkeypatch, tmp_path)
    out = tmp_path / "off"
    assert TR.main(["--out", str(out), *RUN, "--steps", "3"]) in (0, None)
    sc = _sc(out)
    assert "stop_aux_probe" not in sc and "stop_aux" not in sc["levers"] and "stop_aux_extrinsics" not in sc
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and not any(k.startswith("stop") for k in log[0])
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert not any(k.startswith(M.STOPAUX_KEY_PREFIXES) for k in ck["model"]) and not (set(M.STOPAUX_CFG_FIELDS) & set(ck["cfg"]))


def test_end_to_end_refuses_thin_coverage_and_a_probe_with_no_positive(monkeypatch, tmp_path):
    ext = _patched_world(monkeypatch, tmp_path, cover=False)
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "thin"), *RUN, "--steps", "3", "--stop-aux", "on", "--w-stop-aux", "0.5", "--stop-aux-extr", str(ext)])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "sensor_extrinsics row" in str(e.value)
    monkeypatch.undo()
    ext2 = _patched_world(monkeypatch, tmp_path / "np", positives=False) if (tmp_path / "np").mkdir() is None else None
    with pytest.raises((SystemExit, AssertionError)) as e2:
        TR.main(["--out", str(tmp_path / "nopos"), *RUN, "--steps", "3", "--stop-aux", "on", "--w-stop-aux", "0.5", "--stop-aux-extr", str(ext2)])
    assert "in-view" in str(e2.value) or "dead" in str(e2.value) or "G-LIVE" in str(e2.value), str(e2.value)[:300]


def test_the_loop_calls_the_stage_before_the_ego_stages():
    src = Path(TR.__file__).read_text(encoding="utf-8")
    i_stage, i_ego, i_attach, i_dev = (src.index("stop_lab = stop_aux_stage(b, SA)"), src.index("b = ego.apply(b, step)", src.index("stop_lab = stop_aux_stage(b, SA)")),
                                       src.index('b["stop_cell"], b["stop_pos"], b["stop_has"] = stop_lab'), src.index("b = to_dev(b, dev)", src.index("stop_lab = stop_aux_stage(b, SA)")))
    assert i_stage < i_ego < i_attach < i_dev
