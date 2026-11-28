"""refcv9 T1 -- ``--r9-bev-warp`` (the temporal BEV warp; REFCV9_DESIGN 4.3, D3).  CPU.  Pure-tensor controls first, then a tiny REAL rig model.

The class this file exists for is "built, tested, and UNREACHABLE FROM ITS CALLER" (CLAUDE.md 2026-09-10/11), so it traces the flag to the
loss and to the parameters, and every control has a RED arm that must FAIL:

  1. the ANALYTIC warp control (a marker at (x, y), ego moved by a known (dx, dy, dyaw) lands at the predicted cell; tolerance stated), zero motion =
     identity BIT-EXACT, cells that leave the grid are exactly 0 with mask 0;  RED: a sign-flipped yaw;
  2. the fuse: gate 0 = the input bit for bit, a non-zero gradient on the gate with the fuse's gradient EXACTLY 0.0 (control), no history = unchanged
     for any gate;  RED: a gate that is not zero-initialised;
  3. ADMISSIBILITY: perturbing any FUTURE pose / frame leaves the history bit-identical (exact 0.0), at the dataset side AND at the batch side; an AST
     audit of what ``item_fields`` / ``stage`` read;  RED: a warp that reads the NEXT row's pose, a ``stage`` that reads ``future_poses``;
  4. the SHUFFLED control (fixed derangement, seed 0): frames from another episode, the item's own poses;
  5. a tiny REAL model: gate 0 => the OFF forward bit for bit, a history-less forward REFUSES, a non-zero gate reads the history, the flag reaches the
     loss and a non-zero gradient on the gate (and the fuse, once the gate has moved), the EMA copy carries its own hook, checkpoint + warm start;
  6. argv, refusals, G-DVB and the trainer's call sites.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses as dc
import inspect
import math
import sys
import textwrap
import types
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
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.models import bev_warp as BW  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402

_ORIG_MODEL = v3.RefCV3Model
XM, YH, CELL = 20.0, 6.0, 0.25
NX, NY = int(XM / CELL), int(2 * YH / CELL)          # 80 x 48


# =========================================================================== #
# 1. the warp: ANALYTIC control, identity, leaving the grid                    #
# =========================================================================== #
def _blob(qx, qy, sigma=1.5):
    """A Gaussian marker at metric (qx, qy) in the PAST ego grid -> [1, 1, NX, NY]."""
    ix = (qx / CELL) - 0.5
    iy = ((qy + YH) / CELL) - 0.5
    xs = torch.arange(NX, dtype=torch.float64).view(-1, 1)
    ys = torch.arange(NY, dtype=torch.float64).view(1, -1)
    return torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2)).float().view(1, 1, NX, NY)


def _predicted_cell(q, pose_now):
    """INDEPENDENT of bev_warp: the past ego frame IS the world frame (pose_past = identity); the marker is fixed in the world, so its NOW-frame
    coordinate is R(-yaw_now) (q - t_now); returned in continuous cell units (centre of cell (i, j) = (i, j))."""
    px, py, ya = pose_now
    ux, uy = q[0] - px, q[1] - py
    x = math.cos(ya) * ux + math.sin(ya) * uy
    y = -math.sin(ya) * ux + math.cos(ya) * uy
    return x / CELL - 0.5, (y + YH) / CELL - 0.5


def _centroid(t):
    w = t[0, 0].double()
    xs = torch.arange(NX, dtype=torch.float64).view(-1, 1)
    ys = torch.arange(NY, dtype=torch.float64).view(1, -1)
    s = w.sum()
    return float((w * xs).sum() / s), float((w * ys).sum() / s)


#: (marker in the past grid, NOW pose (x, y, yaw) in the world == past ego frame): forward / lateral / yaw / combined, both yaw signs.
ANALYTIC_CASES = (((10.0, 1.0), (2.5, 0.0, 0.0)), ((10.0, 1.0), (0.0, -1.0, 0.0)), ((10.0, 1.0), (1.0, 0.5, 0.2)),
                  ((12.0, -1.5), (2.0, -0.25, -0.3)))
CENTROID_TOL_CELLS = 0.10        # the stated tolerance: an intensity-weighted centroid of a sigma-1.5-cell blob under one bilinear resample


def _analytic_problems(rel=BW.relative_se2) -> list:
    out = []
    for q, pn in ANALYTIC_CASES:
        pose_now = torch.tensor([[pn[0], pn[1], pn[2], 5.0]])
        pose_past = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
        warped, mask = BW.warp_bev(_blob(*q), None, rel(pose_now, pose_past), XM, YH)
        got, want = _centroid(warped), _predicted_cell(q, pn)
        err = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
        if err > CENTROID_TOL_CELLS or float(mask[0, 0, round(want[0]), round(want[1])]) != 1.0:      # the marker's own cell is an observed cell
            out.append((q, pn, round(err, 3)))
    return out


def test_ANALYTIC_marker_lands_at_the_predicted_cell():
    assert _analytic_problems() == []


def test_ANALYTIC_integer_shift_is_exact():
    """Ego 2.5 m forward = 10 cells: the hot cell (39, 27) of the past grid is cell (29, 27) of the NOW grid (argmax exact, value within 1e-4)."""
    f = torch.zeros(1, 1, NX, NY)
    f[0, 0, 39, 27] = 1.0
    w, m = BW.warp_bev(f, None, BW.relative_se2(torch.tensor([[2.5, 0, 0, 5.0]]), torch.tensor([[0.0, 0, 0, 5.0]])), XM, YH)
    assert tuple(divmod(int(w.flatten().argmax()), NY)) == (29, 27) and abs(float(w[0, 0, 29, 27]) - 1.0) < 1e-4
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-3)


def test_RED_a_sign_flipped_yaw_turns_the_analytic_control_red():
    flipped = lambda a, b: BW.relative_se2(a, b) * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    bad = _analytic_problems(flipped)
    assert bad, "a sign-flipped yaw must FAIL the analytic control"
    assert {c[1] for c in bad} <= {c[1] for c in ANALYTIC_CASES if c[1][2] != 0.0}      # only the yaw cases move; the pure translations still pass


def test_RED_a_flipped_lateral_axis_turns_the_analytic_control_red():
    assert _analytic_problems(lambda a, b: BW.relative_se2(a, b) * torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))


def test_zero_motion_is_the_identity_bit_exact():
    g = torch.Generator().manual_seed(1)
    f = torch.randn(3, 5, NX, NY, generator=g)
    w, m = BW.warp_bev(f, None, torch.zeros(3, 3, dtype=torch.float64), XM, YH)
    assert torch.equal(w, f) and torch.equal(w.view(torch.int32), f.view(torch.int32)) and float(m.min()) == 1.0
    sv = torch.rand(3, NX, NY, generator=g) > 0.3
    w2, m2 = BW.warp_bev(f, sv, torch.zeros(3, 3, dtype=torch.float64), XM, YH)
    assert torch.equal(w2, f * sv.unsqueeze(1)) and torch.equal(m2[:, 0].bool(), sv)


def test_cells_that_leave_the_grid_are_exactly_zero_with_mask_zero():
    """Past origin 5 m AHEAD of the NOW origin (ego reversed 5 m): NOW rows 0..19 map to x < 0 in the past grid."""
    f = torch.ones(1, 2, NX, NY)
    T_ = BW.relative_se2(torch.tensor([[0.0, 0, 0, 1.0]]), torch.tensor([[5.0, 0, 0, 1.0]]))
    w, m = BW.warp_bev(f, None, T_, XM, YH)
    assert float(w[:, :, :20].abs().max()) == 0.0 and float(m[:, :, :20].max()) == 0.0 and float(m[:, :, 20:].min()) == 1.0


def test_cells_the_past_frame_did_not_observe_are_zero_with_mask_zero():
    f = torch.ones(1, 1, NX, NY)
    sv = torch.ones(1, NX, NY, dtype=torch.bool)
    sv[:, :, :24] = False                                             # the past frame saw only y >= 0
    w, m = BW.warp_bev(f, sv, BW.relative_se2(torch.tensor([[0.0, 0, 0, 1.0]]), torch.tensor([[0.0, 0, 0, 1.0]])), XM, YH)   # T == 0 exactly
    assert float(w[:, :, :, :24].abs().max()) == 0.0 and float(m[:, :, :, :24].max()) == 0.0 and float(m[:, :, :, 24:].min()) == 1.0
    w, m = BW.warp_bev(f, sv, BW.relative_se2(torch.tensor([[1.0, 0, 0, 1.0]]), torch.tensor([[0.0, 0, 0, 1.0]])), XM, YH)   # bilinear path
    assert float(w[:, :, :76, :23].abs().max()) == 0.0 and float(m[:, :, :76, :23].max()) == 0.0 and float(m[:, :, :76, 25:].min()) == 1.0     # rows 76..79 left the far edge


def test_lags_to_rows_and_the_derangement():
    assert BW.lags_to_rows((0.5, 1.0)) == (5, 10) == BW.lags_to_rows(BW.LAGS_S_DEFAULT)
    for bad in ((0.55,), (1.0, 0.5), (0.0,), (), (0.5, 0.5)):
        with pytest.raises(ValueError):
            BW.lags_to_rows(bad)
    for n in (2, 3, 5, 40, 139):
        for seed in (0, 1, 2):
            p = BW.derangement(n, seed)
            assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n)) and p == BW.derangement(n, seed)
    with pytest.raises(ValueError):
        BW.derangement(1)


# =========================================================================== #
# 2. the fuse                                                                  #
# =========================================================================== #
def _fuse(c=4, p=2, seed=0):
    torch.manual_seed(seed)
    return BW.BevWarpFuse(c, p, XM, YH, (5, 10))


def _identity_problems(fz) -> list:
    g = torch.Generator().manual_seed(3)
    bev = torch.randn(2, fz.d_bev, 8, 6, generator=g)
    w = [torch.randn(2, fz.d_bev, 8, 6, generator=g) for _ in range(fz.n_past)]
    m = [(torch.rand(2, 1, 8, 6, generator=g) > 0.5).float() for _ in range(fz.n_past)]
    out = fz.mix(bev, w, m)
    return [] if torch.equal(out.view(torch.int32), bev.view(torch.int32)) else ["gate-0 output differs from the input"]


def test_gate_zero_is_the_identity_bit_for_bit():
    fz = _fuse()
    assert float(fz.gate.detach().abs().max()) == 0.0 and _identity_problems(fz) == []


def test_RED_a_gate_that_is_not_zero_initialised_turns_the_identity_red():
    fz = _fuse()
    torch.nn.init.constant_(fz.gate, 0.5)
    assert _identity_problems(fz), "a non-zero gate must FAIL the identity control"


def test_no_history_is_unchanged_for_any_gate():
    fz = _fuse()
    fz.gate.data.fill_(0.7)
    bev = torch.randn(2, 4, 8, 6)
    z = [torch.zeros(2, 4, 8, 6)] * 2
    zm = [torch.zeros(2, 1, 8, 6)] * 2
    assert torch.equal(fz.mix(bev, z, zm), bev)               # bias-free: fuse(0) == 0 exactly


def test_gradients_gate_non_zero_fuse_exactly_zero_at_gate_zero_then_wakes():
    fz = _fuse()
    g = torch.Generator().manual_seed(5)
    bev = torch.randn(2, 4, 8, 6, generator=g, requires_grad=True)
    w = [torch.randn(2, 4, 8, 6, generator=g) for _ in range(2)]
    m = [torch.ones(2, 1, 8, 6) for _ in range(2)]
    fz.mix(bev, w, m).pow(2).sum().backward()
    assert float(fz.gate.grad.abs().sum()) > 0.0                       # the gate learns first
    assert torch.count_nonzero(fz.fuse.weight.grad) == 0               # the CONTROL: exactly 0.0 until the gate has moved
    fz.zero_grad()
    fz.gate.data.fill_(0.1)
    fz.mix(bev, w, m).pow(2).sum().backward()
    assert torch.count_nonzero(fz.fuse.weight.grad) > 0 and float(fz.gate.grad.abs().sum()) > 0.0


# =========================================================================== #
# 3. ADMISSIBILITY: the history reads the PAST only                            #
# =========================================================================== #
def _ep(n=60, seed=0, h=8, w=8):
    g = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(frames=(torch.rand(n, 9, h, w, generator=g) * 255).to(torch.uint8),
                                 poses=torch.cat([torch.randn(n, 3, generator=g), torch.rand(n, 1, generator=g)], 1), episode_id=seed)


def _perturb_future(eps, e_i, row):
    out = copy.deepcopy(eps)
    ep = out[e_i]
    gen = torch.Generator().manual_seed(99)
    ep.poses[row + 1:] = torch.randn(ep.poses[row + 1:].shape, generator=gen) * 100.0
    ep.frames[row + 1:] = (torch.rand(ep.frames[row + 1:].shape, generator=gen) * 255).to(torch.uint8)
    return out


def _future_problems(eps, e_i=0, row=30, src=None) -> list:
    src = src or BF.ItemSource((5, 10), False, len(eps))
    a = BF.item_fields(eps, e_i, row, src)
    b = BF.item_fields(_perturb_future(eps, e_i, row), e_i, row, src)
    return [k for k in a if float((a[k].float() - b[k].float()).abs().max()) != 0.0]


def test_item_fields_reads_the_PAST_rows_only_exact_zero_change_under_a_future_perturbation():
    eps = [_ep(seed=0), _ep(seed=1)]
    assert _future_problems(eps) == []
    it = BF.item_fields(eps, 0, 30, BF.ItemSource((5, 10), False, 2))
    assert torch.equal(it["bw_pose_now"], eps[0].poses[30]) and torch.equal(it["bw_pose_past"][0], eps[0].poses[25]) \
        and torch.equal(it["bw_pose_past"][1], eps[0].poses[20])
    assert torch.equal(it["bw_frames"][0], eps[0].frames[25][-3:]) and torch.equal(it["bw_frames"][1], eps[0].frames[20][-3:]) and bool(it["bw_has"].all())


def test_RED_a_warp_that_reads_the_NEXT_rows_pose_turns_the_future_test_red(monkeypatch):
    monkeypatch.setattr(BF, "_pose_row", lambda row, back=0: row - back + 1)
    assert _future_problems([_ep(seed=0), _ep(seed=1)]), "reading row + 1 must FAIL the future-pose test"


def test_clip_start_has_false_zero_frames_and_the_now_pose():
    eps = [_ep(seed=0), _ep(seed=1)]
    it = BF.item_fields(eps, 0, 7, BF.ItemSource((5, 10), False, 2))
    assert it["bw_has"].tolist() == [True, False] and int(it["bw_frames"][1].abs().sum()) == 0 and torch.equal(it["bw_pose_past"][1], eps[0].poses[7])


def test_AST_audit_stage_and_item_fields_read_only_past_things():
    src = inspect.getsource(BF)
    assert _audit(src) == []
    leaked = src.replace('pn = batch["bw_pose_now"].to(device)[:, None, :3]', 'pn = batch["future_poses"].to(device)[:, 0:1, :3]')
    assert leaked != src and _audit(leaked), "an audit that cannot see a future-pose read in stage() is not an audit"
    leaked2 = src.replace("pose_now = ep.poses[_pose_row(row)][:4].float()", "pose_now = ep.future_poses[0][:4].float()")
    assert leaked2 != src and _audit(leaked2)


def _audit(src: str) -> list:
    """Every ``batch[...]`` subscript in ``stage`` is a BATCH_KEY; no identifier / attribute / string in ``stage`` or ``item_fields`` names a future,
    a label, a goal, a nav or a situation-classifier quantity.  Docstrings and comments are not code."""
    tree = ast.parse(src)
    bad = []
    banned = ("future", "goal", "nav", "lat_v7", "lon_v7", "tac_", "maneuver", "label", "cls", "situation")
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("stage", "item_fields", "_relative_T")]:   # T1v: the pose read moved into the shared helper _relative_T (stage's T1 and T1v branches both call it)
        doc = ast.get_docstring(fn, clean=False)
        for node in ast.walk(fn):
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != doc:
                names = [node.value]
            for nm in names:
                if any(b in nm.lower() for b in banned) and nm not in BF.BATCH_KEYS:
                    bad.append((fn.name, nm))
            if fn.name in ("stage", "_relative_T") and isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "batch" \
                    and isinstance(node.slice, ast.Constant) and node.slice.value not in BF.BATCH_KEYS:
                bad.append((fn.name, node.slice.value))
    return bad


# =========================================================================== #
# 4. the SHUFFLED control                                                      #
# =========================================================================== #
def test_shuffled_control_takes_frames_from_another_episode_and_keeps_the_own_poses():
    eps = [_ep(seed=s) for s in range(4)]
    perm = BW.derangement(4, 0)
    own = BF.item_fields(eps, 1, 30, BF.ItemSource((5, 10), False, 4))
    shuf = BF.item_fields(eps, 1, 30, BF.ItemSource((5, 10), True, 4))
    d = eps[perm[1]]
    assert perm[1] != 1 and torch.equal(shuf["bw_frames"][0], d.frames[25][-3:]) and torch.equal(shuf["bw_frames"][1], d.frames[20][-3:])
    assert not torch.equal(shuf["bw_frames"], own["bw_frames"])
    for k in ("bw_pose_now", "bw_pose_past", "bw_has"):
        assert torch.equal(shuf[k], own[k]), k
    eps2 = copy.deepcopy(eps)
    eps2[perm[1]].frames[31:] = 0
    eps2[perm[1]].poses[31:] = 1e6
    again = BF.item_fields(eps2, 1, 30, BF.ItemSource((5, 10), True, 4))
    assert all(torch.equal(again[k], shuf[k]) for k in shuf)


# =========================================================================== #
# 5. a tiny REAL rig model                                                     #
# =========================================================================== #
def _args(**over):
    a = argparse.Namespace(r9_bev_warp=True, r9_bev_warp_lags="0.5,1.0", r9_bev_warp_shuffle=False, r9_perc_rig=True, map_hires="on",
                           trunk="timm", trunk_mode="shared", r8_train_only_map=False, r9_train_only=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _rig(monkeypatch, on, **over):
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: _ORIG_MODEL(cfg, perc_rig=True))      # the class as imported (a 2nd call must not nest the lambda)
    model, cfg = R._build()
    args = _args(**over)
    if on:
        BF.pin_args(args)
        BF.attach(model, args)
    return model, cfg, args


def _batch_t1(cfg, model, args, idxs=(12, 14), attach=True):
    """R._batch's body, but on the dataset the flag is attached to, at windows whose NOW row has both lags (t >= 3)."""
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    if attach:
        BF.attach_dataset(ds, model, args)
    batch = torch.utils.data.default_collate([ds[i] for i in idxs])
    v7l = T.v7l
    n = len(idxs)
    batch["lat_v7"] = torch.tensor([0] + [v7l.IGNORE_ID] * (n - 1), dtype=torch.long)
    batch["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1] + [v7l.IGNORE_ID] * (n - 1), dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2][:n], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True] * n)
    batch["tac_goal_y"] = torch.zeros(n, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(n, len(v7l.TAC_GOAL_TOKENS))
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 8, (n, 600, 320), generator=g, dtype=torch.uint8)
    codes[:, :40] = 255
    batch.update({"map_fine": codes, "map_fine_label": torch.tensor([True] * n),
                  "map_ep": torch.tensor([int(stable_episode_id(R.CLIP))] * n)})
    return batch, ds


def _mh(model, batch):
    """One forward of the 10 cm branch (the lift hook fires inside it) on the batch's NEWEST row."""
    model.eval()
    frames = T.frames_to_device(batch["frames"][:, -1:], "cpu")
    with torch.no_grad():
        s8, *_ = model.core.encoder.forward_features_s8(frames[:, -1], torch.arange(frames.shape[0]))
        grid, valid = model._lift_bank_hires.for_episodes(batch["map_ep"], device="cpu")
        return model._map_hires(s8, grid, valid)


def _equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(*(list(_leaves(a)), list(_leaves(b))))) and len(list(_leaves(a))) == len(list(_leaves(b)))


def _leaves(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from _leaves(x[k])
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v)


def test_gate_zero_forward_is_the_OFF_forward_bit_for_bit(monkeypatch):
    off, cfg, _ = _rig(monkeypatch, on=False)
    on, _, args = _rig(monkeypatch, on=True)
    assert all(torch.equal(off.state_dict()[k], v) for k, v in on.state_dict().items() if not k.startswith(BF.KEY_PREFIX))     # the fork_rng: no shared init moved
    batch, _ = _batch_t1(cfg, on, args)
    BF.stage(off, batch, "cpu")                                         # OFF: a no-op (no fuse built)
    ref = _mh(off, batch)
    BF.stage(on, batch, "cpu")
    out = _mh(on, batch)
    assert float(on._map_hires.bev_warp.gate.abs().max()) == 0.0 and _equal(ref, out)


def test_a_history_less_forward_REFUSES(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    batch, _ = _batch_t1(cfg, on, args)
    with pytest.raises(RuntimeError, match="no history was staged"):
        _mh(on, batch)
    batch2, _ = _batch_t1(cfg, on, args, attach=False)                  # a dataset the flag was never attached to
    with pytest.raises(SystemExit, match="carries no"):
        BF.stage(on, batch2, "cpu")


def _open_gate(model, v=0.3):
    fz = model._map_hires.bev_warp
    fz.gate.data.fill_(v)
    return fz


def test_a_non_zero_gate_reads_the_history_and_ONLY_the_history(monkeypatch):
    off, cfg, _ = _rig(monkeypatch, on=False)
    on, _, args = _rig(monkeypatch, on=True)
    _open_gate(on)
    batch, _ = _batch_t1(cfg, on, args)
    ref = _mh(off, batch)
    BF.stage(on, batch, "cpu")
    a = _mh(on, batch)
    assert not _equal(ref, a)                                                                      # the history moved the BEV
    b2 = dict(batch)
    b2["bw_frames"] = (255 - batch["bw_frames"].to(torch.int32)).to(torch.uint8)                   # other CONTENT, same poses
    BF.stage(on, b2, "cpu")
    assert not _equal(a, _mh(on, b2))                                                              # the content is read
    b3 = dict(batch)
    b3["bw_has"] = torch.zeros_like(batch["bw_has"])
    BF.stage(on, b3, "cpu")
    assert _equal(ref, _mh(on, b3))                                                                # no history == the OFF forward EXACTLY, any gate


def _future_batch_problems(model, batch, stage_fn) -> list:
    """The ADMISSIBILITY check at the batch side: perturb EVERY non-history key a future / label / goal could ride in; the T1 forward must not move."""
    stage_fn(model, batch, "cpu")
    a = _mh(model, batch)
    pert = dict(batch)
    n = 0
    for k, v in batch.items():
        if k.startswith(("future_", "goal_", "pose_", "actions", "tac_", "nav", "lat_", "lon_")) and torch.is_tensor(v) and v.dtype.is_floating_point:
            pert[k] = v + 123.0
            n += 1
    assert n >= 4, "the batch carries too few future / label keys to make this test mean something"
    stage_fn(model, pert, "cpu")
    return [] if _equal(a, _mh(model, pert)) else ["the forward moved under a future / label perturbation"]


def test_future_and_label_perturbation_leaves_the_forward_bit_identical_and_the_RED_arm_fails(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    _open_gate(on)
    batch, _ = _batch_t1(cfg, on, args)
    assert _future_batch_problems(on, batch, BF.stage) == []
    real = BF.stage

    def reads_future(model, b, device):
        b = dict(b)
        b["bw_pose_now"] = b["bw_pose_now"] + 0.01 * b["future_poses"][:, 0, :]          # the RED arm: a stage that mixes the first FUTURE pose in
        return real(model, b, device)
    assert _future_batch_problems(on, batch, reads_future), "a stage that reads future_poses must FAIL the batch-side test"


def test_the_flag_reaches_the_loss_and_the_gradient_controls(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    batch, _ = _batch_t1(cfg, on, args)
    fz = on._map_hires.bev_warp

    def run(b):
        on.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        on.train()
        out = T.compute_losses_v3(on, b, "cpu", log_metrics=False)
        out["loss"].backward()
        return out

    out = run(batch)                                                                   # gate 0: the gate learns, the fuse is exactly 0.0
    assert torch.isfinite(out["loss"]) and float(fz.gate.grad.abs().sum()) > 0.0 and torch.count_nonzero(fz.fuse.weight.grad) == 0
    _open_gate(on, 0.05)
    run(batch)
    assert torch.count_nonzero(fz.fuse.weight.grad) > 0 and float(fz.gate.grad.abs().sum()) > 0.0
    nohist = dict(batch)
    nohist["bw_has"] = torch.zeros_like(batch["bw_has"])
    run(nohist)                                                                        # the CONTROL: no history => the fuse's gradient reads EXACTLY 0.0
    assert torch.count_nonzero(fz.fuse.weight.grad) == 0 and float(fz.gate.grad.abs().sum()) == 0.0
    lp = [p for n, p in on.named_parameters() if n.startswith("_map_hires.lift.")]
    run(batch)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in lp)       # the CURRENT frame's lift still trains (the past path is detached)


def test_the_loss_at_gate_zero_equals_the_OFF_loss_exactly(monkeypatch):
    off, cfg, _ = _rig(monkeypatch, on=False)
    on, _, args = _rig(monkeypatch, on=True)
    batch, _ = _batch_t1(cfg, on, args)
    res = []
    for m in (off, on):
        torch.manual_seed(5)
        m.train()
        res.append(T.compute_losses_v3(m, batch, "cpu", log_metrics=False))
    assert float(res[0]["loss"]) == float(res[1]["loss"]) and float(res[0]["map_hires"]) == float(res[1]["map_hires"])


def test_the_shuffled_control_changes_the_forward_but_not_the_poses(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    sh, _, sargs = _rig(monkeypatch, on=True, r9_bev_warp_shuffle=True)
    for m in (on, sh):
        _open_gate(m)
        m._map_hires.bev_warp.fuse.weight.data.copy_(on._map_hires.bev_warp.fuse.weight.data)
    b_true, _ = _batch_t1(cfg, on, args)
    b_shuf, _ = _batch_t1(cfg, sh, sargs)
    assert torch.equal(b_true["bw_pose_past"], b_shuf["bw_pose_past"]) and not torch.equal(b_true["bw_frames"], b_shuf["bw_frames"])
    BF.stage(on, b_true, "cpu")
    BF.stage(sh, b_shuf, "cpu")
    assert not _equal(_mh(on, b_true), _mh(sh, b_shuf))


def test_the_EMA_copy_carries_its_own_hook_and_the_checkpoint_round_trips(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    _open_gate(on, 0.2)
    cp = copy.deepcopy(on)
    hooks = list(cp._map_hires.lift._forward_hooks.values())
    assert len(hooks) == 1 and hooks[0].__self__ is cp._map_hires.bev_warp and hooks[0].__self__ is not on._map_hires.bev_warp
    sd = on.state_dict()
    assert {k for k in sd if k.startswith(BF.KEY_PREFIX)} == {BF.KEY_PREFIX + "gate", BF.KEY_PREFIX + "fuse.weight"}
    fresh, _, a2 = _rig(monkeypatch, on=True)
    fresh.load_state_dict(sd, strict=True)
    assert float(fresh._map_hires.bev_warp.gate[0]) == pytest.approx(0.2)
    off, _, _ = _rig(monkeypatch, on=False)
    with pytest.raises(RuntimeError):                       # a T1 checkpoint does not load into an OFF build: G-EVAL must replay the flag
        off.load_state_dict(sd, strict=True)


def test_the_rig_warm_start_names_the_fuse_and_an_unnamed_missing_key_still_refuses(monkeypatch, tmp_path):
    off, _, _ = _rig(monkeypatch, on=False)
    src = {k: v.clone() for k, v in off.state_dict().items()}
    path = tmp_path / "src.pt"
    torch.save({"model": src, "step": 7}, str(path))
    on, _, _ = _rig(monkeypatch, on=True)
    assert rig.bev_warp_exemptions(on) == (BF.KEY_PREFIX,) and rig.bev_warp_exemptions(off) == ()
    rig.warm_start_perc_rig(on, str(path))
    assert float(on._map_hires.bev_warp.gate.abs().max()) == 0.0
    bad = dict(src)
    bad.pop(next(k for k in bad if k.startswith("_map_hires.encoder.")))
    torch.save({"model": bad, "step": 7}, str(path))
    with pytest.raises(SystemExit, match="rig key"):
        rig.warm_start_perc_rig(on, str(path))
    on2, _, _ = _rig(monkeypatch, on=True)
    torch.save({"model": src, "step": 7}, str(path))
    rig.warm_start_perc_rig(off, str(path))                 # OFF: the tip's warm start, unchanged


# =========================================================================== #
# 6. argv, refusals, G-DVB, the trainer's call sites                           #
# =========================================================================== #
def test_argv_defaults_and_the_registry():
    p = T.build_parser()
    d = {a.dest: a.default for a in p._actions}
    assert d["r9_bev_warp"] is False and d["r9_bev_warp_lags"] == "0.5,1.0" and d["r9_bev_warp_shuffle"] is False
    for dest in ("r9_bev_warp", "r9_bev_warp_lags", "r9_bev_warp_shuffle"):
        assert D.REGISTRY[dest].kind == "built"


def test_pin_args_refusals():
    BF.pin_args(argparse.Namespace(r9_bev_warp=False, r9_bev_warp_lags="0.5,1.0", r9_bev_warp_shuffle=False))          # OFF + defaults: fine
    BF.pin_args(_args())
    BF.pin_args(_args(r9_perc_rig=False))                                   # T1-FULL: the rig is NOT required (the full model's loss stages the history too);
    #                                                                         tests/test_bev_warp_full_model.py proves the full model's path
    for over, match in ((dict(r9_bev_warp_lags="0.5,2.0", r9_bev_warp=False), "dead flag"), (dict(r9_bev_warp_shuffle=True, r9_bev_warp=False), "dead flag"),
                        (dict(map_hires="off"), "map-hires"), (dict(r9_bev_warp_lags="0.55"), "whole number"),
                        (dict(r9_bev_warp_lags="1.0,0.5"), "increasing"), (dict(trunk_mode="inflate"), "trunk-mode"),
                        (dict(r8_train_only_map=True), "freeze"), (dict(ablate_frames=True), "ablate-frames"),
                        (dict(r9_perc_rig=False, ablate_frames=True), "ablate-frames")):
        with pytest.raises(SystemExit, match=match):
            BF.pin_args(_args(**over))


def test_gdvb_catches_an_argv_that_disagrees_with_the_build(monkeypatch):
    on, cfg, args = _rig(monkeypatch, on=True)
    assert BF.dvb_check(on, args) == [] and BF.dvb_check_lags(on, args) == [] and BF.dvb_check_shuffle(on, args) == []
    assert BF.dvb_check(on, _args(r9_bev_warp=False))                                              # built but not declared
    assert BF.dvb_check_lags(on, _args(r9_bev_warp_lags="0.5,0.9"))
    assert BF.dvb_check_lags(on, _args(r9_bev_warp_lags="0.5,1.5"))
    assert BF.dvb_check_shuffle(on, _args(r9_bev_warp_shuffle=True))
    off, _, _ = _rig(monkeypatch, on=False)
    assert BF.dvb_check(off, args)                                                                  # declared but not built
    assert BF.dvb_check(off, _args(r9_bev_warp=False)) == []
    on.__dict__["_modules"]["_map_hires"].lift._forward_hooks.clear()
    assert BF.dvb_check(on, args), "a fuse with no hook on the shared lift is built-but-unreachable and G-DVB must say so"


def test_the_trainers_call_sites():
    def src(f):
        return inspect.getsource(f)
    rigfn = src(T.compute_losses_perc_rig)
    assert rigfn.index("_bw.stage(model, batch, device)") < rigfn.index("out = model(frames, perception_grid=_pgrid, perception_valid=_pvalid)")
    assert "_bw.item_fields(self.episodes, e_i, t + w - 1, self._r9_bw)" in src(T.V3Dataset.__getitem__)
    tr = src(T.train)
    for needle in ("_bw.attach(model, args)", "_bw.attach_dataset(ds, model, args)", "_bw.attach_dataset(e_ds, model, args)", "_bw.stamp(model, args)"):
        assert needle in tr, needle
    assert "_bw.pin_args(args)" in inspect.getsource(T)
    from tanitad.eval import refcv7_loader as L
    assert "tr._bw.attach(model, args)" in inspect.getsource(L)


def test_OFF_the_dataset_item_and_the_stamp_are_the_tips(monkeypatch):
    off, cfg, args = _rig(monkeypatch, on=False, r9_bev_warp=False)
    batch, ds = _batch_t1(cfg, off, args, attach=False)
    assert not any(k.startswith("bw_") for k in batch) and getattr(ds, "_r9_bw", None) is None
    assert BF.stamp(off, args) is None and BF.attach_dataset(ds, off, args) == {} and BF.warm_start_prefixes(off) == ()
    on, _, a_on = _rig(monkeypatch, on=True)
    st = BF.stamp(on, a_on)
    assert st["lag_rows"] == [5, 10] and st["shuffle_control"] is False and st["n_params"] == sum(p.numel() for p in on._map_hires.bev_warp.parameters())
