"""refcv6 §6 -- the conflict detector WIRED INTO ``refc_v3_train.py``, on the
REAL model, with the REAL trajectory loss and the REAL BEV auxiliary head.

``tests/test_refcv6_grad_conflict.py`` proves the instrument on a two-head toy.
This file proves the three seams that connect it to the trainer, against
``RefCV3Model`` built from ``refc_v3_smoke_config`` and the trainer's own
``compute_losses_v3`` -- so "the controls read +1" is a statement about the trunk
that will actually be trained, not about a fixture.

⭐ **It also carries the 30x mutation a second time, on the real model**, because
the finding it produces (the cosine is scale-blind; ``ratio``/``proj`` are not)
is the one that changes how a refcv6 arm gets read, and a finding demonstrated
only on a toy is a finding about the toy.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
_SCRIPTS = os.path.join(_STACK, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_TRAINER_PY = os.path.join(_SCRIPTS, "refc_v3_train.py")

from tanitad.train import grad_conflict as gc          # noqa: E402


def _trainer():
    """``refc_v3_train.py`` by PATH -- the ``test_built_heads_receive_gradient``
    convention."""
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_conflict", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W_BEV = 0.2
N_RNG = 6


def _build(T, w_bev: float = W_BEV):
    from tanitad.refs import refc_bev_aux as RB
    cfg = T.v3.refc_v3_smoke_config(True)
    gs = cfg.core.encoder.grid_shape
    gh, gw = (gs() if callable(gs) else gs)
    cfg.core.bev_aux = RB.BEVAuxConfig(enable=True, kind="col", n_rng=N_RNG,
                                       n_az=gw, d_tok=8, hidden=16, w=w_bev,
                                       pos_weight=30.0)
    torch.manual_seed(0)
    m = T.v3.RefCV3Model(cfg)
    m._w_bev_aux = float(w_bev)
    m._w_goal_point = 0.0
    m._w_tac_goal = 0.0
    m._tac_goal_pos_weight = None
    m._tac_goal_class_mask = None
    m.train()
    return cfg, m, (gh, gw)


def _batch(T, cfg, gw, seed: int = 0):
    eps = T._synth_episodes(2, cfg.core, seed=seed)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    b["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    b["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID],
                               dtype=torch.long)
    b["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    b["nav_valid"] = torch.tensor([True, True])
    k = len(v7l.TAC_GOAL_TOKENS)
    b["tac_goal_y"] = torch.zeros(2, k)
    b["tac_goal_w"] = torch.zeros(2, k)
    g = torch.Generator().manual_seed(7 + seed)
    b["bev_occ"] = (torch.rand(2, N_RNG, gw, generator=g) < 0.1).float()
    # ⛔ bool, not float: `bev_aux_loss` refuses a float mask, because a float
    # one would WEIGHT the IGNORE state instead of removing it.
    b["bev_mask"] = torch.ones(2, N_RNG, gw, dtype=torch.bool)
    return b


@pytest.fixture(scope="module")
def live():
    T = _trainer()
    cfg, model, (gh, gw) = _build(T)
    batch = _batch(T, cfg, gw)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    return T, cfg, model, batch, losses


# --------------------------------------------------------------------------- #
#  the three trainer seams
# --------------------------------------------------------------------------- #
def test_flag_parsing(live):
    T = live[0]
    ns = __import__("argparse").Namespace
    assert T._conflict_override(ns(conflict_detector="auto")) is None
    assert T._conflict_override(ns(conflict_detector="on")) is True
    assert T._conflict_override(ns(conflict_detector="off")) is False
    assert T._conflict_override(ns()) is None          # absent == auto


def test_cli_defaults(tmp_path, live):
    T = live[0]
    a = T.build_parser().parse_args(["--arm", "hier", "--out", str(tmp_path)])
    assert a.conflict_detector == "auto"
    assert a.conflict_mode == "probe"                  # the PRE-REGISTERED one
    assert a.conflict_every == 1                       # every step


def test_aux_weights_exclude_a_dead_weight(live):
    """⛔ A weight of 0.0 is not a perception term: its gradient on the trunk is
    the zero vector and the cosine against it is the degenerate NaN, not a
    measurement. It must not switch the detector on."""
    T, _cfg, model, _b, _l = live
    assert T._conflict_aux_weights(model) == {"bev": pytest.approx(W_BEV)}
    model._w_bev_aux = 0.0
    try:
        assert T._conflict_aux_weights(model) == {}
        assert gc.enabled_for_arm(object(), aux_present=False) is False
    finally:
        model._w_bev_aux = W_BEV


def test_conflict_terms_are_the_WEIGHTED_losses(live):
    """The cosine is between the gradients that actually ARRIVE at the trunk, so
    both sides carry the weight the term enters ``loss`` with."""
    T, _cfg, model, _b, losses = live
    lt, la = T._conflict_terms(model, losses)
    assert lt is not None and la is not None
    assert float(lt.detach()) == pytest.approx(
        T.TRAJ_WEIGHT * float(losses["traj"].detach()))
    assert float(la.detach()) == pytest.approx(
        W_BEV * float(losses["bev"].detach()))
    assert lt.requires_grad and la.requires_grad


def test_the_trainer_REFUSES_a_detector_that_can_never_read_anything(tmp_path, live):
    """⛔⛔ MEASURED 2026-09-17 on the smoke path, and the reason this refusal
    exists: ``--conflict-detector on`` with no live perception weight built the
    detector, printed "ON", stamped ``conflict_detector.enabled=true`` in
    ``config.json`` -- and emitted **zero** ``cd_*`` rows for the whole run,
    because there is no second gradient. That is the ``--w-agent`` defect
    verbatim: a run whose record claims an instrument it never read.

    ⭐ Two-sided. ``auto`` must NOT refuse -- it already returns False with no
    aux -- or the guard would break every non-refcv6 arm in the repo.
    """
    T = live[0]
    argv = ["--arm", "hier", "--out", str(tmp_path / "run"), "--smoke",
            "--synth-episodes", "2", "--steps", "1", "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "1"]
    with pytest.raises(SystemExit, match="NO perception term has a live weight"):
        T.train(T.build_parser().parse_args(argv + ["--conflict-detector", "on"]))
    assert not (tmp_path / "run" / "config.json").exists(), (
        "the refusal must land BEFORE config.json is written -- the "
        "`test_smoke_train_refuses_legacy_file_without_the_override` convention")
    # the same argv with `auto` is the repo's default path and must run
    T.train(T.build_parser().parse_args(argv + ["--conflict-detector", "auto"]))
    cfg = __import__("json").loads(
        (tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert "conflict_detector" not in cfg, (
        "a non-refcv6 arm stamped a detector block it never ran")


def test_conflict_terms_return_nothing_when_the_aux_is_absent(live):
    """A batch with no supervised aux is a genuine ABSENCE. It is logged as a
    missing row, never as a zero conflict."""
    T, _cfg, model, _b, losses = live
    stripped = {k: v for k, v in losses.items() if k != "bev"}
    assert T._conflict_terms(model, stripped) == (None, None)


# --------------------------------------------------------------------------- #
#  the instrument, on the REAL trunk
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def det_and_terms(live):
    T, _cfg, model, _b, losses = live
    d = gc.GradientConflictDetector.for_model(model, gc.ConflictConfig(
        enabled=True, trunk_prefixes=gc.resolve_trunk_prefixes(model)))
    lt, la = T._conflict_terms(model, losses)
    return d, lt, la, model


def test_the_trunk_the_trainer_would_measure_is_the_real_encoder(det_and_terms):
    d, _lt, _la, model = det_and_terms
    # ⚠️ `RefCV3Model` wraps the trunk -- a hard-coded `encoder.` reads nothing.
    assert gc.resolve_trunk_prefixes(model) == ("core.encoder.",)
    p = d.provenance()
    assert p["n_trunk_params"] == sum(
        q.numel() for n, q in model.named_parameters()
        if n.startswith("core.encoder."))
    assert set(p["groups"]) == {"stem", "stage_stages0", "stage_stages1",
                                "stage_stages2", "stage_stages3"}


def test_the_three_controls_read_their_known_values_ON_THE_REAL_MODEL(det_and_terms):
    """⛔ THE POINT OF THE WHOLE EXERCISE. +1 / detached-0 / -1, exactly, on the
    trunk a refcv6 arm will actually train."""
    d, lt, la, _m = det_and_terms
    res = d.self_check(lt, la)                       # RAISES if any misses
    assert res.self_cos == 1.0
    assert res.negated_cos == -1.0
    assert math.isnan(res.detached_cos)
    assert res.detached_conflict == 0.0
    assert res.detached_norm_aux == 0.0
    assert res.ok and res.failures == ()


def test_per_group_rows_on_the_real_model(det_and_terms):
    d, lt, la, _m = det_and_terms
    r = d.measure(lt, la)
    row = r.row()
    assert row["cd_plan_side"] == "traj"
    assert r.pooled.norm_traj > 0.0 and r.pooled.norm_aux > 0.0
    assert not r.pooled.degenerate
    for g in ("stem", "stage_stages0", "stage_stages1", "stage_stages2",
              "stage_stages3"):
        assert f"cd_{g}_cos" in row and f"cd_{g}_ratio" in row
        assert -1.0 <= r.groups[g].cos <= 1.0
    # the structural control, on the real heads
    assert r.groups[gc.HEADS_NAME].cos == 0.0
    assert r.groups[gc.HEADS_NAME].degenerate is False


def test_the_30x_MUTATION_on_the_real_model(det_and_terms):
    """⛔⛔ THE MUTATION ARM, ON THE REAL TRUNK. Same verdict as the toy: the
    pre-registered cosine is BLIND to a 30x aux and the magnitude channels are
    not. If this ever changes, the module docstring's algebra is wrong."""
    d, lt, la, _m = det_and_terms
    r1 = d.measure(lt, la)
    r30 = d.measure(lt, la * 30.0)
    r32 = d.measure(lt, la * 32.0)
    assert r32.pooled.cos == r1.pooled.cos               # exact invariance
    assert abs(r30.pooled.cos - r1.pooled.cos) < 1e-6    # 30x: rounding only
    assert r30.pooled.ratio == pytest.approx(30.0 * r1.pooled.ratio, rel=1e-5)
    assert r30.pooled.proj == pytest.approx(30.0 * r1.pooled.proj, rel=1e-5)
    # and the magnitude channel lands in E-DEC-18's measured 10-30x regime
    assert r1.pooled.ratio < 1.0 < r30.pooled.ratio


def test_subtract_mode_on_the_real_model_reads_the_broader_plan_side(live):
    """The cheap mode, on the real model: ONE extra backward, ``.grad`` read not
    written, and the row says which planning loss it answered against."""
    T, _cfg, model, batch, _l = live
    d = gc.GradientConflictDetector.for_model(model, gc.ConflictConfig(
        enabled=True, mode=gc.MODE_SUBTRACT,
        trunk_prefixes=gc.resolve_trunk_prefixes(model)))
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    _lt, la = T._conflict_terms(model, losses)
    model.zero_grad(set_to_none=True)
    losses["loss"].backward(retain_graph=True)
    r = d.measure_after_backward(la)
    model.zero_grad(set_to_none=True)
    assert r.plan_side == "total_minus_aux"
    assert r.row()["cd_plan_side"] == "total_minus_aux"
    assert not r.pooled.degenerate
    assert r.pooled.norm_traj > 0.0 and r.pooled.norm_aux > 0.0
