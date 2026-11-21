"""refcv9 R9-PF -- the clean-split, regime-factored progress head (REFCV9_INMODEL_DESIGN.md, 2026-10-26).

Pins, each with a deliberate-regression arm that must go RED:

1. GEOMETRY: the torch re-timing equals ``tanitad.retime_stage`` (an independent numpy implementation) in float64;
   analytic targets (straight line, circle of radius 20 m); the identity ``s' = own arc`` returns the plan bit for bit.
   RED: the end-ray extension dropped.
2. LATERAL: every re-timed point lies on the selected polyline / its end ray. RED: a position re-scale (RS_LON-style).
3. FACTORED DECISION: the speed is the argmax decision's profile. RED: the soft (expected) profile.
4. GT DECISION: the MD/FD metric's own definition, on literal trajectories.
5. FLAGS OFF / ON: ON leaves every non-R9 state tensor and every forward output bit-identical (train AND eval; at eval
   only ``traj`` / ``wp_seq`` change, and ``traj_preretime`` IS the OFF plan). RED: the head built from the GLOBAL RNG.
6. STOP-GRADIENT: the calibration loss reaches the head and NO other tensor. RED: ``z_tac`` not detached.
7. TRAINER: the pin refusals, the DVB entries, the warm start that admits only R9 keys as new.

Every expectation is a LITERAL. CPU only.
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import _refcv8_rig as R  # noqa: E402
from tanitad import retime_stage as RS  # noqa: E402
from tanitad.refs import refcv9_progress as r9p  # noqa: E402

HZ = (5, 10, 15, 20, 30, 40, 50, 60)


# ------------------------------------------------------------------------------------------------ helpers
def _dist_to_polyline(path: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Independent of the code under test: min distance of each point to the segments (0,0)->p1->...->pS and to the
    end ray beyond pS (direction of the last positive-length segment, +x if none)."""
    V = np.concatenate([np.zeros((1, 2)), path], 0)
    segs = [(V[i], V[i + 1]) for i in range(len(V) - 1)]
    d_end = np.array([1.0, 0.0])
    for a, b in reversed(segs):
        if np.linalg.norm(b - a) > 0:
            d_end = (b - a) / np.linalg.norm(b - a)
            break
    out = []
    for p in pts:
        best = np.inf
        for a, b in segs:
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0 else min(max(float((p - a) @ ab) / L2, 0.0), 1.0)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        t = max(float((p - V[-1]) @ d_end), 0.0)
        best = min(best, float(np.linalg.norm(p - (V[-1] + t * d_end))))
        out.append(best)
    return np.asarray(out)


def _circle_plan(R_m: float = 20.0, ds: float = 3.0, n: int = 8) -> np.ndarray:
    """Points on a circle of radius R through the origin, heading +x, centre (0, R), every ds metres of ARC."""
    th = np.arange(1, n + 1) * ds / R_m
    return np.stack([R_m * np.sin(th), R_m * (1 - np.cos(th))], 1)


def _retime_np(O, s):
    return RS.eval_polyline(RS.polyline_geometry(O), s)


# ------------------------------------------------------------------------------------------------ 1. geometry
def test_GEOM_straight_line_lands_exactly_on_the_x_axis_and_extends_along_the_end_ray():
    O = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]]],
                     dtype=torch.float64)
    s = torch.tensor([[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 9.5]], dtype=torch.float64)
    P = r9p.retime(O, s)
    assert P[0, :, 0].tolist() == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 9.5]
    assert P[0, :, 1].tolist() == [0.0] * 8


def test_GEOM_own_arc_returns_the_plan_BIT_FOR_BIT():
    O = torch.from_numpy(_circle_plan())[None]
    P = r9p.retime(O, r9p.arc_at_slots(O))
    assert torch.equal(P, O)


def test_GEOM_circle_vertices_stay_on_the_radius_20_circle():
    O = torch.from_numpy(_circle_plan())[None]
    L = r9p.arc_at_slots(O)
    P = r9p.retime(O, L[:, [3, 1, 6, 0, 7, 2, 5, 4]])           # the vertices, re-ordered by progress
    r = torch.hypot(P[0, :, 0] - 0.0, P[0, :, 1] - 20.0)
    assert float((r - 20.0).abs().max()) < 1e-9


def test_GEOM_torch_port_equals_retime_stage_numpy_in_float64():
    g = np.random.default_rng(7)
    O = np.cumsum(g.normal(1.0, 0.8, (64, 8, 2)), axis=1)
    O[3] = 0.0                                                    # an all-zero plan: the +x fallback
    O[5, 2:4] = O[5, 1]                                           # zero-length segments inside a plan
    s = np.abs(g.normal(5.0, 6.0, (64, 8)))                       # incl. far beyond the end (the end ray)
    ref = _retime_np(O, s)
    got = r9p.retime(torch.from_numpy(O), torch.from_numpy(s)).numpy()
    assert float(np.abs(got - ref).max()) < 1e-12


def test_GEOM_MUTATION_no_end_ray_goes_RED(monkeypatch):
    O = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]]],
                     dtype=torch.float64)
    s = torch.full((1, 8), 9.5, dtype=torch.float64)

    real = r9p.eval_polyline

    def no_ray(geo, ss):                                          # clamps at the last vertex instead of extending
        return real(geo, torch.minimum(ss, geo["L"][:, -1:]))
    monkeypatch.setattr(r9p, "eval_polyline", no_ray)
    P = r9p.retime(O, s)
    assert P[0, 0, 0].item() != 9.5                               # the analytic target is missed -> the pin is RED


# ------------------------------------------------------------------------------------------------ 2. lateral
def test_LATERAL_every_retimed_point_is_on_the_selected_polyline():
    O = _circle_plan()
    s = np.array([0.7, 2.9, 5.0, 9.1, 13.3, 17.0, 23.9, 31.0])    # incl. beyond the 24 m arc
    P = r9p.retime(torch.from_numpy(O)[None], torch.from_numpy(s)[None])[0].numpy()
    assert float(_dist_to_polyline(O, P).max()) < 1e-9


def test_LATERAL_MUTATION_a_position_rescale_leaves_the_path_RED():
    O = _circle_plan()
    L = r9p.arc_at_slots(torch.from_numpy(O)[None])[0].numpy()
    s = 0.6 * L                                                   # a decel-like shrink
    P_bad = O * (s / L)[:, None]                                  # RS_LON-style: scale the POSITIONS, not the timing
    assert float(_dist_to_polyline(O, P_bad).max()) > 0.05


# ------------------------------------------------------------------------------------------------ 3. factored decision
def _crafted():
    logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 3.0, 0.0], [-2.0, 0.0, 5.0]])
    prof = torch.tensor([[[-1.0] * 8, [0.0] * 8, [1.0] * 8]] * 3)
    return logits, prof


def _is_hard(fn) -> bool:
    logits, prof = _crafted()
    return fn(prof, logits)[:, 0].tolist() == [-1.0, 0.0, 1.0]


def test_FACTORED_the_speed_is_the_ARGMAX_decision_profile():
    assert _is_hard(lambda prof, lg: r9p.select_profile(prof, r9p.decide(lg)))


def test_FACTORED_MUTATION_the_soft_expected_profile_goes_RED():
    assert not _is_hard(lambda prof, lg: (lg.softmax(-1)[..., None] * prof).sum(1))


def test_FACTORED_tau_moves_the_operating_point_to_decel():
    lg = torch.log(torch.tensor([[0.4, 0.5, 0.1]]))
    assert r9p.decide(lg).tolist() == [1]
    assert r9p.decide(lg, 0.3).tolist() == [0]
    assert r9p.decide(lg, 0.45).tolist() == [1]


def test_FACTORED_trunk_lon_argmax_maps_LON6_groups_to_regimes():
    lon = torch.full((6, 8), -5.0)
    for i, c in enumerate((0, 1, 3, 4, 5, 7)):                    # FOLLOW CRUISE BRAKE_TO CREEP HOLD ACCELERATE
        lon[i, c] = 5.0
    lon[:, 2] = 9.0                                               # a NON-LON6 column must be ignored
    assert r9p.trunk_lon_regime(lon).tolist() == [1, 1, 0, 0, 0, 2]


# ------------------------------------------------------------------------------------------------ 4. GT decision
def test_GT_regime_is_the_metric_definition_on_literal_trajectories():
    t = torch.tensor([h * 0.1 for h in HZ])
    tgt = torch.zeros(3, 8, 2)
    tgt[0, :, 0] = 10.0 * t                                       # constant 10 m/s, v0 10 -> cruise
    tgt[1, :, 0] = 10.0 * t
    tgt[1, 7, 0] = tgt[1, 6, 0] + 5.0                             # last chord 5 m/s, v0 10 -> dv -5 -> decel
    tgt[2, :, 0] = 5.0 * t
    tgt[2, 7, 0] = tgt[2, 6, 0] + 7.0                             # 7 m/s vs v0 5 -> dv +2 -> accel
    sv = torch.ones(3, 8, dtype=torch.bool)
    sv[2, 7] = False
    r, ok = r9p.gt_regime(tgt, sv, torch.tensor([10.0, 10.0, 5.0]), HZ)
    assert r.tolist() == [1, 0, 2]
    assert ok.tolist() == [True, True, False]


def test_ADMISSIBLE_the_scalar_set_is_H2cns_and_carries_no_curvature():
    assert r9p.SCALAR_NAMES == ("v0", "a0", "omega0", "v0_omega0", "v0_sq", "a0_v0", "ceiling", "ceiling_minus_v0",
                                "ceiling_valid")
    x = r9p.admissible_scalars(torch.tensor([10.0, 4.0]), torch.tensor([0.5, -1.0]), torch.tensor([0.1, 0.0]),
                               torch.tensor([13.89, 50.0]), torch.tensor([1.0, 0.0]))
    assert x[0].tolist() == pytest.approx([10.0, 0.5, 0.1, 1.0, 100.0, 5.0, 13.89, 3.89, 1.0], abs=1e-5)
    assert x[1].tolist() == pytest.approx([4.0, -1.0, 0.0, 0.0, 16.0, -4.0, 40.0, 36.0, 0.0], abs=1e-5)


def test_PARAMS_literal_count_at_d_tac_512():
    assert r9p.count_params(r9p.R9ProgressHead(512, 8, 128)) == 70299


# ------------------------------------------------------------------------------------------------ 5./6. the model
@pytest.fixture(scope="module")
def T():
    return R.trainer()


def _build(T, r9: bool):
    cfg, m = R.build(T, True, seed=0)
    if r9:
        import copy
        cfgr = copy.deepcopy(cfg)          # the rig's own cfg, + the R9 switch; the SAME seed and build order
        cfgr.refcv9 = r9p.R9Config(enable=True, calib_stage=True)
        torch.manual_seed(0)
        m2 = T.v3.RefCV3Model(cfgr)
        m2._perception = R.FakeBranch(R.D_BEV)
        with torch.no_grad():
            m2.core.decoder.anchor_controls.copy_(R.grid())
        return cfgr, m2
    return cfg, m


def test_FLAGS_OFF_builds_nothing_and_ON_leaves_every_other_tensor_BIT_IDENTICAL(T):
    _c0, m0 = _build(T, False)
    _c1, m1 = _build(T, True)
    assert m0.r9_prog is None and m1.r9_prog is not None
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert not [k for k in s0 if r9p.is_refcv9_key(k)]
    assert sorted(set(s1) - set(s0)) == sorted(k for k in s1 if r9p.is_refcv9_key(k))
    assert [k for k in s0 if not torch.equal(s0[k], s1[k])] == []


def _outs(cfg, m, train):
    bt = R.batch(cfg, b=3, seed=1)
    return R.forward(cfg, m, bt, seed=2, train=train)


def _diff_keys(o0, o1):
    bad = []
    for k, v in o0.items():
        if torch.is_tensor(v) and not torch.equal(v, o1[k]):
            bad.append(k)
    return sorted(bad)


def test_FLAGS_ON_forward_changes_ONLY_the_emitted_plan_and_only_at_eval(T):
    c0, m0 = _build(T, False)
    c1, m1 = _build(T, True)
    tr0, tr1 = _outs(c0, m0, True), _outs(c1, m1, True)
    assert _diff_keys(tr0, tr1) == []                             # training: no output moves at all
    ev0, ev1 = _outs(c0, m0, False), _outs(c1, m1, False)
    assert _diff_keys(ev0, ev1) == ["traj", "wp_seq"]
    assert torch.equal(ev1["traj_preretime"], ev0["traj"])
    P, O = ev1["traj"].double().numpy(), ev1["traj_preretime"].double().numpy()
    for i in range(P.shape[0]):                                   # the speed moved, the PATH did not
        assert float(_dist_to_polyline(O[i], P[i]).max()) < 1e-4
    assert torch.allclose(ev1["traj"], r9p.retime(ev1["traj_preretime"], ev1["r9_s_target"]).to(ev1["traj"].dtype))


def test_FLAGS_MUTATION_a_head_built_from_the_GLOBAL_RNG_goes_RED(T, monkeypatch):
    monkeypatch.setattr(torch.random, "fork_rng", lambda *a, **k: contextlib.nullcontext())
    _c0, m0 = _build(T, False)
    _c1, m1 = _build(T, True)
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert [k for k in s0 if not torch.equal(s0[k], s1[k])] != []    # the module built after the model moved


def _calib_loss(cfg, m, fwd_fn=None):
    bt = R.batch(cfg, b=3, seed=1)
    m.eval()
    m.r9_prog.train()
    torch.manual_seed(2)
    ph = bt["pose_hist"]
    m.core.set_ego_window(ph, int(ph.shape[1]))
    out = m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=cfg.core.decoder.diffusion_steps)
    t = torch.tensor([h * 0.1 for h in HZ])
    tgt = bt["v0"][:, None, None] * torch.stack([t, torch.zeros_like(t)], -1)[None]
    sv = torch.ones(3, 8, dtype=torch.bool)
    loss, tele = r9p.r9_losses(cfg.refcv9, out, tgt, sv, bt["v0"], HZ)
    return loss, tele


def test_STOPGRAD_the_calibration_loss_reaches_the_head_and_NOTHING_else(T):
    cfg, m = _build(T, True)
    m.zero_grad(set_to_none=True)
    loss, tele = _calib_loss(cfg, m)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert r9p.frozen_grad_leak(m) == 0.0
    g = [float(p.grad.abs().sum()) for n, p in m.named_parameters() if r9p.is_refcv9_key(n) and p.grad is not None]
    assert len(g) == 6 and min(g) > 0.0                           # body W/b, regime W/b, profile W/b all receive it


def test_STOPGRAD_MUTATION_an_undetached_z_tac_leaks_into_the_trunk_RED(T, monkeypatch):
    cfg, m = _build(T, True)
    real = r9p.head_forward

    def leaky(head, c, z, x, s_prior, lon):
        o = real(head, c, z, x, s_prior, lon)
        h2 = head(z, x)                                           # the same head WITHOUT the detach
        o["r9_profiles"] = h2["profiles"]
        o["r9_regime_logits"] = h2["regime_logits"]
        return o
    monkeypatch.setattr(r9p, "head_forward", leaky)
    m.zero_grad(set_to_none=True)
    loss, _ = _calib_loss(cfg, m)
    loss.backward()
    assert r9p.frozen_grad_leak(m) > 0.0


def test_CALIB_FREEZE_leaves_only_the_head_trainable(T):
    _cfg, m = _build(T, True)
    rec = r9p.apply_calib_freeze(m)
    assert rec["n_trainable"] == r9p.count_params(m.r9_prog)
    assert [n for n, p in m.named_parameters() if p.requires_grad and not r9p.is_refcv9_key(n)] == []


# ------------------------------------------------------------------------------------------------ 7. trainer
_BASE = ["--arm", "hier", "--out", "_r9_unused"]


_W = ["--w-r9-prog", "1.0", "--w-r9-regime", "1.0"]


@pytest.mark.parametrize("extra, msg", [(["--r9-progress-head"] + _W, "trains ONLY in the calibration stage"),
                                        (["--r9-calib-stage"], "nothing to calibrate"),
                                        (["--r9-progress-head", "--r9-calib-stage", "--r9-decel-tau", "1.5"] + _W,
                                         "must lie in (0, 1)"),
                                        (["--r9-progress-head", "--r9-calib-stage"], "--w-r9-prog 0"),
                                        (["--r9-progress-head", "--r9-calib-stage", "--w-r9-prog", "1.0"],
                                         "--w-r9-regime 0")])
def test_PIN_refusals(T, extra, msg):
    args = T.build_parser().parse_args(_BASE + extra)
    with pytest.raises(SystemExit, match=None) as ei:
        r9p.pin_args(args)
    assert msg in str(ei.value)


def test_PIN_on_sets_the_declared_config(T):
    args = T.build_parser().parse_args(_BASE + ["--r9-progress-head", "--r9-calib-stage", "--r9-decision", "trunk_lon",
                                                "--w-r9-prog", "0.5"])
    r9p.pin_args(args)                       # trunk_lon: the regime CE may be 0 (the decision is not the head's)
    c = r9p.config_from_args(args)
    assert (c.enable, c.calib_stage, c.decision, c.hidden, c.decel_tau, c.w_prog, c.w_regime) == \
        (True, True, "trunk_lon", 128, None, 0.5, 0.0)


def test_DVB_the_eight_r9_flags_are_registered_with_their_kinds(T):
    from tanitad.train import declared_vs_built as dvb
    got = {d: dvb.REGISTRY[d].kind for d in ("r9_progress_head", "r9_calib_stage", "r9_decision", "r9_hidden",
                                             "r9_decel_tau", "r9_seed", "w_r9_prog", "w_r9_regime")}
    assert got == {"r9_progress_head": "built", "r9_calib_stage": "built", "r9_decision": "built",
                   "r9_hidden": "built", "r9_decel_tau": "built", "r9_seed": "built", "w_r9_prog": "loss",
                   "w_r9_regime": "loss"}


def test_DVB_declared_but_not_built_is_NAMED(T):
    from tanitad.train import declared_vs_built as dvb
    _c, m = _build(T, False)
    args = T.build_parser().parse_args(_BASE + ["--r9-progress-head", "--r9-calib-stage"])
    got = dvb.REGISTRY["r9_progress_head"].check(m, args)
    assert [g.lever for g in got] == ["--r9-progress-head"]


def test_WARM_START_admits_only_R9_keys_as_new(T, tmp_path):
    from tanitad.train import refcv8_train as r8train
    _c0, m0 = _build(T, False)
    p = tmp_path / "src.pt"
    torch.save({"model": m0.state_dict(), "step": 7}, str(p))
    _c1, m1 = _build(T, True)
    rec = r8train.warm_start_from(m1, str(p))
    # 6 params (body / regime / profile W, b) + BN running_mean / running_var; BatchNorm back-fills a missing
    # num_batches_tracked itself, so torch does not report it missing (9 R9 state keys, 8 reported)
    assert len([k for k in m1.state_dict() if r9p.is_refcv9_key(k)]) == 9
    assert rec["n_new_refcv8_keys"] == 8


def test_DVB_the_calibration_freeze_is_DECLARED_and_an_undeclared_one_goes_RED(T):
    from tanitad.train import declared_vs_built as dvb
    _c, m = _build(T, True)
    args = T.build_parser().parse_args(["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
                                        "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
                                        "--residual-prior", "ha0_ext_pose", "--r9-progress-head", "--r9-calib-stage",
                                        "--out", "_r9_unused"] + R.F1_F6)
    r9p.apply_calib_freeze(m)

    def frozen_rows():
        return [x for x in dvb.check_grad_unreachable(m, args) if x.lever == "frozen parameters"]
    assert frozen_rows() == []
    assert dvb.REGISTRY["r9_calib_stage"].check(m, args) == []
    delattr(m, r9p.CALIB_MARKER_ATTR)                             # the SAME freeze, no declaration
    assert len(frozen_rows()) == 1
    next(p for n, p in m.named_parameters() if not r9p.is_refcv9_key(n)).requires_grad_(True)
    assert [x.lever for x in dvb.REGISTRY["r9_calib_stage"].check(m, args)] == ["--r9-calib-stage"]


def test_WEIGHTS_default_ZERO_and_are_GATED_on_head_plus_stage(T):
    args = T.build_parser().parse_args(_BASE)
    assert (args.w_r9_prog, args.w_r9_regime) == (0.0, 0.0)
    on = T.build_parser().parse_args(_BASE + ["--r9-progress-head", "--r9-calib-stage"] + _W)
    for d in ("w_r9_prog", "w_r9_regime"):
        assert T.REFC_WEIGHT_GATES[d]["gate"](args)[0] is False
        assert T.REFC_WEIGHT_GATES[d]["gate"](on)[0] is True
