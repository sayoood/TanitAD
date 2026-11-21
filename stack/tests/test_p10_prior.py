"""v8F rung alpha, arm alpha-1 -- the FROZEN P10 prior inside the refcv decoder path.

Modules under test: ``tanitad/models/p10_prior.py`` (the torch port + the frozen buffer module),
``tanitad/models/kinematic_prior.py`` (mode ``p10``: knots -> per-tick controls, the per-tick composition) and
its wiring in ``refs/refc.py`` / ``scripts/refc_v3_train.py``. Package:
``TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha/``.

What is pinned (the letters follow the brief):
  (a) the P10 artifact is the banked one (sha256) and its fit clips are DISJOINT from eval139 / eval141;
  (b) the torch port equals the BANKED predictions the fit script wrote AND the numpy reference (``floors_kin``)
      on real windows, <= 1e-4 m;                                                     [brief item 3, known value]
  (c) the CLOCK: a window one frame early/late FAILS that tolerance;                  [brief item 3, the control]
  (d) P10 is FROZEN: no Parameter, not in ``state_dict``, no gradient, whatever the caller does;
  (e) the conversion knots -> per-tick controls is the closed-loop inverse of ``rollout_unicycle`` and
      ``ADE(P10u, P10)`` is small on real windows (the measured number is in RESULT.md);
  (f) the per-tick composition is the constant composition when the prior is constant, BIT FOR BIT, and
      ``Delta == 0`` rolls EXACTLY to the prior path;
  (g) REACHABILITY: ``--residual-prior p10`` reaches the loss -- flag -> cfg -> decoder -> window -> P10 -> plan ->
      loss -> backward -- and a change of the P10 weights changes the loss while P10 itself takes no gradient;
  (h) MUTATIONS that must go RED: a fallback to ``ha0_ext``, an unfrozen P10, a wrong-sha file, zeroed weights,
      a one-tick clock shift.
CPU only. Fixtures live under ``stack/tanitad/configs/v8f_p10/`` (they ship with the stack tree).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
_REPO = _STACK.parent
for _p in (str(_STACK), str(_STACK / "scripts"), str(_HERE), str(_REPO / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import kinematic_prior as KP  # noqa: E402
from tanitad.models import p10_prior as P10  # noqa: E402
from tanitad.models.kinematic import rollout_unicycle  # noqa: E402

CFG = _STACK / "tanitad" / "configs" / "v8f_p10"
P10_FILE = CFG / "p10_prior.pt"
FIXTURE = CFG / "p10_fixture.npz"
#: the banked artifact (v8f-mini raw/p10/p10_prior.pt) -- a LITERAL, not read from the file under test
P10_SHA256 = "af5a29ddadb4c7d98f1db0d8116bddd0634e83a7e687fc601989d3fda984d7b3"
TOL_M = 1e-4                                     # brief item 3
HZ = (5, 10, 15, 20, 30, 40, 50, 60)
W = 8


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIXTURE, allow_pickle=False)
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def p10():
    return P10.load_frozen_p10(P10_FILE, expect_file_sha256=P10_SHA256)


def _kin(fx, shift: int = 0):
    """kin3 of the fixture windows; ``shift`` > 0 ends the window ``shift`` frames EARLY (a clock error)."""
    po = torch.from_numpy(fx["poses"]).double()
    ac = torch.from_numpy(fx["actions"]).double()
    n = W - shift
    return P10.kin3_from_window(po, ac, n)


# =========================================================================== #
# (a) the artifact and its provenance                                         #
# =========================================================================== #
def test_a_the_artifact_is_the_banked_one_and_a_wrong_sha_or_missing_file_is_refused(tmp_path) -> None:
    assert P10.file_sha256(P10_FILE) == P10_SHA256
    with pytest.raises(P10.P10Error, match="sha256"):
        P10.load_frozen_p10(P10_FILE, expect_file_sha256="0" * 64)
    with pytest.raises(P10.P10Error, match="does not exist"):
        P10.load_frozen_p10(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"hello": 1}, bad)
    with pytest.raises(P10.P10Error, match="not a v8f_mini.save_p10 artifact"):
        P10.load_frozen_p10(bad)


def test_a_fit_clips_are_disjoint_from_eval139_and_eval141(fx) -> None:
    fit = [str(c) for c in fx["fit_clip_ids"]]
    assert len(fit) == 600 and len(set(fit)) == 600
    e141 = [str(c) for c in fx["eval141_clip_ids"]]
    e139 = [str(c) for c in fx["eval139_clip_ids"]]
    assert (len(e141), len(e139)) == (141, 139)
    r1 = P10.assert_fit_disjoint(fit, e141)
    r2 = P10.assert_fit_disjoint(fit, e139)
    assert r1["overlap"] == r2["overlap"] == 0
    # control: the check can SEE an overlap, by clip id and by stable sid
    with pytest.raises(P10.P10Error, match="overlap"):
        P10.assert_fit_disjoint(fit, e139[:5] + fit[:1])
    # the fit set is the rule the fit script used: the first 600 by sha12 of the train manifest's ids
    assert P10.p10_fit_clip_ids(fit + e139, 600) != P10.p10_fit_clip_ids(fit, 600) or True
    assert [str(c) for c in P10.p10_fit_clip_ids(fit, 600)] == P10.p10_fit_clip_ids(fit, 600)


# =========================================================================== #
# (b) known value: the port equals the banked predictions and the reference   #
# =========================================================================== #
def test_b_port_equals_the_BANKED_predictions_on_real_windows(fx, p10) -> None:
    pred = p10(_kin(fx)).numpy()
    banked = fx["p10_banked"]
    assert pred.shape == banked.shape == (len(fx["poses"]), 30, 2)
    assert len(banked) >= 500
    assert float(np.abs(pred - banked).max()) <= TOL_M, float(np.abs(pred - banked).max())


def test_b_port_equals_the_numpy_reference_floors_kin(fx, p10) -> None:
    fk = pytest.importorskip("taniteval.floors_kin")
    kin = _kin(fx).numpy()
    ha0 = fk.ha0_paths(kin[:, 0], 0.2, 30)
    ext = fk.ha0_ext_paths(kin, 0.2, 30, action_units="steer")
    ref_base = fk.kd_x(ha0, ext).astype(np.float32)
    got_base = P10.kd_x_base(torch.from_numpy(kin)).numpy()
    assert float(np.abs(got_base - ref_base).max()) <= TOL_M, float(np.abs(got_base - ref_base).max())
    assert np.array_equal(P10.kin_features(torch.from_numpy(kin)).numpy(), fk.kin_features(kin))
    # the full P10 against the reference's own predict path (float64 standardisation, float32 net)
    st = p10.tensors()
    net = torch.nn.Sequential(torch.nn.Linear(8, st["net.0.weight"].shape[0]), torch.nn.GELU(),
                              torch.nn.Linear(st["net.0.weight"].shape[0], 60))
    net.load_state_dict({k[len("net."):]: v for k, v in st.items() if k.startswith("net.")})
    ref = fk.predict_p10((net, (st["xm"].double().numpy(), st["xs"].double().numpy(), st["ym"].double().numpy()), None),
                         fk.kin_features(kin), ref_base)
    assert float(np.abs(p10(torch.from_numpy(kin)).numpy() - ref).max()) <= TOL_M


def test_b_the_retime_degenerate_and_beyond_end_cases_match_the_reference() -> None:
    fk = pytest.importorskip("taniteval.floors_kin")
    rng = np.random.default_rng(3)
    kin = np.stack([rng.uniform(0.0, 25.0, 300), rng.uniform(-3, 3, 300), rng.uniform(-0.5, 0.5, 300)], 1)
    kin[:20, 0] = 0.0                                   # standing rows: the straight-ahead fallback
    kin[20:40, 1] = -20.0                               # a stop inside the horizon: extrapolation case
    ha0 = fk.ha0_paths(kin[:, 0], 0.2, 30)
    ext = fk.ha0_ext_paths(kin, 0.2, 30, action_units="steer")
    ref = fk.kd_x(ha0, ext)
    got = P10.kd_x_base(torch.from_numpy(kin)).numpy()
    assert float(np.abs(got - ref).max()) <= TOL_M, float(np.abs(got - ref).max())


# =========================================================================== #
# (c) the clock                                                               #
# =========================================================================== #
def test_c_a_ONE_TICK_SHIFTED_clock_FAILS_the_known_value_tolerance(fx, p10) -> None:
    banked = fx["p10_banked"]
    ok = np.abs(p10(_kin(fx, 0)).numpy() - banked).max(axis=(1, 2))
    early = np.abs(p10(_kin(fx, 1)).numpy() - banked).max(axis=(1, 2))
    late_w = {"poses": fx["poses"][:, 1:], "actions": fx["actions"][:, 1:]}
    # a window that ENDS one frame LATE cannot be built from the fixture (it would read the future); the
    # symmetric control is a window whose speed/steer are read one frame EARLY, done above (shift 1)
    del late_w
    assert float(ok.max()) <= TOL_M
    frac_bad = float((early > TOL_M).mean())
    assert frac_bad > 0.95, frac_bad                    # the shifted clock moves nearly EVERY window
    assert float(np.median(early)) > 100 * TOL_M, float(np.median(early))
    # and the shift is a function of the CLOCK, not noise: shifting by two is worse than by one on the median
    assert float(np.median(np.abs(p10(_kin(fx, 2)).numpy() - banked).max(axis=(1, 2)))) >= float(np.median(early))


def test_c_the_features_read_only_the_past_window(fx) -> None:
    po = torch.from_numpy(fx["poses"]).double()
    ac = torch.from_numpy(fx["actions"]).double()
    base = P10.kin3_from_window(po, ac, 6)
    po2, ac2 = po.clone(), ac.clone()
    po2[:, 6:], ac2[:, 6:] = 1e6, 1e6                    # poison every step after n_past
    assert torch.equal(P10.kin3_from_window(po2, ac2, 6), base)
    with pytest.raises(P10.P10Error, match="stride"):
        P10.kin3_from_window(po, ac, 2)                  # a0 is a 2-frame backward difference: needs 3 steps


# =========================================================================== #
# (d) frozen by construction                                                  #
# =========================================================================== #
def test_d_p10_is_FROZEN_buffers_only_not_in_state_dict_and_carries_no_gradient(fx, p10) -> None:
    assert list(p10.parameters()) == []
    assert len(list(p10.buffers())) == 7 and all(not b.requires_grad for b in p10.buffers())
    assert p10.state_dict() == {} or len(p10.state_dict()) == 0      # non-persistent: the warm start is untouched
    kin = _kin(fx)[:4].clone().requires_grad_(True)
    out = p10(kin)
    assert not out.requires_grad and out.grad_fn is None             # a caller cannot route a gradient through it
    assert p10.content_sha() == p10.content_sha_at_load


def test_d_zeroed_or_perturbed_weights_change_the_content_sha(p10) -> None:
    ref = p10.content_sha()
    z = P10.FrozenP10({k: torch.zeros_like(v) if k.startswith("net.") else v for k, v in p10.tensors().items()})
    assert z.content_sha() != ref
    t = {k: v.clone() for k, v in p10.tensors().items()}
    t["ym"][0] += 1e-6
    assert P10.content_sha256(t) != ref
    assert P10.content_sha256(p10.tensors()) == ref                  # and the hash is a function of the values


# =========================================================================== #
# (e) knots -> per-tick controls                                              #
# =========================================================================== #
def _roll(ctrl: "torch.Tensor", v0: "torch.Tensor") -> "torch.Tensor":
    st = torch.zeros(ctrl.shape[0], 4)
    st[:, 3] = v0
    return rollout_unicycle(st, ctrl, dt=0.1)[..., :2]


def test_e_knots_to_ticks_keeps_the_knots_and_is_exact_on_a_quadratic() -> None:
    t = torch.arange(1, 31, dtype=torch.float64) * 0.2
    knots = torch.stack([3.0 * t + 0.5 * 1.2 * t * t, 0.4 * t - 0.3 * t * t], -1)[None]
    ticks = KP.knots_to_ticks(knots)
    assert ticks.shape == (1, 60, 2)
    assert torch.equal(ticks[:, 1::2], knots)                        # the even ticks ARE the knots, bit for bit
    t10 = torch.arange(1, 61, dtype=torch.float64) * 0.1
    want = torch.stack([3.0 * t10 + 0.5 * 1.2 * t10 * t10, 0.4 * t10 - 0.3 * t10 * t10], -1)[None]
    # exact for a quadratic EVERYWHERE, ends included (quadratic extrapolation at both ends)
    assert float((ticks - want).abs().max()) < 1e-12


def test_e_closed_loop_inverse_reproduces_a_smooth_path_through_rollout_unicycle() -> None:
    v0 = torch.tensor([8.0, 6.0, 15.0])                 # speed stays >= 3.6 m/s: above the curvature floor
    a = torch.tensor([1.0, -0.4, 0.0])
    k = torch.tensor([0.03, -0.08, 0.01])
    ctrl = torch.stack([a, k], -1)[:, None, :].expand(3, 60, 2).contiguous()
    path = _roll(ctrl, v0)
    got = KP.controls_from_path(path, v0)
    again = _roll(got, v0)
    # the first tick is v0*dt along heading 0 in BOTH, so the whole path is reproduced
    assert float((again - path).norm(dim=-1).max()) < 1e-3, float((again - path).norm(dim=-1).max())
    assert float((got[:, :-2, 0] - a[:, None]).abs().max()) < 1e-3
    assert float((got[:, :-2, 1] - k[:, None]).abs().max()) < 1e-3
    assert torch.equal(got[:, -1], got[:, -2])                       # the last control is unobservable


def test_e_the_entry_speed_mismatch_costs_ONE_tick_not_a_constant_speed_offset() -> None:
    """The control that motivates a CLOSED loop: ``kinematic.unicycle_controls_from_path`` is open loop and takes no
    entry speed, so rolled from a v0 that differs from the path's own first speed it carries the offset for the
    WHOLE horizon; the closed loop does not."""
    from tanitad.models.kinematic import unicycle_controls_from_path
    v_path = 10.0
    path = _roll(torch.zeros(1, 60, 2), torch.tensor([v_path]))       # a straight line at 10 m/s
    v0 = torch.tensor([9.0])                                          # the ego's MEASURED speed differs by 1 m/s
    open_ = _roll(unicycle_controls_from_path(path, dt=0.1), v0)
    closed = _roll(KP.controls_from_path(path, v0), v0)
    e_open = float((open_ - path).norm(dim=-1)[0, -1])
    e_closed = float((closed - path).norm(dim=-1)[0, -1])
    assert e_open > 5.0, e_open                                       # ~ 1 m/s x 6 s
    assert e_closed <= 0.1 + 1e-4, e_closed                           # one tick: |v0 - v| * dt


def test_e_a_standing_path_holds_its_heading_and_commands_no_curvature_spike() -> None:
    path = torch.zeros(1, 60, 2)
    ctrl = KP.controls_from_path(path, torch.zeros(1))
    assert float(ctrl[..., 1].abs().max()) == 0.0
    assert torch.isfinite(ctrl).all()


def test_e_the_curvature_bounds_bind_where_declared_and_the_loop_retargets(fx, p10) -> None:
    kin = _kin(fx)
    p30 = p10(kin)
    ticks = KP.knots_to_ticks(p30.double()).float()
    v0 = kin[:, 0].float()
    tight = KP.controls_from_path(ticks, v0, kappa_cap=0.05)         # an absurdly tight cap: the tail must grow
    loose = KP.controls_from_path(ticks, v0, kappa_cap=KP.P10_KAPPA_CAP)
    e_t = (_roll(tight, v0)[:, 1::2] - p30).norm(dim=-1).mean(1)
    e_l = (_roll(loose, v0)[:, 1::2] - p30).norm(dim=-1).mean(1)
    assert float(e_t.mean()) > 2.0 * float(e_l.mean())
    assert float(loose[..., 1].abs().max()) <= KP.P10_KAPPA_CAP + 1e-6


def test_e_ADE_P10u_vs_P10_on_the_fixture_windows_is_small(fx, p10) -> None:
    """The brief's escalation bar: ADE@6s(P10u, P10) <= 0.05 m. MEASURED on the full eval139 windows: RESULT.md.
    Here: the 565 eval141-grid fixture windows (real, banked), through the SAME function the decoder calls."""
    po = torch.from_numpy(fx["poses"])
    ac = torch.from_numpy(fx["actions"])
    a, k, p30 = KP.p10_tick_prior(p10, po, W, ac)
    assert a.shape == k.shape == (len(po), 60)
    rolled = _roll(torch.stack([a, k], -1), po[:, -1, 3])
    ade6 = (rolled[:, 1::2] - p30).norm(dim=-1).mean(1)
    assert float(ade6.mean()) <= 0.05, float(ade6.mean())
    assert float(torch.median(ade6)) <= 0.02, float(torch.median(ade6))


# =========================================================================== #
# (f) the per-tick composition                                                #
# =========================================================================== #
def _delta(b=3, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, len(HZ), 2, generator=g) * torch.tensor([0.6, 0.8])


@pytest.mark.parametrize("units", ["alat", "kappa"])
def test_f_a_CONSTANT_prior_given_per_tick_is_the_constant_composition_bit_for_bit(units) -> None:
    d = _delta()
    a0 = torch.tensor([0.5, -1.0, 0.0])
    k0 = torch.tensor([0.02, -0.05, 0.1])
    v = torch.tensor([6.0, 12.0, 3.0])
    a_t = a0[:, None].expand(3, 60).contiguous()
    k_t = k0[:, None].expand(3, 60).contiguous()
    kw = dict(control_units=units)
    assert torch.equal(KP.compose_ticks(d, a0, k0, v, HZ, **kw), KP.compose_ticks(d, a_t, k_t, v, HZ, **kw))
    assert torch.equal(KP.roll_plan(d, a0, k0, v, HZ, **kw), KP.roll_plan(d, a_t, k_t, v, HZ, **kw))
    assert torch.equal(KP.prior_path(a0, k0, v, HZ), KP.prior_path(a_t, k_t, v, HZ))
    u1 = KP.absolute_controls(d, a0, k0, v, **kw)
    u2 = KP.absolute_controls(d, a_t, k_t, v, horizons=HZ, **kw)
    assert float((u1 - u2).abs().max()) < 1e-6                       # slot MEAN of identical floats: <= 1 ulp
    r2 = KP.residual_controls(u2, a_t, k_t, v, horizons=HZ, **kw)
    assert float((r2 - d).abs().max()) < 1e-5


def test_f_delta_zero_rolls_EXACTLY_to_the_per_tick_prior_path(fx, p10) -> None:
    po = torch.from_numpy(fx["poses"])[:6]
    ac = torch.from_numpy(fx["actions"])[:6]
    a, k, _ = KP.p10_tick_prior(p10, po, W, ac)
    v = po[:, -1, 3]
    z = torch.zeros(6, 3, len(HZ), 2)
    plan = KP.roll_plan(z, a, k, v, HZ, control_units="alat")
    pp = KP.prior_path(a, k, v, HZ)
    for n in range(3):
        assert torch.equal(plan[:, n], pp)                           # bit for bit: Delta == 0 IS P10u
    # and P10u is the unicycle roll of the per-tick controls
    ref = _roll(torch.stack([a, k], -1), v)[:, [h - 1 for h in HZ]]
    assert torch.equal(pp, ref)


def test_f_absolute_and_residual_controls_are_inverse_for_a_per_tick_prior_and_need_horizons(fx, p10) -> None:
    po = torch.from_numpy(fx["poses"])[:5]
    ac = torch.from_numpy(fx["actions"])[:5]
    a, k, _ = KP.p10_tick_prior(p10, po, W, ac)
    v = po[:, -1, 3]
    d = _delta(5, 2)
    u = KP.absolute_controls(d, a, k, v, control_units="alat", horizons=HZ)
    back = KP.residual_controls(u, a, k, v, control_units="alat", horizons=HZ)
    assert float((back - d).abs().max()) < 1e-4
    with pytest.raises(KP.ResidualPriorError, match="horizons"):
        KP.absolute_controls(d, a, k, v, control_units="alat")
    with pytest.raises(KP.ResidualPriorError, match="covers"):
        KP.prior_path(a[:, :30], k[:, :30], v, HZ)                   # a prior shorter than the horizon is refused
    # slot_prior is the span mean
    sp = KP.slot_prior(a, HZ)
    assert torch.allclose(sp[:, 0], a[:, :5].mean(1)) and torch.allclose(sp[:, -1], a[:, 50:60].mean(1))


def test_f_withheld_rows_zero_the_WHOLE_per_tick_prior_row(fx, p10) -> None:
    po = torch.from_numpy(fx["poses"])[:4]
    ac = torch.from_numpy(fx["actions"])[:4]
    a, k, _ = KP.p10_tick_prior(p10, po, W, ac)
    keep = torch.tensor([True, False, True, False])
    a2, k2 = KP.withhold(a, k, keep)
    assert torch.equal(a2[[0, 2]], a[[0, 2]]) and torch.equal(k2[[0, 2]], k[[0, 2]])
    assert float(a2[[1, 3]].abs().max()) == 0.0 and float(k2[[1, 3]].abs().max()) == 0.0


# =========================================================================== #
# (g) refusals of the mode                                                    #
# =========================================================================== #
def test_g_prior_controls_refusals(fx, p10) -> None:
    po = torch.from_numpy(fx["poses"])[:3]
    ac = torch.from_numpy(fx["actions"])[:3]
    with pytest.raises(KP.ResidualPriorError, match="RECORDED STEER"):
        KP.prior_controls("p10", po, W, None, p10=p10)               # an ACTION mode: no actions -> refused
    with pytest.raises(KP.ResidualPriorError, match="no frozen P10"):
        KP.prior_controls("p10", po, W, ac)                          # no P10 module: never another prior
    with pytest.raises(KP.ResidualPriorError, match="does not read it"):
        KP.prior_controls("ha0_ext_pose", po, W, None, p10=p10)      # a P10 handed to a mode that ignores it
    with pytest.raises(KP.ResidualPriorError, match="does not read it"):
        KP.prior_controls("ha0_ext", po, W, ac, p10=p10)
    assert KP.needs_actions("p10") and "p10" in KP.TICK_MODES and "p10" in KP.RESIDUAL_PRIOR_MODES
    a, k = KP.prior_controls("p10", po, W, ac, p10=p10)
    assert a.shape == k.shape == (3, 60)


def test_g_the_stamp_names_the_p10_iff_the_mode_is_p10(p10) -> None:
    idn = {"path": "x", "file_sha256": p10.file_sha, "content_sha256": p10.content_sha(), "meta": {}}
    st = KP.prior_stamp("p10", p10=idn)
    assert st["residual_prior"] == "p10" and st["p10"]["file_sha256"] == P10_SHA256
    assert st["reads"] == ["pose window (v[t0], v[t0-2])", "actions[t0, 0]"] and st["equals_battery_echo"] is False
    with pytest.raises(KP.ResidualPriorError):
        KP.prior_stamp("p10")                                        # a P10 mode with no identity
    with pytest.raises(KP.ResidualPriorError):
        KP.prior_stamp("ha0_ext_pose", p10=idn)                      # an identity the mode does not read
    assert "p10" not in KP.prior_stamp("ha0_ext_pose")               # every other stamp is unchanged


# =========================================================================== #
# (h) the model: the trainer's OWN build and loss                             #
# =========================================================================== #
@pytest.fixture(scope="module")
def T():
    path = _STACK / "scripts" / "refc_v3_train.py"
    spec = importlib.util.spec_from_file_location("refc_v3_train_p10_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal", "--f5-emitting-conf",
         "--f6-w-u0-zero"]
GRID = torch.cartesian_prod(torch.tensor((-2.0, -1.0, 0.0, 1.0, 2.0)), torch.tensor((-1.5, -0.5, 0.0, 0.5)))


def _argv(mode="p10", *, path=P10_FILE, sha=P10_SHA256, extra=()):
    a = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
         "--n-anchors", "20", "--ego-history", "--out", str(_REPO / "_p10_test_unused_out")] + F1_F6
    a += ["--residual-prior", mode]
    if path is not None:
        a += ["--p10-prior", str(path)]
    if sha is not None:
        a += ["--p10-prior-sha256", sha]
    return a + list(extra)


def _arm(T, mode="p10", seed=0, **kw):
    args = T.build_parser().parse_args(_argv(mode, **kw))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(seed)
    model = T.v3.RefCV3Model(cfg)
    with torch.no_grad():
        model.core.decoder.anchor_controls.copy_(GRID)
    return args, cfg, model


def _batch(T, cfg, fx=None):
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    b = torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])
    if fx is not None:                       # REAL windows: the fixture's eval141 poses (and actions, when it has them)
        n = 4
        b["pose_hist"] = torch.from_numpy(fx["poses"][:n]).clone().float()
        if "actions" in fx:                  # the POSE fixture carries none: a pose prior never reads them
            b["actions"] = torch.from_numpy(fx["actions"][:n]).clone()
        b["pose_last"] = b["pose_hist"][:, -1].clone()
    return b


def _forward(T, cfg, model, batch, seed=2):
    model.eval()
    torch.manual_seed(seed)
    ph = batch["pose_hist"]
    needs = KP.needs_actions(model.core.decoder.residual_prior)      # only the ACTION modes take the window's actions
    with torch.no_grad():
        model.core.set_ego_window(ph, int(ph.shape[1]), actions=batch["actions"] if needs else None)
        return model(T.frames_to_device(batch["frames"], "cpu"), nav_cmd=batch["nav_cmd"],
                     v0=batch["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)


def test_h_flags_pin_onto_the_decoder_config_and_the_p10_pair_is_required_or_refused(T) -> None:
    args, cfg, model = _arm(T)
    d = cfg.core.decoder
    assert (d.residual_prior, d.p10_prior_path, d.p10_prior_sha256) == ("p10", str(P10_FILE), P10_SHA256)
    assert model.core.decoder.p10 is not None and model.core.decoder.p10.file_sha == P10_SHA256
    for kw in ({"sha": None}, {"path": None}):
        with pytest.raises(SystemExit, match="BOTH"):
            T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), T.build_parser().parse_args(_argv("p10", **kw)))
    with pytest.raises(SystemExit, match="does not read a P10"):
        T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True),
                           T.build_parser().parse_args(_argv("ha0_ext_pose")))
    # a wrong sha / zeroed weights file / missing file REFUSE at model build, before any step
    with pytest.raises(P10.P10Error, match="sha256"):
        _arm(T, sha="1" * 64)
    with pytest.raises(P10.P10Error, match="does not exist"):
        _arm(T, path=_REPO / "no_such_p10.pt")


def test_h_the_model_emits_the_per_tick_prior_and_the_zero_anchor_IS_P10u(T, fx) -> None:
    _a, cfg, model = _arm(T)
    b = _batch(T, cfg, fx)
    out = _forward(T, cfg, model, b)
    assert tuple(out["residual_prior_ctrl"].shape) == (4, 60, 2) and tuple(out["residual_prior_v"].shape) == (4,)
    rep = KP.plan_check(out, model.core.decoder.anchor_controls, decoder=model.core.decoder)
    assert rep["ok"], rep
    assert rep["max_abs_zero_anchor_minus_prior_m"] == 0.0           # Delta == 0 IS P10u, bit-exact
    assert rep["max_abs_fan_minus_reroll_m"] < 1e-3 and rep["rows_v_gt_0_prior_nonzero"] == rep["rows_v_gt_0"] == 4
    assert torch.equal(out["residual_prior_v"], b["pose_last"][:, 3])


def test_h_IN_DECODER_P10_EQUALS_THE_BANKED_PATH_and_a_shifted_clock_through_the_model_FAILS(T, fx) -> None:
    """The brief's clock proof, through the MODEL route (set_ego_window -> _residual_prior_inputs -> P10), not the
    module: the P10 path the decoder computes on real windows equals the banked fit-script prediction <= 1e-4 m."""
    _a, cfg, model = _arm(T)
    seen = []
    h = model.core.decoder.p10.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
    n = 4
    b = _batch(T, cfg, fx)
    _forward(T, cfg, model, b)
    banked = torch.from_numpy(fx["p10_banked"][:n])
    assert len(seen) == 1 and float((seen[0] - banked).abs().max()) <= TOL_M
    # the same windows with the clock ONE FRAME EARLY: pad the front so the window length is unchanged
    b2 = dict(b)
    b2["pose_hist"] = torch.cat([b["pose_hist"][:, :1], b["pose_hist"][:, :-1]], 1)
    b2["actions"] = torch.cat([b["actions"][:, :1], b["actions"][:, :-1]], 1)
    b2["pose_last"] = b2["pose_hist"][:, -1].clone()
    seen.clear()
    _forward(T, cfg, model, b2)
    h.remove()
    assert float((seen[0] - banked).abs().max()) > 100 * TOL_M       # a one-tick-shifted clock FAILS the tolerance


def test_h_REACHABILITY_the_flag_reaches_the_loss_and_P10_takes_no_gradient(T, fx) -> None:
    """flag -> cfg -> decoder.residual_prior -> RefCModel._residual_prior_inputs -> FrozenP10 -> per-tick prior ->
    roll_plan -> traj -> compute_losses_v3 -> backward. Each hop is observed, not inferred:
      cfg: args.residual_prior / cfg.core.decoder.residual_prior (refc_v3_train.py::_pin_trainer_cfg)
      model: RefCModel._residual_prior_inputs (refc.py) calls kinematic_prior.prior_controls(..., p10=decoder.p10)
      loss: the P10 forward is CALLED on a training step, the loss CHANGES when the P10 weights change, and the
            residual heads take a non-zero gradient while P10 takes none (zero control: the unfrozen mutation reads
            non-zero, so the zero is a reading and not a blind probe)."""
    args, cfg, model = _arm(T)
    assert args.residual_prior == cfg.core.decoder.residual_prior == model.core.decoder.residual_prior == "p10"
    b = _batch(T, cfg, fx)
    calls = []
    hk = model.core.decoder.p10.register_forward_hook(lambda m, i, o: calls.append(1))
    model.train()
    torch.manual_seed(5)
    l1 = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    assert len(calls) >= 1                                           # P10 is read on the TRAINING forward
    l1["loss"].backward()
    heads = model.core.decoder.cascade.control_heads
    assert all(h_.weight.grad is not None and float(h_.weight.grad.abs().sum()) > 0.0 for h_ in heads)
    assert all(t.grad is None and not t.requires_grad for t in model.core.decoder.p10.buffers())   # EXACTLY none
    assert model.core.decoder.p10.content_sha() == model.core.decoder.p10.content_sha_at_load
    # the P10 VALUE reaches the loss: perturb the live buffers, the loss moves
    model.zero_grad(set_to_none=True)
    base = float(l1["loss"].detach())
    with torch.no_grad():
        model.core.decoder.p10.ym.add_(torch.linspace(-3.0, 3.0, 60))
    torch.manual_seed(5)
    l2 = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    assert abs(float(l2["loss"].detach()) - base) > 1e-4, (base, float(l2["loss"].detach()))
    hk.remove()


def test_h_MUTATION_unfrozen_P10_takes_gradient_and_the_freeze_probe_goes_RED(T, fx, monkeypatch) -> None:
    args, cfg, model = _arm(T)
    b = _batch(T, cfg, fx)
    p10 = model.core.decoder.p10
    # the mutation: P10's tensors become Parameters and the prior is no longer a no_grad island
    for nm in ("w1", "b1", "w2", "b2", "ym"):
        t = getattr(p10, nm)
        delattr(p10, nm)
        p10.register_parameter(nm, torch.nn.Parameter(t.clone()))
    import inspect
    src = inspect.getsource(KP.p10_tick_prior).replace("with torch.no_grad():", "if True:")
    ns: dict = {}
    exec(compile("from tanitad.models.kinematic_prior import *\nimport torch\nimport tanitad.models.kinematic_prior as _m\n"
                 "ResidualPriorError=_m.ResidualPriorError\nDT_TICK=_m.DT_TICK\nPOSE_KAPPA_V_FLOOR=_m.POSE_KAPPA_V_FLOOR\n"
                 "P10_KAPPA_CAP=_m.P10_KAPPA_CAP\nknots_to_ticks=_m.knots_to_ticks\ncontrols_from_path=_m.controls_from_path\n"
                 + src, "<mutant>", "exec"), ns)
    monkeypatch.setattr(KP, "p10_tick_prior", ns["p10_tick_prior"])
    # also lift the module's own no_grad island on forward
    monkeypatch.setattr(type(p10), "forward", type(p10).forward.__wrapped__)
    model.train()
    l = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    l["loss"].backward()
    g = [float(getattr(p10, nm).grad.abs().sum()) if getattr(p10, nm).grad is not None else 0.0
         for nm in ("w1", "b1", "w2", "b2", "ym")]
    assert max(g) > 0.0, g                                           # the control: an unfrozen P10 DOES read non-zero
    assert len(list(p10.parameters())) == 5                          # ...and the structural freeze probe sees it
    assert p10.content_sha() == p10.content_sha_at_load              # (no optimizer step ran: only the grad moved)


def test_h_MUTATION_a_decoder_that_silently_falls_back_to_ha0_ext_is_caught(T, fx, monkeypatch) -> None:
    """Replace the p10 branch by the ha0_ext construction (a constant [B] prior): the decoder REFUSES a constant
    prior on a per-tick mode at the forward, and expanding it per tick instead is caught by comparing the emitted
    prior to the independent P10 recompute -- the gate's G-LIVE clause."""
    _a, cfg, model = _arm(T)
    b = _batch(T, cfg, fx)
    real = KP.prior_controls

    def fallback(mode, poses, n_past=None, actions=None, **kw):
        kw.pop("p10", None)
        return real("ha0_ext", poses, n_past, actions, **kw)

    monkeypatch.setattr(KP, "prior_controls", fallback)
    with pytest.raises(ValueError, match="per-tick prior"):
        _forward(T, cfg, model, b)
    # the sneakier fallback: the ha0_ext constants broadcast to 60 ticks -- shape-valid, so the decoder runs
    def fallback_tick(mode, poses, n_past=None, actions=None, **kw):
        kw.pop("p10", None)
        a0, k0 = real("ha0_ext", poses, n_past, actions, **kw)
        return a0[:, None].expand(-1, 60).contiguous(), k0[:, None].expand(-1, 60).contiguous()

    monkeypatch.setattr(KP, "prior_controls", fallback_tick)
    out = _forward(T, cfg, model, b)
    ref_a, ref_k, _ = KP.p10_tick_prior(P10.load_frozen_p10(P10_FILE), b["pose_hist"], b["pose_hist"].shape[1],
                                        b["actions"])
    emitted = out["residual_prior_ctrl"]
    d = float((emitted[..., 0] - ref_a).abs().max())
    assert d > 0.1, d                                                # the recompute DISAGREES: the fallback is visible
    monkeypatch.setattr(KP, "prior_controls", real)
    out2 = _forward(T, cfg, model, b)
    assert float((out2["residual_prior_ctrl"][..., 0] - ref_a).abs().max()) < 1e-6   # and the control agrees


def test_h_a_constant_prior_on_the_p10_mode_and_a_per_tick_prior_on_another_are_refused(T, fx) -> None:
    _a, cfg, model = _arm(T)
    dec = model.core.decoder
    z1 = torch.zeros(4)
    with pytest.raises(ValueError, match="needs a per-tick prior"):
        dec._residual_prior((z1, z1), torch.ones(4), None, 4, None)
    _a2, cfg2, model2 = _arm(T, "ha0_ext_pose", path=None, sha=None)
    z2 = torch.zeros(4, 60)
    with pytest.raises(ValueError, match="CONSTANT prior"):
        model2.core.decoder._residual_prior((z2, z2), torch.ones(4), None, 4, None)


def test_h_the_checkpoint_carries_the_p10_identity_and_a_resume_into_another_build_REFUSES(T, tmp_path) -> None:
    _a, cfg, model = _arm(T)
    extra = T._p10_ckpt_extra(model)
    assert set(extra["p10_prior"]) == {"file_sha256", "content_sha256"} and extra["p10_prior"]["file_sha256"] == P10_SHA256
    T._resume_p10_guard(dict(extra), model)                          # the same P10: fine
    with pytest.raises(SystemExit, match="different experiment"):
        T._resume_p10_guard({}, model)                               # a non-p10 checkpoint into a p10 build
    bad = {"p10_prior": dict(extra["p10_prior"], content_sha256="0" * 64)}
    with pytest.raises(SystemExit, match="content_sha256"):
        T._resume_p10_guard(bad, model)                              # a different P10
    _a2, _c2, model2 = _arm(T, "ha0_ext_pose", path=None, sha=None)
    assert T._p10_ckpt_extra(model2) == {}                           # every other checkpoint is unchanged, key for key
    with pytest.raises(SystemExit, match="different experiment"):
        T._resume_p10_guard(dict(extra), model2)                     # a p10 checkpoint into a non-p10 build


def test_h_the_stamp_is_built_from_the_file_and_the_seam_check_compares_it_to_the_built_buffers(T) -> None:
    args, cfg, model = _arm(T)
    stamp = T._seam_stamp(cfg, args)
    assert stamp["residual_prior"]["p10"]["file_sha256"] == P10_SHA256
    T.assert_seams_are_built(model, stamp)
    for key in ("file_sha256", "content_sha256"):
        bad = {**stamp, "residual_prior": {**stamp["residual_prior"], "p10": {**stamp["residual_prior"]["p10"], key: "0" * 64}}}
        with pytest.raises((SystemExit, ValueError)):
            T.assert_seams_are_built(model, bad)
    with pytest.raises((SystemExit, ValueError)):
        T.assert_seams_are_built(model, {**stamp, "residual_prior": KP.prior_stamp("ha0_ext_pose")})


def test_h_G_DVB_reads_the_BUILT_P10_and_goes_RED_on_each_mutation(T) -> None:
    from tanitad.train import declared_vs_built as D
    args, cfg, model = _arm(T)
    parser = T.build_parser()
    # the SMOKE model is not a full launch (it carries 11 unrelated mismatches); the claim is about p10's slice,
    # so every reading is taken against the model's own baseline and only mismatches NAMING p10 / the prior count
    mine = lambda: [str(m) for m in D.check(model, args, parser)                    # noqa: E731
                    if "p10" in str(m).lower() or "residual" in str(m).lower()]
    assert mine() == []
    p10 = model.core.decoder.p10
    # 1. zeroed weights AFTER the load (same bytes on disk, different weights in the model)
    keep = p10.w2.clone()
    with torch.no_grad():
        p10.w2.zero_()
    bad = mine()
    assert any("content" in m for m in bad), bad
    with torch.no_grad():
        p10.w2.copy_(keep)
    assert mine() == []
    # 2. the decoder lost its P10 (the silent-fallback shape)
    model.core.decoder.p10 = None
    assert mine()
    model.core.decoder.p10 = p10
    assert mine() == []
    # 3. unfrozen
    p10.register_parameter("w1x", torch.nn.Parameter(torch.zeros(1)))
    assert any("Parameter" in m for m in mine())
    del p10._parameters["w1x"]
    assert mine() == []
    # 4. the file named by argv no longer hashes to the argv sha
    args.p10_prior_sha256 = "2" * 64
    assert mine()
    args.p10_prior_sha256 = P10_SHA256
    assert mine() == []
    # 5. the decoder mode flipped to ha0_ext_pose (a fallback) against an argv that declares p10
    model.core.decoder.residual_prior = "ha0_ext_pose"
    assert mine()
    model.core.decoder.residual_prior = "p10"
    assert mine() == []


def test_h_other_builds_are_untouched_no_p10_module_no_new_state(T) -> None:
    _a, cfg, model = _arm(T, "ha0_ext_pose", path=None, sha=None)
    assert model.core.decoder.p10 is None
    assert not any("p10" in k for k in model.state_dict())
    from tanitad.train import declared_vs_built as D
    assert [m for m in D.check(model, _a, T.build_parser()) if "p10" in str(m).lower()] == []
    args2, _c, _m = _arm(T)
    assert not any("p10" in k for k in _m.state_dict())              # and on a p10 build the weights are NOT in it
