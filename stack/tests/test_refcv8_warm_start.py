"""refcv8 WP-B -- the WARM-START IDENTITY and the wiring of the tactical conditioning on a REAL RefCV3Model
(`_refcv8_rig`: DDIM control-space sampler, residual prior, F1-F6, behaviour decoder over agents + BEV).

The binding claim (PLAN_REFCV8 sec. 0.1, DESIGN.md sec. 3.7): a refcv8 build loaded from a refcv7 checkpoint
reproduces refcv7's forward BIT FOR BIT at step 0 -- with every refcv8 switch ON. Each identity test has a
DELIBERATE-REGRESSION twin that moves one new gate and must go RED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402

KEYS = ("traj", "sel_idx", "anchor_traj", "sel_score", "sel_score_v3", "anchor_logits", "u0_hat", "traj_base",
        "sel_idx_base")


@pytest.fixture(scope="module")
def T():
    return R.trainer()


@pytest.fixture(scope="module")
def base(T):
    cfg7, m7 = R.build(T, False)
    bt = R.batch(cfg7)
    return cfg7, m7, bt, R.forward(cfg7, m7, bt)


def _assert_same(o7, o8, nb=None):
    for k in KEYS:
        a, b = o7[k], o8[k]
        if k in ("anchor_traj", "sel_score", "sel_score_v3", "u0_hat") and nb is not None:
            b = b[:, :nb]                     # the candidate axis: base slice; the extras are appended after it
        assert torch.equal(a, b), k


def test_seams_on_reproduce_refcv7_bit_for_bit(T, base) -> None:
    cfg7, m7, bt, o7 = base
    cfg8, m8 = R.build(T, True)
    new = R.copy_into(m7, m8)
    assert new and all(".r8_" in k or k.startswith("tac_decoder_v6.r8_") for k in new), new
    o8 = R.forward(cfg8, m8, bt)
    _assert_same(o7, o8)
    assert int(o8["r8_n_base"]) == o8["anchor_traj"].shape[1]       # no extension on this build


def test_allocation_and_prior_free_group_leave_the_emitted_plan_and_the_base_fan_unchanged(T, base) -> None:
    cfg7, m7, bt, o7 = base
    cfg8, m8 = R.build(T, True, n_alloc=8, prior_free_group=True)
    R.copy_into(m7, m8)
    o8 = R.forward(cfg8, m8, bt)
    nb = int(o8["r8_n_base"])
    assert o8["anchor_traj"].shape[1] == nb + 8 + nb
    # the GEOMETRY and the EMITTED plan are bit-identical; the base candidates' SCORES are equal to float rounding
    # only (MEASURED 9.5e-7: the score heads run one GEMM over N + M + N rows instead of N, so the blocking differs)
    # -- stated, not hidden. The no-extension build above is the bitwise warm-start identity.
    for k in ("traj", "sel_idx", "traj_base", "sel_idx_base"):
        assert torch.equal(o7[k], o8[k]), k
    assert torch.equal(o7["anchor_traj"], o8["anchor_traj"][:, :nb])
    assert torch.equal(o7["u0_hat"], o8["u0_hat"][:, :nb])
    for k in ("sel_score", "sel_score_v3"):
        assert float((o7[k] - o8[k][:, :nb]).abs().max()) <= 1e-5, k
    kind = o8["r8_kind"].tolist()
    assert kind == [0] * nb + [1] * 8 + [2] * nb
    assert not bool(o8["r8_emit_keep"][:, nb:].any())               # trained, never emitted (flags off)


def test_deliberate_regression_one_gate_moved_goes_red(T, base) -> None:
    cfg7, m7, bt, o7 = base
    cfg8, m8 = R.build(T, True)
    R.copy_into(m7, m8)
    with torch.no_grad():
        m8.core.decoder.r8_mod.proj[-1].bias.fill_(0.5)
    o8 = R.forward(cfg8, m8, bt)
    with pytest.raises(AssertionError):
        _assert_same(o7, o8)
    cfg9, m9 = R.build(T, True)
    R.copy_into(m7, m9)
    with torch.no_grad():
        m9.core.decoder.r8_sel.beta_lat.fill_(50.0)
    o9 = R.forward(cfg9, m9, bt)
    assert not torch.equal(o7["sel_score_v3"], o9["sel_score_v3"])


def test_training_draws_never_touch_the_global_stream(T, base) -> None:
    """Every refcv8 training-time draw (cond dropout, prior dropout, allocation choice, extra-candidate noise) uses
    the DEDICATED generator: after one TRAINING forward the global RNG state equals refcv7's."""
    # a FRESH refcv7 build: a training forward moves BN statistics, which must not leak into the shared fixture
    _c, bt = None, base[2]
    cfg7, m7 = R.build(T, False)
    cfg8, m8 = R.build(T, True, n_alloc=8, prior_free_group=True, lat_prior_dropout=0.5, cond_dropout=0.5)
    R.copy_into(m7, m8)
    R.forward(cfg7, m7, bt, seed=5, train=True)
    s7 = torch.get_rng_state()
    R.forward(cfg8, m8, bt, seed=5, train=True)
    s8 = torch.get_rng_state()
    assert torch.equal(s7, s8)


def test_forced_class_moves_the_allocated_fan_and_the_score(T, base) -> None:
    cfg7, m7, bt, _ = base
    cfg8, m8 = R.build(T, True, n_alloc=8)
    R.copy_into(m7, m8)
    m8.eval()
    m8.set_r8_force(lat3=C.LAT3.index("TURN_L"))
    of = R.forward(cfg8, m8, bt)
    hl, _ = C.split_joint(of["r8_hyp_alloc"])
    assert bool((hl == C.LAT3.index("TURN_L")).all())
    m8.set_r8_force(lat3=C.LAT3.index("TURN_R"))
    og = R.forward(cfg8, m8, bt)
    hr, _ = C.split_joint(og["r8_hyp_alloc"])
    assert bool((hr == C.LAT3.index("TURN_R")).all())
    # a forced STOP-at-d imposes ONE joint hypothesis with its constraint on every allocated candidate
    d = 12.0
    cons = torch.tensor([[0.0, 0.0, float(torch.log1p(torch.tensor(d)) / C.PROG_LOG_SCALE), 0.0]] * 3)
    m8.set_r8_force(lat3=0, lon6=C.LON6.index("BRAKE_TO"), cons=cons)
    os_ = R.forward(cfg8, m8, bt)
    want = int(C.joint_id(torch.tensor(0), torch.tensor(C.LON6.index("BRAKE_TO"))))
    assert bool((os_["r8_hyp_alloc"] == want).all())
    nb = int(os_["r8_n_base"])
    phi_c = C.denorm_cons(os_["r8_phi"][:, nb:, 11:15])
    assert torch.allclose(phi_c[..., 2], torch.full_like(phi_c[..., 2], d), atol=1e-4)
    with pytest.raises(ValueError):
        m8.train()
        m8.set_r8_force(lat3=0)


def test_inputs_without_seams_are_refused_not_dropped(T, base) -> None:
    cfg7, m7, bt, _ = base
    with pytest.raises(ValueError):
        R.forward(cfg7, m7, bt, r8_rc=torch.zeros(3, 4))
    with pytest.raises(ValueError):
        m7.tac_decoder_v6(torch.zeros(1, 10), agent_tokens=torch.zeros(1, 2, int(cfg7.core.decoder.d)),
                          cond_extra=torch.zeros(1, 10))


def test_the_route_checkpoint_path_carries_nothing_from_the_tactical_layer(T, base) -> None:
    """Admissibility (PI 2026-08-03): the RC reaching the operative condition is bit-identical whatever the tactical
    decoder emits -- perturbing every tactical output (a forced posterior) leaves it unchanged."""
    cfg7, m7, bt, _ = base
    cfg8, m8 = R.build(T, True, n_alloc=4)
    R.copy_into(m7, m8)
    seen = []
    m8.core.decoder.r8_rc_to_cond.register_forward_pre_hook(lambda _m, a: seen.append(a[0].detach().clone()))
    rc = torch.tensor([[0.9, 0.3, 1.0, 1.0], [0.8, -0.4, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    R.forward(cfg8, m8, bt, r8_rc=rc)
    m8.eval()
    m8.set_r8_force(lat3=C.LAT3.index("TURN_R"), lon6=C.LON6.index("HOLD"))
    R.forward(cfg8, m8, bt, r8_rc=rc)
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])
    assert torch.equal(seen[0][2], torch.zeros(4))                  # an invalid row is exactly zeros


def test_new_parameters_take_gradient_through_the_trainer_side_losses(T, base) -> None:
    from tanitad.train import refcv8_train as RT
    cfg7, m7, bt, _ = base
    cfg8, m8 = R.build(T, True, n_alloc=8, w_sat=0.1, w_listwise=1.0, w_subscore=0.5)
    R.copy_into(m7, m8)
    m8.train()
    b = bt["frames"].shape[0]
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    sv = torch.ones(b, 8, dtype=torch.bool)
    fut = torch.zeros(b, 60, 4)
    fut[:, :, 3] = 6.0
    fut[:, :, 2] = torch.linspace(0, 0.8, 60)
    m8._r8_tf_ratio = 1.0
    prep = RT.r8_before_forward(m8, {}, "cpu", traj, sv, bt["pose_hist"][:, -1], fut,
                                torch.ones(b, 60, dtype=torch.bool), bt["v0"])
    out = R.forward(cfg8, m8, bt, train=True)
    loss, tele = RT.r8_losses(m8, out, prep, traj, sv)
    assert torch.isfinite(loss)
    for k in ("r8_cons", "r8_alloc_l1", "r8_sat", "r8_listwise", "r8_subscore"):
        assert k in tele, k
    loss.backward()
    dec = m8.core.decoder
    for name, p in (("mod_last", dec.r8_mod.proj[-1].weight), ("beta_lat", dec.r8_sel.beta_lat),
                    ("gamma_prog", dec.r8_sel.gamma_prog), ("sub_w", dec.r8_sub.w),
                    ("cons_lat", m8.tac_decoder_v6.r8_cons_lat.weight),
                    ("cons_lon", m8.tac_decoder_v6.r8_cons_lon.weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name


def test_post_hoc_enable_on_a_loaded_refcv7_model_is_the_identity(T, base) -> None:
    """The warm start as the harness / branch run performs it: build refcv7, load its weights, THEN attach the seams."""
    from tanitad.refs import refcv8_conditioning as C8
    # FRESH builds (the module fixture's model has had TRAINING forwards in other tests -- its BN statistics moved)
    cfg7, m7 = R.build(T, False)
    bt = R.batch(cfg7)
    o7 = R.forward(cfg7, m7, bt)
    _cfg, m = R.build(T, False)                                   # a second refcv7 build ...
    R.copy_into(m7, m)                                            # ... loaded with refcv7's weights
    n = m.enable_refcv8(C8.R8Config(n_alloc=0))
    assert n > 0 and m.r8_enabled
    o = R.forward(cfg7, m, bt)
    _assert_same(o7, o)
    with pytest.raises(ValueError):
        m.enable_refcv8(C8.R8Config())


def test_derange_diagnostic_feeds_another_windows_tactical_output(T, base) -> None:
    cfg7, m7, bt, _ = base
    cfg8, m8 = R.build(T, True, n_alloc=4)
    R.copy_into(m7, m8)
    with torch.no_grad():
        m8.core.decoder.r8_mod.proj[-1].weight.normal_(0.0, 0.1)    # make phi matter
    a = R.forward(cfg8, m8, bt)
    m8._r8_derange_feed = True
    b = R.forward(cfg8, m8, bt)
    m8._r8_derange_feed = False
    assert not torch.equal(a["r8_phi"], b["r8_phi"])
    assert torch.equal(a["r8_lat3"][:, :int(a["r8_n_base"])], b["r8_lat3"][:, :int(b["r8_n_base"])])  # tags = geometry
