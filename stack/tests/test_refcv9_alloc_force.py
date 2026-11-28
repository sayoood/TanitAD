"""refcv9 ``--r9-alloc-force-schedule`` (H-R9-ALLOC-ORACLE-1): the GT joint hypothesis is forced into a training
window's allocated pool with a SCHEDULED probability, so training can see the predicted-only pool inference sees.

⛔ Expectations are LITERALS (or the INFERENCE-time call ``allocate(..., force=None)`` -- the pool this flag exists to
reproduce), never an expression over the wiring under test. Two mutation arms re-force the GT inside the flag-on path
and must turn the const:0.0 check RED.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refc as RC  # noqa: E402
from tanitad.refs import refcv8_conditioning as C8  # noqa: E402
from tanitad.train import refcv9_alloc_force as AF  # noqa: E402

N_ALLOC = 8


@pytest.fixture(scope="module")
def T():
    return R.trainer()


# =========================================================================== #
# 1. the schedule -- parsing and refusals                                     #
# =========================================================================== #
def test_const_schedule_literals():
    s = AF.parse_schedule("const:1.0")
    assert (s.kind, s.p0, s.p1) == ("const", 1.0, 1.0)
    assert s.p_at(0) == 1.0 and s.p_at(48852) == 1.0
    s0 = AF.parse_schedule("const:0.0")
    assert s0.p_at(0) == 0.0 and s0.p_at(10 ** 7) == 0.0
    assert AF.parse_schedule("const:0.25").p_at(123) == 0.25


def test_linear_schedule_literals():
    s = AF.parse_schedule("linear:1.0:0.0:100:300")
    assert (s.kind, s.p0, s.p1, s.start, s.end) == ("linear", 1.0, 0.0, 100, 300)
    assert s.p_at(0) == 1.0
    assert s.p_at(100) == 1.0
    assert s.p_at(200) == 0.5
    assert s.p_at(250) == 0.25
    assert s.p_at(300) == 0.0
    assert s.p_at(5000) == 0.0
    s2 = AF.parse_schedule("linear:0.2:0.8:0:10")
    assert abs(s2.p_at(5) - 0.5) < 1e-12 and s2.p_at(0) == 0.2 and s2.p_at(10) == 0.8


@pytest.mark.parametrize("spec", [
    "", "const", "const:", "const:1.5", "const:-0.1", "const:nan", "const:inf", "const:abc", "const:1.0:2",
    " const:1.0", "const: 1.0", "CONST:1.0", "cosine:1:0:0:10", "linear:1:0:0", "linear:1:0:0:10:20",
    "linear:1:0:300:100", "linear:1:0:100:100", "linear:1:0:-1:10", "linear:1:0:1.5:10", "linear:1:0:1e3:2000",
    "linear:2:0:0:10", "linear:1:-1:0:10", None, 1.0,
])
def test_malformed_specs_are_refused(spec):
    with pytest.raises(ValueError):
        AF.parse_schedule(spec)


def _args(**kw):
    a = dict(r9_alloc_force_schedule=None, r9_alloc_force_seed=AF.DEFAULT_SEED, refcv8=True, r8_n_alloc=N_ALLOC,
             r9_calib_stage=False, steps=10)
    a.update(kw)
    return SimpleNamespace(**a)


def test_pin_args_refusals():
    assert AF.pin_args(_args()) is None                                           # absent: nothing
    assert AF.pin_args(_args(r9_alloc_force_schedule="const:0.0")).spec == "const:0.0"
    for bad in (dict(r9_alloc_force_schedule="const:2"),                        # malformed
                dict(r9_alloc_force_schedule="const:0.0", refcv8=False),        # no refcv8 seams
                dict(r9_alloc_force_schedule="const:0.0", r8_n_alloc=0),        # no allocated pool
                dict(r9_alloc_force_schedule="const:0.0", r9_calib_stage=True),  # the inference path: dead
                dict(r9_alloc_force_seed=7)):                                    # a seed that seeds nothing
        with pytest.raises(SystemExit):
            AF.pin_args(_args(**bad))


def test_trainer_argv_refuses_a_malformed_schedule(T):
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "_unused", "--refcv8", "--r8-n-alloc", "8",
                                        "--r9-alloc-force-schedule", "linear:1:0:10:5"])
    with pytest.raises(SystemExit):
        T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    absent = T.build_parser().parse_args(["--arm", "hier", "--out", "_unused"])
    assert absent.r9_alloc_force_schedule is None and absent.r9_alloc_force_seed == 20261027


# =========================================================================== #
# 2. the coin stream -- exact at p in {0, 1}, private, a pure function        #
# =========================================================================== #
def test_p1_forces_every_window_and_p0_none():
    hyp = torch.zeros(257, dtype=torch.long)
    assert AF.PoolForce(AF.parse_schedule("const:1.0")).mask(hyp).tolist() == [True] * 257
    assert AF.PoolForce(AF.parse_schedule("const:0.0")).mask(hyp).tolist() == [False] * 257


def test_coins_are_a_pure_function_of_seed_and_step():
    pf = AF.PoolForce(AF.parse_schedule("const:0.5"))
    pf.step = 17
    a, b = pf.coins(64), pf.coins(64)
    assert torch.equal(a, b)                                       # the same step twice: the same coins
    pf.step = 18
    assert not torch.equal(a, pf.coins(64))                        # another step: other coins
    pf2 = AF.PoolForce(AF.parse_schedule("const:0.5"), seed=1)
    pf2.step = 17
    assert not torch.equal(a, pf2.coins(64))                       # another seed: other coins
    pf.step = 3
    m = pf.mask(torch.zeros(4096, dtype=torch.long)).to(torch.float64).mean().item()
    assert 0.45 < m < 0.55


def test_the_private_stream_moves_no_other_draw(T):
    cfg, m = R.build(T, True, n_alloc=N_ALLOC)
    gen = m.core.decoder.r8_gen
    g = gen.get("cpu")
    s_glob, s_r8 = torch.get_rng_state().clone(), g.get_state().clone()
    pf = AF.PoolForce(AF.parse_schedule("const:0.5"))
    for st in range(5):
        pf.step = st
        pf.mask(torch.zeros(9, dtype=torch.long))
    assert torch.equal(torch.get_rng_state(), s_glob)
    assert torch.equal(g.get_state(), s_r8)


# =========================================================================== #
# 3. the wiring -- through the REAL training forward (allocate is spied)      #
# =========================================================================== #
def _targets(b: int):
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    sv = torch.ones(b, 8, dtype=torch.bool)
    fut = torch.zeros(b, 60, 4)
    fut[:, :, 3] = 6.0
    fut[:, :, 2] = torch.linspace(0, 0.8, 60)
    return traj, sv, fut, torch.ones(b, 60, dtype=torch.bool)


def _spy(monkeypatch, crafted_gt_floor: bool = False):
    """Record every `allocate` call the decoder makes. ``crafted_gt_floor``: hand allocate a posterior in which the
    window's GT hypothesis has probability ~0 (so the PREDICTED pool cannot hold it)."""
    real = C8.allocate
    calls = []

    def spy(logp_joint, m_total, top_k=4, min_each=2, force=None, force_min=4):
        lp = logp_joint
        if crafted_gt_floor and _spy.gt is not None:
            lp = torch.zeros_like(logp_joint)
            lp[torch.arange(lp.shape[0]), _spy.gt.clamp_min(0)] = -60.0
            lp = torch.log_softmax(lp, -1)
        out = real(lp, m_total, top_k, min_each, force=force, force_min=force_min)
        calls.append({"logp": lp.detach().clone(), "m": int(m_total), "k": int(top_k), "min_each": int(min_each),
                      "force": None if force is None else force.detach().clone(), "force_min": int(force_min),
                      "out": out.clone()})
        return out
    monkeypatch.setattr(C8, "allocate", spy)
    return calls, real


_spy.gt = None


def _run(T, monkeypatch, spec, crafted=False, perturb_cons=False, pool_override=None):
    from tanitad.train import refcv8_train as RT
    cfg, m = R.build(T, True, n_alloc=N_ALLOC)
    bt = R.batch(cfg)
    b = bt["frames"].shape[0]
    if spec is not None:
        rec = AF.attach(m, _args(r9_alloc_force_schedule=spec))
        assert rec["spec"] == spec
    m.train()
    m._r8_tf_ratio = 1.0                                           # every window teacher-forces its GT constraint
    traj, sv, fut, fv = _targets(b)
    prep = RT.r8_before_forward(m, {}, "cpu", traj, sv, bt["pose_hist"][:, -1], fut, fv, bt["v0"])
    if perturb_cons:
        m._r8_teacher["cons"] = m._r8_teacher["cons"] + 0.75
    if pool_override is not None:
        m._r8_teacher["pool"] = torch.tensor(pool_override)
    _spy.gt = prep["gt_hyp"]
    calls, real = _spy(monkeypatch, crafted_gt_floor=crafted)
    out = R.forward(cfg, m, bt, train=True)
    _spy.gt = None
    monkeypatch.setattr(C8, "allocate", real)
    assert len(calls) == 1, len(calls)
    return prep, calls[0], out


def _gt_rows(prep):
    g = prep["gt_hyp"].tolist()
    assert all(x >= 0 for x in g), g                                # NOT VACUOUS: every rig window has a GT hypothesis
    return g


def _assert_const0_pool_is_predicted_only(T, monkeypatch, crafted: bool):
    prep, c, out = _run(T, monkeypatch, "const:0.0", crafted=crafted)
    g = _gt_rows(prep)
    assert c["force"] is not None and c["force"].tolist() == [-1, -1, -1], c["force"]
    # the pool IS the inference-time pool: allocate on the same posterior with NO force
    want = C8.allocate(c["logp"], c["m"], c["k"], c["min_each"], force=None)
    assert torch.equal(c["out"], want)
    assert torch.equal(out["r8_hyp_alloc"], want)
    if crafted:                                                     # the posterior gives GT ~0: GT is NOT in the pool
        assert [int((out["r8_hyp_alloc"][r] == g[r]).sum()) for r in range(3)] == [0, 0, 0]
    return prep, c, out


def test_const0_the_pool_is_the_predicted_posteriors_alone(T, monkeypatch):
    _assert_const0_pool_is_predicted_only(T, monkeypatch, crafted=False)


def test_const0_a_gt_the_posterior_does_not_allocate_is_absent(T, monkeypatch):
    _assert_const0_pool_is_predicted_only(T, monkeypatch, crafted=True)


def test_const1_forces_the_gt_into_every_pool_exactly_as_refcv8(T, monkeypatch):
    prep, c, out = _run(T, monkeypatch, "const:1.0", crafted=True)
    g = _gt_rows(prep)
    assert c["force"].tolist() == g
    assert c["force_min"] == 4                                      # refcv8: 2 x alloc_min_each (2)
    assert [int((out["r8_hyp_alloc"][r] == g[r]).sum()) >= 4 for r in range(3)] == [True, True, True]


def test_the_coin_is_per_window(T, monkeypatch):
    prep, c, out = _run(T, monkeypatch, "const:1.0", crafted=True, pool_override=[True, False, True])
    g = _gt_rows(prep)
    assert c["force"].tolist() == [g[0], -1, g[2]]
    assert [int((out["r8_hyp_alloc"][r] == g[r]).sum()) for r in (1,)] == [0]
    assert [int((out["r8_hyp_alloc"][r] == g[r]).sum()) >= 4 for r in (0, 2)] == [True, True]


def test_gt_constraint_tf_never_reaches_a_pool_without_the_gt(T, monkeypatch):
    """const:0.0 with GT absent from every pool: the GT CONSTRAINT is read by NO candidate -- perturbing it changes
    nothing (bit-identical). CONTROL: under const:1.0 the same perturbation reaches phi."""
    _p, _c, o0 = _run(T, monkeypatch, "const:0.0", crafted=True)
    _p, _c, o1 = _run(T, monkeypatch, "const:0.0", crafted=True, perturb_cons=True)
    for k in ("r8_phi", "anchor_traj", "r8_hyp_alloc"):
        assert torch.equal(o0[k], o1[k]), k
    _p, _c, f0 = _run(T, monkeypatch, "const:1.0", crafted=True)
    _p, _c, f1 = _run(T, monkeypatch, "const:1.0", crafted=True, perturb_cons=True)
    assert not torch.equal(f0["r8_phi"], f1["r8_phi"])


# =========================================================================== #
# 4. MUTATION arms: re-forcing the GT inside the flag-on path must go RED     #
# =========================================================================== #
def test_MUTATION_the_teacher_drops_the_pool_coin_goes_RED(T, monkeypatch):
    real = T.v3.RefCV3Model.set_r8_teacher

    def mutant(self, hyp, cons, tf, pool=None):                    # the coin never reaches the decoder
        return real(self, hyp, cons, tf)
    monkeypatch.setattr(T.v3.RefCV3Model, "set_r8_teacher", mutant)
    with pytest.raises(AssertionError):
        _assert_const0_pool_is_predicted_only(T, monkeypatch, crafted=True)


def test_MUTATION_the_decoder_reforces_the_gt_goes_RED(T, monkeypatch):
    src = textwrap.dedent(inspect.getsource(RC.AnchoredDiffusionDecoder._r8_extend))
    assert src.count("force=force_a,") == 1                         # the anchor the mutation rewrites exists
    ns: dict = {}
    exec(compile(src.replace("force=force_a,", "force=force,"), RC.__file__, "exec"), RC.__dict__, ns)
    monkeypatch.setattr(RC.AnchoredDiffusionDecoder, "_r8_extend", ns["_r8_extend"])
    with pytest.raises(AssertionError):
        _assert_const0_pool_is_predicted_only(T, monkeypatch, crafted=True)


# =========================================================================== #
# 5. flag absent / const:1.0 == the tip on a REAL compute_losses_v3 + one     #
#    AdamW step over the CI-synthetic corpus; const:0.0 moves it (control)    #
# =========================================================================== #
def _step(T, spec):
    cfg, m = R.build(T, True, n_alloc=N_ALLOC, w_alloc_l1=1.0, w_sat=0.1)
    if spec is not None:
        AF.attach(m, _args(r9_alloc_force_schedule=spec))
    eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=120)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=60, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    n = len(ds)
    batch = torch.utils.data.default_collate([ds[i] for i in range(0, n, max(1, n // 3))][:3])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0], dtype=torch.long)
    batch["tac_goal_y"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
    m._w_tac_v6 = 1.0
    m.train()
    AF.apply_step(m, 7)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(set_to_none=True)
    torch.manual_seed(5)
    L = T.compute_losses_v3(m, batch, "cpu", mode="diffusion")
    L["loss"].backward()
    grads = {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}
    opt.step()
    losses = {k: v.detach().clone() for k, v in L.items() if torch.is_tensor(v) and not k.startswith("r9_pool")}
    scal = {k: v for k, v in L.items() if isinstance(v, (int, float)) and not k.startswith("r9_pool")}
    return losses, scal, grads, {k: v.detach().clone() for k, v in m.state_dict().items()}, L


def test_absent_and_const1_are_bit_identical_through_a_real_step_and_const0_is_not(T):
    la, sa, ga, sda, _ = _step(T, None)
    lb, sb, gb, sdb, Lb = _step(T, "const:1.0")
    assert "r8_alloc_l1" in la and float(la["loss"]) == float(la["loss"])     # NOT VACUOUS: the pool loss ran
    assert sorted(la) == sorted(lb) and all(torch.equal(la[k], lb[k]) for k in la)
    assert sa == sb
    assert sorted(ga) == sorted(gb) and all(torch.equal(ga[k], gb[k]) for k in ga)
    assert all(torch.equal(sda[k], sdb[k]) for k in sda)
    assert float(Lb["r9_pool_p"]) == 1.0 and float(Lb["r9_pool_forced_frac"]) == 1.0
    lc, _sc, gc, sdc, Lc = _step(T, "const:0.0")
    assert float(Lc["r9_pool_p"]) == 0.0 and float(Lc["r9_pool_forced_frac"]) == 0.0
    moved = [k for k in sda if not torch.equal(sda[k], sdc[k])]
    assert moved, "const:0.0 changed no tensor after a step -- the schedule is not live"


# =========================================================================== #
# 6. G-DVB: argv against the built model's attached schedule                 #
# =========================================================================== #
def test_dvb_reads_the_attached_schedule(T):
    from tanitad.train import declared_vs_built as dvb
    parser = T.build_parser()
    args = parser.parse_args(["--arm", "hier", "--out", "_unused", "--refcv8", "--r8-n-alloc", "8",
                              "--r9-alloc-force-schedule", "const:0.0"])
    cfg, m = R.build(T, True, n_alloc=N_ALLOC)

    def r9(model):
        return [str(x) for x in dvb.check(model, args, parser) if str(x).startswith("--r9-alloc-force")]
    assert len(r9(m)) == 1                                            # declared, NOT attached -> a mismatch
    AF.attach(m, args)
    assert r9(m) == []
    getattr(m, AF.MODEL_ATTR).sched = AF.parse_schedule("const:1.0")  # built != declared -> a mismatch
    assert len(r9(m)) == 1
