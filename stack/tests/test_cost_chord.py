"""L3 / R29 — the CHORD goal metric, and the weight it may not ship without.

⛔ THE DEFECT, MEASURED (`D-REFAV1-COST-SURFACE`, `…/Research/
2026-09-03-refav1-cost-surface/raw/cost_surface_{ep2,incumbent}.json`): the
shipped goal term is ``1 - cosine_similarity(z, g)`` evaluated in FLOAT32.
Just below ``cos = 1`` the representable float32 values are
``spacing(1f)/2 = 5.9604645e-08`` apart, so the term can only be an integer
multiple of that step — whatever its own magnitude. The world model's whole
contribution along kappa over the planner's candidate box measures
**1.63e-10**: **1/366th of one representable step**, against a
``0.05*kappa^2`` penalty that is **99.5 %** of all cost variation.

⭐ THE REPAIR UNDER TEST. ``cost_metric="chord"`` computes
``||z_hat - g_hat||`` DIRECTLY. On unit vectors ``chord = sqrt(2*(1-cos))``, so
it is MONOTONE-EQUIVALENT and cannot re-rank anything in exact arithmetic; what
changes is that the component difference of two nearly-equal normalised vectors
is EXACT (Sterbenz), so the value is resolved at the DIFFERENCE's own scale
instead of at 1.0's.

⭐ THE ASSERTION THAT MATTERS IS ABOUT **RANKING**, NOT ABOUT PRECISION FOR ITS
OWN SAKE. `test_e2_*` puts 16 candidates whose TRUE goal terms are 1e-10 apart
— the measured regime — into the REAL goal-space dimension (64 queries x 1024 =
65,536) and asks which metric still orders them. MEASURED here: the shipped
cosine gets **9 of 120** pairs right (chance is 60) and is wrong about the
term's own magnitude by **448x**; the chord gets **120 of 120** and reproduces
the true value to **2.3e-05** relative.

⭐ AND THE DELIBERATE REGRESSION THAT MAKES THAT FALSIFIABLE (`test_f_*`):
``sqrt(2*(1-cos))`` computed FROM the float32 cosine is the SAME ALGEBRA and
recovers **NOTHING** — 1 distinct value, 0 of 120 pairs — because the
quantisation already happened before the square root saw it. *Computing it as a
norm* is the load-bearing half of the repair; *taking a square root* is not.
If that arm ever starts recovering resolution, `test_e2_*` is not measuring
what it claims and no chord result may be reported.

⚠️ THE FLAG IS NOT WEIGHT-NEUTRAL, and `test_i_*` asserts it rather than
letting a reader assume otherwise: the chord multiplies the goal term by
``1/chord`` (MEASURED 5,792.6x on the banked windows) against an UNCHANGED
`W_KAPPA`, so flipping it alone is a metric change AND an implicit
re-weighting. `PREREG_TACTICAL_DECODER.md` §5 L3: *"Swapping the metric while
holding 0.05 fixed is therefore NOT a one-variable arm."*

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import (COST_METRICS, W_JERK, W_KAPPA, W_VEND,
                                  RefAV1, RefAV1Config, _check_cost_metric,
                                  _goal_term)
from tanitad.refs.refa_v1_plan import PlanConfig, _baseline_controls

#: float32 resolution of `1 - cos` near cos = 1: `cos` lands in [0.5, 1), whose
#: spacing is 2**-24. This is the quantum the shipped term inherits.
COS32_QUANTUM = 5.9604644775390625e-08

#: the SHIPPED weights. Pinned because L4 is only meaningful against a baseline
#: that cannot drift silently.
SHIPPED_W_JERK, SHIPPED_W_KAPPA, SHIPPED_W_VEND = 0.02, 0.05, 0.10

#: the REAL goal space: `tac_queries` 64 x `d_state` 1024 (banked
#: `cost_surface_ep2.json` -> `model.cfg`). The dimension is load-bearing —
#: the cosine's accumulated error grows with it.
GOAL_DIM = 64 * 1024
#: the MEASURED kappa-response of the goal term over the whole candidate box.
MEASURED_KAPPA_RESPONSE = 1.0e-10


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _shipped_cos(zt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The pre-2026-09-03 expression, verbatim from `_cost_chunk`."""
    return 1.0 - F.cosine_similarity(zt.flatten(1), g.flatten(1), dim=-1)


def _sqrt_of_1mcos_f32(zt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """⛔ THE DELIBERATE REGRESSION: the chord obtained ALGEBRAICALLY from the
    float32 cosine. Mathematically identical to `_goal_term(..., "chord")`;
    numerically it recovers nothing, because the quantisation already happened.
    """
    return torch.sqrt(2.0 * _shipped_cos(zt, g).clamp_min(0.0))


def _ladder(d: int, delta: float, n: int, seed: int = 31):
    """``(zt [n,1,d], g [n,1,d], true_1mcos [n])`` — n candidates whose TRUE
    goal term walks in exact steps of ``delta``.

    Built in float64 from a unit ``g`` and a unit ``p`` orthogonal to it:
    ``z_k = cos(a_k) g + sin(a_k) p`` gives ``1 - cos(z_k, g) = 1 - cos(a_k)``,
    and ``a_k = sqrt(2 k delta)`` makes that exactly ``k*delta`` to O(a^4).
    Cast to float32 ONCE, so the fields are honest float32 activations and the
    only question left is what the METRIC does with them.
    """
    gen = torch.Generator().manual_seed(seed)
    g64 = torch.randn(1, d, dtype=torch.float64, generator=gen)
    g64 = g64 / g64.norm()
    p64 = torch.randn(1, d, dtype=torch.float64, generator=gen)
    p64 = p64 - (p64 @ g64.T) * g64
    p64 = p64 / p64.norm()
    ang = [math.sqrt(2.0 * (k + 1) * delta) for k in range(n)]
    z64 = torch.cat([math.cos(a) * g64 + math.sin(a) * p64 for a in ang], 0)
    true = np.array([1.0 - math.cos(a) for a in ang])
    return (z64.to(torch.float32)[:, None, :],
            g64.to(torch.float32)[None].expand(n, 1, d).contiguous(),
            true)


def _concordant(v: np.ndarray, true: np.ndarray) -> tuple[int, int]:
    """(pairs ordered correctly, total pairs). A metric that has lost the
    ordering scores ~half; a correct one scores all of them."""
    v = np.asarray(v, dtype=np.float64)
    ok = tot = 0
    for i in range(v.size):
        for j in range(i + 1, v.size):
            tot += 1
            ok += int(np.sign(v[j] - v[i]) == np.sign(true[j] - true[i]))
    return ok, tot


def _n_distinct(t) -> int:
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return int(np.unique(a).size)


def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
                op_layers=1, op_heads=2, op_window=2, tac_layers=1,
                tac_queries=4, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_ctx=16,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                  n_heads=2, d_intent=16))
    base.update(kw)
    return RefAV1Config(**base)


def _model(seed: int = 0, **kw) -> RefAV1:
    torch.manual_seed(seed)
    m = RefAV1(_cfg(**kw)).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    return m


def _pc(c: RefAV1Config) -> PlanConfig:
    """Deliberately tiny: this file is about the COST, not about the search."""
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=8, n_iters=2, n_elites=4)


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


def _proposal_of(m: RefAV1, feats: torch.Tensor) -> torch.Tensor:
    """`plan()`'s own proposal selection, mirrored (`refa_v1.py` :1746-1757)."""
    with torch.no_grad():
        field = m.encode(feats)
        pooled = field.mean(dim=-2)[:, -1]
        modes = m.proposal(pooled).reshape(m.cfg.proposal_k, m.cfg.plan_steps,
                                           m.cfg.a_dim)
        if m.proposal_score is not None:
            modes = modes[m.proposal_score(pooled)[0].argsort(descending=True)]
        return modes[0]


# --------------------------------------------------------------------------- #
#  A — "cos" IS the shipped expression, bit for bit                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(5, 1, 64), (3, 4, 16), (7, 64, 32)])
def test_a_cos_branch_is_bit_identical_to_the_shipped_expression(shape):
    g = torch.Generator().manual_seed(11)
    zt = torch.randn(*shape, generator=g)
    gg = torch.randn(*shape, generator=g)
    ours, ship = _goal_term(zt, gg, "cos"), _shipped_cos(zt, gg)
    assert torch.equal(ours, ship), (
        "the 'cos' branch must BE the pre-2026-09-03 expression, not merely "
        f"agree with it: max abs diff {float((ours - ship).abs().max()):.3e}")


def test_a2_cos_is_the_default_argument():
    g = torch.Generator().manual_seed(12)
    zt = torch.randn(4, 1, 32, generator=g)
    gg = torch.randn(4, 1, 32, generator=g)
    assert torch.equal(_goal_term(zt, gg), _shipped_cos(zt, gg))


@pytest.mark.parametrize("d", [64, 4096, GOAL_DIM])
def test_a3_only_the_chord_reads_its_own_identity_control(d):
    """⭐ THE IDENTITY CONTROL, AND THE SHIPPED METRIC FAILS IT.

    A candidate whose rollout IS the goal rollout must pay EXACTLY nothing.
    MEASURED here (CPU float32): the chord reads **0.0 exactly** at every
    dimension; the shipped cosine reads **1.19e-07** at d = 64, **1.79e-07** at
    d = 4,096 and **1.31e-06** at the real goal dimension 65,536 — the latter
    is **2.7x** the `4*spacing(1f) = 4.77e-07` tolerance that the E0 package's
    K1 control uses, and it is pure reduction noise on two IDENTICAL tensors.

    ⛔ If the cosine ever reads 0.0 here, this box's reduction changed and the
    float32-saturation premise behind every conclusion in
    `…/Research/2026-09-03-cost-repair/` must be re-derived before it is
    quoted again.
    """
    g = torch.Generator().manual_seed(13)
    zt = torch.randn(6, 1, d, generator=g)
    # ⚠️ the two REFERENCE-FREE branches only. `"ccos"` (added 2026-09-04)
    # requires a centring reference by construction and cannot be evaluated
    # from (zt, g) alone; its own identity control -- which it PASSES exactly,
    # for a different reason -- is
    # `test_cost_ccos.py::test_c_identity_control_is_exactly_zero`.
    err = {m: float(_goal_term(zt, zt.clone(), m).abs().max())
           for m in ("cos", "chord")}
    assert err["chord"] == 0.0, (
        f"d={d}: the chord must read EXACTLY 0.0 on identical fields, got "
        f"{err['chord']:.3e}")
    assert err["cos"] > 0.0, (
        f"d={d}: the shipped cosine read {err['cos']:.3e} — a 0.0 here means "
        f"the saturation premise no longer holds on this box")
    assert err["cos"] > err["chord"]


# --------------------------------------------------------------------------- #
#  B/C/D — the flag itself                                                     #
# --------------------------------------------------------------------------- #
def test_b_unknown_metric_is_rejected_loudly():
    with pytest.raises(ValueError, match="cost_metric must be one of"):
        _check_cost_metric("cosine")
    m = _model()
    feats, v0, nav = _window(m.cfg)
    with pytest.raises(ValueError, match="cost_metric must be one of"):
        m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
               cost_metric="euclid")


def test_c_the_default_changes_nothing():
    """⭐ THE LOAD-BEARING REGRESSION. `plan(...)` and
    `plan(..., cost_metric="cos")` must be indistinguishable — otherwise every
    banked refav1 number moved the day this flag landed."""
    m = _model(seed=3)
    feats, v0, nav = _window(m.cfg, seed=3)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
    assert torch.equal(a.controls, b.controls)
    assert float(a.cost) == float(b.cost)
    assert a.source == b.source
    assert set(a.baseline_costs) == set(b.baseline_costs)
    for k in a.baseline_costs:
        assert float(a.baseline_costs[k]) == float(b.baseline_costs[k]), k
    fa, fb = getattr(a, "fine_costs", None), getattr(b, "fine_costs", None)
    assert (fa is None) == (fb is None)
    if fa is not None:
        assert set(fa) == set(fb)
        for k in fa:
            assert float(fa[k]) == float(fb[k]), k
    assert a.cost_metric == "cos" and b.cost_metric == "cos"


def test_d_the_flag_is_live_and_is_stamped_on_the_result():
    """A provenance-only flag is worse than no flag: it reads as a change that
    was made. The chord MUST move at least one candidate's cost."""
    m = _model(seed=4)
    feats, v0, nav = _window(m.cfg, seed=4)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="chord")
    assert a.cost_metric == "cos" and b.cost_metric == "chord"
    moved = [k for k in a.baseline_costs
             if float(a.baseline_costs[k]) != float(b.baseline_costs[k])]
    assert moved, ("the chord left every baseline cost bit-identical — the "
                   "branch is dead, or this window has no goal term")


# --------------------------------------------------------------------------- #
#  E — monotone equivalence in f64, and the ranking it RESTORES in f32         #
# --------------------------------------------------------------------------- #
def test_e_chord_is_monotone_equivalent_in_float64():
    """`chord = sqrt(2*(1-cos))` — in exact arithmetic the two metrics induce
    the SAME ordering. 0 discordant pairs is the assertion; anything else means
    the implementation is not the chord and `test_e2_*` is void."""
    g = torch.Generator().manual_seed(21)
    zt = torch.randn(40, 1, 512, generator=g, dtype=torch.float64)
    gg = torch.randn(1, 1, 512, generator=g,
                     dtype=torch.float64).expand(40, 1, 512).contiguous()
    c = _goal_term(zt, gg, "cos").numpy()
    h = _goal_term(zt, gg, "chord").numpy()
    assert np.allclose(h ** 2 / 2.0, c, rtol=1e-11, atol=1e-13), (
        "chord^2 / 2 must equal 1 - cos in float64")
    disc = int(np.sum(np.sign(c[:, None] - c[None, :])
                      != np.sign(h[:, None] - h[None, :])))
    assert disc == 0, f"{disc} discordant pairs — the chord re-ranked in f64"


def test_e2_the_chord_keeps_the_ordering_the_cosine_loses():
    """⭐ THE CLAIM, in the MEASURED regime and at the REAL goal dimension.

    16 candidates 1e-10 apart in the true goal term, in 65,536 dims, over
    EIGHT independent draws — because the cosine's failure is noise-driven and
    a single draw would pin a number that is not reproducible.

    MEASURED (this rig, CPU float32, seeds 31-38):
      * chord — **120/120** concordant pairs and **16/16** distinct values on
        **every** draw; value error <= 5.6e-05 relative;
      * cosine — **5 to 78** of 120 concordant (chance is 60), never perfect,
        3-6 distinct values, and the term's own magnitude wrong by
        **223x to 671x**.
    """
    tot = 120
    ok_cos, ok_chord, rel_cos, rel_chord, nd_chord = [], [], [], [], []
    for seed in range(31, 39):
        zt, gg, true = _ladder(GOAL_DIM, MEASURED_KAPPA_RESPONSE, 16, seed)
        cos = _goal_term(zt, gg, "cos").double().numpy()
        chord = _goal_term(zt, gg, "chord").double().numpy()
        tch = np.sqrt(2.0 * true)
        ok_cos.append(_concordant(cos, true)[0])
        ok_chord.append(_concordant(chord, true)[0])
        rel_cos.append(float(np.abs(cos - true).max() / true.max()))
        rel_chord.append(float(np.abs(chord - tch).max() / tch.max()))
        nd_chord.append(_n_distinct(_goal_term(zt, gg, "chord")))

    assert min(ok_chord) == tot, (
        f"the chord lost the ordering on some draw: {ok_chord} of {tot}")
    assert min(nd_chord) == 16, f"chord distinct values per draw: {nd_chord}"
    assert max(rel_chord) < 1e-3, f"chord value error: {max(rel_chord):.3e}"
    assert max(ok_cos) < tot, (
        f"the shipped cosine ordered ALL {tot} pairs correctly on some draw "
        f"({ok_cos}) at a true separation of {MEASURED_KAPPA_RESPONSE:g}. Its "
        f"premise (float32 saturation) no longer holds on this box and EVERY "
        f"conclusion downstream of it must be re-derived before it is quoted.")
    assert float(np.median(ok_cos)) <= tot / 2.0, (
        f"the shipped cosine's median concordance {np.median(ok_cos)} is above "
        f"chance ({tot / 2.0}) — see above")
    assert min(rel_cos) > 10.0, (
        f"the shipped cosine's value error is only {min(rel_cos):.3g}x — see "
        f"above")


def test_f_the_deliberate_regression_recovers_nothing():
    """⛔ IF THIS FAILS, `test_e2_*` PROVES NOTHING.

    `sqrt(2*(1-cos))` in float32 is the SAME ALGEBRA as the chord and must
    recover NO ordering, because the cosine quantised the value before the
    square root saw it. This is what makes *compute it as a norm* the
    load-bearing part of the repair rather than *take a square root*.
    """
    zt, gg, true = _ladder(GOAL_DIM, MEASURED_KAPPA_RESPONSE, 16)
    ok_cos, tot = _concordant(_goal_term(zt, gg, "cos").double().numpy(), true)
    ok_sqrt, _ = _concordant(_sqrt_of_1mcos_f32(zt, gg).double().numpy(), true)
    ok_chord, _ = _concordant(_goal_term(zt, gg, "chord").double().numpy(),
                              true)
    assert ok_sqrt <= ok_cos, (
        f"the algebraic square root recovered ordering ({ok_sqrt} vs "
        f"{ok_cos} of {tot}) — the gate cannot see the defect it exists to "
        f"catch")
    assert ok_chord > ok_sqrt


def test_g_away_from_degeneracy_the_two_metrics_agree():
    """The chord must not pretend to buy anything where the cosine can still
    see. At a 1e-3 separation both order all 16 correctly and `chord^2/2`
    tracks `1 - cos`.

    ⚠️ MEASURED, and worth carrying: even at a goal term of ~1e-2 the shipped
    cosine is still **7.4e-04** off the true value in relative terms, so
    `rtol` here is 5e-3 and not 1e-6. The saturation is a spectrum, not a
    cliff.
    """
    zt, gg, true = _ladder(GOAL_DIM, 1e-3, 16, seed=41)
    cos = _goal_term(zt, gg, "cos").double().numpy()
    chord = _goal_term(zt, gg, "chord").double().numpy()
    assert _concordant(cos, true) == (120, 120)
    assert _concordant(chord, true) == (120, 120)
    assert np.allclose(chord ** 2 / 2.0, cos, rtol=5e-3)
    assert float(np.max(np.abs(cos - true) / true)) < 1e-2


# --------------------------------------------------------------------------- #
#  H/I — the weights, and the fact that the metric silently moves them         #
# --------------------------------------------------------------------------- #
def test_h_the_shipped_weights_are_pinned():
    """⛔ If this fails, every banked refav1 cost number is on a different
    scale and the L4 derivation in
    `…/Research/2026-09-03-cost-repair/RESULT.md` §4 is void."""
    assert (W_JERK, W_KAPPA, W_VEND) == (SHIPPED_W_JERK, SHIPPED_W_KAPPA,
                                         SHIPPED_W_VEND)


def test_h2_the_cost_charges_the_named_W_KAPPA_and_nothing_else():
    """The constants must be what `_cost_chunk` charges, not decoration.

    Scaling `W_KAPPA` by 10 must (a) leave every ZERO-CURVATURE baseline
    EXACTLY unchanged — which is itself the proof that the penalty is charged
    on curvature and on nothing else — and (b) move the curved `proposal`
    candidate by EXACTLY `9 * W_KAPPA * mean(kappa^2)`.
    """
    import tanitad.refs.refa_v1 as R
    m = _model(seed=6)
    feats, v0, nav = _window(m.cfg, seed=6)
    pc = _pc(m.cfg)
    prop = _baseline_controls(pc, v0, feats.device,
                             _proposal_of(m, feats))["proposal"]
    k2 = float(prop[..., 1].pow(2).mean())
    assert k2 > 0.0, "the proposal is straight — this test needs a curved one"
    with torch.no_grad():
        base = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc).baseline_costs
        old = R.W_KAPPA
        try:
            R.W_KAPPA = old * 10.0
            hi = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc).baseline_costs
        finally:
            R.W_KAPPA = old
        back = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc).baseline_costs
    for k in ("cv", "hold_v0", "decel_1.5"):
        assert float(base[k]) == float(hi[k]), (
            f"{k} has kappa == 0 yet moved when W_KAPPA changed")
    d = float(hi["proposal"]) - float(base["proposal"])
    assert d == pytest.approx(9.0 * SHIPPED_W_KAPPA * k2, rel=1e-5), (
        f"proposal moved by {d:.6e}, expected "
        f"{9.0 * SHIPPED_W_KAPPA * k2:.6e}")
    for k in base:
        assert float(base[k]) == float(back[k]), f"{k}: W_KAPPA not restored"


def test_i_the_chord_is_not_weight_neutral():
    """⚠️ THE TRAP `PREREG_TACTICAL_DECODER.md` §5 L3 names: swapping the
    metric while holding `W_KAPPA` fixed is a metric change AND an implicit
    re-weighting (MEASURED 5,792.6x on the banked windows). This test fails if
    anyone ever 'simplifies' the chord into something weight-neutral."""
    zt, gg, _ = _ladder(4096, 1e-9, 8, seed=51)
    cos = _goal_term(zt, gg, "cos").double().clamp_min(1e-30)
    chord = _goal_term(zt, gg, "chord").double()
    lev = float((chord / cos).median())
    assert lev > 100.0, (
        f"chord/cos leverage {lev:.3g} — the metric swap is being reported as "
        f"weight-neutral when the banked measurement is 5,792.6x")


def test_j_the_metrics_tuple_is_exactly_the_supported_branches():
    """⚠️ Extended 2026-09-04: `"ccos"` (centred cosine) is a THIRD branch and
    is a different KIND of change from the chord -- it RE-RANKS, which the
    chord provably cannot. `test_cost_ccos.py` owns its behaviour; this file
    keeps owning the chord's, and the tuple is pinned here so a fourth branch
    cannot arrive unannounced."""
    assert COST_METRICS == ("cos", "chord", "ccos")
    for metric in COST_METRICS:
        assert _check_cost_metric(metric) == metric
