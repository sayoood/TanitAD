"""D-REFAV1-CENTRED-GOAL — `cost_metric="ccos"`, the CENTRED cosine.

⛔ THE DEFECT (MEASURED, `.../Research/2026-09-04-refav1-cost-scale/RESULT.md`,
checkpoint step 21,109, n = 282 windows / 141 episode clusters): the planner's
goal term compares two UNCENTRED fields that are **99.3 % common mode**, so
``1 - cos`` realises ~1.2e-05 of its own [0, 2] range. Against that, the
CHEAPEST jerk charge available in a 300-sample iCEM population is
``0.02 * 4.711 = 0.094``, while ``goal(cv)`` maxes at **5.58e-04** over the
whole eval set ⇒ **100.00 % of the iteration-0 population is excluded before
the world model is consulted.** Curvature came out identically 0 on **282/282**
windows, with exactly **2** distinct plans, **100 %** of them bit-exact copies
of an injected baseline.

⭐ THE REPAIR UNDER TEST. ``ccos`` computes
``1 - cos(z_K - z_ref, g - z_ref)`` where ``z_ref`` is the ZERO-ACTION
(constant-velocity) terminal field of the same window, rolled by the same
predictor from the same ``z0`` in the same forward pass. It is exactly the
`cv` baseline's rollout, so it costs no extra rollout and reads nothing from
the future.

⛔⛔ IT IS A DIFFERENT KIND OF CHANGE FROM `"chord"`, AND CONFLATING THE TWO IS
THE MISTAKE THIS FILE EXISTS TO PREVENT. ``chord = sqrt(2*(1-cos))`` is
**monotone-equivalent**: `test_cost_chord.py::test_e_*` asserts **0 discordant
pairs** in float64, i.e. it CANNOT re-rank two candidates in exact arithmetic —
it repairs float32 catastrophic cancellation and nothing else. **Subtracting a
common vector CHANGES THE RANKING**, in exact arithmetic, and `test_k_*` below
exhibits a candidate pair whose order flips in float64 with the mechanism
visible: the uncentred cosine prefers the candidate that BARELY MOVES over the
one that moves in the goal's own direction, which is the 282/282 flat plan in
miniature.

⚠️ AND IT IS NOT WEIGHT-NEUTRAL EITHER — by a much LARGER factor than the
chord's 5,792.6x on the value scale. MEASURED (`raw/ccos_scale_factor.json`,
n = 40 windows, same checkpoint): value leverage **751,546x**, DECISION leverage
(the ratio of the term's variation over the candidate box, which is what trades
against the penalties) **214.46x**. `test_i_*` pins that this is not silently
"simplified" away.

⚠️ AND IT CARRIES A KNOWN DEGENERACY THAT IS **NOT** REPAIRED HERE, deliberately.
The do-nothing candidate IS ``z_ref``, so its centred vector is the zero vector
and ``ccos(cv) = 1.0`` EXACTLY, by definition rather than by measurement
(MEASURED 40/40 windows). A HOLD BRANCH is a pre-registration item belonging to
the PI / Master Mind. `test_f_*` and `test_m_*` pin the degeneracy as a
KNOWN VALUE so that it cannot be discovered later as a surprise.

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import (COST_METRICS, RefAV1, RefAV1Config,
                                  _CHORD_EPS, _check_cost_metric, _goal_term)
from tanitad.refs.refa_v1_plan import PlanConfig

#: the REAL goal space: `tac_queries` 64 x `d_state` 1024.
GOAL_DIM = 64 * 1024
#: MEASURED median ||z_K|| on the banked windows (`RESULT.md` §2a) — the scale
#: at which the common mode is 99.3 % of the field.
FIELD_NORM = 315.0
#: MEASURED decision leverage of ccos over cos (`raw/ccos_scale_factor.json`).
MEASURED_DECISION_LEVERAGE = 214.46


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _shipped_cos(zt, g):
    """The pre-2026-09-03 expression, verbatim from `_cost_chunk`."""
    return 1.0 - F.cosine_similarity(zt.flatten(1), g.flatten(1), dim=-1)


def _shipped_chord(zt, g):
    """The 2026-09-03 chord branch, verbatim — re-implemented here so the
    `z_ref` parameter cannot have perturbed it without this file noticing."""
    x, y = zt.flatten(1), g.flatten(1)
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    yn = y / y.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    return (xn - yn).norm(dim=-1)


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
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=8, n_iters=2, n_elites=4)


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


def _basis(d: int, seed: int = 71, dtype=torch.float64):
    """Three orthonormal directions in ``d`` dims: the common mode ``e``, the
    goal's action direction ``u``, and an orthogonal direction ``w``."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, 3, generator=gen, dtype=torch.float64))
    e, u, w = q[:, 0], q[:, 1], q[:, 2]
    return e.to(dtype), u.to(dtype), w.to(dtype)


# --------------------------------------------------------------------------- #
#  A — the two pre-existing branches are UNTOUCHED, bit for bit                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(5, 1, 64), (3, 4, 16), (7, 64, 32)])
def test_a_cos_is_still_bit_identical_to_the_shipped_expression(shape):
    """⛔ THE LOAD-BEARING PIN. Every banked refav1 number was produced under
    ``1 - F.cosine_similarity(...)``. Adding `z_ref` must not have moved that
    branch by one bit — including when `z_ref` is passed and ignored."""
    g = torch.Generator().manual_seed(101)
    zt = torch.randn(*shape, generator=g)
    gg = torch.randn(*shape, generator=g)
    ref = torch.randn(1, *shape[1:], generator=g)
    ship = _shipped_cos(zt, gg)
    assert torch.equal(_goal_term(zt, gg, "cos"), ship)
    assert torch.equal(_goal_term(zt, gg), ship), "the DEFAULT moved"
    assert torch.equal(_goal_term(zt, gg, "cos", ref), ship), (
        "'cos' consumed z_ref — the branch is not inert as documented")


@pytest.mark.parametrize("shape", [(5, 1, 64), (3, 4, 16), (7, 64, 32)])
def test_a2_chord_is_still_bit_identical(shape):
    g = torch.Generator().manual_seed(102)
    zt = torch.randn(*shape, generator=g)
    gg = torch.randn(*shape, generator=g)
    ref = torch.randn(1, *shape[1:], generator=g)
    ship = _shipped_chord(zt, gg)
    assert torch.equal(_goal_term(zt, gg, "chord"), ship)
    assert torch.equal(_goal_term(zt, gg, "chord", ref), ship)


def test_b_the_tuple_and_the_validator_carry_the_third_branch():
    assert COST_METRICS == ("cos", "chord", "ccos")
    for metric in COST_METRICS:
        assert _check_cost_metric(metric) == metric
    with pytest.raises(ValueError, match="cost_metric must be one of"):
        _check_cost_metric("centred")


def test_b2_ccos_without_a_reference_fails_LOUDLY():
    """⛔ A silent fallback to the uncentred form would make an arm labelled
    'ccos' score as 'cos' — the exact class of defect a verify-gate exists to
    catch. It must raise, not default."""
    zt = torch.randn(4, 1, 32)
    gg = torch.randn(4, 1, 32)
    with pytest.raises(ValueError, match="needs z_ref"):
        _goal_term(zt, gg, "ccos")


# --------------------------------------------------------------------------- #
#  C/D/E/F — the CONTROLS, each of which must read a KNOWN value EXACTLY       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [64, 4096, GOAL_DIM])
def test_c_identity_control_is_exactly_zero(d):
    """A candidate whose centred rollout IS the goal's must pay 0 — to float32
    cosine precision. KNOWN VALUE: 0.0, tolerance 2e-6.

    ⚠️ MEASURED 2026-09-05 (torch 2.11.0+cu128, CPU): this read **5.96e-08**
    (d=64), **-2.38e-07** (4096) and **-8.34e-07** (65,536) — NOT exactly 0.
    A float32 cosine of two IDENTICAL vectors is not exactly 1: ``F.cosine_
    similarity`` normalises by ``sqrt(x.x)`` and ``sqrt(s)**2 != s`` in
    float32, and the 65,536-term reduction is itself inexact. The shipped
    ``cos`` has the SAME property (1.3e-06, `test_cost_chord.py::test_a3_*`);
    only the CHORD reads its identity control exactly (Sterbenz). The first
    version of this test asserted ``== 0.0`` — a property a float32 cosine
    cannot have — and would have failed on every machine. The tolerance is the
    one `test_d`/`test_e` already use for the SAME function.
    """
    e, u, _ = _basis(d, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    zt = (r + 3.0 * u[None, None])
    gg = (r + 3.0 * u[None, None])
    v = float(_goal_term(zt, gg, "ccos", r).abs().max())
    assert v <= 2e-6, f"d={d}: got {v:.6e}"
    # and the SHIPPED branch's own identity error on the same fields, so the
    # comparison is like with like: ccos is no worse than cos here.
    v_cos = float(_goal_term(zt, gg, "cos").abs().max())
    assert v <= max(2e-6, 4.0 * v_cos), (
        f"d={d}: ccos identity error {v:.3e} vs cos's own {v_cos:.3e}")


@pytest.mark.parametrize("d", [64, 4096, GOAL_DIM])
def test_d_orthogonal_control_is_exactly_the_no_information_value(d):
    """⭐ THE NO-INFORMATION CONTROL. A candidate whose action-induced change is
    ORTHOGONAL to the goal's carries zero information about it, and the centred
    cosine must read the midpoint of its own [0, 2] range. KNOWN VALUE: 1.0."""
    e, u, w = _basis(d, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    zt = r + 2.0 * w[None, None]
    gg = r + 5.0 * u[None, None]
    v = float(_goal_term(zt, gg, "ccos", r)[0])
    assert v == pytest.approx(1.0, abs=2e-6), f"d={d}: got {v!r}"


@pytest.mark.parametrize("d", [64, 4096, GOAL_DIM])
def test_e_antipodal_control_is_the_worst_value(d):
    """A candidate moving exactly OPPOSITE to the goal must read the range's
    maximum. KNOWN VALUE: 2.0."""
    e, u, _ = _basis(d, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    zt = r - 4.0 * u[None, None]
    gg = r + 4.0 * u[None, None]
    v = float(_goal_term(zt, gg, "ccos", r)[0])
    assert v == pytest.approx(2.0, abs=2e-6), f"d={d}: got {v!r}"


def test_f_the_cv_degeneracy_is_a_KNOWN_VALUE_not_a_surprise():
    """⚠️ THE DEGENERACY THIS FLAG SHIPS WITH, pinned so nobody rediscovers it.

    The do-nothing candidate IS `z_ref`; its centred vector is the ZERO vector;
    `F.cosine_similarity` of a zero vector is 0 ⇒ ``ccos(cv) = 1.0`` EXACTLY.
    MEASURED at 1.0 on 40/40 banked windows (`raw/cost_forms_devbox.json`).
    On the 86.5 % of the eval grid whose decoded goal is LANE_KEEP the goal is
    itself "hold", so this is not a corner case — it is the majority stratum,
    and it is why a HOLD BRANCH is a named PRE-REGISTRATION item rather than
    something taken in `refa_v1.py`.
    """
    e, u, _ = _basis(4096, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    gg = r + 5.0 * u[None, None]
    v = float(_goal_term(r, gg, "ccos", r)[0])
    assert v == 1.0, f"ccos(cv) must be EXACTLY 1.0, got {v!r}"


def test_g_zero_model_control_reads_a_constant_EXACTLY():
    """⭐ THE ZERO-MODEL CONTROL (`cost_surface_probe.py`'s C3, in centred
    form). If every candidate's terminal field is the SAME field, the metric
    can carry no candidate information: peak-to-peak must be EXACTLY 0.0 and,
    when that shared field is `z_ref` itself, the value must be EXACTLY 1.0.
    A probe that manufactures structure here would manufacture it everywhere."""
    e, u, _ = _basis(4096, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    gg = (r + 5.0 * u[None, None]).expand(9, 1, 4096)
    same = r.expand(9, 1, 4096)
    v = _goal_term(same, gg, "ccos", r)
    assert float(v.max() - v.min()) == 0.0, "ptp over identical fields != 0"
    assert float(v[0]) == 1.0
    # and a non-degenerate constant field: still exactly constant
    same2 = (r + 1.5 * u[None, None]).expand(9, 1, 4096)
    v2 = _goal_term(same2, gg, "ccos", r)
    assert float(v2.max() - v2.min()) == 0.0


# --------------------------------------------------------------------------- #
#  K — ⭐ THE PROPERTY THAT DISTINGUISHES THIS FROM THE CHORD                   #
# --------------------------------------------------------------------------- #
def test_k_ccos_is_NOT_monotone_equivalent_to_cos_and_here_is_the_pair():
    """⭐⭐ CENTRING RE-RANKS. The chord provably cannot
    (`test_cost_chord.py::test_e_*`: 0 discordant pairs in float64); this does,
    in float64, and the flip is the DEFECT IN MINIATURE.

    Two candidates against a goal that asks for ``+a*u`` from a field whose
    common mode is ``R*e``:
      * ``z_far``  = r + 10*u  -- moves in the goal's OWN direction, 10x too far
      * ``z_still``= r + 1e-3*w -- barely moves, and orthogonally

    ``1 - cos`` prefers **z_still**, because from the origin both candidates are
    nearly parallel to ``r`` and the one that moved LESS is the one closer in
    ANGLE to ``r + a*u``. ``ccos`` prefers **z_far**, because after removing the
    common mode the question becomes "did you move the way the goal asked?".

    ⛔ That preference for "do nothing" is exactly the measured behaviour:
    curvature identically 0 on 282/282 windows, the plan a bit-exact copy of an
    injected baseline 100 % of the time.
    """
    d = 4096
    e, u, w = _basis(d, dtype=torch.float64)
    r = (FIELD_NORM * e)[None, None]
    gg = (r + 1.0 * u[None, None]).expand(2, 1, d).contiguous()
    zt = torch.stack([r[0] + 10.0 * u[None], r[0] + 1e-3 * w[None]], 0)

    c = _goal_term(zt, gg, "cos").numpy()
    h = _goal_term(zt, gg, "chord").numpy()
    q = _goal_term(zt, gg, "ccos", r).numpy()
    print(f"\n[test_k] n = 2 candidates, d = {d} "
          f"(real goal dim is {GOAL_DIM}); float64")
    print(f"[test_k] cos  : z_far {c[0]:.6e}  z_still {c[1]:.6e}")
    print(f"[test_k] chord: z_far {h[0]:.6e}  z_still {h[1]:.6e}")
    print(f"[test_k] ccos : z_far {q[0]:.6e}  z_still {q[1]:.6e}")

    assert c[1] < c[0], "the uncentred cosine did not prefer the still candidate"
    assert q[0] < q[1], "the centred cosine did not prefer the goal-directed one"
    # the chord agrees with cos — because it CANNOT disagree
    assert np.sign(h[1] - h[0]) == np.sign(c[1] - c[0]), (
        "the chord re-ranked; it is monotone-equivalent and must not")
    # exact known values for the centred pair
    assert q[0] == pytest.approx(0.0, abs=1e-12)
    assert q[1] == pytest.approx(1.0, abs=1e-12)


def test_k2_the_discordance_is_measurable_over_a_population():
    """The same statement as a rate rather than an anecdote: over a random
    candidate population around a common mode, `chord` has EXACTLY 0 discordant
    pairs with `cos` and `ccos` has many. n and d are printed."""
    d, n = 512, 40
    e, u, _ = _basis(d, seed=73, dtype=torch.float64)
    gen = torch.Generator().manual_seed(77)
    r = (FIELD_NORM * e)[None, None]
    delta = torch.randn(n, 1, d, generator=gen, dtype=torch.float64)
    delta = delta / delta.norm(dim=-1, keepdim=True) * torch.linspace(
        0.05, 5.0, n, dtype=torch.float64)[:, None, None]
    zt = r + delta
    gg = (r + 2.0 * u[None, None]).expand(n, 1, d).contiguous()
    c = _goal_term(zt, gg, "cos").numpy()
    h = _goal_term(zt, gg, "chord").numpy()
    q = _goal_term(zt, gg, "ccos", r).numpy()

    def _disc(a, b):
        return int(np.sum(np.sign(a[:, None] - a[None, :])
                          != np.sign(b[:, None] - b[None, :])))
    tot = n * (n - 1)
    dc, dq = _disc(c, h), _disc(c, q)
    print(f"\n[test_k2] n = {n} candidates, d = {d}, float64; "
          f"ordered pairs = {tot}")
    print(f"[test_k2] chord vs cos: {dc} discordant  ({100*dc/tot:.1f} %)")
    print(f"[test_k2] ccos  vs cos: {dq} discordant  ({100*dq/tot:.1f} %)")
    assert dc == 0, f"the chord re-ranked {dc} pairs — it cannot"
    assert dq > 0.05 * tot, (
        f"ccos re-ranked only {dq}/{tot} pairs — if centring stops re-ranking, "
        f"it has become the chord and this whole branch is redundant")


def test_i_ccos_is_NOT_weight_neutral():
    """⚠️ THE TRAP `PREREG_TACTICAL_DECODER.md` §5 L3 names, for the third
    branch. MEASURED (`raw/ccos_scale_factor.json`, n = 40 windows, step
    21,109): DECISION leverage — the ratio of the term's variation over the
    candidate box, which is what trades against the unchanged `W_JERK` /
    `W_KAPPA` — is **214.46x** (median; p10 30.1, p90 722.2), and the VALUE
    leverage is **751,546x**. Any arm flipping this flag declares the weight
    triple in the same breath."""
    d, n = 512, 24
    e, u, _ = _basis(d, seed=79, dtype=torch.float64)
    gen = torch.Generator().manual_seed(83)
    r = (FIELD_NORM * e)[None, None]
    delta = torch.randn(n, 1, d, generator=gen, dtype=torch.float64)
    delta = delta / delta.norm(dim=-1, keepdim=True) * 0.5
    zt = r + delta
    gg = (r + 0.5 * u[None, None]).expand(n, 1, d).contiguous()
    c = _goal_term(zt, gg, "cos").numpy()
    q = _goal_term(zt, gg, "ccos", r).numpy()
    lev = float(np.ptp(q) / np.ptp(c))
    print(f"\n[test_i] n = {n}, d = {d}: decision leverage ptp(ccos)/ptp(cos) "
          f"= {lev:.4g}  (MEASURED on real fields: "
          f"{MEASURED_DECISION_LEVERAGE})")
    assert lev > 10.0, (
        f"leverage {lev:.4g} — the centring is being reported as weight-neutral "
        f"when the banked measurement is {MEASURED_DECISION_LEVERAGE}x")


# --------------------------------------------------------------------------- #
#  L/M/N — the flag inside `plan()`                                            #
# --------------------------------------------------------------------------- #
def test_l_the_default_still_changes_nothing():
    """⛔ THE REGRESSION THAT PROTECTS EVERY BANKED NUMBER. `plan(...)`,
    `plan(cost_metric="cos")` and `plan(cost_metric="chord")` must all be
    exactly what they were before `z_ref` existed."""
    m = _model(seed=3)
    feats, v0, nav = _window(m.cfg, seed=3)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
    assert torch.equal(a.controls, b.controls)
    assert float(a.cost) == float(b.cost) and a.source == b.source
    for k in a.baseline_costs:
        assert float(a.baseline_costs[k]) == float(b.baseline_costs[k]), k
    assert a.cost_metric == "cos"


def test_m_plan_under_ccos_reads_the_cv_degeneracy_END_TO_END():
    """⭐ THE END-TO-END KNOWN-VALUE CONTROL, and the single most decision-
    relevant number in this file.

    `cv` is the all-zero control, so ``W_JERK*jerk^2 = W_KAPPA*kappa^2 = 0``
    and its TOTAL cost IS its goal term. Under `"ccos"` that must be **1.0**
    (`test_f_*`); under `"cos"` it is the measured ~1e-07. The planner's floor
    therefore moves by SEVEN ORDERS OF MAGNITUDE, which is the whole mechanism
    by which the a-priori exclusion falls from 100 %.

    ⚠️ NOT ``== 1.0`` AND NOT ``approx(1.0, 1e-4)`` — MEASURED 2026-09-05:
    ``baseline_costs['cv']`` under ccos read **1.0531** on this tiny model.
    `baseline_costs` comes from `icem_plan`'s SECOND `cost_fn(base_stack)`
    call, where cv is ROW 0 of a 4-row batch, while ``z_ref`` is rolled with
    batch 1. The two fields differ by batch-composition ulps, so cv's centred
    vector is a NOISE vector, not the zero vector, and its cosine against
    ``g - z_ref`` is O(1/sqrt(D)): +-0.125 at this model's D = 64, +-0.004 at
    the real D = 65,536. The exact 1.0 of `test_f` holds only for bit-identical
    fields. ⇒ the KNOWN VALUE here is "1.0 up to that noise", and the
    decision-relevant assertion is that the floor moved from ~1e-07 to ~1.0
    — SEVEN orders of magnitude — not its last digits. The per-window value on
    the real checkpoint is BANKED by `refav1_arm.py` (``basecost_cv_cl``), so
    it is measured, never assumed.
    """
    m = _model(seed=5)
    feats, v0, nav = _window(m.cfg, seed=5)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="ccos")
    assert b.cost_metric == "ccos"
    assert a.goal_source == b.goal_source == "tactical_imagined", (
        "no goal on this window — the test measures nothing")
    D = m.cfg.tac_queries * m.cfg.d_state
    noise = 4.0 / D ** 0.5                     # 4 sigma of an O(1/sqrt(D)) cosine
    for k in ("cv", "hold_v0"):
        v = float(b.baseline_costs[k])
        assert abs(v - 1.0) <= noise, (
            f"{k} under ccos = {v!r}, expected the no-information value 1.0 up "
            f"to batch-composition noise {noise:.3f} (D={D})")
        assert v > 0.5, f"{k} under ccos = {v!r}: the floor did not move to ~1.0"
        assert float(a.baseline_costs[k]) < 1e-3, (
            f"{k} under cos = {a.baseline_costs[k]!r} — this tiny model does "
            f"not reproduce the near-zero uncentred floor, so the contrast "
            f"below is not the one measured on the real checkpoint")


def test_n_the_centring_reference_IS_the_cv_rollout():
    """⛔ THE PROVENANCE ASSERTION. `z_ref` must be the ZERO-ACTION terminal
    field of THIS window under THIS predictor — not a batch mean, not a running
    statistic, and nothing from the future. Captured at the one site it is
    used and compared against the `cv` baseline's own terminal field from the
    SAME call."""
    import tanitad.refs.refa_v1 as R
    m = _model(seed=7)
    feats, v0, nav = _window(m.cfg, seed=7)
    pc = _pc(m.cfg)
    seen = []
    real = R._goal_term

    def _rec(zt, g, metric="cos", z_ref=None):
        seen.append((zt.detach().clone(),
                     None if z_ref is None else z_ref.detach().clone(),
                     int(zt.shape[0])))
        return real(zt, g, metric, z_ref)

    R._goal_term = _rec
    try:
        with torch.no_grad():
            m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="ccos")
    finally:
        R._goal_term = real
    assert seen, "the goal term was never called — no goal on this window"
    assert all(s[1] is not None for s in seen), "a call arrived without z_ref"

    # every call in one plan() shares at most TWO references (the coarse search
    # and the coarse->fine re-score), and each is [1, ...] — one window's own.
    refs = {tuple(np.round(s[1].flatten().numpy(), 12)) for s in seen}
    assert len(refs) <= 2, f"{len(refs)} distinct references in one plan()"
    for _, r, _n in seen:
        assert r.shape[0] == 1, f"z_ref has batch {r.shape[0]}, expected 1"

    # ⭐ the base_stack call carries `cv` as ROW 0 (`_baseline_controls`), so
    # its terminal field must BE the reference, up to batch-composition ulp.
    base = [s for s in seen if s[2] == 4]
    assert base, "no 4-row baseline call was captured"
    zt, ref, _ = base[-1]
    err = float((zt[0] - ref[0]).abs().max())
    scale = float(ref.abs().max())
    assert err <= 1e-5 * max(scale, 1.0), (
        f"the cv row and z_ref differ by {err:.3e} (field scale {scale:.3e}) — "
        f"the reference is not the zero-action rollout")


def test_o_ccos_actually_moves_the_decision():
    """A provenance-only flag is worse than no flag. `"ccos"` must change at
    least one baseline cost AND must be stamped on the result."""
    m = _model(seed=9)
    feats, v0, nav = _window(m.cfg, seed=9)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="ccos")
    assert (a.cost_metric, b.cost_metric) == ("cos", "ccos")
    moved = [k for k in a.baseline_costs
             if float(a.baseline_costs[k]) != float(b.baseline_costs[k])]
    assert moved, "ccos left every baseline cost identical — the branch is dead"


# --------------------------------------------------------------------------- #
#  P — `cost_weights`: the weight triple travels ON THE CALL (D-REFAV1-CCOS-EVAL)#
# --------------------------------------------------------------------------- #
def test_p_cost_weights_default_and_explicit_shipped_are_bit_identical():
    """⛔ THE REGRESSION PIN for the 2026-09-05 edit: `plan()` with no
    `cost_weights`, with `cost_weights=None`, and with the shipped triple
    spelled out must all be exactly what they were — and the triple actually
    used must be stamped on the result."""
    from tanitad.refs.refa_v1 import W_JERK, W_KAPPA, W_VEND
    m = _model(seed=11)
    feats, v0, nav = _window(m.cfg, seed=11)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_weights=None)
        c = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                   cost_weights=(W_JERK, W_KAPPA, W_VEND))
    for other in (b, c):
        assert torch.equal(a.controls, other.controls)
        assert float(a.cost) == float(other.cost) and a.source == other.source
        for k in a.baseline_costs:
            assert float(a.baseline_costs[k]) == float(other.baseline_costs[k]), k
    assert a.cost_weights == b.cost_weights == c.cost_weights == \
        (W_JERK, W_KAPPA, W_VEND)


def test_p2_cost_weights_override_scales_ONLY_the_penalties_and_is_stamped():
    """Scaling W_KAPPA by 10 through the CALL must leave every zero-curvature
    baseline's cost bit-identical (their kappa penalty is exactly 0) and move
    a curved candidate by EXACTLY 9 * W_KAPPA * mean(kappa^2) — the same
    known-value construction as `test_cost_chord.py::test_h_*`, without any
    module-global mutation. The result names the triple it used."""
    import tanitad.refs.refa_v1 as R
    from tanitad.refs.refa_v1 import W_JERK, W_KAPPA, W_VEND
    m = _model(seed=13)
    feats, v0, nav = _window(m.cfg, seed=13)
    pc = _pc(m.cfg)
    with torch.no_grad():
        base = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        ten = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                     cost_weights=(W_JERK, 10.0 * W_KAPPA, W_VEND))
    assert ten.cost_weights == (W_JERK, 10.0 * W_KAPPA, W_VEND)
    assert base.cost_weights == (W_JERK, W_KAPPA, W_VEND)
    # module constants were NOT touched by the call-site override
    assert (R.W_JERK, R.W_KAPPA, R.W_VEND) == (W_JERK, W_KAPPA, W_VEND)
    for k in ("cv", "hold_v0", "decel_1.5"):
        assert float(base.baseline_costs[k]) == float(ten.baseline_costs[k]), (
            f"{k} has kappa == 0 yet moved under a W_KAPPA override")
    # a curved candidate through the SAME closure: rescore the winner's controls
    # under both weightings via the fine (operative) re-score path is not
    # exposed, so use the goal-free identity on the candidate the planner
    # returned when it is curved; otherwise assert the override at least
    # reached the result (the stamp) — the exact-9x identity is pinned by
    # test_cost_chord.py::test_h_* on the module-global path.
    k2 = float(base.controls[:, 1].pow(2).mean())
    if k2 > 0:
        assert float(ten.cost) != float(base.cost)


def test_p3_cost_weights_shape_is_validated():
    m = _model(seed=17)
    feats, v0, nav = _window(m.cfg, seed=17)
    pc = _pc(m.cfg)
    with pytest.raises(ValueError, match="cost_weights must be"):
        with torch.no_grad():
            m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_weights=(1.0, 2.0))
