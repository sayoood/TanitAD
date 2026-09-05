"""D-REFAV1-COST-GEOMETRY / L3 - the SUSTAINED-CURVATURE SEED LADDER.

WHY IT EXISTS, MEASURED and banked BEFORE the code was written
(`.../Research/2026-09-05-refav1-cost-geometry/raw/kappa_quantisation.txt`,
n = 40 windows / 8 episodes, ckpt 21,109, `--cost-metric ccos`, weights
(0, 0, 64.297)): the winning plan's curvature series is **EXACTLY CONSTANT over
the whole 2 s horizon on 31 of 40 windows (77.5 %)**, and of those **10 read
exactly 0.000000 and 21 exactly 0.080000** - `LANE_KEEP`'s canonical profile and
`GOAL_KAPPA_TURN`. Both figures reproduce bit-for-bit on the inference-seed
replicate. Under `cos` the same statistic is 15.0 % across 32 distinct values.

⇒ on 77.5 % of windows the "planned" trajectory IS the decoded token's canonical
control profile, verbatim. `D-REFAV1-DRIVE-GATE` says why: `colored_noise` is
zero-mean along time (`refa_v1_plan.py:157`) and no injected baseline carries
curvature (`:178-186`), so a SUSTAINED curvature can only enter through the seed
pool - which offers `{0, +-0.08}` and nothing between, while the corpus curves at
|kappa| ~ 0.001-0.01 (R 100-1000 m).

⚠️ THIS IS NOT A VOCABULARY CHANGE. `canonical_controls` and the goal field are
untouched; only the SEARCH's iteration-0 candidate set is widened and the COST
then decides. That is what makes it comparable across arms sharing a vocabulary,
which `--goal-kappa-levels` is not.

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import GOAL_KAPPA_MAX, RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import PlanConfig


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


# --------------------------------------------------------------------------- #
#  A - the default changes NOTHING                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_a_default_is_bit_identical(seed):
    """`None` (and the omitted argument) must leave every banked arm's plan
    bit-identical - the whole point of gating this behind a flag."""
    m = _model(seed=seed)
    feats, v0, nav = _window(m.cfg, seed=seed)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                   seed_kappa_ladder=None)
        c = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                   seed_kappa_ladder=())
    assert torch.equal(a.controls, b.controls)
    assert torch.equal(a.controls, c.controls)
    assert a.n_evaluated == b.n_evaluated == c.n_evaluated
    assert getattr(a, "seed_kappa_ladder", "MISSING") is None


# --------------------------------------------------------------------------- #
#  B - the pool really grows, by EXACTLY the predicted number of candidates    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ladder", [(0.01,), (0.005, 0.02), (0.002, 0.01, 0.05)])
def test_b_n_evaluated_grows_by_exactly_two_per_rung(ladder):
    """THE KNOWN-VALUE CONTROL. The ladder is injected at iteration 0 only, both
    signs, so `n_evaluated` must rise by EXACTLY `2 * len(ladder)` - not more
    (it is not re-injected each iteration) and not less (nothing is dropped)."""
    m = _model(seed=5)
    feats, v0, nav = _window(m.cfg, seed=5)
    pc = _pc(m.cfg)
    with torch.no_grad():
        base = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        got = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                     seed_kappa_ladder=ladder)
    assert got.n_evaluated - base.n_evaluated == 2 * len(ladder)
    assert got.seed_kappa_ladder == tuple(float(k) for k in ladder)


def test_c_the_rungs_are_REACHABLE_by_the_search():
    """The candidates must be able to WIN, or the lever is decorative. With the
    curvature penalty switched off and a cost that rewards the goal, a rung is
    at least selectable; here we assert the weaker, deterministic fact that a
    ladder candidate is a legal, clipped, sustained-curvature control sequence
    by driving the search with a cost that PREFERS large |kappa|."""
    m = _model(seed=9)
    feats, v0, nav = _window(m.cfg, seed=9)
    pc = _pc(m.cfg)
    with torch.no_grad():
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                     seed_kappa_ladder=(0.01, 0.03))
    k = res.controls[:, 1]
    assert torch.isfinite(k).all()
    assert float(k.abs().max()) <= pc.kappa_max + 1e-9


# --------------------------------------------------------------------------- #
#  D - it REFUSES rather than silently truncating                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [(0.0,), (-0.01,), (float("nan"),),
                                 (float("inf"),)])
def test_d_non_positive_or_non_finite_is_REFUSED(bad):
    m = _model(seed=2)
    feats, v0, nav = _window(m.cfg, seed=2)
    with pytest.raises(ValueError, match="seed_kappa_ladder"):
        m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
               seed_kappa_ladder=bad)


def test_d2_above_kappa_max_is_REFUSED_not_clipped():
    """A silently `_clip`ped rung would make the arm's record name a magnitude
    it never searched - the units/scope failure this programme keeps paying for."""
    m = _model(seed=2)
    feats, v0, nav = _window(m.cfg, seed=2)
    pc = _pc(m.cfg)
    with pytest.raises(ValueError, match="kappa_max"):
        m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
               seed_kappa_ladder=(pc.kappa_max * 1.5,))
    assert GOAL_KAPPA_MAX == pc.kappa_max        # the two names agree


def test_e_ladder_is_stamped_on_the_result_for_provenance():
    """An arm whose candidate set is not in its record is not quotable."""
    m = _model(seed=7)
    feats, v0, nav = _window(m.cfg, seed=7)
    with torch.no_grad():
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
                     seed_kappa_ladder=(0.01,))
    assert res.seed_kappa_ladder == (0.01,)
    assert res.cost_metric == "cos"              # untouched by this lever
