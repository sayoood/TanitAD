"""refcv5 WP-7 / ``E-DDA-2b`` — the four-family sub-metric selector, its score
composition, and the rule-based targets it is trained against.

Every test names its kind in its own docstring, and the kinds are the three the
programme's operating standard requires:

* **FAILS WITHOUT THE FEATURE** — the test that would go green if the feature
  were removed is not a test of the feature.
* **CONTROL THAT MUST READ A KNOWN VALUE** — an identity, a floor, or a
  no-information reading. Without one of these a probe manufactures results:
  MEASURED 2026-08-22, **four distinct failures in one ridge probe in one
  afternoon**, each producing a confident publishable-looking number, and
  three of the four were caught ONLY because a control read the same value as
  the thing being measured.
* **DELIBERATE REGRESSION** — an arm that MUST fail, or the instrument cannot
  see what it is cited for.

⚠️ Everything here is CPU-sized and tiny on purpose (d=32, N<=16, S=9). The
dev-box GPU is contended and nothing measured at this size is a model claim.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.refs import refc_selector as S
from tanitad.refs import refc_selector_targets as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cfg(**kw) -> S.SelectorConfig:
    """A tiny but structurally faithful config: 1 coarse + 3 fine layers."""
    base = dict(d=32, n_heads=4, ff_mult=2, coarse_layers=1, fine_layers=3,
                top_k=8, n_freq=4)
    base.update(kw)
    return S.SelectorConfig(**base)


def _fan(b: int = 2, n: int = 12, s: int = 9, seed: int = 0) -> torch.Tensor:
    """A random-but-forward-moving fan ``[B, N, S, 2]``."""
    g = torch.Generator().manual_seed(seed)
    step = torch.rand(b, n, s - 1, generator=g) * 4.0 + 0.5
    lat = (torch.rand(b, n, s - 1, generator=g) - 0.5) * 0.6
    x = torch.cat([torch.zeros(b, n, 1), step.cumsum(-1)], dim=-1)
    y = torch.cat([torch.zeros(b, n, 1), lat.cumsum(-1)], dim=-1)
    return torch.stack([x, y], dim=-1)


def _scene(b: int = 2, m: int = 10, dim: int = 16, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, m, dim, generator=g)


def _build(cfg=None, scene_dim: int = 16, n_steps: int = 9, seed: int = 7,
           **kw):
    torch.manual_seed(seed)
    return S.SubMetricSelector(scene_dim, n_steps, cfg or _cfg(**kw))


def _fast_and_slow():
    """The synthetic 2-candidate fan used by the deliberate regression.

    Candidate 0 — FAST and COLLIDING: 0 -> 60 m in 9 waypoints (7.5 m spacing,
    so it lands EXACTLY on the obstacle at x = 45).
    Candidate 1 — SLOW and SAFE: 0 -> 30 m, nearest approach 15 m.

    ⚠️ The 7.5 m spacing is chosen so a waypoint coincides with the obstacle.
    ``rewards._collision`` tests WAYPOINTS against obstacle centres, not
    segments — a fan whose waypoints straddle an obstacle passes through it
    undetected. That is a real property of the predicate and this fixture makes
    it visible rather than accidentally hiding it.
    """
    s = 9
    fast = torch.linspace(0.0, 60.0, s)
    slow = torch.linspace(0.0, 30.0, s)
    x = torch.stack([fast, slow])[None]                     # [1, 2, S]
    y = torch.zeros_like(x)
    cand = torch.stack([x, y], dim=-1)                      # [1, 2, S, 2]
    obstacles = torch.tensor([[[45.0, 0.0]]])               # [B=1, K=1, 2]
    ctx = {"v0": torch.tensor([20.0]), "obstacles": obstacles, "dt": 0.5}
    return cand, ctx


# ===========================================================================
# 1. Shapes and finiteness through coarse -> top-K -> fine
# ===========================================================================
def test_shapes_coarse_topk_fine():
    """FAILS WITHOUT THE FEATURE. The two-stage structure must actually be two
    stages: the fine scorer sees ``top_k`` survivors, not the whole fan."""
    b, n, s = 2, 12, 9
    sel = _build(_cfg(top_k=5))
    out = sel(_fan(b, n, s), _scene(b))

    assert out["score"].shape == (b, n)
    assert out["score_coarse"].shape == (b, n)
    assert out["score_fine"].shape == (b, 5)
    assert out["topk_idx"].shape == (b, 5)
    assert out["sel_idx_selector"].shape == (b,)
    assert out["n_candidates"] == n
    assert out["n_scored"] == 5
    assert out["pruned"] is True
    for h in S.HEAD_NAMES:
        assert out["coarse_logits"][h].shape == (b, n)
        assert out["fine_logits"][h].shape == (b, 5)
    assert torch.isfinite(out["score_fine"]).all()
    assert torch.isfinite(out["score_coarse"]).all()

    # the pruned candidates carry the sentinel and NOTHING else
    pruned_mask = torch.ones(b, n, dtype=torch.bool)
    pruned_mask.scatter_(1, out["topk_idx"], False)
    assert (out["score"][pruned_mask] == S.NOT_SCORED).all()
    assert (out["score"][~pruned_mask] > 0.0).all()

    # ⭐ the sentinel is unreachable by a real score: the composition is a
    # product of a sigmoid and a convex sum of sigmoids, so it lies in (0, 1).
    assert S.NOT_SCORED < 0.0
    assert out["sel_idx_selector"].max() < n


def test_no_prune_when_topk_exceeds_fan():
    """CONTROL that must read a known value. ``top_k >= N`` must score EVERY
    candidate — this is the exact-scoring path the coupling tests rely on, and
    if it silently pruned they would be testing something else."""
    b, n = 2, 6
    out = _build(_cfg(top_k=64))(_fan(b, n), _scene(b))
    assert out["n_scored"] == n
    assert out["pruned"] is False
    assert (out["score"] > S.NOT_SCORED).all()
    assert torch.equal(out["topk_idx"],
                       torch.arange(n).unsqueeze(0).expand(b, n))


def test_backward_reaches_every_head():
    """FAILS WITHOUT THE FEATURE. A head with no gradient is a dead seam that
    reports as a null result — the failure mode `refc_agents` documents at its
    own seam. Every head's bias must receive gradient from the loss."""
    b, n = 2, 8
    sel = _build(_cfg(top_k=4))
    cand, scene = _fan(b, n), _scene(b)
    out = sel(cand, scene)
    tgt = {h: torch.rand(b, n) for h in S.HEAD_NAMES}
    res = S.selector_losses(out, tgt, cfg=sel.cfg,
                            generator=torch.Generator().manual_seed(0))
    res["loss"].backward()
    for h in S.HEAD_NAMES:
        for mod in (sel.coarse_heads, sel.fine_heads):
            g = mod[h].bias.grad
            assert g is not None and torch.isfinite(g).all()
            assert g.abs().sum() > 0.0, f"head {h} received no gradient"


# ===========================================================================
# 2. The score composition — three controls
# ===========================================================================
def test_equal_head_logits_give_a_constant_and_argmax_zero():
    """CONTROL that must read a known value. With every head logit forced
    EQUAL the composition must be constant across candidates and the argmax
    must be index 0 — i.e. there is no hidden tie-break bias (a positional
    prior, a sorted-index leak, a scatter that favours late indices).

    ⛔ Why this control and not a smoke test: a selector with an index-shaped
    bias would look perfectly healthy on random inputs and would then silently
    prefer the same anchor on every window — which is precisely the
    ``D-REFC-DDAUDIT-3`` shape (``sel_idx_base`` unchanged on 201/201).
    """
    b, n = 2, 10
    sel = _build(_cfg(top_k=64))               # no prune: every candidate scored
    with torch.no_grad():
        for mod in (sel.coarse_heads, sel.fine_heads):
            for h in S.HEAD_NAMES:
                mod[h].weight.zero_()
                mod[h].bias.zero_()
    out = sel(_fan(b, n), _scene(b))

    # all logits are exactly 0 -> every sigmoid is 0.5 -> score = 0.5 * 6/12
    assert torch.allclose(out["score"], torch.full((b, n), 0.25), atol=1e-6)
    assert (out["score"].std(dim=1) == 0).all()
    assert (out["sel_idx_selector"] == 0).all()


def test_composition_reproduces_a_hand_computed_value():
    """CONTROL that must read a known value. The composition is checked
    against a number computed by hand, not against itself.

    sigma(NC)=0.75, sig(TTC)=0.5, sig(prog)=0.8, sig(comf)=0.25,
    sig(compl)=0.9, sig(tac)=0.1.

      w_c = w_t = 0 : 0.75 * (5*0.5 + 5*0.8 + 2*0.25) / 12
                    = 0.75 * 7.0 / 12          = 0.437500
      w_c = 2, w_t = 3 : 0.75 * (7.0 + 1.8 + 0.3) / 17
                    = 0.75 * 9.1 / 17          = 0.401470588...
    """
    def lg(p: float) -> torch.Tensor:
        return torch.tensor([math.log(p / (1.0 - p))], dtype=torch.float64)

    logits = {"nc": lg(0.75), "ttc": lg(0.5), "progress": lg(0.8),
              "comfort": lg(0.25), "compliance": lg(0.9), "tactical": lg(0.1)}

    got = S.compose_score(logits)
    assert float(got) == pytest.approx(0.75 * 7.0 / 12.0, abs=1e-9)
    assert float(got) == pytest.approx(0.4375, abs=1e-9)

    got2 = S.compose_score(logits, w_compliance=2.0, w_tactical=3.0)
    assert float(got2) == pytest.approx(0.75 * 9.1 / 17.0, abs=1e-9)
    assert float(got2) == pytest.approx(0.4014705882352941, abs=1e-9)


def test_zero_weights_reproduce_the_pdms_twelve_denominator_exactly():
    """CONTROL that must read a known value. At ``w_c = w_t = 0`` the score is
    the pure PDMS-shaped composition with a denominator of exactly 12, and the
    two extra heads are INERT — moving them across their whole range must not
    change the score by a single float."""
    torch.manual_seed(3)
    base = {h: torch.randn(4, 7, dtype=torch.float64) for h in S.HEAD_NAMES}
    ref = S.compose_score(base)

    manual = (torch.sigmoid(base["nc"])
              * (5.0 * torch.sigmoid(base["ttc"])
                 + 5.0 * torch.sigmoid(base["progress"])
                 + 2.0 * torch.sigmoid(base["comfort"])) / 12.0)
    assert torch.equal(ref, manual)

    for wild in (-40.0, 0.0, 40.0):
        alt = dict(base)
        alt["compliance"] = torch.full_like(base["compliance"], wild)
        alt["tactical"] = torch.full_like(base["tactical"], wild)
        assert torch.equal(S.compose_score(alt), ref)

    # ...and the moment a weight is non-zero they DO move it (the switch works)
    moved = S.compose_score(
        {**base, "compliance": torch.full_like(base["compliance"], 40.0)},
        w_compliance=2.0)
    assert not torch.allclose(moved, ref)
    assert S.SelectorConfig().as_dict()["score_denominator"] == 12.0


def test_negative_composition_weight_is_refused():
    """CONTROL. A negative weight INVERTS a head — the score would reward
    NON-compliance. Refused at construction rather than clamped, so the arm
    cannot silently run the inverted objective."""
    with pytest.raises(ValueError, match="NEGATIVE"):
        S.SelectorConfig(w_compliance=-1.0)


# ===========================================================================
# 3. Candidate self-attention — the ONE place coupling is intended
# ===========================================================================
def _coupling_delta(self_attn: bool) -> float:
    """|score(cand 0) before - after| when a DIFFERENT candidate is changed."""
    b, n = 1, 6
    sel = _build(_cfg(top_k=64, self_attn=self_attn), seed=11)
    scene = _scene(b)
    cand = _fan(b, n, seed=5)
    with torch.no_grad():
        a = sel(cand, scene)["score"][0, 0]
        alt = cand.clone()
        alt[0, 4] = alt[0, 4] * 0.25 + 3.0       # rewrite candidate 4 only
        c = sel(alt, scene)["score"][0, 0]
    assert torch.equal(cand[0, 0], alt[0, 0]), "candidate 0 must be untouched"
    return float((a - c).abs())


def test_candidate_self_attention_actually_couples_candidates():
    """FAILS WITHOUT THE FEATURE — and carries its own control.

    Changing candidate 4 must change candidate 0's score. This is the ONE place
    in the programme where candidates are allowed to see each other, so it is
    proved rather than asserted.

    ⚠️ A PERMUTATION test would NOT prove this: self-attention is permutation-
    EQUIVARIANT, so re-ordering the fan leaves every candidate's score
    unchanged whether or not the coupling exists. What discriminates is
    changing a candidate's CONTENT.

    ⛔ The paired control is the whole point: with ``self_attn=False`` the same
    edit must leave candidate 0 EXACTLY alone. Without it, "the score moved"
    could be float noise from a different reduction order, which is the class of
    error MEASURED at ~1.19e-07 on a subset decode in `test_refc_agents`.
    """
    coupled = _coupling_delta(self_attn=True)
    independent = _coupling_delta(self_attn=False)

    assert independent < 1e-7, (
        "with candidate self-attention OFF the candidates must be independent "
        "— this is the structural fact `SelectionConfig.anchor_prefilter` "
        f"(S2b) rests on; got {independent:.3e}")
    assert coupled > 1e-5, (
        f"candidate self-attention is not coupling anything: {coupled:.3e}")
    assert coupled > 100.0 * max(independent, 1e-12)


def test_self_attention_off_matches_a_per_candidate_decode():
    """CONTROL that must read a known value. With ``self_attn=False``, scoring
    the fan one candidate at a time must reproduce the full-fan scores. This is
    the ``anchor_prefilter`` exactness argument in miniature.

    ⚠️ The tolerance is 1e-5, not 0. MEASURED (`test_refc_agents::
    test_candidate_independence_survives_the_agent_branch`): a subset decode
    disagrees with a full decode by ~1.19e-07 with NOTHING changed, because a
    different candidate count selects a different GEMM reduction order. The
    STRUCTURAL claim is what matters; bit equality was never true here.
    """
    b, n = 1, 5
    sel = _build(_cfg(top_k=64, self_attn=False), seed=13)
    scene, cand = _scene(b), _fan(b, n, seed=6)
    with torch.no_grad():
        full = sel(cand, scene)["score"][0]
        one = torch.stack([sel(cand[:, i:i + 1], scene)["score"][0, 0]
                           for i in range(n)])
    assert torch.allclose(full, one, atol=1e-5)


def test_agents_without_the_branch_are_refused_not_ignored():
    """CONTROL. Feeding agent tokens to a selector built without the agent
    branch must RAISE. A silent drop reads exactly like "agents made no
    difference", which is a manufactured null result."""
    b, n = 1, 4
    sel = _build(_cfg(top_k=64, cross_agent=False))
    with pytest.raises(ValueError, match="cross_agent"):
        sel(_fan(b, n), _scene(b), agents=torch.randn(b, 3, 32))


def test_agent_branch_runs_and_a_fully_padded_row_is_not_nan():
    """FAILS WITHOUT THE FEATURE. A window with NO detected agents is an
    ordinary empty road, and an all-True ``key_padding_mask`` row normalises a
    softmax over all ``-inf``: the output is NaN and the BACKWARD stays NaN even
    after ``nan_to_num``. Both directions are checked."""
    b, n, k = 2, 6, 3
    sel = _build(_cfg(top_k=64, cross_agent=True))
    agents = torch.randn(b, k, 32, requires_grad=True)
    pad = torch.zeros(b, k, dtype=torch.bool)
    pad[1] = True                                   # row 1 has NO agents
    out = sel(_fan(b, n), _scene(b), agents=agents, agent_pad=pad)
    assert torch.isfinite(out["score"]).all()
    out["score"].sum().backward()
    assert torch.isfinite(agents.grad).all()


# ===========================================================================
# 4. The targets — collision control and the deliberate regression
# ===========================================================================
def test_collision_target_reads_zero_with_no_agents_and_fires_with_one():
    """CONTROL that must read a known value, BOTH directions in one test.

    * a scene with NO agents -> EXACTLY 0 collisions;
    * the same fan with an agent ON candidate 0's path -> EXACTLY 1.

    ⛔ Only the pair is admissible. A "0 collisions" reading alone is
    indistinguishable from a predicate that cannot fire at all — which is the
    E-DETECT-1 failure in another costume: an all-zero FLOOR arm scores at
    chance and makes every other arm look like it wins.
    """
    cand, ctx = _fast_and_slow()

    empty = {k: v for k, v in ctx.items() if k != "obstacles"}
    nc_empty, _ = T.nc_target(cand, T.broadcast_scene(empty))
    assert nc_empty.shape == (1, 2)
    assert int((nc_empty == 0.0).sum()) == 0
    assert torch.equal(nc_empty, torch.ones_like(nc_empty))

    nc, _ = T.nc_target(cand, T.broadcast_scene(ctx))
    assert int((nc == 0.0).sum()) == 1
    assert float(nc[0, 0]) == 0.0, "the fast candidate drives through it"
    assert float(nc[0, 1]) == 1.0, "the slow candidate stops 15 m short"


def test_progress_only_selector_picks_the_colliding_candidate():
    """DELIBERATE REGRESSION. A progress-only selector MUST prefer the fast,
    COLLIDING candidate over the slow, safe one.

    ⛔ If it does not, the collision readout is blind and the whole E-DDA-2b
    panel is VOID — because a selector that cannot be hacked by progress is one
    whose collision term is firing for the wrong reason, and no later "the fan
    got safer" result would mean anything. This is the
    ``rewards.HACKABLE_WEIGHTS = {"progress": 1.0}`` arm, ported to the
    selector: the audit MUST flag it, and it is shipped precisely so it can.

    The progress-only selector is CONSTRUCTED AT CONVERGENCE rather than
    trained — a 3-epoch CPU run would be a weaker instrument, not a stronger
    one, and its outcome would depend on the seed rather than on the objective.
    """
    cand, ctx = _fast_and_slow()
    ctxb = T.broadcast_scene(ctx)
    cfg = T.TargetConfig(dt=0.5)

    prog, _ = T.progress_target(cand, ctxb, cfg)
    nc, _ = T.nc_target(cand, ctxb, cfg)

    # the fixture is honest: fast really does progress more, and really collides
    assert float(prog[0, 0]) > float(prog[0, 1])
    assert 0.0 < float(prog[0, 1]) < float(prog[0, 0]) < 1.0, \
        "both targets must be UNSATURATED or the ranking is a clamp artifact"
    assert float(nc[0, 0]) == 0.0 and float(nc[0, 1]) == 1.0

    # a progress-only selector: only the progress head discriminates.
    def logit(p):
        p = p.clamp(1e-4, 1 - 1e-4)
        return torch.log(p / (1 - p))

    flat = torch.zeros_like(prog)
    heads = {h: flat.clone() for h in S.HEAD_NAMES}
    heads["progress"] = logit(prog)
    score = S.compose_score(heads)
    assert int(score.argmax(dim=1)) == 0, (
        "the progress-only objective did NOT pick the colliding candidate — "
        "the deliberate regression failed to regress, so the collision readout "
        "cannot be trusted and the panel is VOID")

    # ...and the same fan with the NC head switched on flips the pick. This is
    # the control on the control: it proves the fixture is winnable both ways.
    heads["nc"] = logit(nc)
    assert int(S.compose_score(heads).argmax(dim=1)) == 1


def test_targets_refuse_a_context_carrying_the_answer():
    """CONTROL. The environment targets must REFUSE a ctx holding the ego's
    logged future or a model-produced ranking. This is the SEL-1 winner's-curse
    door lock; a warning would be a log line nobody re-reads."""
    cand, ctx = _fast_and_slow()
    for bad in ("gt_traj", "sel_score", "anchor_logits", "nav_true"):
        with pytest.raises(KeyError):
            T.selector_targets(cand, {**ctx, bad: torch.zeros(1)})
    # ...and the same call without it succeeds — the same-breath control that
    # proves the refusal is about the key and not about the call.
    ok = T.selector_targets(cand, ctx)
    assert "nc" in ok["targets"]


def test_comfort_is_masked_where_undefined_and_scored_where_defined():
    """CONTROL that must read a known value. A candidate that does not move has
    NO RIDE to be comfortable about, so its comfort target is UNDEFINED
    (masked), not 0. Scoring it as 0 would teach the head that stillness is
    harshness; ramping it to 1 hands the frozen policy free reward (MEASURED:
    0.70 at the default weights, which is why ``_motion_gate`` exists)."""
    s = 9
    still = torch.zeros(s, 2)
    moving = torch.stack([torch.linspace(0, 30, s), torch.zeros(s)], dim=-1)
    cand = torch.stack([still, moving])[None]              # [1, 2, S, 2]
    ctx = {"v0": torch.tensor([10.0]), "dt": 0.5}
    tgt, mask, extras = T.comfort_target(cand, ctx, T.TargetConfig(dt=0.5))
    assert bool(mask[0, 0]) is False, "a standstill must be MASKED"
    assert bool(mask[0, 1]) is True
    assert 0.0 <= float(tgt[0, 1]) <= 1.0
    assert "peak_g" in extras and torch.isfinite(extras["peak_g"]).all()
    # the straight constant-speed candidate is maximally comfortable: no jerk,
    # no lateral acceleration, no friction load. A known value, not a range.
    assert float(tgt[0, 1]) == pytest.approx(1.0, abs=1e-6)
    assert float(extras["peak_g"][0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_compliance_is_omitted_rather_than_invented_without_a_tolerance():
    """CONTROL. Without a derived tau the compliance head is OMITTED and the
    reason is recorded — it is never computed against an invented tolerance,
    and never zero-filled (a zero-filled head trains toward 'never complies')."""
    cand, ctx = _fast_and_slow()
    nav = torch.tensor([1])                                  # "left"
    res = T.selector_targets(cand, ctx, nav_cmd=nav)
    assert "compliance" not in res["targets"]
    assert "compliance" in res["provenance"]["omitted"]
    assert "tolerance" in res["provenance"]["reasons"]["compliance"]

    cfg = T.TargetConfig(dt=0.5, compliance_tau_rad=0.10)
    res2 = T.selector_targets(cand, ctx, cfg=cfg, nav_cmd=nav)
    assert "compliance" in res2["targets"]
    # both fixture candidates drive dead straight, so under "left" both FAIL
    # compliance while the window IS informative: a known value.
    assert res2["masks"]["compliance"].all()
    assert float(res2["targets"]["compliance"].sum()) == 0.0

    # ...and a genuinely left-turning candidate complies. The same-breath
    # control: without it, "0.0" is indistinguishable from a dead predicate.
    s = 9
    left = torch.stack([torch.linspace(0, 30, s),
                        torch.linspace(0, 30, s) ** 2 * 0.02], dim=-1)
    got, m = T.compliance_target(left[None, None], nav, 0.10)
    assert bool(m[0, 0]) and float(got[0, 0]) == 1.0


def test_compliance_is_undefined_for_follow_and_straight():
    """CONTROL that must read a known value. ``follow`` and ``straight`` carry
    no commanded SIDE, so the predicate is UNDEFINED, not failed. Scoring those
    windows lets the head reach a fine BCE by predicting 'never complies'."""
    s = 9
    left = torch.stack([torch.linspace(0, 30, s),
                        torch.linspace(0, 30, s) ** 2 * 0.02], dim=-1)
    fan = left[None, None]
    for cmd in (T.NAV_COMMANDS.index("follow"),
                T.NAV_COMMANDS.index("straight")):
        _, m = T.compliance_target(fan, torch.tensor([cmd]), 0.10)
        assert not bool(m.any()), f"nav={cmd} must be uninformative"
    _, m = T.compliance_target(fan, torch.tensor([T.NAV_COMMANDS.index("left")]),
                               0.10)
    assert bool(m.all())


def test_tactical_target_is_flagged_gt_derived_and_reports_its_horizon():
    """CONTROL. The tactical head is the ONE quarantined target. It must be
    reported as GT-derived, must be OFF in the composition by default, and must
    declare the horizon mismatch (label 2 s vs candidate 6 s) rather than
    silently comparing two different quantities."""
    cand, ctx = _fast_and_slow()
    lat = torch.tensor([0])                       # lane_keep
    lon = torch.tensor([0])                       # brake_stop
    res = T.selector_targets(cand, ctx, nav_cmd=None,
                             lat_label=lat, lon_label=lon)
    assert "tactical" in res["targets"]
    assert res["provenance"]["gt_derived_heads"] == ["tactical"]
    assert "GT-DERIVED" in res["provenance"]["tactical_quarantine"]
    assert res["extras"]["tactical"]["horizon_mismatch"] is True
    assert res["extras"]["tactical"]["label_horizon_steps"] == 20
    assert S.SelectorConfig().w_tactical == 0.0
    assert "tactical" in S.GT_DERIVED_HEADS
    assert "tactical" not in S.ENV_HEADS


# ===========================================================================
# 5. n is returned, not just the loss
# ===========================================================================
def test_losses_return_every_n_not_just_the_loss():
    """CONTROL. ``n_candidates`` and ``n_scored`` — and a per-head valid count
    — are part of the contract, not telemetry.

    ⛔ MEASURED 2026-08-22: a probe with 2,050 features on ~700 rows correctly
    chose maximal regularisation and every arm read +0.0000. It was
    UNDERPOWERED BY CONSTRUCTION and looked exactly like an absence. The only
    thing that separates the two is the n printed beside the number.
    """
    b, n, k = 2, 12, 5
    sel = _build(_cfg(top_k=k))
    out = sel(_fan(b, n), _scene(b))
    tgt = {h: torch.rand(b, n) for h in S.HEAD_NAMES}
    masks = {"comfort": torch.zeros(b, n, dtype=torch.bool)}   # fully undefined
    res = S.selector_losses(out, tgt, masks, cfg=sel.cfg,
                            generator=torch.Generator().manual_seed(1))

    assert res["n_candidates"] == n
    assert res["n_scored"] == k
    assert res["n_live_candidates"] == b * n
    for h in S.HEAD_NAMES:
        assert f"n_{h}_coarse" in res and f"n_{h}_fine" in res
    assert res["n_nc_coarse"] == b * n
    assert res["n_nc_fine"] == b * k
    # ⭐ the fully-masked head reads n == 0 AND loss == 0. Without the n, that
    # 0.0 is indistinguishable from a perfectly-fit head.
    assert res["n_comfort_coarse"] == 0
    assert float(res["loss_comfort_coarse"].detach()) == 0.0
    assert res["heads_gt_derived"] == ["tactical"]
    assert res["n_margin_coarse"] > 0
    assert torch.isfinite(res["loss"])


def test_selector_losses_refuses_an_empty_target_set():
    """CONTROL. A loss with no supervised head is not a loss — it would train
    nothing while reporting a finite number every step."""
    b, n = 1, 4
    sel = _build(_cfg(top_k=64))
    out = sel(_fan(b, n), _scene(b))
    with pytest.raises(ValueError, match="no known head targets"):
        S.selector_losses(out, {"not_a_head": torch.rand(b, n)}, cfg=sel.cfg)


def test_coverage_reports_a_zero_spread_target():
    """CONTROL that must read a known value. A target identical for every
    candidate cancels EXACTLY in any ranking objective — it is present and
    carries no signal. MEASURED (A0, 240 held-out windows): the first headway
    component fired on 28.3 % of windows with a MEDIAN FAN SPREAD OF 0.0000.
    Only a spread field can say so."""
    cand, ctx = _fast_and_slow()
    res = T.selector_targets(cand, ctx)
    # no lead in this fixture -> the ttc head is OMITTED, not silently zeroed
    assert "ttc" in res["provenance"]["omitted"]
    assert res["coverage"]["ttc"]["present"] is False
    # nc DOES discriminate here (one collides, one does not)
    assert res["coverage"]["nc"]["spread_where_defined"] > 0.0
    assert res["coverage"]["nc"]["n_defined"] == 2
    assert res["coverage"]["progress"]["frac_defined"] == 1.0


# ===========================================================================
# 6. The config is a run record
# ===========================================================================
def test_config_as_dict_is_serialisable_and_names_the_absent_map_head():
    """CONTROL. ``config.json['seams']['selector']`` must be able to rebuild
    the model. It must also record that DAC is ABSENT — a missing factor that
    is not written down reads later as an oversight rather than as the pinned
    fact that PhysicalAI carries no map."""
    import json
    d = S.SelectorConfig(w_compliance=2.0, w_tactical=1.0).as_dict()
    json.loads(json.dumps(d))                       # must round-trip
    assert d["d"] == 256 and d["n_heads"] == 8
    assert d["coarse_layers"] == 1 and d["fine_layers"] == 3
    assert d["top_k"] == 32
    assert d["head_names"] == list(S.HEAD_NAMES)
    assert d["score_denominator"] == 15.0
    assert "no map" in d["dac_head"]
    assert d["pdms_weights"] == {"ttc": 5.0, "progress": 5.0, "comfort": 2.0}
    # the fields the plan calls the ablatable switches are the ones present
    assert d["w_compliance"] == 2.0 and d["w_tactical"] == 1.0
    assert d["self_attn"] is True and d["cross_agent"] is False


def test_param_count_is_pinned_and_the_plan_estimate_priced_the_ablation():
    """CONTROL that must read a known value — and it is the value that
    CONTRADICTS the plan, which is why it is pinned rather than described.

    ``REFCV5_DESIGN_PLAN.md`` §7.1 prices WP-7 at **"≈ 3.5 M, outside the
    generator"**. MEASURED by building at ``scene_dim=704, n_steps=8``:

        default                      4,530,444
        cross_agent=True             5,585,164
        self_attn=False              3,475,724   <- what "3.5 M" actually is

    ⇒ the estimate priced the arm WITHOUT its candidate self-attention, i.e.
    the ablation rather than the arm. The 1,054,720 gap is exactly the four
    self-attention blocks (4 x [262,144 + 1,024 + 512]).

    ⛔ A described delta rots; an asserted one cannot. The launch preflight
    must pin the measured number the way ``REGISTERED_DELTA_KEYS_V4`` pins
    v4's. If this test fails, the module changed size — update the plan and
    the preflight in the SAME turn, do not edit the number here alone.
    """
    def n_params(**kw) -> int:
        torch.manual_seed(0)
        m = S.SubMetricSelector(scene_dim=704, n_steps=8,
                                cfg=S.SelectorConfig(**kw))
        return sum(p.numel() for p in m.parameters())

    default = n_params()
    with_agents = n_params(cross_agent=True)
    no_self = n_params(self_attn=False)

    assert default == 4_530_444
    assert with_agents == 5_585_164
    assert no_self == 3_475_724
    # the gap IS the four self-attention blocks, to the parameter
    assert default - no_self == 4 * (256 * 256 * 4 + 256 * 4 + 2 * 256)
    assert default - no_self == 1_054_720


def test_pdms_nc_binarise_is_a_noop_on_our_binary_target():
    """CONTROL that must read a known value. V2 maps NAVSIM's NC 0.5 (an
    at-fault-LESS collision) to 0. We have no fault model, so our NC target is
    already binary and the mapping is a NO-OP. Asserting the parity is what
    makes it a fact instead of an assumption."""
    cand, ctx = _fast_and_slow()
    nc, _ = T.nc_target(cand, T.broadcast_scene(ctx))
    assert torch.equal(S.pdms_nc_binarise(nc), nc)
    # ...and it still does the V2 thing to a genuine 0.5
    half = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(S.pdms_nc_binarise(half), torch.tensor([0., 0., 1.]))
