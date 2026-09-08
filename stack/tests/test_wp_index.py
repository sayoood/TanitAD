"""refcv5 WP-B / ``E-WP-INDEX-1`` — the guard set for the waypoint index.

⛔⛔ **EVERY EXPECTATION IN THIS FILE IS A LITERAL, NEVER AN EXPRESSION OVER THE
CODE UNDER TEST.** ``CLAUDE.md`` (`e4af94f`) records four failures in one night
from cross-checks that shared the defect they checked for: a label builder that
verified its buckets against its own rounded ladder, a census whose test pinned
a set against a union with that set, an RL guard asserting *"expected value ==
whatever the code does"*, and a conditioning test whose stub could not express
the fields it was checking. Re-running the producer's own derivation and finding
agreement measures **determinism, not correctness**.

⇒ the targets here are of exactly three kinds, in order of strength:

1. **ANALYTIC** — a waypoint at a known range and bearing must address a
   computable slot, and the numbers (``6.0``, ``sqrt(436) - 20``, ``e**6``) are
   computed by hand or by ``math``, never by ``refc_wp_index``.
2. **IDENTITY** — a control whose known value is exact and structural: a
   constant address gives a bias that is bitwise EQUAL across the anchor axis;
   a shuffled address gives, for row ``b``, exactly the geometric bias of row
   ``perm[b]``.
3. **MUTATION** — the historical defect is reintroduced and the guard must go
   RED. ``raw/MUTATION_LOG.md`` in the WP-B package carries the GREEN/RED runs;
   the two most important mutations are also pinned here so a future edit that
   silently disables the guard fails in CI rather than in an audit.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.refs import refc_agents as ra
from tanitad.refs import refc_wp_index as wi
from tanitad.refs.refc import RefCModel, param_breakdown, refc_smoke_config


# ---------------------------------------------------------------------------
# The scene, its literals, and why they are literals
# ---------------------------------------------------------------------------
#: A straight-ahead plan: four waypoints on the ego x-axis at 5/10/15/20 m.
STRAIGHT = torch.tensor([[[[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]]]])
#: A left-going plan reaching (20, +6) — the SAME terminal range band, a
#: different bearing, so the two plans can only be told apart by GEOMETRY.
LEFT = torch.tensor([[[[5.0, 0.5], [10.0, 1.5], [15.0, 3.5], [20.0, 6.0]]]])
#: three agents: dead ahead at 20 m, left at (20, +6), and close ahead at 5 m.
AGENTS = torch.tensor([[[20.0, 0.0], [20.0, 6.0], [5.0, 0.0]]])


def test_ANALYTIC_the_address_is_a_metric_relation_not_an_array_position():
    """⭐ THE CLOSED-FORM CHECK. A straight-ahead plan whose LAST waypoint sits
    exactly on agent 0, 6 m to the right of agent 1, and whose FIRST waypoint
    sits exactly on agent 2.

    Every number here is computable by hand from the scene definition above and
    none of them is read back from the implementation:

    * ``d_min`` — 0 m to the agent a waypoint lands on, exactly **6.0** m to
      the one 6 m to its left, 0 m to the one under the first waypoint;
    * ``s_star`` — **3** (the last of four) and **0** (the first), so the index
      also carries WHEN the plan is closest;
    * ``lat`` — the plan's heading at ``s* = 3`` is ``+x``, whose left-normal is
      ``+y``, so the cross-track offset of an agent at ``y = +6`` is exactly
      **+6.0** and its along-track offset is exactly **0.0**.

    ⛔ ``d_min[0] == d_min[2] == 0`` while ``s_star`` differs is the whole point:
    an index by ARRAY POSITION could never produce that pair.
    """
    g = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    assert g["d_min"].shape == (1, 1, 3)
    assert g["d_min"].reshape(-1).tolist() == [0.0, 6.0, 0.0]
    assert g["s_star"].reshape(-1).tolist() == [3, 3, 0]
    assert g["tau"].reshape(-1).tolist() == [1.0, 1.0, 0.0]
    assert g["lat"].reshape(-1).tolist() == [0.0, 6.0, 0.0]
    assert g["lon"].reshape(-1).tolist() == [0.0, 0.0, 0.0]


def test_ANALYTIC_range_and_bearing_against_hand_trigonometry():
    """⭐ The polar half of the address, against ``math``, not against us.

    Agent 1 is at ``(20, 6)``; the closest waypoint of the straight plan is
    ``(20, 0)``. Then, exactly:

    * ``d_range = |(20, 6)| - |(20, 0)| = sqrt(436) - 20``
    * ``cos_db = (20*20 + 0*6) / (20 * sqrt(436)) = 20 / sqrt(436)``
    * ``sin_db = (20*6 - 0*20) / (20 * sqrt(436)) =  6 / sqrt(436)``
    """
    g = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    r = math.sqrt(436.0)
    assert g["d_range"][0, 0, 1].item() == pytest.approx(r - 20.0, abs=1e-4)
    assert g["cos_db"][0, 0, 1].item() == pytest.approx(20.0 / r, abs=1e-6)
    assert g["sin_db"][0, 0, 1].item() == pytest.approx(6.0 / r, abs=1e-6)
    # ...and the two agents a waypoint lands on are at zero bearing offset.
    assert g["cos_db"][0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert g["sin_db"][0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_ANALYTIC_the_left_plan_addresses_the_LEFT_agent():
    """⭐ THE DISCRIMINATING CONTROL: the same agent set, a different plan, and
    the addressed slot MOVES. A positive assertion on the straight plan alone
    would pass happily on an implementation that always returns slot 0."""
    gs = wi.waypoint_agent_geometry(STRAIGHT, AGENTS)
    gl = wi.waypoint_agent_geometry(LEFT, AGENTS)
    assert int(gs["d_min"].argmin()) == 0        # straight -> the agent ahead
    assert int(gl["d_min"].argmin()) == 1        # left     -> the agent left
    assert gl["d_min"][0, 0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert gl["d_min"][0, 0, 0].item() == pytest.approx(6.0, abs=1e-6)


def test_ANALYTIC_attention_mass_ratio_is_exp_of_the_bias_gap():
    """⭐⭐ THE END-TO-END CLOSED-FORM CHECK — the address really steers the
    attention, and by a computable amount.

    Construction: every agent token is made **identical**, so every content
    logit is equal and the softmax reduces to ``softmax(bias)`` exactly. The
    bias head is then set by hand to ``b = -1.0 * (d_min / scale) * scale =
    -d_min`` — a weight of ``-scale`` on feature 0 and zero on the rest.

    ⇒ the attention mass on the agent at ``d_min = 0`` divided by the mass on
    the one at ``d_min = 6`` is exactly ``exp(0 - (-6)) = e**6 = 403.4288…``,
    a number produced by ``math.exp`` and by no line of ``refc_wp_index``.
    """
    cfg = wi.WaypointIndexConfig(enable=True, hidden=4, scale_m=10.0)
    head = wi.WaypointIndexBias(cfg, n_heads=1)
    # a hand-set LINEAR map: kill the GELU branch by making layer 0 the
    # identity-on-feature-0 and layer 1 read only that unit.
    with torch.no_grad():
        head.mlp[0].weight.zero_(); head.mlp[0].bias.zero_()
        head.mlp[0].weight[0, 0] = 100.0          # 100 * (d/10) = 10*d, GELU~id
        head.mlp[2].weight.zero_(); head.mlp[2].bias.zero_()
        head.mlp[2].weight[0, 0] = -1.0
    rel, d_min = wi.build_relation(STRAIGHT, AGENTS[:, :2], cfg)
    bias = head(rel)                                    # [1, 1, 2]
    assert bias.shape == (1, 1, 2)
    # the bias gap is analytic: -(10 * 0) - (-(10 * 6)) = 60... but GELU is only
    # ~identity for large positive inputs, so assert the RATIO of the softmax,
    # which is what the attention actually applies, against exp of the gap.
    gap = float(bias[0, 0, 0] - bias[0, 0, 1])
    w = torch.softmax(bias[0, 0], dim=-1)
    assert float(w[0] / w[1]) == pytest.approx(math.exp(gap), rel=1e-4)
    # ...and the direction is not a coin flip: the NEAR agent gets the mass.
    assert gap > 0.0 and float(w[0]) > 0.99

    # Now the literal the whole test exists for. Re-scale the head so the gap
    # is exactly 6.0, and the ratio must be e**6 = 403.42879349...
    with torch.no_grad():
        head.mlp[2].weight[0, 0] = -1.0 * (6.0 / gap)
    bias = head(rel)
    w = torch.softmax(bias[0, 0], dim=-1)
    assert float(w[0] / w[1]) == pytest.approx(403.4287934927351, rel=1e-3)


# ---------------------------------------------------------------------------
# ⛔ THE CONTROLS — each reads a KNOWN value, and each is VERIFIED not asserted
# ---------------------------------------------------------------------------
def test_CONTROL_const_index_gives_a_bias_IDENTICAL_across_the_anchor_axis():
    """⛔ KNOWN VALUE: with every waypoint at one fixed point, every anchor asks
    the SAME question of the agent set, so the emitted bias cannot depend on
    the anchor. **Bitwise** equality, not ``allclose`` — the rows are literally
    the same computation.

    ⭐ This is the ``n``/``d`` -style no-information control: an index whose
    output still varied across anchors here would be reading something other
    than the address."""
    wp = torch.randn(2, 5, 4, 2) * 10.0
    pos = torch.randn(2, 7, 2) * 10.0
    cfg = wi.WaypointIndexConfig(enable=True, mode="const", const_xy=(10.0, 0.0))
    rel, _ = wi.build_relation(wp, pos, cfg)
    head = wi.WaypointIndexBias(cfg, n_heads=2)
    with torch.no_grad():                     # zero-init would pass trivially
        torch.nn.init.normal_(head.mlp[-1].weight, std=1.0)
        torch.nn.init.normal_(head.mlp[-1].bias, std=1.0)
    bias = head(rel).reshape(2, 2, 5, 7)
    for n in range(5):
        assert torch.equal(bias[:, :, n], bias[:, :, 0]), n
    # and the treatment must NOT have that property, or the control is vacuous
    rel_g, _ = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    bias_g = head(rel_g).reshape(2, 2, 5, 7)
    assert not torch.equal(bias_g[:, :, 1], bias_g[:, :, 0])


def test_CONTROL_shuffled_index_is_EXACTLY_the_partner_rows_bias():
    """⛔ KNOWN VALUE, and it is an IDENTITY rather than a "it changed".

    ``shuffle`` permutes the WAYPOINTS across the batch, so row ``b`` is
    addressed with row ``perm[b]``'s geometry against its OWN agents. The bias
    it produces must therefore equal — bitwise — the bias the geometric mode
    would produce from ``wp[perm[b]]`` and ``pos[b]``.

    ⭐ Why this control exists at all: a ``shuffled`` arm indistinguishable from
    ``pos_only`` is what PROVED WP-A's fit ladder was feature-driven. If a
    trained ``shuffle`` arm matches the treatment, the coupling is CAPACITY,
    not CONTENT.
    """
    torch.manual_seed(0)
    wp = torch.randn(4, 3, 4, 2) * 10.0
    pos = torch.randn(4, 5, 2) * 10.0
    cfg = wi.WaypointIndexConfig(enable=True, mode="shuffle", shuffle_seed=3)
    rel_s, _ = wi.build_relation(wp, pos, cfg)
    wp_p, perm = wi.shuffle_waypoints(wp, 3)
    rel_ref, _ = wi.build_relation(
        wp_p, pos, wi.WaypointIndexConfig(enable=True))
    assert torch.equal(rel_s, rel_ref)
    # the permutation must actually move something, or the control is vacuous
    assert not torch.equal(perm, torch.arange(4))
    rel_g, _ = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    assert not torch.equal(rel_s, rel_g)


def test_CONTROL_shuffle_REFUSES_batch_one_where_it_would_be_the_identity():
    """⛔ A CONTROL THAT SILENTLY BECOMES THE TREATMENT IS WORSE THAN NO
    CONTROL. A batch permutation of one row is the identity, so ``shuffle`` at
    ``B == 1`` would 'confirm' the coupling no matter what it does — the
    guard-that-cannot-go-red failure, applied to a control."""
    wp = torch.randn(1, 3, 4, 2)
    pos = torch.randn(1, 5, 2)
    cfg = wi.WaypointIndexConfig(enable=True, mode="shuffle")
    with pytest.raises(ValueError, match="batch >= 2"):
        wi.build_relation(wp, pos, cfg)


def test_CONTROL_detached_index_passes_EXACTLY_ZERO_gradient():
    """⛔ KNOWN VALUE: **exactly** zero, not "small".

    The address is a function of the waypoints AND of the decoded agent boxes,
    so it is differentiable back into the trunk through both. The detached arm
    blocks that path; its known value is a gradient of exactly ``0`` on both
    operands, and the treatment's is non-zero — which is what makes the control
    a control rather than a second copy of the treatment."""
    # (a) the TREATMENT: gradient reaches BOTH operands, and is non-zero.
    wp = (torch.randn(2, 3, 4, 2) * 10.0).requires_grad_(True)
    pos = (torch.randn(2, 5, 2) * 10.0).requires_grad_(True)
    rel, _ = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    assert rel.requires_grad is True
    rel.sum().backward()
    assert wp.grad is not None and float(wp.grad.abs().sum()) > 0.0
    assert pos.grad is not None and float(pos.grad.abs().sum()) > 0.0

    # (b) the CONTROL: the graph is SEVERED, which is stronger than "the
    # gradient is zero" -- there is no path at all, so `rel.backward()` cannot
    # even be called. Both `.grad` slots stay exactly None.
    wp2 = (torch.randn(2, 3, 4, 2) * 10.0).requires_grad_(True)
    pos2 = (torch.randn(2, 5, 2) * 10.0).requires_grad_(True)
    rel2, _ = wi.build_relation(wp2, pos2,
                                wi.WaypointIndexConfig(enable=True,
                                                       detach=True))
    assert rel2.requires_grad is False and rel2.grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        rel2.sum().backward()
    assert wp2.grad is None and pos2.grad is None


def test_CONTROL_n_and_d_are_reported_in_the_config_record():
    """⛔ ``n`` and ``d`` printed in every table (the probe-panel rule). Here
    ``d`` is the address dimension and it travels in the run record, named."""
    d = wi.WaypointIndexConfig(enable=True).as_dict()
    assert d["feat_dim"] == 8
    assert d["features"] == list(wi.WP_INDEX_FEATURES)
    assert len(wi.WP_INDEX_FEATURES) == wi.WP_INDEX_FEAT_DIM == 8


# ---------------------------------------------------------------------------
# ⛔⛔ THE NaN TRAPS — both are the common case on a real road, not edge cases
# ---------------------------------------------------------------------------
def test_a_padded_slot_at_the_origin_produces_a_FINITE_address():
    """⛔ A padded slot decodes to ``(0, 0)`` and a stopped plan has coincident
    waypoints; a bearing computed without a clamped denominator is NaN there.
    ``NaN + (-inf)`` is ``NaN``, so the softmax over that row would be NaN
    **everywhere, including the live keys** — a poisoned batch that reads as an
    exploding planner, not as an empty scene."""
    wp = torch.zeros(1, 2, 4, 2)                    # a plan that goes nowhere
    pos = torch.zeros(1, 3, 2)                      # every slot at the origin
    rel, d_min = wi.build_relation(wp, pos, wi.WaypointIndexConfig(enable=True))
    assert torch.isfinite(rel).all()
    assert torch.isfinite(d_min).all()


def test_radius_gate_never_empties_a_query_row_and_counts_PADDING_as_empty():
    """⛔⛔ TWO failure modes, and the second is the one a radius-only test
    misses.

    (a) a query whose every agent is out of range would be all ``-inf`` and
        softmax would return NaN — its gate is dropped;
    (b) a query whose every LIVE agent is out of range but whose PADDED slots
        sit at ``(0, 0)`` — inside any radius — looks non-empty to a
        radius-only test while every surviving key is padding, so the mass
        would land entirely on slots that do not exist.
    """
    bias = torch.zeros(2, 2, 3)                     # [B*H=2, N=2, M=3]
    d_min = torch.tensor([[[50.0, 60.0, 70.0],      # b0/q0: all far
                           [1.0, 60.0, 70.0]],      # b0/q1: one near
                          [[50.0, 60.0, 0.5],       # b1/q0: near slot is PAD
                           [2.0, 3.0, 4.0]]])       # b1/q1: all near
    pad = torch.tensor([[False, False, True],
                        [False, False, True]])
    out = wi.apply_radius_gate(bias, d_min, radius_m=8.0, n_heads=1, pad=pad)
    finite = torch.isfinite(out)
    assert bool(finite.any(dim=-1).all()), "a fully -inf row would give NaN"
    # (a) b0/q0 had nothing in range at all -> the gate is dropped entirely
    assert finite[0, 0].tolist() == [True, True, True]
    # (b) b1/q0's only in-range slot is PADDING -> also treated as empty
    assert finite[1, 0].tolist() == [True, True, True]
    # ...and where the gate is legitimate it really bites
    assert finite[0, 1].tolist() == [True, False, False]
    assert finite[1, 1].tolist() == [True, True, True]


def test_radius_zero_is_the_soft_index_and_returns_the_bias_UNTOUCHED():
    bias = torch.randn(2, 3, 4)
    d_min = torch.rand(2, 3, 4) * 100.0
    out = wi.apply_radius_gate(bias, d_min, 0.0, 1, None)
    assert out is bias


# ---------------------------------------------------------------------------
# ⛔ SHAPES — the transposed bias is a SILENT mis-index whenever N == M
# ---------------------------------------------------------------------------
def test_bias_shape_is_BH_N_M_and_the_axes_are_not_interchangeable():
    """The scene is built with ``N != M`` on purpose: with ``N == M`` a
    transposed bias is a shape-valid, silent mis-index, and no assertion on
    shape alone could catch it. The mutation set carries the transposed
    variant; this test pins the LITERAL shape so the mutation has something to
    break."""
    cfg = wi.WaypointIndexConfig(enable=True)
    rel, _ = wi.build_relation(torch.randn(2, 5, 4, 2), torch.randn(2, 7, 2),
                               cfg)
    assert rel.shape == (2, 5, 7, 8)
    bias = wi.WaypointIndexBias(cfg, n_heads=3)(rel)
    assert bias.shape == (6, 5, 7)          # (B*H, N_anchors, M_slots)


def test_relation_refuses_a_normalised_or_mis_shaped_trajectory():
    """⛔ THE UNITS TRAP IN SHAPE FORM. A correct formula under the wrong units
    reads exactly like an answer (the `anchors.pt` kappa/a_lat retraction), and
    this module cannot check metres — so it checks everything it can, loudly."""
    cfg = wi.WaypointIndexConfig(enable=True)
    with pytest.raises(ValueError, match=r"\[B, N, S, 2\] metres"):
        wi.build_relation(torch.randn(2, 5, 4), torch.randn(2, 7, 2), cfg)
    with pytest.raises(ValueError, match=r"\[B, M, 2\] metres"):
        wi.build_relation(torch.randn(2, 5, 4, 2), torch.randn(2, 7), cfg)
    with pytest.raises(ValueError, match="batch mismatch"):
        wi.build_relation(torch.randn(2, 5, 4, 2), torch.randn(3, 7, 2), cfg)


def test_config_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="not in"):
        wi.WaypointIndexConfig(mode="geometric")


# ---------------------------------------------------------------------------
# ⛔⛔ REMOVABILITY — the requirement, proven three ways
# ---------------------------------------------------------------------------
def _models(index, agents=True, seed=1234):
    cfg = refc_smoke_config()
    if agents:
        cfg.agents = ra.AgentSeamConfig(enable=True, queries=6, d_model=32,
                                        depth=1, n_heads=2, enforce_band=False)
        cfg.decoder.cross_agent = True
    if index is not None:
        cfg.decoder.wp_index = index
    torch.manual_seed(seed)
    return cfg, RefCModel(cfg)


def test_off_is_not_constructed_at_all():
    """⛔ ``enable=False`` must mean NOT CONSTRUCTED, not constructed-and-idle.
    A disabled-but-present module still draws from the global RNG at
    ``__init__`` and still lands in ``state_dict``, and both break the two
    proofs below."""
    for index in (None, wi.WaypointIndexConfig(enable=False)):
        _, m = _models(index)
        assert all(ly.wp_index is None for ly in m.decoder.layers)
        assert m.decoder.wp_index_cfg is None
        assert not any("wp_index" in k for k in m.state_dict())
        assert param_breakdown(m)["wp_index"] == 0


def test_shared_params_bit_identical():
    """⛔⛔ THE ONE-VARIABLE PROOF AT STEP 0. The bias heads are attached as the
    LAST statement of ``RefCModel.__init__`` precisely so that every module
    above them draws the same RNG in both builds. Attach them anywhere earlier
    and the "index on vs index off" A/B would differ in the SEED as well as in
    the lever — invisibly, in every log."""
    _, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True))
    so, sn = off.state_dict(), on.state_dict()
    shared = [k for k in so if k in sn]
    assert len(shared) == len(so) and len(sn) > len(so)
    bad = [k for k in shared if not torch.equal(so[k], sn[k])]
    assert bad == [], bad


@pytest.mark.parametrize("index", [
    None,
    wi.WaypointIndexConfig(enable=False),
    wi.WaypointIndexConfig(enable=True),
    wi.WaypointIndexConfig(enable=True, mode="const"),
    wi.WaypointIndexConfig(enable=True, mode="shuffle"),
    wi.WaypointIndexConfig(enable=True, detach=True),
    wi.WaypointIndexConfig(enable=True, radius_m=8.0),
])
def test_planner_output_bit_identical_at_the_SHIPPED_init(index):
    """⛔⛔ THE REMOVABILITY PROOF AT THE SHIPPED INITIALISATION: with the
    index attached in ANY mode — including both controls and the radius gate —
    the planner emits **bitwise** the tensors an index-free build emits, so a
    fresh ``+index`` run starts exactly where the agent-seam run was.

    ⚠️⚠️ **AND THIS TEST IS NECESSARY, NOT SUFFICIENT — IT IS GREEN FOR A
    REASON THAT IS NOT WP-B.** MEASURED while writing it: ``agent_gate`` is
    zero-init (WP-6's own discipline), so the ENTIRE agent branch contributes
    exactly ``0`` at step 0 and this assertion passes even with a
    **deliberately corrupted** bias head. A guard whose PASS does not depend on
    the thing it guards is the `CLAUDE.md` "a check that shares the defect it
    checks for" family, and it is why the two tests immediately below exist:
    ``..._is_not_carried_by_the_index_alone`` states the confound as an
    executable fact, and ``..._at_a_LIVE_agent_gate`` re-runs the comparison in
    the regime where the bias head is actually load-bearing.

    ⚠️ Scope: dev-box CPU, float32, torch as installed.
    """
    cfg, off = _models(None)
    _, on = _models(index)
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    assert set(a) == set(b)
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]), k
            assert torch.isfinite(b[k]).all() if b[k].is_floating_point() \
                else True, k


def test_the_shipped_init_proof_is_not_carried_by_the_index_alone():
    """⛔⛔ THE DISCRIMINATING CONTROL FOR THE TEST ABOVE, and it MEASURES
    the confound rather than warning about it in prose.

    A bias head filled with garbage still leaves the planner bitwise unchanged
    at the shipped init — because ``agent_gate`` is 0 and multiplies the whole
    agent branch away. ⇒ the previous test's PASS is evidence about
    ``agent_gate``, not about WP-B's zero-init, and anyone reading it as the
    latter is reading a determinism check as a correctness check.
    """
    cfg, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True))
    with torch.no_grad():
        for ly in on.decoder.layers:               # deliberately corrupted
            torch.nn.init.normal_(ly.wp_index.mlp[-1].weight, std=10.0)
            torch.nn.init.normal_(ly.wp_index.mlp[-1].bias, std=10.0)
        assert all(float(ly.agent_gate) == 0.0 for ly in on.decoder.layers)
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    assert torch.equal(a["traj"], b["traj"]), (
        "if this ever fails the confound is gone and the note above is stale")


def test_at_a_LIVE_agent_gate_the_zero_init_index_agrees_to_float32_ROUNDING():
    """⛔ THE HONEST NUMBER, and it is NOT bit-identical.

    With ``agent_gate`` opened to 1.0 the agent branch is live, so the
    zero-init bias head is finally load-bearing — and the two arms then differ
    by **1.907e-06 max-abs on ``traj``** (MEASURED, dev-box CPU, float32, smoke
    config). The cause is not the bias, which is exactly ``0.0``: it is that
    supplying a float ``attn_mask`` AT ALL changes which scaled-dot-product
    kernel torch dispatches to, and the two kernels do not round identically.

    ⇒ **State it; do not claim bit-identity.** ``2.17e-03`` — the movement a
    trained index produces on the same rig (the mutation test below) — is three
    orders above it, and every pre-registered bar is far above both. But a
    report that said "bit-identical" here would be false, and the next person
    to rely on it would be relying on a claim that was never true in the live
    regime.
    """
    cfg, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True))
    for m in (off, on):
        with torch.no_grad():
            for ly in m.decoder.layers:
                ly.agent_gate.fill_(1.0)
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    d = float((a["traj"] - b["traj"]).abs().max())
    assert d < 1e-4, d          # a LITERAL bound, three orders under the lever
    assert torch.isfinite(b["traj"]).all()


def test_param_breakdown_carries_wp_index_and_the_decoder_line_is_unchanged():
    """⛔ The cost is on its OWN line and is carved OUT of ``decoder`` — a lever
    whose cost is buried in a 40 M-parameter row is a lever nobody can audit.

    The literal: ``8*h + h + h*H + H`` per decoder layer. The smoke config has
    ``h = 32``, ``H = 4``, ``layers = 2`` -> ``(256 + 32 + 128 + 4) * 2 = 840``.
    """
    _, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True, hidden=32))
    bo, bn = param_breakdown(off), param_breakdown(on)
    assert bo["wp_index"] == 0
    assert bn["wp_index"] == 840
    assert bn["decoder"] == bo["decoder"]          # carved out, not added on
    assert bn["total"] - bo["total"] == 840
    assert {k: v for k, v in bo.items() if k not in ("wp_index", "total")} == \
           {k: v for k, v in bn.items() if k not in ("wp_index", "total")}


def test_attach_REFUSES_without_the_agent_seam():
    """⛔ An index with no tokens to address is not a degraded arm, it is a
    NON-ARM: the heads would be built, stamped and never called, and the run
    would read as 'the waypoint index does not help' while never having had an
    index. It refuses at construction, before the GPU."""
    cfg = refc_smoke_config()
    cfg.decoder.wp_index = wi.WaypointIndexConfig(enable=True)
    torch.manual_seed(1234)
    with pytest.raises(ValueError, match="no agent cross-attention"):
        RefCModel(cfg)


def test_the_index_actually_reaches_the_attention_once_it_is_trained():
    """⛔ A SEAM THAT CANNOT MOVE THE OUTPUT IS DEAD, AND THE ZERO-INIT MAKES
    THAT HARD TO SEE. The proof above shows the arm starts bit-identical; this
    one shows it does not STAY that way — the bias head is given a non-zero
    output layer and the planner's trajectory must move."""
    cfg, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True))
    with torch.no_grad():
        for ly in on.decoder.layers:
            torch.nn.init.normal_(ly.wp_index.mlp[-1].weight, std=1.0)
            ly.agent_gate.fill_(1.0)
        for ly in off.decoder.layers:
            ly.agent_gate.fill_(1.0)
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    assert not torch.equal(a["traj"], b["traj"])
    assert torch.isfinite(b["traj"]).all()


def test_the_index_carries_GRADIENT_back_into_the_trunk_unless_detached():
    """⛔ The seam must be GATED, not dead: with ``detach`` off, the planner
    loss must reach the encoder THROUGH the address, and with it on, it must
    not — measured on the same rig so the two are comparable."""
    def trunk_grad(index):
        cfg, m = _models(index)
        with torch.no_grad():
            for ly in m.decoder.layers:
                torch.nn.init.normal_(ly.wp_index.mlp[-1].weight, std=1.0)
                ly.agent_gate.fill_(1.0)
        x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
        m.zero_grad()
        m(x)["traj"].sum().backward()
        return float(sum(p.grad.abs().sum() for p in m.encoder.parameters()
                         if p.grad is not None))
    torch.manual_seed(11)
    live = trunk_grad(wi.WaypointIndexConfig(enable=True))
    torch.manual_seed(11)
    det = trunk_grad(wi.WaypointIndexConfig(enable=True, detach=True))
    # both are non-zero (the AGENT seam itself still trains the trunk); what
    # the detach removes is the SECOND path, through the address.
    assert live > 0.0 and det > 0.0
    assert live != det, ("detach changed nothing: the address is not "
                         "differentiable on this rig and the control is void")


# ---------------------------------------------------------------------------
# ⛔⛔ MUTATION — the guard must be shown ABLE TO FAIL
# ---------------------------------------------------------------------------
def test_MUTATION_indexing_by_ARRAY_POSITION_turns_the_analytic_guard_RED():
    """⛔⛔ THE REAL DEFECT, REINTRODUCED. The whole work package is "the
    trajectory's geometry is the address"; the defect it is written against is
    an index that uses the slot's ARRAY POSITION instead. A guard that cannot
    go red proves nothing, so here is the red."""
    def by_position(wp, pos):
        b, n, s, _ = wp.shape
        m = pos.shape[1]
        idx = torch.arange(m, dtype=wp.dtype).reshape(1, 1, m).expand(b, n, m)
        z = torch.zeros(b, n, m, dtype=wp.dtype)
        return {"d_min": idx, "s_star": torch.zeros(b, n, m, dtype=torch.long),
                "tau": z, "lon": z, "lat": z, "d_range": z,
                "cos_db": z + 1.0, "sin_db": z}
    g = by_position(STRAIGHT, AGENTS)
    assert g["d_min"].reshape(-1).tolist() != [0.0, 6.0, 0.0]
    assert g["lat"].reshape(-1).tolist() != [0.0, 6.0, 0.0]
    # and the discriminating control dies too: both plans address slot 0
    assert int(by_position(LEFT, AGENTS)["d_min"].argmin()) == 0


def test_MUTATION_swapping_x_and_y_turns_the_lateral_literal_RED():
    """⛔ The range/bearing transposition — the ``anchors.pt`` units family, one
    axis over. Both tables look plausible; only the literal separates them."""
    swapped = AGENTS.flip(-1)
    g = wi.waypoint_agent_geometry(STRAIGHT, swapped)
    assert g["lat"].reshape(-1).tolist() != [0.0, 6.0, 0.0]


def test_ANALYTIC_the_first_waypoint_heading_comes_from_the_EGO_ORIGIN():
    """⭐⛔ **THE GUARD THE MUTATION PROOF DEMANDED.** Every other analytic
    check here uses the STRAIGHT plan, whose per-step heading is constant — so
    borrowing step 1's heading for step 0 changes nothing and the ``M6``
    mutation (drop the origin prepend) turned **no guard red** on the first
    run of ``code/mutation_proof.py``. That is the guard set being incomplete,
    not the mutation being unrealistic, and this is the repair.

    The discriminating scene is a TURNING plan whose closest approach is at
    ``s* = 0``. ``LEFT``'s first step is ``(0,0) -> (5, 0.5)``; an agent at
    ``(5, 1.5)`` is exactly ``1.0`` m from waypoint 0 and ``5.0`` m from
    waypoint 1, so ``s* = 0`` and ``delta = (0, 1)``. With
    ``|(5, 0.5)| = sqrt(25.25)`` the literals are

        lon = delta . h      = 0.5 / sqrt(25.25) = 0.0995037...
        lat = delta . h_perp = 5.0 / sqrt(25.25) = 0.9950372...

    and the off-by-one heading ``(5, 1.0)/sqrt(26)`` would read ``0.196116`` /
    ``0.980581`` — both entirely plausible, neither equal to the literal.
    """
    agent = torch.tensor([[[5.0, 1.5]]])
    g = wi.waypoint_agent_geometry(LEFT, agent)
    r = math.sqrt(25.25)
    assert int(g["s_star"]) == 0
    assert float(g["d_min"]) == pytest.approx(1.0, abs=1e-5)
    assert float(g["lon"]) == pytest.approx(0.5 / r, abs=1e-5)
    assert float(g["lat"]) == pytest.approx(5.0 / r, abs=1e-5)
    # ...and the off-by-one value is NOT what we read
    assert float(g["lon"]) != pytest.approx(1.0 / math.sqrt(26.0), abs=1e-3)


def test_MUTATION_dropping_the_origin_prepend_shifts_every_heading():
    """⛔ The off-by-one in the local heading. Without the ego origin
    prepended, waypoint 0's heading is borrowed from waypoint 1's step, and on
    a turning plan the cross-track literal moves. Measured here on ``LEFT``,
    whose first step (0,0)->(5,0.5) differs from its second."""
    b, n, s, _ = LEFT.shape
    step_bad = LEFT[:, :, 1:, :] - LEFT[:, :, :-1, :]
    step_bad = torch.cat([step_bad[:, :, :1], step_bad], dim=2)   # borrow s=1
    h_bad = step_bad / step_bad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    wp0 = torch.cat([LEFT.new_zeros(b, n, 1, 2), LEFT], dim=2)
    step_ok = wp0[:, :, 1:, :] - wp0[:, :, :-1, :]
    h_ok = step_ok / step_ok.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    assert not torch.allclose(h_bad[:, :, 0], h_ok[:, :, 0])


def test_MUTATION_a_nonzero_init_breaks_the_removability_proof():
    """⛔ The zero-init IS the removability guarantee, not decoration. With the
    output layer initialised normally the planner moves at step 0 and the A/B
    is confounded from its first gradient step."""
    cfg, off = _models(None)
    _, on = _models(wi.WaypointIndexConfig(enable=True))
    for m in (off, on):                 # the LIVE regime -- see the note above
        with torch.no_grad():
            for ly in m.decoder.layers:
                ly.agent_gate.fill_(1.0)
    with torch.no_grad():
        for ly in on.decoder.layers:
            torch.nn.init.normal_(ly.wp_index.mlp[-1].weight, std=1.0)
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    assert not torch.equal(a["traj"], b["traj"])
    # ...and the movement is REAL, not the 1.9e-06 kernel-rounding floor
    assert float((a["traj"] - b["traj"]).abs().max()) > 1e-4
