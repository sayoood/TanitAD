"""v6 staged trainer + composition module — CPU-only pins.

NO GPU, NO CORPUS, NO CHECKPOINT. Everything here runs on synthetic tensors, so
the pod launch is verifiable dev-side (the CLAUDE.md pod rules: a pod's
``stack/`` checkout drifts silently, ``git fetch`` HANGS with no credentials,
and a launch from stale code resurrects fixed bugs — the defence is a test that
proves the assembly BEFORE the files are shipped).

What is pinned, group by group:
  * **budget** — the default stack is under the sub-300M INVARIANT, groups
    partition every parameter, and shared goal tables are counted ONCE;
  * **X3 isolation** — ``assert_isolation`` PASSES on the wired stack and
    RAISES on both deliberate mis-wires (planner→encoder, uplink), including
    the case where the emission head's zero-init would mask a live path;
  * **§5 vocabulary sharing** — the emitting head and the consuming conditioner
    hold the SAME embedding tensor (``is`` identity, not equality);
  * **§4b horizon** — ONE 60-step (a, κ) sequence integrated to [B, 60, 2]
    waypoints; the 0–2 s / 2–6 s bands are SLICES of that one rollout, contiguous
    and seam-free, and the config REFUSES a gap or overlap;
  * **O2 time-to-reach** — speed-adaptive: equal time-to-reach ⇒ equal weight,
    and the half-weight METRE distance scales linearly with speed;
  * **O4 interaction sampling** — saliency and weights rise on high-jerk /
    braking / reversal windows and are computed from ACTIONS ONLY;
  * **X5 staging** — a FAILED gate refuses the next stage with no override; an
    INCONCLUSIVE one refuses unless overridden WITH a reason;
  * **the trainer** — every stage's loss assembles, steps, and writes its config
    through the real ``--dry-run`` path.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    CONSTRAINT_SLOTS, GOAL_ARG_SLOTS, HORIZON_S, MODULE_GROUPS, PARAM_BUDGET,
    PLAN_STEPS, STAGE_GROUPS, STAGES, STRATEGIC_GOAL_TOKENS,
    TACTICAL_GOAL_TOKENS, GoalVocabulary, InteractionSampler,
    IsolationViolation, V6Config, V6Stack, apply_stage_freeze,
    half_weight_distance_m, kinematic_saliency, near_field_band_mask,
    readout_grid_ranges, sample_cell_block_mask, saliency_weights,
    spectrum_report, stage_trainable_groups, time_to_reach,
    time_to_reach_weights)
from train_v6_staged import (  # noqa: E402
    STAGE_GATE_SPEC, STAGE_LAMBDA_PLAN, STAGE_PRECONDITION,
    GatePreconditionError, V6LossWeights, assert_stage_precondition,
    build_o4_weights, build_parser, dry_run, o2_near_field_loss,
    o3_masked_cell_loss, o5_rollout_consistency_loss, rollout_step_weights,
    stage_gate_dict, synthetic_train_batch, v6_loss_step, write_stage_gate)


# ============================================================================
# fixtures — a tiny stack that is architecturally IDENTICAL to the real one
# ============================================================================
def tiny_cfg(**kw) -> V6Config:
    """Same wiring, same seams, ~1000x fewer params. The horizon spec is NOT
    shrunk: ``plan_steps`` stays 60 because §4b is the thing under test."""
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


@pytest.fixture(scope="module")
def stack() -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(tiny_cfg())


def tiny_args(tmp_path, stage="S-W", **over):
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack")
    argv = ["--stage", stage, "--out", str(tmp_path), "--dry-run",
            "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
            "--patch", "16", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "8",
            "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "2", "--d-tac", "32",
            "--d-str", "16", "--d-goal-embed", "16", "--adapter-hidden", "32",
            "--n-candidates", "3", "--sigreg-slices", "8",
            "--dry-steps", "2", "--dry-batch", "2", "--dry-k", "12"]
    for k, v in over.items():
        argv += [k] if v is True else [k, str(v)]
    return ap.parse_args(argv)


# ============================================================================
# 1. CONFIG + THE SUB-300M INVARIANT
# ============================================================================
def test_default_stack_is_under_the_300m_invariant():
    """CLAUDE.md: sub-300M is an INVARIANT, not a preference."""
    s = V6Stack(V6Config())
    rep = s.param_report()
    assert rep["budget"] == PARAM_BUDGET == 300_000_000
    assert rep["within_budget"], rep
    assert rep["total"] < PARAM_BUDGET
    assert rep["headroom"] > 0
    # the groups must PARTITION the model: no double counting, nothing missed
    assert sum(rep["per_group"].values()) == rep["total"]
    assert set(rep["per_group"]) == set(MODULE_GROUPS)
    assert rep["total"] == sum(p.numel() for p in s.parameters())


def test_per_layer_encoder_arm_costs_params_and_stays_in_budget():
    """E-ENC arm (b) must EARN its params — it is measurably more expensive,
    and the budget check is the thing that keeps the comparison honest."""
    shared = V6Stack(tiny_cfg(shared_encoder=True)).param_report()
    per = V6Stack(tiny_cfg(shared_encoder=False)).param_report()
    assert per["total"] > shared["total"]
    assert per["per_group"]["encoder"] > shared["per_group"]["encoder"]
    assert shared["arm"] == "shared-encoder+adapters"
    assert per["arm"] == "per-layer-encoders"
    assert per["within_budget"]


def test_matched_param_config_respects_the_head_divisibility():
    """E-ENC decides at MATCHED TOTAL PARAMS. A probe grid that ignores
    ``d_model % n_heads == 0`` silently prunes to a handful of widths and
    reports a bad match as the best one — MEASURED at n_heads=2 here, and at
    n_heads=12 on the real config where a multiples-of-64 grid leaves only
    384/768/1152."""
    from tanitad.models.v6 import matched_param_config
    base = tiny_cfg()
    target = V6Stack(tiny_cfg(shared_encoder=False)).param_report()["total"]
    cfg, rep = matched_param_config(base, target, lo=32, hi=128)
    assert cfg.predictor.d_model % base.predictor.n_heads == 0
    assert all(d % base.predictor.n_heads == 0 for d in rep["grid"])
    assert rep["gap"] == abs(rep["chosen_total"] - target)
    assert 0.0 <= rep["gap_frac"]
    assert V6Stack(cfg).param_report()["within_budget"]


def test_over_budget_stack_is_refused():
    s = V6Stack(tiny_cfg())
    with pytest.raises(ValueError, match="sub-300M"):
        s.assert_param_budget(budget=1000)


def test_config_derives_state_dim_from_the_readout_firewall():
    cfg = tiny_cfg()
    assert cfg.d_op == cfg.n_cells * cfg.readout.d_readout == 4 * 4 * 8
    assert cfg.grid_shape == (4, 4)
    assert V6Config().d_op == 2048          # the width the whole stack expects


def test_clocks_and_strides():
    cfg = V6Config()
    assert (cfg.hz_op, cfg.hz_tac, cfg.hz_str) == (10.0, 2.0, 0.5)
    assert cfg.stride_tac == 5 and cfg.stride_str == 20
    with pytest.raises(ValueError, match="non-increasing"):
        V6Config(hz_tac=20.0)
    with pytest.raises(ValueError, match="contradicts"):
        V6Config(hz_op=5.0)


def test_config_serialises_with_its_derived_geometry():
    d = V6Config().to_dict()
    assert d["_derived"]["d_op"] == 2048
    assert d["_derived"]["horizon_s"] == 6.0
    assert d["_derived"]["op_band_steps"] == [0, 20]
    assert d["_derived"]["tac_band_steps"] == [20, 60]
    json.dumps(d)                            # must be serialisable as written


# ============================================================================
# 2. X3 — the gradient-isolation matrix
# ============================================================================
def test_isolation_passes_on_the_wired_stack(stack):
    rep = stack.assert_isolation()
    assert rep["pass"] is True
    assert rep["violations"] == {}
    assert all(v == 0 for v in rep["n_violations"].values())
    assert rep["n_probed"]["planner_to_encoder"] > 0     # not vacuous
    assert rep["_evidence_class"].startswith("MEASURED")


def test_isolation_catches_a_mis_wired_planner_to_encoder_path():
    """THE headline check: a planner gradient reaching the encoder must RAISE.

    ``isolate_planner_from_encoder=False`` is the deliberately mis-wired arm —
    exactly the co-training path ``JEPA_PHYSICS_SURVEY`` §4 says nobody at the
    frontier takes."""
    bad = V6Stack(tiny_cfg(isolate_planner_from_encoder=False))
    with pytest.raises(IsolationViolation, match="X3 gradient-isolation"):
        bad.assert_isolation()
    rep = bad.assert_isolation(strict=False)
    assert rep["pass"] is False
    assert rep["n_violations"]["planner_to_encoder"] > 0
    # and it must name the ENCODER parameters, not something adjacent
    named = rep["violations"]["planner_to_encoder"]
    assert any(n.startswith(("encoder.", "readout.")) for n in named), named


def test_isolation_catches_a_mis_wired_uplink():
    bad = V6Stack(tiny_cfg(isolate_uplink=False))
    with pytest.raises(IsolationViolation):
        bad.assert_isolation()
    rep = bad.assert_isolation(strict=False)
    assert rep["n_violations"]["tactical_to_below"] > 0
    assert rep["n_violations"]["strategic_to_below"] > 0


def test_isolation_is_not_fooled_by_the_zero_init_emission_head():
    """The emission's final layer is ZERO-INIT (the CV warm start), so its
    gradient w.r.t. its own INPUT is exactly 0 at init. A probe that squared
    the outputs, or that only looked at the emitted controls, would report a
    live mis-wire as isolated. The declared surface includes the pre-emission
    feature — this test is what keeps that true."""
    bad = V6Stack(tiny_cfg(isolate_planner_from_encoder=False))
    out = bad.forward(**bad.synthetic_batch(2))
    assert float(out["plan"]["a"].detach().abs().max()) == 0.0   # zero-init
    assert float(out["plan"]["kappa"].detach().abs().max()) == 0.0
    enc = list(bad.encoder_parameters())
    # probing ONLY the emitted controls would find nothing...
    only_controls = V6Stack._probe_scalar([out["plan"]["a"],
                                           out["plan"]["kappa"]])
    assert V6Stack._live_edges(only_controls, enc) == []
    # ...but the DECLARED surface finds the live path.
    assert V6Stack._live_edges(
        V6Stack._probe_scalar(out["planner_side"]), enc) != []


def test_isolation_survives_a_stage_freeze(stack):
    """X3 is an ARCHITECTURE property. Run mid-S-T, with the encoder frozen, the
    check must still probe the encoder — a frozen parameter records no autograd
    edge, and a vacuous pass is how an isolation guarantee rots."""
    before = [(p, p.requires_grad) for p in stack.parameters()]
    apply_stage_freeze(stack, "S-T")
    rep = stack.assert_isolation()
    assert rep["pass"] is True
    assert rep["n_probed"]["planner_to_encoder"] > 0
    # requires_grad must be restored EXACTLY as the freeze left it
    assert [p.requires_grad for p, _ in before] == \
           [p.requires_grad for p in stack.parameters()]
    for p, rg in before:
        p.requires_grad_(rg)


@pytest.mark.parametrize("selector", ["none", "goal", "mlp"])
def test_planner_surface_is_total(selector):
    """Every planner-group parameter must be REACHABLE from the DECLARED
    planner-side surface. A head added without appending to the declaration
    would silently escape the isolation probe — this is the guard.

    ⚠️ Measured on a stack whose emission output layer has been perturbed off
    its zero init. At zero init the emission's FIRST layer receives exactly no
    gradient through the emitted controls (``dL/dW0 ∝ W_last = 0``), so
    "reachable" is not measurable there — which is itself the reason the
    declared surface carries the pre-emission ``feat`` tensor as well.

    ⛔ AND IT MUST RUN ON EVERY SELECTOR ARM, NOT ONLY THE DEFAULT. Until
    2026-08-16 this ran on ``tiny_cfg()`` alone, i.e. with NO scorer built — so
    the ``"goal"`` and ``"mlp"`` arms' parameters were never probed at all, and
    with ``selector="mlp"`` ``cand_score.fc1.{weight,bias}`` are additionally
    INVISIBLE at init for the same zero-init reason as the emission and the
    FiLM (``fc2`` is zero-init BY DESIGN so the capacity control starts FLAT
    over the fan, hence ``dL/dfc1 ∝ W_fc2 = 0`` EXACTLY). A mis-wire in
    ``MLPCandidateScorer.fc1`` would not have been caught — in the arm whose
    whole job is to be the CONTROL that decides whether SEL-1 is mechanism or
    capacity.

    ⭐ THE FIX IS A TEST-LOCAL PERTURBATION, NOT A CHANGED INITIALISATION.
    ``MLPCandidateScorer.__init__`` is untouched, so the TRAINED arm still
    starts flat over the fan and any ranking it acquires is still visibly
    LEARNED. Reachability is an ARCHITECTURE property (the X3 probe's own
    philosophy), so the probe — and only the probe — runs off the zero init, on
    a stack built for the probe and discarded after it. Sensitivity is proved
    separately: ``tests/test_v6_anchor_loss.py`` shows the hole EXISTS at zero
    init, CLOSES under this perturbation, and that a genuinely disconnected
    ``fc1`` is still reported MISSING.
    """
    s = V6Stack(tiny_cfg(selector=selector))
    with torch.no_grad():                     # off the CV warm start
        s.emission.net[-1].weight.normal_(0.0, 0.1)
        s.emission.net[-1].bias.normal_(0.0, 0.1)
        # ⛔ Same class as the emission perturbation above, found 2026-08-13
        # when intent_proj joined the planner group: FiLM.to_scale_shift is
        # ZERO-INIT (identity start, deliberate), so at init
        # dL/d(intent_proj.W) ∝ W_film = 0 EXACTLY — the seam is
        # architecturally reachable but first-order-dead until S-W moves the
        # FiLM. Reachability is an ARCHITECTURE property (the X3 probe's own
        # philosophy), so the probe runs off the zero init. S-W training
        # guarantees W_film != 0 by the time S-T needs the seam: the same
        # FiLM carries the action conditioning that S-W trains.
        for blk in s.predictor_op.blocks:
            blk.film.to_scale_shift.weight.normal_(0.0, 0.1)
        # ⛔ THE THIRD ZERO-INIT, and the one that hid a whole module. Every
        # scorer's OUTPUT layer starts at zero on the same discipline; probe
        # off it or the layer BEFORE it reads as unreachable.
        for name in ("fc2", "goal_point"):
            head = getattr(s.cand_score, name, None)
            if head is not None:
                head.weight.normal_(0.0, 0.1)
                head.bias.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    planner = list(s.group_parameters("planner"))
    assert planner, "nothing probed — a probe over an empty set proves nothing"
    if selector == "mlp":                     # the arm this guard was blind to
        assert any(n.startswith("cand_score.fc1") for n, _ in planner)
    live = V6Stack._live_edges(V6Stack._probe_scalar(out["planner_side"]),
                              planner)
    missing = {n for n, _ in planner} - set(live)
    assert not missing, f"planner params invisible to the probe: {missing}"


def test_ema_uplink_arm_is_isolated_and_updates():
    s = V6Stack(tiny_cfg(uplink="ema"))
    assert s.ema_adapter_tac is not None and s.ema_adapter_str is not None
    assert all(not p.requires_grad for p in s.ema_adapter_tac.parameters())
    assert s.assert_isolation()["pass"] is True
    before = s.ema_adapter_tac.module[0].weight.clone()
    with torch.no_grad():
        for p in s.adapter_tac.parameters():
            p.add_(1.0)
    s.ema_update()
    assert not torch.equal(before, s.ema_adapter_tac.module[0].weight)


# ============================================================================
# 3. §5 — one vocabulary, two views
# ============================================================================
def test_goal_token_embedding_is_the_SAME_tensor_emitting_and_consuming(stack):
    """HIERARCHY_VOCABULARY §5: *"Token embeddings are shared between the
    goal-emitting head (above) and the goal-consuming conditioner (below) — one
    vocabulary, two views."* Identity, not equality: two tables that merely
    start equal are two vocabularies."""
    # tactical seam: emitted by layer T, consumed by layer O
    assert stack.goal_head_tac.vocab is stack.cond_op.vocab
    assert stack.goal_head_tac.vocab.table.weight is \
           stack.cond_op.vocab.table.weight
    assert id(stack.goal_head_tac.vocab.table.weight) == \
           id(stack.cond_op.vocab.table.weight)
    # strategic seam: emitted by layer S, consumed by layer T
    assert stack.goal_head_str.vocab is stack.cond_tac.vocab
    assert stack.goal_head_str.vocab.table.weight is \
           stack.cond_tac.vocab.table.weight
    # and the two seams are DIFFERENT vocabularies
    assert stack.vocab_tac is not stack.vocab_str


def test_shared_table_is_counted_once(stack):
    """If a table were accidentally copied the total would jump — the param
    report is therefore also the sharing check."""
    names = [n for n, _ in stack.named_parameters()]
    assert names.count("vocab_tac.table.weight") == 1
    assert not any(n.startswith("cond_op.vocab.") for n in names)
    assert not any(n.startswith("goal_head_tac.vocab.") for n in names)


def test_a_moving_shared_table_moves_both_views(stack):
    with torch.no_grad():
        stack.vocab_tac.table.weight.add_(0.5)
    ids = torch.tensor([0, 1])
    assert torch.equal(stack.goal_head_tac.vocab.embed_tokens(ids),
                       stack.cond_op.vocab.embed_tokens(ids))
    with torch.no_grad():
        stack.vocab_tac.table.weight.sub_(0.5)


def test_vocabulary_contents_match_the_spec():
    assert "FOLLOW_MAIN_ROAD" in STRATEGIC_GOAL_TOKENS    # THE no-route default
    assert "NONE_ABSTAIN" in STRATEGIC_GOAL_TOKENS        # ambiguous geometry
    for t in ("YIELD_AT", "STOP_POINT", "WAIT_FOR_ONCOMING",
              "EVADE_IN_CORRIDOR", "TRAFFIC_LIGHT_REACT", "SPEED_BAND"):
        assert t in TACTICAL_GOAL_TOKENS, t
    assert CONSTRAINT_SLOTS == ("within_m", "by_time_s", "at_arc_m",
                                "hold_for_s")
    assert GOAL_ARG_SLOTS == 8


def test_goal_vocabulary_soft_and_hard_views_agree():
    v = GoalVocabulary(("A", "B", "C"), d_embed=4)
    hard = v.encode(torch.tensor([2]))
    soft = v.encode(torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(hard, soft, atol=1e-6)
    assert v.id_of("B") == 1
    with pytest.raises(KeyError):
        v.id_of("NOPE")


def test_unset_constraint_slots_contribute_nothing():
    """§2: *"Unset = unconstrained."* A masked slot must contribute EXACTLY
    zero — the same IGNORE discipline as the factorised CE."""
    v = GoalVocabulary(("A", "B"), d_embed=4)
    ids = torch.tensor([0])
    args = torch.randn(1, GOAL_ARG_SLOTS)
    masked = v.encode(ids, args, torch.zeros(1, GOAL_ARG_SLOTS))
    zeroed = v.encode(ids, torch.zeros(1, GOAL_ARG_SLOTS))
    assert torch.allclose(masked, zeroed, atol=1e-6)


def test_goal_head_refuses_an_undeclared_conditioning_path(stack):
    """X1's disjointness audit in miniature: a head built without a cond port
    must not silently accept one."""
    z = torch.randn(2, stack.cfg.d_str)
    with pytest.raises(ValueError, match="undeclared conditioning path"):
        stack.goal_head_str(z, cond=torch.randn(2, stack.cfg.d_goal_embed))


# ============================================================================
# 4. §4b — ONE 60-step rollout, bands are slices of it
# ============================================================================
def test_horizon_constants_are_the_binding_spec():
    assert PLAN_STEPS == 60
    assert HORIZON_S == 6.0
    cfg = V6Config()
    assert cfg.plan_steps == 60 and cfg.dt == 0.1 and cfg.horizon_s == 6.0
    assert cfg.op_band_s == (0.0, 2.0) and cfg.tac_band_s == (2.0, 6.0)


def test_emission_returns_60_step_controls_and_integrated_waypoints(stack):
    cfg = stack.cfg
    b = 3
    z = torch.randn(b, cfg.d_op)
    g = torch.randn(b, cfg.d_goal_embed)
    v0 = torch.full((b,), 12.0)
    plan = stack.emit(z, g, v0)
    assert plan["controls"].shape == (b, cfg.n_candidates, 60, 2)
    assert plan["waypoints"].shape == (b, cfg.n_candidates, 60, 2)
    assert plan["a"].shape == plan["kappa"].shape == (b, cfg.n_candidates, 60)
    # ONE candidate -> the [B, 60, 2] contract of the brief
    assert plan["controls"][:, 0].shape == (b, 60, 2)
    assert plan["waypoints"][:, 0].shape == (b, 60, 2)
    # feasible BY CONSTRUCTION (W4: a = a_max*tanh, kappa = kappa_max*tanh)
    assert float(plan["a"].detach().abs().max()) <= cfg.a_max + 1e-5
    assert float(plan["kappa"].detach().abs().max()) <= cfg.kappa_max + 1e-5


def test_zero_control_rollout_is_the_constant_velocity_path(stack):
    """Zero-init emission ⇒ a = κ = 0 ⇒ the integrated path is straight at v0,
    which makes the 6 s waypoints analytically checkable: x_k = v0 * k * dt."""
    b, v = 2, 10.0
    plan = stack.emit(torch.randn(b, stack.cfg.d_op),
                      torch.randn(b, stack.cfg.d_goal_embed),
                      torch.full((b,), v))
    wp = plan["waypoints"][:, 0].detach()                 # [B, 60, 2]
    want_x = torch.arange(1, 61).float() * v * stack.cfg.dt
    assert torch.allclose(wp[..., 0], want_x.expand(b, 60), atol=1e-3)
    assert float(wp[..., 1].abs().max()) < 1e-5           # no lateral motion


def test_band_slicing_is_correct_and_seam_free(stack):
    cfg = stack.cfg
    assert cfg.band_slice("op") == slice(0, 20)
    assert cfg.band_slice("tac") == slice(20, 60)
    x = torch.arange(60).float()[None, :, None].expand(2, 60, 2)
    op, tac = cfg.split_bands(x, dim=-2)
    assert op.shape == (2, 20, 2) and tac.shape == (2, 40, 2)
    # CONTIGUOUS: the tactical band starts exactly where the operative ends —
    # bands are slices of ONE rollout, never two stitched trajectories.
    assert float(op[0, -1, 0]) == 19.0 and float(tac[0, 0, 0]) == 20.0
    assert torch.equal(torch.cat([op, tac], dim=-2), x)


def test_the_two_bands_come_from_one_rollout(stack):
    """The seam-free-by-construction claim, MEASURED: slicing the emitted
    waypoints reproduces them exactly, and the last operative point and the
    first tactical point are one unicycle step apart at v0."""
    b, v = 2, 8.0
    wp = stack.emit(torch.randn(b, stack.cfg.d_op),
                    torch.randn(b, stack.cfg.d_goal_embed),
                    torch.full((b,), v))["waypoints"][:, 0]
    op, tac = stack.cfg.split_bands(wp, dim=-2)
    step = (tac[:, 0] - op[:, -1]).norm(dim=-1)
    assert torch.allclose(step, torch.full((b,), v * stack.cfg.dt), atol=1e-3)


def test_config_refuses_a_band_gap_or_overlap():
    with pytest.raises(ValueError, match="seam-free-by-construction"):
        V6Config(op_band_s=(0.0, 1.5))           # gap before the tactical band
    with pytest.raises(ValueError, match="seam-free-by-construction"):
        V6Config(op_band_s=(0.0, 3.0))           # overlap
    with pytest.raises(ValueError, match="plan horizon"):
        V6Config(tac_band_s=(2.0, 8.0))          # past the 6 s horizon


def test_forward_plumbs_the_horizon_end_to_end(stack):
    out = stack.forward(**stack.synthetic_batch(2))
    assert out["plan"]["waypoints"].shape[-2] == 60
    op, tac = stack.cfg.split_bands(out["plan"]["waypoints"], dim=-2)
    assert op.shape[-2] == 20 and tac.shape[-2] == 40


# ============================================================================
# 5. O2 — TIME-TO-REACH weighting is speed-adaptive
# ============================================================================
def test_equal_time_to_reach_gives_equal_weight_at_any_speed():
    """The definition of speed-adaptive: the TIME band is fixed, so the metre
    band moves with speed. 10 m at 10 m/s and 20 m at 20 m/s are the same
    1 s away and must weigh the same."""
    slow = time_to_reach_weights(torch.tensor([[10.0, 20.0, 30.0]]),
                                 torch.tensor([10.0]), normalize=False)
    fast = time_to_reach_weights(torch.tensor([[20.0, 40.0, 60.0]]),
                                 torch.tensor([20.0]), normalize=False)
    assert torch.allclose(slow, fast, atol=1e-6)


def test_higher_speed_widens_the_metre_band():
    """*"a fixed 40 m band cannot cover a 6 s horizon (180 m at 30 m/s)"* — the
    half-weight distance is linear in speed, so the band widens by construction
    and no metre constant appears anywhere."""
    d = torch.tensor([[40.0]])
    w_slow = time_to_reach_weights(d, torch.tensor([5.0]), normalize=False)
    w_fast = time_to_reach_weights(d, torch.tensor([30.0]), normalize=False)
    assert float(w_fast) > float(w_slow)     # 40 m is "nearer" in TIME at speed
    h5 = float(half_weight_distance_m(5.0))
    h30 = float(half_weight_distance_m(30.0))
    assert h30 == pytest.approx(h5 * 6.0, rel=1e-5)
    assert h30 > 40.0 > h5                   # the 40 m constant is not a band


def test_time_to_reach_is_capped_at_the_horizon_and_floored_in_speed():
    """*"capped at the 6 s horizon"* — and a STOPPED ego must not send every
    cell to +inf (v_floor), which would make the weights NaN, not merely
    small."""
    t = time_to_reach(torch.tensor([[1e6]]), torch.tensor([1.0]))
    assert float(t) == pytest.approx(HORIZON_S)           # capped
    stopped = time_to_reach(torch.tensor([[3.0]]), torch.tensor([0.0]))
    assert float(stopped) == pytest.approx(3.0)           # v_floor = 1 m/s
    assert torch.isfinite(stopped).all()
    far_and_stopped = time_to_reach(torch.tensor([[500.0]]),
                                    torch.tensor([0.0]))
    assert float(far_and_stopped) == pytest.approx(HORIZON_S)
    assert torch.isfinite(time_to_reach_weights(torch.tensor([[500.0]]),
                                                torch.tensor([0.0]))).all()


def test_time_to_reach_weights_normalise_to_mean_one():
    w = time_to_reach_weights(torch.rand(4, 16) * 80.0,
                              torch.rand(4) * 25 + 1.0)
    assert torch.allclose(w.mean(dim=-1), torch.ones(4), atol=1e-5)


def test_readout_grid_ranges_are_monotone_near_at_the_bottom():
    r = readout_grid_ranges(4, 4, near_m=3.0, far_m=80.0)
    col = r[:, 0]
    assert float(col[0]) == pytest.approx(80.0)     # image top = far
    assert float(col[-1]) == pytest.approx(3.0)     # image bottom = near
    assert bool((torch.diff(col) < 0).all())        # strictly decreasing
    assert torch.equal(r[:, 0], r[:, 3])            # no lateral depth cue


def test_o2_loss_weights_the_near_field_more(stack):
    b, c, dr = 2, stack.cfg.n_cells, stack.cfg.readout.d_readout
    true = torch.zeros(b, c, dr)
    ranges = stack.cell_ranges_m
    near = int(ranges.argmin())
    far = int(ranges.argmax())
    pred_near, pred_far = true.clone(), true.clone()
    pred_near[:, near] = 1.0
    pred_far[:, far] = 1.0
    v0 = torch.full((b,), 10.0)
    l_near, _ = o2_near_field_loss(pred_near, true, ranges, v0)
    l_far, _ = o2_near_field_loss(pred_far, true, ranges, v0)
    assert float(l_near) > float(l_far)


# ============================================================================
# 6. O3 / O4 / O5 / O6 primitives
# ============================================================================
def test_block_mask_is_contiguous_and_bounded():
    g = torch.Generator().manual_seed(3)
    m = sample_cell_block_mask(4, 6, n_blocks=1, block_h=2, block_w=3,
                               batch=8, generator=g).reshape(8, 4, 6)
    for b in range(8):
        rows = torch.where(m[b].any(dim=1))[0]
        cols = torch.where(m[b].any(dim=0))[0]
        assert len(rows) == 2 and len(cols) == 3            # exact block size
        assert int(rows[-1] - rows[0]) == 1                 # CONTIGUOUS
        assert int(cols[-1] - cols[0]) == 2
    assert int(m.sum()) == 8 * 6
    with pytest.raises(ValueError, match="does not fit"):
        sample_cell_block_mask(2, 2, block_h=3, block_w=1)


def test_near_field_band_mask_takes_the_bottom_rows():
    m = near_field_band_mask(4, 4, rows=2).reshape(4, 4)
    assert bool(m[2:].all()) and not bool(m[:2].any())
    assert not bool(near_field_band_mask(4, 4, rows=0).any())


def test_o3_scores_only_masked_cells(stack):
    b, c, dr = 2, stack.cfg.n_cells, stack.cfg.readout.d_readout
    ctx = torch.randn(b, c, dr)
    true = torch.randn(b, c, dr)
    mask = torch.zeros(b, c, dtype=torch.bool)
    mask[:, :4] = True
    loss, log = o3_masked_cell_loss(stack.masked_cells, ctx, true, mask)
    assert torch.isfinite(loss) and loss.requires_grad
    assert log["o3_n_masked"] == 8
    assert log["o3_mask_rate"] == pytest.approx(4 / c)
    # an empty mask contributes zero but stays in the graph
    z, zlog = o3_masked_cell_loss(stack.masked_cells, ctx, true,
                                  torch.zeros(b, c, dtype=torch.bool))
    assert float(z.detach()) == 0.0 and zlog["o3_n_masked"] == 0
    assert z.requires_grad          # still in the graph, contributes nothing


def test_o4_saliency_rises_on_high_jerk_windows():
    """O4/LF1: |jerk|, |decel|, steering reversals — from ACTIONS ONLY."""
    t = torch.arange(40).float()
    free_flow = torch.zeros(1, 40, 2)
    free_flow[..., 1] = 0.05                       # gentle steady accel
    jerky = torch.zeros(1, 40, 2)
    jerky[0, :, 1] = torch.sin(t) * 3.0            # hard accel/brake cycling
    braking = torch.zeros(1, 40, 2)
    braking[0, :, 1] = -3.0                        # sustained deceleration
    weaving = torch.zeros(1, 40, 2)
    weaving[0, :, 0] = torch.sin(t * 1.9) * 0.3    # steering reversals
    s = kinematic_saliency(torch.cat([free_flow, jerky, braking, weaving], 0))
    assert float(s[1]) > float(s[0])               # jerk
    assert float(s[2]) > float(s[0])               # decel
    assert float(s[3]) > float(s[0])               # reversals
    w = saliency_weights(s)
    assert float(w[1]) > float(w[0])
    assert float(w.sum()) == pytest.approx(1.0)
    assert float(w[0]) > 0.0                       # free flow stays REACHABLE


def test_o4_alpha_zero_is_uniform_sampling():
    """The attributability control: alpha=0 must reproduce uniform exactly, so
    the O4 arm is a single lever."""
    s = torch.tensor([0.0, 1.0, 5.0, 20.0])
    w = saliency_weights(s, alpha=0.0)
    assert torch.allclose(w, torch.full((4,), 0.25), atol=1e-6)


def test_o4_never_removes_a_window_parity_is_sacred():
    s = torch.zeros(50)
    s[7] = 100.0
    w = saliency_weights(s, floor=0.25)
    assert bool((w > 0).all()), "a zero weight would re-select the corpus"


def test_build_o4_weights_reports_its_skew():
    acts = torch.randn(32, 20, 2) * 0.3
    w, log = build_o4_weights(acts)
    assert w.shape == (32,) and float(w.sum()) == pytest.approx(1.0)
    assert log["o4_n"] == 32 and len(log["o4_saliency_quantiles"]) == 5
    assert log["o4_weight_max_over_min"] >= 1.0


def test_interaction_sampler_prefers_salient_windows():
    index = [(0, t) for t in range(200)]
    w = torch.full((200,), 1e-6)
    w[:10] = 1.0                                   # 10 salient windows
    smp = InteractionSampler(index, w, eps_per_batch=1,
                            generator=torch.Generator().manual_seed(0))
    drawn = smp(400)
    assert len(drawn) == 400
    assert sum(1 for i in drawn if i < 10) > 350    # overwhelmingly salient
    with pytest.raises(ValueError, match="align 1:1"):
        InteractionSampler(index, torch.ones(3))


def test_o5_scores_every_step_not_just_the_endpoint():
    """The P5 lesson: an endpoint-only loss is minimisable by a trajectory that
    is wrong throughout and right at the end."""
    k = 6
    true = [torch.zeros(2, 8) for _ in range(k)]
    wrong_middle = [torch.ones(2, 8) for _ in range(k - 1)] + [torch.zeros(2, 8)]
    w_uni = rollout_step_weights(k, "uniform")
    w_end = rollout_step_weights(k, "endpoint")
    l_uni, log = o5_rollout_consistency_loss(wrong_middle, true, w_uni)
    l_end, _ = o5_rollout_consistency_loss(wrong_middle, true, w_end)
    assert float(l_uni) > 0.0
    assert float(l_end) == pytest.approx(0.0)      # blind to the whole path
    assert log["o5_k"] == k and log["o5_step1"] == pytest.approx(1.0)
    assert torch.allclose(w_uni.mean(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="uniform"):
        rollout_step_weights(4, "nonsense")


def test_o6_spectrum_monitor_detects_collapse():
    torch.manual_seed(0)
    full = spectrum_report(torch.randn(64, 16))
    rank1 = spectrum_report(torch.randn(64, 1) @ torch.randn(1, 16))
    assert full["participation_ratio"] > 8.0
    assert rank1["participation_ratio"] < 1.5
    assert rank1["effective_rank"] < full["effective_rank"]
    assert full["n"] == 64 and full["d"] == 16 and full["top_k"] == 8


def test_o6_sigreg_is_differentiable(stack):
    from train_v6_staged import o6_sigreg_loss
    z = torch.randn(16, stack.cfg.d_op, requires_grad=True)
    loss = o6_sigreg_loss(stack.sigreg, z, 0)
    loss.backward()
    assert torch.isfinite(loss) and z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0


# ============================================================================
# 7. X5 — staging, freezing, and the gate that refuses
# ============================================================================
def test_stage_groups_partition_the_model(stack):
    assert set(STAGES) == set(STAGE_GROUPS)
    for st in STAGES:
        assert set(stage_trainable_groups(st)) <= set(MODULE_GROUPS)
    assert set(STAGE_GROUPS["S-J"]) == set(MODULE_GROUPS)
    for n, _ in stack.named_parameters():
        assert stack.group_of(n) in MODULE_GROUPS
    with pytest.raises(KeyError, match="belongs to no group"):
        stack.group_of("some_new_head.weight")


def test_stage_freeze_trains_exactly_the_declared_groups(stack):
    for st in STAGES:
        rep = apply_stage_freeze(stack, st)
        want = set(stage_trainable_groups(st))
        for n, p in stack.named_parameters():
            assert p.requires_grad == (stack.group_of(n) in want), (st, n)
        assert rep["n_trainable"] + rep["n_frozen"] == \
               stack.param_report()["total"]
        assert set(rep["shared_goal_tables"]) == {"vocab_tac", "vocab_str"}
    # S-W: WM only — the planner is ABSENT
    rep = apply_stage_freeze(stack, "S-W")
    assert rep["per_group"]["planner"]["trainable"] == 0
    assert rep["per_group"]["encoder"]["trainable"] > 0
    # S-T: the trunk is frozen, the tactical layer + planner train
    rep = apply_stage_freeze(stack, "S-T")
    assert rep["per_group"]["encoder"]["trainable"] == 0
    assert rep["per_group"]["predictor_op"]["trainable"] == 0
    assert rep["per_group"]["layer_tac"]["trainable"] > 0
    assert rep["per_group"]["planner"]["trainable"] > 0
    # S-S: only the strategic layer
    rep = apply_stage_freeze(stack, "S-S")
    assert rep["per_group"]["layer_tac"]["trainable"] == 0
    assert rep["per_group"]["layer_str"]["trainable"] > 0
    apply_stage_freeze(stack, "S-J")


def test_lambda_plan_is_zero_in_S_W_by_construction():
    assert STAGE_LAMBDA_PLAN["S-W"] == 0.0
    w = V6LossWeights(lambda_plan=1.0).for_stage("S-W")
    assert w.lambda_plan == 0.0 and w.t1_latent == 0.0 and w.s1_latent == 0.0
    assert w.o1_ctrl > 0 and w.o5_rollout > 0
    assert V6LossWeights().for_stage("S-T").o1_ctrl == 0.0   # trunk frozen
    assert V6LossWeights().for_stage("S-S").t1_latent == 0.0


def test_gate_dict_pass_fail_and_inconclusive():
    ok = {p: {"pass": True} for p in STAGE_GATE_SPEC["S-W"]["required"]}
    g = stage_gate_dict("S-W", ok)
    assert g["pass"] is True and g["verdict"] == "PASS"
    assert g["next_stage"] == "S-T"
    bad = dict(ok, P3={"pass": False})
    g = stage_gate_dict("S-W", bad)
    assert g["pass"] is False and g["failed_required"] == ["P3"]
    # a MISSING required probe is INCONCLUSIVE, never a pass
    g = stage_gate_dict("S-W", {"P1": {"pass": True}})
    assert g["pass"] is None and g["verdict"] == "INCONCLUSIVE"
    assert set(g["missing_required"]) == {"P3", "P6"}
    # an explicitly-null probe is also inconclusive
    g = stage_gate_dict("S-W", dict(ok, P6={"pass": None}))
    assert g["pass"] is None and g["inconclusive_required"] == ["P6"]
    assert "INCONCLUSIVE" in g["bound_outcomes"]


def test_a_failed_gate_refuses_the_next_stage(tmp_path):
    """X5: *"a failed stage never propagates upward"* — and there is NO
    override for a FAIL."""
    gate = stage_gate_dict("S-W", {"P1": {"pass": True}, "P3": {"pass": False},
                                   "P6": {"pass": True}})
    p = write_stage_gate(tmp_path, gate)
    assert gate["pass"] is False
    with pytest.raises(GatePreconditionError, match="FAILED its gate"):
        assert_stage_precondition("S-T", p)
    # not even the inconclusive override opens a FAIL
    with pytest.raises(GatePreconditionError, match="no override for a FAIL"):
        assert_stage_precondition("S-T", p, allow_inconclusive=True,
                                  off_reason="we are in a hurry")


def test_a_passed_gate_admits_the_next_stage(tmp_path):
    gate = stage_gate_dict("S-W", {p: {"pass": True}
                                   for p in STAGE_GATE_SPEC["S-W"]["required"]})
    p = write_stage_gate(tmp_path, gate)
    rep = assert_stage_precondition("S-T", p)
    assert rep["ok"] and rep["prev_verdict"] == "PASS"


def test_an_inconclusive_gate_refuses_without_a_stated_reason(tmp_path):
    p = write_stage_gate(tmp_path, stage_gate_dict("S-W", {"P1": {"pass": True}}))
    with pytest.raises(GatePreconditionError, match="INCONCLUSIVE IS NOT A PASS"):
        assert_stage_precondition("S-T", p)
    with pytest.raises(GatePreconditionError):
        assert_stage_precondition("S-T", p, allow_inconclusive=True,
                                  off_reason="   ")
    rep = assert_stage_precondition("S-T", p, allow_inconclusive=True,
                                    off_reason="P3/P6 pod is down; battery "
                                               "runs tomorrow")
    assert rep["ok"] and rep["override"] == "allow-inconclusive-gate"
    assert "pod is down" in rep["off_reason"]


def test_a_missing_or_mismatched_gate_file_refuses(tmp_path):
    assert assert_stage_precondition("S-W")["ok"]          # S-W starts free
    with pytest.raises(GatePreconditionError, match="no --prev-gate"):
        assert_stage_precondition("S-T")
    with pytest.raises(GatePreconditionError, match="does not exist"):
        assert_stage_precondition("S-T", tmp_path / "nope.json")
    # pointing a stage at the WRONG stage's gate file is refused: S-T needs
    # S-W's gate, and being handed a (passing!) S-T gate is not a pass for S-T.
    wrong = write_stage_gate(tmp_path, stage_gate_dict(
        "S-T", {p: {"pass": True} for p in STAGE_GATE_SPEC["S-T"]["required"]}))
    with pytest.raises(GatePreconditionError, match="requires 'S-W'"):
        assert_stage_precondition("S-T", wrong)
    # ...while the SAME file is the correct precondition for S-S
    assert assert_stage_precondition("S-S", wrong)["prev_verdict"] == "PASS"


def test_stage_ladder_is_the_spec_order():
    assert STAGE_PRECONDITION == {"S-W": None, "S-T": "S-W", "S-S": "S-T",
                                  "S-J": "S-S"}


# ============================================================================
# 8. the trainer — loss assembly and the --dry-run path
# ============================================================================
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_assembles_a_finite_loss_with_grads(stack, stage):
    apply_stage_freeze(stack, stage)
    b = synthetic_train_batch(stack, batch=2, k=12, seed=1)
    b["gt_wp"] = torch.randn(2, 10, 2)
    L = v6_loss_step(stack, b, stage=stage, weights=V6LossWeights(
        lambda_plan=STAGE_LAMBDA_PLAN[stage]), o1_k=10, o5_k=12)
    assert torch.isfinite(L["loss"]) and L["loss"].requires_grad
    assert L["log"]["stage"] == stage and L["log"]["terms"]
    L["loss"].backward()
    got = [n for n, p in stack.named_parameters()
           if p.requires_grad and p.grad is not None
           and float(p.grad.abs().sum()) > 0]
    assert got, f"stage {stage} produced no gradient anywhere"
    for n in got:
        assert stack.group_of(n) in stage_trainable_groups(stage), (stage, n)
    stack.zero_grad(set_to_none=True)
    apply_stage_freeze(stack, "S-J")


def test_S_W_touches_no_planner_parameter(stack):
    apply_stage_freeze(stack, "S-W")
    b = synthetic_train_batch(stack, batch=2, k=12, seed=2)
    b["gt_wp"] = torch.randn(2, 10, 2)
    L = v6_loss_step(stack, b, stage="S-W", weights=V6LossWeights(),
                     o1_k=10, o5_k=12)
    assert "plan" not in L and "t1" not in L and "s1" not in L
    L["loss"].backward()
    for n, p in stack.group_parameters("planner"):
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
    stack.zero_grad(set_to_none=True)
    apply_stage_freeze(stack, "S-J")


def test_loss_step_refuses_an_all_zero_weight_set(stack):
    b = synthetic_train_batch(stack, batch=2, k=12, seed=3)
    b["gt_wp"] = torch.randn(2, 10, 2)
    zero = V6LossWeights(o1_ctrl=0, o1_fact=0, o1_scene=0, o2_nearfield=0,
                         o3_masked=0, o5_rollout=0, o6_sigreg=0, t1_latent=0,
                         s1_latent=0, lambda_plan=0, seam_op=0)
    with pytest.raises(RuntimeError, match="NO loss terms"):
        v6_loss_step(stack, b, stage="S-J", weights=zero, o1_k=10, o5_k=12)


def test_plan_loss_reports_both_bands_separately(stack):
    """§4b's eval consequence: four families + oracle/selected at BOTH 0–2 s and
    0–6 s. A pooled number cannot see the seam."""
    apply_stage_freeze(stack, "S-J")
    b = synthetic_train_batch(stack, batch=2, k=12, seed=4)
    b["gt_wp"] = torch.randn(2, 10, 2)
    L = v6_loss_step(stack, b, stage="S-T",
                     weights=V6LossWeights(lambda_plan=1.0), o1_k=10, o5_k=12)
    assert "plan_ade_0_2s" in L["log"] and "plan_ade_2_6s" in L["log"]
    assert L["log"]["plan_ade_0_2s"] != L["log"]["plan_ade_2_6s"]


def test_lambda_plan_without_a_target_is_refused(stack):
    b = synthetic_train_batch(stack, batch=2, k=12, seed=5)
    b["gt_wp"] = torch.randn(2, 10, 2)
    b.pop("plan_target")
    with pytest.raises(ValueError, match="plan_target"):
        v6_loss_step(stack, b, stage="S-T",
                     weights=V6LossWeights(lambda_plan=1.0), o1_k=10, o5_k=12)


@pytest.mark.parametrize("stage", STAGES)
def test_dry_run_writes_config_and_steps(tmp_path, stage):
    """The pre-launch verification the pod runbook depends on: it must exercise
    the REAL loss assembly, not merely prove the module imports."""
    a = tiny_args(tmp_path / stage, stage=stage)
    res = dry_run(a)
    assert len(res["steps"]) == 2
    assert res["isolation"]["pass"] is True
    assert res["param_report"]["within_budget"]
    assert res["n_trainable_tensors"] > 0
    cfg = json.loads((tmp_path / stage / "config.json").read_text())
    assert cfg["stage"] == stage
    assert cfg["horizon_spec"]["plan_steps"] == 60
    assert cfg["horizon_spec"]["horizon_s"] == 6.0
    assert cfg["trainable_groups"] == list(stage_trainable_groups(stage))
    assert cfg["loss_weights_in_force"]["lambda_plan"] == \
           STAGE_LAMBDA_PLAN[stage]
    dr = json.loads((tmp_path / stage / "dry_run.json").read_text())
    assert dr["mode"] == "dry-run" and "quotable" in dr["_read"]


def test_dry_run_of_the_per_layer_encoder_arm(tmp_path):
    a = tiny_args(tmp_path, stage="S-W")
    a.per_layer_encoders = True
    res = dry_run(a)
    assert res["param_report"]["arm"] == "per-layer-encoders"
    assert res["isolation"]["pass"] is True


def test_preflight_refuses_lambda_plan_in_S_W(tmp_path):
    from train_v6_staged import preflight
    a = tiny_args(tmp_path, stage="S-W")
    a.lambda_plan = 1.0
    probs = preflight(a)
    assert any("S-W is the WORLD stage" in p for p in probs)


def test_preflight_refuses_a_staged_run_that_starts_from_nothing(tmp_path):
    """The OTHER half of X5. A gate saying "S-W passed" is worthless if S-T then
    trains on a randomly-initialised trunk — that is not the staged protocol,
    it is four unrelated models with a gate between them."""
    from train_v6_staged import preflight
    for stage in ("S-T", "S-S", "S-J"):
        a = tiny_args(tmp_path, stage=stage)
        a.dry_run = False
        a.v2_cache = ["/data/train"]
        assert any("--init-from" in p for p in preflight(a)), stage
        a.init_from = "/w/prev/ckpt.pt"
        assert not any("--init-from" in p for p in preflight(a)), stage
    a = tiny_args(tmp_path, stage="S-W")           # S-W starts the ladder
    a.dry_run, a.v2_cache = False, ["/data/train"]
    assert not any("--init-from" in p for p in preflight(a))


def test_load_stage_init_round_trips_and_reports_the_trunk_md5(tmp_path):
    """The ladder must be ONE lineage: stage N+1 loads stage N's whole stack,
    and the report names the trunk it stands on."""
    from train_v6_staged import _save_ckpt, load_stage_init
    torch.manual_seed(7)
    src = V6Stack(tiny_cfg())
    with torch.no_grad():                          # move it off its init
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ck = tmp_path / "ckpt.pt"
    opt = torch.optim.AdamW(list(src.parameters())[:1], lr=1e-4)
    _save_ckpt(ck, stack=src, opt=opt, step=1234,
               cfg_json={"stage": "S-W"})
    dst = V6Stack(tiny_cfg())
    assert not torch.equal(next(iter(dst.parameters())),
                           next(iter(src.parameters())))
    rep = load_stage_init(dst, ck)
    assert rep["missing_keys"] == [] and rep["unexpected_keys"] == []
    assert rep["init_step"] == 1234 and rep["prev_stage"] == "S-W"
    assert len(rep["trunk_md5_after_load"]) == 32
    for (n, a), (m, b) in zip(dst.named_parameters(), src.named_parameters()):
        assert n == m and torch.equal(a, b), n
    # the md5 is over the TRUNK only, so it identifies which S-W this stands on
    dst2 = V6Stack(tiny_cfg())
    rep2 = load_stage_init(dst2, ck)
    assert rep2["trunk_md5_after_load"] == rep["trunk_md5_after_load"]


def test_load_stage_init_refuses_a_geometry_mismatch(tmp_path):
    """A key mismatch means the stages were built with DIFFERENT geometry.
    Loading it loosely is how a stage ends up on a random trunk while its log
    looks healthy."""
    from train_v6_staged import _save_ckpt, load_stage_init
    src = V6Stack(tiny_cfg())
    ck = tmp_path / "ckpt.pt"
    _save_ckpt(ck, stack=src, opt=torch.optim.AdamW(
        list(src.parameters())[:1], lr=1e-4), step=1, cfg_json={})
    other = V6Stack(tiny_cfg(d_tac=64))            # different tactical width
    with pytest.raises(RuntimeError):
        load_stage_init(other, ck)
    with pytest.raises(SystemExit, match="does not exist"):
        load_stage_init(src, tmp_path / "nope.pt")


def test_resume_guard_refuses_to_relaunch_a_DONE_run(tmp_path):
    """MEASURED 2026-08-09/11: the v5f run finished but never wrote its
    done-marker; its supervisor relaunched for TWO DAYS, and when the
    crash-cause was fixed a relaunch succeeded, resumed from a stale ckpt.pt,
    and started overwriting the canonical run directory next to a live eval.
    The done-marker IS the remote off-switch — so it has to actually stop a
    launch."""
    from train_v6_staged import resume_guard
    (tmp_path / "summary.json").write_text(json.dumps(
        {"done": True, "stage": "S-W", "steps": 30000,
         "gate_verdict": "PASS"}))
    (tmp_path / "ckpt.pt").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="says this run is DONE"):
        resume_guard(tmp_path, resume="auto", force_rerun=False)
    # --force-rerun is the ONLY way past it, and it is explicit
    rg = resume_guard(tmp_path, resume="auto", force_rerun=True)
    assert rg["mode"] == "resume"
    # a marker that does NOT claim done must not block
    (tmp_path / "summary.json").write_text(json.dumps({"done": False}))
    assert resume_guard(tmp_path, resume="auto",
                        force_rerun=False)["mode"] == "resume"
    # and a corrupt marker must not block either (it is not a done claim)
    (tmp_path / "summary.json").write_text("{not json")
    assert resume_guard(tmp_path, resume="auto",
                        force_rerun=False)["mode"] == "resume"


def test_resume_guard_refuses_to_restart_over_a_live_checkpoint(tmp_path):
    """`supervise_run.sh` SOURCES ITS MANIFEST ONCE and replays the captured
    command, so a relaunch runs the SAME flags. With `--resume off` that would
    restart at step 0 on top of a live ckpt.pt."""
    from train_v6_staged import resume_guard
    assert resume_guard(tmp_path, resume="off",
                        force_rerun=False)["mode"] == "fresh"
    (tmp_path / "ckpt.pt").write_bytes(b"live")
    with pytest.raises(SystemExit, match="--resume off with an existing"):
        resume_guard(tmp_path, resume="off", force_rerun=False)
    assert resume_guard(tmp_path, resume="off",
                        force_rerun=True)["mode"] == "fresh"
    assert resume_guard(tmp_path, resume="auto",
                        force_rerun=False) == {"mode": "resume",
                                               "from": str(tmp_path
                                                           / "ckpt.pt")}


def test_load_resume_restores_stack_optimiser_and_step(tmp_path):
    from train_v6_staged import _save_ckpt, load_resume
    torch.manual_seed(11)
    src = V6Stack(tiny_cfg())
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    opt = torch.optim.AdamW(src.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    next(iter(src.parameters())).grad = torch.ones_like(
        next(iter(src.parameters())))
    opt.step()                                   # give the optimiser real state
    ck = tmp_path / "ckpt.pt"
    _save_ckpt(ck, stack=src, opt=opt, step=4321, cfg_json={"stage": "S-W"})

    dst = V6Stack(tiny_cfg())
    dopt = torch.optim.AdamW(dst.parameters(), lr=1e-4)
    step = load_resume(dst, dopt, ck)
    assert step == 4321
    for (n, a), (m, b) in zip(dst.named_parameters(), src.named_parameters()):
        assert n == m and torch.equal(a, b), n
    assert dopt.state_dict()["state"], "optimiser moments were not restored"


def test_preflight_refuses_a_reasonless_gate_override(tmp_path):
    from train_v6_staged import preflight
    a = tiny_args(tmp_path)
    a.allow_inconclusive_gate = True
    a.gate_off_reason = ""
    assert any("gate-off-reason" in p for p in preflight(a))


def test_preflight_flags_the_isolation_control_arm(tmp_path):
    from train_v6_staged import preflight
    a = tiny_args(tmp_path)
    a.no_isolate_planner = True
    assert any("ISOLATION DISABLED" in p for p in preflight(a))


def test_preflight_requires_the_corpus_for_a_real_run(tmp_path):
    from train_v6_staged import preflight
    a = tiny_args(tmp_path)
    a.dry_run = False
    assert any("--v2-cache is required" in p for p in preflight(a))


def test_run_stage_gate_names_what_it_could_not_run(tmp_path, stack):
    """Rule 2 applied to the battery: an unavailable probe must SAY what was
    not reachable and WHERE it lives — never silently vanish, never pass."""
    from train_v6_staged import run_stage_gate
    gate = run_stage_gate(stack, "S-W", out_dir=tmp_path)
    assert gate["pass"] is None                      # nothing ran -> NOT a pass
    assert gate["verdict"] == "INCONCLUSIVE"
    for name in STAGE_GATE_SPEC["S-W"]["required"]:
        assert gate["probes"][name]["pass"] is None
        assert gate["probes"][name]["owner"] != "?"
        assert gate["probes"][name]["reason"]
    # X3 is the one gate this module can always measure on its own
    assert gate["probes"]["X3_isolation"]["pass"] is True
    assert (tmp_path / "stage_gate.json").exists()


def test_gate_folds_in_externally_run_probes(tmp_path, stack):
    from train_v6_staged import run_stage_gate
    ext = {p: {"pass": True, "status": "run", "artifact": "w3_gate.json"}
           for p in STAGE_GATE_SPEC["S-W"]["required"]}
    gate = run_stage_gate(stack, "S-W", out_dir=tmp_path, extra_probes=ext)
    assert gate["pass"] is True and gate["verdict"] == "PASS"
    assert gate["param_report"]["within_budget"]


def test_the_whole_ladder_hands_off_through_the_WRITTEN_files(tmp_path, stack):
    """END-TO-END X5: the gate file a stage WRITES must be the gate file the
    next stage READS.

    The pieces are tested separately above, but the handoff is the thing X5 is
    actually about — and a seam that is only ever tested from both sides at
    once is exactly the kind that fails on the pod. This walks the real ladder
    S-W → S-T → S-S → S-J through :func:`run_stage_gate`'s files, then plants a
    FAIL at the tactical rung and checks the ladder stops there.
    """
    from train_v6_staged import run_stage_gate
    paths = {}
    for stage in STAGES:                       # every rung PASSES
        d = tmp_path / stage
        ext = {p: {"pass": True, "status": "run"}
               for p in STAGE_GATE_SPEC[stage]["required"]}
        g = run_stage_gate(stack, stage, out_dir=d, extra_probes=ext)
        assert g["pass"] is True, stage
        paths[stage] = d / "stage_gate.json"
        assert paths[stage].exists()
    # each stage accepts the PREVIOUS stage's own written file, and that file
    # names the stage after it — the ladder is linked, not four islands
    for stage, prev in STAGE_PRECONDITION.items():
        if prev is None:
            continue
        rep = assert_stage_precondition(stage, paths[prev])
        assert rep["ok"] and rep["prev_verdict"] == "PASS"
        assert json.loads(paths[prev].read_text())["next_stage"] == stage

    # now FAIL the tactical rung: S-S must refuse, and no flag opens it
    d = tmp_path / "S-T-failed"
    bad = {p: {"pass": True, "status": "run"}
           for p in STAGE_GATE_SPEC["S-T"]["required"]}
    bad["sel_gap"] = {"pass": False, "status": "run", "value": 0.91}
    g = run_stage_gate(stack, "S-T", out_dir=d, extra_probes=bad)
    assert g["pass"] is False and g["failed_required"] == ["sel_gap"]
    assert "MUST NOT launch" in g["bound_outcomes"]["FAIL"]
    with pytest.raises(GatePreconditionError, match="FAILED its gate"):
        assert_stage_precondition("S-S", d / "stage_gate.json")
    with pytest.raises(GatePreconditionError, match="no override for a FAIL"):
        assert_stage_precondition("S-S", d / "stage_gate.json",
                                  allow_inconclusive=True,
                                  off_reason="the schedule slipped")


def test_the_g_tac_seam_TRAINS_in_ST_and_not_in_SW():
    """⛔ FOUND BY A PI QUESTION, 2026-08-13: "how will the operative predictor
    learn to be actioned by the tactical goals if it is trained alone?"

    With intent_proj grouped under `predictor_op`, S-W trained it while
    intent=None (zero gradient — dead weight at random init) and S-T FROZE it
    exactly when g_tac first flows. The downlink of the hierarchy would have
    silently stayed random until S-J — the same failure shape as v5's
    nav-echo: a conditioning path that exists in the diagram but not in the
    optimisation. The seam is now grouped with the PLANNER (the
    goal-conditioning side); the trunk DYNAMICS stay frozen in S-T."""
    from tanitad.models.v6 import V6Config, V6Stack, apply_stage_freeze
    m = V6Stack(V6Config())
    apply_stage_freeze(m, "S-W")
    sw = {n for n, p in m.named_parameters()
          if "intent" in n and p.requires_grad}
    assert sw == set(), f"S-W must not train the unused seam: {sw}"
    apply_stage_freeze(m, "S-T")
    st = {n for n, p in m.named_parameters()
          if "intent" in n and p.requires_grad}
    assert "predictor_op.intent_proj.weight" in st, (
        "S-T must train the g_tac->operative seam — otherwise the hierarchy's "
        "downlink learns only at S-J")
    # and the trunk dynamics stay frozen in S-T
    blocked = {n for n, p in m.named_parameters()
               if n.startswith("predictor_op.blocks") and p.requires_grad}
    assert blocked == set(), f"S-T must not train trunk dynamics: {blocked}"


def test_seam_loss_trains_intent_proj_in_ST(stack):
    """⛔ THE OTHER HALF of the seam fix (the grouping was the first half): an
    S-T loss must actually FLOW through the goal-conditioned operative
    prediction. Before this term, t1_latent trained layer_tac and lambda_plan
    trained the planner, and zhat_op_seam fed NOTHING — the downlink existed in
    forward and not in the optimisation."""
    import torch
    from tanitad.models.v6 import apply_stage_freeze
    with torch.no_grad():                     # off the zero inits (see the
        for blk in stack.predictor_op.blocks:  # planner-surface test)
            blk.film.to_scale_shift.weight.normal_(0.0, 0.1)
    apply_stage_freeze(stack, "S-T")
    b = synthetic_train_batch(stack, batch=2, k=2, seed=5)
    b["plan_target"] = torch.randn(2, stack.cfg.plan_steps, 2)
    w = V6LossWeights().for_stage("S-T")
    assert w.seam_op > 0
    r = v6_loss_step(stack, b, stage="S-T", weights=w, o1_k=1, o5_k=1)
    assert "seam" in r and "seam_op" in r["log"]
    r["loss"].backward()
    g = stack.predictor_op.intent_proj.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, (
        "seam loss must reach intent_proj — otherwise the hierarchy downlink "
        "still does not learn in S-T")


def test_modern_predictor_blocks_have_the_vit5_properties():
    """ViT-5-recipe predictor (PI 2026-08-13): RMSNorm, QK-Norm, LayerScale,
    bias-free attention, GeLU (NOT SwiGLU), and the SAME FiLM seam."""
    import torch
    from tanitad.config import PredictorConfig
    from tanitad.models.predictor import ModernCausalBlock, OperativePredictor
    cfg = PredictorConfig(d_model=128, depth=2, n_heads=4, action_dim=3,
                          modern=True)
    m = OperativePredictor(cfg, state_dim=256, intent_dim=64)
    blk = m.blocks[0]
    assert isinstance(blk, ModernCausalBlock)
    assert blk.qkv.bias is None and blk.proj.bias is None
    assert torch.allclose(blk.ls1, torch.full_like(blk.ls1, 1e-5))
    acts = [type(x).__name__ for x in blk.mlp]
    assert "GELU" in acts and len(blk.mlp) == 3      # 2-matmul FFN, no gating
    # FiLM seam identical: zero-init -> identity start
    assert float(blk.film.to_scale_shift.weight.abs().sum()) == 0.0
    # legacy arm unchanged
    leg = OperativePredictor(PredictorConfig(d_model=128, depth=2, n_heads=4,
                                             action_dim=3), state_dim=256)
    from tanitad.models.predictor import CausalBlock
    assert isinstance(leg.blocks[0], CausalBlock)


def test_modern_predictor_forward_backward_and_distinct_state_dict():
    """The modern arm must run end-to-end AND be un-resumable from a legacy
    checkpoint (strict load must fail loudly, never silently mis-map)."""
    import torch
    from tanitad.config import PredictorConfig
    from tanitad.models.predictor import OperativePredictor
    kw = dict(d_model=128, depth=2, n_heads=4, action_dim=3)
    new = OperativePredictor(PredictorConfig(modern=True, **kw), state_dim=256)
    old = OperativePredictor(PredictorConfig(**kw), state_dim=256)
    out = new(torch.randn(2, 6, 256), torch.randn(2, 6, 3))
    sum(o.sum() for o in out.values()).backward()
    import pytest as _pt
    with _pt.raises(RuntimeError):
        new.load_state_dict(old.state_dict(), strict=True)
