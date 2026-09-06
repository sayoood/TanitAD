"""⭐⭐ THE SIX LINKS, ASSERTED POSITIVELY, PLUS THE MUTATION THAT DECIDES.

The tactical-GOAL vocabulary was minted on 4,572 clips and trained by nothing
(audit ``…/incoming/2026-09-06-label-vocab-audit/AUDIT_RESULT.json``). Wiring it
is six links, and **each of the six has failed separately in this programme
within one week**:

  L1 label present in a REAL record
  L2 label reaches the BATCH
  L3 the batch reaches the FORWARD
        -- a hand-maintained tuple mirroring a forward signature drifted TWICE
           (``rl/refc_adapter.py`` FORWARD_KEYS): a channel absent from it is
           never passed, WITH NO ERROR.
  L4 the forward output enters a LOSS WITH NON-ZERO WEIGHT
        -- 42 of 138 optimizer tensors got no gradient (52.2 % of a declared
           budget) because terms were weighted 0.0 AND the loss GUARDED them.
           ⭐ ``p.grad is None`` is the discriminator, never the weight's value.
  L5 the head is EVALUATED
        -- every refcv3/refcv4b eval read ``UNKNOWN_SCOPE`` to the census.
  L6 it is produced AT INFERENCE from vision + measured ego only.

⛔ A test that only checks shapes proves link 3 and nothing else. So every
assertion here is POSITIVE (a value that must be there), each is paired with a
SAME-BREATH DISCRIMINATING CONTROL that must read a different known value, and
the decisive one is a MUTATION: perturb the label, require the loss AND a
gradient to move, and require a control mutation NOT to move them.

⚠️ MASK BEFORE DEATH. A sibling found the route head's apparent zero gradient
was a VALIDITY MASK, not a dead head — forcing ``route_valid=True`` moved the
loss 0.0 → 0.687. :func:`test_zero_loss_is_the_mask_not_a_dead_head` reproduces
that discrimination for this head, so the same misreading cannot happen twice.

⛔ CPU only, random-init tiny model, synthetic records. Touches no pod, no Thor,
no checkpoint, no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.data import v7_labels as v7l
from tanitad.models import vocab_v7 as V7
from tanitad.refs import refc_v3 as v3
from tanitad.refs import refc
from tanitad.refs import tac_goal_head as TG

N_STACK, SIZE, WIN = 3, 64, 2
RED, GREEN = "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _frames(b: int) -> torch.Tensor:
    """[B, W, C, H, W'] - the shape RefCModel.forward documents."""
    return torch.randn(b, WIN, 3 * N_STACK, SIZE, SIZE)


def _label(clip: str, goals: dict, *, t0: float = 8.0) -> v7l.V7Label:
    """A record shaped exactly as ``load_v7_labels`` builds one."""
    return v7l.V7Label(
        clip_id=clip, tac_lat="LANE_KEEP", tac_lon="CRUISE",
        str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
        tac_anchor=None, bands={"tactical_s": [2.0, 6.0]}, t0_s=t0,
        horizon={}, tac_goals=frozenset(goals),
        tac_goal_meta={k: dict(v) for k, v in goals.items()})


def _corpus() -> list[v7l.V7Label]:
    """Three records that exercise BOTH provenances and the exclusion table."""
    return [
        _label("clip_red", {
            "SPEED_BAND": {}, "FOLLOW_LANE": {"provenance": "geometry"},
            RED: {"provenance": "vlm-cot", "state": "red"}}),
        _label("clip_green", {
            "SPEED_BAND": {}, "FOLLOW_LANE": {"provenance": "geometry"},
            GREEN: {"provenance": "vlm-cot", "state": "green"}}),
        _label("clip_turn", {
            "SPEED_BAND": {}, "TURN_L": {}, "YIELD_FOR_TURN_L": {}}),
    ]


@pytest.fixture()
def geom_tokens(monkeypatch):
    """Pin the split-derived negative policy, as ``load_v7_labels`` would."""
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", frozenset(
        {"FOLLOW_LANE", "SPEED_BAND", "STOP_POINT", "TURN_L", "TURN_R",
         "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"}))
    return v7l._MEASURED_GEOMETRY_TOKENS


def _model():
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=3 * N_STACK, image_size=SIZE, base_width=8,
        blocks=(1, 1, 1, 1))
    cfg.tac_vocab_version = "v7.0"          # the flag `--v7-labels` pins
    return v3.RefCV3Model(cfg), cfg


# ==========================================================================
# L1 — the label is in a real record, and it is a SET
# ==========================================================================
def test_L1_the_goal_set_reaches_the_label_object(geom_tokens):
    labs = _corpus()
    assert RED in labs[0].tac_goals, "L1 FAILED: the goal set never loaded"
    # CONTROL, same breath: a token that is NOT in this record must be absent,
    # so the assertion above is about content and not about a truthy container.
    assert GREEN not in labs[0].tac_goals
    assert labs[0].tac_goal_meta[RED]["state"] == "red"


def test_L1b_the_head_is_not_in_HEADS_and_that_is_deliberate():
    """⛔ ``HEADS`` stays the four SOFTMAX heads. The goal set is MULTI-LABEL
    (2-7 tokens/record, mean 2.751 MEASURED), and ``head_mask`` /
    ``class_weights`` / ``assert_mask_matches_presence`` all index a
    single-label vocabulary. Merging it in would make a SET look like a CHOICE
    — the 5-way-softmax defect one layer up — and silently break three
    functions that iterate ``HEADS`` generically."""
    assert set(v7l.HEADS) == {"tac_lat", "tac_lon", "str_action", "str_goal"}
    assert v7l.TAC_GOAL_HEAD == "tac_goal" and v7l.TAC_GOAL_HEAD not in v7l.HEADS
    assert len(v7l.TAC_GOAL_TOKENS) == 22


# ==========================================================================
# L2 — it reaches the BATCH, with the right validity
# ==========================================================================
def test_L2_targets_carry_positives_entailed_negatives_and_ignores(geom_tokens):
    lab = _corpus()[0]                              # carries RED
    y, w = v7l.tactical_goal_targets(lab, lab.t0_s)
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}

    assert y[ix[RED]] == 1.0 and w[ix[RED]] == 1.0, "positive not supervised"
    # ⭐ GREEN is FALSE BY ENTAILMENT (the frozen exclusion table), not by an
    # assumption that the caption was exhaustive. This is the single change
    # that makes the traffic-light group trainable at all.
    assert y[ix[GREEN]] == 0.0 and w[ix[GREEN]] == 1.0, \
        "entailed negative not supervised — the TL group cannot train"
    # CONTROL, same breath: a CoT-backed token with NO exclusion partner here
    # must be IGNORED, not supervised as a negative. If this reads 1.0 the
    # policy has silently become 'absence = negative' and would teach the head
    # that ~78 % of the corpus has no traffic light, on no evidence.
    assert w[ix["GAP_TARGET"]] == v7l.IGNORE_W
    # and a geometry token IS supervised as a negative, because that emitter
    # is exhaustive
    assert w[ix["STOP_POINT"]] == 1.0


def test_L2b_out_of_band_is_all_ignored_not_all_negative(geom_tokens):
    """⛔ The band rule is the record's OWN ±(hi-lo)/2, never a literal."""
    lab = _corpus()[0]
    _, w_in = v7l.tactical_goal_targets(lab, lab.t0_s + 1.9)
    y_out, w_out = v7l.tactical_goal_targets(lab, lab.t0_s + 2.1)
    assert max(w_in) == 1.0, "in-band window supervised nothing"
    assert max(w_out) == v7l.IGNORE_W, \
        "out-of-band window trains — an unlabelled cell must not train a class"
    assert set(y_out) == {0.0}


def test_L2c_emitter_join_refuses_an_unmapped_episode(geom_tokens):
    labs = _corpus()
    em = v7l.TacGoalEmitter(labs, {0: "clip_red", 1: "clip_green"})
    y, w = em(torch.tensor([0, 1]), t_now_s=torch.tensor([8.0, 8.0]))
    assert y.shape == (2, 22) and w.shape == (2, 22)
    ix = v7l.TAC_GOAL_TOKENS.index(RED)
    assert float(y[0, ix]) == 1.0 and float(y[1, ix]) == 0.0
    # CONTROL: a default would attach one clip's goals to another's windows.
    with pytest.raises(v7l.NavTokenMissing):
        em(torch.tensor([7]), t_now_s=torch.tensor([8.0]))


# ==========================================================================
# L3 — the forward EMITS it (the FORWARD_KEYS drift class)
# ==========================================================================
def test_L3_forward_emits_tac_goal_logits_at_vocabulary_width():
    model, cfg = _model()
    assert model.tac_goal_tok_head is not None, \
        "L3 FAILED: no head was built under a v7 vocabulary"
    frames = _frames(2)
    out = model(frames, v0=torch.zeros(2))
    assert "tac_goal_logits" in out, \
        ("L3 FAILED: the head exists but its output never reaches the forward "
         "dict — this is the FORWARD_KEYS drift class, which is SILENT.")
    assert out["tac_goal_logits"].shape == (2, 22)
    # CONTROL, same breath: the pre-existing GEOMETRIC goal head is a different
    # object and must still be there at its own width. Two things called 'the
    # tactical goal head' is how one arm's number gets quoted for the other.
    assert "g_tac" in out and out["g_tac"].shape[-1] != 22


def test_L3b_kin3_builds_no_head_rather_than_dead_logits():
    """⛔ ``kin3`` has no tactical goal vocabulary. Building 22 logits there
    would be 22 classes that can never be supervised."""
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=3 * N_STACK, image_size=SIZE, base_width=8,
        blocks=(1, 1, 1, 1))
    cfg.tac_vocab_version = "kin3"          # no --v7-labels on the command line
    m = v3.RefCV3Model(cfg)
    assert m.tac_goal_tok_head is None
    out = m(_frames(1), v0=torch.zeros(1))
    assert "tac_goal_logits" not in out
    # CONTROL: the same build still emits the action heads, so the absence
    # above is scoped to the goal head and is not a broken forward.
    assert "lat_logits_tac" in out


def test_L3c_a_width_mismatch_is_REFUSED_not_broadcast():
    """BCE broadcasts silently; the z_tac refusal exists because an 8-wide head
    once trained against 3-class labels (2026-09-01). Mirror it."""
    with pytest.raises(TG.TacGoalVocabMismatch):
        TG.assert_head_matches_vocabulary(torch.zeros(2, 8))
    TG.assert_head_matches_vocabulary(torch.zeros(2, 22))     # control


# ==========================================================================
# L4 — a LOSS WITH NON-ZERO WEIGHT, and a real gradient
# ==========================================================================
def test_L4_loss_produces_gradient_on_the_head(geom_tokens):
    model, _ = _model()
    labs = _corpus()
    em = v7l.TacGoalEmitter(labs, {0: "clip_red", 1: "clip_green"})
    frames = _frames(2)
    out = model(frames, v0=torch.zeros(2))
    y, w = em(torch.tensor([0, 1]), t_now_s=torch.tensor([8.0, 8.0]))
    loss, n_sup = TG.tac_goal_loss(out["tac_goal_logits"], y, w)
    assert n_sup > 0, "L4 FAILED: nothing supervised — check the band/mask"
    assert float(loss.detach()) > 0.0
    model.zero_grad(set_to_none=True)
    loss.backward()
    p = model.tac_goal_tok_head.net.weight
    assert p.grad is not None, "L4 FAILED: the head received NO gradient"
    assert float(p.grad.abs().sum()) > 0.0
    # ⭐ CONTROL, same breath: a parameter NOT on this path must read grad None
    # under this loss alone. Without it, 'grad is not None' could be an artifact
    # of a leftover gradient rather than of this term.
    assert model.core.route_head.weight.grad is None


def test_L4b_zero_weight_still_yields_a_ZEROS_grad_not_None(geom_tokens):
    """⛔⛔ THE 52.2 %-OF-BUDGET DEFECT, PINNED.

    Terms weighted 0.0 AND guarded produce ``p.grad is None``, which is
    indistinguishable from a head that was never wired. Here the term is always
    computed and the weight multiplies, so a switched-OFF head reads
    ``grad is not None`` with an all-zero tensor and the two states can never
    again be confused.
    """
    model, _ = _model()
    labs = _corpus()
    em = v7l.TacGoalEmitter(labs, {0: "clip_red"})
    out = model(_frames(1), v0=torch.zeros(1))
    y, w = em(torch.tensor([0]), t_now_s=torch.tensor([8.0]))
    loss, _ = TG.tac_goal_loss(out["tac_goal_logits"], y, w)
    model.zero_grad(set_to_none=True)
    (0.0 * loss).backward()
    g = model.tac_goal_tok_head.net.weight.grad
    assert g is not None, "a zero-weighted term must still be WIRED"
    assert float(g.abs().sum()) == 0.0
    # CONTROL: the same term at weight 1.0 moves it, so the zero above is the
    # weight and not a detached graph.
    model.zero_grad(set_to_none=True)
    out2 = model(_frames(1), v0=torch.zeros(1))
    l2, _ = TG.tac_goal_loss(out2["tac_goal_logits"], y, w)
    l2.backward()
    assert float(model.tac_goal_tok_head.net.weight.grad.abs().sum()) > 0.0


def test_zero_loss_is_the_mask_not_a_dead_head(geom_tokens):
    """⚠️ CHECK THE MASK BEFORE DECLARING ANYTHING DEAD.

    Out of band every cell is ignored, so the loss is 0.0 with ``n_sup == 0``.
    That is the route-head shape: a validity mask reading as a dead head.
    Forcing the mask on must move it — as forcing ``route_valid=True`` moved
    that loss 0.0 -> 0.687.
    """
    model, _ = _model()
    lab = _corpus()[0]
    out = model(_frames(1), v0=torch.zeros(1))
    y_out, w_out = v7l.tactical_goal_targets(lab, lab.t0_s + 9.0)   # out of band
    y_t = torch.tensor([list(y_out)])
    loss_masked, n0 = TG.tac_goal_loss(out["tac_goal_logits"], y_t,
                                       torch.tensor([list(w_out)]))
    assert n0 == 0 and float(loss_masked) == 0.0
    # ⛔⛔ THE ASSERTION THAT CATCHES THE GUARD, AND IT WAS MISSING.
    # MEASURED 2026-09-06: reintroducing `if n_sup == 0: return zeros(), 0`
    # into `tac_goal_loss` left this file at 17/17 GREEN. A suite that cannot
    # fail on the defect it names certifies nothing — the same lesson as the
    # AST census that read 0 suspects on BOTH the fixed and the broken trainer.
    # An ALL-IGNORED batch must still be WIRED: the term stays in the graph, so
    # every parameter receives a ZEROS gradient and `p.grad is None` keeps its
    # meaning of "never wired". A guarded early return breaks exactly this.
    model.zero_grad(set_to_none=True)
    loss_masked.backward(retain_graph=True)
    g = model.tac_goal_tok_head.net.weight.grad
    assert g is not None, (
        "GUARD DEFECT: an all-ignored batch returned a DETACHED zero, so "
        "`p.grad is None` no longer discriminates 'not wired' from 'nothing "
        "in band'. This is the 42-of-138 (52.2 %) no-gradient defect.")
    assert float(g.abs().sum()) == 0.0
    forced, n1 = TG.tac_goal_loss(out["tac_goal_logits"], y_t,
                                  torch.ones_like(y_t))
    assert n1 == 22 and float(forced) > 0.0, \
        "forcing the mask did not move the loss — THIS one really is dead"


# ==========================================================================
# ⭐⭐ THE MUTATION — perturb the label, require loss AND gradient to move
# ==========================================================================
def test_MUTATION_flipping_red_to_green_moves_loss_and_gradient(geom_tokens):
    model, _ = _model()
    frames = _frames(2)
    torch.manual_seed(1)
    out = model(frames, v0=torch.zeros(2))
    logits = out["tac_goal_logits"]

    base = _corpus()
    mutated = [_label("clip_red", {                     # RED  -> GREEN
        "SPEED_BAND": {}, "FOLLOW_LANE": {"provenance": "geometry"},
        GREEN: {"provenance": "vlm-cot", "state": "green"}}), base[1], base[2]]

    def run(labs):
        em = v7l.TacGoalEmitter(labs, {0: labs[0].clip_id, 1: "clip_green"})
        y, w = em(torch.tensor([0, 1]), t_now_s=torch.tensor([8.0, 8.0]))
        loss, n = TG.tac_goal_loss(logits, y, w)
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return (float(loss), n,
                model.tac_goal_tok_head.net.weight.grad.clone())

    l0, n0, g0 = run(base)
    l1, n1, g1 = run(mutated)
    assert n0 == n1, ("the mutation changed HOW MANY cells are supervised, so "
                      "a loss difference would not isolate the label")
    assert abs(l1 - l0) > 1e-6, \
        "MUTATION FAILED: flipping the traffic-light colour did not move the loss"
    assert float((g1 - g0).abs().sum()) > 1e-6, \
        "MUTATION FAILED: the gradient is identical under a flipped label"

    # ⭐⭐ THE DISCRIMINATING CONTROL, SAME BREATH, SAME FUNCTION. Flip the
    # TARGET on a cell the policy IGNORES. The target tensor genuinely changes,
    # and the loss and the gradient must not move at all. The two mutations
    # differ in exactly ONE thing — whether the cell carries evidence — so
    # together they prove the movement above is THIS LABEL and not any
    # perturbation of a tensor.
    em = v7l.TacGoalEmitter(base, {0: "clip_red", 1: "clip_green"})
    y0, w0 = em(torch.tensor([0, 1]), t_now_s=torch.tensor([8.0, 8.0]))
    ig = v7l.TAC_GOAL_TOKENS.index("GAP_TARGET")
    assert float(w0[:, ig].sum()) == 0.0, "the probe cell is not actually ignored"
    y_ctrl = y0.clone()
    y_ctrl[:, ig] = 1.0
    assert not torch.equal(y_ctrl, y0), "the control did not change the target"
    l2t, n2 = TG.tac_goal_loss(logits, y_ctrl, w0)
    model.zero_grad(set_to_none=True)
    l2t.backward(retain_graph=True)
    g2 = model.tac_goal_tok_head.net.weight.grad.clone()
    l2 = float(l2t)
    assert n2 == n0, "the control changed the supervised-cell count"
    assert abs(l2 - l0) < 1e-9, \
        ("CONTROL FAILED: an IGNORED cell moved the loss — the ignore policy "
         "is not being applied and CoT-absence is training as a negative")
    assert float((g2 - g0).abs().sum()) < 1e-9


# ==========================================================================
# L5 — it is EVALUATED, per class, against a control that must read its value
# ==========================================================================
def test_L5_per_class_recall_never_pooled_accuracy(geom_tokens):
    labs = _corpus() * 8
    em = v7l.TacGoalEmitter(labs, {i: l.clip_id for i, l in enumerate(labs)})
    y, w = em(torch.arange(len(labs)),
              t_now_s=torch.full((len(labs),), 8.0))
    logits = torch.full((len(labs), 22), -4.0)          # a NEVER-FIRE head
    sc = TG.per_class_scores(logits, y, w)
    assert set(sc) == set(v7l.TAC_GOAL_TOKENS)
    # ⭐ THE SHAPE THE PROGRAMME HAS ALREADY BANKED: a class with real positives
    # whose recall is EXACTLY 0.0000 while any pooled score looks healthy.
    assert sc[RED]["n_pos"] > 0 and sc[RED]["recall"] == 0.0
    assert sc[RED]["n_fired"] == 0
    # CONTROL, same breath: an all-fire head must read recall 1.0 on the same
    # cells, so the 0.0 above is the head and not the scorer.
    sc2 = TG.per_class_scores(torch.full((len(labs), 22), 4.0), y, w)
    assert sc2[RED]["recall"] == 1.0
    # and an IGNORED class is scored on nothing rather than counted as a miss
    assert sc[ "GAP_TARGET"]["n_pos"] == 0 and sc["GAP_TARGET"]["recall"] is None


def test_L5b_majority_control_reads_its_KNOWN_value(geom_tokens):
    """⭐⭐ The control that must read a known value, or the panel is a bug.

    A majority-class predictor's recall is known a priori: 0.0 where the
    majority is ABSENT, 1.0 where it is PRESENT. Three of four manufactured
    results on 2026-08-22 were caught only because a control read the same
    value as the thing being measured.
    """
    labs = _corpus() * 8
    em = v7l.TacGoalEmitter(labs, {i: l.clip_id for i, l in enumerate(labs)})
    y, w = em(torch.arange(len(labs)), t_now_s=torch.full((len(labs),), 8.0))
    ctrl = TG.majority_control_scores(y, w)
    # RED is 8 of 24 supervised rows -> majority ABSENT -> recall EXACTLY 0.0
    assert ctrl[RED]["majority"] == "absent" and ctrl[RED]["recall"] == 0.0
    # FOLLOW_LANE is 16 of 24 -> majority PRESENT -> recall EXACTLY 1.0
    assert ctrl["FOLLOW_LANE"]["majority"] == "present"
    assert ctrl["FOLLOW_LANE"]["recall"] == 1.0
    # ⛔ and the majority control's PRECISION on FOLLOW_LANE is its prevalence:
    # a head must beat this, not merely be positive.
    assert ctrl["FOLLOW_LANE"]["precision"] == pytest.approx(16 / 24)


def test_L5c_pos_weight_and_mask_come_from_the_split(geom_tokens):
    labs = _corpus() * 8
    pw = v7l.goal_pos_weight(labs)
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}
    assert pw[ix[RED]] == pytest.approx(1.0)     # 8 pos / 8 entailed neg
    # ⛔ a class with NO supervised positive gets 0.0 AND must be masked
    assert pw[ix["MERGE"]] == 0.0
    census = v7l.goal_supervision_census(labs)
    rep = TG.mask_report(census)
    assert rep["mask"][ix[RED]] is True
    assert rep["mask"][ix["SPEED_BAND"]] is False, \
        ("SPEED_BAND is present on 100 % of records and has no supervised "
         "negative; an unmasked logit there can only be pushed towards 1")
    assert "no supervised negative" in rep["masked_why"]["SPEED_BAND"]


# ==========================================================================
# L6 — produced at INFERENCE from vision + measured ego, with no label input
# ==========================================================================
def test_L6_inference_needs_only_frames_and_measured_v0(geom_tokens):
    """⛔ The goal SET is a TARGET. If the forward needed it as an input the
    head would be echoing its own label — the flagship-v1 route echo (a
    bijection of its own input that scored 1.0000)."""
    model, _ = _model()
    model.eval()
    with torch.no_grad():
        out = model(_frames(1),
                    v0=torch.zeros(1))
    assert out["tac_goal_logits"].shape == (1, 22)
    assert torch.isfinite(out["tac_goal_logits"]).all()
    # ⭐ CONTROL, same breath: the forward signature must expose NO tactical-goal
    # target channel at all, so the echo is impossible by construction rather
    # than by convention.
    import inspect
    params = set(inspect.signature(model.forward).parameters)
    assert not (params & {"tac_goal", "tac_goals", "tac_goal_gt",
                          "tac_goal_target"}), sorted(params)


def test_L6b_the_head_reads_the_same_latent_as_the_action_heads():
    """A goal and the action serving it are predicted from ONE tactical latent —
    the emitter's own ``serves_goals`` invariant, kept on the model side."""
    model, cfg = _model()
    assert model.tac_goal_tok_head.net.in_features == cfg.d_tac
    assert model.lat_head_tac.in_features == cfg.d_tac
