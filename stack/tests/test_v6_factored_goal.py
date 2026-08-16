"""The FACTORED `g_tac` goal head — gated, default-off, and proved inert when off.

⛔ WHY EVERY TEST HERE EXISTS. `ANCHOR_GOAL_SUPERVISION.md` (2026-08-16, 881
windows / 40 episodes, LOEO, paired episode-cluster bootstrap) measured four
things that together say the goal layer repeats the programme's oldest defect:

* the 2 s goal's corpus variance is **98.8 % LONGITUDINAL** (σ_long 19.0578 vs
  σ_lat 2.0723);
* a K-way `anchor_id` classifier is near-adequate LATERALLY (1.3310 against a
  0.6802 floor) and hopeless LONGITUDINALLY (13.3502 against 0.8954);
* ⇒ ONE K-way index carries BOTH axes — the 5-way manoeuvre softmax defect,
  one level up, **after** `a_tac` was factored LAT × LON to retire it;
* and the failure is the **ESTIMATOR, not the estimand**: ``snap`` (the same
  ridge, rounded to the nearest anchor) is NOT separated from the ridge
  (−0.0002 [−0.1031, +0.0703]) while the one-hot target costs **+4.7502
  [+3.0514, +6.3981]**.

⛔ AND THE CONSTRAINT THAT OUTRANKS ALL OF IT: v6F S-W is training on Thor from
a checkpoint of this exact architecture. Every addition is gated so the DEFAULT
build's ``state_dict`` stays byte-identical — proved here per tensor with
``torch.equal`` against the module as it exists at git HEAD, never by hashing a
``torch.save`` container (RETRACTION_LOG C72: those bytes are not canonical).
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.v6 import (  # noqa: E402
    GOAL_CAT_ARG_NAMES, GOAL_CAT_ARG_TOKENS, LIGHT_STATES, STOP_REASONS,
    TACTICAL_GOAL_TOKENS, TACTICAL_GOAL_TOKENS_LAT, TACTICAL_GOAL_TOKENS_LON,
    AnchorGoalHead, GoalConditioner, GoalHead, GoalVocabulary, V6Config,
    V6Stack, goal_token_axis)

#: MEASURED on this box 2026-08-16, HEAD 37220d2, torch 2.11.0+cu128, seed 0,
#: BEFORE any of this turn's edits — the numbers the default build must not move.
HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS = 611_293, 223
HEAD_FULL_PARAMS, HEAD_FULL_KEYS = 87_893_449, 405

#: MEASURED this turn on the FULL ``V6Config()`` (d_tac 512, d_goal_embed 128,
#: head hidden 256, K 256, n_lat_bins 16, n_agent_slots 8). Recompute, never
#: inherit: ``selector="mlp"``'s design-time estimate of +41,089 was never
#: realised and the implementation cost +33,801.
FACTORED_DELTA_FULL = 470_939
CAT_DELTA_FULL = 110_880
ANCHOR_SNAP_DELTA_FULL = 258           # Linear(d_goal_embed -> 2), on top of cat
ANCHOR_ONEHOT_DELTA_FULL = 33_024      # Linear(d_goal_embed -> K), on top of cat

#: the config keys the pre-2026-08-16 ``V6Config`` knows — the HEAD snapshot is
#: built from exactly these, so the two arms differ by the new code and nothing
#: else.
_SMALL_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1,
    op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0,
    hz_str=0.5, d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
    n_candidates=8)


def _sub_cfgs() -> dict:
    """Fresh sub-configs per stack — a shared mutable default would make one
    test's build depend on another's."""
    return dict(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  horizons=(1,), action_dim=3, residual=True))


def _small(**kw) -> V6Config:
    return V6Config(**{**_sub_cfgs(), **_SMALL_KW, **kw})


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


def _load_table(stack: V6Stack, seed: int = 3) -> dict:
    """A synthetic vocabulary AT THE PLAN HORIZON, so the head will accept it.
    (The shipped ones are all 2 s and are correctly refused — see the 6 s test.)"""
    torch.manual_seed(seed)
    cfg = stack.cfg
    steps = list(range(1, cfg.plan_steps + 1))
    return stack.anchor_head.load_anchor_table(
        torch.randn(cfg.n_anchors, len(steps), 2) * 3.0, horizons=steps,
        dt=cfg.dt)


# =========================================================================== #
# 1. ⛔ THE BYTE-IDENTITY PROOF — nothing else matters if the live resume breaks
# =========================================================================== #

def test_every_new_lever_defaults_off():
    c = V6Config()
    assert (c.goal_factored, c.goal_multilabel, c.goal_cat_args,
            c.anchor_goal) == (False, False, False, "none")
    s = _build(_small())
    assert s.goal_head_tac_lat is None and s.goal_head_tac_lon is None
    assert s.vocab_tac_lat is None and s.vocab_tac_lon is None
    assert s.anchor_head is None
    assert s.vocab_tac.cat_cards is None and s.goal_head_tac.cat_head is None
    assert s.goal_head_tac.multilabel is False


def test_default_counts_are_the_MEASURED_head_counts():
    """Integer-exact and version-robust: the two literals were measured on the
    unmodified file before the first edit."""
    s = _build(_small())
    assert (_n(s), len(s.state_dict())) == (HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS)


@pytest.mark.slow
def test_default_FULL_config_counts_are_the_MEASURED_head_counts():
    """The config the live run actually uses. Kept separate because building an
    87.9 M-parameter stack is seconds, not milliseconds."""
    f = _build(V6Config())
    assert (_n(f), len(f.state_dict())) == (HEAD_FULL_PARAMS, HEAD_FULL_KEYS)


#: The marker that separates the pre-change architecture from this one. Any
#: revision of ``v6.py`` WITHOUT it is a revision the live v6F S-W checkpoint
#: could have been built from.
_PRE_CHANGE_MARKER = "goal_factored"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_change_module():
    """``tanitad.models.v6`` as it was BEFORE this turn's change, imported
    side-by-side so the two builds can be compared tensor by tensor.

    ⚠️ NOT "HEAD". HEAD is the wrong reference and this is not a style point:
    while this very file was being written HEAD moved 37220d2 -> a558b79 and
    swept the in-progress ``v6.py`` into a commit — after which a HEAD
    comparison is a module compared with itself, i.e. a test that passes by
    construction and proves nothing. Instead we walk ``v6.py``'s own history
    for the NEWEST revision that does not yet carry the new flag. That
    reference is stable no matter how many commits land afterwards, and it is
    also the semantically right one: it IS the architecture the live S-W
    checkpoint was built from.

    Returns None when git cannot answer, in which case the caller skips — a
    skipped test is honest; a self-comparison dressed as a real one is not.
    """
    root = _STACK.parent
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", _V6_REL],
                             cwd=root, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{_V6_REL}"], cwd=root,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if _PRE_CHANGE_MARKER.encode() in r.stdout:
                continue                      # already carries the change
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    # the emission lives in stack/scripts and is imported by bare name
    sp = str(_STACK / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_change.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_change", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_change"] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


def test_all_off_is_byte_identical_to_the_PRE_CHANGE_architecture():
    """⛔ THE ONE THAT PROTECTS THE LIVE RUN. Per tensor, ``torch.equal``.

    Not a digest of a ``torch.save`` file: that container's bytes are not
    canonical (RETRACTION_LOG C72), so a hash over it can differ for two
    identical state_dicts and agree for two different ones.
    """
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")

    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config(**{**_sub_cfgs(), **_SMALL_KW}))
    rng_old = torch.random.get_rng_state()
    torch.manual_seed(0)
    new = V6Stack(_small())
    rng_new = torch.random.get_rng_state()

    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn), (
        f"state_dict KEYS moved against {head._ref}: "
        f"only-old={set(so) - set(sn)}, only-new={set(sn) - set(so)}")
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} MOVED against {head._ref}"
    # ⭐ and the RNG stream itself: a default path that consumed one extra draw
    # would leave the state_dict identical and still desynchronise everything
    # initialised after the model.
    assert torch.equal(rng_old, rng_new), \
        "the default build consumed a different number of random draws"


@pytest.mark.parametrize("flags", [
    dict(goal_factored=True),
    dict(goal_multilabel=True),
    dict(goal_cat_args=True),
    dict(goal_cat_args=True, anchor_goal="snap_lat"),
    dict(goal_cat_args=True, anchor_goal="snap_xy"),
    dict(goal_cat_args=True, anchor_goal="onehot"),
    dict(goal_factored=True, goal_cat_args=True, goal_multilabel=True,
         anchor_goal="snap_lat"),
])
def test_turning_a_flag_on_perturbs_NO_pre_existing_tensor(flags):
    """The capacity-control discipline, applied to every new lever: the ON arm
    is the OFF arm plus tensors, never the OFF arm re-initialised. This is what
    makes a delta attributable — and it only holds because every new module is
    constructed at the END of ``__init__``."""
    off, on = _build(_small()), _build(_small(**flags))
    so, sn = off.state_dict(), on.state_dict()
    assert set(so) <= set(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} moved when {flags} flipped"


def test_default_forward_emits_exactly_the_pre_existing_keys():
    """Byte-identity of the weights is necessary and not sufficient: a new
    output key on the default path would change what every consumer sees."""
    s = _build(_small())
    out = s.forward(**s.synthetic_batch(2))
    assert set(out["g_tac"]) == {"logits", "args", "probs"}
    assert out["g_tac_lat"] is None and out["g_tac_lon"] is None
    assert not any(k.startswith("anchor_") for k in out["plan"])


# =========================================================================== #
# 2. ⭐ THE LAT × LON FACTORING — the a_tac idiom, applied to g_tac
# =========================================================================== #

def test_the_partition_is_total_and_disjoint():
    """LAT + LON reproduce §4's nine exactly, plus one ABSTAIN per axis. A
    partition that dropped or duplicated a token would silently change the
    vocabulary while looking like a refactor."""
    lat = set(TACTICAL_GOAL_TOKENS_LAT) - {"LAT_UNCONSTRAINED"}
    lon = set(TACTICAL_GOAL_TOKENS_LON) - {"LON_UNCONSTRAINED"}
    assert lat | lon == set(TACTICAL_GOAL_TOKENS)
    assert not lat & lon
    assert len(lat) + len(lon) == len(TACTICAL_GOAL_TOKENS) == 9
    # the two the measurement names, on the axes it names them on
    assert goal_token_axis("ANCHOR_GOAL") == "lat"
    assert goal_token_axis("SPEED_BAND") == "lon"
    with pytest.raises(KeyError):
        goal_token_axis("KEEP_CORRIDOR")           # a STRATEGIC token
    # each axis can say "unconstrained" — §2's "Unset = unconstrained"
    assert "LAT_UNCONSTRAINED" in TACTICAL_GOAL_TOKENS_LAT
    assert "LON_UNCONSTRAINED" in TACTICAL_GOAL_TOKENS_LON


def test_it_follows_the_a_tac_IDIOM_rather_than_inventing_a_second_one():
    """``act_head_lat`` / ``act_head_lon`` already factor the tactical ACTION.
    The goal pair must be the same construction — same class, same input, same
    shared-vocabulary rule — or the programme has two idioms for one job."""
    s = _build(_small(goal_factored=True))
    for h in (s.goal_head_tac_lat, s.goal_head_tac_lon):
        assert isinstance(h, GoalHead)
        assert type(h) is type(s.act_head_lat)
        assert h.d_in == s.act_head_lat.d_in == s.cfg.d_tac
    # ...and, like the MIXED goal head it replaces, both are conditioned on the
    # strategic goal handed down (goals flow DOWN, §5). The action heads are
    # not: an action is not conditioned on a goal from above, its own layer's
    # goal is.
    assert s.goal_head_tac_lat.d_cond == s.goal_head_tac.d_cond
    assert s.act_head_lat.d_cond == 0
    # §5 "one vocabulary, two views": id() identity, not a copy that starts equal
    assert s.goal_head_tac_lat.vocab is s.cond_op_lat.vocab
    assert s.goal_head_tac_lon.vocab is s.cond_op_lon.vocab
    assert s.goal_head_tac_lat.vocab is not s.goal_head_tac_lon.vocab


def test_factored_param_delta_is_MEASURED_on_a_full_V6Stack():
    """⚠️ MEASURE, DO NOT ARITHMETIC. ``selector="mlp"`` cost +33,801 measured
    against a design-time +41,089 that was never realised; a delta quoted from
    a shape calculation is an ESTIMATE wearing a measurement's stamp.

    This asserts the delta on the SMALL config (fast) and pins the FULL-config
    number as a literal recorded from a real build.
    """
    off, on = _build(_small()), _build(_small(goal_factored=True))
    delta = _n(on) - _n(off)
    # the shape, stated so a future reader can see WHERE the parameters went
    v = on.vocab_tac_lat
    per_head = (on.cfg.d_tac + v.d_embed) * 256 + 256 + 256 * 256 + 256
    expect = 0
    for voc in (on.vocab_tac_lat, on.vocab_tac_lon):
        expect += voc.n_tokens * voc.d_embed          # embedding table
        expect += voc.n_args * voc.d_embed + voc.d_embed   # arg_proj
        expect += 2 * voc.d_embed                     # LayerNorm
        expect += per_head                            # head trunk
        expect += 256 * voc.n_tokens + voc.n_tokens   # type_head
        expect += 256 * voc.n_args + voc.n_args       # arg_head
    assert delta == expect > 0
    # the conditioners' SEAM is free: d_out == d_embed, so proj is nn.Identity.
    # (`_n(cond_op_lat)` itself is NOT zero — a conditioner holds the shared
    # vocabulary as a submodule, which is the §5 identity, not a second copy;
    # ``parameters()`` de-duplicates so the stack total counts it once.)
    assert isinstance(on.cond_op_lat, GoalConditioner)
    assert _n(on.cond_op_lat.proj) == _n(on.cond_op_lon.proj) == 0
    assert on.cond_op_lat.vocab is on.vocab_tac_lat


def test_factored_delta_on_the_REAL_config_is_the_recorded_literal():
    """The number the writeup quotes, asserted rather than remembered."""
    off = _build(V6Config())
    on = _build(V6Config(goal_factored=True))
    assert _n(on) - _n(off) == FACTORED_DELTA_FULL


def test_the_pair_emits_ANCHOR_GOAL_AND_SPEED_BAND_TOGETHER():
    """⭐ THE POINT OF THE FACTORING. §6.4 says the tactical goal must be
    `ANCHOR_GOAL` (lateral) ∧ `SPEED_BAND` (longitudinal) at once. One 9-way
    softmax cannot represent a pair; two axis-heads emit one each BY
    CONSTRUCTION, and each half is separately scoreable — which the
    unstructured multi-label form cannot give you."""
    s = _build(_small(goal_factored=True))
    out = s.forward(**s.synthetic_batch(2))
    lat, lon = out["g_tac_lat"], out["g_tac_lon"]
    assert lat["probs"].shape[-1] == len(TACTICAL_GOAL_TOKENS_LAT)
    assert lon["probs"].shape[-1] == len(TACTICAL_GOAL_TOKENS_LON)
    # both simplices, independently — the pair is always emitted
    assert torch.allclose(lat["probs"].sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(lon["probs"].sum(-1), torch.ones(2), atol=1e-5)
    i = s.vocab_tac_lat.id_of("ANCHOR_GOAL")
    j = s.vocab_tac_lon.id_of("SPEED_BAND")
    assert lat["probs"][:, i].shape == lon["probs"][:, j].shape == (2,)
    # ⛔ and the un-factored head CANNOT: one simplex over nine, so mass spent
    # on ANCHOR_GOAL is mass taken from SPEED_BAND.
    p = out["g_tac"]["probs"].detach()
    a, b = (s.vocab_tac.id_of("ANCHOR_GOAL"), s.vocab_tac.id_of("SPEED_BAND"))
    assert float((p[:, a] + p[:, b]).max()) <= 1.0 + 1e-5


def test_the_mixed_head_SURVIVES_as_the_control():
    """A comparison with no control is unattributable (C6). The factored arm
    must not delete the head it is being compared against — and deleting it
    would also break the live strict resume."""
    s = _build(_small(goal_factored=True))
    assert s.goal_head_tac is not None and s.vocab_tac is not None
    out = s.forward(**s.synthetic_batch(2))
    assert out["g_tac"]["logits"].shape[-1] == 9
    assert any(k.startswith("goal_head_tac.") for k in s.state_dict())


def test_the_downlink_moves_to_the_pair_and_keeps_its_width():
    """``e_g_tac`` feeds ``predictor_op.intent`` and the emission; a width
    change there would be a silent architecture change, not an arm."""
    mixed, fact = _build(_small()), _build(_small(goal_factored=True))
    a = mixed.forward(**mixed.synthetic_batch(2))["e_g_tac"]
    b = fact.forward(**fact.synthetic_batch(2))["e_g_tac"]
    assert a.shape == b.shape == (2, mixed.cfg.d_goal_embed)
    # the MEAN, not the sum: the factored embedding must live on the same scale
    # as the mixed one, or the arm and its control differ in conditioning
    # magnitude as well as in structure.
    assert float(b.abs().mean()) < 2.0 * float(a.abs().mean())


def test_the_factored_pair_is_grouped_with_the_head_it_replaces():
    """It is a shape change, not a stage change. A head trained in a different
    stage from its control would not be its control."""
    s = _build(_small(goal_factored=True))
    names = [n for n, _ in s.named_parameters()
             if n.startswith(("goal_head_tac_", "vocab_tac_l"))]
    assert names
    assert {s.group_of(n) for n in names} == {"layer_tac"}
    assert s.group_of("goal_head_tac.type_head.weight") == "layer_tac"


# =========================================================================== #
# 3. ⭐ REGRESS-THEN-SNAP as the default; the one-hot CE as the CONTROL
# =========================================================================== #

def test_the_default_mode_is_the_one_the_measurement_prescribes():
    assert AnchorGoalHead.MODES[0] == "snap_lat"
    h = AnchorGoalHead(16, 8, plan_horizon_s=1.0, n_lat_bins=4)
    assert h.mode == "snap_lat"
    assert h.cls is None                       # no one-hot classifier built
    assert h.goal_point is not None            # the metric-aware regression


def test_the_onehot_CONTROL_is_reachable_and_is_the_other_estimator():
    """⛔ E-AG2 measured it +4.7502 [+3.0514, +6.3981] WORSE. It stays buildable
    because a comparison with no control is unattributable — and it must be a
    genuinely DIFFERENT estimator, not the same one with a flag."""
    h = AnchorGoalHead(16, 8, mode="onehot", plan_horizon_s=1.0, n_lat_bins=4)
    assert h.cls is not None and h.goal_point is None
    with pytest.raises(ValueError, match="4.7502|onehot|snap_lat"):
        AnchorGoalHead(16, 8, mode="argmin_cheat", plan_horizon_s=1.0)


def test_quantisation_is_STRAIGHT_THROUGH_so_the_estimator_stays_metric_aware():
    """⭐ The finding this implements: the ESTIMAND (quantisation) is free —
    ``snap`` is NOT separated from the ridge (−0.0002 [−0.1031, +0.0703]) — and
    the ESTIMATOR (a one-hot target) is what costs +4.75 m. Straight-through is
    what keeps the emitted point quantised while the gradient still reaches a
    continuous regression that a metric-aware loss can train."""
    torch.manual_seed(0)
    h = AnchorGoalHead(4, 8, mode="snap_xy", plan_horizon_s=0.4, n_lat_bins=4)
    h.load_anchor_table(torch.randn(8, 4, 2) * 5, horizons=[1, 2, 3, 4], dt=0.1)
    with torch.no_grad():                      # off the zero init
        h.goal_point.weight.normal_(0.0, 1.0)
    g = torch.randn(3, 4, requires_grad=True)
    out = h(g)
    # forward: exactly a table row
    for b in range(3):
        assert torch.equal(out["goal_point"][b], h.anchors[out["anchor_id"][b]])
    # backward: the gradient is the RAW regression's, not the table's
    out["goal_point"].sum().backward()
    g2 = g.detach().clone().requires_grad_(True)
    h.goal_point(g2).sum().backward()
    assert torch.allclose(g.grad, g2.grad)
    assert float(g.grad.abs().sum()) > 0.0


def test_snap_lat_quantises_ONLY_the_lateral_axis():
    """⭐⭐ THE FACTORING, AT THE GEOMETRY LEVEL. §6.4: the goal's variance is
    98.8 % longitudinal while FPS quantisation is isotropic (0.5674 / 0.5599),
    so a joint K-way index spends half its resolution on the axis carrying
    1.2 % of the variance. ``snap_lat`` quantises the lateral coordinate and
    leaves the longitudinal one continuous — the shape the measurement asks
    for."""
    torch.manual_seed(0)
    h = AnchorGoalHead(4, 16, mode="snap_lat", plan_horizon_s=0.4, n_lat_bins=4)
    h.load_anchor_table(torch.randn(16, 4, 2) * 5, horizons=[1, 2, 3, 4], dt=0.1)
    with torch.no_grad():
        h.goal_point.weight.normal_(0.0, 1.0)
    g = torch.randn(6, 4)
    out = h(g)
    raw = out["goal_point_raw"]
    # LONGITUDINAL: bit-identical to the free regression — not quantised at all
    assert torch.equal(out["goal_point"][:, 0], raw[:, 0])
    # LATERAL: on the bin grid, and it really moved
    assert torch.equal(out["goal_point"][:, 1], h.lat_bins[out["lat_bin"]])
    assert not torch.equal(out["goal_point"][:, 1], raw[:, 1])
    assert torch.allclose(out["quant_err_m"], (raw[:, 1] - h.lat_bins[out["lat_bin"]]).abs())
    # and it snaps to the NEAREST bin, not merely to some bin
    for b in range(6):
        d = (raw[b, 1] - h.lat_bins).abs()
        assert int(out["lat_bin"][b]) == int(d.argmin())


def test_the_onehot_control_offers_NO_differentiable_point():
    """Its defining property. Adding a straight-through path would quietly turn
    the control into a third arm — and the thing being controlled for is
    exactly that the target is metric-BLIND."""
    torch.manual_seed(0)
    h = AnchorGoalHead(4, 8, mode="onehot", plan_horizon_s=0.4, n_lat_bins=4)
    h.load_anchor_table(torch.randn(8, 4, 2) * 5, horizons=[1, 2, 3, 4], dt=0.1)
    out = h(torch.randn(3, 4))
    assert set(out) == {"mode", "cls_logits", "anchor_id", "goal_point"}
    assert "goal_point_raw" not in out
    assert not out["goal_point"].requires_grad          # a hard table lookup
    assert out["cls_logits"].requires_grad              # CE is the only loss
    assert out["cls_logits"].shape == (3, 8)


def test_it_refuses_to_emit_without_a_table():
    """A zero table would snap every goal to the origin AND STILL RETURN A
    NUMBER — the failure class this refusal exists to make impossible."""
    h = AnchorGoalHead(4, 8, plan_horizon_s=0.4, n_lat_bins=4)
    assert not bool(h.table_ready)
    with pytest.raises(RuntimeError, match="no anchor table is loaded"):
        h(torch.randn(2, 4))


def test_the_table_SHIPS_WITH_THE_CHECKPOINT():
    """`anchor_id` is meaningless without the exact table, and a rebuilt table
    silently re-labels the whole corpus (§1 field 3). The buffers are therefore
    persistent, and a strict load carries them."""
    s = _build(_small(goal_cat_args=True, anchor_goal="snap_xy"))
    _load_table(s)
    sd = s.state_dict()
    for k in ("anchors", "lat_bins", "table_ready", "table_horizon_s"):
        assert f"anchor_head.{k}" in sd
    t = _build(_small(goal_cat_args=True, anchor_goal="snap_xy"))
    t.load_state_dict(sd)                        # strict
    assert bool(t.anchor_head.table_ready)
    assert torch.equal(t.anchor_head.anchors, s.anchor_head.anchors)
    assert float(t.anchor_head.table_horizon_s) == pytest.approx(
        s.cfg.horizon_s)


# =========================================================================== #
# 4. ⚠️ THE 6 s EXPRESSIBILITY BLOCKER — enforced, not documented
# =========================================================================== #

@pytest.mark.parametrize("horizons,name", [
    ([5, 10, 15, 20], "refc_anchors_full_REBUILD.pt / refc_anchors_small64.pt"),
    (list(range(1, 21)), "anchors_dev256.pt / flagship_v4_anchors_dense.pt"),
])
def test_every_SHIPPED_vocabulary_is_refused_against_the_6s_plan_horizon(
        horizons, name):
    """⛔ MEASURED 2026-08-16 over all five banked tables: every anchor
    vocabulary the programme owns stops at step 20 = **2.0 s**, while
    ``PLAN_STEPS`` is 60 and the v6f selector scores the **6 s** endpoint.
    Scoring one against the other would produce a NUMBER rather than an error.

    ⇒ the factored head is NOT expressible at 6 s today. That is a blocker, and
    this test is where it lives so it cannot be forgotten into a launch.
    """
    h = AnchorGoalHead(128, 256, plan_horizon_s=6.0, n_lat_bins=16)
    with pytest.raises(ValueError):
        h.load_anchor_table(torch.randn(256, len(horizons), 2),
                            horizons=horizons, dt=0.1)
    assert not bool(h.table_ready), f"{name} was accepted at 6 s"


def test_a_2s_table_IS_accepted_by_a_2s_head_so_the_refusal_is_the_horizon():
    """The negative control for the refusal above: it must fire on the HORIZON,
    not on the table. A guard that refuses everything guards nothing."""
    h = AnchorGoalHead(8, 256, plan_horizon_s=2.0, n_lat_bins=8)
    meta = h.load_anchor_table(torch.randn(256, 4, 2), horizons=[5, 10, 15, 20],
                               dt=0.1)
    assert bool(h.table_ready) and meta["horizon_s"] == pytest.approx(2.0)


def test_a_headless_endpoint_table_still_cannot_smuggle_a_horizon():
    h = AnchorGoalHead(8, 4, plan_horizon_s=6.0, n_lat_bins=4)
    with pytest.raises(ValueError, match="takes no"):
        h.load_anchor_table(torch.randn(4, 2), horizons=[20])
    with pytest.raises(ValueError, match="needs its"):
        h.load_anchor_table(torch.randn(4, 3, 2))


def test_the_horizon_refusal_has_ONE_definition_in_the_programme():
    """It reuses ``tanitad.data.anchor_goal.anchor_endpoints`` — the LABEL
    side's own refusal — rather than re-implementing it. Two copies of a
    refusal is one copy that will drift."""
    src = (_STACK / "tanitad" / "models" / "v6.py").read_text(encoding="utf-8")
    assert "from tanitad.data.anchor_goal import anchor_endpoints" in src


# =========================================================================== #
# 5. ⭐ THE CATEGORICAL ARG CHANNEL — 2 of 9 tokens expressible -> 9 of 9
# =========================================================================== #

def _expressible(vocab: GoalVocabulary) -> set[str]:
    """A token is EXPRESSIBLE when every categorical arg it needs has a typed
    slot to live in. Before the channel, only the two purely-continuous tokens
    qualified."""
    ok = set()
    for t in vocab.tokens:
        need = GOAL_CAT_ARG_TOKENS.get(t, ())
        if not need:
            ok.add(t)
        elif vocab.cat_names and any(s in vocab.cat_names for s in need):
            ok.add(t)
    return ok


def test_before_the_channel_only_TWO_of_the_nine_are_expressible():
    """The gap, MEASURED from the code rather than asserted from a doc: both
    ends are continuous (`arg_head` emits 8 floats, `arg_proj` consumes 8), and
    an index is not a physical quantity."""
    s = _build(_small())
    assert s.vocab_tac.cat_cards is None
    assert _expressible(s.vocab_tac) == {"CORRIDOR_OFFSET", "SPEED_BAND"}
    assert s.goal_head_tac.arg_head.out_features == s.vocab_tac.n_args == 8
    assert s.vocab_tac.arg_proj.in_features == 8


def test_with_the_channel_ALL_NINE_are_expressible():
    """⭐ The code gap closed. It says nothing about labels existing — that is a
    separate and still-open gap (PH0 emits none of the nine)."""
    s = _build(_small(goal_cat_args=True))
    assert _expressible(s.vocab_tac) == set(TACTICAL_GOAL_TOKENS)
    assert len(TACTICAL_GOAL_TOKENS) == 9
    assert s.vocab_tac.cat_names == GOAL_CAT_ARG_NAMES
    assert s.vocab_tac.cat_cards == (s.cfg.n_anchors, s.cfg.n_lat_bins,
                                     s.cfg.n_agent_slots, len(STOP_REASONS),
                                     len(LIGHT_STATES))
    out = s.forward(**s.synthetic_batch(2))
    assert out["g_tac"]["cat_logits"].shape == (2, s.vocab_tac.n_cat)


def test_the_five_slot_args_are_ONE_kind_and_no_token_needs_two():
    """`agent_slot_id` / `gap_slot` / `oncoming_slot` / `obstacle_slot` /
    `light_slot_id` are all one id into the window's agent-slot vocabulary, and
    checking token by token shows none needs two at once — which is why ONE
    channel serves all five instead of five channels serving one each."""
    for tok, slots in GOAL_CAT_ARG_TOKENS.items():
        assert tok in TACTICAL_GOAL_TOKENS
        assert len(set(slots)) == len(slots)
        assert sum(s == "agent_slot" for s in slots) <= 1
        for s in slots:
            assert s in GOAL_CAT_ARG_NAMES
    # `anchor_id` (joint) and `lat_bin` (factored) are DIFFERENT vocabularies —
    # conflating them is the type error the channel exists to remove.
    assert GOAL_CAT_ARG_TOKENS["ANCHOR_GOAL"] == ("anchor_id", "lat_bin")
    assert "anchor_id" in GOAL_CAT_ARG_NAMES and "lat_bin" in GOAL_CAT_ARG_NAMES


def test_an_unset_slot_contributes_EXACTLY_zero():
    """§2's "Unset = unconstrained", the same IGNORE discipline the continuous
    slots already follow: a slot no label can fill must not be regressed
    against a fabricated 0 — and must not leak an embedding either."""
    torch.manual_seed(0)
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    v.attach_cat_channel((4, 3, 2, 4, 4))
    ids = torch.tensor([v.id_of("SPEED_BAND")] * 2)      # needs NO cat slot
    cat = torch.rand(2, v.n_cat)
    mask = v.cat_mask_from_tokens(ids)
    assert float(mask.abs().sum()) == 0.0
    a = v.encode(ids, cat=cat, cat_mask=mask)
    b = v.encode(ids)
    assert torch.allclose(a, b, atol=1e-6)
    # ...and a token that DOES use a slot is not zeroed
    ids2 = torch.tensor([v.id_of("STOP_POINT")] * 2)
    assert float(v.cat_mask_from_tokens(ids2).sum()) == 2.0     # 'reason' only
    assert not torch.allclose(v.encode(ids2, cat=cat,
                                       cat_mask=v.cat_mask_from_tokens(ids2)),
                              v.encode(ids2), atol=1e-6)


def test_the_soft_mask_coincides_with_the_hard_one_at_a_one_hot():
    """The differentiable generalisation must not be a SECOND convention."""
    torch.manual_seed(0)
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    v.attach_cat_channel((4, 3, 2, 4, 4))
    ids = torch.tensor([1, 5, 0])
    probs = torch.nn.functional.one_hot(ids, v.n_tokens).float()
    assert torch.equal(v.cat_mask_from_tokens(ids),
                       v.cat_mask_from_tokens(probs))


def test_multilabel_gates_cannot_amplify_a_slot_mask():
    """Two emitted tokens that both use `agent_slot` would give that slot a
    mask of 2 and silently scale its embedding. A mask says whether a slot is
    set; it is not a weight."""
    torch.manual_seed(0)
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    v.attach_cat_channel((4, 3, 2, 4, 4))
    gates = torch.zeros(1, v.n_tokens)
    for t in ("GAP_TARGET", "YIELD_AT", "WAIT_FOR_ONCOMING"):
        gates[0, v.id_of(t)] = 1.0                # all three use agent_slot
    assert float(v.cat_mask_from_tokens(gates).max()) == 1.0


def test_each_typed_slot_is_its_OWN_categorical_variable():
    """One softmax over the concatenation would make picking an `anchor_id`
    compete with picking a `reason` — a different, and wrong, model."""
    torch.manual_seed(0)
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    v.attach_cat_channel((4, 3, 2, 4, 4))
    h = GoalHead(v, 8)
    h.attach_cat_head()
    with torch.no_grad():
        h.cat_head.weight.normal_(0.0, 1.0)
        h.cat_head.bias.normal_(0.0, 1.0)
    cp = h(torch.randn(3, 8))["cat_probs"]
    assert cp.shape == (3, v.n_cat)
    for name in v.cat_names:
        block = cp[:, v.cat_slice(name)]
        assert torch.allclose(block.sum(-1), torch.ones(3), atol=1e-5)
    assert not torch.allclose(cp.sum(-1), torch.ones(3), atol=1e-3)


def test_an_undeclared_categorical_path_is_REFUSED():
    """Same discipline as ``GoalHead``'s refusal of an unexpected ``cond``: an
    undeclared conditioning path is what the disjointness audit (X1) forbids."""
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    with pytest.raises(ValueError, match="NO categorical arg channel"):
        v.encode(torch.tensor([0]), cat=torch.rand(1, 5))
    h = GoalHead(v, 8)
    with pytest.raises(RuntimeError, match="attach the vocabulary"):
        h.attach_cat_head()
    v.attach_cat_channel((4, 3, 2, 4, 4))
    with pytest.raises(RuntimeError, match="already"):
        v.attach_cat_channel((4, 3, 2, 4, 4))


def test_cat_usage_is_derived_and_NOT_shipped_in_the_checkpoint():
    s = _build(_small(goal_cat_args=True))
    assert "vocab_tac.cat_usage" not in s.state_dict()
    assert s.vocab_tac.cat_usage.shape == (9, len(GOAL_CAT_ARG_NAMES))


# =========================================================================== #
# 6. ⭐ MULTI-LABEL EMISSION — the pair a single 9-way softmax cannot represent
# =========================================================================== #

def test_multilabel_holds_no_parameters_and_keeps_probs():
    """It is the SAME ``type_head`` logits read through a sigmoid. A second
    output layer would make it a capacity change wearing an expressivity
    change's name."""
    off, on = _build(_small()), _build(_small(goal_multilabel=True))
    assert _n(on) == _n(off)
    assert set(on.state_dict()) == set(off.state_dict())
    out = on.forward(**on.synthetic_batch(2))
    assert "gates" in out["g_tac"] and "probs" in out["g_tac"]
    assert torch.allclose(out["g_tac"]["gates"],
                          out["g_tac"]["logits"].sigmoid())


def test_multilabel_can_express_a_PAIR_the_softmax_cannot():
    """⭐ §2.3's second representational limit. Under a simplex the two masses
    are in competition and cannot both approach 1; independent gates can."""
    torch.manual_seed(0)
    v = GoalVocabulary(TACTICAL_GOAL_TOKENS, 16)
    h = GoalHead(v, 8)
    h.enable_multilabel(True)
    i, j = v.id_of("ANCHOR_GOAL"), v.id_of("SPEED_BAND")
    with torch.no_grad():
        h.type_head.weight.zero_()
        h.type_head.bias.fill_(-6.0)
        h.type_head.bias[i] = h.type_head.bias[j] = 6.0
    out = h(torch.randn(2, 8))
    assert float(out["gates"][:, [i, j]].min()) > 0.99      # BOTH on
    assert float(out["probs"][:, [i, j]].max()) < 0.51      # the simplex splits
    # and the conditioner consumes a SET through the same matmul path
    e = GoalConditioner(v)(out["gates"])
    assert e.shape == (2, v.d_embed) and torch.isfinite(e).all()


# =========================================================================== #
# 7. ⛔ X3 ISOLATION AND ADMISSIBILITY — unchanged for every new arm
# =========================================================================== #

_ARMS = [
    dict(goal_factored=True),
    dict(goal_multilabel=True),
    dict(goal_cat_args=True),
    dict(goal_factored=True, goal_cat_args=True),
    dict(goal_cat_args=True, anchor_goal="snap_lat"),
    dict(goal_cat_args=True, anchor_goal="snap_xy", selector="goal"),
    dict(goal_cat_args=True, anchor_goal="onehot", selector="goal"),
    dict(goal_factored=True, goal_cat_args=True, goal_multilabel=True,
         anchor_goal="snap_lat", selector="goal"),
]


@pytest.mark.parametrize("flags", _ARMS)
def test_X3_isolation_holds_for_every_new_arm(flags):
    """``{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}``
    — measured on a real autograd graph, not asserted in a comment."""
    s = _build(_small(**flags))
    if s.anchor_head is not None:
        _load_table(s)
    iso = s.assert_isolation(batch_size=2, strict=True)
    assert iso["pass"], iso
    assert iso["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}
    assert iso["violations"] == {}
    # ...and the probe actually looked at something — a probe over an empty
    # parameter set reports zero violations and has established nothing.
    assert all(v > 0 for v in iso["n_probed"].values()), iso["n_probed"]


@pytest.mark.parametrize("flags", _ARMS)
def test_the_declared_planner_surface_stays_TOTAL(flags):
    """Every planner-group parameter must be reachable from the DECLARED
    surface. A head added without appending to the declaration escapes the
    isolation probe — the `intent_proj` defect, where a path present in the
    diagram was absent from the optimisation."""
    s = _build(_small(**flags))
    if s.anchor_head is not None:
        _load_table(s)
    with torch.no_grad():                       # off the CV warm start
        s.emission.net[-1].weight.normal_(0.0, 0.1)
        s.emission.net[-1].bias.normal_(0.0, 0.1)
        for blk in s.predictor_op.blocks:
            blk.film.to_scale_shift.weight.normal_(0.0, 0.1)
        if getattr(s.cand_score, "fc2", None) is not None:
            s.cand_score.fc2.weight.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    planner = list(s.group_parameters("planner"))
    live = V6Stack._live_edges(V6Stack._probe_scalar(out["planner_side"]),
                               planner)
    assert not {n for n, _ in planner} - set(live)


def test_the_superseded_free_decode_stays_reachable_and_is_the_pair_control():
    """When the anchor head supplies ĝ, the selector's own free decode is no
    longer used for scoring. It is still computed and DECLARED — otherwise it
    is a planner parameter no declared output reaches — and it hands E-AG2's
    paired free-vs-structured comparison on the same window for free."""
    s = _build(_small(goal_cat_args=True, anchor_goal="snap_lat",
                      selector="goal"))
    _load_table(s)
    plan = s.forward(**s.synthetic_batch(2))["plan"]
    assert "sel_goal_point_free" in plan
    assert torch.equal(plan["sel_goal_point"], plan["anchor_goal_point"])
    # and with no anchor head the incumbent path is bit-for-bit unchanged
    t = _build(_small(selector="goal"))
    p2 = t.forward(**t.synthetic_batch(2))["plan"]
    assert "sel_goal_point_free" not in p2


def test_the_mlp_capacity_control_is_NOT_handed_a_ready_made_goal_point():
    """Handing it one would make it an INFORMATION control and its result would
    stop speaking to capacity (§5.3). It keeps reading ``e_g_tac``."""
    s = _build(_small(goal_cat_args=True, anchor_goal="snap_lat",
                      selector="mlp"))
    _load_table(s)
    plan = s.forward(**s.synthetic_batch(2))["plan"]
    assert plan["sel_mechanism"] == "mlp"
    assert "sel_goal_point" not in plan and "anchor_goal_point" in plan


def test_no_new_signature_opens_an_ego_or_situation_port():
    """⛔ BINDING (PI 2026-08-03), audited on MY OWN design: vision-only at
    inference, and the goal path stays information-disjoint from the situation
    classifier. The check is structural — no port exists to carry one."""
    import inspect
    for fn in (AnchorGoalHead.forward, GoalHead.forward,
               GoalConditioner.forward, GoalVocabulary.encode,
               V6Stack._encode_goal):
        names = set(inspect.signature(fn).parameters)
        assert not (names & {"situation", "situations", "sit", "ego", "v0",
                             "speed", "kwargs"}), (fn, names)
    src = (_STACK / "tanitad" / "models" / "v6.py").read_text(encoding="utf-8")
    assert "tanitad.data.situations" not in src
    assert "import situations" not in src


def test_nothing_new_depends_on_v0_being_admissible():
    """🟥 ``v0``'s admissibility is an OPEN PI DECISION (V6F_PLANNER_DESIGN §1.4
    vs e_wc2_sigma_star.py:188) worth a MEASURED 2.85x on this very quantity.
    The anchor head must not pre-empt it: its only input is ``e_g_tac``."""
    import inspect
    sig = inspect.signature(AnchorGoalHead.forward)
    assert list(sig.parameters) == ["self", "g_embed"]
    s = _build(_small(goal_cat_args=True, anchor_goal="snap_lat"))
    _load_table(s)
    b = s.synthetic_batch(2)
    a = s.forward(**b)["plan"]["anchor_goal_point"]
    b2 = dict(b, v0=b["v0"] * 3.0 + 7.0)
    assert torch.equal(a, s.forward(**b2)["plan"]["anchor_goal_point"])


# =========================================================================== #
# 8. Config refusals
# =========================================================================== #

def test_anchor_goal_requires_the_categorical_channel():
    """An emitted id that reaches nothing downstream is a head wearing an
    emission's name — and writing it into a metres slot is the very type error
    the channel removes."""
    with pytest.raises(ValueError, match="goal_cat_args"):
        V6Config(anchor_goal="snap_lat")
    assert V6Config(anchor_goal="snap_lat", goal_cat_args=True)


def test_an_unknown_anchor_mode_is_refused_and_the_refusal_names_the_control():
    with pytest.raises(ValueError, match="onehot"):
        V6Config(anchor_goal="learned", goal_cat_args=True)
    for bad in ("n_anchors", "n_lat_bins", "n_agent_slots"):
        with pytest.raises(ValueError, match=bad):
            V6Config(**{bad: 1})
