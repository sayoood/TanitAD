"""The METRIC-AWARE ``ANCHOR_GOAL`` objective — and the two identity guards
that have to hold before any of it is allowed to exist.

⛔ THE CONSTRAINT THAT OUTRANKS EVERYTHING HERE. v6F S-W is training on Thor
from a checkpoint of this exact architecture, and this turn edits the file that
BUILDS that architecture from the CLI (``build_stack_from_args``) and the file
that computes its LOSS. So there are two identity proofs, not one:

  1. ARCHITECTURE — the default CLI build's ``state_dict`` is byte-identical,
     per tensor with ``torch.equal``, against the newest revision of ``v6.py``
     that PREDATES the goal-head change (found BY CONTENT, never ``HEAD``);
  2. ⭐ LOSS — ``v6_loss_step`` at default weights returns a bit-identical
     total AND bit-identical per-term tensors against the newest revision of
     ``train_v6_staged.py`` that predates ``w_anchor``, on the same stack, the
     same batch and the same seed.

⛔ WHY "BY CONTENT, NEVER ``HEAD``" — RETRACTION_LOG C75, logged hours ago. A
``HEAD``-anchored byte-identity test SKIPPED itself with *"matches HEAD
byte-for-byte"* after a sibling agent's whole-index commit swept the file under
test into ``HEAD``. A guard whose reference is a MOVING POINTER degenerates into
a self-comparison: it runs, reports green, and has measured nothing.

⛔ AND NEVER BY HASHING ``torch.save`` — C72: that container's bytes are not
canonical, so a digest can differ for two identical ``state_dict``s and agree
for two different ones.

⚠️ EVERY GUARD HERE IS SHOWN TO FAIL. A negative control the same night proved a
*discarded* ``nn.Linear`` — one that registers nothing and only consumes an RNG
draw — passes param-count AND flag-flip checks and is caught ONLY by the
per-tensor comparison. Both identity guards therefore carry that control.

WHAT THE OBJECTIVE IS, AND WHY (INHERITED, ANCHOR_GOAL_SUPERVISION.md
2026-08-16, 881 windows / 40 episodes, LOEO, paired episode-cluster bootstrap):
``snap`` — the same ridge rounded to the nearest anchor — is NOT separated from
the free ridge (−0.0002 [−0.1031, +0.0703]), while a K-way one-hot classifier
costs +4.7502 [+3.0514, +6.3981] WORSE, separated at every K and replicated on
REF-C-base. ⇒ QUANTISATION IS FREE; THE ONE-HOT TARGET IS WHAT COSTS. So the
default objective is metric-aware and the CE is reachable only as the named,
acknowledged CONTROL.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.v6 import (  # noqa: E402
    AnchorGoalHead, MLPCandidateScorer, V6Config, V6Stack)
from train_v6_staged import (  # noqa: E402
    ANCHOR_AXIS_W_DEFAULT, ANCHOR_OBJ_MODES, ANCHOR_OBJECTIVES,
    V6LossWeights, _read_anchor_table, anchor_goal_loss,
    build_parser, build_stack_from_args, preflight, synthetic_train_batch,
    v6_loss_step)

# ---------------------------------------------------------------------------
# INHERITED, MEASURED 2026-08-16 (ANCHOR_GOAL_SUPERVISION.md §6.4, 2 s, 881
# windows / 40 episodes). Quoted here because the AXIS-WEIGHT decision is
# derived from them arithmetically below rather than asserted.
SIGMA_LONG_CORPUS, SIGMA_LAT_CORPUS = 19.0578, 2.0723      # the echo null
SIGMA_LONG_RIDGE, SIGMA_LAT_RIDGE = 6.6132, 1.0667         # the free ridge
SIGMA_LONG_CLF, SIGMA_LAT_CLF = 13.3502, 1.3310            # the K-way clf
SIGMA_LONG_FLOOR, SIGMA_LAT_FLOOR = 0.8954, 0.6802         # the quantisation


# =========================================================================== #
# fixtures
# =========================================================================== #
def _sub_cfgs() -> dict:
    return dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  horizons=(1,), action_dim=3, residual=True))


#: the config keys the PRE-CHANGE ``V6Config`` knows, so the two builds differ
#: by the new code and nothing else.
#: ⚠️ ONLY FIELDS THE PARSER CAN SET. The §4b horizon spec, ``aux_hidden``,
#: ``d_plan_feat`` and ``emission_hidden`` have NO CLI flags, so overriding them
#: here would diverge the two builds for a reason that has nothing to do with
#: this turn — which is how a byte-identity test starts failing for the wrong
#: cause and gets "fixed" by loosening it.
_SMALL_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, sigreg_slices=8, d_goal_embed=16, n_candidates=4)

#: the same geometry, spelled as a CLI command — so ``build_stack_from_args``
#: (the function this turn actually edited) is what gets compared, not a
#: hand-built V6Config that would route around the edit.
_SMALL_ARGV = [
    "--stage", "S-T", "--out", "unused",
    "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
    "--patch", "16", "--enc-dim", "32", "--enc-depth", "1", "--enc-heads", "2",
    "--readout-grid", "2", "--readout-dim", "8",
    "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
    "--window", "2", "--horizons", "1",
    "--d-tac", "32", "--d-str", "16", "--d-goal-embed", "16",
    "--adapter-hidden", "32", "--f-hidden-tac", "32", "--f-hidden-str", "32",
    "--f-blocks", "1", "--sigreg-slices", "8", "--n-candidates", "4",
]


def _small(**kw) -> V6Config:
    return V6Config(**{**_sub_cfgs(), **_SMALL_KW, **kw})


def _args(*extra) -> object:
    return build_parser().parse_args(_SMALL_ARGV + list(extra))


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def _load_table(stack: V6Stack, seed: int = 3) -> dict:
    """A synthetic vocabulary AT THE PLAN HORIZON, so the head accepts it. The
    shipped ones are all 2 s and are correctly refused — see §5."""
    torch.manual_seed(seed)
    cfg = stack.cfg
    steps = list(range(1, cfg.plan_steps + 1))
    return stack.anchor_head.load_anchor_table(
        torch.randn(cfg.n_anchors, len(steps), 2) * 3.0, horizons=steps,
        dt=cfg.dt)


def _anchor_stack(mode: str = "snap_lat", *, k: int = 8, seed: int = 0
                  ) -> V6Stack:
    s = _build(_small(anchor_goal=mode, goal_cat_args=True, n_anchors=k,
                      n_lat_bins=4), seed=seed)
    _load_table(s)
    return s


def _side_by_side(rel: str, marker: str, modname: str):
    """Import the newest revision of ``rel`` that does NOT contain ``marker``.

    ⛔ THE REFERENCE IS RESOLVED BY CONTENT, NOT BY ``HEAD`` (C75). ``HEAD``
    moves — a sibling's whole-index commit swept an in-progress file into it
    hours ago — and a HEAD-relative identity test then compares a module with
    itself. Walking the FILE's own history for the last revision lacking the
    change marker is stable no matter how many commits land afterwards, and it
    is the semantically correct reference: it IS the code the live checkpoint /
    the live loss came from.

    Returns ``None`` when git cannot answer; the caller then SKIPS. A skipped
    test is honest; a self-comparison dressed as a real one is not.
    """
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", rel],
                             cwd=_ROOT, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=_ROOT,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if marker.encode() in r.stdout:
                continue                        # already carries the change
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / f"{modname}.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location(modname, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


# =========================================================================== #
# 1. ⛔ BYTE-IDENTITY — nothing else matters if the live resume breaks
# =========================================================================== #

def test_every_new_lever_defaults_off_on_the_COMMAND_LINE():
    """The V6Config levers had no CLI path at all until this turn. Adding one
    is only safe if its default is the incumbent — checked on the parser, which
    is what an operator actually types."""
    a = _args()
    assert (a.anchor_goal, a.w_anchor, a.anchor_table) == ("none", 0.0, None)
    assert (a.goal_factored, a.goal_multilabel, a.goal_cat_args) == \
        (False, False, False)
    assert a.anchor_objective == "metric"
    assert list(a.anchor_axis_w) == list(ANCHOR_AXIS_W_DEFAULT) == [1.0, 1.0]
    assert V6LossWeights().w_anchor == 0.0


def test_default_CLI_build_is_byte_identical_to_the_PRE_CHANGE_architecture():
    """⛔ THE ONE THAT PROTECTS THE LIVE RUN. Per tensor, ``torch.equal``.

    It goes through ``build_stack_from_args`` — the function this turn edited —
    rather than through a hand-built ``V6Config``, because a test that routes
    around the edited code proves nothing about it.
    """
    old = _side_by_side("stack/tanitad/models/v6.py", "goal_factored",
                        "v6_pre_change_anchorloss")
    if old is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")

    torch.manual_seed(0)
    ref = old.V6Stack(old.V6Config(**{**_sub_cfgs(), **_SMALL_KW}))
    rng_ref = torch.random.get_rng_state()
    torch.manual_seed(0)
    new = build_stack_from_args(_args())
    rng_new = torch.random.get_rng_state()

    so, sn = ref.state_dict(), new.state_dict()
    assert list(so) == list(sn), (
        f"state_dict KEYS moved against {old._ref}: "
        f"only-old={set(so) - set(sn)}, only-new={set(sn) - set(so)}")
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} MOVED against {old._ref}"
    assert sum(p.numel() for p in ref.parameters()) == \
        sum(p.numel() for p in new.parameters())
    # ⭐ and the RNG STREAM: a default path that consumed one extra draw would
    # leave the state_dict identical here and desynchronise everything
    # initialised after the model.
    assert torch.equal(rng_ref, rng_new), \
        "the default CLI build consumed a different number of random draws"


def test_the_byte_identity_guard_CAN_FAIL_on_a_discarded_draw():
    """⭐ THE NEGATIVE CONTROL, in the exact shape that was measured to slip
    through the cheap checks: a module that is CONSTRUCTED AND DISCARDED
    registers no key and changes no count — it only consumes one RNG draw — and
    is caught ONLY by the per-tensor comparison.

    A guard that has never failed is a guard whose sensitivity is unknown.
    """
    clean = _build(_small())
    torch.manual_seed(0)
    torch.nn.Linear(3, 3)                        # constructed, DISCARDED
    poisoned = V6Stack(copy.deepcopy(_small()))

    a, b = clean.state_dict(), poisoned.state_dict()
    # the CHEAP checks both pass...
    assert list(a) == list(b)
    assert sum(p.numel() for p in clean.parameters()) == \
        sum(p.numel() for p in poisoned.parameters())
    # ...and the per-tensor comparison FIRES.
    assert any(not torch.equal(a[k], b[k]) for k in a), \
        "the per-tensor guard did not fire on a shifted RNG stream"


def test_default_loss_is_BIT_IDENTICAL_to_the_pre_w_anchor_trainer():
    """⭐ THE SECOND IDENTITY PROOF, and the one this turn actually risks.

    Same stack, same batch, same seed, both trainers — the total AND every
    per-term tensor must be bit-identical. The reference is resolved from
    ``train_v6_staged.py``'s OWN history by content (the newest revision
    without ``w_anchor``), for the C75 reason.

    ⚠️ The old module imports the CURRENT ``tanitad.models.v6``, which is
    deliberate: this holds the MODEL fixed and varies only the LOSS code, so a
    difference can only be mine.
    """
    old = _side_by_side("stack/scripts/train_v6_staged.py", "w_anchor",
                        "train_v6_staged_pre_anchor")
    if old is None:
        pytest.skip("git could not produce a pre-w_anchor trainer revision")

    stack = _build(_small())
    # ⚠️ MEASURED while writing this: in TRAIN mode the S-W total is not
    # reproducible even between two calls of the SAME code (3.9301 vs 3.9227) —
    # a stochastic op reads the GLOBAL RNG, which the passed ``generator`` does
    # not cover. A guard run in that regime would fire on noise and be
    # "fixed" by loosening it into uselessness. ⇒ eval() AND a re-seed per call.
    stack.eval()
    for stage in ("S-W", "S-T", "S-S", "S-J"):
        batch = synthetic_train_batch(stack, batch=2, k=4, seed=7)
        batch["gt_wp"] = torch.zeros(2, 2, 2)
        kw = dict(stage=stage, o1_k=2, o5_k=2)
        torch.manual_seed(3)
        lo = old.v6_loss_step(stack, batch, weights=old.V6LossWeights(),
                              generator=torch.Generator().manual_seed(11),
                              **kw)
        torch.manual_seed(3)
        ln = v6_loss_step(stack, batch, weights=V6LossWeights(),
                          generator=torch.Generator().manual_seed(11), **kw)
        assert torch.equal(lo["loss"], ln["loss"]), \
            f"{stage}: total loss MOVED against {old._ref}"
        assert lo["log"]["terms"] == ln["log"]["terms"], \
            f"{stage}: the TERM SET moved against {old._ref}"
        for t in lo["log"]["terms"]:
            assert torch.equal(lo[t], ln[t]), \
                f"{stage}: term {t!r} MOVED against {old._ref}"
        # the log is what an operator reads; a silently changed key is a run
        # row that stops being comparable with the ones before it.
        assert set(lo["log"]) == set(ln["log"])


def test_the_loss_identity_guard_CAN_FAIL():
    """The negative control for the guard above: turn the new term ON and the
    comparison must fire. (Run against the CURRENT trainer twice, so it tests
    the COMPARISON, not the reference resolution.)"""
    stack = _anchor_stack("snap_lat")
    batch = synthetic_train_batch(stack, batch=2, k=4, seed=7)
    batch["gt_wp"] = torch.zeros(2, 2, 2)
    kw = dict(stage="S-T", o1_k=2, o5_k=2)
    off = v6_loss_step(stack, batch, weights=V6LossWeights(), **kw)
    on = v6_loss_step(stack, batch,
                      weights=V6LossWeights(w_anchor=1.0), **kw)
    assert "anchor" not in off["log"]["terms"]
    assert "anchor" in on["log"]["terms"]
    assert not torch.equal(off["loss"], on["loss"])


def test_w_anchor_is_zero_in_the_stages_whose_planner_is_frozen():
    """S-W's planner is ABSENT and S-S trains ``layer_str`` ONLY. A term in
    force where its module cannot move is a launch line that lies about what
    trained — the same reason ``w_select`` is zeroed in both."""
    w = V6LossWeights(w_anchor=1.0)
    assert w.for_stage("S-W").w_anchor == 0.0
    assert w.for_stage("S-S").w_anchor == 0.0
    assert w.for_stage("S-T").w_anchor == 1.0
    assert w.for_stage("S-J").w_anchor == 1.0


def test_the_anchor_term_is_SKIPPED_not_multiplied_by_zero():
    """A skipped term costs no compute and — the part that bites — cannot
    appear in the log looking like it trained something."""
    stack = _anchor_stack("snap_lat")
    batch = synthetic_train_batch(stack, batch=2, k=4, seed=1)
    batch["gt_wp"] = torch.zeros(2, 2, 2)
    L = v6_loss_step(stack, batch, stage="S-T", o1_k=2, o5_k=2,
                     weights=V6LossWeights(w_anchor=0.0))
    assert "anchor" not in L["log"]["terms"]
    assert not any(k.startswith("anchor_") for k in L["log"])


# =========================================================================== #
# 2. ⭐ THE METRIC-AWARE OBJECTIVE — and why CE is not the default
# =========================================================================== #

def test_the_default_objective_is_metric_aware_and_never_ce():
    assert build_parser().get_default("anchor_objective") == "metric"
    assert inspect.signature(anchor_goal_loss).parameters[
        "objective"].default == "metric"
    assert inspect.signature(v6_loss_step).parameters[
        "anchor_objective"].default == "metric"
    assert "CONTROL ONLY" in ANCHOR_OBJECTIVES["ce"]


def test_metric_gradient_reaches_the_CONTINUOUS_regression_through_the_snap():
    """⭐ THE PROPERTY THAT MAKES REGRESS-THEN-SNAP A TRAINABLE OBJECT.

    The emitted point is QUANTISED and the gradient still reaches the
    CONTINUOUS regression — including on the axis that was quantised. That is
    what keeps the objective metric-aware instead of collapsing into the CE the
    measurement refuses.

    Checked ANALYTICALLY, not by comparison with another autograd call: for
    ``raw = W·g + b`` the straight-through estimator makes
    ``dL/db == mean_B (emitted − target) / ||emitted − target||`` on BOTH
    coordinates. A hard (non-straight-through) snap would zero the LATERAL
    component — the negative control below shows exactly that.
    """
    s = _anchor_stack("snap_lat")
    head = s.anchor_head
    torch.manual_seed(5)
    g = torch.randn(4, s.cfg.d_goal_embed)
    with torch.no_grad():                 # off the zero init, or there is no ĝ
        head.goal_point.weight.normal_(0.0, 0.3)
        head.goal_point.bias.normal_(0.0, 0.3)
    tgt = torch.randn(4, 2) * 2.0

    out = head(g)
    loss, log = anchor_goal_loss(out, tgt, head.anchors)
    gw, gb = torch.autograd.grad(loss, [head.goal_point.weight,
                                        head.goal_point.bias])
    assert float(gw.abs().max()) > 0 and float(gb.abs().max()) > 0, \
        "the metric objective reached NO parameter — the snap ate the gradient"

    d = out["goal_point"].detach() - tgt
    expect = (d / d.norm(dim=-1, keepdim=True)).mean(dim=0)
    assert torch.allclose(gb, expect, atol=1e-6)
    # ⭐ the LATERAL component — the quantised axis — is NOT zero. That single
    # number is the difference between a trained regress-then-snap and a
    # post-hoc rounding.
    assert float(gb[1].abs()) > 1e-4
    assert log["anchor_objective"] == "metric"


def test_a_HARD_snap_would_kill_the_lateral_gradient_the_straight_through_keeps():
    """⭐ THE NEGATIVE CONTROL for the test above. Detach the snapped lateral
    coordinate entirely — i.e. round it post hoc instead of straight through —
    and the lateral gradient goes to EXACTLY zero, which is the failure the
    straight-through estimator exists to prevent. A property nobody has seen
    fail is a property whose test might be measuring nothing."""
    s = _anchor_stack("snap_lat")
    head = s.anchor_head
    torch.manual_seed(5)
    g = torch.randn(4, s.cfg.d_goal_embed)
    with torch.no_grad():
        head.goal_point.weight.normal_(0.0, 0.3)
        head.goal_point.bias.normal_(0.0, 0.3)
    tgt = torch.randn(4, 2) * 2.0

    raw = head.goal_point(g)
    j = (raw[:, 1:2] - head.lat_bins[None]).abs().argmin(dim=-1)
    hard = torch.stack([raw[:, 0], head.lat_bins[j]], dim=-1)   # NO passthrough
    loss, _ = anchor_goal_loss({"mode": "snap_lat", "goal_point": hard},
                               tgt, head.anchors)
    gb, = torch.autograd.grad(loss, [head.goal_point.bias])
    assert float(gb[1].abs()) == 0.0, \
        "the hard-snap control did not lose the lateral gradient"
    assert float(gb[0].abs()) > 0.0     # the un-quantised axis still trains


def test_metric_on_the_onehot_control_is_REFUSED_not_silently_dead():
    """⛔ The onehot head's emitted point is a HARD table lookup with NO
    gradient. A metric loss on it would fall to a constant and train NOTHING
    while the log showed a metre-scale term — a number instead of an error."""
    s = _anchor_stack("onehot")
    out = s.anchor_head(torch.randn(3, s.cfg.d_goal_embed))
    # first: MEASURE the claim rather than assert it.
    assert not out["goal_point"].requires_grad
    with pytest.raises(ValueError, match="needs an anchor_goal mode"):
        anchor_goal_loss(out, torch.randn(3, 2), s.anchor_head.anchors)


@pytest.mark.parametrize("obj", ["softanchor", "ce"])
def test_the_anchor_distribution_objectives_refuse_a_snap_mode(obj):
    """No ``cls_logits`` means there is nothing to put a distribution on."""
    s = _anchor_stack("snap_xy")
    out = s.anchor_head(torch.randn(3, s.cfg.d_goal_embed))
    with pytest.raises(ValueError, match="needs an anchor_goal mode"):
        anchor_goal_loss(out, torch.randn(3, 2), s.anchor_head.anchors,
                         objective=obj)


def test_CE_IS_METRIC_BLIND_AND_THE_OTHERS_ARE_NOT():
    """⭐⭐ THE DISCRIMINATING TEST — the property being controlled for, made
    measurable rather than asserted.

    Two predictions that are WRONG IN THE SAME WAY categorically (identical
    logits over a permuted table) but differ by ~40 m versus ~0.1 m in the
    METRE the car is actually scored in. CE gives the two the SAME loss. The
    metric-aware objectives do not.

    This is E-AG2's +4.7502 [+3.0514, +6.3981] explained in one assertion.
    """
    tgt = torch.zeros(1, 2)
    near = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    far = torch.tensor([[0.0, 0.0], [40.0, 0.0], [80.0, 0.0]])
    # all mass on index 1 — the WRONG anchor in both tables, by construction
    logits = torch.tensor([[0.0, 9.0, 0.0]])

    ce_n, _ = anchor_goal_loss({"mode": "onehot", "cls_logits": logits},
                               tgt, near, objective="ce")
    ce_f, _ = anchor_goal_loss({"mode": "onehot", "cls_logits": logits},
                               tgt, far, objective="ce")
    assert torch.allclose(ce_n, ce_f), \
        "CE is supposed to be metric-BLIND — that is the control's defect"

    sa_n, ln = anchor_goal_loss({"mode": "onehot", "cls_logits": logits},
                                tgt, near, objective="softanchor")
    sa_f, lf = anchor_goal_loss({"mode": "onehot", "cls_logits": logits},
                                tgt, far, objective="softanchor")
    assert float(sa_f) > float(sa_n) * 50, \
        "softanchor did NOT see the 400x metre difference CE cannot see"
    # and the metre-valued diagnostics separate them even for the CE arm, so a
    # control run is still comparable on the quantity that matters.
    assert lf["anchor_expected_err_m"] > ln["anchor_expected_err_m"] * 50


def test_softanchor_optimum_is_the_NEAREST_anchor_not_a_soft_blur():
    """E-OBJ-1 measured metric-awareness HELPING (−0.0974 / −0.1670 m,
    separated) and target-SOFTENING HURTING (+0.0909 m at every tau). The
    expected-distance form keeps the first and refuses the second: its optimum
    is still all mass on the nearest anchor."""
    tgt = torch.zeros(1, 2)
    anchors = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    sharp_right = torch.tensor([[20.0, 0.0, 0.0]])
    blurred = torch.tensor([[0.0, 0.0, 0.0]])
    sharp_wrong = torch.tensor([[0.0, 0.0, 20.0]])
    ls, _ = anchor_goal_loss({"mode": "onehot", "cls_logits": sharp_right},
                             tgt, anchors, objective="softanchor")
    lb, _ = anchor_goal_loss({"mode": "onehot", "cls_logits": blurred},
                             tgt, anchors, objective="softanchor")
    lw, _ = anchor_goal_loss({"mode": "onehot", "cls_logits": sharp_wrong},
                             tgt, anchors, objective="softanchor")
    assert float(ls) < float(lb) < float(lw)
    assert float(ls) == pytest.approx(0.0, abs=1e-6)


def test_the_objective_mode_table_is_total_and_the_refusals_quote_it():
    """Every objective names the modes it can train, and every mode is a real
    :class:`AnchorGoalHead` mode. A table that drifted from ``MODES`` would let
    a launch through that the head then refuses at the first batch."""
    assert set(ANCHOR_OBJ_MODES) == set(ANCHOR_OBJECTIVES)
    for obj, modes in ANCHOR_OBJ_MODES.items():
        assert modes, f"{obj} names no mode"
        assert set(modes) <= set(AnchorGoalHead.MODES)
    # and the two snap modes are covered exactly by the metric objective
    assert set(ANCHOR_OBJ_MODES["metric"]) == \
        set(AnchorGoalHead.MODES) - {"onehot"}


def test_anchor_goal_loss_rejects_a_bad_objective_and_bad_shapes():
    a = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="anchor objective must be"):
        anchor_goal_loss({"mode": "snap_lat", "goal_point": torch.zeros(1, 2)},
                         torch.zeros(1, 2), a, objective="softade")
    with pytest.raises(ValueError, match=r"target_xy must be \[B, 2\]"):
        anchor_goal_loss({"mode": "snap_lat", "goal_point": torch.zeros(1, 2)},
                         torch.zeros(1, 3), a)
    with pytest.raises(ValueError, match=r"anchors must be \[K, 2\]"):
        anchor_goal_loss({"mode": "snap_lat", "goal_point": torch.zeros(1, 2)},
                         torch.zeros(1, 2), torch.zeros(4, 3, 2))
    with pytest.raises(ValueError, match="non-negative"):
        anchor_goal_loss({"mode": "snap_lat", "goal_point": torch.zeros(1, 2)},
                         torch.zeros(1, 2), a, axis_w=(-1.0, 1.0))


# =========================================================================== #
# 3. ⭐ THE AXIS WEIGHTING — decided on the 98.8 % measurement, not on symmetry
# =========================================================================== #

def test_the_default_axis_weights_are_RAW_METRES():
    assert ANCHOR_AXIS_W_DEFAULT == (1.0, 1.0)
    assert list(build_parser().get_default("anchor_axis_w")) == [1.0, 1.0]


def test_raw_metres_ALREADY_allocates_the_gradient_where_the_headroom_is():
    """⭐ THE ARITHMETIC BEHIND THE CHOICE, computed rather than asserted.

    Under the MEASURED residual (ridge σ_long 6.6132 / σ_lat 1.0667) a raw-metre
    squared-error loss spends ~97 % of its gradient LONGITUDINALLY — the axis
    that carries 98.8 % of the corpus variance and 14.9x (vs 1.96x) of the
    distance-to-floor. WHITENING would move it to ~50/50, i.e. spend half the
    gradient on the axis carrying 1.2 % of the variance — which is §6.4's own
    diagnosis of what the isotropic FPS vocabulary does WRONG, repeated one
    level up in the objective.
    """
    raw_share = SIGMA_LONG_RIDGE ** 2 / (SIGMA_LONG_RIDGE ** 2
                                         + SIGMA_LAT_RIDGE ** 2)
    assert raw_share == pytest.approx(0.974, abs=0.005)
    wl, wt = 1.0 / SIGMA_LONG_CORPUS, 1.0 / SIGMA_LAT_CORPUS
    white = (wl * SIGMA_LONG_RIDGE) ** 2 / (
        (wl * SIGMA_LONG_RIDGE) ** 2 + (wt * SIGMA_LAT_RIDGE) ** 2)
    assert white < 0.55, "whitening was supposed to EQUALISE the axes"
    # the headroom (distance to the quantisation floor) is longitudinal too
    assert (SIGMA_LONG_CLF - SIGMA_LONG_FLOOR) > \
        18.0 * (SIGMA_LAT_CLF - SIGMA_LAT_FLOOR)


def test_the_axis_weights_actually_reweight_the_loss():
    """A DECLARED knob that did nothing would be worse than no knob."""
    head_out = {"mode": "snap_xy", "goal_point": torch.tensor([[3.0, 4.0]]),
                "goal_point_raw": torch.tensor([[3.0, 4.0]])}
    anchors = torch.zeros(2, 2)
    anchors[1] = torch.tensor([100.0, 100.0])
    tgt = torch.zeros(1, 2)
    base, _ = anchor_goal_loss(head_out, tgt, anchors)
    lon_only, _ = anchor_goal_loss(head_out, tgt, anchors, axis_w=(1.0, 0.0))
    lat_only, _ = anchor_goal_loss(head_out, tgt, anchors, axis_w=(0.0, 1.0))
    assert float(base) == pytest.approx(5.0)
    assert float(lon_only) == pytest.approx(3.0)
    assert float(lat_only) == pytest.approx(4.0)


def test_the_log_reports_LON_and_LAT_separately_never_pooled():
    """⛔ PI 2026-08-02, binding: per-family, never pooled into one score. 98.8 %
    of this quantity's variance is longitudinal, so a single scalar would hide
    exactly the axis that is the problem."""
    s = _anchor_stack("snap_lat")
    with torch.no_grad():
        s.anchor_head.goal_point.weight.normal_(0.0, 0.3)
    out = s.anchor_head(torch.randn(4, s.cfg.d_goal_embed))
    _, log = anchor_goal_loss(out, torch.randn(4, 2) * 3.0,
                              s.anchor_head.anchors)
    for key in ("anchor_err_lon_m", "anchor_err_lat_m", "anchor_lon_share_sq",
                "anchor_floor_m", "anchor_free_err_m", "anchor_axis_w"):
        assert key in log, f"{key} missing — the per-family read is incomplete"
    assert 0.0 <= log["anchor_lon_share_sq"] <= 1.0


def test_the_onehot_control_still_reports_metres_and_top1():
    """A control arm has to be comparable with the default on the quantity that
    matters, or it cannot function as a control. CE's own loss is in NATS, so
    the metre-valued diagnostics travel with it."""
    s = _anchor_stack("onehot", k=8)
    out = s.anchor_head(torch.randn(5, s.cfg.d_goal_embed))
    _, log = anchor_goal_loss(out, torch.randn(5, 2) * 3.0,
                              s.anchor_head.anchors, objective="ce")
    assert log["anchor_chance"] == pytest.approx(1 / 8)
    for key in ("anchor_top1_acc", "anchor_expected_err_m",
                "anchor_argmax_err_m", "anchor_err_lon_m", "anchor_err_lat_m"):
        assert key in log


# =========================================================================== #
# 4. ⛔ THE REFUSALS — every one fires in MILLISECONDS, not after a GPU-day
# =========================================================================== #

def _problems(*extra) -> str:
    return " || ".join(preflight(_args(*extra)))


def test_ce_is_REACHABLE_ONLY_as_the_acknowledged_control():
    """⭐ THE MECHANISM THAT KEEPS A REFUTED OBJECTIVE FROM BECOMING A DEFAULT.

    It is the SAME acknowledgement ``--no-isolate-planner`` already uses, so a
    control arm is always a conscious act. Both directions are checked: the
    refusal fires without the flag, and — the half a one-sided test would miss —
    the arm IS launchable with it. A control nobody can run is not a control.
    """
    base = ["--anchor-goal", "onehot", "--goal-cat-args",
            "--anchor-table", "x.pt", "--w-anchor", "1.0",
            "--anchor-objective", "ce"]
    assert "MEASURED-REFUTED" in _problems(*base)
    # the ack flag is registered in ``main`` with dest="control_arm_ack" (it is
    # argparse.SUPPRESSed in the parser), so the namespace carries the DEST.
    a = _args(*base)
    a.control_arm_ack = True
    assert not any("MEASURED-REFUTED" in p for p in preflight(a))
    # ...and the DEFAULT objective never needs the flag.
    assert "MEASURED-REFUTED" not in _problems(
        "--anchor-goal", "snap_lat", "--goal-cat-args",
        "--anchor-table", "x.pt", "--w-anchor", "1.0")


@pytest.mark.parametrize("extra,needle", [
    (["--w-anchor", "1.0"], "with --anchor-goal none"),
    (["--anchor-goal", "snap_lat", "--goal-cat-args"], "without --anchor-table"),
    (["--anchor-goal", "snap_lat", "--anchor-table", "x.pt"],
     "without --goal-cat-args"),
    (["--anchor-goal", "onehot", "--goal-cat-args", "--anchor-table", "x.pt",
      "--w-anchor", "1.0", "--anchor-objective", "metric"],
     "needs --anchor-goal in"),
    (["--anchor-goal", "snap_lat", "--goal-cat-args", "--anchor-table", "x.pt",
      "--w-anchor", "1.0", "--anchor-objective", "softanchor"],
     "needs --anchor-goal in"),
])
def test_preflight_refuses_the_mis_specified_arms(extra, needle):
    assert needle in _problems(*extra)


def test_S_W_refuses_the_anchor_head_because_the_LIVE_RUN_resumes_from_it():
    """⛔ The planner group is FROZEN in S-W, so the head would be untrainable
    dead weight AND would change the state_dict — which breaks the strict
    resume of the run that is training on Thor right now."""
    a = build_parser().parse_args(
        ["--stage", "S-W", "--out", "u", "--anchor-goal", "snap_lat",
         "--goal-cat-args", "--anchor-table", "x.pt"])
    assert any("breaks a strict resume" in p for p in preflight(a))


def test_S_S_refuses_a_w_anchor_that_would_not_be_in_force():
    """``for_stage('S-S')`` zeroes it, so a launch line carrying it would
    advertise a term that trains nothing — a run row that lies about what
    moved. The GEOMETRY must still be carried forward, so only the WEIGHT is
    refused."""
    a = build_parser().parse_args(
        ["--stage", "S-S", "--out", "u", "--anchor-goal", "snap_lat",
         "--goal-cat-args", "--anchor-table", "x.pt", "--w-anchor", "1.0",
         "--init-from", "x.pt", "--v2-cache", "c"])
    probs = preflight(a)
    assert any("not in force" in p for p in probs)
    assert not any("Keep --anchor-goal none" in p for p in probs)


def test_the_default_command_line_raises_NO_anchor_problem():
    """The negative control for the whole refusal block: a guard that refuses
    everything guards nothing."""
    a = build_parser().parse_args(
        ["--stage", "S-T", "--out", "u", "--init-from", "x.pt",
         "--v2-cache", "c"])
    assert not any("anchor" in p.lower() for p in preflight(a))


def test_the_loss_refuses_a_missing_head_and_a_missing_target():
    stack = _build(_small())            # no anchor head at all
    batch = synthetic_train_batch(stack, batch=2, k=4, seed=2)
    batch["gt_wp"] = torch.zeros(2, 2, 2)
    with pytest.raises(ValueError, match="anchor loss with no anchor head"):
        v6_loss_step(stack, batch, stage="S-T", o1_k=2, o5_k=2,
                     weights=V6LossWeights(w_anchor=1.0))
    s2 = _anchor_stack("snap_lat")
    b2 = synthetic_train_batch(s2, batch=2, k=4, seed=2)
    b2["gt_wp"] = torch.zeros(2, 2, 2)
    b2.pop("plan_target")
    with pytest.raises(ValueError, match="needs batch..plan_target"):
        v6_loss_step(s2, b2, stage="S-T", o1_k=2, o5_k=2,
                     weights=V6LossWeights(w_anchor=1.0, lambda_plan=0.0))


# =========================================================================== #
# 5. ⛔ THE 6 s BLOCKER — enforced end to end, on the REAL banked artifacts
# =========================================================================== #

_BANKED = [
    "TanitAD Research Hub/Data Engineering/Implementation/incoming/"
    "2026-08-04-instrument-durability/refc_anchors_full_REBUILD.pt",
    "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
    "2026-07-27-percandidate-labels/raw/anchors_dev256.pt",
]


@pytest.mark.parametrize("rel", _BANKED)
def test_a_REAL_banked_table_is_refused_at_the_6s_plan_horizon(rel):
    """⛔ Every vocabulary the programme owns stops at step 20 = 2.0 s while
    ``PLAN_STEPS`` is 60. Scoring a 6 s ground truth against a 2 s anchor would
    produce a NUMBER rather than an error.

    ⭐ WITH ITS NEGATIVE CONTROL in the same test: the SAME table is ACCEPTED at
    a 2 s plan horizon. The refusal fires on the HORIZON, not on the table — a
    guard that refuses everything guards nothing.
    """
    p = _ROOT / rel
    if not p.exists():
        pytest.skip(f"banked artifact not on this box: {rel}")
    anchors, horizons = _read_anchor_table(p)
    k = int(anchors.shape[0])

    six = AnchorGoalHead(16, k, mode="snap_lat", plan_horizon_s=6.0)
    with pytest.raises(ValueError, match="REFUSING an anchor table|does NOT"):
        six.load_anchor_table(anchors, horizons, dt=0.1)
    assert not bool(six.table_ready)

    two = AnchorGoalHead(16, k, mode="snap_lat", plan_horizon_s=2.0)
    prov = two.load_anchor_table(anchors, horizons, dt=0.1)
    assert bool(two.table_ready) and prov["horizon_s"] == pytest.approx(2.0)


def test_read_anchor_table_never_INVENTS_horizons():
    """A bare ``[K, S, 2]`` carries no horizons, and the two real shipped shapes
    are ``[5,10,15,20]`` and ``[1..20]`` — guessing between them mislabels the
    whole corpus. ``horizons=None`` is passed through so the head refuses it BY
    NAME instead of silently reading ``anchors[:, -1]``."""
    tmp = Path(tempfile.mkdtemp()) / "bare.pt"
    torch.save(torch.randn(8, 4, 2), tmp)
    anchors, horizons = _read_anchor_table(tmp)
    assert horizons is None
    head = AnchorGoalHead(16, 8, mode="snap_lat", plan_horizon_s=2.0)
    with pytest.raises(ValueError, match="needs its `horizons`"):
        head.load_anchor_table(anchors, horizons, dt=0.1)


def test_read_anchor_table_refuses_a_non_artifact():
    tmp = Path(tempfile.mkdtemp()) / "nope.pt"
    torch.save({"weights": torch.zeros(2)}, tmp)
    with pytest.raises(SystemExit, match="no 'anchors' key"):
        _read_anchor_table(tmp)
    with pytest.raises(SystemExit, match="does not exist"):
        _read_anchor_table(tmp.parent / "missing.pt")


# =========================================================================== #
# 6. ⛔ X3 + ADMISSIBILITY — audited on the arms this turn makes launchable
# =========================================================================== #

@pytest.mark.parametrize("mode", ["snap_lat", "snap_xy", "onehot"])
def test_X3_holds_on_every_anchor_arm_built_FROM_ARGS(mode):
    """``{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}``
    on a real autograd graph, with ``n_probed > 0`` — a probe over an empty
    parameter set reports zero violations and has established nothing."""
    s = _anchor_stack(mode)
    rep = s.assert_isolation(batch_size=1)
    assert rep["pass"] is True
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}
    assert all(v > 0 for v in rep["n_probed"].values())


def test_the_objective_reads_no_situation_classifier_no_ego_no_v0():
    """⛔ PI 2026-08-03 (both halves). The goal path stays information-disjoint
    from the situation classifier, and inference is vision-only.

    Checked on the SIGNATURE and on the SOURCE, because a rule enforced only by
    a comment is a rule that drifts. ``target_xy`` is a LABEL (future ego
    poses) — labels may use ego; nothing here enters the head, whose only input
    is the vision-derived ``e_g_tac``.
    """
    params = list(inspect.signature(anchor_goal_loss).parameters)
    assert params == ["head_out", "target_xy", "anchors", "objective",
                      "axis_w"]
    # the CODE, with the docstring removed: the docstring NAMES these rules and
    # a scan that could not tell the two apart would either pass vacuously or
    # fail on its own explanation.
    src = inspect.getsource(anchor_goal_loss)
    body = src.split('"""')[2]
    assert "objective not in ANCHOR_OBJECTIVES" in body    # the scan found code
    for forbidden in ("situations", "situation", "detect_lane_change",
                      "v0", "ego", "nav_cmd"):
        assert forbidden not in body, \
            f"{forbidden!r} appears in the objective's CODE"


def test_the_emitted_goal_is_INVARIANT_to_v0():
    """🟥 ``v0``'s admissibility is an OPEN PI DECISION worth a MEASURED 2.85x.
    Nothing built here depends on it — pinned BEHAVIOURALLY (tripling and
    offsetting ``v0`` leaves the emitted goal bit-identical), not by reading the
    code."""
    s = _anchor_stack("snap_lat")
    with torch.no_grad():
        s.anchor_head.goal_point.weight.normal_(0.0, 0.3)
    b = s.synthetic_batch(2)
    a = s.forward(**b)["plan"]["anchor_goal_point"]
    b2 = dict(b, v0=b["v0"] * 3.0 + 7.0)
    c = s.forward(**b2)["plan"]["anchor_goal_point"]
    assert torch.equal(a, c)


# =========================================================================== #
# 7. ⭐ THE ``"mlp"`` PROBE HOLE — fc1 was invisible to the X3 surface probe
# =========================================================================== #

def test_mlp_fc2_is_STILL_zero_init_after_the_probe_fix():
    """⛔ THE CONSTRAINT ON THE FIX. ``fc2`` is zero-init BY DESIGN so the
    capacity control starts FLAT over the fan — any ranking it acquires must be
    something it LEARNED. The probe hole must be closed WITHOUT giving the
    control a ranking at initialisation, so the perturbation lives in the TEST
    and the module is untouched: the TRAINED arm still starts flat."""
    m = MLPCandidateScorer(8, 4)
    assert float(m.fc2.weight.detach().abs().max()) == 0.0
    assert float(m.fc2.bias.detach().abs().max()) == 0.0
    wp = torch.randn(2, 4, 6, 2)
    score = m(wp, torch.randn(2, 8))["score"]
    # flat over the fan at init: every candidate scores its bias and nothing
    # else, so the argmax is not a ranking.
    assert torch.allclose(score, m.cand_bias.expand_as(score), atol=1e-6)


def test_fc1_IS_reachable_once_the_probe_runs_off_the_zero_init():
    """⭐ THE HOLE, AND ITS CLOSURE. With ``fc2`` at zero, ``dL/dfc1 ∝ W_fc2 =
    0`` EXACTLY, so the X3 planner-surface probe reports ``fc1`` unreachable and
    a mis-wire there would go uncaught. Reachability is an ARCHITECTURE
    property (the probe's own philosophy — the emission and FiLM zero-inits are
    already handled this way), so the probe runs off the zero init.

    Both halves are measured here: the hole EXISTS at zero init, and it CLOSES
    under the perturbation.
    """
    s = _build(_small(selector="mlp"))
    names = {"cand_score.fc1.weight", "cand_score.fc1.bias"}

    out = s.forward(**s.synthetic_batch(2))
    live0 = set(V6Stack._live_edges(
        V6Stack._probe_scalar(out["planner_side"]),
        list(s.group_parameters("planner"))))
    assert not (names & live0), \
        "fc1 was already visible — this test no longer measures the hole"

    with torch.no_grad():                       # off the zero init, restored by
        s.cand_score.fc2.weight.normal_(0.0, 0.1)   # rebuilding, never reused
        s.cand_score.fc2.bias.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    live1 = set(V6Stack._live_edges(
        V6Stack._probe_scalar(out["planner_side"]),
        list(s.group_parameters("planner"))))
    assert names <= live1, f"fc1 STILL invisible to the probe: {names - live1}"


def test_the_fc1_probe_CAN_FAIL_when_fc1_is_genuinely_unreachable():
    """⭐ THE NEGATIVE CONTROL. A probe that reports "reachable" for a
    genuinely disconnected layer would be worse than no probe. Here ``fc1`` is
    bypassed in the forward — architecturally dead — and the extended probe must
    report it MISSING even with ``fc2`` perturbed."""
    s = _build(_small(selector="mlp"))
    sc = s.cand_score
    d_in = sc.fc1.out_features

    def bypass(waypoints, g_embed, _sc=sc, _d=d_in):
        b, n = waypoints.shape[0], waypoints.shape[1]
        h = torch.zeros(b, n, _d, dtype=waypoints.dtype) + 1.0
        return {"score": _sc.fc2(h).squeeze(-1) + _sc.cand_bias,
                "mechanism": "mlp"}

    sc.forward = bypass
    with torch.no_grad():
        sc.fc2.weight.normal_(0.0, 0.1)
        sc.fc2.bias.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    live = set(V6Stack._live_edges(
        V6Stack._probe_scalar(out["planner_side"]),
        list(s.group_parameters("planner"))))
    assert "cand_score.fc1.weight" not in live, \
        "the probe reported a DISCONNECTED fc1 as reachable"
