"""E-OBJ-1 — the RANKED-SCORE OBJECTIVE. Three forms, ZERO parameters, one implementation.

WHY THIS FILE EXISTS
============================================================================
``refc_train.loss_rcls`` is a ONE-HOT cross-entropy over ~128 **near-duplicate** candidates: one
winner, 127 losers, and the loser that missed by a centimetre is penalised exactly as hard as the
one that missed by ten metres. MEASURED (E-S1-0 §3.1, frozen 30 k weights, 881 windows, LOEO,
paired episode-cluster bootstrap): under that objective **every** fitted ranker is separated WORSE
than the incumbent selector — *including feature sets that contain the incumbent's own score* —
with a C-leak gap of −0.001 to −0.003 m, i.e. **not** overfitting.

E-OBJ-1 pre-registered the swap and split it in two, because they are different one-line changes:

  ``softade``  the EXPECTED fan error under the score's own softmax — METRIC-AWARE.
               ⭐ MEASURED to recover **−0.0974 m** (base) / **−0.1670 m** (XL) of the deficit,
               separated, and the recovery is **LONGITUDINAL** (``speed_abs`` −0.1102 / −0.1816),
               which is the family carrying 87.6–89.9 % of the selection gap.
  ``softce``   the CE FORM with a softened TARGET ``softmax(-fan_err/tau)``.
               ⚠️ MEASURED SEPARATED **WORSE** than the incumbent (+0.0909 m base) at every tau
               in {0.1, 0.25, 0.5}. It is implemented as the **CONTROL** that separates
               metric-awareness from target-softness in training, never as a candidate.

⭐ THE STRUCTURAL REASON THIS IS NOT ONLY ABOUT S1. ``refc_train.py`` states it in source: ``argmax``
has no gradient and the other consumers of the selected trajectory differentiate w.r.t. the FAN,
never w.r.t. the SCORE — so **without this term ``cons_gate`` (S3) and ``route_to_anchor`` (S5) get
EXACTLY ZERO gradient**. Every lever that grafts onto the ranked score is trained by this one loss.

WHAT THIS FILE PINS
============================================================================
(a) CAPACITY: all three objectives cost **EXACTLY 0** parameters. The check that caught a
    +272,001-parameter head where +897 sufficed.
(b) THE INCUMBENT IS BIT-IDENTICAL: ``objective="ce"`` reproduces ``F.cross_entropy`` exactly,
    and the ``weight`` knob is refused on that path so no inert number is recorded.
(c) CONTINUITY: ``softce`` converges to the incumbent as ``tau -> 0``. A knob that does not
    reproduce what it claims to generalise is not the knob it is documented as.
(d) ⛔ ``softade`` is a SELECTION objective, not a trajectory one: ``ce_err`` is DETACHED, so no
    gradient reaches the fan. Without this the arm would be unattributable.
(e) MASKED SUPPORT: every objective is finite in loss AND gradient when ``sel_ce_reach`` masks
    the fan to survivors — ``+inf * 0`` and ``0 * -inf`` are the two standard NaN factories here.
(f) THE FAIL-LOUDS, at PARSE time: a silently-inert flag that ``config.json`` records as ON is the
    D-TAC1 ``tactical_speed_input`` failure ("a conservative guard that makes an effect
    unattributable is not conservative").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_train                                                   # noqa: E402
from refc_train import ranked_score_loss                            # noqa: E402
from tanitad.refs.refc import (RefCModel, param_breakdown,          # noqa: E402
                               refc_config, refc_smoke_config)
from tests.test_refc import _make_cached_root                       # noqa: E402

OBJECTIVES = ("ce", "softade", "softce")


def _synthetic(b: int = 6, n: int = 32, seed: int = 0, mask: bool = False):
    """A ranked score and a per-candidate error on the SAME support the trainer builds."""
    torch.manual_seed(seed)
    score = torch.randn(b, n, dtype=torch.float64, requires_grad=True)
    err = torch.rand(b, n, dtype=torch.float64) * 3.0
    if not mask:
        return score, err, None
    keep = torch.rand(b, n) > 0.6
    keep[:, 0] = True                       # the decoder's own empty-set fallback
    return (score.detach().masked_fill(~keep, torch.finfo(torch.float64).min / 4)
            .requires_grad_(True),
            err.masked_fill(~keep, float("inf")), keep)


# ---------- (a) capacity: EXACTLY zero ---------------------------------------

def test_every_objective_costs_exactly_zero_parameters():
    """The swap changes a scalar loss, never a module — and "zero" is a claim that has to be
    measured. The +272,001-parameter head passed every review that did not count."""
    with torch.device("meta"):
        base = param_breakdown(RefCModel(refc_config()))
        assert base["total"] == 104_191_577              # the registry's number
        for flags in ({"sel_ce_objective": "softade"},
                      {"sel_ce_objective": "softce", "sel_ce_soft_tau": 0.25},
                      {"sel_ce_objective": "softade", "sel_ce_weight": 0.1},
                      {"sel_refined": True, "sel_reach_clamp": True,
                       "sel_ce_reach": True, "sel_ce_objective": "softade"}):
            cfg = refc_config()
            for k, v in flags.items():
                setattr(cfg, k, v)
            got = param_breakdown(RefCModel(cfg))
            assert got["total"] - base["total"] == 0, f"{flags}: +{got['total'] - base['total']}"
            assert got == base, f"{flags}: the param BREAKDOWN moved, not only the total"


def test_defaults_are_the_incumbent():
    c = refc_config()
    assert c.sel_ce_objective == "ce"
    assert c.sel_ce_soft_tau == 0.0
    assert c.sel_ce_weight == 1.0


# ---------- (b) the incumbent path is bit-identical --------------------------

def test_ce_reproduces_cross_entropy_bit_for_bit():
    """⛔ The published control arm must not move. If this fails, every D-SEL number that was
    ever compared against `refc-base-30k` has been silently re-baselined."""
    score, err, _ = _synthetic()
    got = ranked_score_loss(score, err, objective="ce")
    want = F.cross_entropy(score, err.argmin(dim=1))
    assert torch.equal(got, want)


def test_the_weight_knob_is_inert_on_the_incumbent_path_by_construction():
    """`sel_ce_weight` exists only because `softade` is in METRES while the CE is in NATS. On the
    `ce` path it must not be applied at all — a knob that silently rescales the control arm is
    worse than no knob."""
    score, err, _ = _synthetic()
    a = ranked_score_loss(score, err, objective="ce", weight=1.0)
    b = ranked_score_loss(score, err, objective="ce", weight=7.0)
    assert torch.equal(a, b)
    c = ranked_score_loss(score, err, objective="softade", weight=1.0)
    d = ranked_score_loss(score, err, objective="softade", weight=7.0)
    assert torch.allclose(d, 7.0 * c)


# ---------- (c) continuity: softce -> the incumbent as tau -> 0 --------------

@pytest.mark.parametrize("mask", [False, True])
def test_softce_converges_to_the_incumbent_ce_as_tau_goes_to_zero(mask):
    """⭐ THE CONTROL THAT MAKES `softce` A GENERALISATION AND NOT A NEW LOSS.

    ``softmax(-e/tau) -> onehot(argmin e)`` as ``tau -> 0``, so the softened CE must converge to
    the one-hot CE. ⚠️ This is EXACT at the LOSS level and only asymptotic at the SELECTION level:
    E-OBJ-1's probe MEASURED that at tau = 0.01 the selection ADE still differs, because the fan's
    top candidates are separated by LESS than 0.01 m. Both facts are real; this test pins the one
    that is a property of the code."""
    score, err, _ = _synthetic(mask=mask)
    ce = ranked_score_loss(score, err, objective="ce")
    prev = None
    for tau in (1.0, 1e-1, 1e-2, 1e-4):
        got = ranked_score_loss(score, err, objective="softce", tau=tau)
        assert torch.isfinite(got)
        gap = float((got - ce).abs().detach())
        if prev is not None:
            assert gap <= prev + 1e-12, "the gap must not grow as tau shrinks"
        prev = gap
    assert prev < 1e-6, f"softce(tau=1e-4) did not converge to the CE: gap {prev}"


def test_softce_without_a_temperature_is_refused_rather_than_silently_one_hot():
    score, err, _ = _synthetic()
    with pytest.raises(ValueError, match="tau > 0"):
        ranked_score_loss(score, err, objective="softce", tau=0.0)
    with pytest.raises(ValueError, match="unknown sel_ce_objective"):
        ranked_score_loss(score, err, objective="nope")


# ---------- (d) softade is a SELECTION objective, not a trajectory one -------

def test_softade_never_pushes_the_fan():
    """⛔ THE ATTRIBUTION BOUND. `ce_err` is differentiable w.r.t. `anchor_traj`; if the objective
    were allowed through it, a `softade` arm would be optimising the TRAJECTORY as well as the
    SELECTION and its result could not be assigned to either. The incumbent detaches its own
    target for the same reason — this is that discipline, not a new one."""
    score, _, _ = _synthetic()
    err = (torch.rand(*score.shape, dtype=torch.float64) * 3.0).requires_grad_(True)
    loss = ranked_score_loss(score, err, objective="softade")
    loss.backward()
    assert err.grad is None, "gradient reached the fan — softade is not confined to selection"
    assert score.grad is not None and torch.isfinite(score.grad).all()


def test_softade_is_metric_aware_where_the_one_hot_ce_is_not():
    """The property the whole experiment is about: moving mass from the oracle onto a candidate
    that is 10 m worse must cost more than moving it onto one that is 1 cm worse. The one-hot CE
    cannot see that difference; `softade` must."""
    score = torch.zeros(1, 3, dtype=torch.float64)
    near = torch.tensor([[0.10, 0.11, 0.12]], dtype=torch.float64)
    far = torch.tensor([[0.10, 0.11, 10.0]], dtype=torch.float64)
    ce_n = ranked_score_loss(score, near, objective="ce")
    ce_f = ranked_score_loss(score, far, objective="ce")
    assert torch.equal(ce_n, ce_f), "the one-hot CE is blind to the magnitude, by construction"
    sa_n = ranked_score_loss(score, near, objective="softade")
    sa_f = ranked_score_loss(score, far, objective="softade")
    assert float(sa_f) > float(sa_n) + 1.0


# ---------- (e) the masked support is finite in loss AND gradient ------------

@pytest.mark.parametrize("objective", OBJECTIVES)
def test_masked_support_is_finite_in_loss_and_gradient(objective):
    """With `sel_ce_reach` on, `ce_err` is +inf and `ce_score` is min/4 off the survivor set.
    `+inf * 0` (softade) and `0 * -inf` (softce) are the two standard NaN factories here, and the
    target of a masked CE that is itself masked is the third."""
    score, err, keep = _synthetic(mask=True)
    loss = ranked_score_loss(score, err, objective=objective,
                             tau=(0.25 if objective == "softce" else 0.0))
    assert torch.isfinite(loss), f"{objective}: non-finite loss on the masked support"
    loss.backward()
    assert torch.isfinite(score.grad).all(), f"{objective}: non-finite gradient"
    tgt = err.argmin(dim=1)
    assert bool(keep.gather(1, tgt[:, None]).all()), "the CE target is not a survivor"
    assert float(keep.double().mean()) < 1.0        # the support really is smaller than the fan


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_the_objective_reaches_the_ranked_score_end_to_end(objective):
    """The whole point of `loss_rcls` is that it is the ONLY gradient path to anything grafted
    onto the ranked score — `refc_train` says so in source: without it `cons_gate` and
    `route_to_anchor` receive EXACTLY ZERO gradient. So every objective must actually reach the
    decoder's confidence head through a real forward, not only through a synthetic tensor."""
    cfg = refc_smoke_config()
    cfg.sel_refined = cfg.sel_reach_clamp = cfg.sel_ce_reach = True
    cfg.sel_ce_objective = objective
    if objective == "softce":
        cfg.sel_ce_soft_tau = 0.25
    torch.manual_seed(0)
    m = RefCModel(cfg).train()
    out = m(torch.randn(4, 4, 1, 64, 64), v0=torch.tensor([3.0, 10.0, 0.5, 7.0]), steps=2)
    keep = out["reach_keep"]
    fan_err = out["anchor_traj"].norm(dim=-1).mean(-1)          # stand-in target
    ce_err = fan_err.masked_fill(~keep, float("inf"))
    ce_score = out["sel_score"].masked_fill(
        ~keep, torch.finfo(out["sel_score"].dtype).min / 4)
    loss = ranked_score_loss(ce_score, ce_err, objective=objective,
                            tau=cfg.sel_ce_soft_tau)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------- (f) the fail-louds, at PARSE time --------------------------------

def test_the_objective_is_refused_when_there_is_no_ranked_score_to_train(tmp_path):
    """⛔ `loss_rcls` is only CONSTRUCTED when one of `sel_refined` / `graft_cons` /
    `graft_route` is on. With none of them there is no ranked-score objective to change, and the
    flag would be SILENTLY INERT while config.json recorded it as ON."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="SILENTLY"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "a"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-ce-objective", "softade"])


def test_softce_without_a_temperature_is_refused_at_parse_time(tmp_path):
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="requires --sel-ce-soft-tau"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "b"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-refined", "--sel-ce-objective", "softce"])


def test_a_temperature_without_softce_is_refused_at_parse_time(tmp_path):
    """The mirror of the above: a number the run never reads must not reach config.json."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="only read by"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "c"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-refined", "--sel-ce-soft-tau", "0.25"])


def test_the_weight_is_refused_on_the_incumbent_path(tmp_path):
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="only applied off the incumbent"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "d"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-refined", "--sel-ce-weight", "0.1"])


def test_a_negative_temperature_is_refused(tmp_path):
    """A negative temperature puts the CE's mass on the WORST candidate — a sign error that
    would train the selector to be maximally wrong and still produce a falling loss curve."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="must be >= 0"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "e"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-refined", "--sel-ce-objective", "softce",
            "--sel-ce-soft-tau", "-1.0"])
