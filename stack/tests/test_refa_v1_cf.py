"""Change #10 — the counterfactual-action term in REF-A v1.

⛔ WHY v1 NEEDED IT. Change #4 makes the primary loss "predict the future patch
features". That target is TEACHER-FORCED: it already contains the action's effect, so a
predictor can match it WITHOUT using the action. UWM-JEPA (2605.25313) states the
mechanism and says it "applies beyond the unitary parameterisation"; our own campaign
measured the same thing three independent ways. Without this term v1 would have inherited
REF-A's original symptom -- scores on context, ignores actions.

⭐ THE PROPERTY UNDER TEST IS THE FLOOR, NOT THE LOSS VALUE. An action-independent
predictor emits the same rollout for every action, so every logit is equal, the softmax is
uniform, and the loss is EXACTLY ln(1 + n_neg). That makes "did the predictor use the
action" answerable against arithmetic rather than against a baseline someone can dispute.
"""
import math

import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _tiny(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v6.0", d_enc=32, d_state=32, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=1.5, str_steps=4, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _batch(c: RefAV1Config, b=4, k=8):
    return (torch.randn(b, c.op_window, c.n_tokens, c.d_enc),
            torch.randn(b, max(k, c.op_steps), c.a_dim),
            torch.randn(b, k, c.n_tokens, c.d_enc))


def test_the_term_is_OFF_by_default_and_adds_no_parameters():
    """⚠️ Default-off means every existing launch and checkpoint is untouched;
    zero new parameters means the state_dict is byte-identical either way."""
    assert RefAV1Config().w_cf == 0.0
    n_off = sum(p.numel() for p in RefAV1(_tiny()).parameters())
    n_on = sum(p.numel() for p in RefAV1(_tiny(w_cf=1.0)).parameters())
    assert n_off == n_on, "the counterfactual term must add no parameters"
    a = RefAV1(_tiny()).state_dict().keys()
    b = RefAV1(_tiny(w_cf=1.0)).state_dict().keys()
    assert set(a) == set(b)


def test_off_by_default_emits_no_cf_keys():
    m = RefAV1(_tiny())
    f, a, fut = _batch(m.cfg)
    out = m(f, a, future_feats=fut)
    assert "cf_loss" not in out and "loss" in out


def test_on_it_emits_the_floor_and_the_excess():
    m = RefAV1(_tiny(w_cf=1.0, cf_negs=3))
    f, a, fut = _batch(m.cfg)
    out = m(f, a, future_feats=fut)
    assert out["cf_no_info_floor"] == pytest.approx(math.log(4.0))
    assert out["cf_chance_acc"] == pytest.approx(0.25)
    assert out["cf_excess"] == pytest.approx(
        out["cf_no_info_floor"] - float(out["cf_loss"]), abs=1e-6)
    assert torch.isfinite(out["cf_loss"])


def test_AN_ACTION_BLIND_PREDICTOR_SCORES_EXACTLY_THE_FLOOR():
    """⭐⭐ THE LOAD-BEARING TEST. This is the whole reason the term is an
    instrument: make the rollout ignore its action, and the loss must land on
    ln(1+n_neg) to numerical precision -- not near it, ON it."""
    m = RefAV1(_tiny(w_cf=1.0, cf_negs=3))
    f, a, fut = _batch(m.cfg)
    real_rollout = m.operative.rollout

    def action_blind(last, actions, intent=None):
        return real_rollout(last, torch.zeros_like(actions), intent=intent)

    m.operative.rollout = action_blind
    out = m(f, a, future_feats=fut)
    assert float(out["cf_loss"]) == pytest.approx(math.log(4.0), abs=1e-5)
    assert out["cf_excess"] == pytest.approx(0.0, abs=1e-5)


def test_negatives_are_a_DERANGEMENT_never_a_permutation():
    """⛔ A permutation fixes points with probability ~1/B, handing that row its
    OWN actions as a 'counterfactual' -- which pulls the loss toward the floor
    and reads as action-blindness that is not there. The source must roll."""
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "tanitad" / "refs" / "refa_v1.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_cf_term")
    # ⚠️ STRIP THE DOCSTRING FIRST. The prose deliberately NAMES randperm to
    # explain why it is forbidden, so a raw substring search fails on the very
    # comment that documents the rule -- a scope error, not a defect.
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    code = chr(10).join(ast.unparse(n) for n in body)
    assert "torch.roll" in code, "negatives must be drawn by a cyclic roll"
    assert "randperm" not in code, "randperm fixes points -- forbidden here"


def test_it_refuses_a_batch_too_small_for_a_counterfactual():
    m = RefAV1(_tiny(w_cf=1.0))
    f, a, fut = _batch(m.cfg, b=1)
    with pytest.raises(ValueError) as e:
        m(f, a, future_feats=fut)
    assert "batch >= 2" in str(e.value)


def test_sanity_refuses_an_ADVERTISED_BUT_INERT_configuration():
    """⚠️ w_cf > 0 with zero negatives makes the InfoNCE a constant: the launch
    line would advertise the term and the loss would not contain it."""
    with pytest.raises(ValueError) as e:
        _tiny(w_cf=1.0, cf_negs=0).sanity()
    assert "inert" in str(e.value)
    with pytest.raises(ValueError):
        _tiny(w_cf=1.0, cf_at_step=999).sanity()
    with pytest.raises(ValueError):
        _tiny(w_cf=-1.0).sanity()


def test_the_term_reaches_the_total_loss_and_backprops():
    """⚠️ A term that is computed and not added is the advertised-but-inert
    defect one level in — it would log beautifully and train nothing."""
    m = RefAV1(_tiny(w_cf=2.0, cf_negs=2))
    f, a, fut = _batch(m.cfg)
    out = m(f, a, future_feats=fut)
    feat_only = (m.cfg.w_feat_op * out["loss_feat_op"]
                 + m.cfg.w_feat_tac * out["loss_feat_tac"]
                 + m.cfg.w_feat_str * out["loss_feat_str"])
    assert float(out["loss"]) == pytest.approx(
        float(feat_only) + 2.0 * float(out["cf_loss"]), abs=1e-5)
    out["loss"].backward()
    g = [p.grad for p in m.operative.parameters() if p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g)
