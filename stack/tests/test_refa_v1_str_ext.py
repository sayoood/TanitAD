"""The long-horizon strategic extension (PI decision 2026-08-31).

Verbatim: *"I think you can use 3.0 x2. The strategic layer must have its long
horizon predictor in the abstract latent space not only 6 seconds."*

⇒ two commitments, each pinned here:
  1. the in-window ladder is 0.2 / 0.6 / **3.0** s — 1 : 3 : 15, the ratio
     MM-E15 read off the corpus label bands;
  2. the strategic SUBSPACE predictor continues PAST the 6.0 s operative grid,
     supervised from cached features at 6.0 + k*str_dt (default: 9.0 / 12.0 s,
     reaching INSIDE the strategic label band [8, 30) at MM-E15's median
     manoeuvre start of 12.5 s).

⛔ D-REFAV1-LADDER's lesson travels with this file: assert the realised
behaviour (which tick lands where, whether the term reaches the loss), never
the config arithmetic alone.
"""
import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _inputs(c: RefAV1Config, b=2, ext=True):
    feats = torch.randn(b, c.op_window, c.n_tokens, c.d_enc)
    actions = torch.randn(b, c.op_steps, c.a_dim)
    future = torch.randn(b, c.op_steps, c.n_tokens, c.d_enc)
    if not ext:
        return feats, actions, future, None, None
    k = c.str_ext_steps
    return (feats, actions, future,
            torch.randn(b, k, c.n_tokens, c.d_enc),
            torch.randn(b, k, c.a_dim))


# ------------------------------------------------------------ the decision --
def test_the_default_ladder_is_the_PI_decided_1_3_15():
    c = RefAV1Config()
    c.sanity()
    assert (c.op_dt, c.tac_dt, c.str_dt) == (0.2, 0.6, 3.0)
    assert c.str_steps == 2
    assert c.tac_dt / c.op_dt == pytest.approx(3.0)
    assert c.str_dt / c.op_dt == pytest.approx(15.0)          # 1 : 3 : 15


def test_the_default_extension_reaches_INSIDE_the_strategic_band():
    """MM-E15: strategic_s = [8, 30) s, median manoeuvre start 12.5 s. The 6 s
    rung provably never reached the band; the extension must."""
    c = RefAV1Config()
    assert c.str_horizon_s == pytest.approx(12.0)
    assert c.str_ext_steps == 2                               # 9.0 s, 12.0 s
    assert c.str_horizon_s >= 8.0                             # inside the band


def test_ext_ticks_land_at_nine_and_twelve_seconds():
    c = _cfg()
    m = RefAV1(c)
    feats, actions, future, et, ea = _inputs(c)
    out = m(feats, actions, future_feats=future,
            str_ext_targets=et, str_ext_actions=ea)
    assert out["str_ext_target_s"] == [9.0, 12.0]
    assert out["str_pred"].shape[1] == c.str_steps            # 6 s in-window
    assert out["str_pred_ext"].shape[1] == c.str_ext_steps


# ---------------------------------------------------- one continued rollout --
def test_the_extension_CONTINUES_the_inwindow_rollout_not_a_second_head():
    """⭐ Feed identical inputs with and without the extension: the in-window
    prediction must be BIT-IDENTICAL. If adding the extension changed the 6 s
    ticks, it would be a parallel head wearing a rollout's name."""
    c = _cfg()
    torch.manual_seed(0)
    m = RefAV1(c).eval()
    feats, actions, future, et, ea = _inputs(c)
    with torch.no_grad():
        a = m(feats, actions, future_feats=future)
        b = m(feats, actions, future_feats=future,
              str_ext_targets=et, str_ext_actions=ea)
    assert torch.equal(a["str_pred"], b["str_pred"])
    # and the ext ticks continue FROM the last in-window state: tick 3 differs
    # from tick 2 (it rolled), and no in-window tensor was recomputed.
    assert not torch.equal(b["str_pred_ext"][:, 0], b["str_pred"][:, -1])


def test_the_ext_term_reaches_the_loss_and_backprops_to_the_strategic_predictor():
    c = _cfg(w_feat_str_ext=0.5)
    m = RefAV1(c)
    feats, actions, future, et, ea = _inputs(c)
    base = m(feats, actions, future_feats=future)
    full = m(feats, actions, future_feats=future,
             str_ext_targets=et, str_ext_actions=ea)
    assert "loss_feat_str_ext" in full
    assert float(full["loss"].detach()) == pytest.approx(
        float(base["loss"].detach())
        + 0.5 * float(full["loss_feat_str_ext"].detach()), abs=1e-5)
    full["loss"].backward()
    g = [p.grad for p in m.strategic.parameters() if p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g)


# ------------------------------------------------------------- the refusals --
def test_targets_without_actions_is_a_contract_error_and_vice_versa():
    c = _cfg()
    m = RefAV1(c)
    feats, actions, future, et, ea = _inputs(c)
    with pytest.raises(ValueError, match="PAIR"):
        m(feats, actions, future_feats=future, str_ext_targets=et)
    with pytest.raises(ValueError, match="PAIR"):
        m(feats, actions, future_feats=future, str_ext_actions=ea)


def test_ext_without_a_training_loss_is_refused_not_silently_dropped():
    """⛔ Without future_feats there is no `loss` to attach to — accepting the
    targets and using none of them is the silent-no-op family."""
    c = _cfg()
    m = RefAV1(c)
    feats, actions, _, et, ea = _inputs(c)
    with pytest.raises(ValueError, match="silently unused"):
        m(feats, actions, str_ext_targets=et, str_ext_actions=ea)


def test_a_wrong_tick_count_is_refused_by_name():
    c = _cfg()
    m = RefAV1(c)
    feats, actions, future, et, ea = _inputs(c)
    with pytest.raises(ValueError, match="str_ext_targets carries 1"):
        m(feats, actions, future_feats=future,
          str_ext_targets=et[:, :1], str_ext_actions=ea)


def test_sanity_refuses_an_inexpressible_or_shrunken_horizon():
    with pytest.raises(ValueError, match="not an integer"):
        _cfg(str_horizon_s=10.0).sanity()          # (10-6)/3 = 1.33 ticks
    with pytest.raises(ValueError, match="cannot shrink"):
        _cfg(str_horizon_s=3.0).sanity()
    with pytest.raises(ValueError, match="advertised .* inert|inert"):
        _cfg(str_horizon_s=6.0).sanity()           # w_ext > 0, zero ticks
    _cfg(str_horizon_s=6.0, w_feat_str_ext=0.0).sanity()      # explicit off: OK
    _cfg(str_horizon_s=15.0).sanity()              # (15-6)/3 = 3 ticks: OK


def test_the_smoke_path_without_extension_inputs_is_untouched():
    c = _cfg()
    m = RefAV1(c)
    feats, actions, future, _, _ = _inputs(c, ext=False)
    out = m(feats, actions, future_feats=future)
    assert "str_pred_ext" not in out and "loss_feat_str_ext" not in out
    assert torch.isfinite(out["loss"].detach())
