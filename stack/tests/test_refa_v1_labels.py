"""v7.2 label supervision (PI 2026-08-31: "It must be trained with this data").

⛔ THE DEFECT PINNED FIRST: `lat_head`, `lon_head` and `route_logits` were
EMITTED by forward and appeared in NO loss term — inert parameters that would
have sat at initialisation through an entire 30k run while the eval decoded
them as "the tactical action". `test_the_DEFECT_heads_get_no_gradient_without_
labels` keeps that fact on the record; the rest pins the repair.
"""
import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v6.0", d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1,
                                                    n_heads=2, d_ctx=8,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1,
                                                  n_heads=2, d_intent=8))
    base.update(kw)
    return RefAV1Config(**base)


def _inputs(c: RefAV1Config, b=3):
    return (torch.randn(b, c.op_window, c.n_tokens, c.d_enc),
            torch.randn(b, c.op_steps, c.a_dim),
            torch.randn(b, c.op_steps, c.n_tokens, c.d_enc))


def _labels(m: RefAV1, b=3):
    g = torch.Generator().manual_seed(7)
    return (torch.randint(0, m.n_lat, (b,), generator=g),
            torch.randint(0, m.n_lon, (b,), generator=g),
            torch.randint(0, 3, (b,), generator=g))      # route: L/S/R


def test_the_DEFECT_heads_get_no_gradient_without_labels():
    """⭐ Without labels, backprop leaves every lat/lon head parameter with no
    gradient — the heads were decorative. This is the state every earlier
    commit shipped."""
    m = RefAV1(_cfg())
    f, a, fut = _inputs(m.cfg)
    m(f, a, future_feats=fut)["loss"].backward()
    for head in (m.lat_head, m.lon_head):
        assert all(p.grad is None for p in head.parameters())


def test_with_labels_every_head_trains():
    m = RefAV1(_cfg())
    f, a, fut = _inputs(m.cfg)
    lat, lon, route = _labels(m)
    out = m(f, a, future_feats=fut,
            lat_label=lat, lon_label=lon, route_label=route)
    for k in ("loss_lat_label", "loss_lon_label", "loss_route_label"):
        assert k in out and torch.isfinite(out[k])
    out["loss"].backward()
    for head in (m.lat_head, m.lon_head):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in head.parameters())
    # the route CE reaches the strategic policy's route head too
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.strategic_policy.parameters())


def test_the_weights_compose_exactly():
    c = _cfg(w_tac_label=0.3, w_str_label=0.2)
    m = RefAV1(c)
    f, a, fut = _inputs(c)
    lat, lon, route = _labels(m)
    torch.manual_seed(0)
    base = m(f, a, future_feats=fut)
    full = m(f, a, future_feats=fut,
             lat_label=lat, lon_label=lon, route_label=route)
    want = (float(base["loss"].detach())
            + 0.3 * float((full["loss_lat_label"]
                           + full["loss_lon_label"]).detach()) / 2
            + 0.2 * float(full["loss_route_label"].detach()))
    assert float(full["loss"].detach()) == pytest.approx(want, abs=1e-5)


def test_partial_labels_are_fine_missing_ones_just_skip():
    m = RefAV1(_cfg())
    f, a, fut = _inputs(m.cfg)
    lat, _, _ = _labels(m)
    out = m(f, a, future_feats=fut, lat_label=lat)
    assert "loss_lat_label" in out
    assert "loss_lon_label" not in out and "loss_route_label" not in out


def test_out_of_range_labels_are_refused_by_name():
    m = RefAV1(_cfg())
    f, a, fut = _inputs(m.cfg)
    with pytest.raises(ValueError, match="lat_label outside"):
        m(f, a, future_feats=fut,
          lat_label=torch.tensor([0, 1, m.n_lat]))
    with pytest.raises(ValueError, match="route_label outside"):
        m(f, a, future_feats=fut, route_label=torch.tensor([0, 1, 3]))


def test_labels_without_a_hierarchy_are_refused():
    """The no_hierarchy ablation has no heads — accepting labels there would
    silently score nothing."""
    m = RefAV1(_cfg(strategic_cfg=None, tactical_cfg=None))
    f, a, fut = _inputs(m.cfg)
    with pytest.raises(ValueError, match="hierarchy is off"):
        m(f, a, future_feats=fut, lat_label=torch.zeros(3, dtype=torch.long))


def test_labels_without_future_feats_are_refused():
    m = RefAV1(_cfg())
    f, a, _ = _inputs(m.cfg)
    with pytest.raises(ValueError, match="silently unused"):
        m(f, a, lat_label=torch.zeros(3, dtype=torch.long))
