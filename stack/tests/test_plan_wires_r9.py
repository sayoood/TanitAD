"""refcv9 C8 -- the pure-tensor core of the planner wires (W-MAP-1 critic feature, W-MAP-2 channels, W-BOX tokens + the exact gate).
Each guard has a deliberate-regression arm that must go RED (monkeypatch the one line that makes it work, require the check to fail)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import BEVGrid  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import plan_wires_r9 as PW  # noqa: E402

GEOM = MB.band_geometry(100.0, 30.0)
I = {c: i for i, c in enumerate(MB.CHANNELS)}


def banded(half_m=3.0, unobs_all=False, unobs_x=None, b=1):
    """drv logit +8 for |y| <= half_m else -8; sdf = the signed lateral distance to the corridor edge (m)."""
    zs, us = [], []
    for g in GEOM:
        y = (torch.arange(g.cols, dtype=torch.float32) + 0.5) * g.cell - 30.0
        z = torch.zeros(b, MB.N_CHANNELS, g.rows, g.cols)
        inside = (y.abs() <= half_m)[None, None, :].expand(b, g.rows, g.cols)
        z[:, I["drv"]] = torch.where(inside, 8.0, -8.0)
        z[:, I["drv_sdf"]] = (half_m - y.abs())[None, None, :].expand(b, g.rows, g.cols)
        z[:, I["sdw"]] = torch.where(inside, -8.0, 8.0)
        z[:, I["lane"]] = -8.0
        u = torch.zeros(b, g.rows, g.cols, dtype=torch.bool)
        if unobs_all:
            u[:] = True
        if unobs_x is not None:
            x = g.x0 + (torch.arange(g.rows, dtype=torch.float32) + 0.5) * g.cell
            u |= ((x >= unobs_x[0]) & (x < unobs_x[1]))[None, :, None].expand(b, g.rows, g.cols)
        zs.append(z), us.append(u)
    return {"logits": zs, "unobs": us}


def straight(y=0.0, xs=(5.0, 15.0, 30.0, 45.0, 70.0)):
    return torch.tensor([[[[x, y] for x in xs]]], dtype=torch.float32)               # [1, 1, S, 2]


# ---------------------------------------------------------------- W-MAP-1
def test_critic_known_values_corridor_vs_shifted_and_all_bands_are_read():
    f0 = PW.critic_feature_banded(straight(0.0), banded())[0, 0]
    f3 = PW.critic_feature_banded(straight(3.0), banded())[0, 0]
    assert float(f0[0]) > 0.999 and float(f0[2]) == 1.0 and float(f0[3]) > 0.0           # in the corridor, every corner read
    assert float(f3[0]) < 0.001                                                          # the left corners reach y = 4.15 m
    # each band contributes: a path that lives only in the far band (40-100 m) still has evidence
    far = PW.critic_feature_banded(straight(0.0, xs=(50.0, 60.0, 80.0, 95.0)), banded())[0, 0]
    assert float(far[2]) == 1.0 and float(far[0]) > 0.999


def test_critic_out_of_extent_and_unobs_read_no_evidence_exactly():
    want = torch.tensor(PW.NO_EVIDENCE)
    beyond = PW.critic_feature_banded(straight(0.0, xs=(110.0, 120.0)), banded())[0, 0]
    lateral = PW.critic_feature_banded(straight(40.0), banded())[0, 0]
    unobs = PW.critic_feature_banded(straight(3.0), banded(unobs_all=True))[0, 0]       # would read P ~ 0 if it were read
    for f in (beyond, lateral, unobs):
        assert torch.equal(f, want), f
    # partial: only the unobserved stretch drops out
    part = PW.critic_feature_banded(straight(3.0), banded(unobs_x=(0.0, 100.0)))[0, 0]
    assert torch.equal(part, want)


def test_critic_feature_is_detached():
    mb = banded()
    for z in mb["logits"]:
        z.requires_grad_()
    f = PW.critic_feature_banded(straight(0.0), mb)
    assert not f.requires_grad


def test_RED_unobs_corner_read_as_drivable_FAILS(monkeypatch):
    def green():
        f = PW.critic_feature_banded(straight(3.0), banded(unobs_all=True))[0, 0]
        return torch.equal(f, torch.tensor(PW.NO_EVIDENCE))
    assert green()
    monkeypatch.setattr(PW, "_evidence_mask", lambda inside, unobs: inside)             # unobs ignored: the cell's logits are read
    assert not green()


# ---------------------------------------------------------------- W-MAP-2
def test_channels_1m_shape_values_and_unobs():
    mb = banded(unobs_x=(20.0, 30.0))
    c = PW.channels_1m(mb)
    assert tuple(c.shape) == (1, 5, 100, 60) and not c.requires_grad
    assert float(c[0, 0, 5, 30]) > 0.99 and float(c[0, 0, 5, 5]) < 0.01                  # near band: drv in / out of the corridor
    assert float(c[0, 4, 25, 30]) == 1.0 and float(c[0, 0, 25, 30]) == 0.0              # unobs rows: value channels 0, unobs 1
    assert float(c[0, 4, 5, 30]) == 0.0 and float(c[0, 4, 80, 30]) == 0.0
    assert float(c[0, 2, 5, 30]) < 0.01 and float(c[0, 2, 5, 5]) > 0.99                  # sdw


def test_map_chan_proj_is_zero_init_windowed_and_has_a_gradient_at_step_0():
    grid = BEVGrid(60.0, 16.0, 0.5)
    p = PW.MapChanProj(8, grid)
    c = PW.channels_1m(banded())
    y = p(c)
    assert tuple(y.shape) == (1, 8, 120, 64) and float(y.abs().max()) == 0.0              # zero-init: the pool is untouched
    p(c).sum().backward()
    assert float(p.proj.weight.grad.abs().sum()) > 0.0 and float(p.proj.bias.grad.abs().sum()) > 0.0   # the gate learns at step 0
    with torch.no_grad():
        p.proj.weight.fill_(0.5)
    assert float(p(c).abs().max()) > 0.0                                                  # the RED arm for "non-zero init changes step 0"


def test_map_chan_proj_refuses_a_window_outside_the_extent():
    with pytest.raises(ValueError):
        PW.MapChanProj(4, BEVGrid(120.0, 16.0, 0.5))(PW.channels_1m(banded()))


# ---------------------------------------------------------------- W-BOX
def slots(b=2, n=50, ncls=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    box = torch.cat([torch.rand(b, n, 2, generator=g) * torch.tensor([60.0, 10.0]), 2 + torch.rand(b, n, 2, generator=g)], -1)
    yaw = r(b, n)
    return {"box": box, "yaw_vec": torch.stack([yaw.sin(), yaw.cos()], -1), "rates": r(b, n, 3), "cls_logits": r(b, n, ncls),
            "presence_logit": r(b, n)}


def test_box_tokens_pick_the_top_32_by_presence_and_are_detached():
    s = slots()
    for k in ("box", "presence_logit"):
        s[k].requires_grad_()
    feat, pres, top = PW.box_features(s)
    assert feat.shape[:2] == (2, 32) and not feat.requires_grad
    want = s["presence_logit"].detach().topk(32, dim=1).indices
    assert torch.equal(top, want)
    assert float(pres.min()) >= float(torch.sigmoid(s["presence_logit"].detach().sort(1, descending=True).values[:, 31]).min()) - 1e-6
    tok, pad = PW.BoxTokens(16, 6)(s)
    assert tok.shape == (2, 32, 16) and not bool(pad.any())


def test_gated_mix_is_exact_at_zero_gate_and_has_gradient():
    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    q = torch.randn(2, 4, 8)
    agents, box = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    tok = torch.cat([agents, box], 1)
    att = lambda t, p: mha(q, t, t, key_padding_mask=p, need_weights=False)[0]
    gate = torch.zeros(1, requires_grad=True)
    out = PW.gated_mix(att, tok, None, 3, gate)
    assert torch.equal(out, att(agents, None))                                           # bit-exact at g = 0
    out.sum().backward()
    assert float(gate.grad.abs()) > 0.0                                                   # and the gate learns
    # the naive concatenation (extra keys in ONE softmax) is NOT inert -- the reason gated_mix exists
    assert not torch.equal(att(tok, None), att(agents, None))
    # RED: a mix that forgets the agents-only baseline is not the identity at g = 0
    bad = gate * att(tok, None)
    assert not torch.equal(bad, att(agents, None))
