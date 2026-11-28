"""refcv7 heads — WTA proposals and the disentangled scorer."""
import pytest
import torch

from tanitad.refs.refcv7_heads import (DisentangledScorer, Refcv7HeadConfig,
                                       WTAProposalDecoder, scorer_loss, wta_loss)

SLOTS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
SRC = {"img": 32, "bev": 16}


def sources(b=2):
    g = torch.Generator().manual_seed(0)
    return {"img": torch.randn(b, 10, 32, generator=g), "bev": torch.randn(b, 6, 16, generator=g)}


def small(**kw):
    return Refcv7HeadConfig(d_model=32, n_layers=2, n_heads=4, n_queries=8, cond_dim=5, **kw)


def test_wta_shapes_and_start_near_standstill():
    torch.manual_seed(0)
    m = WTAProposalDecoder(small(), SRC, SLOTS)
    out = m(sources(), torch.zeros(2, 5))
    assert out.shape == (2, 8, 8, 2)


def test_wta_loss_supervises_only_the_closest_proposal():
    gt = torch.zeros(1, 8, 2); gt[..., 0] = torch.tensor(SLOTS) * 10
    props = torch.stack([gt[0] + 5.0, gt[0] + 0.1, gt[0] - 3.0])[None]       # [1, 3, 8, 2]
    props.requires_grad_(True)
    loss, best = wta_loss(props, gt)
    assert int(best) == 1
    # DrivoR's ||tau - tau_hat||_1 sums |dx| + |dy| per waypoint, then averages over slots:
    # a 0.1 m offset on BOTH axes is 0.2 per waypoint
    assert abs(float(loss.detach()) - 0.2) < 1e-5
    loss.backward()
    g = props.grad.abs().sum(dim=(2, 3))[0]
    assert float(g[0]) == 0.0 and float(g[2]) == 0.0 and float(g[1]) > 0.0


def test_wta_loss_respects_the_gt_mask():
    gt = torch.zeros(1, 8, 2)
    props = torch.zeros(1, 2, 8, 2); props[0, 0, 4:] = 100.0
    valid = torch.zeros(1, 8, dtype=torch.bool); valid[0, :4] = True
    loss, best = wta_loss(props, gt, valid)
    assert float(loss) == 0.0            # the masked slots, where proposal 0 is wrong, do not count


def test_scorer_refuses_candidates_that_carry_the_generator_gradient():
    s = DisentangledScorer(small(), SRC, SLOTS)
    traj = torch.zeros(2, 3, 8, 2, requires_grad=True)
    with pytest.raises(ValueError, match="DETACHED"):
        s(sources(), traj, torch.ones(2), torch.zeros(2, 5))
    s(sources(), traj.detach(), torch.ones(2), torch.zeros(2, 5))       # detached is fine


def _score_of_first(s, companions):
    torch.manual_seed(1)
    base = torch.cumsum(torch.full((1, 1, 8, 2), 1.0), dim=2).expand(2, 1, 8, 2)
    traj = torch.cat([base, companions], dim=1)
    with torch.no_grad():
        return s(sources(), traj, torch.ones(2), torch.zeros(2, 5))["nc"][:, 0]


def test_a_candidates_score_does_not_depend_on_its_companions():
    torch.manual_seed(0)
    s = DisentangledScorer(small(), SRC, SLOTS).eval()
    a = _score_of_first(s, torch.zeros(2, 3, 8, 2))
    b = _score_of_first(s, torch.randn(2, 5, 8, 2) * 20)
    assert torch.allclose(a, b, atol=1e-6)


def test_control_self_attention_DOES_couple_companions():
    # the same test must be able to go RED: with DrivoR's self-attention it does
    torch.manual_seed(0)
    s = DisentangledScorer(small(scorer_self_attn=True), SRC, SLOTS).eval()
    a = _score_of_first(s, torch.zeros(2, 3, 8, 2))
    b = _score_of_first(s, torch.randn(2, 5, 8, 2) * 20)
    assert not torch.allclose(a, b, atol=1e-6)


def test_scorer_depends_on_the_condition():
    torch.manual_seed(0)
    s = DisentangledScorer(small(), SRC, SLOTS).eval()
    traj = torch.zeros(2, 3, 8, 2)
    with torch.no_grad():
        c0 = s(sources(), traj, torch.ones(2), torch.zeros(2, 5))["nc"]
        c1 = s(sources(), traj, torch.ones(2), torch.ones(2, 5))["nc"]
    assert not torch.allclose(c0, c1)


def test_scorer_loss_is_masked():
    lg = {"nc": torch.tensor([[10.0, -10.0]]), "dac": torch.tensor([[0.0, 0.0]])}
    orc = {"nc": torch.tensor([[1.0, 0.0]]), "nc_mask": torch.tensor([[True, True]]),
           "dac": torch.tensor([[1.0, 1.0]]), "dac_mask": torch.tensor([[False, False]])}
    loss, log = scorer_loss(lg, orc)
    assert float(loss) < 1e-3                 # nc is right; dac is fully masked
    assert log["dac"] != log["dac"]           # NaN: reported as "not scored", not as 0


def test_padded_tokens_are_ignored():
    torch.manual_seed(0)
    s = DisentangledScorer(small(), SRC, SLOTS).eval()
    src = sources()
    traj = torch.zeros(2, 3, 8, 2)
    pad = torch.zeros(2, 6, dtype=torch.bool); pad[:, 3:] = True
    with torch.no_grad():
        a = s(src, traj, torch.ones(2), torch.zeros(2, 5), pads={"bev": pad})["nc"]
        src2 = dict(src); src2["bev"] = src["bev"].clone(); src2["bev"][:, 3:] = 99.0
        b = s(src2, traj, torch.ones(2), torch.zeros(2, 5), pads={"bev": pad})["nc"]
        c = s(src2, traj, torch.ones(2), torch.zeros(2, 5))["nc"]
    assert torch.allclose(a, b, atol=1e-5)      # changing PADDED content changes nothing
    assert not torch.allclose(a, c, atol=1e-5)  # control: unmasked, the same change moves it
