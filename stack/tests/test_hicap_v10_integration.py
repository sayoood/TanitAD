"""v10 knowledge wired into HiCAPFull: null is bit-exact, gate 0 is bit-exact, the goal query is never written, the country loss
enters FullLoss, the supplied (GNSS) arm is a one-hot, and the v10 country labels load with rules.COUNTRIES ids."""
import os
import numpy as np
import pytest
import torch
from tanitad.hicap.full import HiCAPFull, FullLoss
from tanitad.hicap.knowledge import hashed_text_embedding
from tanitad.hicap.model import LEVELS
from tanitad.hicap import rules as RU
from tanitad.hicap.model import HiCAPConfig
from tanitad.hicap import vocab as V
from test_hicap_architecture import SMALL, label_batch, build_vt, synth_world


def _emb(texts, d): return np.stack([np.resize(hashed_text_embedding(t, 64), d) for t in texts]).astype(np.float32)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    c = HiCAPConfig(**SMALL); p, lat, dv, lon, v0 = synth_world()
    vocab = V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2); m = HiCAPFull(c).eval()
    return c, vocab, m


def _knowledge(m, c):
    return _emb(m.book.texts(), c.d_text), _emb(m.book.country_nodes(), c.d_text)


def test_knowledge_null_and_gate_zero_bit_exact(setup):
    c, vocab, m = setup
    b = label_batch(c, vocab, 3)
    with torch.no_grad():
        s0, g0, *_ = m.encode(b)
        m.set_knowledge(*_knowledge(m, c))
        s1, g1, _, _, k = m.encode_k(b)
    assert k and k["rules"]["idx"].shape == (3, 8)
    for kk in s0: assert torch.equal(s0[kk], s1[kk]), kk           # gate g = 0 at init: memory adds exactly 0
    assert torch.equal(g0, g1)
    m.rule_emb = torch.zeros(0, c.d_text)                            # back to null for other tests


def test_memory_never_writes_goal_and_moves_s1_s2(setup):
    c, vocab, m = setup
    m.set_knowledge(*_knowledge(m, c))
    b = label_batch(c, vocab, 3)
    with torch.no_grad():
        s0, g0, *_ = m.encode(b)
        m.gated_memory.g.fill_(1.0)
        s1, g1, *_ = m.encode(b)
        m.gated_memory.g.zero_()
    assert torch.equal(g0, g1)                                       # goal query untouched (admissibility)
    assert not torch.equal(s0["s1"], s1["s1"]) and not torch.equal(s0["s2lat"], s1["s2lat"])
    assert torch.equal(s0["s3P"], s1["s3P"])                         # s3 queries are not memory targets
    m.rule_emb = torch.zeros(0, c.d_text)


def test_country_loss_in_fullloss_and_supplied_arm(setup):
    c, vocab, m0 = setup
    m = HiCAPFull(c); m.set_knowledge(*_knowledge(m, c))
    V = build_vt(c, vocab)
    b = label_batch(c, vocab, 4); b["country_y"] = torch.tensor([0, 9, 24, -100])
    total, parts = FullLoss()(m, V, b, stage=1)
    assert "country" in parts and torch.isfinite(parts["country"])
    total.backward()
    assert m.country_readout.h[0].weight.grad.abs().sum() > 0
    b["country_supplied"] = torch.tensor([9, 9, 9, 9])               # GNSS arm: Germany one-hot, retrieval keyed on it
    with torch.no_grad(): *_, k = m.encode_k(b)
    de = set(np.flatnonzero(m.book.membership()[RU.COUNTRIES.index("Germany")]))
    got = k["rules"]["idx"][k["rules"]["valid"]].tolist()
    assert got and set(got) <= de


@pytest.mark.skipif(not os.path.exists("DataEng/v10/country/v10_country.parquet"), reason="v10 country file absent")
def test_v10_country_labels_load():
    from tanitad.hicap.data import load_country
    import pandas as pd
    m = load_country("DataEng/v10/country/v10_country.parquet")
    d = pd.read_parquet("DataEng/v10/country/v10_country.parquet")
    assert len(m) == 4508 and all(RU.COUNTRIES[d.country_id[i]] == d.country[i] for i in range(len(d)))


def test_audio_bridge_wired_present_mask_and_null():
    from tanitad.hicap.audio import BridgeConfig
    c = HiCAPConfig(**{**SMALL, "d_audio": SMALL.get("d_v", 2048)})
    with pytest.raises(ValueError): HiCAPFull(HiCAPConfig(**{**SMALL, "d_audio": c.d_v + 1}), bridge=BridgeConfig(d_v=c.d_v))
    torch.manual_seed(0)
    m = HiCAPFull(c, bridge=BridgeConfig(d_enc=64, d_v=c.d_v, width=64, heads=4)).eval()
    p, lat, dv, lon, v0 = synth_world(); vocab = V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2)
    b = label_batch(c, vocab, 3)
    with torch.no_grad():
        s0, *_ = m.encode(b)
        b2 = dict(b, audio_feat=torch.randn(3, 25, 64))
        s1, *_ = m.encode(b2)                                       # gate 0: bridge tokens change nothing
        for k in s0: assert torch.equal(s0[k], s1[k]), k
        m.fusion.audio.g.fill_(1.0)
        b3 = dict(b2, audio_present=torch.tensor([1.0, 0.0, 1.0]))
        s2, *_ = m.encode(b3)
        m.fusion.audio.g.zero_()
    assert torch.equal(s0["s1"][1], s2["s1"][1])                    # sample without audio: untouched in a mixed batch
    assert not torch.equal(s0["s1"][0], s2["s1"][0])                # sample with audio: moved once the gate opens
    assert len(m.audio_params()) > 0


@pytest.mark.skipif(not os.path.exists("DataEng/v10/imag/v10_imag_eval139.npz"), reason="imag sample absent")
def test_imag_rows_feed_fullloss_and_masks():
    from tanitad.hicap.data import ImagRows, collate
    from tanitad.hicap.imagination import EHead, exo_loss
    R = ImagRows("DataEng/v10/imag/v10_imag_eval139.npz")
    sha = next(iter(R.row0)); items = [R.item(sha, j) for j in range(0, 40, 5)] + [R.item("000000000000", 0)]
    b = collate(items)
    assert b["exo_w"][-1] == 0 and b["exo_mode"][-1] == -100          # clip absent -> unsupervised, never 'no lead'
    out = EHead(64)(torch.randn(len(items), 64))
    tgt = dict(gap=b["exo_gap"], closing=b["exo_close"], lacc=b["exo_acc"], mode=b["exo_mode"], onset=b["exo_onset"],
               has_lead=b["exo_has_lead"], w=b["exo_w"], hmask=b["exo_hmask"])
    L = exo_loss(out, tgt); assert torch.isfinite(L); L.backward()


def test_ehead_gap_is_linear():
    from tanitad.hicap.imagination import EHead
    h = EHead(8)
    with torch.no_grad():
        h.m.net[-1].bias.fill_(-1.0); h.m.net[-1].weight.zero_()
    assert (h(torch.randn(2, 8))["gap"] < 0).all()                     # negative gap targets are representable (stream S: up to 3.7 %)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_clm_pure_arm_trains_with_hicap_losses(stage):
    from tanitad.hicap.clm_pure import CLMPure
    from tanitad.hicap.losses import HiCAPLoss
    c = HiCAPConfig(**SMALL); torch.manual_seed(0)
    p, lat, dv, lon, v0 = synth_world(); vocab = V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2)
    m = CLMPure(c, width=32); b = label_batch(c, vocab, 4)
    total, parts = HiCAPLoss()(m, build_vt(c, vocab), b, stage=stage)
    assert torch.isfinite(total); total.backward()
    assert not hasattr(m, "fusion")
    b2 = dict(b, frames=torch.zeros_like(b["frames"]), ego=torch.zeros_like(b["ego"]), vlm=torch.zeros_like(b["vlm"]))
    with torch.no_grad():                                                  # reads ONLY the last-token state
        s1, *_ = m.encode(b); s2, *_ = m.encode(b2)
    for k in s1: assert torch.equal(s1[k], s2[k])
