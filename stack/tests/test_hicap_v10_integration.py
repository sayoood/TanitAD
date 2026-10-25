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
