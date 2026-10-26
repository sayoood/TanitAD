"""Stream R: RuleBook schema + append-only versioning, retrieval specificity, country-swap counterfactual, gate=0 bit-exactness,
country readout learnability, guard-set survival, and the frozen country table."""
import copy, json
import numpy as np, pytest
torch = pytest.importorskip("torch")
from tanitad.hicap import rules as RU, vocab as V, prune as P
from tanitad.hicap.knowledge import hashed_text_embedding
from tanitad.hicap.model import HiCAP, HiCAPConfig, LEVELS
from test_hicap_architecture import SMALL, make_batch, synth_world


@pytest.fixture(scope="module")
def book():
    return RU.RuleBook.load()


@pytest.fixture(scope="module")
def vocab():
    p, lat, dv, lon, v0 = synth_world()
    return V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2)


def onehot(c):
    p = np.zeros(len(RU.COUNTRIES)); p[RU.COUNTRY_ID[c]] = 1.0; return p


# ---------------------------------------------------------------- schema + content
def test_every_record_valid_and_sourced(book):
    raw = json.loads(RU.RULEBOOK_PATH.read_text(encoding="utf-8"))
    assert tuple(raw["countries"]) == RU.COUNTRIES and raw["version"] == 1
    for r in raw["records"]:
        assert RU.validate_record(r) == [], r["rule_id"]
        assert r["evidence"] in ("PUBLISHED", "UNVERIFIED") and r["source"]["url"].startswith("https://")
    ids = [r["rule_id"] for r in raw["records"]]
    assert len(ids) == len(set(ids))


def test_coverage_every_country(book):
    M = book.membership()
    assert M.shape == (25, len(book)) and (M.sum(1) >= 8).all()           # >= top-k rules per country
    for c in RU.COUNTRIES:
        cats = {r["category"] for r in book.rules_for(c)}
        assert {"speed_limit", "right_turn_on_red", "priority_to_right", "roundabout", "emergency_vehicle", "driving_side"} <= cats, c
        assert book.speed_rule(c, "urban") is not None and book.rtor_mode(c) is not None
    assert book.rtor_mode("United States") == "permitted_after_stop_unless_signed"
    assert book.rtor_mode("Germany") == "green_arrow_sign_after_stop" and book.rtor_mode("France") == "prohibited"
    us = book.speed_rule("United States", "motorway")
    assert us["args"]["v_default_kmh"] is None and "STATE-LEVEL" in us["caveat"] and us["effect"] != "prune"
    assert book.speed_rule("Germany", "motorway")["args"]["v_default_kmh"] is None
    assert any(r["category"] == "school_bus" for r in book.rules_for("United States"))
    assert not any(r["category"] == "school_bus" for r in book.rules_for("Germany"))


def test_validator_rejects_bad_records(book):
    r = copy.deepcopy(book.records[0])
    bad = [dict(r, effect="maybe"), dict(r, countries=["Atlantis"]), dict(r, evidence="INHERITED"),
           dict(r, args=dict(v_default_kmh=900)), dict(r, source=dict(title="x", url="http://x")),
           {k: v for k, v in r.items() if k != "source"}, dict(r, args=dict(v_default_kmh=None), effect="prune")]
    for b in bad: assert RU.validate_record(b), b


def test_append_only_versioning(book):
    new = dict(rule_id="TEST-NEW", countries=["Austria"], category="tram", road_type="any", effect="memory",
               text="A test rule appended to the rule book.", args={}, source=dict(title="t", url="https://example.org"),
               evidence="UNVERIFIED", verification="test")
    b2 = book.add_rule(new)
    assert b2.version == book.version + 1 and len(b2) == len(book) + 1
    assert b2.ids[: len(book)] == book.ids and b2.index("TEST-NEW") == len(book)   # existing ids never move
    assert b2.records[-1]["version_added"] == book.version + 1 and len(book) == len(b2) - 1   # original untouched
    with pytest.raises(ValueError): b2.add_rule(new)                                      # duplicate id refused
    with pytest.raises(ValueError): book.add_rule(dict(new, rule_id="TEST-BAD", effect="nope"))


def test_country_table_frozen_and_nodes(book):
    assert list(RU.COUNTRIES) == sorted(RU.COUNTRIES) and len(RU.COUNTRIES) == 25
    nodes = book.country_nodes()
    assert len(nodes) == 25 and nodes[RU.COUNTRY_ID["Germany"]].startswith("driving in Germany: ")
    assert "green-arrow sign" in nodes[RU.COUNTRY_ID["Germany"]] and "after stopping unless signed" in nodes[RU.COUNTRY_ID["United States"]]
    with pytest.raises(ValueError): RU.RuleBook(book.records, countries=tuple(reversed(RU.COUNTRIES)))


def test_embedding_cache(book):
    calls = []
    def emb(s): calls.append(s); return hashed_text_embedding(s, 16)
    e1 = book.embed_rules(emb, key="h16"); n = len(calls); e2 = book.embed_rules(emb, key="h16")
    assert e1.shape == (len(book), 16) and e2 is e1 and len(calls) == n                   # cached, embedded once per version
    assert book.embed_countries(emb, key="h16").shape == (25, 16)


# ---------------------------------------------------------------- retrieval
def _mem(book, d=32, d_rule=16, k=8, seed=0):
    torch.manual_seed(seed)
    R = torch.from_numpy(book.embed_rules(lambda s: hashed_text_embedding(s, d_rule), key=f"h{d_rule}"))
    return RU.RuleMemory(d, d_rule, book.membership(), k=k), R


def test_retrieval_only_dominant_country(book):
    mem, R = _mem(book)
    q = torch.randn(len(RU.COUNTRIES), 32)
    P_ = torch.eye(len(RU.COUNTRIES))
    out = mem(q, P_, R)
    assert out["idx"].shape == (25, 8) and out["tokens"].shape == (25, 8, 16)
    M = book.membership()
    for ci, c in enumerate(RU.COUNTRIES):
        idx = out["idx"][ci][out["valid"][ci]].numpy()
        assert len(idx) == 8 and M[ci, idx].all(), c                                        # every retrieved rule applies in c
        assert (out["score"][ci][out["valid"][ci]] > 0).all()


def test_retrieval_mixture_and_gradient(book):
    mem, R = _mem(book)
    p = torch.zeros(1, 25); p[0, RU.COUNTRY_ID["Germany"]] = 0.5; p[0, RU.COUNTRY_ID["United States"]] = 0.5
    out = mem(torch.randn(1, 32), p, R)
    M = book.membership(); idx = out["idx"][0].numpy()
    assert (M[RU.COUNTRY_ID["Germany"], idx] | M[RU.COUNTRY_ID["United States"], idx]).all()
    gm = RU.GatedMemory(32, 16, heads=4)
    with torch.no_grad(): gm.g.fill_(1.0)
    q = torch.randn(1, len(LEVELS), 32, requires_grad=True)
    gm(q, mem(q[:, LEVELS.index("s2lat")], p, R)).sum().backward()
    assert mem.wq.weight.grad is not None and mem.wq.weight.grad.abs().sum() > 0           # relevance projections are trained


# ---------------------------------------------------------------- gated memory: bit-exact no-ops, goal untouched
def test_gate_zero_and_null_bit_exact(book):
    mem, R = _mem(book)
    gm = RU.GatedMemory(32, 16, heads=4)
    q = torch.randn(3, len(LEVELS), 32)
    rules = mem(q[:, 1], torch.softmax(torch.randn(3, 25), -1), R)
    assert torch.equal(gm(q, rules), q)                                                     # g = 0 at init -> bit-exact
    assert gm(q, None) is q                                                                 # rules-null -> identity
    with torch.no_grad(): gm.g.fill_(2.0)
    out = gm(q, rules)
    assert not torch.equal(out, q)
    assert torch.equal(out[:, LEVELS.index("goal")], q[:, LEVELS.index("goal")])            # goal query never written
    assert torch.equal(out[:, LEVELS.index("s3P")], q[:, LEVELS.index("s3P")])
    with pytest.raises(ValueError): RU.GatedMemory(32, 16, heads=4, targets=("s1", "goal"))


def test_gated_memory_no_nan_without_valid_rules(book):
    gm = RU.GatedMemory(32, 16, heads=4)
    with torch.no_grad(): gm.g.fill_(1.0)
    q = torch.randn(2, len(LEVELS), 32)
    rules = dict(tokens=torch.randn(2, 8, 16), score=torch.zeros(2, 8), valid=torch.zeros(2, 8, dtype=torch.bool))
    assert torch.equal(gm(q, rules), q)


def test_gated_memory_on_real_hicap_queries_is_noop_at_init(book):
    c = HiCAPConfig(**SMALL); m = HiCAP(c).eval()
    with torch.no_grad():
        b = make_batch(2, c)
        q, _ = m.fusion(b["vlm"], b["last"], b["age_s"], b["frames"], b["ego"], b["nav"], b["nav_args"], b["route"])
        R = torch.from_numpy(book.embed_rules(lambda s: hashed_text_embedding(s, 16), key="h16"))
        mem = RU.RuleMemory(c.d, 16, book.membership()); gm = RU.GatedMemory(c.d, 16, c.heads)
        out = gm(q, mem(q[:, LEVELS.index("s2lat")], torch.softmax(torch.randn(2, 25), -1), R))
    assert torch.equal(out, q)


# ---------------------------------------------------------------- country readout
def test_country_readout_learns_toy_mapping(book):
    torch.manual_seed(0)
    T = torch.from_numpy(book.embed_countries(lambda s: hashed_text_embedding(s, 32), key="h32"))
    cr = RU.CountryReadout(16, 32)
    proto = torch.randn(25, 16)
    y = torch.arange(25).repeat(8); x = proto[y] + 0.1 * torch.randn(len(y), 16)
    opt = torch.optim.Adam(cr.parameters(), 3e-3)
    for _ in range(300):
        opt.zero_grad(); l = cr.loss(cr(x, T), y); l.backward(); opt.step()
    acc = (cr(x, T).argmax(-1) == y).float().mean().item()
    assert acc > 0.95, acc
    assert abs(float(cr.tau().detach()) - float(torch.exp(cr.theta.detach().clamp(max=np.log(100.0))))) < 1e-6
    assert torch.isfinite(cr.loss(cr(x[:2], T), torch.tensor([-100, -100])))                 # all-IGNORE batch is finite
    p = cr.posterior(x[:3], T, supplied=torch.tensor([0, 5, 24]))                            # SUPPLIED (GNSS) arm = one-hot
    assert torch.equal(p.argmax(-1), torch.tensor([0, 5, 24])) and torch.allclose(p.sum(-1), torch.ones(3))


# ---------------------------------------------------------------- prune route
def test_rule_masks_noop_when_context_unknown(book, vocab):
    r1 = P.profile_mask(vocab, 8.0)
    m = RU.rule_masks(book, vocab, onehot("Germany"), 8.0, r1)
    assert np.array_equal(m["profile"], r1) and m["cell"].all() and not m["info"]["speed_active"]


def test_speed_mask_subset_of_r1_and_respects_limit(book, vocab):
    v0 = 13.0                                                                                # 46.8 km/h, urban 50 -> cap 55 km/h
    r1 = P.profile_mask(vocab, v0)
    m = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, road_type="urban")
    assert m["info"]["speed_active"] and not (m["profile"] & ~r1).any()                       # never resurrects an R1-removed profile
    vmax = (v0 + vocab.profiles).max(1)
    assert (vmax[m["profile"]] <= 50 * 1.1 / 3.6 + 0.05 + 1e-9).all() or m["info"]["fallback"]
    free = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, road_type="motorway")       # no general limit -> no prune
    assert np.array_equal(free["profile"], r1)
    us = RU.rule_masks(book, vocab, onehot("United States"), v0, r1, road_type="urban")       # state-level range -> no prune
    assert np.array_equal(us["profile"], r1)


def test_speed_mask_never_demands_impossible_brake(book, vocab):
    v0 = 25.0                                                                                # 90 km/h in a 50 zone
    r1 = np.ones(vocab.K_V, bool)
    ok = RU.speed_allowed(vocab, v0, 50.0)
    vmax = (v0 + vocab.profiles).max(1)
    assert np.array_equal(ok, vmax <= v0 + 0.05)                                             # non-accelerating profiles survive
    m = RU.rule_masks(book, vocab, onehot("Austria"), v0, r1, road_type="urban")
    assert m["profile"].any()


def test_country_swap_counterfactual_rtor(book, vocab):
    v0 = 0.0                                                                                 # stopped at a red light
    r1 = P.profile_mask(vocab, v0)
    us = RU.rule_masks(book, vocab, onehot("United States"), v0, r1, signal="red")
    de = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, signal="red")
    de_arrow = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, signal="red", green_arrow=True)
    tr = RU.LAT_TURN_R
    assert us["cell"].all()                                                                  # US: right-on-red candidate allowed
    assert not de["cell"][tr, V.LON_V9.index("CREEP")] and de["cell"][tr, V.LON_V9.index("HOLD")]   # DE: masked (no arrow)
    assert de_arrow["cell"].all()                                                            # DE with green-arrow sign: allowed
    diff = us["cell"] != de["cell"]
    assert diff.any() and (np.where(diff)[0] == tr).all()                                    # swap changes ONLY rule-relevant cells
    assert np.array_equal(us["profile"], de["profile"])                                      # and no profile without a road type
    moving = RU.rule_masks(book, vocab, onehot("United States"), 6.0, None, signal="red")     # US must stop first
    assert not moving["cell"][tr, V.LON_V9.index("KEEP")] and moving["cell"][tr, V.LON_V9.index("STOP")]
    green = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, signal="green")
    assert green["cell"].all()


def test_country_swap_speed_only_changes_speed_relevant_profiles(book, vocab):
    v0 = 22.0                                                                                # rural road: AT 100 vs SE 70
    r1 = np.ones(vocab.K_V, bool)
    at = RU.rule_masks(book, vocab, onehot("Austria"), v0, r1, road_type="rural")
    se = RU.rule_masks(book, vocab, onehot("Sweden"), v0, r1, road_type="rural")
    diff = at["profile"] != se["profile"]
    vmax = (v0 + vocab.profiles).max(1)
    assert (vmax[diff] > 70 * 1.1 / 3.6).all()                                               # only profiles above the lower cap differ
    assert np.array_equal(at["cell"], se["cell"])


def test_posterior_vote(book, vocab):
    p = np.zeros(25); p[RU.COUNTRY_ID["United States"]] = 0.7; p[RU.COUNTRY_ID["Germany"]] = 0.3
    m = RU.rule_masks(book, vocab, p, 0.0, None, signal="red")
    assert m["cell"].all()                                                                   # 0.3 prohibiting mass < vote 0.5
    p2 = np.zeros(25); p2[RU.COUNTRY_ID["United States"]] = 0.3; p2[RU.COUNTRY_ID["Germany"]] = 0.7
    assert not RU.rule_masks(book, vocab, p2, 0.0, None, signal="red")["cell"].all()


def test_guard_set_always_survives_rule_masks(book, vocab):
    v0, yaw = 20.0, 0.0
    m = RU.rule_masks(book, vocab, onehot("France"), v0, P.profile_mask(vocab, v0), road_type="urban", signal="red")
    guards = P.guard_set(v0, yaw, prev_traj=np.cumsum(np.full((60, 2), [2.0, 0.0]), 0))
    n_open = 30
    rng = np.random.default_rng(0)
    prof = np.concatenate([rng.integers(0, vocab.K_V, n_open), -np.ones(len(guards), int)])
    lat = np.concatenate([np.full(n_open, RU.LAT_TURN_R), np.zeros(len(guards), int)])
    lon = np.concatenate([rng.integers(0, 7, n_open), np.zeros(len(guards), int)])
    is_g = np.concatenate([np.zeros(n_open, bool), np.ones(len(guards), bool)])
    keep = RU.candidate_keep(m, prof, lat, lon, is_g)
    assert keep[is_g].all() and not keep[~is_g].all()                                        # guards always, some open removed
    all_off = dict(profile=np.zeros(vocab.K_V, bool), cell=np.zeros((3, 7), bool))
    assert RU.candidate_keep(all_off, prof, lat, lon, is_g)[is_g].all()


def test_empty_after_rules_falls_back_to_r1(book, vocab):
    v0 = 13.0
    r1 = np.zeros(vocab.K_V, bool); vmax = (v0 + vocab.profiles).max(1); r1[np.argmax(vmax)] = True   # only the fastest survives R1
    m = RU.rule_masks(book, vocab, onehot("Germany"), v0, r1, road_type="urban", tol=0.0)
    if vmax[np.argmax(vmax)] > 50 / 3.6 + 0.05:
        assert m["info"]["fallback"] and np.array_equal(m["profile"], r1)


def test_v10_country_labels_match_frozen_table():
    """The v10 country release (built by the orchestrator) must use exactly the frozen COUNTRIES ids; synthetic check otherwise."""
    from pathlib import Path
    pq = Path(__file__).resolve().parents[2] / "DataEng" / "v10" / "country" / "v10_country.parquet"
    if not pq.exists(): pytest.skip(f"{pq} not built yet")
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(pq)
    assert {"split", "sha12", "clip_id", "country", "country_id"} <= set(d.columns)
    assert set(d["country"]) <= set(RU.COUNTRIES)
    assert (d["country_id"].astype(int) == d["country"].map(RU.COUNTRY_ID)).all()
