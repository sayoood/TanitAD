"""HiCAP architecture tests (CPU, synthetic data). They check mechanisms and invariants, not driving quality."""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from tanitad.hicap import vocab as V, prune, losses as L
from tanitad.hicap.model import HiCAP, HiCAPConfig, LEVELS
from tanitad.hicap.decode import decide, CascadeCfg

SMALL = dict(d=64, layers=2, heads=4, d_v=48, n_vlm_tokens=8, n_motion_tokens=16, d_text=32, d_audio=24, rerank_layers=1)


def synth_world(n=300, seed=0):
    rng = np.random.default_rng(seed)
    paths, dvs, lat, lon, v0s = [], [], [], [], []
    for i in range(n):
        v0 = rng.choice([0.0, 3.0, 8.0, 15.0]); c = rng.integers(0, 3)
        curv = {0: 0.0, 1: 0.03, 2: -0.03}[c] + rng.normal(0, 0.003)
        a = rng.choice([0.0, -3.0, 1.5, -1.0]); l = {0.0: 6, -3.0: 4, 1.5: 5, -1.0: 3}[a] if v0 > 0 else rng.choice([0, 1, 2])
        t = (np.arange(60) + 1) * 0.1
        v = np.maximum(v0 + a * t, 0)
        if abs(curv) * v.max() ** 2 > 7.0: curv = np.sign(curv) * 7.0 / v.max() ** 2      # keep the synthetic world physically feasible
        th = np.cumsum(v * curv * 0.1)
        tr = np.stack([np.cumsum(v * np.cos(th) * 0.1), np.cumsum(v * np.sin(th) * 0.1)], 1)
        p, dv = V.extract_path_profile(tr, v0)
        paths.append(p); dvs.append(dv); lat.append(c); lon.append(l); v0s.append(v0)
    return np.array(paths), np.array(lat), np.array(dvs), np.array(lon), np.array(v0s)


@pytest.fixture(scope="module")
def world():
    return synth_world()


@pytest.fixture(scope="module")
def vocab(world):
    p, lat, dv, lon, v0 = world
    return V.build_factored_vocab(p, lat, dv, lon, v0, K_P=48, K_V=24, floor_P=4, floor_V=2)


def test_compose_roundtrip_straight_constant_speed():
    tr = np.stack([np.arange(1, 61) * 1.0, np.zeros(60)], 1)               # 10 m/s straight
    p, dv = V.extract_path_profile(tr, 10.0)
    out = V.compose(p, dv, 10.0)
    assert np.abs(out - tr).max() < 1e-3


def test_compose_roundtrip_curve_and_braking(world):
    p, lat, dv, lon, v0 = world
    i = int(np.where((lon == 4) & (lat == 1))[0][0])
    tr = V.compose(p[i], dv[i], v0[i])
    p2, dv2 = V.extract_path_profile(tr, v0[i])
    assert np.abs(V.compose(p2, dv2, v0[i]) - tr).max() < 0.3          # 1 m path grid: double-quantisation, < 1% of path length


def test_vocab_classes_and_standstill(vocab):
    assert set(np.unique(vocab.path_cls)) <= {0, 1, 2}
    assert vocab.K_P >= 3 * 4 and vocab.standstill.any()
    assert (vocab.paths.shape[1] == len(V.S_GRID)) and vocab.profiles.shape[1] == V.T_STEPS


def test_pair_admissibility_mask():
    j = np.zeros((3, 7)); j[0, :] = 5; j[1, 6] = 3
    b = V.pair_admissibility(j)
    assert b[0, 0] == 0 and np.isinf(b[2, 3]) and b[1, 6] == 0


def make_batch(B=4, c=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = c or HiCAPConfig(**SMALL)
    return dict(vlm=torch.randn(B, 3, c.n_vlm_tokens, c.d_v, generator=g), last=torch.randn(B, c.d_v, generator=g),
                age_s=torch.rand(B, generator=g), frames=torch.randn(B, 3, 3, 64, 160, generator=g),
                ego=torch.randn(B, 3, generator=g), nav=torch.randint(0, 3, (B,), generator=g), nav_args=torch.randn(B, 4, generator=g),
                route=torch.randn(B, 4, generator=g))


def test_default_param_budget():
    n = HiCAP().n_params()
    assert 50e6 < n < 62e6, n          # MEASURED code count at the defaults: 56.4 M (paper's earlier 27 M estimate omitted projector, towers, re-ranker)
    bd = HiCAP().breakdown()
    assert abs(sum(bd.values()) - n) < 1


def test_forward_shapes_and_unit_norm():
    m = HiCAP(HiCAPConfig(**SMALL)); b = make_batch()
    s, g, gp, scene = m.encode(b)
    assert set(s) == {k for k in LEVELS if k != "goal"} and g.shape == (4, 22) and gp.shape == (4, 3, 3)
    for v in s.values(): assert torch.allclose(v.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_matryoshka_widths_unit_norm():
    m = HiCAP(HiCAPConfig(**SMALL)); b = make_batch()
    for w in (16, 32, 64):
        s, *_ = m.encode(b, width=w)
        assert s["s2lon"].shape[-1] == w and torch.allclose(s["s2lon"].norm(dim=-1), torch.ones(4), atol=1e-5)


def test_audio_null_is_bit_exact_and_gate_opens():
    m = HiCAP(HiCAPConfig(**SMALL)).eval(); b = make_batch()
    s0, *_ = m.encode(b)
    b2 = dict(b); b2["audio"] = torch.randn(4, 8, 24)
    s1, *_ = m.encode(b2)
    for k in s0: assert torch.equal(s0[k], s1[k])                # g = 0  =>  audio changes nothing, bit for bit
    with torch.no_grad(): m.fusion.audio.g.fill_(0.5)
    s2, *_ = m.encode(b2)
    assert not torch.equal(s0["s2lon"], s2["s2lon"])


def test_ego_has_no_curvature_channel():
    m = HiCAP(HiCAPConfig(**SMALL))
    assert m.fusion.ego.k == 3                                    # v, a_lon, yaw rate only (PI ruling 2026-10-25)


def test_null_rows_change_output():
    m = HiCAP(HiCAPConfig(**SMALL)).eval(); b = make_batch()
    s0, *_ = m.encode(b)
    b2 = dict(b); b2["route_present"] = torch.zeros(4)
    with torch.no_grad(): m.fusion.route.null.add_(torch.randn_like(m.fusion.route.null))
    s1, *_ = m.encode(b2)
    assert not torch.equal(s0["s3P"], s1["s3P"])


def test_soft_target_and_ancestor_sum():
    dist = torch.tensor([[0.0, 0.5, 5.0, 5.0]])
    y = L.soft_target(dist)
    assert abs(float(y.sum()) - 1) < 1e-6 and y[0, 0] > y[0, 1] > y[0, 2]
    parent = L.subtree_sum(y, torch.tensor([0, 0, 1, 1]))
    assert abs(float(parent.sum()) - 1) < 1e-6 and parent[0, 0] > 0.9


def test_infonce_mask_changes_loss():
    s = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1); a = s.clone()
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    t = torch.tensor(20.0)
    assert float(L.infonce_bidirectional(s, a, t, ids)) < float(L.infonce_bidirectional(s, a, t, None)) + 1e-6


def build_vt(c, vocab):
    rng = np.random.default_rng(1)
    pf = np.stack([V.path_features(p) for p in vocab.paths]); qf = np.stack([V.profile_features(d) for d in vocab.profiles])
    return L.VocabTensors(rng.normal(size=(7, c.d_text)), rng.normal(size=(7, 4)), rng.normal(size=(3, c.d_text)), rng.normal(size=(3, 4)),
                          rng.normal(size=(7, c.d_text)), rng.normal(size=(7, 4)), pf, vocab.path_cls, qf, vocab.prof_cls,
                          strat_mask=np.array([1, 1, 1, 1, 1, 0, 0], bool), cell_bias=V.pair_admissibility(np.ones((3, 7))))


def label_batch(c, vocab, B=4):
    b = make_batch(B, c)
    b.update(strat_y=torch.tensor([0, 1, -100, 3]), lat_y=torch.tensor([0, 1, 2, -100]), lon_y=torch.tensor([6, 4, 5, 3]),
             yP=torch.softmax(torch.randn(B, vocab.K_P), -1), yQ=torch.softmax(torch.randn(B, vocab.K_V), -1),
             goal_y=torch.randint(0, 2, (B, 22)).float(), goal_w=torch.ones(B, 22), goal_pt=torch.randn(B, 3, 3), goal_pt_w=torch.ones(B, 3),
             cand_p=torch.randint(0, vocab.K_P, (B, 5)), cand_q=torch.randint(0, vocab.K_V, (B, 5)),
             cost_t=torch.randn(B, 5, 3), refine_t=torch.randn(B, 5, 16))
    return b


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_loss_all_stages_finite_and_grads(vocab, stage):
    c = HiCAPConfig(**SMALL); m = HiCAP(c); Vt = build_vt(c, vocab); crit = L.HiCAPLoss()
    tot, parts = crit(m, Vt, label_batch(c, vocab), stage=stage)
    assert torch.isfinite(tot)
    tot.backward()
    assert m.fusion.queries.grad is not None and m.theta["s3P"].grad is not None


def test_overfit_tiny_batch(vocab):
    torch.manual_seed(0)
    c = HiCAPConfig(**SMALL); m = HiCAP(c); Vt = build_vt(c, vocab); crit = L.HiCAPLoss()
    b = label_batch(c, vocab); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0)
    l0 = None
    for i in range(60):
        opt.zero_grad(); tot, _ = crit(m, Vt, b, stage=1); tot.backward(); opt.step()
        l0 = float(tot) if l0 is None else l0
    assert float(tot) < 0.6 * l0


def test_lr_rule():
    assert abs(L.lr_rule(1024, 1024) - 2e-3) < 1e-12 and L.lr_rule(1024, 2048) > 2e-3


def test_kinematic_masks_keep_logged_future(world, vocab):
    p, lat, dv, lon, v0 = world
    kept = 0; n = 0
    for i in range(0, len(p), 3):
        tr = V.compose(p[i], dv[i], v0[i]); n += 1
        a0 = float(dv[i][0] - 0.0) / V.DT
        kept += prune.pair_feasible(tr)
    assert kept / n > 0.95                                        # physical limits never remove a logged (feasible) future


def test_guard_set_never_empty_and_shapes():
    g = prune.guard_set(10.0, 0.05, None)
    assert g.shape == (4, 60, 2) and g[1, -1, 0] < g[0, -1, 0]    # brake stops short of hold
    assert prune.guard_set(10.0, 0.0, g[0]).shape[0] == 5


def test_conformal_class_conditional_coverage():
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 3, 3000); sc = rng.normal(cls * 0.5, 1.0)
    cal, te = slice(0, 1500), slice(1500, None)
    cc = prune.ClassConditionalConformal.fit(sc[cal], cls[cal], alpha=0.1)
    for c in range(3):
        mk = cls[te] == c
        cov = np.mean(sc[te][mk] >= -cc.q[c])
        assert cov > 0.85


def test_episode_blocked_coverage():
    cov = np.array([1, 1, 0, 1, 1, 1], bool); ep = np.array([0, 0, 0, 1, 1, 1])
    r = prune.episode_blocked_coverage(cov, ep)
    assert r["n_episodes"] == 2 and abs(r["pooled"] - 5 / 6) < 1e-9


def test_end_to_end_decide(vocab):
    c = HiCAPConfig(**SMALL); m = HiCAP(c); Vt = build_vt(c, vocab)
    b = make_batch(1, c)
    out = decide(m, Vt, vocab, b, v0=8.0, a0=0.0, yaw=0.0, cfg=CascadeCfg(k_p=6, k_v=4, shortlist=12, deploy_k=8))
    assert out["slots"].shape == (8, 2) and out["traj"].shape == (60, 2) and np.isfinite(out["slots"]).all()
    assert out["n_composed"] >= 4                                   # guards are always present
