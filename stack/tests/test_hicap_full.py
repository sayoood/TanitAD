"""Imagination (Tiers 0-2), elastic sensing, knowledge injection, verifier, distillation, strategic supply, full loss + decision."""
import numpy as np, pytest, gzip, json, hashlib
torch = pytest.importorskip("torch")
from tanitad.hicap import imagination as I, elastic as E, knowledge as K, distill as D, strategic as S, vocab as V, losses as L
from tanitad.hicap.full import HiCAPFull, FullLoss
from tanitad.hicap.model import HiCAP, HiCAPConfig
from tanitad.hicap.decode import decide, CascadeCfg
from test_hicap_architecture import SMALL, make_batch, build_vt, synth_world
from test_hicap_data_train import setup  # noqa: F401  (fixture)


# ---------------- Tier 0
def test_tier0_target_mix_and_zero_cost_ordering():
    J = torch.tensor([[0.0, 1.0, 5.0]]); y = I.tier0_target(J, torch.tensor([2]))
    assert abs(float(y.sum()) - 1) < 1e-6 and y[0, 2] > 0.3 and y[0, 0] > y[0, 1]            # 0.3 on the expert + softmax(-J)


def test_teacher_costs_collision_and_comfort():
    t = (np.arange(1, 61) * 1.0)[:, None] * np.array([[1.0, 0.0]])                              # 10 m/s straight
    free = I.teacher_costs(t, None, 10.0); hit = I.teacher_costs(t, t[None] + np.array([0.0, 0.5]), 10.0)
    assert hit[0] > free[0] == 0.0
    brake = np.stack([np.cumsum(np.maximum(10 - 9 * (np.arange(60) + 1) * 0.1, 0) * 0.1), np.zeros(60)], 1)
    assert I.teacher_costs(brake, None, 10.0)[1] > free[1]


# ---------------- Tier 1
def _eout(gap0=20.0, has=5.0):
    o = dict(gap=torch.full((1, 4), gap0), closing=torch.zeros(1, 4), lacc=torch.zeros(1, 4), logsig=torch.zeros(1, 4),
             mode=torch.tensor([[2.0, 0.0, 0.0, 0.0]]), onset=torch.tensor([[0.5, 1.0, 2.0]]), has_lead=torch.tensor([has]))
    return o


def test_ehead_shapes_and_onset_sorted():
    o = I.EHead(32)(torch.randn(5, 32))
    assert o["gap"].shape == (5, 4) and o["mode"].shape == (5, 4) and (o["onset"][:, 1:] >= o["onset"][:, :-1]).all()


def test_tier1_cost_penalises_fast_candidate_more_than_brake():
    t = (np.arange(1, 61) * 0.1)
    fast = np.stack([15.0 * t, 0 * t], 1); slow = np.stack([4.0 * t, 0 * t], 1)
    J = I.tier1_cost(_eout(12.0), 0, np.stack([fast, slow]), v0=10.0)
    assert J[0] > J[1] >= 0
    assert I.tier1_cost(_eout(12.0, has=-9.0), 0, np.stack([fast]), 10.0)[0] == 0.0           # no lead -> zero cost


def test_exo_loss_finite_and_masked():
    o = I.EHead(32)(torch.randn(6, 32)); B = 6
    tgt = dict(gap=torch.rand(B, 4) * 30, closing=torch.randn(B, 4), lacc=torch.randn(B, 4), mode=torch.tensor([0, 1, 2, 3, -100, 0]),
               onset=torch.rand(B) * 3, has_lead=torch.ones(B), w=torch.tensor([1, 1, 1, 1, 1, 0.]))
    l = I.exo_loss(o, tgt); assert torch.isfinite(l)


# ---------------- Tier 2 + composition + guards
def test_tier2_veto_is_not_trigger_and_gate_shapes():
    t2 = I.Tier2(32); s = torch.randn(3, 32); v = torch.randn(3, 5, 32)
    c = t2(s, v); assert c.shape == (2, 3, 5, 4) and (c >= 0).all()
    assert t2.veto(c, 1e9).sum() == 0 and t2.benefit_logit(s, torch.rand(3), torch.rand(3)).shape == (3,)


def test_tier2_loss_respects_family_mask():
    c = torch.ones(2, 2, 3, 4); cf = torch.zeros(2, 3, 4); m0 = torch.zeros(2, 3, 4)
    assert float(I.tier2_loss(c, cf, m0)) == 0.0                                              # G2: no supervision -> no loss


def test_compose_scores_feasibility_after_learned_terms_and_ablation():
    base = torch.tensor([1.0, 2.0, 3.0]); J1 = torch.tensor([0.0, 0.0, 5.0])
    assert int(I.compose_scores(base, J1, None, 1.0, 0.0).argmax()) == 1
    assert int(I.compose_scores(base, J1, None, 0.0, 0.0).argmax()) == 2                       # beta=0 ablation
    s = I.compose_scores(base, None, None, 0, 0, feasible=torch.tensor([True, True, False]))
    assert int(s.argmax()) == 1                                                                 # G1 mask applied last
    assert I.confidence(base) <= 1.0


# ---------------- elastic
def test_option_tokens_and_costs():
    assert E.option_tokens(0) == 160 and E.option_tokens(E.N_OPT - 1) == 160 + 6 * 64 and E.option_cost(0) == 1.0


def test_keepalive_escalates_instantly_deescalates_slowly_and_shadows():
    ka = E.KeepAlive(); p_front = np.eye(E.N_OPT)[0]; p_all = np.eye(E.N_OPT)[E.N_OPT - 1]
    r = ka.step(p_all); assert r["option"] == E.N_OPT - 1 and r["cameras_shadow"] and 0 in r["cameras_used"]    # shadow mode first
    for _ in range(2): r = ka.step(p_all)
    assert len(r["cameras_used"]) == 7
    for i in range(2): r = ka.step(p_front); assert r["option"] == E.N_OPT - 1                 # holds 2 ticks
    r = ka.step(p_front); assert r["option"] == 0                                              # de-escalates on the 3rd


def test_gate_is_vision_only_and_loss_prefers_cheap_correct_option():
    g = E.CameraGate(32); assert g.m.net[0].in_features == 32 + E.N_OPT                          # front embedding + own previous option only
    logits = torch.zeros(4, E.N_OPT, requires_grad=True)
    Eo = torch.ones(4, E.N_OPT); Eo[:, 1] = 0.0
    E.gate_loss(logits, Eo, torch.ones(4)).backward()
    assert logits.grad[0, 1] < logits.grad[0, 0]                                                # gradient descent raises option 1


def test_dropout_curriculum_proportions():
    rng = np.random.default_rng(0); n = 6000
    k = [E.dropout_option(rng, 2, return_kind=True)[1] for _ in range(n)]
    for name, p in (("oracle", 0.40), ("oracle+1", 0.20), ("all", 0.15), ("front+rand", 0.25)): assert abs(np.mean([x == name for x in k]) - p) < 0.03
    assert all(0 <= E.dropout_option(rng, 2) < E.N_OPT for _ in range(200))


# ---------------- knowledge injection + verifier
def test_new_token_gets_meaningful_init_from_text_only():
    toks = ["LANE_KEEP", "TURN_L", "TURN_R"]
    emb = lambda s: K.hashed_text_embedding(s, 32)
    lib = K.NodeLibrary("lat", toks, np.stack([emb(K.define(t)) for t in toks]), np.zeros((3, 4)))
    lib2 = lib.add_token("PREPARE_TURN_L", emb, definition="prepare to turn left at a junction")
    assert lib2.version == 2 and lib.version == 1 and lib2.tokens[:3] == toks                    # ids never move
    c = HiCAPConfig(**{**SMALL, "d_text": 32}); tower = HiCAP(c).tower_lat
    nn_ = lib2.nearest(tower, "PREPARE_TURN_L")
    emb_sim = {t: float(emb(K.define("PREPARE_TURN_L")) @ emb(K.define(t))) for t in toks}
    assert max(emb_sim, key=emb_sim.get) == "TURN_L" and len(nn_) == 3
    with pytest.raises(ValueError): lib.add_token("TURN_L", emb)


def test_verifier_fires_only_when_uncertain_and_never_generates():
    v = K.Verifier(m_min=0.5, k2=3, lam=2.0)
    sc = torch.tensor([3.0, 1.0, 0.5, 0.2])
    out, fired = v(sc, ["a", "b", "c", "d"], lambda t: np.ones(len(t)))
    assert not fired and torch.equal(out, sc)                                                    # confident tick: untouched
    sc2 = torch.tensor([1.0, 0.9, 0.5, 0.2])
    out2, fired2 = v(sc2, ["a", "b", "c", "d"], lambda t: np.array([0.0, 1.0, 0.0][: len(t)]))
    assert fired2 and int(out2.argmax()) == 1 and v.rate == 0.5
    _, f3 = v(sc, ["a", "b", "c", "d"], lambda t: np.zeros(len(t)), disagree=True); assert f3
    assert K.render_candidate("LANE_KEEP", "DECELERATE", 6.0).startswith("lat=LANE_KEEP, lon=DECELERATE, target 6")


# ---------------- distillation (optional student)
def test_distill_share_tables_and_margin_condition():
    t = HiCAP(HiCAPConfig(**SMALL)); sc = student = HiCAP(HiCAPConfig(**{**SMALL, "layers": 1}))
    D.share_action_tables(student, t)
    assert torch.equal(student.tower_lat.g.net[0].weight, t.tower_lat.g.net[0].weight) and not student.tower_lat.g.net[0].weight.requires_grad
    b = make_batch(2); st, *_ = student.encode(b); te, *_ = t.encode(b)
    assert float(D.distill_loss(st, te)) > 0
    assert float(D.distill_loss(te, te)) < 1e-6
    sco = torch.tensor([[10.0, 0.0, 0.0], [1.0, 0.9, 0.0]])
    assert D.margin_condition(sco, 0.01, 20.0).tolist() == [True, False]


# ---------------- strategic supply
def test_strategic_rows_window_and_tokens(tmp_path):
    cid = "01acc9de-8bb9-465e-a664-3098ecd92276"; sha = hashlib.sha256(cid.encode()).hexdigest()[:12]
    p = tmp_path / "s.jsonl.gz"
    with gzip.open(p, "wt") as f: f.write(json.dumps(dict(clip_id=cid, a_str=dict(token="HOLD_MAIN_ROAD", provenance="geometry"), g_str=dict(token="FOLLOW_ROUTE"), t0_s=8.0)) + "\n")
    rec = S.load_strategic(str(p)); toks = S.token_table(rec, ("UNPOPULATED_1", "UNPOPULATED_2"))
    class V9:  # minimal v9 reader
        def __init__(s): s.index = {sha: 0}; s.z = {"lat_cls_a": np.zeros(200), "now_s": np.arange(200) * 0.1}
        def clip(s, h): return dict(rows=slice(0, 200))
        def __getitem__(s, k): return s.z[k]
    y = S.strategic_rows(V9(), rec, toks)
    assert (y[78:82] == toks.index("HOLD_MAIN_ROAD")).all() and y[0] == -100 and y[150] == -100 and (y >= 0).sum() == 41


# ---------------- full model: parameters, loss with every supervision present, decision with imagination
def test_full_param_budget_and_extra_heads():
    m = HiCAPFull(); n = m.n_params(); base = HiCAP().n_params(); kn = sum(p.numel() for p in m.knowledge_params())
    assert 1e6 < n - base - kn < 3e6, n - base - kn          # imagination + elastic heads (1.88 M)
    assert 4e6 < kn < 8e6, kn                                # v10 knowledge: country readout + rule memory + gated memory (6.1 M)


def _full_batch(c, vocab, B=4):
    from test_hicap_architecture import label_batch
    b = label_batch(c, vocab, B); K_ = 5
    b.update(exo_gap=torch.rand(B, 4) * 30, exo_close=torch.randn(B, 4), exo_acc=torch.randn(B, 4), exo_mode=torch.tensor([0, 1, 2, -100]),
             exo_onset=torch.rand(B) * 3, exo_has_lead=torch.ones(B), exo_w=torch.ones(B),
             cf_cost=torch.rand(B, K_, 4), cf_mask=torch.ones(B, K_, 4), J_teacher=torch.rand(B, K_), star=torch.zeros(B, dtype=torch.long),
             E_opt=torch.rand(B, E.N_OPT), crit=torch.ones(B), prev_opt=torch.zeros(B, dtype=torch.long), cfg_opt=torch.randint(0, E.N_OPT, (B,)))
    return b


def test_full_loss_all_terms_finite_with_grads(setup):
    c = HiCAPConfig(**SMALL); m = HiCAPFull(c); Vt = build_vt(c, setup["vocab"])
    tot, parts = FullLoss()(m, Vt, _full_batch(c, setup["vocab"]), stage=3)
    assert {"exo", "tier2", "tier0", "gate", "cfg"} <= set(parts) and torch.isfinite(tot)
    tot.backward()
    assert m.ehead.m.net[0].weight.grad is not None and m.tier2.h[0].net[0].weight.grad is not None and m.camera_gate.m.net[0].weight.grad is not None


def test_imagination_heads_do_not_move_selector_through_detach(setup):
    c = HiCAPConfig(**SMALL); m = HiCAPFull(c); Vt = build_vt(c, setup["vocab"])
    b = _full_batch(c, setup["vocab"])
    for k in ("cf_cost", "J_teacher", "E_opt"): b.pop(k, None)
    m.zero_grad(); tot, parts = FullLoss()(m, Vt, b, stage=3)
    # exo loss alone must not send gradient into the fusion transformer via the E-head
    m.zero_grad(); parts["exo"].backward(retain_graph=False)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in m.fusion.blocks.parameters())


def test_decide_with_tier1_tier2_and_verifier_fields(setup):
    c = HiCAPConfig(**SMALL); m = HiCAPFull(c); Vt = build_vt(c, setup["vocab"])
    b = make_batch(1, c)
    with torch.no_grad(): m.ehead.m.net[2].bias[23] = 5.0                                        # has lead
    out = decide(m, Vt, setup["vocab"], b, 8.0, 0.0, 0.0, CascadeCfg(k_p=6, k_v=4, shortlist=12, deploy_k=8, beta1=1.0, beta2=1.0, veto_thr=1e9, gate_thr=-1.0))
    assert out["tier1_used"] and out["tier2_ran"] and 0 < out["confidence"] <= 1 and np.isfinite(out["slots"]).all()
    out0 = decide(m, Vt, setup["vocab"], b, 8.0, 0.0, 0.0, CascadeCfg(k_p=6, k_v=4, shortlist=12, deploy_k=8))
    assert not out0["tier1_used"] and not out0["tier2_ran"]


def test_train_loop_runs_full_model(setup, tmp_path):
    from tanitad.hicap.train import train, TrainCfg
    m = HiCAPFull(HiCAPConfig(**SMALL))
    h = train(m, setup["Vt"], setup["vocab"], setup["ds"], TrainCfg(steps=(4, 2, 2), batch=8, workers=0, log_every=2, ckpt_dir=str(tmp_path), device="cpu"), log=lambda *_: None)
    assert all(np.isfinite(x["loss"]) for x in h)
