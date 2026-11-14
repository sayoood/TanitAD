"""MOTION levers of the level-1 world-model predictor. CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_motion_levers.py

  lever A  --wm-l1-residual {off,on}: level 1 returns z_ctx + f(z_ctx, a_ctx), last layer of f zero-initialised, flat and cell predictor, seen by every caller.
  lever B  --wm-l1-roll-k K with --w-wm-roll W: a K-step rollout term roll/L1 in the level-1 loss, targets detached, the start state keeps its graph.
  diag     --wm-l1-diag {off,on}: extra train-log keys, no effect on the loss.

Controls (each reads a known value):
  m2  at init the residual predictor equals its input exactly (0.0); a non-residual predictor does not.
  m3  roll/L1 alone: predictor parameters and the state path get a non-zero gradient; a frame that is only ever a target gets EXACTLY 0.0;
      control: the one-step loss reads non-zero on that same frame.
  m4  the rollout inside the loss equals HJEPAStack.rollout(1, ..., context=<predictor context>) (max abs difference reported, bar 1e-6).
  m5  total(W) - total(0) == W x roll/L1; the SAME assertion is run on a copy of hjepa.py with the line that adds the term deleted and must go red.
"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
CELL = GRID + ["--wm-l1-pred", "cell"]
KINDS = ("flat", "cell")


# ------------------------------------------------------------------------------------------------ small stacks (stack level)
def _stack(kind="flat", residual=False, k=0, w=0.0, seed=0, d1=64, width=32):
    kw = dict(pred_kind="cell", pred_cells=8, pred_cell_dim=d1 // 8) if kind == "cell" else {}
    mk = lambda **x: H.LevelConfig(pred_width=width, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16, beta=0.08, gamma=10.0, **x)   # noqa: E731
    lv = [mk(latent_dim=d1, action_dim=2, act_embed_dim=16, pred_residual=residual, roll_k=k, roll_weight=w, **kw),
          mk(latent_dim=32, action_dim=16, stride=2), mk(latent_dim=32, action_dim=32, stride=2)]
    torch.manual_seed(seed)
    return H.HJEPAStack(lv, sigreg_projections=32).eval()


def _nonzero_last(st, scale=0.3):
    """Give the zero-initialised last layer of a residual predictor a non-zero value (so that every trunk parameter is live)."""
    g = torch.Generator().manual_seed(5)
    last = st.predictors[0].inner.out[-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * scale)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * scale)


def _seq(b=3, t=12, d=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g), torch.randn(b, t - 1, 2, generator=g)


def _groups(pred):
    inner = pred.inner if isinstance(pred, H.ResidualPredictor) else pred
    out = {}
    for n, c in inner.named_children():
        out[n] = list(c.parameters())
    for n, p in inner.named_parameters(recurse=False):
        out[n] = [p]
    return out


def _gsum(ps):
    return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))


# ------------------------------------------------------------------------------------------------ defaults: nothing built, nothing added
@pytest.mark.parametrize("kind", KINDS)
def test_defaults_build_no_wrapper_no_new_key_and_no_new_loss_term(kind):
    st = _stack(kind)
    assert type(st.predictors[0]) in (H.CausalPredictor, H.CellCausalPredictor)
    assert not any(".inner." in k or k.startswith("predictors.0.inner") for k in st.state_dict())
    z, a = _seq()
    out = st.loss(z, a)
    assert sorted(out) == sorted(["total", *[f"{p}/L{i}" for i in (1, 2, 3) for p in ("pred", "sigreg", "idm", "level")], "act_sigreg/L2", "act_sigreg/L3"])
    assert not [k for k in out if k.startswith(("roll", "diag"))]


# ------------------------------------------------------------------------------------------------ lever A, m2
@pytest.mark.parametrize("kind", KINDS)
def test_m2_residual_at_init_equals_the_copy_exactly_and_the_control_does_not(kind):
    st = _stack(kind, residual=True)
    assert isinstance(st.predictors[0], H.ResidualPredictor)
    pred = st.predictors[0]
    assert (pred.context, pred.latent_dim, pred.action_dim) == (3, 64, 16)
    assert pred.in_proj is pred.inner.in_proj and (kind == "flat" or (pred.n_cells, pred.cell_dim) == (8, 8))      # fall-through attributes
    g = torch.Generator().manual_seed(3)
    for L in (1, 2, 3):
        z, a = torch.randn(4, L, 64, generator=g), torch.randn(4, L, 16, generator=g)
        assert float((pred(z, a) - z).abs().max()) == 0.0                                    # m2
    ctl = _stack(kind, residual=False).predictors[0]
    assert float((ctl(z, a) - z).abs().max()) > 1e-3                                         # control: the plain predictor is not the copy
    roll = st.rollout(1, z[:, 0], torch.randn(4, 5, 2, generator=g), context=3)
    assert float((roll - z[:, 0][:, None]).abs().max()) == 0.0                                # every caller sees it: the rollout of a copy is the start state
    assert sorted(k for k in st.state_dict() if k.startswith("predictors.0.")) == sorted(
        "predictors.0.inner." + k[len("predictors.0."):] for k in _stack(kind).state_dict() if k.startswith("predictors.0."))


@pytest.mark.parametrize("kind", KINDS)
def test_residual_is_the_same_trunk_plus_the_skip_and_uses_no_extra_random_draw(kind):
    a, b = _stack(kind, residual=False), _stack(kind, residual=True)
    sa, sb = a.state_dict(), b.state_dict()
    for k, v in sa.items():                                     # the RNG stream is the unwrapped model's: every tensor but the zeroed last layer is identical
        kb = k.replace("predictors.0.", "predictors.0.inner.", 1) if k.startswith("predictors.0.") else k
        if kb.startswith("predictors.0.inner.out.2."):
            assert float(sb[kb].abs().sum()) == 0.0
        else:
            assert torch.equal(v, sb[kb]), k
    _nonzero_last(b)
    pa, pb = a.predictors[0], b.predictors[0]
    z, ac = torch.randn(2, 3, 64), torch.randn(2, 3, 16)
    inner_out = pb.inner(z, ac)
    assert torch.equal(pb(z, ac), z + inner_out)


def test_residual_refused_where_it_cannot_be_built():
    with pytest.raises(ValueError, match="level-1"):
        H.HJEPAStack([H.LevelConfig(latent_dim=32, action_dim=2), H.LevelConfig(latent_dim=32, action_dim=32, stride=2, pred_residual=True)])
    with pytest.raises(ValueError, match="override"):
        H.HJEPAStack([H.LevelConfig(latent_dim=32, action_dim=2, pred_residual=True)], predictor_overrides={1: torch.nn.Identity()})
    with pytest.raises(ValueError, match="last output layer"):
        H.ResidualPredictor(torch.nn.Identity())


# ------------------------------------------------------------------------------------------------ lever B, m4
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("residual", [False, True])
def test_m4_the_rollout_in_the_loss_equals_the_stack_rollout_with_the_predictor_context(kind, residual):
    K = 5
    st = _stack(kind, residual=residual, k=K, w=0.5)
    if residual:
        _nonzero_last(st)
    z, raw = _seq(t=12)
    a = st.act_embed(raw)
    pr, tgt, starts = st.roll_pairs(z, a, K)
    assert starts == list(range(0, 12 - K, 2)) and pr.shape == (len(starts), 3, K, 64) and not tgt.requires_grad
    ctx = st.predictors[0].context
    assert ctx == 3 and K > ctx                                          # the context grows 1 -> 3 and then slides
    worst = 0.0
    for i, s in enumerate(starts):
        ref = st.rollout(1, z[:, s], raw[:, s:s + K], context=ctx)
        worst = max(worst, float((pr[i] - ref).abs().max()))
        assert torch.equal(tgt[i], z[:, s + 1:s + K + 1])
    assert worst <= 1e-6, worst
    assert float((pr - tgt).abs().max()) > 1e-3                          # control: the rollout is not trivially the target


def test_roll_loss_is_the_mean_squared_error_over_starts_steps_and_batch():
    K = 4
    st = _stack("flat", k=K, w=1.0)
    z, raw = _seq(t=10)
    a = st.act_embed(raw)
    got = float(st._roll_loss(z, a, K))
    ref = []
    for s in range(0, 10 - K, 2):
        r = st.rollout(1, z[:, s], raw[:, s:s + K], context=3)
        ref.append((r - z[:, s + 1:s + K + 1]).pow(2).mean(-1))            # [B, K]
    want = float(torch.stack(ref).mean())
    assert abs(got - want) < 1e-6
    st.pred_reduction = "sum"
    assert abs(float(st._roll_loss(z, a, K)) - float(torch.stack([(st.rollout(1, z[:, s], raw[:, s:s + K], context=3) - z[:, s + 1:s + K + 1]).pow(2).sum(-1)
                                                                  for s in range(0, 10 - K, 2)]).mean())) < 1e-4


# ------------------------------------------------------------------------------------------------ lever B, m3
@pytest.mark.parametrize("kind", KINDS)
def test_m3_roll_loss_alone_reaches_the_predictor_and_the_start_state_and_never_a_target_only_frame(kind):
    K = 4
    st = _stack(kind, k=K, w=1.0).train()
    z, raw = _seq(t=11)
    zl = z.clone().requires_grad_(True)                                  # a leaf: its grad IS the state path
    a = st.act_embed(raw)
    st._roll_loss(zl, a, K).backward()
    dead = {n: _gsum(v) for n, v in _groups(st.predictors[0]).items() if not _gsum(v) > 0.0}
    assert not dead, dead                                                # every group of the level-1 predictor
    g = zl.grad.abs().sum(dim=(0, 2))                                    # per frame
    starts = list(range(0, 11 - K, 2))
    assert all(float(g[s]) > 0.0 for s in starts)                         # a start state keeps its graph
    assert float(g[10]) == 0.0                                           # the last frame is only ever a target: EXACTLY 0.0
    assert all(float(g[t]) == 0.0 for t in range(11) if t not in starts)  # so is every frame that is not a start
    assert _gsum(st.act_embed.parameters()) > 0.0                        # the action embedding is trained through it too
    # control: the one-step loss (not detached) reads NON-ZERO on the last frame, so the probe can see a gradient there
    st.zero_grad()
    zc = z.clone().requires_grad_(True)
    st._pred_loss(1, zc, st.act_embed(raw), detach_target=False).backward()
    assert float(zc.grad[:, 10].abs().sum()) > 0.0


def test_m3_state_projection_gets_the_gradient_through_the_start_state():
    cfg = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_roll_k=6, w_wm_roll=1.0, d=32, ff=64, **{k: v for k, v in TINY.items() if k in ("heads", "n_q", "depth_s", "depth_t", "depth_o", "nb")})
    torch.manual_seed(0)
    wm = M.WMAux(cfg)
    g = torch.Generator().manual_seed(2)
    adapt, act = torch.randn(2, 16, cfg.n_tok, 32, generator=g), torch.randn(2, 16, 2, generator=g)
    z1 = wm.z1(adapt)
    a = wm.scale_actions(act)[:, :-1]
    out = wm.stack.loss(z1, a)
    out["roll/L1"].backward()                                            # roll/L1 ALONE
    assert _gsum(wm.proj.parameters()) > 0.0                             # the state projection, through the start state
    assert _gsum(wm.stack.predictors[0].parameters()) > 0.0
    wm.zero_grad()
    z1b = wm.z1(adapt)
    out2 = wm.stack.loss(z1b, a)
    out2["pred/L1"].backward()
    assert _gsum(wm.proj.parameters()) > 0.0                             # control: the one-step term reaches it too (reads non-zero)


# ------------------------------------------------------------------------------------------------ lever B, m5
def _check_total_delta(Hm, kind="flat"):
    """total(W) - total(0) == W x roll/L1 (W changed in the built stack's config, everything else equal), and it is not trivially 0."""
    K, W = 4, 0.7
    mk = lambda w: Hm.HJEPAStack([Hm.LevelConfig(latent_dim=64, action_dim=2, act_embed_dim=16, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16,
                                                 roll_k=K, roll_weight=w, **(dict(pred_kind="cell", pred_cells=8, pred_cell_dim=8) if kind == "cell" else {})),
                                  Hm.LevelConfig(latent_dim=32, action_dim=16, stride=2, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16),
                                  Hm.LevelConfig(latent_dim=32, action_dim=32, stride=2, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16)],
                                 sigreg_projections=32).eval()
    torch.manual_seed(0)
    s1 = mk(W)
    torch.manual_seed(0)
    s0 = mk(W)
    s0.cfg[0].roll_weight = 0.0                                          # the same model, the weight set to 0 after the build
    z, a = _seq(t=11)
    g = torch.Generator().manual_seed(9)
    o1, o0 = s1.loss(z, a, generator=g), s0.loss(z, a, generator=torch.Generator().manual_seed(9))
    delta = float(o1["total"] - o0["total"])
    assert float(o1["roll/L1"]) > 1e-3
    assert abs(delta - W * float(o1["roll/L1"])) < 1e-5, (delta, W * float(o1["roll/L1"]))


@pytest.mark.parametrize("kind", KINDS)
def test_m5_total_moves_by_w_times_the_roll_term_and_a_mutation_of_the_adding_line_goes_red(kind):
    """GUARDS the line in ``HJEPAStack.loss`` that reads ``level_loss = level_loss + c.roll_weight * roll``: the same assertion is run on a copy of hjepa.py
    with that line deleted, and it must fail (the term would be logged and never trained)."""
    _check_total_delta(H, kind)
    src = inspect.getsource(H)
    marker = "level_loss = level_loss + c.roll_weight * roll"
    lines = src.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip().startswith(marker)]
    assert len(idx) == 1, idx
    mut_src = "\n".join(lines[:idx[0]] + lines[idx[0] + 1:])
    mut = types.ModuleType("hjepa_mutant")
    mut.__dict__["__file__"] = "hjepa_mutant.py"
    sys.modules["hjepa_mutant"] = mut                                    # dataclasses resolve annotations through sys.modules
    try:
        exec(compile(mut_src, "hjepa_mutant.py", "exec"), mut.__dict__)
        with pytest.raises(AssertionError):
            _check_total_delta(mut, kind)
    finally:
        sys.modules.pop("hjepa_mutant", None)


@pytest.mark.parametrize("kind", KINDS)
def test_both_levers_together_run_finite_and_roll_term_matters(kind):
    st = _stack(kind, residual=True, k=6, w=0.3).train()
    z, a = _seq(t=12)
    z.requires_grad_(True)
    out = st.loss(z, a)
    assert all(torch.isfinite(v) for v in out.values())
    out["total"].backward()
    assert _gsum(st.predictors[0].parameters()) > 0.0
    with pytest.raises(ValueError, match="does not fit"):
        st.loss(z[:, :6], a[:, :5])                                      # K 6 needs T >= 7


# ------------------------------------------------------------------------------------------------ config refusals
def test_config_refuses_what_it_cannot_build():
    mk = lambda **kw: M.V8FMiniConfig(wm="b", hmax=16, **kw)             # noqa: E731
    mk()
    mk(wm_l1_residual="on")
    mk(wm_l1_roll_k=2, w_wm_roll=0.1)
    mk(wm_l1_roll_k=15, w_wm_roll=0.1)
    for kw, pat in (({"wm_l1_roll_k": 1, "w_wm_roll": 0.1}, "need K = 0"), ({"wm_l1_roll_k": 4, "w_wm_roll": 0.0}, "need K = 0"),
                    ({"wm_l1_roll_k": 0, "w_wm_roll": 0.2}, "need K = 0"), ({"wm_l1_roll_k": 16, "w_wm_roll": 0.1}, "does not fit"),
                    ({"wm_l1_roll_k": 4, "w_wm_roll": float("nan")}, "finite"), ({"wm_l1_roll_k": -2, "w_wm_roll": 0.1}, ">= 0"),
                    ({"wm_l1_residual": "maybe"}, "wm_l1_residual"), ({"wm_l1_diag": "x"}, "wm_l1_diag")):
        with pytest.raises(ValueError, match=pat):
            mk(**kw)
    for kw in ({"wm_l1_residual": "on"}, {"wm_l1_roll_k": 4, "w_wm_roll": 0.1}, {"wm_l1_diag": "on"}):
        with pytest.raises(ValueError, match="world-model"):
            M.V8FMiniConfig(wm="off", **kw)


# ------------------------------------------------------------------------------------------------ the trainer: flags, gate, checkpoints
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("motion")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J}


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _built(world, extra=(), **over):
    a = _args(extra)
    c0 = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(0)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None}
    return a, cfg, model, opt, info


A_ON, B_ON, AB_CELL = ["--wm-l1-residual", "on"], ["--wm-l1-roll-k", "6", "--w-wm-roll", "0.3"], GRID + ["--wm-l1-pred", "cell", "--wm-l1-residual", "on", "--wm-l1-roll-k", "6", "--w-wm-roll", "0.3"]


def test_flags_have_entries_defaults_and_ckpt_match():
    d = {act.dest for act in TR.build_parser()._actions}
    p = TR.build_parser()
    for f, dflt in (("wm_l1_residual", "off"), ("wm_l1_roll_k", 0), ("w_wm_roll", 0.0), ("wm_l1_diag", "off")):
        assert f in d and f in TR.FLAG_CHECKS and p.get_default(f) == dflt
    for f in ("wm_l1_residual", "wm_l1_roll_k", "w_wm_roll"):
        assert f in TR.INIT_CKPT_MATCH
    with pytest.raises(ValueError):
        TR.make_config(_args(["--wm-l1-roll-k", "6"]))                    # K with no weight
    with pytest.raises(ValueError):
        TR.make_config(_args(["--w-wm-roll", "0.3"]))                     # a weight with no K
    with pytest.raises(ValueError):
        TR.make_config(_args(["--wm-l1-roll-k", "16", "--w-wm-roll", "0.3"]))   # K does not fit hmax 16


@pytest.mark.parametrize("extra", [GRID + A_ON, GRID + B_ON, AB_CELL, A_ON, B_ON], ids=["A_grid", "B_grid", "AB_cell", "A_mean", "B_mean"])
def test_declared_vs_built_passes_when_built_and_refuses_every_lie(world, extra):
    a, cfg, model, opt, info = _built(world, extra)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    lv = model.built_levers()
    assert lv["wm_l1_residual"] == ("on" if "--wm-l1-residual" in extra else "off")
    assert lv["wm_l1_roll_k"] == (6 if "--wm-l1-roll-k" in extra else 0) and lv["w_wm_roll"] == (0.3 if "--w-wm-roll" in extra else 0.0)
    base = GRID if "--wm-state" in extra else []
    ab, cb, mb, ob, ib = _built(world, base)                                # everything default: the same argv minus the levers
    assert TR.declared_vs_built(ab, cb, mb, ob, ib) == []
    for f, v in (("wm_l1_residual", "on" if lv["wm_l1_residual"] == "off" else "off"), ("wm_l1_roll_k", 8), ("w_wm_roll", 0.9)):
        wrong = _args(extra)
        setattr(wrong, f, v)
        assert any(f in m for m in TR.declared_vs_built(wrong, cfg, model, opt, info)), f          # a deliberately wrong expectation is refused


def test_declared_but_not_built_is_refused_in_both_directions(world):
    a, cfg, m_on, opt, info = _built(world, GRID + A_ON + B_ON)
    ad, cd, m_off, od, idf = _built(world, GRID)
    assert any("wm_l1_residual" in m for m in TR.declared_vs_built(a, cd, m_off, od, idf))          # declared on, built off
    assert any("wm_l1_roll_k" in m for m in TR.declared_vs_built(a, cd, m_off, od, idf))
    assert any("wm_l1_residual" in m for m in TR.declared_vs_built(ad, cfg, m_on, opt, info))       # declared off, built on
    assert any("w_wm_roll" in m for m in TR.declared_vs_built(ad, cfg, m_on, opt, info))
    m_on.wm.stack.predictors[0] = m_on.wm.stack.predictors[0].inner                                 # the wrapper removed from the built model under an 'on' argv
    assert any("wm_l1_residual" in m for m in TR.declared_vs_built(a, cfg, m_on, opt, info))
    a2, cfg2, m2, o2, i2 = _built(world, GRID + B_ON)
    m2.wm.stack.cfg[0].roll_weight = 0.0                                                            # the loss's own weight changed under an argv that declares 0.3
    assert any("w_wm_roll" in m for m in TR.declared_vs_built(a2, cfg2, m2, o2, i2))


def test_wm_off_with_a_motion_lever_is_refused(world):
    with pytest.raises(ValueError):
        TR.make_config(_args(["--arm", "M1", "--wm-l1-residual", "on"]))
    a1, c1, m1, o1, i1 = _built(world, ["--arm", "M1"])
    assert TR.declared_vs_built(a1, c1, m1, o1, i1) == []
    wrong = _args(["--arm", "M1"])
    wrong.wm_l1_roll_k = 4
    assert any("wm-l1-roll-k" in m for m in TR.declared_vs_built(wrong, c1, m1, o1, i1))


def _loss(world, extra, **over):
    a, cfg, model, opt, info = _built(world, extra, **over)
    model.train()
    out = model.loss(dict(world["batch"]), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    return model, out


def test_diag_off_adds_no_key_on_adds_four_and_the_loss_is_unchanged(world):
    _, off = _loss(world, GRID + B_ON)
    _, on = _loss(world, GRID + B_ON + ["--wm-l1-diag", "on"])
    new = sorted(set(on["terms"]) - set(off["terms"]))
    assert new == ["wm_copy_l1", "wm_pred_l1", "wm_roll_copy_l1", "wm_roll_l1"]
    assert float(on["terms"]["total"]) == float(off["terms"]["total"]) and float(on["terms"]["wm"]) == float(off["terms"]["wm"])
    assert not [k for k in off["terms"] if k.startswith(("wm_pred", "wm_copy", "wm_roll_"))]      # with K > 0 the plain rollout term wm_roll is logged (G-LIVE requires it); the diagnostics are not
    assert "wm_roll" in off["terms"] and abs(float(off["terms"]["wm_roll"]) - float(on["terms"]["wm_roll_l1"])) < 1e-6
    assert not any(v.requires_grad for k, v in on["terms"].items() if k in new)                     # no graph
    _, on0 = _loss(world, GRID + ["--wm-l1-diag", "on"])
    assert sorted(set(on0["terms"]) - set(_loss(world, GRID)[1]["terms"])) == ["wm_copy_l1", "wm_pred_l1"]   # K = 0: no roll keys
    t = on["terms"]
    assert all(np.isfinite(float(t[k])) and float(t[k]) > 0 for k in new)


def test_diag_copy_values_equal_a_hand_computation(world):
    model, out = _loss(world, GRID + B_ON + ["--wm-l1-diag", "on"])
    with torch.no_grad():
        z1 = model.wm.z1(out["adapt"])
        want_copy = float((z1[:, :-1] - z1[:, 1:]).pow(2).mean(-1).mean())
        K = 6
        rc = torch.stack([(z1[:, s][:, None] - z1[:, s + 1:s + K + 1]).pow(2).mean(-1) for s in range(0, z1.shape[1] - K, 2)]).mean()
    assert abs(float(out["terms"]["wm_copy_l1"]) - want_copy) < 1e-6
    assert abs(float(out["terms"]["wm_roll_copy_l1"]) - float(rc)) < 1e-6


def test_the_trainer_runs_both_levers_and_checkpoint_cfg_keys_follow_the_defaults(world, tmp_path):
    base = ["--synthetic", "--arm", "M2b", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--steps", "3", "--log-every", "1"]
    assert TR.main([*base, "--out", str(tmp_path / "grid")] + GRID) == 0
    assert TR.main([*base, "--out", str(tmp_path / "ab")] + AB_CELL + ["--wm-l1-diag", "on"]) == 0
    ld = lambda n: torch.load(tmp_path / n / "ckpt_last.pt", map_location="cpu", weights_only=False)   # noqa: E731
    cg, cb = ld("grid"), ld("ab")
    assert not (set(M.MOTION_CFG_DEFAULTS) & set(cg["cfg"]))                                     # a default checkpoint keeps its key set
    assert set(cb["cfg"]) - set(cg["cfg"]) == set(M.MOTION_CFG_DEFAULTS) | set(M.L1PRED_CFG_FIELDS)
    cfg = M.V8FMiniConfig(**cb["cfg"])                                                           # the T1 eval builder
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(cb["model"])                                                           # strict
    assert model.built_levers()["wm_l1_residual"] == "on" and model.built_levers()["wm_l1_roll_k"] == 6
    rows = [__import__("json").loads(l) for l in open(tmp_path / "ab" / "train_log.jsonl", encoding="utf-8")]
    assert all(k in rows[-1] for k in ("wm_pred_l1", "wm_copy_l1", "wm_roll_l1", "wm_roll_copy_l1"))
    rows_g = [__import__("json").loads(l) for l in open(tmp_path / "grid" / "train_log.jsonl", encoding="utf-8")]
    assert not [k for k in rows_g[-1] if k.startswith(("wm_pred", "wm_copy", "wm_roll"))]
    # fine-tune start: an argv with a different residual lever must not start from this checkpoint, nor the reverse; an absent key = the default
    ov = {k: cb["cfg"][k] for k in ("d", "ff", "heads", "n_q", "depth_s", "depth_t", "depth_o", "nb", "hmax", "hist")}
    _, c_plain, m_plain, _, _ = _built(world, GRID + ["--wm-l1-pred", "cell"], **ov)
    with pytest.raises(ValueError, match="wm_l1_residual"):
        TR.load_init_ckpt(str(tmp_path / "ab" / "ckpt_last.pt"), m_plain, c_plain)
    _, c_a, m_a, _, _ = _built(world, GRID + A_ON, **ov)
    with pytest.raises(ValueError, match=r"wm_l1_residual: 'off' != 'on'"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m_a, c_a)                    # an older checkpoint (key absent = off) is refused by an 'on' argv
    _, c_ab, m_ab, _, _ = _built(world, AB_CELL, **ov)
    assert TR.load_init_ckpt(str(tmp_path / "ab" / "ckpt_last.pt"), m_ab, c_ab)["n_mismatch_after_load"] == 0


def test_g_live_requires_and_sees_the_roll_term_and_every_group_trains(world):
    for extra in (GRID + B_ON, AB_CELL):
        a, cfg, model, opt, info = _built(world, extra)
        res = TR.live_check(model, dict(world["batch"]), cfg)
        assert "wm_roll" in res["terms"] and res["terms"]["wm_roll"] > 0.0 and res["missing_terms"] == [] and res["unexpected_dead"] == [], (res["missing_terms"], res["unexpected_dead"])
    a, cfg, model, opt, info = _built(world, GRID)
    res = TR.live_check(model, dict(world["batch"]), cfg)
    assert "wm_roll" not in res["terms"] and res["missing_terms"] == []         # no lever, no key
