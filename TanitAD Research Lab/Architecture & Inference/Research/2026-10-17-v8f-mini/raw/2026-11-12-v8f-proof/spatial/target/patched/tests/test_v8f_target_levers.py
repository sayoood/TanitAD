"""TARGET levers of the level-1 world-model predictor (PREREG_V8F_PROOF_ADD19). CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_target_levers.py

  lever A  EGOV    --wm-l1-act {acc_yaw,speed}: the level-1 action gets a third channel, the logged ego speed of the start frame / WM_SPEED_SCALE.
  lever B  FUTTOK  --wm-l1-futtok-k K --w-wm-futtok W: a per-cell linear head reads the rolled-out states of steps 1..K against the frozen
                   (layer-normalised, detached) cached token of the same cell at the same frame.

Controls (each reads a known value):
  t2  EGOV: non-zero gradient on the column of the action embedding that reads the third channel; the same column reads EXACTLY 0.0 when the speed is all zeros;
      permuting the speed across the batch moves the level-1 loss; with the lever off a different speed moves nothing.
  t3  FUTTOK alone: tok_head, every level-1 predictor group, the action embedding and the state projection get a non-zero gradient; levels 2 and 3 read EXACTLY 0.0
      (control: the total reaches them).
  t4  the states the term reads equal HJEPAStack.rollout(1, ..., context=<predictor context>) (max abs difference reported).
  t5  total(W) - total(0) == W x futtok/L1 (float64); the SAME assertion is run on a copy of hjepa.py with the adding line deleted and must go red.
"""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
NT, DT = 8, 24                                                       # cells and token width of the small stacks


# ------------------------------------------------------------------------------------------------ small stacks (stack level)
def _stack(kind="flat", fk=0, fw=0.0, rk=0, rw=0.0, adim=2, seed=0, d1=64, width=32):
    kw = dict(pred_kind="cell", pred_cells=NT, pred_cell_dim=d1 // NT) if kind == "cell" else {}
    mk = lambda **x: H.LevelConfig(pred_width=width, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16, beta=0.08, gamma=10.0, **x)   # noqa: E731
    lv = [mk(latent_dim=d1, action_dim=adim, act_embed_dim=16, roll_k=rk, roll_weight=rw, futtok_k=fk, futtok_weight=fw, **kw),
          mk(latent_dim=32, action_dim=16, stride=2), mk(latent_dim=32, action_dim=32, stride=2)]
    torch.manual_seed(seed)
    return H.HJEPAStack(lv, sigreg_projections=32).eval()


def _head(d1=64, seed=3):
    torch.manual_seed(seed)
    return torch.nn.Linear(d1 // NT, DT)


def _seq(b=3, t=12, d=64, adim=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g), torch.randn(b, t - 1, adim, generator=g), torch.randn(b, t, NT, DT, generator=g)


def _ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],))


def _gsum(ps):
    return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))


def _level_modules(st, lv):
    return [st.predictors[lv - 1], st.idm_heads[lv - 1], *([st.encoders[lv - 2], st.macro_encoders[lv - 2]] if lv >= 2 else [])]


# ------------------------------------------------------------------------------------------------ defaults: nothing built, nothing added
@pytest.mark.parametrize("kind", KINDS)
def test_defaults_build_no_head_no_new_key_no_new_loss_term_and_refuse_stray_inputs(kind):
    st = _stack(kind)
    z, a, tg = _seq()
    out = st.loss(z, a)
    assert sorted(out) == sorted(["total", *[f"{p}/L{i}" for i in (1, 2, 3) for p in ("pred", "sigreg", "idm", "level")], "act_sigreg/L2", "act_sigreg/L3"])
    assert not [k for k in out if "futtok" in k]
    with pytest.raises(ValueError, match="futtok_k 0"):
        st.loss(z, a, tok_head=_head(), tok_tgt=_ln(tg))             # inputs the loss would ignore are refused
    on = _stack(kind, fk=4, fw=1.0)
    with pytest.raises(ValueError, match="no tok_head"):
        on.loss(z, a)                                                 # a declared term with no inputs is refused
    with pytest.raises(ValueError, match="come together"):
        on.loss(z, a, tok_head=_head())


def test_levelconfig_refuses_futtok_where_it_cannot_run():
    with pytest.raises(ValueError, match="level-1"):
        H.HJEPAStack([H.LevelConfig(latent_dim=32, action_dim=2), H.LevelConfig(latent_dim=32, action_dim=32, stride=2, futtok_k=4, futtok_weight=1.0)])
    for k, w in ((1, 1.0), (4, 0.0), (0, 0.5), (4, float("nan"))):
        with pytest.raises(ValueError, match="need"):
            H.HJEPAStack([H.LevelConfig(latent_dim=32, action_dim=2, futtok_k=k, futtok_weight=w)])


# ------------------------------------------------------------------------------------------------ t4: the rolled-out states the term reads
class _Rec(torch.nn.Module):
    """Wraps a head and records what it reads."""

    def __init__(self, head):
        super().__init__()
        self.head, self.seen = head, []

    def forward(self, x):
        self.seen.append(x)
        return self.head(x)


@pytest.mark.parametrize("kind", KINDS)
def test_t4_states_read_by_the_term_equal_the_stack_rollout(kind):
    K = 5
    st = _stack(kind, fk=K, fw=1.0)
    z, raw, tg = _seq(t=12)
    rec = _Rec(_head())
    st.loss(z, raw, tok_head=rec, tok_tgt=_ln(tg))
    starts = list(range(0, 12 - K, 2))
    assert len(rec.seen) == len(starts)
    ctx = st.predictors[0].context
    assert ctx == 3 and K > ctx
    worst = 0.0
    for x, s in zip(rec.seen, starts):
        ref = st.rollout(1, z[:, s], raw[:, s:s + K], context=ctx)                               # [B, K, D]
        assert x.shape == (3, K, NT, 64 // NT)
        worst = max(worst, float((x.reshape(3, K, 64) - ref).abs().max()))
    assert worst <= 1e-6, worst
    assert float((rec.seen[0].reshape(3, K, 64) - z[:, 1:K + 1]).abs().max()) > 1e-3             # control: the rollout is not the targets


@pytest.mark.parametrize("kind", KINDS)
def test_futtok_value_is_the_mean_squared_error_to_the_layernormed_token(kind):
    K = 4
    st = _stack(kind, fk=K, fw=1.0)
    z, raw, tg = _seq(t=10)
    head = _head()
    got = float(st.loss(z, raw, tok_head=head, tok_tgt=_ln(tg))["futtok/L1"])
    ref = []
    for s in range(0, 10 - K, 2):
        r = st.rollout(1, z[:, s], raw[:, s:s + K], context=3).reshape(3, K, NT, 64 // NT)
        ref.append((head(r) - _ln(tg[:, s + 1:s + K + 1])).pow(2).mean())                          # mean over width, cells, steps, batch
    assert abs(got - float(torch.stack(ref).mean())) < 1e-6
    assert float(_ln(tg).mean(-1).abs().max()) < 1e-5 and abs(float(_ln(tg).var(-1, unbiased=False).mean()) - 1.0) < 1e-3     # the target is normalised over the feature axis


# ------------------------------------------------------------------------------------------------ t3: reach
@pytest.mark.parametrize("kind", KINDS)
def test_t3_futtok_alone_reaches_head_predictor_embedding_and_start_state_and_not_levels_2_3(kind):
    K = 4
    st = _stack(kind, fk=K, fw=1.0).train()
    z, raw, tg = _seq(t=11)
    zl = z.clone().requires_grad_(True)
    head = _head()
    out = st.loss(zl, raw, tok_head=head, tok_tgt=_ln(tg))
    out["futtok/L1"].backward()                                                    # the term ALONE
    assert _gsum(head.parameters()) > 0.0
    dead = {n: _gsum(c.parameters()) for n, c in st.predictors[0].named_children() if not _gsum(c.parameters()) > 0.0}
    assert not dead, dead                                                          # every group of the level-1 predictor
    assert _gsum(st.act_embed.parameters()) > 0.0
    g = zl.grad.abs().sum(dim=(0, 2))
    starts = list(range(0, 11 - K, 2))
    assert all(float(g[s]) > 0.0 for s in starts)                                  # the start state keeps its graph
    assert all(float(g[t]) == 0.0 for t in range(11) if t not in starts)           # nothing else does: the targets carry no gradient
    for lv in (2, 3):
        assert _gsum(p for m in _level_modules(st, lv) for p in m.parameters()) == 0.0, lv      # EXACTLY 0.0 on levels 2 and 3
    st.zero_grad()
    st.loss(z.clone().requires_grad_(True), raw, tok_head=head, tok_tgt=_ln(tg))["total"].backward()
    for lv in (2, 3):
        assert _gsum(p for m in _level_modules(st, lv) for p in m.parameters()) > 0.0, lv        # control: the total reaches them, so the probe can see a gradient there


def _wm(extra=(), seed=0, **over):
    kw = dict(wm="b", hmax=16, wm_state="grid", wm_cell_dim=16, d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, d_feat=DT, n_tok=NT)
    cfg = M.V8FMiniConfig(**{**kw, **over})
    torch.manual_seed(seed)
    wm = M.WMAux(cfg)
    g = torch.Generator().manual_seed(2)
    return cfg, wm, torch.randn(2, 16, NT, 32, generator=g), torch.randn(2, 16, 2, generator=g), torch.rand(2, 16, generator=g) * 20.0, torch.randn(2, 16, NT, DT, generator=g)


def test_t3_state_projection_gets_the_gradient_through_the_start_state_of_the_token_term():
    cfg, wm, adapt, act, spd, feats = _wm(wm_l1_futtok_k=6, w_wm_futtok=1.0)
    out = wm(adapt, act, feats=feats)
    out["futtok/L1"].backward()
    assert _gsum(wm.proj.parameters()) > 0.0 and _gsum(wm.tok_head.parameters()) > 0.0
    wm.zero_grad()
    wm(adapt, act, feats=feats)["total"].backward()                                  # control: the full loss reaches the projection too
    assert _gsum(wm.proj.parameters()) > 0.0


# ------------------------------------------------------------------------------------------------ t5: the term is in the total
def _check_total_delta(Hm, kind="flat"):
    K, W = 4, 0.7
    mk = lambda w: Hm.HJEPAStack([Hm.LevelConfig(latent_dim=64, action_dim=2, act_embed_dim=16, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16,
                                                 futtok_k=K, futtok_weight=w, **(dict(pred_kind="cell", pred_cells=NT, pred_cell_dim=64 // NT) if kind == "cell" else {})),
                                  Hm.LevelConfig(latent_dim=32, action_dim=16, stride=2, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16),
                                  Hm.LevelConfig(latent_dim=32, action_dim=32, stride=2, pred_width=32, pred_depth=1, pred_heads=4, enc_hidden=32, macro_hidden=16)],
                                 sigreg_projections=32).eval().double()
    torch.manual_seed(0)
    s1 = mk(W)
    torch.manual_seed(0)
    s0 = mk(W)
    s0.cfg[0].futtok_weight = 0.0                                                    # the same model, the weight set to 0 after the build
    z, a, tg = _seq(t=11)
    z, a, tg = z.double(), a.double(), tg.double()
    head = _head().double()
    o1 = s1.loss(z, a, generator=torch.Generator().manual_seed(9), tok_head=head, tok_tgt=_ln(tg))
    o0 = s0.loss(z, a, generator=torch.Generator().manual_seed(9), tok_head=head, tok_tgt=_ln(tg))
    delta = float(o1["total"] - o0["total"])
    assert float(o1["futtok/L1"]) > 1e-3
    assert abs(delta - W * float(o1["futtok/L1"])) < 1e-9, (delta, W * float(o1["futtok/L1"]))


@pytest.mark.parametrize("kind", KINDS)
def test_t5_total_moves_by_w_times_the_term_and_a_mutation_of_the_adding_line_goes_red(kind):
    """GUARDS the line in ``HJEPAStack.loss`` that reads ``level_loss = level_loss + c.futtok_weight * futt``: the same assertion is run on a copy of hjepa.py
    with that line deleted, and it must fail (the term would be logged and never trained)."""
    _check_total_delta(H, kind)
    src = inspect.getsource(H)
    marker = "level_loss = level_loss + c.futtok_weight * futt"
    lines = src.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip().startswith(marker)]
    assert len(idx) == 1, idx
    mut = types.ModuleType("hjepa_mutant_ft")
    mut.__dict__["__file__"] = "hjepa_mutant_ft.py"
    sys.modules["hjepa_mutant_ft"] = mut
    try:
        exec(compile("\n".join(lines[:idx[0]] + lines[idx[0] + 1:]), "hjepa_mutant_ft.py", "exec"), mut.__dict__)
        with pytest.raises(AssertionError):
            _check_total_delta(mut, kind)
    finally:
        sys.modules.pop("hjepa_mutant_ft", None)


# ------------------------------------------------------------------------------------------------ ROLL and FUTTOK share the rollout
def test_roll_and_futtok_with_the_same_k_share_one_rollout_and_different_k_do_not():
    z, a, tg = _seq(t=12)
    for rk, fk, want in ((5, 5, 1), (5, 6, 2), (0, 5, 1), (5, 0, 1)):
        st = _stack("flat", fk=fk, fw=1.0 if fk else 0.0, rk=rk, rw=0.3 if rk else 0.0)
        n = []
        orig = st.roll_pairs
        st.roll_pairs = lambda *x, _o=orig, _n=n: (_n.append(x[2]), _o(*x))[1]
        kw = dict(tok_head=_head(), tok_tgt=_ln(tg)) if fk else {}
        st.loss(z, a, **kw)
        assert len(n) == want, (rk, fk, n)
    st = _stack("flat", fk=5, fw=1.0, rk=5, rw=0.3)
    out = st.loss(z, a, tok_head=_head(), tok_tgt=_ln(tg))
    assert "roll/L1" in out and "futtok/L1" in out and all(torch.isfinite(v) for v in out.values())


def test_roll_term_alone_is_unchanged_by_the_refactor_of_its_call():
    z, a, tg = _seq(t=11)
    st = _stack("flat", rk=4, rw=0.5)
    e = st.act_embed(a)
    assert float(st._roll_loss(z, e, 4)) == float(st._roll_loss(z, e, 4, rolled=st.roll_pairs(z, e, 4)))


# ------------------------------------------------------------------------------------------------ diagnostics
@pytest.mark.parametrize("kind", KINDS)
def test_diag_futtok_copy_is_the_head_on_the_start_state_without_a_graph(kind):
    K = 4
    st = _stack(kind, fk=K, fw=1.0)
    z, raw, tg = _seq(t=10)
    head = _head()
    on = st.loss(z.clone().requires_grad_(True), raw, tok_head=head, tok_tgt=_ln(tg), diag=True, generator=torch.Generator().manual_seed(4))
    off = st.loss(z.clone().requires_grad_(True), raw, tok_head=head, tok_tgt=_ln(tg), generator=torch.Generator().manual_seed(4))
    assert "diag/futtok_copy" in on and "diag/futtok_copy" not in off
    assert not on["diag/futtok_copy"].requires_grad
    assert float(on["total"]) == float(off["total"]) and float(on["futtok/L1"]) == float(off["futtok/L1"])
    ref = []
    for s in range(0, 10 - K, 2):
        h0 = head(z[:, s].reshape(3, NT, 64 // NT))[:, None]
        ref.append((h0 - _ln(tg[:, s + 1:s + K + 1])).pow(2).mean())
    assert abs(float(on["diag/futtok_copy"]) - float(torch.stack(ref).mean())) < 1e-6


# ------------------------------------------------------------------------------------------------ lever A: EGOV, WMAux level
def test_egov_scale_actions_two_channels_when_off_three_when_on_and_a_clear_error_without_speed():
    _, off, adapt, act, spd, feats = _wm()
    _, on, *_ = _wm(wm_l1_act="speed")
    t_off = off.scale_actions(act)
    assert t_off.shape == (2, 16, 2) and torch.equal(off.scale_actions(act, spd), t_off)          # off: today's tensor, a speed passed anyway is not read
    t_on = on.scale_actions(act, spd)
    assert t_on.shape == (2, 16, 3) and torch.equal(t_on[..., :2], t_off) and torch.equal(t_on[..., 2], spd / M.WM_SPEED_SCALE) and M.WM_SPEED_SCALE == 10.0
    with pytest.raises(ValueError, match="no `speed` was given"):
        on.scale_actions(act)
    with pytest.raises(ValueError, match="must match"):
        on.scale_actions(act, spd[:, :-1])
    with pytest.raises(ValueError, match="no `speed`"):
        on(adapt, act)                                                    # a speed-built model rolled out / trained without speed
    assert on.stack.act_embed[0].in_features == 3 and off.stack.act_embed[0].in_features == 2


def test_wm_frame_speed_reads_the_field_the_trainer_reads_and_refuses_a_batch_without_it():
    b = {"wm_speed": torch.arange(6.0).reshape(2, 3)}
    assert M.wm_frame_speed(b) is b["wm_speed"]
    with pytest.raises(KeyError, match="wm_speed"):
        M.wm_frame_speed({"act": torch.zeros(1)})


def test_t2_egov_reaches_the_third_channel_weights_and_the_controls_read_exact_zero_and_the_loss_is_sensitive():
    _, wm, adapt, act, spd, feats = _wm(wm_l1_act="speed")
    wm(adapt, act, speed=spd)["pred/L1"].backward()                         # the level-1 one-step term ALONE
    w = wm.stack.act_embed[0].weight
    assert w.shape == (32, 3) and float(w.grad[:, 2].abs().sum()) > 0.0       # the weights that read the third channel
    assert float(w.grad[:, 0].abs().sum()) > 0.0                             # control: the old channels still train
    for lv in (2, 3):                                                      # control 1: modules the level-1 term cannot reach read EXACTLY 0.0
        assert _gsum(p for m in _level_modules(wm.stack, lv) for p in m.parameters()) == 0.0
    wm.zero_grad()
    wm(adapt, act, speed=torch.zeros_like(spd))["pred/L1"].backward()       # control 2: the column that reads the speed gets EXACTLY 0.0 when the speed is zero
    assert float(wm.stack.act_embed[0].weight.grad[:, 2].abs().sum()) == 0.0 and float(wm.stack.act_embed[0].weight.grad[:, 0].abs().sum()) > 0.0
    # sensitivity: permuting the speed across the batch (same frames, other windows' speeds) moves the level-1 loss
    wm.eval()
    with torch.no_grad():
        base = float(wm(adapt, act, speed=spd)["pred/L1"])
        perm = float(wm(adapt, act, speed=spd.flip(0))["pred/L1"])
        shuf = float(wm(adapt, act, speed=spd[:, torch.randperm(16, generator=torch.Generator().manual_seed(1))])["pred/L1"])
    assert abs(base - perm) > 1e-6 and abs(base - shuf) > 1e-6
    _, off, *_ = _wm()
    off.eval()
    with torch.no_grad():
        assert float(off(adapt, act, speed=spd)["pred/L1"]) == float(off(adapt, act, speed=spd.flip(0))["pred/L1"])      # lever off: the speed is not read


def test_egov_builds_the_rest_of_the_model_unchanged_in_init():
    cfg_a, a, *_ = _wm()
    cfg_b, b, *_ = _wm(wm_l1_act="speed")
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    same = [k for k in sa if sa[k].shape == sb[k].shape]
    assert [k for k in sa if k not in same] == [f"stack.prior_{n}_1" for n in ("mean", "std", "lo", "hi")] + ["stack.act_embed.0.weight", "stack.idm_heads.0.net.4.weight", "stack.idm_heads.0.net.4.bias"]    # only what is sized by the action width (the planning priors, the embedding, the IDM output)
    assert tuple(sb["stack.act_embed.0.weight"].shape) == (32, 3)


# ------------------------------------------------------------------------------------------------ the data: wm_speed
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("target")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J, "store": store, "ds": ds, "wins": wins, "idx": idx}


def test_wm_speed_is_the_logged_speed_of_every_cached_frame(world):
    b = world["batch"]
    assert b["wm_speed"].shape == (8, 16) and b["wm_speed"].dtype == torch.float32 and torch.isfinite(b["wm_speed"]).all()
    ci, t = world["wins"][int(world["idx"][0])][:2]
    d = world["store"].get(str(world["J"]["clip"][ci]))
    idc = np.clip(np.arange(t - 15, t + 1), 0, len(d["pose"]) - 1)
    assert np.array_equal(b["wm_speed"][0].numpy(), np.nan_to_num(d["v"][idc]).astype(np.float32))
    assert float(b["wm_speed"].abs().max()) > 1.0                         # m/s, not scaled
    assert M.wm_frame_speed(b) is b["wm_speed"]
    assert "wm_speed" not in M.INFERENCE_INPUT_KEYS                         # no model input of the driving path reads it


# ------------------------------------------------------------------------------------------------ config refusals
def test_config_refuses_what_it_cannot_build():
    mk = lambda **kw: M.V8FMiniConfig(wm="b", hmax=16, **kw)             # noqa: E731
    mk()
    mk(wm_l1_act="speed")
    mk(wm_state="grid", wm_l1_futtok_k=2, w_wm_futtok=0.1)
    mk(wm_state="grid", wm_l1_futtok_k=15, w_wm_futtok=0.1)
    for kw, pat in (({"wm_l1_futtok_k": 1, "w_wm_futtok": 0.1}, "need K = 0"), ({"wm_l1_futtok_k": 4, "w_wm_futtok": 0.0}, "need K = 0"),
                    ({"wm_l1_futtok_k": 0, "w_wm_futtok": 0.2}, "need K = 0"), ({"wm_l1_futtok_k": 16, "w_wm_futtok": 0.1}, "does not fit"),
                    ({"wm_l1_futtok_k": 4, "w_wm_futtok": float("nan")}, "finite"), ({"wm_l1_futtok_k": -2, "w_wm_futtok": 0.1}, ">= 0"),
                    ({"wm_l1_futtok_k": 4, "w_wm_futtok": 0.1}, "wm_state grid"), ({"wm_l1_act": "steer"}, "wm_l1_act")):
        with pytest.raises(ValueError, match=pat):
            mk(**kw)
    for kw in ({"wm_l1_act": "speed"}, {"wm_state": "grid", "wm_l1_futtok_k": 4, "w_wm_futtok": 0.1}):
        with pytest.raises(ValueError, match="world-model"):
            M.V8FMiniConfig(wm="off", **kw)


def test_futtok_forward_refuses_a_missing_or_mismatched_token_grid():
    _, wm, adapt, act, spd, feats = _wm(wm_l1_futtok_k=4, w_wm_futtok=1.0)
    with pytest.raises(ValueError, match="no `feats`"):
        wm(adapt, act)
    with pytest.raises(ValueError, match="do not match"):
        wm(adapt, act, feats=feats[:, :, :, :-1])
    assert "tok_head.weight" in wm.state_dict() and not [k for k in _wm()[1].state_dict() if "tok_head" in k]
    assert tuple(wm.tok_head.weight.shape) == (DT, 16)                      # cell width -> token width, one map for every cell


# ------------------------------------------------------------------------------------------------ the trainer: flags, gate, checkpoints
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


E_ON = ["--wm-l1-act", "speed"]
F_ON = ["--wm-l1-futtok-k", "6", "--w-wm-futtok", "1.0"]
EF_CELL = CELL + ["--wm-l1-act", "speed"] + F_ON


def test_flags_have_entries_defaults_and_ckpt_match():
    d = {act.dest for act in TR.build_parser()._actions}
    p = TR.build_parser()
    for f, dflt in (("wm_l1_act", "acc_yaw"), ("wm_l1_futtok_k", 0), ("w_wm_futtok", 0.0)):
        assert f in d and f in TR.FLAG_CHECKS and p.get_default(f) == dflt and f in TR.INIT_CKPT_MATCH
    for bad in (["--wm-l1-futtok-k", "6"], ["--w-wm-futtok", "0.3"], GRID + ["--wm-l1-futtok-k", "16", "--w-wm-futtok", "0.3"], ["--wm-l1-futtok-k", "6", "--w-wm-futtok", "1.0"]):
        with pytest.raises(ValueError):
            TR.make_config(_args(bad))                                    # K with no weight, a weight with no K, K that does not fit hmax 16, FUTTOK with a mean state


@pytest.mark.parametrize("extra", [GRID + E_ON, GRID + F_ON, EF_CELL, E_ON], ids=["E_grid", "F_grid", "EF_cell", "E_mean"])
def test_t6_declared_vs_built_passes_when_built_and_refuses_every_wrong_expectation(world, extra):
    a, cfg, model, opt, info = _built(world, extra, d_feat=world["batch"]["feats"].shape[-1])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    lv = model.built_levers()
    assert lv["wm_l1_act"] == ("speed" if "--wm-l1-act" in extra else "acc_yaw")
    assert lv["wm_l1_futtok_k"] == (6 if "--wm-l1-futtok-k" in extra else 0) and lv["w_wm_futtok"] == (1.0 if "--w-wm-futtok" in extra else 0.0)
    for f, v in (("wm_l1_act", "acc_yaw" if lv["wm_l1_act"] == "speed" else "speed"), ("wm_l1_futtok_k", 10), ("w_wm_futtok", 0.9)):
        wrong = _args(extra)
        setattr(wrong, f, v)
        assert any(f in m for m in TR.declared_vs_built(wrong, cfg, model, opt, info)), f          # a deliberately wrong expectation is refused


def test_declared_but_not_built_is_refused_in_both_directions(world):
    dfeat = world["batch"]["feats"].shape[-1]
    a, cfg, m_on, opt, info = _built(world, GRID + E_ON + F_ON, d_feat=dfeat)
    ad, cd, m_off, od, idf = _built(world, GRID, d_feat=dfeat)
    msgs = TR.declared_vs_built(a, cd, m_off, od, idf)                                              # declared on, built off
    assert all(any(f in m for m in msgs) for f in ("wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok"))
    msgs = TR.declared_vs_built(ad, cfg, m_on, opt, info)                                           # declared off, built on
    assert all(any(f in m for m in msgs) for f in ("wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok"))
    a2, cfg2, m2, o2, i2 = _built(world, GRID + F_ON, d_feat=dfeat)
    m2.wm.stack.cfg[0].futtok_weight = 0.0                                                          # the loss's own weight changed under an argv that declares 1.0
    assert any("w_wm_futtok" in m for m in TR.declared_vs_built(a2, cfg2, m2, o2, i2))
    m2.wm.tok_head = None                                                                           # the head removed from a built model under an 'on' argv
    assert any("wm_l1_futtok_k" in m for m in TR.declared_vs_built(a2, cfg2, m2, o2, i2))
    with pytest.raises(ValueError):
        TR.make_config(_args(["--arm", "M1", "--wm-l1-act", "speed"]))
    a1, c1, m1, o1, i1 = _built(world, ["--arm", "M1"])
    wrong = _args(["--arm", "M1"])
    wrong.wm_l1_futtok_k = 4
    assert any("wm-l1-futtok-k" in m for m in TR.declared_vs_built(wrong, c1, m1, o1, i1))


def _loss(world, extra, **over):
    a, cfg, model, opt, info = _built(world, extra, d_feat=world["batch"]["feats"].shape[-1], **over)
    model.train()
    out = model.loss(dict(world["batch"]), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    return model, out


def test_trainer_loss_logs_the_terms_and_diag_changes_no_value(world):
    _, off = _loss(world, GRID + F_ON)
    _, on = _loss(world, GRID + F_ON + ["--wm-l1-diag", "on"])
    assert sorted(set(on["terms"]) - set(off["terms"])) == ["wm_copy_l1", "wm_futtok_copy", "wm_pred_l1"]
    assert float(on["terms"]["total"]) == float(off["terms"]["total"]) and float(on["terms"]["wm_futtok"]) == float(off["terms"]["wm_futtok"])
    assert "wm_futtok" in off["terms"] and float(off["terms"]["wm_futtok"]) > 0.0 and not off["terms"]["wm_futtok"].requires_grad
    _, e = _loss(world, GRID + E_ON)
    _, base = _loss(world, GRID)
    assert sorted(set(e["terms"]) ^ set(base["terms"])) == [] and float(e["terms"]["wm"]) != float(base["terms"]["wm"])
    assert "wm_futtok" not in base["terms"]
    model, _ = _loss(world, GRID + E_ON)
    # the same call through the model with the key missing is a KeyError from wm_frame_speed (not a silent zero)
    b = dict(world["batch"])
    b.pop("wm_speed")
    with pytest.raises(KeyError, match="wm_speed"):
        model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))


def test_the_trainer_runs_both_levers_and_checkpoint_cfg_keys_follow_the_defaults(world, tmp_path):
    base = ["--synthetic", "--arm", "M2b", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--steps", "3", "--log-every", "1"]
    assert TR.main([*base, "--out", str(tmp_path / "grid")] + GRID) == 0
    assert TR.main([*base, "--out", str(tmp_path / "ef")] + EF_CELL + ["--wm-l1-diag", "on"]) == 0
    ld = lambda n: torch.load(tmp_path / n / "ckpt_last.pt", map_location="cpu", weights_only=False)   # noqa: E731
    cg, ce = ld("grid"), ld("ef")
    assert not (set(M.TARGET_CFG_DEFAULTS) & set(cg["cfg"]))                                     # a default checkpoint keeps its key set
    assert set(ce["cfg"]) - set(cg["cfg"]) == set(M.TARGET_CFG_DEFAULTS) | set(M.L1PRED_CFG_FIELDS) | {"wm_l1_diag"}
    cfg = M.V8FMiniConfig(**ce["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ce["model"])                                                           # strict
    lv = model.built_levers()
    assert (lv["wm_l1_act"], lv["wm_l1_futtok_k"], lv["w_wm_futtok"]) == ("speed", 6, 1.0)
    rows = [json.loads(l) for l in open(tmp_path / "ef" / "train_log.jsonl", encoding="utf-8")]
    assert all(k in rows[-1] and np.isfinite(rows[-1][k]) for k in ("wm_futtok", "wm_futtok_copy", "wm_pred_l1", "wm_copy_l1"))
    rows_g = [json.loads(l) for l in open(tmp_path / "grid" / "train_log.jsonl", encoding="utf-8")]
    assert not [k for k in rows_g[-1] if "futtok" in k]
    dfeat = int(cfg.d_feat)
    ov = {k: ce["cfg"][k] for k in ("d", "ff", "heads", "n_q", "depth_s", "depth_t", "depth_o", "nb", "hmax", "hist")}
    _, c_plain, m_plain, _, _ = _built(world, CELL, d_feat=dfeat, **ov)
    with pytest.raises(ValueError, match="wm_l1_act"):
        TR.load_init_ckpt(str(tmp_path / "ef" / "ckpt_last.pt"), m_plain, c_plain)               # a different lever must not start from this checkpoint
    _, c_f, m_f, _, _ = _built(world, GRID + F_ON, d_feat=dfeat, **ov)
    with pytest.raises(ValueError, match=r"wm_l1_futtok_k: 0 != 6"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m_f, c_f)                      # an older checkpoint (key absent = 0) is refused by an 'on' argv
    _, c_ef, m_ef, _, _ = _built(world, EF_CELL, d_feat=dfeat, **ov)
    assert TR.load_init_ckpt(str(tmp_path / "ef" / "ckpt_last.pt"), m_ef, c_ef)["n_mismatch_after_load"] == 0


def test_g_live_requires_and_sees_the_futtok_term_and_every_group_trains(world):
    dfeat = world["batch"]["feats"].shape[-1]
    for extra in (GRID + F_ON, EF_CELL, GRID + E_ON):
        a, cfg, model, opt, info = _built(world, extra, d_feat=dfeat)
        res = TR.live_check(model, dict(world["batch"]), cfg)
        assert res["missing_terms"] == [] and res["unexpected_dead"] == [], (res["missing_terms"], res["unexpected_dead"])
        assert ("wm_futtok" in res["terms"]) == (cfg.wm_l1_futtok_k > 0)
        if cfg.wm_l1_futtok_k:
            assert res["terms"]["wm_futtok"] > 0.0
