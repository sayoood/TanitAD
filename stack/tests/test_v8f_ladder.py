"""PREREG_V8F_LADDER: the world-model ladder levers (``--wm-rollout-k``, ``--wm-ctx``, ``--wm-rollout-stride``, ``--wm-rollout-w``, ``--wm-windows wmonly``, ``--wm-win-stride``, ``--wm-pooled``, ``--wm-zcache``).

Controls of the card (each with its deliberate-regression mutation, run by ``mutate_ladder.py`` on a COPY of the tree; the test named in the MUTATION table must go RED there):
1. k = 1 identity: with k = 1 the new loss equals the old loss bit for bit (``test_k1_identity_*``).
2. Reach reachability: the gradient reaches each level's predictor through every rollout step in use; a deleted k-step line and unapplied strides go red (``test_reach_*``, ``test_gate_refuses_*``).
3. Mask check: a step past the frames that exist contributes exactly 0.0, a step inside contributes non-zero (``test_mask_*``).
4. Planner identity after a run (``test_a_wmonly_run_*``).
Plus the frozen-adapter pooled path (``test_pooled_*``: equal to the live adapter mean to 1e-6), the world-model-only sampler, the K = 3 history at each level's own step, and the G-DVB entries (every new flag read back from the BUILT model).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_wm_windows as WW  # noqa: E402
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0",
          "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]


# ------------------------------------------------------------------------------------------------ hjepa level: the stack
def lc(**kw):
    return H.LevelConfig(16, kw.pop("action_dim", 8), pred_width=32, pred_depth=1, pred_heads=2, enc_hidden=16, macro_hidden=8, **kw)


def stack_of(k=(1, 1, 1), stride=(1, 1, 1), w=1.0, ctx=(3, 3, 3), s=(3, 2), seed=0):
    torch.manual_seed(seed)
    lv = [lc(action_dim=2, act_embed_dim=8, context=ctx[0], rollout_k=k[0], rollout_stride=stride[0], rollout_w=w),
          lc(stride=s[0], context=ctx[1], rollout_k=k[1], rollout_stride=stride[1], rollout_w=w),
          lc(stride=s[1], context=ctx[2], rollout_k=k[2], rollout_stride=stride[2], rollout_w=w)]
    return H.HJEPAStack(lv, sigreg_projections=16)


def data(b=3, t=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, 16, generator=g), torch.randn(b, t - 1, 2, generator=g)


def ref_loss(st, z1, a_seq):
    """The loss of the BASE tree's ``HJEPAStack.loss`` (copied from the tip's text): used to pin control 1."""
    enc = st.encode(z1, a_seq)
    out, total = {}, z1.new_zeros(()).to(torch.float32)
    for lv in range(1, st.n_levels + 1):
        c = st.cfg[lv - 1]
        z, a = enc.z[lv - 1], enc.act[lv - 1]
        pred = st._pred_loss(lv, z, a, False)
        sg = st.sigreg(z.transpose(0, 1), generator=None)
        idm = st.idm_heads[lv - 1].loss(z, enc.idm_target[lv - 1])
        level_loss = pred + c.beta * sg + c.gamma * idm
        out[f"pred/L{lv}"], out[f"sigreg/L{lv}"], out[f"idm/L{lv}"] = pred, sg, idm
        if lv >= 2:
            asg = st.sigreg(a.transpose(0, 1), generator=None)
            out[f"act_sigreg/L{lv}"] = asg
            level_loss = level_loss + c.act_sigreg * asg
        out[f"level/L{lv}"] = level_loss
        total = total + level_loss
    out["total"] = total
    return out


def test_k1_identity_bit_equal_to_the_old_loss():
    z1, a = data()
    for extra in ({}, {"k": (1, 1, 1), "stride": (4, 5, 6), "w": 7.5}):          # k = 1: stride and weight are inert
        st = stack_of(**extra)
        torch.manual_seed(3)
        new = st.loss(z1, a)
        torch.manual_seed(3)
        old = ref_loss(st, z1, a)
        assert list(new) == list(old), "the loss dict grew a key at k = 1"
        assert all(torch.equal(new[k], old[k]) for k in old), [k for k in old if not torch.equal(new[k], old[k])]


def test_k1_identity_holds_inside_the_wm_aux_and_its_state_dict_is_unchanged():
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=28, kin_mode="yaw")
    torch.manual_seed(0)
    w = M.WMAux(cfg)
    assert cfg.wm_rollout_k == (1, 1, 1) and cfg.wm_ctx == (3, 3, 3)
    adapt = torch.randn(2, 28, 40, 32)
    act = torch.randn(2, 28, 2)
    torch.manual_seed(1)
    got = w(adapt, act)
    torch.manual_seed(1)
    z1 = w.z1(adapt)
    old = ref_loss(w.stack, z1, w.scale_actions(act)[:, :-1])
    assert all(torch.equal(got[k], old[k]) for k in old) and list(got) == list(old)
    assert not any("rollout" in k for k in w.stack.built_levers()[2]) and "wm_rollout_k" not in M.V8FMini(cfg, M.P10Prior.synthetic(0)).built_levers()


@pytest.mark.parametrize("ks,stride", [((4, 3, 2), (2, 2, 1)), ((6, 4, 3), (1, 1, 1))])
def test_reach_gradient_flows_through_every_rollout_step_of_every_level(ks, stride):
    z1, a = data(b=2, t=60)
    st = stack_of(k=ks, stride=stride)
    enc = st.encode(z1, a)
    for lv in (1, 2, 3):
        z, ap = enc.z[lv - 1], (enc.idm_target[0] if lv == 1 else enc.act[lv - 1])
        k = ks[lv - 1]
        # one anchor only (stride past the window): step h depends on the actions 0..h and on no later action
        ap = ap.detach().clone().requires_grad_(True)
        ro = st.rollout_loss(lv, z.detach(), ap, k, stride=10 ** 6)
        for h in range(k):
            g, = torch.autograd.grad(ro["contrib"][:, 0, h].sum(), ap, retain_graph=True)
            gn = g.abs().sum(dim=(0, 2))
            assert (gn[:h + 1] > 0).all(), f"level {lv} step {h}: the action it conditions on gets no gradient"
            assert (gn[h + 1:] == 0).all(), f"level {lv} step {h}: a LATER action leaked into the step"
        for p_ in st.predictors[lv - 1].parameters():
            p_.grad = None
        out = st.loss(z1, a)
        out[f"rollout/L{lv}"].backward()
        assert sum(float(p_.grad.abs().sum()) for p_ in st.predictors[lv - 1].parameters() if p_.grad is not None) > 0, f"level {lv}: the k-step term does not reach its predictor"
        st.zero_grad(set_to_none=True)


def test_reach_the_term_enters_the_total_with_exactly_its_weight():
    z1, a = data()
    for w in (0.0, 1.0, 2.5):
        st = stack_of(k=(3, 2, 2), stride=(2, 2, 1), w=w)
        torch.manual_seed(4)
        new = st.loss(z1, a)
        torch.manual_seed(4)
        old = ref_loss(st, z1, a)
        extra = sum(w * new[f"rollout/L{lv}"] for lv in (1, 2, 3))
        assert float((new["total"] - old["total"] - extra).abs()) < 1e-4 * max(1.0, float(new["total"])), w
    st = stack_of(k=(3, 2, 2), stride=(2, 2, 1), w=1.0)
    g1 = torch.autograd.grad(st.loss(z1, a)["total"], list(st.predictors[0].parameters()), allow_unused=True)
    st0 = stack_of(k=(3, 2, 2), stride=(2, 2, 1), w=0.0)
    g0 = torch.autograd.grad(st0.loss(z1, a)["total"], list(st0.predictors[0].parameters()), allow_unused=True)
    assert any(not torch.equal(x, y) for x, y in zip(g1, g0)), "weight 1 and weight 0 give the same gradient: the term is not in the total"


def test_built_rollout_reads_the_loops_the_loss_really_runs():
    st = stack_of(k=(4, 3, 2), stride=(2, 2, 1), w=2.5, ctx=(3, 2, 1))
    rb = st.built_rollout(40)
    assert rb["k"] == [4, 3, 2] and rb["ctx_built"] == [3, 2, 1] and rb["ctx_passed"] == rb["ctx_built"]
    assert rb["anchors"] == [len(range(0, rb["states"][i] - 1, s)) for i, s in enumerate((2, 2, 1))]
    assert all(abs(x - 2.5) < 1e-4 for x in rb["w_in_loss"])
    rb1 = stack_of().built_rollout(40)
    assert rb1["k"] == [0, 0, 0] and rb1["w_in_loss"] == [None] * 3 and rb1["ctx_built"] == [3, 3, 3]


@pytest.mark.parametrize("k,stride", [(10, 1), (6, 1), (5, 3)])
def test_mask_a_step_past_the_frames_contributes_exactly_zero_and_a_step_inside_does_not(k, stride):
    z1, a = data(b=2, t=22)
    st = stack_of(s=(2, 2))
    enc = st.encode(z1, a)
    for lv in (1, 2, 3):
        z, ap = enc.z[lv - 1].detach(), (enc.idm_target[0] if lv == 1 else enc.act[lv - 1]).detach()
        T = z.shape[1]
        ro = st.rollout_loss(lv, z, ap, k, stride=stride)
        idx = np.arange(0, T - 1, stride)
        valid = (idx[:, None] + np.arange(k)[None, :] + 1) <= (T - 1)
        c = ro["contrib"].detach().numpy()
        assert (c[:, ~valid] == 0.0).all() and (c[:, valid] > 0.0).all(), f"level {lv}: mask wrong"
        assert float(ro["n_valid"]) == 2 * valid.sum() and float(ro["n_masked"]) == 2 * (~valid).sum()
        assert abs(float(ro["loss"]) - c.sum() / (2 * valid.sum())) < 1e-6
        if (~valid).any():                                             # moving a masked step's action changes nothing
            ap2 = ap.clone()
            ap2[:, -1] += 100.0
            ro2 = st.rollout_loss(lv, z, ap2, k, stride=stride)
            assert torch.equal(ro2["contrib"][:, ~torch.as_tensor(valid)], ro["contrib"][:, ~torch.as_tensor(valid)])


def test_mask_the_l3_of_the_card_counts_its_masked_steps():
    """91 frames, strides (3, 5): L3 holds 7 states; k = 6 at stride 1 anchors 0..5 give 6+5+4+3+2+1 = 21 valid steps and 15 masked ones per sample."""
    st = stack_of(s=(3, 5))
    z = torch.randn(2, 7, 16)
    a = torch.randn(2, 6, 8)
    ro = st.rollout_loss(3, z, a, 6, stride=1)
    assert float(ro["n_valid"]) == 2 * 21 and float(ro["n_masked"]) == 2 * 15


def test_wm_ctx_sets_the_predictor_parameter_shape_and_the_hmax_guard_refuses_an_unreachable_k():
    st = stack_of(ctx=(1, 2, 3))
    assert [int(p.pos.shape[0]) for p in st.predictors] == [1, 2, 3]
    base = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", kin_mode="yaw", wm_strides=(3, 5))
    M.V8FMiniConfig(**base, hmax=91, wm_rollout_k=(10, 10, 6))
    with pytest.raises(ValueError, match="could never be trained"):
        M.V8FMiniConfig(**base, hmax=90, wm_rollout_k=(10, 10, 6))
    with pytest.raises(ValueError, match="no world model"):
        M.V8FMiniConfig(**{**base, "wm": "off"}, hmax=91, wm_rollout_k=(2, 1, 1))


def test_history_k3_reads_each_level_at_its_own_step():
    """R35H: --wm-k 3 on strides (3, 5): L1 frames t, t-1, t-2 (0.4 s); L2 states 3 frames apart (1.2 s); L3 states 15 frames apart (6.0 s)."""
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=91, kin_mode="yaw", wm_k=3, wm_strides=(3, 5), wm_rollout_k=(10, 10, 6))
    w = M.WMAux(cfg)
    bk = w.built_k()
    assert bk["k"] == [3, 3, 3] and bk["dilation"] == [1, 3, 5] and bk["causal_pad"] == [True, True, True]
    span_s = [0.2 * (k - 1) * d * c for k, d, c in zip(bk["k"], bk["dilation"], (1, 1, 3))]
    assert [round(x, 2) for x in span_s] == [0.4, 1.2, 6.0]
    assert w.stack.states_per_level(91) == [91, 31, 7]


# ------------------------------------------------------------------------------------------------ the pooled path
@pytest.mark.parametrize("d", [32, 256])
def test_pooled_path_equals_the_live_adapter_mean_to_1e6(d):
    cfg = M.V8FMiniConfig(d=d, ff=2 * d, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=28, kin_mode="yaw")
    torch.manual_seed(0)
    m = M.V8FMini(cfg, M.P10Prior.synthetic(0))
    g = torch.Generator().manual_seed(2)
    feats = (torch.randn(3, 9, cfg.n_tok, cfg.d_feat, generator=g) * 2.0 + 0.3).to(torch.float16)
    live = m._encode({"feats": feats}).mean(2)
    nm = WW.norm_mean_tokens(feats)
    assert nm.shape == (3, 9, cfg.d_feat) and nm.dtype == torch.float32
    pooled = m.pooled_adapt_mean(nm)
    chunked = m.pooled_adapt_mean_from_feats(feats, chunk=4)
    assert float((pooled - live).abs().max()) < 1e-6, float((pooled - live).abs().max())
    assert float((chunked - live).abs().max()) < 1e-6
    # the WM loss from the three routes agrees
    act = torch.randn(3, 9, 2)
    torch.manual_seed(9)
    t_live = m.wm_only_loss({"feats": feats, "act": act, "a_t": 0})["terms"]
    torch.manual_seed(9)
    t_pool = m.wm_only_loss({"pool": nm, "act": act})["terms"]
    torch.manual_seed(9)
    t_z = m.wm_only_loss({"zmean": live, "act": act})["terms"]
    for k in t_live:
        assert abs(float(t_live[k]) - float(t_pool[k])) < 1e-4 * max(1.0, abs(float(t_live[k]))), k
        assert abs(float(t_live[k]) - float(t_z[k])) < 1e-4 * max(1.0, abs(float(t_live[k]))), k


# ------------------------------------------------------------------------------------------------ the world-model-only sampler
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("synl")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    return {"root": root, "J": J, "store": store, "clips": w["names"]["train"]}


def test_sampler_has_no_label_requirement_and_every_window_is_full(world):
    J, store, clips = world["J"], world["store"], world["clips"]
    F = 25
    wins = WW.wm_windows(store, J, clips, F=F)
    train = D.train_windows(store, J, clips)
    assert len(wins) == sum(int(J["n"][i]) - F + 1 for i in range(len(clips)))
    assert max(w[1] for w in wins) == 59 > max(w[1] for w in train)         # reaches the clip end; train_windows stops 30 steps short
    assert min(w[1] for w in wins) == F - 1
    assert WW.wm_windows(store, J, clips, F=61) == []                         # a clip shorter than F has no window
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, pooled=True)
    ds2 = WW.V8FMiniWMWindows(store, J, wins, F=F, pooled=False)
    b, b2 = WW.collate_wm([ds.item(*wins[3]), ds.item(*wins[40])]), WW.collate_wm([ds2.item(*wins[3]), ds2.item(*wins[40])])
    assert b["pool"].shape == (2, F, 1024) and b2["feats"].shape[:2] == (2, F) and "feats" not in b
    assert torch.allclose(b["pool"], WW.norm_mean_tokens(b2["feats"]), atol=1e-6) and torch.equal(b["act"], b2["act"])


def test_zcache_dataset_reads_the_per_clip_file_and_refuses_a_missing_or_not_ok_frame(world, tmp_path):
    J, store, clips = world["J"], world["store"], world["clips"]
    F = 25
    wins = WW.wm_windows(store, J, clips, F=F)
    d = store.get(clips[0])
    zm = np.random.default_rng(0).standard_normal((len(d["pose"]), 32)).astype(np.float32)
    ok = np.ones(len(zm), bool)
    np.savez(tmp_path / f"{clips[0]}.npz", zmean=zm, ok=ok, t_s=np.arange(len(zm), dtype=np.float32) * 0.2)
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path))
    assert ds.zcache_missing(clips) == clips[1:]
    w0 = [w for w in wins if str(J["clip"][w[0]]) == clips[0]][0]
    it = ds.item(*w0)
    assert np.array_equal(it["zmean"], zm[w0[1] - F + 1:w0[1] + 1]) and "pool" not in it
    ok[w0[1] - 3] = False
    np.savez(tmp_path / f"{clips[0]}.npz", zmean=zm, ok=ok, t_s=np.arange(len(zm), dtype=np.float32) * 0.2)
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path))
    with pytest.raises(IndexError):
        ds.item(*w0)


def test_zcache_prewarm_leaves_the_windows_identical_and_reads_no_clip_file_afterwards(world, tmp_path):
    J, store, clips = world["J"], world["store"], world["clips"]
    F = 25
    wins = WW.wm_windows(store, J, clips, F=F)
    for c in clips:
        n = len(store.get(c)["pose"])
        np.savez(tmp_path / f"{c}.npz", zmean=np.random.default_rng(1).standard_normal((n, 32)).astype(np.float32), ok=np.ones(n, bool), t_s=np.arange(n, dtype=np.float32) * 0.2)
    cold = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path))
    warm = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path))
    assert warm.prewarm() == len(clips)
    calls = []
    real = store.get
    store.get = lambda c: (calls.append(c), real(c))[1]
    try:
        got = [warm.item(*w) for w in wins[:40]]
    finally:
        store.get = real
    assert calls == []
    for w, g in zip(wins[:40], got):
        c = cold.item(*w)
        assert np.array_equal(c["act"], g["act"]) and np.array_equal(c["zmean"], g["zmean"])


def test_ok_windows_drops_exactly_the_windows_holding_a_frame_that_is_not_ok(world, tmp_path):
    J, store, clips = world["J"], world["store"], world["clips"]
    F = 25
    wins = WW.wm_windows(store, J, clips, F=F)
    bad_clip, bad_frame = clips[0], F + 2
    for c in clips:
        n = len(store.get(c)["pose"])
        ok = np.ones(n, bool)
        if c == bad_clip:
            ok[bad_frame] = False
        np.savez(tmp_path / f"{c}.npz", zmean=np.zeros((n, 32), np.float32), ok=ok, t_s=np.arange(n, dtype=np.float32) * 0.2)
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path))
    hit = lambda w: str(J["clip"][w[0]]) == bad_clip and w[1] - F + 1 <= bad_frame <= w[1]
    want = [w for w in wins if not hit(w)]
    assert len(want) < len(wins)
    assert ds.ok_windows() == want and ds.windows == want
    for w in want[:30]:
        ds.item(*w)


# ------------------------------------------------------------------------------------------------ the trainer: G-DVB and a run
@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    ck = root / "m2b" / "ckpt_last.pt"
    return {"root": root, "ck": ck, "blob": torch.load(ck, map_location="cpu", weights_only=False)}


def gate(init, name, extra=(), ck=True, args=None):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *BASE, *(["--init-ckpt", str(init["ck"])] if ck else []), *(LADDER if args is None else args), *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def test_every_ladder_flag_has_a_declared_vs_built_entry_and_defaults_are_the_earlier_behaviour():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    new = {"wm_rollout_k", "wm_rollout_stride", "wm_rollout_w", "wm_ctx", "wm_windows", "wm_win_stride", "wm_pooled", "wm_zcache"}
    assert new <= dests and dests == set(TR.FLAG_CHECKS) and set(TR.WM_LADDER_FLAGS) == new
    a = TR.build_parser().parse_args(["--out", "x"])
    assert TR.declared_wm_ladder(a) == {"wm_rollout_k": (1, 1, 1), "wm_rollout_stride": (1, 1, 1), "wm_ctx": (3, 3, 3), "wm_rollout_w": 1.0}
    assert (a.wm_windows, a.wm_win_stride, a.wm_pooled, a.wm_zcache) == ("train", 1, "off", "") and TR.wm_ladder_named(a) == []


def test_gate_passes_the_ladder_argv_and_reads_every_lever_back_from_the_built_model(init):
    assert gate(init, "g_ok") == 0
    rb = sc_of(init, "g_ok")["levers"]["wm_ladder_readback"]
    assert rb["built"]["k"] == [4, 3, 2] and rb["built"]["ctx_built"] == [3, 3, 3] and rb["built"]["anchors"] == [len(range(0, n - 1, s)) for n, s in zip(rb["built"]["states"][:2] + [rb["built"]["states"][2]], (2, 2, 1))]
    assert rb["windows_built"]["kind"] == "wmonly" and rb["windows_built"]["F"] == 25 and rb["windows_built"]["t_min"] == 24 and rb["hmax"] == 25
    assert gate(init, "g_pool", ["--wm-pooled", "on"]) == 0 and sc_of(init, "g_pool")["levers"]["wm_ladder_readback"]["windows_built"]["pooled"] is True


def test_gate_passes_a_non_default_ctx_and_weight_and_reads_them_back(init):
    """The card holds ctx at (3, 3, 3) and w at 1.0, both the defaults, so a deleted config line would be invisible to the card's own argv: pin the lever with values that are NOT the defaults."""
    assert gate(init, "g_ctx", args=[*LADDER[:-6], "--wm-ctx", "2", "3", "3", "--wm-rollout-w", "0.5", "--wm-windows", "wmonly"]) == 0
    rb = sc_of(init, "g_ctx")["levers"]["wm_ladder_readback"]
    assert rb["built"]["ctx_built"] == [2, 3, 3], rb["built"]
    assert sc_of(init, "g_ctx")["levers"]["wm_ladder_readback"]["declared"]["wm_rollout_w"] == 0.5


def test_gate_refuses_a_ladder_lever_the_built_model_does_not_carry_k(init, monkeypatch):
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rollout_k=(1, 1, 1)))        # the k-step lines deleted: the config is built at k = 1
    rc = gate(init, "r_k")
    assert isinstance(rc, str) and "lever wm_rollout_k" in rc, rc


def test_gate_refuses_a_ladder_lever_the_built_model_does_not_carry_ctx_stride_weight(init, monkeypatch):
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_ctx=(3, 3, 3)))
    rc = gate(init, "r_ctx", args=[*LADDER, "--wm-ctx", "2", "3", "3"])
    assert isinstance(rc, str) and "lever wm_ctx" in rc, rc
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rollout_stride=(1, 1, 1)))
    rc = gate(init, "r_stride")
    assert isinstance(rc, str) and "lever wm_rollout_stride" in rc, rc
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_rollout_w=0.5))
    rc = gate(init, "r_w")
    assert isinstance(rc, str) and "lever wm_rollout_w" in rc, rc


def test_gate_refuses_the_wm_strides_the_built_model_does_not_carry(init, monkeypatch):
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_strides=(2, 2)))             # the strides line deleted: built at (2, 2) under a (2, 3) declaration
    rc = gate(init, "r_strides")
    assert isinstance(rc, str) and "wm_strides" in rc, rc


def test_gate_refuses_every_lever_that_is_declared_and_builds_nothing(init):
    cases = {
        "no_wm_only": ([x for x in LADDER if x != "--wm-only"], "wmonly declared without --wm-only"),
        "stride_on_k1": (["--hmax", "25", "--wm-only", "--wm-windows", "wmonly", "--wm-rollout-k", "1", "1", "1", "--wm-rollout-stride", "2", "1", "1"], "k is 1"),
        "w_on_k1": (["--hmax", "25", "--wm-only", "--wm-windows", "wmonly", "--wm-rollout-w", "2.0"], "wm_rollout_w"),
        "pooled_train": (["--hmax", "28", "--wm-only", "--wm-pooled", "on"], "wm_pooled"),
        "winstride_train": (["--hmax", "28", "--wm-only", "--wm-win-stride", "3"], "wm_win_stride"),
        "zcache_train": (["--hmax", "28", "--wm-only", "--wm-zcache", "somewhere"], "wm_zcache"),
        "zcache_missing": (["--hmax", "25", "--wm-only", "--wm-windows", "wmonly", "--wm-zcache", "no_such_dir"], "no file"),
    }
    for name, (args, needle) in cases.items():
        rc = gate(init, f"x_{name}", args=args, ck=False)    # no --init-ckpt: the declared-vs-built flag checks refuse before any checkpoint comparison
        assert isinstance(rc, str) and needle in rc, (name, rc)
    try:                                                                       # a reach that cannot fit the window is refused by the config itself (ValueError) or by the gate (str)
        rc = gate(init, "x_short", args=["--hmax", "25", "--wm-only", "--wm-windows", "wmonly", "--wm-rollout-k", "4", "20", "2"], ck=False)
    except ValueError as e:
        rc = str(e)
    assert isinstance(rc, str) and "could never be trained" in rc, rc


def test_the_default_argv_is_unchanged_and_records_no_ladder_key(init):
    assert TR.main(["--out", str(init["root"] / "dflt"), *BASE, "--selfcheck-only"]) == 0
    sc = sc_of(init, "dflt")
    assert "wm_ladder_readback" not in sc["levers"]


@pytest.mark.parametrize("pooled", ["off", "on"])
def test_a_wmonly_run_trains_the_ladder_and_leaves_the_planner_bit_equal(init, pooled):
    out = init["root"] / f"run_{pooled}"
    assert TR.main(["--out", str(out), *BASE, "--init-ckpt", str(init["ck"]), *LADDER, "--wm-pooled", pooled, "--steps", "3", "--log-every", "1", "--ckpt-every", "100"]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["done"] is True and s["wm_only"] is True and s["wm_ladder"]["declared"]["wm_rollout_k"] == [4, 3, 2]
    log = [json.loads(ln) for ln in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and all(np.isfinite(r["wm"]) for r in log)
    ck1 = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    a, b = init["blob"]["model"], ck1["model"]
    rest = [n for n in a if not n.startswith("wm.")]
    assert rest and all(torch.equal(a[n], b[n]) for n in rest), "a non-WM tensor moved"                         # control 4: the planner, readers, heads, adapter are bit-equal
    sc = sc_of(init, f"run_{pooled}")
    assert sc["planner_identity_probe"]["pass"] is True and sc["levers"]["wm_ladder_readback"]["windows_built"]["pooled"] is (pooled == "on")
    assert ck1["cfg"]["wm_rollout_k"] in ((4, 3, 2), [4, 3, 2]) and ck1["cfg"]["wm_strides"] in ((2, 3), [2, 3])
    moved = [n for n in b if n.startswith("wm.stack.predictors.") and n in a and not torch.equal(a[n], b[n])] if any(n.startswith("wm.stack.predictors.") for n in a) else ["fresh"]
    assert moved


def test_g_live_passes_with_the_rollout_terms_and_the_non_wm_control_reads_exactly_zero():
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=25, kin_mode="yaw", wm_strides=(2, 3), wm_rollout_k=(4, 3, 2), wm_rollout_stride=(2, 2, 1))
    torch.manual_seed(0)
    m = M.V8FMini(cfg, M.P10Prior.synthetic(0))
    TR.freeze_all_but_wm(m)
    g = torch.Generator().manual_seed(1)
    batch = {"feats": torch.randn(4, 25, 40, 1024, generator=g).to(torch.float16), "act": torch.randn(4, 25, 2, generator=g)}
    res = TR.wm_only_live_check(m, batch, cfg)
    assert res["pass"] and res["control_nonwm_grad_sum"] == 0.0 and res["control_adapter_grad_sum"] == 0.0
    assert all(f"wm/rollout/L{l}" in res["terms"] for l in (1, 2, 3))
    assert res["dead_wm_groups"] == [] and res["unexpected_dead"] == []
