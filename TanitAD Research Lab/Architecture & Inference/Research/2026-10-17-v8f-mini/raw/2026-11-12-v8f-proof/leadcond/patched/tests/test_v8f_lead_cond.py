"""LEADCOND (`--lead-cond {off,on}`): T and O read u = (p, p * g) of the LEADSUP head at the current frame, DETACHED, through a zero-initialised Linear(2, d) ADDED to their ego token.

Tiny models (d 32) on CPU. A test that the mechanism WORKS is not evidence that it RUNS: every reach test traces the token to the outputs and the weights, with a control that must read EXACTLY 0 or a mutation that must go red.
Run with the tree's own stack on the path (the helper below inserts it)."""
from __future__ import annotations

import copy
import inspect
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
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
LS = GRID + ["--lead-sup", "on", "--w-lead-sup", "1.0"]
LC = LS + ["--lead-cond", "on"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("leadcond")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J}


def _built(world, extra=(), seed=0, **over):
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])
    c0 = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(world["J"])
    torch.manual_seed(seed)
    return a, cfg, M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)


def _wake(model, seed=1):
    """Non-trivial heads: O's zero-init head and the lead head's read-out get random weights, so the plan and u can respond."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head_o[2].weight.normal_(0, 0.1, generator=g)
        for m in model.wm.lead_head.mlp:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
    return model


def _inf(batch):
    return {k: v for k, v in batch.items() if k in M.INFERENCE_INPUT_KEYS}


# ------------------------------------------------------------------------------------------------ default off
def test_off_builds_nothing_and_adds_no_key(world):
    _, cfg, m = _built(world, LS)
    assert m.lead_cond is None and cfg.lead_cond == "off"
    assert not [k for k in m.state_dict() if k.startswith("lead_cond")]
    assert "lead_cond" not in m.built_levers() and "LEADCOND" not in m.trainable_groups()
    cfg_rec = {k: v for k, v in cfg.__dict__.items() if (k not in M.LEADCOND_CFG_DEFAULTS or v != M.LEADCOND_CFG_DEFAULTS[k])}
    assert "lead_cond" not in cfg_rec                                              # a flag-off checkpoint keeps its cfg key set
    _, cfg_on, m_on = _built(world, LC)
    rec_on = {k: v for k, v in cfg_on.__dict__.items() if (k not in M.LEADCOND_CFG_DEFAULTS or v != M.LEADCOND_CFG_DEFAULTS[k])}
    assert rec_on["lead_cond"] == "on" and m_on.built_levers()["lead_cond"] == "on" and "LEADCOND" in m_on.trainable_groups()


def test_on_without_lead_sup_and_unknown_values_are_refused(world):
    with pytest.raises(ValueError, match="lead_sup"):
        _built(world, GRID + ["--lead-cond", "on"])
    with pytest.raises(ValueError, match="lead_cond"):
        _built(world, LS, lead_cond="maybe")
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", *LS, "--lead-cond", "maybe"])
    assert TR.build_parser().get_default("lead_cond") == "off" and "lead_cond" in TR.FLAG_CHECKS


# ------------------------------------------------------------------------------------------------ identity at step 0
def test_zero_init_outputs_equal_the_off_model_exactly(world):
    _, _, off = _built(world, LS, seed=4)
    _, _, on = _built(world, LC, seed=4)
    res = on.load_state_dict(off.state_dict(), strict=False)
    assert sorted(res.missing_keys) == ["lead_cond.bias", "lead_cond.weight"] and not res.unexpected_keys
    _wake(off)
    on.load_state_dict(off.state_dict(), strict=False)
    off.eval(), on.eval()
    b = world["batch"]
    with torch.no_grad():
        for fn in (lambda m: m.forward(dict(b), p_gt=0.0), lambda m: m.forward_infer(_inf(b))):
            o0, o1 = fn(off), fn(on)
            ks = [k for k in o0 if torch.is_tensor(o0[k])]
            assert len(ks) >= 8 and all(torch.equal(o0[k], o1[k]) for k in ks)
    assert float(on.lead_cond.weight.abs().sum()) == 0.0 and float(on.lead_cond.bias.abs().sum()) == 0.0


def test_build_moves_no_other_draw_and_no_other_init(world):
    _, _, off = _built(world, LS, seed=7)
    d_off = torch.rand(3)
    _, _, on = _built(world, LC, seed=7)
    d_on = torch.rand(3)
    assert torch.equal(d_off, d_on)                                                 # the next draw after the build is the same: nothing consumed the stream
    so, sn = off.state_dict(), on.state_dict()
    assert sorted(set(sn) - set(so)) == ["lead_cond.bias", "lead_cond.weight"] and all(torch.equal(so[k], sn[k]) for k in so)
    _, _, _ = _built(world, LS, seed=7)
    torch.nn.Linear(2, 32)                                                          # control: an unrestored default init DOES move the next draw
    assert not torch.equal(torch.rand(3), d_off)


# ------------------------------------------------------------------------------------------------ reach
def _u_recorder(model):
    rec = []
    orig = model._lead_u

    def spy(b, adapt, last):
        u = orig(b, adapt, last)
        rec.append(u.detach().clone())
        return u

    model._lead_u = spy
    return rec, orig


def test_token_reaches_operative_and_tactical_outputs_and_the_weight_gradient(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    b = world["batch"]
    B = b["feats"].shape[0]
    code = torch.full((B, M.CODE_DIM), 0.25)
    m.eval()
    with torch.no_grad():
        base = m.forward_diag(_inf(b), code)
        g = torch.Generator().manual_seed(2)
        m.lead_cond.weight.copy_(torch.randn(m.lead_cond.weight.shape, generator=g) * 0.5)
        moved = m.forward_diag(_inf(b), code)
    assert float((moved["traj60"] - base["traj60"]).abs().max()) > 0.0                # O alone (the code O reads is held fixed)
    assert max(float((moved[k] - base[k]).abs().max()) for k in ("goal_logits", "lat_logits", "lon_logits")) > 0.0     # T
    m.train()
    m.zero_grad(set_to_none=True)
    m.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]["total"].backward()
    gw, gb = float(m.lead_cond.weight.grad.abs().sum()), float(m.lead_cond.bias.grad.abs().sum())
    assert gw > 0.0 and gb > 0.0
    m.zero_grad(set_to_none=True)
    m._lead_u = lambda bb, adapt, last: torch.zeros(adapt.shape[0], 2)                # control: u forced to zeros
    m.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]["total"].backward()
    assert float(m.lead_cond.weight.grad.abs().sum()) == 0.0 and float(m.lead_cond.bias.grad.abs().sum()) > 0.0


def test_no_gradient_reaches_the_head_or_the_trunk_through_u(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    with torch.no_grad():
        m.lead_cond.weight.fill_(0.4)
    b = world["batch"]
    m.train()
    adapt = m._encode(b)
    last = torch.full((adapt.shape[0],), adapt.shape[1] - 1, dtype=torch.long)
    m.lead_cond(m._lead_u(b, adapt, last)).sum().backward()
    through = [p for n, p in m.named_parameters() if n.startswith(("adapter", "pos", "time", "wm.proj", "wm.lead_head")) and p.grad is not None and float(p.grad.abs().sum()) > 0.0]
    assert not through                                                                # exactly 0 (None or zero)
    m.zero_grad(set_to_none=True)
    h = cfg.lead_hist
    pos = last[:, None] - (h - 1 - torch.arange(h))[None]
    win = torch.gather(adapt, 1, pos[:, :, None, None].expand(-1, -1, adapt.shape[2], adapt.shape[3]))
    o = m.wm.lead_head(m.wm.z1(win))[:, 0]
    u_nd = torch.stack([torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 0]) * o[:, 1].clamp(0, M.LEAD_COND_GAP_MAX)], -1)
    m.lead_cond(u_nd).sum().backward()                                                 # control: u built WITHOUT the detach reaches the head and the adapter
    assert float(sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("wm.lead_head") and p.grad is not None)) > 0.0
    assert float(sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("adapter") and p.grad is not None)) > 0.0


def test_the_two_adding_lines_exist_once_each():
    src_o, src_t = inspect.getsource(M.V8FMini.run_o), inspect.getsource(M.V8FMini.run_t)
    assert src_o.count("ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, i))") == 1
    assert src_t.count("ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, torch.full_like(i, adapt.shape[1] - 1)))") == 1


def test_deleting_the_token_line_in_either_layer_reads_zero_reach(world):
    """Mutation: replace the adding line of ONE layer by a no-op (a copy of the module source), build the model from the mutated class, read the reach: that layer's output does not move."""
    import types
    results = {}
    for layer, needle in (("O", "            ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, i))\n"),
                          ("T", "            ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, torch.full_like(i, adapt.shape[1] - 1)))\n")):
        src = inspect.getsource(M)
        assert src.count(needle) == 1
        mod = types.ModuleType("v8f_mini_mut_" + layer)
        mod.__dict__["__file__"] = M.__file__
        sys.modules[mod.__name__] = mod                                               # a dataclass resolves its annotations through sys.modules
        exec(compile(src.replace(needle, "            pass\n"), M.__file__, "exec"), mod.__dict__)
        _, cfg, _ = _built(world, LC)
        mu, sd = M.cons_stats(world["J"])
        torch.manual_seed(3)
        m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
        _wake(m)
        m.eval()
        b = world["batch"]
        code = torch.full((b["feats"].shape[0], M.CODE_DIM), 0.25)
        with torch.no_grad():
            base = m.forward_diag(_inf(b), code)
            m.lead_cond.weight.copy_(torch.randn(m.lead_cond.weight.shape, generator=torch.Generator().manual_seed(2)) * 0.5)
            moved = m.forward_diag(_inf(b), code)
        results[layer] = (float((moved["traj60"] - base["traj60"]).abs().max()), max(float((moved[k] - base[k]).abs().max()) for k in ("goal_logits", "lat_logits", "lon_logits")))
    for layer in ("O", "T"):
        sys.modules.pop("v8f_mini_mut_" + layer, None)
    assert results["O"][0] == 0.0 and results["O"][1] > 0.0                           # O's line deleted: O alone is blind, T still moves
    assert results["T"][1] == 0.0 and results["T"][0] > 0.0                           # T's line deleted: T is blind, O still moves


# ------------------------------------------------------------------------------------------------ time and vision only, parity
def test_u_reads_only_the_window_frames_and_no_ego_input(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    with torch.no_grad():
        m.lead_cond.weight.fill_(0.3)
    m.eval()
    rec, orig = _u_recorder(m)
    b = world["batch"]
    H, F_ = cfg.lead_hist, b["feats"].shape[1]

    def u_of(bb):
        rec.clear()
        with torch.no_grad():
            m.forward_infer(_inf(bb))
        return [r.clone() for r in rec]

    base = u_of(b)
    assert len(base) == 2 and float(base[-1].abs().sum()) > 0.0
    g = torch.Generator().manual_seed(9)
    b1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    n = 0
    for k, v in b.items():
        if k in ("feats", "frame_ok") or not torch.is_tensor(v) or v.dtype == torch.bool:
            continue
        b1[k] = v + (torch.randn(v.shape, generator=g, dtype=v.dtype) * 3.0 if v.dtype.is_floating_point else 1)
        n += int(not torch.equal(b1[k], v))
    assert n >= 12 and all(not torch.equal(b1[k], b[k]) for k in ("act", "p10f", "nav", "pose", "limit", "base"))
    assert all(torch.equal(x, y) for x, y in zip(base, u_of(b1)))                      # every ego input perturbed: u exactly unchanged
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["feats"][:, : F_ - H] += 3.0
    assert all(torch.equal(x, y) for x, y in zip(base, u_of(b2)))                      # frames before the window
    b3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b3["feats"][:, -1] += 3.0
    assert any(not torch.equal(x, y) for x, y in zip(base, u_of(b3)))                  # control: the last frame moves u
    adapt = m._encode(b)
    last = torch.full((adapt.shape[0],), F_ - 3, dtype=torch.long)
    b4 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b4["feats"][:, F_ - 2:] += 3.0
    with torch.no_grad():
        assert torch.equal(orig(b, adapt, last), orig(b4, m._encode(b4), last))      # frames after t never gathered
        assert not torch.equal(orig(b, adapt, last), orig(b, adapt, torch.full_like(last, F_ - 1)))


def test_u_equals_the_training_time_head_output_and_train_equals_infer(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    m.eval()
    rec, _ = _u_recorder(m)
    b = world["batch"]
    with torch.no_grad():
        m.forward(dict(b), p_gt=0.0)
        tr = [r.clone() for r in rec]
        rec.clear()
        m.forward_infer(_inf(b))
        inf = [r.clone() for r in rec]
        wm = m.wm(m._encode(b), b["act"])
    assert max(float((x - y).abs().max()) for x, y in zip(tr, inf)) <= 1e-5
    o = wm["lead_out"][:, -1]
    p = torch.sigmoid(o[:, 0])
    u = torch.stack([p, p * o[:, 1].clamp(0.0, M.LEAD_COND_GAP_MAX)], -1)
    ok = b["frame_ok"].unfold(1, cfg.lead_hist, 1).all(-1)[:, -1]
    u = torch.where(ok[:, None], u, torch.zeros_like(u))
    assert max(float((u - r).abs().max()) for r in inf) <= 1e-5


def test_u_is_zero_where_a_window_frame_is_not_real(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in world["batch"].items()}
    b["frame_ok"][0, -2] = False
    adapt = m._encode(b)
    last = torch.full((adapt.shape[0],), adapt.shape[1] - 1, dtype=torch.long)
    with torch.no_grad():
        u = m._lead_u(b, adapt, last)
    assert float(u[0].abs().sum()) == 0.0 and float(u[1:].abs().sum()) > 0.0 and u.dtype == torch.float32


# ------------------------------------------------------------------------------------------------ init-ckpt, gate, probe
def test_init_ckpt_leaves_exactly_the_two_keys_fresh_and_refuses_any_other_missing_key(world, tmp_path):
    _, cfg_off, off = _built(world, LS, seed=4)
    _, cfg_on, on = _built(world, LC, seed=5)
    ck = tmp_path / "off.pt"
    torch.save({"model": off.state_dict(), "cfg": {k: v for k, v in cfg_off.__dict__.items() if (k not in M.LEADCOND_CFG_DEFAULTS or v != M.LEADCOND_CFG_DEFAULTS[k])}, "step": 3}, ck)
    rec = TR.load_init_ckpt(str(ck), on, cfg_on)
    assert rec["fresh_keys"] == ["lead_cond.bias", "lead_cond.weight"] and rec["n_mismatch_after_load"] == 0
    sd = dict(off.state_dict())
    drop = next(k for k in sd if k.startswith("head_o."))
    del sd[drop]
    ck2 = tmp_path / "bad.pt"
    torch.save({"model": sd, "cfg": {k: v for k, v in cfg_off.__dict__.items()}, "step": 3}, ck2)
    with pytest.raises(ValueError, match="not the declared lever"):
        TR.load_init_ckpt(str(ck2), _built(world, LC, seed=6)[2], cfg_on)
    a_ok = TR.build_parser().parse_args(["--out", "x", *LC, "--init-ckpt", str(ck)])
    assert TR.bad_init_ckpt(a_ok, {"init_ckpt": {**rec, "path": str(Path(ck).resolve()), "sha256": "0" * 64}}) == []
    extra = dict(rec, fresh_keys=["lead_cond.weight", "lead_cond.bias", "head_o.2.weight"], declared_new_keys=["lead_cond.bias", "lead_cond.weight", "head_o.2.weight"])
    assert any("not exactly" in x or "declared" in x for x in TR.bad_init_ckpt(a_ok, {"init_ckpt": {**extra, "path": str(Path(ck).resolve()), "sha256": "0" * 64}}))


def test_gate_declared_vs_built(world):
    a, cfg, m = _built(world, LC)
    assert TR.bad_lead_cond(a, cfg, m) == []
    a_off, cfg_off, m_off = _built(world, LS)
    assert TR.bad_lead_cond(a_off, cfg_off, m_off) == []
    assert TR.bad_lead_cond(a, cfg_off, m_off)                                          # declared on, nothing built
    assert TR.bad_lead_cond(a_off, cfg, m)                                              # built, not declared
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        m2.lead_cond.weight.fill_(0.1)
    assert any("zero-initialised" in x for x in TR.bad_lead_cond(a, cfg, m2))          # a non-zero start with no checkpoint to explain it
    m3 = copy.deepcopy(m)
    m3.lead_cond = torch.nn.Linear(2, 16)
    assert any("Linear(2" in x for x in TR.bad_lead_cond(a, cfg, m3))
    assert "lead_cond" in TR.FLAG_CHECKS


def test_probe_reports_reach_and_goes_red_without_the_detach(world):
    _, cfg, m = _built(world, LC, seed=3)
    _wake(m)
    p = TR.lead_cond_probe(m, dict(world["batch"]), cfg)
    assert p["pass"] and p["n_rows"] > 0 and p["grad_norm_weight"] > 0.0 and p["control_zero_u_grad_norm_weight"] == 0.0 and p["control_zero_u_grad_norm_bias"] > 0.0
    assert p["grad_lead_head_through_u_max_abs_diff"] == 0.0 and np.isfinite(p["p_mean"]) and np.isfinite(p["g_mean"]) and np.isfinite(p["g_std"])
    assert TR.lead_cond_probe(_built(world, LS)[2], dict(world["batch"]), _built(world, LS)[1]) == {"lead_cond": "off", "pass": True}
    # mutation: u built WITHOUT the detach -> the probe refuses
    m2 = copy.deepcopy(m)

    def undetached(self, b, adapt, last):
        h = cfg.lead_hist
        pos = last[:, None] - (h - 1 - torch.arange(h))[None]
        win = torch.gather(adapt, 1, pos[:, :, None, None].expand(-1, -1, adapt.shape[2], adapt.shape[3]))
        o = self.wm.lead_head(self.wm.z1(win))[:, 0].float()
        pr = torch.sigmoid(o[:, 0])
        return torch.stack([pr, pr * o[:, 1].clamp(0.0, M.LEAD_COND_GAP_MAX)], -1)

    import types
    m2._lead_u = types.MethodType(undetached, m2)                                      # a bound method: deepcopy inside the probe rebinds it to the probe's copy
    with torch.no_grad():
        m2.lead_cond.weight.fill_(0.3)
    with pytest.raises(AssertionError, match="detach"):
        TR.lead_cond_probe(m2, dict(world["batch"]), cfg)
