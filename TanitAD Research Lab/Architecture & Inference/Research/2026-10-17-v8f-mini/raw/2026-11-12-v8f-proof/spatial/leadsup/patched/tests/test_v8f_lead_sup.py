"""LEADSUP (supervised lead-vehicle head on the level-1 grid states): the labels, the batch, the head, the term, its reach (and the exact-zero controls), the gate, the checkpoint and the read hook.

Everything runs on tiny models (d 32) on CPU. A test that the mechanism WORKS is not evidence that it RUNS: the reach tests trace the flag to the term and the term to the weights, with a control that must read EXACTLY 0.0.
Run with the tree's own stack on the path (the helper below inserts it)."""
from __future__ import annotations

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


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("leadsup")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds0 = D.V8FMiniWindows(store, J, wins, hmax=16)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    default_batch = D.collate([ds0.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "default_batch": default_batch, "J": J, "ds": ds, "wins": wins, "idx": idx}


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
    return a, cfg, model


def _terms(model, batch, **over):
    b = dict(batch)
    b.update(over)
    return model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]


def _gsum(ps):
    return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))


def _outside_head(model):
    return [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0 and not n.startswith("wm.lead_head.")]


# ------------------------------------------------------------------------------------------------ labels
def _ref_clos(off, n, valid, gap):
    ok = (valid == 1) & np.isfinite(gap) & (gap >= 0) & (gap <= 60)
    out = np.full(int(sum(n)), np.nan)
    for c in range(len(off)):
        for r in range(off[c], off[c] + n[c]):
            if not ok[r]:
                continue
            idx, j = [r], r - 1
            while j >= off[c] and len(idx) < 5 and ok[j] and abs(gap[j + 1] - gap[j]) <= 3.0:
                idx.append(j)
                j -= 1
            if len(idx) < 3:
                continue
            idx = idx[::-1]
            t = np.arange(len(idx)) * 0.2
            out[r] = ((t - t.mean()) * (gap[idx] - gap[idx].mean())).sum() / ((t - t.mean()) ** 2).sum()
    return np.where(ok, gap, np.nan), out


def test_label_rule_equals_the_literal_hist_run_ls_slope_rule():
    rng = np.random.default_rng(5)
    n = np.array([60, 45, 7, 80])
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    N = int(n.sum())
    valid = (rng.random(N) < 0.8).astype(np.int8)
    gap = 20 + np.cumsum(rng.normal(0, 1.5, N))
    gap[rng.random(N) < 0.05] += 8.0
    gap[rng.random(N) < 0.03] = np.nan
    gap[rng.random(N) < 0.02] = 75.0
    gap[rng.random(N) < 0.02] = -1.0
    gap = gap.astype(np.float32)
    L = D.lead_sup_label_arrays({"off": off, "n": n, "lead_valid": valid, "lead_gap": gap})
    rg, rc = _ref_clos(off, n, valid, gap.astype(np.float64))
    assert np.array_equal(np.isfinite(L["gap"]), np.isfinite(rg)) and np.allclose(L["gap"][np.isfinite(rg)], rg[np.isfinite(rg)])
    assert np.array_equal(np.isfinite(L["clos"]), np.isfinite(rc)) and int(np.isfinite(rc).sum()) > 50
    assert np.abs(L["clos"][np.isfinite(rc)] - rc[np.isfinite(rc)]).max() < 1e-4
    assert np.array_equal(L["pres"], valid.astype(np.float32))
    assert len(L["sha256"]) == 64 and L["n_clos"] == int(np.isfinite(rc).sum())


def test_planted_series_give_the_exact_slope_and_the_run_breaks_where_it_must():
    off, n = np.array([0, 12, 18]), np.array([12, 6, 10])
    g = np.concatenate([30.0 - 0.4 * np.arange(12), 10.0 + 0.2 * np.arange(6), 40.0 - 0.1 * np.arange(5), 50.0 - 0.1 * np.arange(5)])
    g[23] = g[22] - 2.9
    g[24] = g[23] - 3.1                                                       # above 3.0 m: breaks the run
    c = D.lead_sup_label_arrays({"off": off, "n": n, "lead_valid": np.ones(28, np.int8), "lead_gap": g.astype(np.float32)})["clos"]
    assert np.isnan(c[:2]).all() and np.allclose(c[2:12], -2.0, atol=1e-5)    # fewer than 3 frames: no label; a run capped at 5 frames keeps the exact slope of a line
    assert np.isnan(c[12:14]).all() and np.allclose(c[14:18], 1.0, atol=1e-5)  # a clip start breaks the run
    assert np.isnan(c[18:20]).all() and np.allclose(c[20:23], -0.5, atol=1e-5)
    assert np.isnan(c[24])                                                    # the 3.1 m step: a run of one frame


def test_the_label_file_is_not_read_from_the_lead_head_arrays_and_gap_range_is_0_to_60():
    off, n = np.array([0]), np.array([6])
    L = D.lead_sup_label_arrays({"off": off, "n": n, "lead_valid": np.array([1, 1, 1, 0, 1, 1], np.int8), "lead_gap": np.array([0.0, 60.0, 61.0, 20.0, np.nan, -0.5], np.float32)})
    assert list(np.isfinite(L["gap"])) == [True, True, False, False, False, False]
    assert list(L["pres"]) == [1, 1, 1, 0, 1, 1]


# ------------------------------------------------------------------------------------------------ the batch
def test_batch_carries_the_labels_only_when_the_dataset_is_made_with_lead_sup(world):
    b, d = world["batch"], world["default_batch"]
    assert sorted(set(b) - set(d)) == ["lead_clos", "lead_gapm", "lead_pres"] and not (set(d) - set(b))
    for k in ("lead_pres", "lead_gapm", "lead_clos"):
        assert tuple(b[k].shape) == (8, 16) and b[k].dtype == torch.float32
    assert int(torch.isfinite(b["lead_gapm"]).sum()) > 0 and int(torch.isfinite(b["lead_clos"]).sum()) > 0
    for k in d:                                                               # every other key is the default batch's, value for value
        if torch.is_tensor(d[k]):
            assert torch.equal(torch.nan_to_num(d[k].float()), torch.nan_to_num(b[k].float())) or k == "lead_gap_h"


def test_synthetic_lead_labels_are_deterministic():
    root_J = lambda: {"off": np.array([0, 30]), "n": np.array([30, 30]), "lead_valid": np.zeros(60, np.int8), "lead_gap": np.full(60, np.nan, np.float32)}   # noqa: E731
    a, b = D.synthetic_lead_labels(root_J(), 3), D.synthetic_lead_labels(root_J(), 3)
    assert np.array_equal(a["lead_gap"], b["lead_gap"], equal_nan=True) and int(a["lead_valid"].sum()) > 0


# ------------------------------------------------------------------------------------------------ flags, config, defaults
def test_flags_have_entries_defaults_and_the_config_refuses_what_it_cannot_build():
    d = TR.build_parser()
    assert (d.get_default("lead_sup"), d.get_default("w_lead_sup"), d.get_default("lead_hist"), d.get_default("lead_sup_detach")) == ("off", 0.0, 5, "off")
    for f in ("lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach"):
        assert TR.FLAG_CHECKS[f] == f"lever:{f}" and f in TR.INIT_CKPT_MATCH
    mk = lambda **kw: M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", **kw)       # noqa: E731
    mk()
    mk(lead_sup="on", w_lead_sup=1.0)
    mk(lead_sup="on", w_lead_sup=0.5, lead_hist=3, lead_sup_detach="on")
    for kw, pat in (({"lead_sup": "on"}, "no loss weight"), ({"lead_sup": "on", "w_lead_sup": float("nan")}, "finite"), ({"w_lead_sup": 1.0}, "does nothing"),
                    ({"lead_sup_detach": "on"}, "does nothing"), ({"lead_hist": 3}, "does nothing"), ({"lead_sup": "on", "w_lead_sup": 1.0, "lead_hist": 17}, "does not fit"),
                    ({"lead_sup": "on", "w_lead_sup": 1.0, "lead_hist": 0}, ">= 1"), ({"lead_sup": "maybe"}, "lead_sup must"), ({"lead_sup": "on", "w_lead_sup": -1.0}, ">= 0")):
        with pytest.raises(ValueError, match=pat):
            mk(**kw)
    with pytest.raises(ValueError, match="grid"):
        M.V8FMiniConfig(wm="b", hmax=16, wm_state="mean", lead_sup="on", w_lead_sup=1.0)
    with pytest.raises(ValueError, match="grid"):
        M.V8FMiniConfig(wm="off", hmax=16, wm_state="grid", lead_sup="on", w_lead_sup=1.0)


def test_default_model_has_no_head_no_key_no_term(world):
    a, cfg, model = _built(world, GRID)
    assert model.wm.lead_head is None and not [k for k in model.state_dict() if "lead_head" in k and k.startswith("wm.")]
    T = _terms(model, world["default_batch"])
    assert not [k for k in T if k.startswith("lead_sup")]
    lv = model.built_levers()
    assert (lv["lead_sup"], lv["lead_hist"], lv["lead_sup_detach"], lv["w_lead_sup"]) == ("off", 5, "off", 0.0)


def test_init_of_the_head_arm_is_G_plus_the_head_only_and_the_count(world):
    _, _, g = _built(world, GRID)
    _, _, l = _built(world, LS)
    sg, sl = g.state_dict(), l.state_dict()
    assert all(k in sl and torch.equal(sg[k], sl[k]) for k in sg)
    assert all(k.startswith("wm.lead_head.") for k in sl if k not in sg)
    n = sum(p.numel() for p in l.wm.lead_head.parameters())
    assert n == 640 * 64 + 64 + 320 * 128 + 128 + 128 * 3 + 3 == 82499
    assert sum(p.numel() for p in l.parameters()) - sum(p.numel() for p in g.parameters()) == n


# ------------------------------------------------------------------------------------------------ reach (flag -> model -> term -> weights) with exact-zero controls
def test_reach_term_alone_head_and_state_encoder_nonzero_everything_else_exactly_zero(world):
    a, cfg, model = _built(world, LS)
    assert cfg.lead_sup == "on" and cfg.w_lead_sup == 1.0 and model.wm.lead_head is not None            # argparse -> config -> built head
    model.train()
    T = _terms(model, world["batch"])
    assert float(T["lead_sup"]) > 0 and all(torch.isfinite(T[k]) for k in ("lead_sup_bce", "lead_sup_gap", "lead_sup_clos"))
    assert float(T["lead_sup_n_gap"]) > 0 and float(T["lead_sup_n_clos"]) > 0
    T["lead_sup"].backward()
    st = model.wm.stack
    assert _gsum(model.wm.lead_head.parameters()) > 0 and _gsum(model.wm.proj.parameters()) > 0 and _gsum(model.adapter.parameters()) > 0
    assert _gsum(st.predictors[0].parameters()) == 0.0 and _gsum(st.act_embed[0].parameters()) == 0.0 and _gsum(st.idm_heads[0].parameters()) == 0.0       # controls
    assert _gsum(p for m in (st.predictors[1], st.predictors[2]) for p in m.parameters()) == 0.0
    assert _gsum(p for m in (model.read_s, model.head_s, model.read_t, model.trunk_t, model.read_o, model.head_o) for p in m.parameters()) == 0.0
    model.zero_grad()
    _terms(model, world["batch"])["total"].backward()
    assert _gsum(st.predictors[0].parameters()) > 0                                                                                                      # the total does reach the predictor


def test_detach_on_the_term_reaches_nothing_outside_the_head_exactly(world):
    _, _, off = _built(world, LS)
    _, _, on = _built(world, LS + ["--lead-sup-detach", "on"])
    off.train()
    on.train()
    t_off, t_on = _terms(off, world["batch"]), _terms(on, world["batch"])
    assert float(t_off["lead_sup"]) == float(t_on["lead_sup"])                                       # same init, same value: only the wiring differs
    t_off["lead_sup"].backward()
    t_on["lead_sup"].backward()
    assert len(_outside_head(off)) > 0                                                              # detach off: the state encoder moves (so the zero below is the wiring, not the data)
    assert _outside_head(on) == [] and _gsum(on.wm.lead_head.parameters()) > 0                      # detach on: EXACTLY 0.0 outside the head; the head itself trains
    assert on.built_levers()["lead_sup_detach"] == "on" and off.built_levers()["lead_sup_detach"] == "off"


def test_all_labels_masked_reads_exactly_zero_term_parts_and_gradients(world):
    _, _, model = _built(world, LS)
    b = world["batch"]
    nan = torch.full_like(b["lead_gapm"], float("nan"))
    T = _terms(model, b, lead_gapm=nan, lead_clos=nan, lead_pres=torch.zeros_like(b["lead_pres"]), frame_ok=torch.zeros_like(b["frame_ok"]))
    assert float(T["lead_sup"]) == 0.0 and float(T["lead_sup_n_pres"]) == 0.0
    T["lead_sup"].backward()
    assert _gsum(model.wm.lead_head.parameters()) == 0.0
    _, _, m2 = _built(world, LS)
    T2 = _terms(m2, b, lead_gapm=nan, lead_clos=nan)
    assert float(T2["lead_sup_bce"]) > 0 and float(T2["lead_sup_gap"]) == 0.0 and float(T2["lead_sup_clos"]) == 0.0


def test_frame_alignment_labels_of_non_end_frames_change_nothing_end_frames_do(world):
    _, _, model = _built(world, LS)
    model.eval()
    b = world["batch"]
    ref = float(_terms(model, b)["lead_sup"])
    moved = []
    for f in range(16):
        g, p = b["lead_gapm"].clone(), b["lead_pres"].clone()
        g[:, f] += 7.0
        p[:, f] = 1.0 - p[:, f]
        diff = float(_terms(model, b, lead_gapm=g, lead_pres=p)["lead_sup"]) - ref
        if f < 4:
            assert diff == 0.0, f                                                                   # frames 0 .. hist-2 are never an end frame
        elif diff != 0.0:
            moved.append(f)
    assert len(moved) >= 8
    adapt = torch.randn(2, 16, model.cfg.n_tok, model.cfg.d, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        base = model.wm.lead_head(model.wm.z1(adapt))
        for f in range(16):
            a2 = adapt.clone()
            a2[:, f] += 1.0
            d = (model.wm.lead_head(model.wm.z1(a2)) - base).abs().amax((0, 2))
            assert [int(i) + 4 for i in torch.nonzero(d > 0).flatten()] == [e for e in range(4, 16) if e - 4 <= f <= e], f          # the output at end frame e reads frames e-4 .. e, nothing else


def test_the_total_gains_exactly_w_times_the_term_and_the_line_is_load_bearing(world):
    _, _, model = _built(world, LS)
    model.eval()
    b = dict(world["batch"])
    import copy
    twin = copy.deepcopy(model)                                                                       # made before any call: the SIGReg term draws random projections from the GLOBAL rng (two calls differ by 1.7e-4, measured): reseed before each call
    object.__setattr__(twin.cfg, "w_lead_sup", 0.0)
    w = model.cfg.w_lead_sup
    with torch.no_grad():
        torch.manual_seed(77)
        t = model.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
        torch.manual_seed(77)
        t0 = twin.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
    assert float(t["lead_sup"]) > 0 and abs(float(t["total"]) - float(t0["total"]) - w * float(t["lead_sup"])) < 8 * 1.1920929e-07 * max(abs(float(t["total"])), abs(float(t0["total"])))   # a few float32 ulps of the total
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert sum(1 for ln in src.splitlines() if ln.lstrip().startswith("total = total + cfg.w_lead_sup * T[")) == 1      # the ONE line a mutation deletes (checks/t3_checks.py does it and goes red)


# ------------------------------------------------------------------------------------------------ the gate (declared vs built)
def _info(world, on=True):
    ll = {k: world["ds"].lead_sup_labels[k] for k in ("sha256", "n_rows", "n_pres", "n_gap", "n_clos")}
    return {"ds_lead_sup": on, "lead_sup_labels": ll if on else None}


def test_declared_vs_built_goes_red_on_every_lie(world):
    a, cfg, model = _built(world, LS)
    assert TR.bad_lead_sup(a, cfg, model, _info(world)) == []
    a0, cfg0, model0 = _built(world, GRID)
    assert TR.bad_lead_sup(a0, cfg0, model0, {}) == []
    assert TR.bad_lead_sup(a, cfg0, model0, _info(world))                                            # declared on, built off
    assert TR.bad_lead_sup(a0, cfg, model, _info(world))                                             # built on, declared off
    for extra, what in ((LS[:-1] + ["2.0"], "w"), (LS + ["--lead-hist", "3"], "hist"), (LS + ["--lead-sup-detach", "on"], "detach")):
        assert TR.bad_lead_sup(_args(extra), cfg, model, _info(world)), what                         # the built head is the LS one; the declaration differs
    assert TR.bad_lead_sup(a, cfg, model, _info(world, on=False))                                    # the dataset carries no labels
    bad_hash = _info(world)
    bad_hash["lead_sup_labels"] = {**bad_hash["lead_sup_labels"], "sha256": None}
    assert TR.bad_lead_sup(a, cfg, model, bad_hash)
    none_gap = _info(world)
    none_gap["lead_sup_labels"] = {**none_gap["lead_sup_labels"], "n_gap": 0}
    assert TR.bad_lead_sup(a, cfg, model, none_gap)
    zero_w = _args(LS[:-1] + ["0.0"])
    assert any("no loss" in m for m in TR.bad_lead_sup(zero_w, cfg, model, _info(world)))             # a head with weight 0 is refused by the gate too (the config already refuses it)
    assert TR.bad_lead_sup(_args(GRID + ["--w-lead-sup", "1.0"]), cfg0, model0, {})                  # a weight with the head off
    # mutation of the readback: a built_levers that lies is caught by the gate through the declared/built comparison
    lie = type("L", (), {})()
    lie.wm, lie.built_levers = model.wm, lambda: {**model.built_levers(), "lead_hist": 4}
    assert TR.bad_lead_sup(a, cfg, lie, _info(world))


# ------------------------------------------------------------------------------------------------ the checkpoint and the read hook
def test_checkpoint_cfg_round_trip_loads_in_the_eval_builder_and_the_read_hook_runs(world, tmp_path):
    a, cfg, model = _built(world, LS + ["--lead-hist", "3"])
    cfg_rec = {k: v for k, v in cfg.__dict__.items() if (k not in M.LEADSUP_CFG_DEFAULTS or v != M.LEADSUP_CFG_DEFAULTS[k])}
    assert {k: cfg_rec[k] for k in M.LEADSUP_CFG_DEFAULTS if k in cfg_rec} == {"lead_sup": "on", "w_lead_sup": 1.0, "lead_hist": 3}              # defaults are not written
    off_rec = {k: v for k, v in _built(world, GRID)[1].__dict__.items() if (k not in M.LEADSUP_CFG_DEFAULTS or v != M.LEADSUP_CFG_DEFAULTS[k])}
    assert not [k for k in off_rec if k in M.LEADSUP_CFG_DEFAULTS]                                    # a flag-off checkpoint keeps its key set
    path = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg_rec, "grid": "pool4x10"}, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg2 = M.V8FMiniConfig(**ck["cfg"])                                                               # exactly what v8f_t1_eval.cmd_run does
    m2 = M.V8FMini(cfg2, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m2.load_state_dict(ck["model"])                                                                   # strict: the extra head keys are built, so they load
    assert m2.wm.lead_head is not None and m2.wm.lead_head.hist == 3
    # the read hook on synthetic frames: tokens and states give the same numbers, and they equal the head called by hand
    import lead_dump as LD
    cfg3, m3 = LD.load_model(str(path))
    feats = torch.randn(4, 6, cfg3.n_tok, cfg3.d_feat, generator=torch.Generator().manual_seed(9)).half()
    o1 = LD.head_outputs(m3, cfg3, feats, "feats")
    with torch.no_grad():
        adapt = m3.adapter(feats.float()) + m3.pos[None, None]
        z1 = m3.wm.z1(adapt)
    o2 = LD.head_outputs(m3, cfg3, z1, "states")
    assert o1.shape == (4, 4, 3) and torch.equal(o1, o2)
    np.save(tmp_path / "f.npy", feats.numpy())
    LD.main(["--ckpt", str(path), "--feats", str(tmp_path / "f.npy"), "--out", str(tmp_path / "o.npz"), "--end-frame", "4"])
    z = np.load(tmp_path / "o.npz")
    assert z["presence_prob"].shape == (4,) and np.allclose(z["gap_m"], o1[:, 4 - 2, 1].numpy() * 50.0, atol=1e-4) and np.allclose(z["closing_ms"], o1[:, 4 - 2, 2].numpy() * 10.0, atol=1e-4)
    assert ((z["presence_prob"] >= 0) & (z["presence_prob"] <= 1)).all() and int(z["end_frame"]) == 4
    off_ck = tmp_path / "off.pt"
    torch.save({"model": _built(world, GRID)[2].state_dict(), "cfg": off_rec, "grid": "pool4x10"}, off_ck)
    with pytest.raises(SystemExit, match="no lead head"):
        LD.load_model(str(off_ck))


def test_the_driving_output_path_does_not_read_the_head_or_the_labels(world):
    for k in ("lead_pres", "lead_gapm", "lead_clos"):
        assert k not in M.INFERENCE_INPUT_KEYS
    _, _, model = _built(world, LS)
    _, _, ref = _built(world, GRID)
    model.eval()
    ref.eval()
    b = {k: v for k, v in world["batch"].items() if k in M.INFERENCE_INPUT_KEYS or k == "frame_ok"}
    # the heads share every parameter with G at init (test above); the inference output of both is the same function of the batch: the head adds a branch that forward_infer never calls
    with torch.no_grad():
        o1, o2 = model.forward_infer(dict(b)), ref.forward_infer(dict(b))
    assert all(torch.equal(o1[k], o2[k]) for k in o1 if torch.is_tensor(o1[k]) and k in o2)
