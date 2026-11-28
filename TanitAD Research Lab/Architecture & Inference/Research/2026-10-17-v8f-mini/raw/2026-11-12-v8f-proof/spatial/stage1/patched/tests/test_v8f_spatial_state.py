"""PREREG_V8F_PROOF_ADD12 (SPATIAL): the level-1 world-model state that keeps location. CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_spatial_state.py

Pinned, and the control that must go red for each:
  * widths: mean 128 / predictor 128 (the tip), grid 40 x 16 = 640 / predictor 256, mean --wm-dim 640 / predictor 256; levels 2 and 3 stay 128.
  * the config refuses grid with wm off, a cell width below 1, an unknown mode.
  * every new flag has its declared-vs-built entry; the gate goes RED when z1 ignores the flag, on a declared grid / built mean, the reverse, a wrong cell width.
  * reachability: the WM term alone gives the new projection and the adapter a non-zero gradient; with w_wm 0 the gradient is EXACTLY 0.0.
  * location: permuting the cells of one frame leaves the mean state unchanged (1e-5) and changes the grid state.
  * a default (mean) checkpoint records no new cfg key; a grid checkpoint loads through the T1 builder and is refused as a fine-tune start by a mean argv.
  * rung beta refuses a grid model.
"""
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


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("spatial")
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


@pytest.mark.parametrize("extra,l1,pw", [([], 128, 128), (["--wm-state", "grid"], 640, 256), (["--wm-dim", "640"], 640, 256),
                                          (["--wm-state", "grid", "--wm-cell-dim", "8"], 320, 256)])
def test_widths_are_the_declared_ones(world, extra, l1, pw):
    a, cfg, model, opt, info = _built(world, extra)
    b = model.built_levers()
    assert (b["wm_l1_dim"], b["wm_z1_width"], b["wm_l1_pred_width"], b["wm_upper_dims"]) == (l1, l1, pw, [128, 128])
    assert b["wm_state"] == ("grid" if "grid" in extra else "mean")
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []


def test_default_is_the_tip_shape(world):
    a, cfg, model, opt, info = _built(world)
    assert [tuple(p.shape) for p in model.wm.proj.parameters()] == [(32,), (32,), (128, 32), (128,)]
    assert model.wm.stack.predictors[0].in_proj.weight.shape[0] == 128
    assert (cfg.wm_state, cfg.wm_cell_dim, cfg.wm_dim) == ("mean", 16, 128)


def test_config_refuses_what_it_cannot_build():
    with pytest.raises(ValueError, match="needs the world-model"):
        M.V8FMiniConfig(wm="off", wm_state="grid")
    with pytest.raises(ValueError, match="wm_cell_dim"):
        M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_cell_dim=0)
    with pytest.raises(ValueError, match="wm_state"):
        M.V8FMiniConfig(wm="b", hmax=16, wm_state="ring")


def test_every_new_flag_has_an_entry_and_a_default():
    d = {act.dest for act in TR.build_parser()._actions}
    for f, dflt in (("wm_state", "mean"), ("wm_cell_dim", 16), ("wm_dim", 128)):
        assert f in d and f in TR.FLAG_CHECKS and TR.build_parser().get_default(f) == dflt
    assert {"wm_state", "wm_cell_dim", "wm_dim"} <= set(TR.INIT_CKPT_MATCH)


def test_declared_vs_built_goes_red_on_every_lie(world, monkeypatch):
    a, cfg, model, opt, info = _built(world, ["--wm-state", "grid"])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    with monkeypatch.context() as mp:
        mp.setattr(M.WMAux, "z1", lambda self, adapt: self.proj(adapt.mean(2)))       # z1 ignores the flag
        assert any("wm_state" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    am, cm, mm, om, im = _built(world)                                                  # declared grid, built mean
    assert any("wm_state" in m for m in TR.declared_vs_built(_args(["--wm-state", "grid"]), cm, mm, om, im))
    assert any("wm_state" in m for m in TR.declared_vs_built(_args(), cfg, model, opt, info))                  # declared mean, built grid
    assert any("wm_cell_dim" in m for m in TR.declared_vs_built(_args(["--wm-state", "grid", "--wm-cell-dim", "8"]), cfg, model, opt, info))
    a1, c1, m1, o1, i1 = _built(world, ["--arm", "M1"])                                 # wm off: a lever flag with nothing to build is refused
    assert any("no WM auxiliary" in m for m in TR.declared_vs_built(_args(["--arm", "M1", "--wm-dim", "640"]), c1, m1, o1, i1))


def _wm_grads(world, extra, w_wm):
    a, cfg, model, opt, info = _built(world, extra, w_wm=w_wm)
    model.train()
    out = model.loss(dict(world["batch"]), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    (cfg.w_wm * out["terms"]["wm"]).backward()
    g = lambda ps: float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))
    return g(model.wm.proj.parameters()), g(model.adapter.parameters())


def test_the_wm_term_alone_reaches_projection_and_adapter_and_zero_weight_reads_exactly_zero(world):
    gp, ga = _wm_grads(world, ["--wm-state", "grid"], 0.1)
    assert gp > 0.0 and ga > 0.0
    assert _wm_grads(world, ["--wm-state", "grid"], 0.0) == (0.0, 0.0)


def test_location_is_in_the_grid_state_and_not_in_the_mean_state(world):
    feats = world["batch"]["feats"].float()
    perm = torch.randperm(40, generator=torch.Generator().manual_seed(5))
    f2 = feats.clone()
    f2[:, 2] = feats[:, 2][:, perm]                                                     # the cells of one frame
    for extra, moves in (([], False), (["--wm-dim", "640"], False), (["--wm-state", "grid"], True)):
        a, cfg, model, opt, info = _built(world, extra)
        with torch.no_grad():
            z0, z1 = (model.wm.z1(model._encode({"feats": f})) for f in (feats, f2))
            tok = (model.adapter(feats).mean(2) - model.adapter(f2).mean(2)).abs().max()
        assert float(tok) < 1e-5                                                         # the tokens before the position term
        diff = float((z0[:, 2] - z1[:, 2]).abs().max())
        assert (diff > 1e-3) if moves else (diff < 1e-5)
        assert float((z0[:, [0, 1, 3]] - z1[:, [0, 1, 3]]).abs().max()) == 0.0           # the other frames are untouched


def test_a_default_checkpoint_keeps_the_tip_cfg_keys_and_a_grid_one_round_trips(world, tmp_path):
    base = ["--synthetic", "--arm", "M2b", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--steps", "2"]
    assert TR.main([*base, "--out", str(tmp_path / "mean")]) == 0
    assert TR.main([*base, "--wm-state", "grid", "--out", str(tmp_path / "grid")]) == 0
    cm = torch.load(tmp_path / "mean" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    cg = torch.load(tmp_path / "grid" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert not (set(M.SPATIAL_CFG_FIELDS) & set(cm["cfg"])) and set(M.SPATIAL_CFG_FIELDS) <= set(cg["cfg"])
    cfg = M.V8FMiniConfig(**cg["cfg"])                                                  # the T1 eval builder (v8f_t1_eval.cmd_run)
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(cg["model"])
    assert model.built_levers()["wm_state"] == "grid"
    a, cfg2, m2, o2, i2 = _built(world)                                                  # a mean argv must not start from a grid checkpoint
    with pytest.raises(ValueError, match="wm_state"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m2, cfg2)


def test_rung_beta_refuses_a_grid_model(world):
    from tanitad.models import v8f_beta as B
    a, cfg, model, opt, info = _built(world, ["--wm-state", "grid"])
    with pytest.raises(ValueError, match="REFUSES"):
        B.refuse_grid_state(model.wm)
    out = {"adapt": model._encode({"feats": world["batch"]["feats"].float()})}
    with pytest.raises(ValueError, match="REFUSES"):
        B.collect_beta_inputs(model, dict(world["batch"]), out)
    B.refuse_grid_state(_built(world)[2].wm)                                              # the mean model passes
