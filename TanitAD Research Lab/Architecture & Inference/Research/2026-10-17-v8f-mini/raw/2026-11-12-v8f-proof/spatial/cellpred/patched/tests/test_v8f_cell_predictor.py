"""PREREG_V8F_PROOF_ADD16 (CELLPRED): the level-1 predictor that attends across cells and time. CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_cell_predictor.py

Each control of section 7 of the card, and the deliberate regression that must make it go red:
  * interface: same call signature and attributes as CausalPredictor; HJEPAStack loss and rollout run unchanged on it.
  * c2: parameter count of the cell predictor at or below the flat predictor of the grid arm (both printed).
  * c3: output at step i does not move (exactly 0) for a change at a later step; it moves for a change at step i.
  * c4: one input cell moves OTHER predicted cells; with the spatial attention replaced by identity it moves exactly 0 of them;
        a flat build is reported flat and a cell build cell, read from the built module (a flat module planted under a cell config reads flat).
  * c5: one real-data step: every parameter group of the new predictor has a non-zero gradient; with a zero WM weight and for a parameter
        outside the graph the reading is exactly 0.0.
  * refusals: cell with a mean state, cell with wm off, declared-vs-built mismatch in both directions.
  * checkpoint: a default checkpoint keeps the tip cfg key set; a cell checkpoint adds the key, loads strictly through the T1 builder, and is
        refused as a fine-tune start by a flat argv.
"""
from __future__ import annotations

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
from tanitad.models import hjepa as H  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
GRID = ["--wm-state", "grid", "--wm-cell-dim", "16"]
CELL = GRID + ["--wm-l1-pred", "cell"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cellpred")
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


def _cell(**kw):
    torch.manual_seed(0)
    return H.CellCausalPredictor(640, 32, 40, 128, 2, 4, None, 3, **kw).eval()


# ------------------------------------------------------------------------------------------------ interface and widths
def test_interface_equals_causal_predictor():
    flat, cell = H.CausalPredictor(640, 32, 256, 2, 4, None, 3), _cell()
    names = lambda m: list(inspect.signature(m.forward).parameters)   # noqa: E731
    assert names(flat) == names(cell) == ["z_ctx", "a_ctx"]
    for att in ("latent_dim", "action_dim", "context"):
        assert getattr(flat, att) == getattr(cell, att)
    for L in (1, 2, 3):
        z, a = torch.randn(2, L, 640), torch.randn(2, L, 32)
        assert flat(z, a).shape == cell(z, a).shape == (2, L, 640)
    with pytest.raises(ValueError, match="exceeds"):
        cell(torch.randn(1, 4, 640), torch.randn(1, 4, 32))
    assert cell.in_proj.weight.shape[0] == 128                            # the stage-1 reader of the predictor width


def test_stack_loss_and_rollout_run_unchanged_and_are_finite():
    lv = [H.LevelConfig(latent_dim=640, action_dim=2, act_embed_dim=32, pred_width=128, pred_depth=2, pred_heads=4,
                        pred_kind="cell", pred_cells=40, pred_cell_dim=16, enc_hidden=128, macro_hidden=32),
          H.LevelConfig(latent_dim=128, action_dim=32, stride=2, pred_width=128, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32),
          H.LevelConfig(latent_dim=128, action_dim=32, stride=2, pred_width=128, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32)]
    torch.manual_seed(0)
    st = H.HJEPAStack(lv, sigreg_projections=64)
    assert isinstance(st.predictors[0], H.CellCausalPredictor) and type(st.predictors[1]) is H.CausalPredictor
    z, a = torch.randn(2, 16, 640), torch.randn(2, 15, 2)
    out = st.loss(z, a)
    assert torch.isfinite(out["total"]) and torch.isfinite(out["pred/L1"])
    roll = st.rollout(1, torch.randn(640), torch.randn(5, 7, 2), context=1)       # leading dim 5, horizon 7
    assert roll.shape == (5, 7, 640) and torch.isfinite(roll).all()
    roll3 = st.rollout(1, torch.randn(640), torch.randn(7, 2), context=3)
    assert roll3.shape == (7, 640) and torch.isfinite(roll3).all()
    st.param_groups(1e-3)                                                          # every parameter in exactly one group
    n_groups = sum(len(g["params"]) for g in st.param_groups(1e-3))
    assert n_groups == len(list(st.parameters()))
    assert st.parameter_counts()[1]["predictor"] == sum(p.numel() for p in st.predictors[0].parameters())
    assert st.built_levers()[1]["context"] == 3
    with pytest.raises(ValueError, match="pred_kind"):
        H.HJEPAStack([H.LevelConfig(latent_dim=640, action_dim=2, pred_kind="ring")])
    with pytest.raises(ValueError, match="pred_cells"):
        H.HJEPAStack([H.LevelConfig(latent_dim=640, action_dim=2, pred_kind="cell", pred_cells=40, pred_cell_dim=8)])


# ------------------------------------------------------------------------------------------------ c2
def test_c2_parameters_at_or_below_the_flat_predictor_of_the_grid_arm(world):
    _, _, mg, _, _ = _built(world, GRID)
    _, _, mc, _, _ = _built(world, CELL)
    flat_n = sum(p.numel() for p in mg.wm.stack.predictors[0].parameters())
    cell_n = sum(p.numel() for p in mc.wm.stack.predictors[0].parameters())
    print(f"\nc2: level-1 predictor parameters: flat (grid arm G) {flat_n:,}  cell (arm P) {cell_n:,}")
    assert cell_n <= flat_n
    bc, bg = mc.built_levers(), mg.built_levers()
    assert bc["wm_l1_pred_params"] == cell_n and bc["wm_l1_pred_params_flat_ref"] == flat_n and bc["wm_l1_pred_params_le_flat"] is True
    assert bg["wm_l1_pred_params"] == flat_n == bg["wm_l1_pred_params_flat_ref"]
    assert (bc["wm_l1_pred_width"], bg["wm_l1_pred_width"]) == (128, 256)
    assert bc["wm_z1_width"] == bc["wm_l1_dim"] == 640 and bc["wm_state"] == "grid"


# ------------------------------------------------------------------------------------------------ c3
def test_c3_causality_exact_zero_for_later_steps_and_nonzero_for_the_same_step():
    m = _cell()
    g = torch.Generator().manual_seed(1)
    z, a = torch.randn(2, 3, 640, generator=g), torch.randn(2, 3, 32, generator=g)
    with torch.no_grad():
        base = m(z, a)
        for i in (0, 1):
            for what in ("z", "a"):
                z2, a2 = z.clone(), a.clone()
                (z2 if what == "z" else a2)[:, i + 1:] += torch.randn_like((z2 if what == "z" else a2)[:, i + 1:])
                later = (m(z2, a2) - base)[:, :i + 1].abs().max()
                assert float(later) == 0.0, (i, what, float(later))
        for i in (0, 1, 2):
            for what in ("z", "a"):
                z2, a2 = z.clone(), a.clone()
                (z2 if what == "z" else a2)[:, i] += torch.randn_like((z2 if what == "z" else a2)[:, i])
                same = (m(z2, a2) - base)[:, i].abs().max()
                assert float(same) > 1e-6, (i, what)


# ------------------------------------------------------------------------------------------------ c4
def test_c4_one_cell_moves_the_other_cells_and_the_regression_that_removes_it_goes_red(world):
    m = _cell()
    g = torch.Generator().manual_seed(2)
    z, a = torch.randn(1, 3, 640, generator=g), torch.randn(1, 3, 32, generator=g)
    z2 = z.clone()
    z2[:, 2, 16 * 7:16 * 8] += 1.0                                         # ONE cell (7) at the last step
    with torch.no_grad():
        d = (m(z2, a) - m(z, a))[:, 2]
    other = torch.cat([d[:, :16 * 7], d[:, 16 * 8:]], dim=1).abs().max()
    assert float(other) > 1e-6
    # deliberate regression: every spatial block is the identity -> cells never talk, other cells read EXACTLY 0
    for blk in m.spatial:
        blk.forward = lambda x: x
    with torch.no_grad():
        d0 = (m(z2, a) - m(z, a))[:, 2]
    assert float(torch.cat([d0[:, :16 * 7], d0[:, 16 * 8:]], dim=1).abs().max()) == 0.0
    assert float(d0[:, 16 * 7:16 * 8].abs().max()) > 1e-6                  # the perturbed cell itself still moves
    # the probe reads the BUILT module
    _, _, mc, _, _ = _built(world, CELL)
    _, _, mg, _, _ = _built(world, GRID)
    assert mc.built_levers()["wm_l1_pred"] == "cell" and mc.built_levers()["wm_l1_pred_cross_cell"] is True
    assert mg.built_levers()["wm_l1_pred"] == "flat"
    mm = _built(world)[2]
    assert mm.built_levers()["wm_l1_pred"] == "flat"                       # the mean default is flat too
    mc.wm.stack.predictors[0] = H.CausalPredictor(640, 32, 128, 2, 4, None, 3)   # a flat module planted under a cell config
    assert mc.cfg.wm_l1_pred == "cell" and mc.built_levers()["wm_l1_pred"] == "flat"
    mg.wm.stack.predictors[0] = _cell()                                    # a cell module planted under a flat config
    assert mg.cfg.wm_l1_pred == "flat" and mg.built_levers()["wm_l1_pred"] == "cell"


# ------------------------------------------------------------------------------------------------ c5
def _loss_grads(world, extra, w_wm):
    a, cfg, model, opt, info = _built(world, extra, w_wm=w_wm)
    model.train()
    out = model.loss(dict(world["batch"]), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    (cfg.w_wm * out["terms"]["wm"]).backward()                                      # the WM term as the trainer weights it
    return model


def _gsum(ps):
    return float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))


def _groups(pred):
    gs = {"in_proj": [pred.in_proj], "cell_pos": [pred.cell_pos], "time_pos": [pred.time_pos], "act_tok": [pred.act_tok],
          "final_ln": [pred.final_ln], "out": [pred.out]}
    for i, (s, t) in enumerate(zip(pred.spatial, pred.temporal)):
        gs[f"spatial{i}"], gs[f"temporal{i}"] = [s], [t]
    return {k: [q for m in v for q in (m.parameters() if isinstance(m, torch.nn.Module) else [m])] for k, v in gs.items()}


def test_c5_every_new_group_gets_a_gradient_through_the_real_loss_and_zero_weight_reads_exactly_zero(world):
    model = _loss_grads(world, CELL, 0.1)
    pred = model.wm.stack.predictors[0]
    groups = _groups(pred)
    assert sum(len(v) for v in groups.values()) == len(list(pred.parameters()))     # the list covers the whole module
    dead = {k: _gsum(v) for k, v in groups.items() if not _gsum(v) > 0.0}
    assert not dead, dead
    assert _gsum(model.wm.proj.parameters()) > 0.0 and _gsum(model.adapter.parameters()) > 0.0
    zero = _loss_grads(world, CELL, 0.0)
    zp = zero.wm.stack.predictors[0]
    assert all(_gsum(v) == 0.0 for v in _groups(zp).values())                       # control: a zero WM weight reads exactly 0.0
    orphan = torch.nn.Parameter(torch.ones(3))                                       # a parameter outside the graph
    assert orphan.grad is None and _gsum([orphan]) == 0.0


# ------------------------------------------------------------------------------------------------ refusals and the gate
def test_config_refuses_what_it_cannot_build():
    with pytest.raises(ValueError, match="wm_state grid"):
        M.V8FMiniConfig(wm="b", hmax=16, wm_state="mean", wm_l1_pred="cell")
    with pytest.raises(ValueError, match="world-model"):
        M.V8FMiniConfig(wm="off", wm_state="mean", wm_l1_pred="cell")
    with pytest.raises(ValueError, match="wm_l1_pred"):
        M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred="ring")
    c = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred="cell")
    assert (c.wm_l1_dim, c.wm_l1_pred_width) == (640, 128)
    assert M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid").wm_l1_pred_width == 256        # the flat grid arm is unchanged
    with pytest.raises(ValueError, match="wm_state grid"):
        TR.make_config(_args(["--wm-l1-pred", "cell"]))                                       # the flag with a mean state is refused


def test_the_flag_has_its_entries_and_a_default():
    d = {act.dest for act in TR.build_parser()._actions}
    assert "wm_l1_pred" in d and "wm_l1_pred" in TR.FLAG_CHECKS and TR.build_parser().get_default("wm_l1_pred") == "flat"
    assert "wm_l1_pred" in TR.INIT_CKPT_MATCH


def test_declared_vs_built_goes_red_on_every_lie_in_both_directions(world):
    a, cfg, model, opt, info = _built(world, CELL)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    ag, cg, mg, og, ig = _built(world, GRID)
    assert TR.declared_vs_built(ag, cg, mg, og, ig) == []
    assert any("wm_l1_pred" in m for m in TR.declared_vs_built(_args(CELL), cg, mg, og, ig))      # declared cell, built flat
    assert any("wm_l1_pred" in m for m in TR.declared_vs_built(_args(GRID), cfg, model, opt, info))  # declared flat, built cell
    mg.wm.stack.predictors[0] = _cell()                                                           # the module itself is swapped under a flat config
    assert any("wm_l1_pred" in m for m in TR.declared_vs_built(ag, cg, mg, og, ig))
    a1, c1, m1, o1, i1 = _built(world, ["--arm", "M1"])                                           # wm off: a flat default is clean; the lever with nothing to build is refused at the config
    assert TR.declared_vs_built(a1, c1, m1, o1, i1) == []
    with pytest.raises(ValueError):
        TR.make_config(_args(["--arm", "M1", "--wm-l1-pred", "cell"]))                             # cell with wm off is refused at the config


def test_the_gate_refuses_a_cell_predictor_above_the_flat_budget(world, monkeypatch):
    a, cfg, model, opt, info = _built(world, CELL)
    monkeypatch.setattr(M.WMAux, "probe_pred", lambda self: {"kind": "cell", "cross_cell": True, "params": 10 ** 7, "params_flat_ref": 10 ** 6})
    assert any("c2" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


# ------------------------------------------------------------------------------------------------ checkpoints
def test_checkpoint_cfg_keys_and_the_t1_builder_round_trip(world, tmp_path):
    base = ["--synthetic", "--arm", "M2b", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--steps", "2"]
    assert TR.main([*base, "--out", str(tmp_path / "grid")] + GRID) == 0
    assert TR.main([*base, "--out", str(tmp_path / "cell")] + CELL) == 0
    assert TR.main([*base, "--out", str(tmp_path / "mean")]) == 0
    ld = lambda n: torch.load(tmp_path / n / "ckpt_last.pt", map_location="cpu", weights_only=False)   # noqa: E731
    cm, cg, cc = ld("mean"), ld("grid"), ld("cell")
    assert not (set(M.L1PRED_CFG_FIELDS) & set(cm["cfg"])) and not (set(M.L1PRED_CFG_FIELDS) & set(cg["cfg"]))
    assert set(cc["cfg"]) == set(cg["cfg"]) | set(M.L1PRED_CFG_FIELDS) and cc["cfg"]["wm_l1_pred"] == "cell"
    cfg = M.V8FMiniConfig(**cc["cfg"])                                                    # the T1 eval builder (v8f_t1_eval.cmd_run lines 506-509)
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(cc["model"])                                                     # strict
    assert model.built_levers()["wm_l1_pred"] == "cell"
    cfg_g = M.V8FMiniConfig(**cg["cfg"])                                                   # an earlier (grid, flat) checkpoint still loads as flat
    mg = M.V8FMini(cfg_g, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    mg.load_state_dict(cg["model"])
    assert mg.built_levers()["wm_l1_pred"] == "flat" and cfg_g.wm_l1_pred == "flat"
    ov = {k: cc["cfg"][k] for k in ("d", "ff", "heads", "n_q", "depth_s", "depth_t", "depth_o", "nb", "hmax", "hist")}
    a, cfg2, m2, o2, i2 = _built(world, GRID, **ov)                                              # a flat argv must not start from a cell checkpoint, nor the reverse
    with pytest.raises(ValueError, match="wm_l1_pred"):
        TR.load_init_ckpt(str(tmp_path / "cell" / "ckpt_last.pt"), m2, cfg2)
    a3, cfg3, m3, o3, i3 = _built(world, CELL, **ov)
    with pytest.raises(ValueError, match="wm_l1_pred"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m3, cfg3)
    rec = TR.load_init_ckpt(str(tmp_path / "cell" / "ckpt_last.pt"), m3, cfg3)             # the same lever is accepted
    assert rec["n_mismatch_after_load"] == 0
