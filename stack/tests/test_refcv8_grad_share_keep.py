"""refcv8 X4 -- the gradient-share reading's COST lever: ``--grad-share-terms`` / ``grad_share.term_tensors(keep=)``.

MEASURED 2026-10-11 (the R2 ladder's V0 on Thor): the first fp32-replay reading at step 250 ran > 20 min -- one
retained ``autograd.grad`` pass per term, each re-running the checkpointed forward -- and the arm died at its 2400 s
cap with no checkpoint. ``keep`` names the terms a reading must SEPARATE; every other term folds into ``rest``.

Pinned here:
  * ``parse_keep`` literals, and an unknown term is REFUSED (a typo would silently fold the meant term into rest);
  * ANALYTIC: three terms with known gradients, keep one -> its proj / norm are the closed-form literals of the
    full reading, rest absorbs the others, the shares still sum to 1;
  * on the REAL agent rig: keep=("agent",) gives the SAME agent proj/norm, norm_total and lin_rel_err as the full
    reading, in 3 passes (total + agent + rest) -- and two deliberate regressions go RED (keep ignored; the kept
    term dropped);
  * the timing keys are in the row;
  * the trainer: the pin refuses an unknown term and a term list with the instrument off; both call sites pass keep.
"""
from __future__ import annotations

import importlib
import inspect
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig  # noqa: E402,F401  (puts scripts/ on sys.path)
from tanitad.train import grad_share as GS  # noqa: E402

T = importlib.import_module("refc_v3_train")
_B, _N = 2, 8


# =========================================================================== #
# parse_keep                                                                    #
# =========================================================================== #
def test_parse_keep_literals():
    assert GS.parse_keep(None) is None
    assert GS.parse_keep("") is None
    assert GS.parse_keep("all") is None
    assert GS.parse_keep(" , ") is None
    assert GS.parse_keep("tac_v6,traj") == ("traj", "tac_v6")              # TERMS order, not argv order
    assert GS.parse_keep(["map_hires"]) == ("map_hires",)
    assert GS.parse_keep("agent, agent") == ("agent",)
    for bad in ("tacv6", "traj,rest", "loss"):                              # rest / loss are not separable terms
        with pytest.raises(ValueError, match="not in TERMS"):
            GS.parse_keep(bad)


# =========================================================================== #
# ANALYTIC                                                                      #
# =========================================================================== #
def _three():
    """theta in R^2; traj = 3 th0, agent = 4 th1 + 2 th0, box3d = th0 + th1 (weights 1).
    g_traj = (3, 0), g_agent = (2, 4), g_box = (1, 1), g_tot = (6, 5), |g_tot|^2 = 61.
    proj_traj = 18/61, proj_agent = 32/61, proj_box = 11/61."""
    th = torch.nn.Parameter(torch.zeros(2))
    traj = 3.0 * th[0]
    agent = 4.0 * th[1] + 2.0 * th[0]
    box = th[0] + th[1]
    total = traj + agent + box
    losses = {"loss": total, "traj": traj, "agent": agent, "box3d": box}
    return th, losses, {"traj": 1.0, "agent": 1.0, "box3d": 1.0}


def test_ANALYTIC_keep_one_term_gives_its_closed_form_and_rest_absorbs_the_others():
    th, losses, w = _three()
    full = GS.measure(GS.term_tensors(losses, w), losses["loss"], {"g": [th]})
    th, losses, w = _three()
    terms = GS.term_tensors(losses, w, keep=("agent",))
    assert set(terms) == {"agent", "rest"}
    one = GS.measure(terms, losses["loss"], {"g": [th]})
    assert full["gs_g_proj_agent"] == pytest.approx(32.0 / 61.0, abs=1e-12)
    assert one["gs_g_proj_agent"] == pytest.approx(32.0 / 61.0, abs=1e-12)
    assert one["gs_g_norm_agent"] == pytest.approx(math.sqrt(20.0), abs=1e-12)
    assert one["gs_g_proj_rest"] == pytest.approx(29.0 / 61.0, abs=1e-12)      # traj + box3d = (18 + 11) / 61
    assert one["gs_g_norm_total"] == pytest.approx(math.sqrt(61.0), abs=1e-12)
    assert one["gs_g_lin_rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert "gs_g_proj_traj" not in one and "gs_g_proj_box3d" not in one
    # total + agent + rest  /  total + 3 named + rest (the tip ALWAYS attaches rest, even when it is identically 0)
    assert one["gs_n_passes"] == 3.0 and full["gs_n_passes"] == 5.0
    for k in ("gs_measure_s", "gs_pass_s_max"):
        assert k in one and one[k] >= 0.0


def test_keep_accepts_the_cli_string_form():
    th, losses, w = _three()
    assert set(GS.term_tensors(losses, w, keep="agent,box3d")) == {"agent", "box3d", "rest"}
    th, losses, w = _three()
    assert set(GS.term_tensors(losses, w, keep="all")) == {"traj", "agent", "box3d", "rest"}   # = the tip's set
    with pytest.raises(ValueError):
        GS.term_tensors(losses, w, keep=("agnet",))


# =========================================================================== #
# the real rig                                                                  #
# =========================================================================== #
@pytest.fixture(scope="module")
def agent_rig():
    argv = ["--arm", "hier", "--out", "/tmp/r8_gs_keep_test", "--seed", "0", "--agents", "head",
            "--agent-queries", str(_N), "--w-agent", "1.0", "--agent-join", "j.xz"]
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(1234)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = 0.0
    model._w_tac_goal = 0.0
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model._w_agent = 0.7
    model.train()
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    g = torch.Generator().manual_seed(7)
    batch["agent_box"] = torch.rand(_B, _N, 4, generator=g) * 20.0 + 1.0
    batch["agent_yaw"] = torch.rand(_B, _N, generator=g) * 2.0 - 1.0
    batch["agent_cls"] = torch.randint(0, 3, (_B, _N), generator=g)
    valid = torch.ones(_B, _N, dtype=torch.bool)
    valid[1, _N // 2:] = False
    batch["agent_valid"] = valid
    batch["agent_occ"] = torch.zeros(_B, _N)
    batch["agent_rates"] = torch.zeros(_B, _N, 3)
    batch["agent_rates_mask"] = torch.zeros(_B, _N, dtype=torch.bool)
    batch["agent_label"] = torch.ones(_B, dtype=torch.bool)
    torch.manual_seed(99)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    return model, batch, losses


def _same_reading(full: dict, kept: dict, term: str, group: str = "trunk") -> bool:
    keys = (f"gs_{group}_proj_{term}", f"gs_{group}_norm_{term}", f"gs_{group}_norm_total")
    if not all(k in kept for k in keys):
        return False
    ok = all(math.isclose(full[k], kept[k], rel_tol=1e-5, abs_tol=1e-9) for k in keys)
    return ok and kept[f"gs_{group}_lin_rel_err"] <= 1e-4


def test_on_the_real_rig_keep_agent_is_the_full_reading_in_three_passes(agent_rig):
    model, batch, losses = agent_rig
    groups = GS.param_groups(model)
    w = GS.weights_of(model, T.TRAJ_WEIGHT)
    full = GS.measure(GS.term_tensors(losses, w), losses["loss"], groups)
    kept = GS.measure(GS.term_tensors(losses, w, keep=("agent",)), losses["loss"], groups)
    assert full["gs_n_passes"] >= 4.0                                # total + traj + agent + rest at least
    assert kept["gs_n_passes"] == 3.0
    assert _same_reading(full, kept, "agent")
    assert kept["gs_trunk_proj_agent"] != 0.0
    s = sum(v for k, v in kept.items() if k.startswith("gs_trunk_proj_"))
    assert s == pytest.approx(1.0, abs=1e-6)
    assert all(p.grad is None for p in model.parameters())


def test_DELIBERATE_REGRESSIONS_keep_ignored_and_kept_term_dropped_go_RED(agent_rig, monkeypatch):
    model, batch, losses = agent_rig
    groups = GS.param_groups(model)
    w = GS.weights_of(model, T.TRAJ_WEIGHT)
    full = GS.measure(GS.term_tensors(losses, w), losses["loss"], groups)
    real = GS.term_tensors

    # (1) keep IGNORED: the reading is right but costs every pass -- the defect that killed V0
    monkeypatch.setattr(GS, "term_tensors", lambda l, ww, keep=None: real(l, ww))
    ign = GS.measure(GS.term_tensors(losses, w, keep=("agent",)), losses["loss"], groups)
    assert ign["gs_n_passes"] != 3.0
    # (2) the KEPT term DROPPED: everything folds into rest -- cheap, and about nothing
    monkeypatch.setattr(GS, "term_tensors", lambda l, ww, keep=None: {"rest": l["loss"]})
    drop = GS.measure(GS.term_tensors(losses, w, keep=("agent",)), losses["loss"], groups)
    assert not _same_reading(full, drop, "agent")


# =========================================================================== #
# the trainer                                                                   #
# =========================================================================== #
def _pin(extra):
    from tanitad.refs import refcv8_conditioning as r8c
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--log-every", "50"] + extra)
    T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    return args


def test_the_pin_refuses_an_unknown_term_and_terms_with_the_instrument_off():
    with pytest.raises(SystemExit, match="grad-share-terms"):
        _pin(["--grad-share-every", "250", "--grad-share-terms", "tacv6"])
    with pytest.raises(SystemExit, match="grad-share-terms"):
        _pin(["--grad-share-terms", "tac_v6"])                              # every defaults to 0
    a = _pin(["--grad-share-every", "250", "--grad-share-terms", "tac_v6"])
    assert a.grad_share_terms == "tac_v6"
    assert _pin(["--grad-share-every", "250"]).grad_share_terms is None     # the tip's reading
    _pin(["--grad-share-terms", "all"])                                     # 'all' is None: nothing to refuse


def test_both_call_sites_pass_keep():
    src = inspect.getsource(T).replace("\r\n", "\n")
    tr = src[src.index("\ndef train("):]
    assert '_gs_keep = _gshare.parse_keep(getattr(args, "grad_share_terms", None))' in tr
    assert "_gshare.weights_of(model, TRAJ_WEIGHT), keep=_gs_keep)" in tr       # the fp32 replay
    assert "_gshare.weights_of(model, TRAJ_WEIGHT), _gs_keep)," in tr            # the in-graph reading
    # DELIBERATE REGRESSION: the in-graph site without keep is the tip's line, and it must not pass
    mut = tr.replace("_gshare.weights_of(model, TRAJ_WEIGHT), _gs_keep),", "_gshare.weights_of(model, TRAJ_WEIGHT)),")
    assert mut != tr and "_gshare.weights_of(model, TRAJ_WEIGHT), _gs_keep)," not in mut
