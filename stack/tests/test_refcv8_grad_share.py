"""refcv8 WP-B -- (a) the conflict detector sees the AGENT head; (b) the in-run GRADIENT-SHARE instrument.

(a) WP-D's P-GRAD (MEASURED, refcv7-50,400): the agent loss pays for 50.1 % of the whole-trunk update and was in
    NEITHER side of ``CONFLICT_PERCEPTION_TERMS``. Pinned: the row exists; the producer writes ``losses["agent"]``
    ATTACHED on a real ``compute_losses_v3`` with the agent head trained; ``_conflict_terms``' aux side contains
    ``w_agent * agent`` (value AND gradient); the pre-existing terms survive (control); a table without the row goes
    RED on the same check (deliberate regression).
(b) ``tanitad.train.grad_share`` -- P-GRAD's projection statistic in the trainer:
    * ANALYTIC: two terms with known gradients on a known parameter -> the projection shares are the literals of the
      closed form; a norm-share implementation (the plausible wrong statistic) goes RED;
    * a zero-weight term is not a term; ``rest`` closes the sum; the linearity control reads ~0;
    * on the REAL agent rig: shares sum to 1, linearity <= 1e-4 (P-GRAD's bar), and the instrument leaves ``.grad``
      and the global RNG untouched (the step is the step without it);
    * the trainer wiring: measured before the backward on logged steps only, merged after the rounding; the pin
      refuses a cadence that is not a multiple of --log-every.
"""
from __future__ import annotations

import importlib
import inspect
import math
import sys
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
# a real compute_losses_v3 with the AGENT head trained (smoke width, CPU)      #
# =========================================================================== #
@pytest.fixture(scope="module")
def agent_rig():
    argv = ["--arm", "hier", "--out", "/tmp/r8_gs_test", "--seed", "0", "--agents", "head",
            "--agent-queries", str(_N), "--w-agent", "1.0", "--agent-join", "j.xz"]
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(1234)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = 0.0
    model._w_tac_goal = 0.0
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model._w_agent = 0.7                                   # a non-unit weight, so a missing multiply shows
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


# =========================================================================== #
# (a) the conflict detector sees the agent head                                #
# =========================================================================== #
def _agent_in_aux(table, model, losses) -> bool:
    """The check, against a GIVEN table: the aux side of the detector contains w_agent * agent."""
    old = T.CONFLICT_PERCEPTION_TERMS
    T.CONFLICT_PERCEPTION_TERMS = table
    try:
        if "agent" not in T._conflict_aux_weights(model):
            return False
        _plan, aux = T._conflict_terms(model, losses)
        # RELATIVE (float32): the aux sum is a float32 product; an absolute 1e-6 at |aux| ~ 50 is below float32
        # resolution and flipped with the thread count (OMP 4 PASS / 6 FAIL, MEASURED 2026-10-04)
        return aux is not None and math.isclose(float(aux), 0.7 * float(losses["agent"]), rel_tol=1e-6)
    finally:
        T.CONFLICT_PERCEPTION_TERMS = old


def test_the_agent_row_is_in_the_table_and_the_old_rows_survive():
    names = [n for n, _ in T.CONFLICT_PERCEPTION_TERMS]
    assert "agent" in names
    assert {"bev", "map", "box3d", "tac_v6", "r7_wta", "r7_scorer", "map_hires"} <= set(names)    # the control

    class _M:
        _w_agent = 0.7
    assert T._conflict_aux_weights(_M()) == {"agent": pytest.approx(0.7)}

    class _Z:
        pass
    assert T._conflict_aux_weights(_Z()) == {}                          # a zero weight is not a term


def test_the_producer_writes_the_agent_total_ATTACHED(agent_rig):
    model, batch, losses = agent_rig
    assert "agent" in losses and losses["agent"].requires_grad and bool(torch.isfinite(losses["agent"]))
    assert float(losses["agent"]) > 0.0


def test_the_detector_aux_side_carries_w_agent_times_agent_and_its_gradient(agent_rig):
    model, batch, losses = agent_rig
    assert _agent_in_aux(T.CONFLICT_PERCEPTION_TERMS, model, losses)
    _plan, aux = T._conflict_terms(model, losses)
    p = next(p for n, p in model.named_parameters() if n.startswith("core.encoder.") and p.requires_grad)
    g_aux = torch.autograd.grad(aux, p, retain_graph=True, allow_unused=True)[0]
    g_ag = torch.autograd.grad(0.7 * losses["agent"], p, retain_graph=True, allow_unused=True)[0]
    assert g_ag is not None and float(g_ag.abs().sum()) > 0.0                # the agent loss reaches the trunk
    assert torch.allclose(g_aux, g_ag, atol=1e-7, rtol=1e-6)                 # the only live aux on this rig


def test_DELIBERATE_REGRESSION_a_table_without_the_agent_row_goes_RED(agent_rig):
    model, batch, losses = agent_rig
    without = tuple(r for r in T.CONFLICT_PERCEPTION_TERMS if r[0] != "agent")
    assert not _agent_in_aux(without, model, losses)


# =========================================================================== #
# (b) the gradient-share instrument -- analytic                                #
# =========================================================================== #
def test_ANALYTIC_projection_shares_are_the_closed_form_literals():
    """theta in R^2; L1 = 3 * theta_0, L2 = 4 * theta_1 + 2 * theta_0 -> g1 = (3, 0), g2 = (2, 4), g_tot = (5, 4).
    proj_1 = <g1, g_tot> / |g_tot|^2 = 15 / 41, proj_2 = 26 / 41; norms 3 and sqrt(20)."""
    th = torch.nn.Parameter(torch.zeros(2))
    l1 = 3.0 * th[0]
    l2 = 4.0 * th[1] + 2.0 * th[0]
    row = GS.measure({"a": l1, "b": l2}, l1 + l2, {"g": [th]})
    assert row["gs_g_proj_a"] == pytest.approx(15.0 / 41.0, abs=1e-12)
    assert row["gs_g_proj_b"] == pytest.approx(26.0 / 41.0, abs=1e-12)
    assert row["gs_g_norm_a"] == pytest.approx(3.0, abs=1e-12)
    assert row["gs_g_norm_b"] == pytest.approx(math.sqrt(20.0), abs=1e-12)
    assert row["gs_g_norm_total"] == pytest.approx(math.sqrt(41.0), abs=1e-12)
    assert row["gs_g_lin_rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert th.grad is None                                                    # .grad never touched
    # DELIBERATE REGRESSION: the norm share (|g_t| / sum |g|) is the plausible wrong statistic -- it is NOT this
    wrong_a = 3.0 / (3.0 + math.sqrt(20.0))
    assert abs(wrong_a - row["gs_g_proj_a"]) > 0.03


def test_term_tensors_drop_zero_weights_and_rest_closes_the_sum():
    th = torch.nn.Parameter(torch.ones(3))
    traj, box = (th * 2).sum(), (th * th).sum()
    total = 1.5 * traj + 0.0 * box + th.sum()                                # th.sum() plays "the planner aux"
    terms = GS.term_tensors({"loss": total, "traj": traj, "box3d": box, "agent": torch.tensor(1.0)},
                            {"traj": 1.5, "box3d": 0.0, "agent": 2.0})
    assert set(terms) == {"traj", "rest"}                                     # zero weight / detached: not terms
    row = GS.measure(terms, total, {"g": [th]})
    assert row["gs_g_proj_traj"] + row["gs_g_proj_rest"] == pytest.approx(1.0, abs=1e-12)
    assert row["gs_g_lin_rel_err"] == pytest.approx(0.0, abs=1e-12)


# =========================================================================== #
# (b) on the real rig                                                          #
# =========================================================================== #
def test_on_the_real_rig_shares_sum_to_one_and_the_linearity_control_passes(agent_rig):
    model, batch, losses = agent_rig
    groups = GS.param_groups(model)
    assert "trunk" in groups and sum(p.numel() for p in groups["trunk"]) > 0
    terms = GS.term_tensors(losses, GS.weights_of(model, T.TRAJ_WEIGHT))
    assert {"traj", "agent", "rest"} <= set(terms)
    rng = torch.get_rng_state()
    for p in model.parameters():
        p.grad = None
    row = GS.measure(terms, losses["loss"], groups)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())
    s = sum(v for k, v in row.items() if k.startswith("gs_trunk_proj_"))
    assert s == pytest.approx(1.0, abs=1e-6)
    assert row["gs_trunk_lin_rel_err"] <= 1e-4
    assert row["gs_trunk_proj_agent"] != 0.0


# =========================================================================== #
# (b) the trainer wiring                                                       #
# =========================================================================== #
def test_the_cadence_must_be_a_multiple_of_log_every():
    import types
    from tanitad.refs import refcv8_conditioning as r8c
    for bad in (["--grad-share-every", "70"], ["--grad-share-every", "-50"]):
        args = T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--log-every", "50"] + bad)
        with pytest.raises(SystemExit, match="grad-share-every"):
            T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--log-every", "50", "--grad-share-every",
                                        "500"])
    T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    assert T.build_parser().parse_args(["--arm", "hier", "--out", "X"]).grad_share_every == 0


def _wiring_ok(src: str) -> bool:
    tr = src[src.index("\ndef train("):]
    m = tr.index("_gs_row = _gshare.measure(")
    b = tr.index('losses["loss"].backward()')
    r = tr.index("row.update(_gs_row)")
    rnd = tr.index("row = _train_row_scalars(losses, model)")
    gate = tr[tr.rindex("if (_gs_every > 0", 0, m):m]
    return m < b and r > rnd and "_logged_after(step, args.log_every, args.steps)" in gate


def test_the_trainer_measures_before_the_backward_on_logged_steps_and_merges_after_rounding():
    src = inspect.getsource(T).replace("\r\n", "\n")
    assert _wiring_ok(src)
    # DELIBERATE REGRESSION: measuring on every step (the logged-step gate removed) goes RED
    assert not _wiring_ok(src.replace("if (_gs_every > 0 and _logged_after(step, args.log_every, args.steps)",
                                      "if (_gs_every > 0 and True", 1))
