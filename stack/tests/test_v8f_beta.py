"""v8F rung BETA: the world model as an eval-time scorer of the 117-candidate fan (PREREG_V8F_BETA.md). CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts pytest -q stack/tests/test_v8f_beta.py

What is pinned, and the control that must go red for each:
  * the fan IS the shipped 13 x 9 grid and IS what a real ``AnchoredDiffusionDecoder.roll_bank`` rolls;
  * the path -> action derivation recovers constant controls, matches the cache's convention (``ego_channels_from_poses``), and the route
    goal is the polyline timed by P10;
  * KNOWN ANSWER: on a world model whose level-2 endpoint is an analytic function of the summed actions, beta picks exactly the candidate
    the closed-form cost says (and ``fan_idx`` / ``cost_min`` / ``cost_inc`` equal it);
  * IDENTITY: a constant cost reproduces M2's own output bit for bit, at the planner and END TO END through ``v8f_mini_eval run --plan beta``;
  * PARTIAL: replacement is neither all nor nothing;
  * DELIBERATE-REGRESSION arms (source mutants, exec'd): delete the scoring line / flip argmin / drop the goal / share the actions / make
    the improvement non-strict => the check that owns it goes RED ("MUTATE, do not inspect": CLAUDE.md, built-tested-unreachable rule 3);
  * REACHABILITY from the actual caller: ``--plan beta`` makes ``HJEPAStack.score_candidates`` run on [B, 118, 30, 2] blocks at level 2; the
    default ``--plan none`` runs it ZERO times; the output records how many times it ran;
  * ADMISSIBILITY (binding 2026-08-03): the goal is a function of (ck_now, p30) only, the module names no situation classifier, and
    perturbing T's heads moves M2's plan but NOT the goal;
  * the instrument: per-window family components reproduce the family blocks' printed numbers; known-value route adherence; the decision table.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_beta as B  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from tanitad.models.hjepa import HJEPAStack  # noqa: E402
from tanitad.models.kinematic import rollout_unicycle  # noqa: E402
from taniteval import four_families as ff  # noqa: E402
from taniteval import v8f_beta_eval as BE  # noqa: E402
import v8f_mini_eval as EV  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
WM_DIM = 128                                    # LevelConfig default via V8FMiniConfig.wm_dim


# ------------------------------------------------------------------------------------------------ analytic world models
class _LinStub(nn.Module):
    """z_{t+1} = z_t + a_t @ P: the level-2 endpoint is P applied to the SUMMED actions, so beta's cost has a closed form."""

    def __init__(self, dim, a_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.P = nn.Parameter(torch.randn(a_dim, dim, generator=g), requires_grad=False)
        self.context = 3

    def forward(self, z_ctx, a_ctx):
        return z_ctx + a_ctx @ self.P


class _ConstStub(nn.Module):
    """A world model that ignores the actions: every candidate lands on the same latent => a CONSTANT goal cost."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.context = 3

    def forward(self, z_ctx, a_ctx):
        return z_ctx * 1.0


def make_wm(kind="lin", seed=0):
    cfg = M.V8FMiniConfig(**{**TINY, "wm": "a", "hmax": 16, "kin_mode": "yaw"})
    wm = M.WMAux(cfg).eval()
    st = wm.stack
    st.act_embed = nn.Identity()
    st.encoders[0] = nn.Identity()
    st.predictors[0] = _LinStub(cfg.wm_dim, 2, seed) if kind == "lin" else _ConstStub()
    return wm


def make_inputs(mod, n=6, seed=0, incumbent="stationary"):
    g = torch.Generator().manual_seed(seed)
    v0 = 3.0 + 25.0 * torch.rand(n, generator=g)
    y = (torch.rand(n, 3, generator=g) - 0.5) * 8.0
    ck = torch.stack([torch.stack([torch.full((n,), 30.0), y[:, 0], 0.05 * y[:, 0]], -1),
                      torch.stack([torch.full((n,), 50.0), y[:, 1], 0.05 * y[:, 1]], -1),
                      torch.stack([torch.full((n,), 80.0), y[:, 2], 0.05 * y[:, 2]], -1)], 1)
    j = torch.arange(1, 31, dtype=torch.float32)
    p30 = torch.stack([v0[:, None] * 0.2 * j[None], torch.zeros(n, 30)], -1)
    t = torch.arange(1, 61, dtype=torch.float32) * 0.1
    if incumbent == "runaway":                                    # a = 20 m/s^2 straight: far costlier than any fan member under the stub
        traj60 = torch.stack([10.0 * t ** 2, torch.zeros(60)], -1)[None].expand(n, 60, 2).contiguous()
    elif incumbent == "stationary":
        traj60 = torch.zeros(n, 60, 2)
    else:
        traj60 = torch.randn(n, 60, 2, generator=g).cumsum(1) * 0.05
    return mod.BetaInputs(z1_0=torch.zeros(n, WM_DIM), v0=v0, now_act=torch.zeros(n, 2), ck_now=ck, p30=p30, traj60=traj60)


def expected_costs(wm, inp, goal_xy60):
    """Closed-form costs on the linear world model: ||(S_cand - S_goal) P||^2 with S = the summed SCALED actions."""
    P = wm.stack.predictors[0].P
    fan = B.fan_paths(inp.v0)
    xy = torch.cat([inp.traj60[:, None], fan], 1)
    now = inp.now_act[:, None].expand(-1, xy.shape[1], -1)
    s_c = wm.scale_actions(B.wm_action_sequence(xy, now)).sum(2)                                      # [n, 118, 2]
    s_g = wm.scale_actions(B.wm_action_sequence(goal_xy60[:, None], inp.now_act[:, None])).sum(2)     # [n, 1, 2]
    return ((s_c - s_g) @ P).pow(2).sum(-1), (s_g @ P)[:, 0]


# ------------------------------------------------------------------------------------------------ the checks the mutants must fail
def check_known_answer(mod):
    wm = make_wm("lin")
    inp = make_inputs(mod, n=6, seed=1, incumbent="runaway")
    gxy = mod.route_goal_xy(inp.ck_now, inp.p30)
    goal = mod.goal_latents(wm, inp, gxy, chunk=4)
    cost, want_goal = expected_costs(wm, inp, gxy)
    assert torch.allclose(goal, want_goal, rtol=1e-4, atol=1e-4), "the goal latent is not the world model's endpoint under the goal path"
    plan = mod.plan_beta(wm, inp, goal, chunk=4)
    best_c, best_i = cost[:, 1:].min(1)
    assert torch.equal(plan["fan_idx"].cpu(), best_i), "beta did not pick the closed-form best candidate"
    assert torch.allclose(plan["cost_min"], best_c, rtol=1e-3, atol=1e-3)
    assert torch.allclose(plan["cost_inc"], cost[:, 0], rtol=1e-3, atol=1e-3)
    assert torch.equal(plan["replaced"].cpu(), best_c < cost[:, 0]) and plan["replaced"].all(), "the runaway incumbent must lose to the fan"
    fan30 = B.fan_paths(inp.v0)[:, :, 1::2]
    assert torch.equal(plan["traj30"], fan30[torch.arange(6), best_i])


def check_identity(mod):
    wm = make_wm("const")
    inp = make_inputs(mod, n=6, seed=2, incumbent="random")
    goal = torch.randn(6, WM_DIM, generator=torch.Generator().manual_seed(3))
    plan = mod.plan_beta(wm, inp, goal, chunk=4)
    assert torch.equal(plan["cost_inc"], plan["cost_min"]), "the stub is supposed to make the cost constant across candidates"
    assert not plan["replaced"].any(), "a constant cost replaced the incumbent"
    assert torch.equal(plan["traj30"], inp.traj60[:, 1::2]), "a constant cost did not reproduce M2's own output bit for bit"


def check_partial(mod):
    wm = make_wm("lin")
    inp = make_inputs(mod, n=6, seed=4, incumbent="runaway")
    gxy = mod.route_goal_xy(inp.ck_now, inp.p30)
    traj60 = inp.traj60.clone()
    traj60[:3] = gxy[:3]                                       # the incumbent IS the goal path: cost 0, nothing can beat it strictly
    inp = mod.BetaInputs(inp.z1_0, inp.v0, inp.now_act, inp.ck_now, inp.p30, traj60)
    plan = mod.plan_beta(wm, inp, mod.goal_latents(wm, inp, gxy, chunk=4), chunk=4)
    assert plan["replaced"].tolist() == [False] * 3 + [True] * 3
    assert torch.equal(plan["traj30"][:3], traj60[:3, 1::2])


# ------------------------------------------------------------------------------------------------ fan
def test_beta_fan_is_the_shipped_13x9_grid():
    c = B.fan_controls()
    assert c.shape == (117, 2)
    assert len(set(c[:, 0].tolist())) == 13 and len(set(c[:, 1].tolist())) == 9
    assert ((c.abs() < 1e-12).all(-1)).any(), "the no-action control must exist EXACTLY"
    assert set(c[:, 1].tolist()) == {-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0}
    assert float(c[:, 0].min()) == -4.0 and abs(float(c[:, 0].max()) - 5 * 7 / 12) < 1e-6


def test_beta_fan_paths_equal_a_real_decoder_roll_bank():
    from tanitad.refs import refc
    hz = (5, 10, 15, 20, 30, 40, 50, 60)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=64, n_steps=len(hz), d_meas=8, d_ctx=8, tac_latent_dim=8, anchors=torch.zeros(117, len(hz), 2),
        cfg=refc.DecoderConfig(), hierarchy=False, graft_maneuver=False, graft_target_latent=False, grounded_selector=False,
        horizons=hz, v0_conditioned=True, ref_speed_ms=10.0, control_units="alat", alat_v_floor_ms=4.0, kappa_cap=0.12)
    with torch.no_grad():
        dec.anchor_controls.copy_(B.fan_controls())
    v = torch.tensor([0.0, 0.5, 3.0, 4.0, 10.0, 18.0, 30.0])
    bank = dec.roll_bank(v, None, v.shape[0], torch.float32)                     # [B, 117, 8, 2]: the decoder's own bank
    mine = B.fan_paths(v)[:, :, dec.anchor_slots]
    assert bank.shape == mine.shape == (7, 117, 8, 2)
    assert torch.allclose(bank, mine, rtol=0, atol=1e-5)
    # CONTROL: the same bank rolled at a DIFFERENT speed differs, so the comparison can fail
    assert not torch.allclose(bank, B.fan_paths(v + 2.0)[:, :, dec.anchor_slots], atol=1e-3)


def test_beta_fan_zero_control_is_constant_velocity_and_a_standing_ego_stays_put():
    p = B.fan_paths(torch.tensor([0.0, 10.0, 25.0]))
    i0 = int(((B.fan_controls().abs() < 1e-12).all(-1)).nonzero()[0])
    assert [round(float(p[i, i0, -1, 0]), 3) for i in range(3)] == [0.0, 60.0, 150.0]
    c = B.fan_controls()
    assert float(p[0, i0].abs().max()) == 0.0                       # a standing ego with no control stays exactly where it is
    assert float(p[0, c[:, 0] <= 0].abs().max()) == 0.0             # ... and so does one told to brake (speed is clamped at 0)
    assert float(p[0, :, 0, 1].abs().max()) == 0.0                  # step 1 starts at v = 0: nothing can turn yet
    assert float(p[0, c[:, 0] > 0].abs().amax((-1, -2)).min()) > 0.0   # control: every accelerating member DOES move


def test_beta_fan_artifact_is_read_through_the_declared_units_reader(tmp_path):
    ctrl = B.fan_controls()
    good = tmp_path / "anchors_117_alat_declared.pt"
    torch.save({"anchors": torch.zeros(117, 8, 2), "controls": ctrl, "control_units": "alat"}, good)
    got, rec = B.load_fan_controls(good)
    assert torch.equal(got, ctrl) and rec["max_abs_diff_vs_pinned_grid"] == 0.0 and len(rec["sha256"]) == 64
    legacy = tmp_path / "legacy.pt"                                              # no declared units: refused (the 396 g incident)
    torch.save({"anchors": torch.zeros(117, 8, 2), "controls": ctrl}, legacy)
    with pytest.raises(ValueError):
        B.load_fan_controls(legacy)
    wrong = tmp_path / "wrong.pt"
    torch.save({"anchors": torch.zeros(64, 8, 2), "controls": ctrl[:64], "control_units": "alat"}, wrong)
    with pytest.raises(ValueError, match="117 x 2"):
        B.load_fan_controls(wrong)


# ------------------------------------------------------------------------------------------------ path -> the WM's action channels
def test_beta_action_sequence_recovers_constant_controls():
    a, k, v = 1.5, 0.04, 8.0
    s0 = torch.zeros(1, 4)
    s0[:, 3] = v
    xy = rollout_unicycle(s0, torch.tensor([[a, k]]).expand(1, 60, 2), dt=0.1)[..., :2]
    seq = B.wm_action_sequence(xy, torch.tensor([[a, 0.123]]))
    assert seq.shape == (1, 30, 2)
    assert torch.allclose(seq[0, :, 0], torch.full((30,), a), atol=2e-4)                         # a_j = a at every j (j = 0 is the measured one)
    assert float(seq[0, 0, 1]) == pytest.approx(0.123)                                            # transition 0 = the MEASURED omega
    want = v * k + a * k * 0.1 * (2 * torch.arange(1, 30) - 2)                                   # omega = v(t) * kappa, v at the start of the step
    assert torch.allclose(seq[0, 1:, 1], want, atol=2e-3)


def test_beta_action_sequence_matches_the_caches_own_convention():
    """Poses from a constant-control roll; the CACHE's channels are a = (v[2j]-v[2j-2])/0.2 and omega = ego_channels_from_poses (one-frame)."""
    from tanitad.models.ego_history import ego_channels_from_poses
    a, k, v = -1.0, 0.03, 14.0
    s0 = torch.zeros(1, 4)
    s0[:, 3] = v
    st = rollout_unicycle(s0, torch.tensor([[a, k]]).expand(1, 60, 2), dt=0.1)
    poses = torch.cat([s0[:, None], st], 1)                                                       # [1, 61, 4] x y yaw v at frames 0..60
    ch = ego_channels_from_poses(poses, 61, dt=0.1)[0]                                            # (v, dv, dyaw) per frame, backward diffs
    seq = B.wm_action_sequence(st[..., :2], torch.zeros(1, 2))[0]
    for j in range(2, 30):
        cache_a = float(poses[0, 2 * j, 3] - poses[0, 2 * j - 2, 3]) / 0.2
        assert float(seq[j, 0]) == pytest.approx(cache_a, abs=2e-3), j
        assert float(seq[j, 1]) == pytest.approx(float(ch[2 * j, 2]), abs=2.5e-2), j              # chord vs frame heading: a one-step lag in v
    with pytest.raises(ValueError):
        B.wm_action_sequence(st[..., :2, :], torch.zeros(1, 2))                                   # not 60 points
    with pytest.raises(ValueError):
        B.wm_action_sequence(st[..., :2], torch.zeros(1, 3))                                      # now_act must be (a, omega)


def test_beta_chord_kinematics_holds_the_heading_through_a_stop():
    xy = torch.zeros(1, 60, 2)
    xy[0, :10, 0] = torch.arange(1, 11) * 1.0                       # drives 10 steps along +x ... then stops
    xy[0, 10:, 0] = 10.0
    sp, hd = B.chord_kinematics(xy)
    assert float(sp[0, 0]) == pytest.approx(10.0) and float(sp[0, 20]) == 0.0
    assert float(hd[0, 30]) == float(hd[0, 9]) == 0.0
    xy2 = xy.clone()
    xy2[0, 10:, 1] = torch.arange(1, 51) * 0.0 + 0.0
    assert torch.isfinite(B.wm_action_sequence(xy2, torch.zeros(1, 2))).all()


# ------------------------------------------------------------------------------------------------ the goal
def test_beta_route_goal_is_the_polyline_timed_by_p10():
    n = 2
    p30 = torch.stack([torch.arange(1, 31).float() * 2.0, torch.zeros(30)], -1)[None].expand(n, 30, 2).contiguous()     # 10 m/s straight
    straight = torch.tensor([[30., 0, 0], [50., 0, 0], [80., 0, 0]])[None].expand(n, 3, 3).contiguous()
    g = B.route_goal_xy(straight, p30)
    assert g.shape == (n, 60, 2)
    assert torch.allclose(g[0, :, 0], torch.arange(1, 61).float() * 1.0, atol=1e-3) and float(g[0, :, 1].abs().max()) < 1e-4
    # a 90-degree left turn at 30 m: the route runs up the y axis; P10's 60 m of pace puts the goal 30 m past the corner
    left = torch.tensor([[30., 0, 0], [30., 20., math.pi / 2], [30., 50., math.pi / 2]])[None].expand(n, 3, 3).contiguous()
    g2 = B.route_goal_xy(left, p30)
    assert torch.allclose(g2[0, -1], torch.tensor([30.0, 30.0]), atol=1e-3)
    # the goal travels exactly P10's arc length along ANY route
    arc = torch.linalg.norm(g2[0, 1:] - g2[0, :-1], dim=-1).sum() + torch.linalg.norm(g2[0, 0])
    assert float(arc) == pytest.approx(60.0, abs=1e-2)
    # a standing P10 keeps the goal at the ego
    still = B.route_goal_xy(straight, torch.zeros(n, 30, 2))
    assert float(still.abs().max()) < 1e-4


def test_beta_goal_is_a_function_of_ck_and_p10_only():
    sig = list(inspect.signature(B.route_goal_xy).parameters)
    assert sig == ["ck_now", "p30"], sig
    # the module's CODE names no situation classifier / tactical output (docstrings and comments are not code)
    assert M.scan_source_for_forbidden(inspect.getsource(B)) == []
    import ast
    fn = ast.parse(inspect.getsource(B.route_goal_xy)).body[0]
    idents = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    idents |= {a.arg for a in fn.args.args}
    assert idents.isdisjoint({"goal_prob", "lat_logits", "lon_logits", "cons_pred", "out", "batch", "model", "gt", "lab", "labels"}), idents


# ------------------------------------------------------------------------------------------------ the scorer: known answer, identity, partial
def test_beta_known_answer_on_an_analytic_world_model():
    check_known_answer(B)


def test_beta_constant_cost_reproduces_the_unplanned_output_exactly():
    check_identity(B)


def test_beta_replacement_is_partial_not_all_or_nothing():
    check_partial(B)


def test_beta_scores_through_hjepa_score_candidates_on_118_blocks_at_level_2(monkeypatch):
    calls = []
    orig = HJEPAStack.score_candidates

    def spy(self, z1_t, blocks, goal, level=2, **kw):
        calls.append((tuple(blocks.shape), level, kw.get("return_details", False)))
        return orig(self, z1_t, blocks, goal, level, **kw)
    monkeypatch.setattr(HJEPAStack, "score_candidates", spy)
    wm = make_wm("lin")
    inp = make_inputs(B, n=3)
    B.reset_counters()
    g = B.goal_latents(wm, inp, B.route_goal_xy(inp.ck_now, inp.p30), chunk=3)
    B.plan_beta(wm, inp, g, chunk=3)
    assert ((3, 1, 30, 2), 2, True) in calls and ((3, 118, 30, 2), 2, True) in calls, calls
    assert B.SCORE_CALLS == {"goal": 1, "candidates": 1}


# ------------------------------------------------------------------------------------------------ deliberate-regression arms
def _mutant(edits, name="v8f_beta_mutant"):
    """The real module's source with each ``(old, new)`` applied exactly once, exec'd as a module that is IN sys.modules while it executes
    (a dataclass with ``from __future__ import annotations`` resolves its annotations through it)."""
    src = Path(B.__file__).read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, f"mutation anchor must occur exactly once, found {src.count(old)}: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = B.__file__
    sys.modules[name] = mod
    try:
        exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


SCORING = "cost, det = wm.stack.score_candidates(z1, acts, goal[sl].to(dev), level=WM_LEVEL, return_details=True)   # BETA-SCORING-LINE"
MUTANTS = {
    "delete_the_scoring_line": [(SCORING, "cost = torch.zeros(acts.shape[:2], device=acts.device)\n"
                                          "        det = {'lifted': torch.zeros(acts.shape[0], acts.shape[1], 2, goal.shape[-1], device=acts.device)}")],
    "pick_the_worst_candidate": [("best_c, best_i = cost[:, 1:].min(1)", "best_c, best_i = cost[:, 1:].max(1)")],
    "drop_the_goal": [("goal[sl].to(dev), level=WM_LEVEL", "goal[sl].to(dev) * 0.0, level=WM_LEVEL")],
    "share_one_action_sequence_across_candidates": [("acts = wm.scale_actions(wm_action_sequence(xy, now",
                                                     "acts = wm.scale_actions(wm_action_sequence(xy[:, :1].expand_as(xy), now")],
    "non_strict_improvement": [("replaced = best_c < cost[:, 0]", "replaced = best_c <= cost[:, 0]")],
}
OWNER = {"delete_the_scoring_line": check_known_answer, "pick_the_worst_candidate": check_known_answer, "drop_the_goal": check_known_answer,
         "share_one_action_sequence_across_candidates": check_known_answer, "non_strict_improvement": check_identity}


def test_beta_the_real_module_passes_every_owner_check_the_mutants_must_fail():
    for fn in set(OWNER.values()):
        fn(B)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_beta_deliberate_regression_arm_goes_red(name):
    mut = _mutant(MUTANTS[name], name=f"v8f_beta_mut_{name}")
    with pytest.raises(AssertionError):
        OWNER[name](mut)


def test_beta_the_scoring_line_marker_exists_once_in_the_implementation():
    """A grep for the guard that finds it only in the test file means it does not exist (CLAUDE.md, reachability rule 5)."""
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert src.count("# BETA-SCORING-LINE") == 1 and src.count("score_candidates(") >= 2
    assert src.count(SCORING) == 1


# ------------------------------------------------------------------------------------------------ admissibility
def test_beta_collect_reads_only_inference_keys_and_no_tactical_output(tmp_path):
    root = tmp_path / "w"
    w = D.make_synthetic_world(root, n_train=4, n_eval=2, T=60, seed=0)
    model, b = _tiny_m2_and_batch(root, w)
    out = model.forward_infer(b)
    slim = {k: out[k] for k in ("adapt", "ckpt", "p10", "traj60")}                               # NOTHING of T's: no goal_prob / lat / lon / cons
    row = B.collect_beta_inputs(model, b, slim)
    assert set(row) == set(B._INPUT_FIELDS) and all(torch.isfinite(v).all() for v in row.values())
    assert row["traj60"].shape[-2:] == (60, 2) and row["ck_now"].shape[-2:] == (3, 3)
    # a guard that can fail: with 'act' removed from the admissible set the same call is refused
    keep = M.INFERENCE_INPUT_KEYS
    M.INFERENCE_INPUT_KEYS = keep - {"act"}
    try:
        with pytest.raises(AssertionError, match="outside the admissible set"):
            B.collect_beta_inputs(model, b, slim)
    finally:
        M.INFERENCE_INPUT_KEYS = keep
    # a model without a world model is refused with the arm's name
    off = M.V8FMini(M.V8FMiniConfig(**TINY), M.P10Prior.synthetic(0))
    with pytest.raises(ValueError, match="M2a"):
        B.collect_beta_inputs(off, b, slim)


def test_beta_perturbing_the_tactical_layer_moves_the_plan_but_not_the_goal(tmp_path):
    root = tmp_path / "w"
    w = D.make_synthetic_world(root, n_train=4, n_eval=2, T=60, seed=0)
    model, b = _tiny_m2_and_batch(root, w)
    with torch.no_grad():
        model.head_o[2].weight.add_(0.3)                                                     # O reads T's code only if its head is non-zero
    r0 = B.collect_beta_inputs(model, b, model.forward_infer(b))
    with torch.no_grad():
        for h in (model.h_goal, model.h_lat, model.h_lon, model.h_cons):
            h.weight.add_(torch.randn_like(h.weight) * 2.0)                                  # a different tactical layer
    r1 = B.collect_beta_inputs(model, b, model.forward_infer(b))
    assert not torch.equal(r0["traj60"], r1["traj60"]), "control: the perturbation must move M2's plan (the incumbent)"
    for k in ("ck_now", "p30", "z1_0", "v0", "now_act"):
        assert torch.equal(r0[k], r1[k]), k
    g0 = B.route_goal_xy(r0["ck_now"], r0["p30"])
    g1 = B.route_goal_xy(r1["ck_now"], r1["p30"])
    assert torch.equal(g0, g1)
    wm = model.wm
    inp0, inp1 = B.cat_inputs([r0]), B.cat_inputs([r1])
    assert torch.equal(B.goal_latents(wm, inp0, g0), B.goal_latents(wm, inp1, g1))


def test_beta_wmaux_refactor_is_bit_identical_to_the_inline_training_formula(tmp_path):
    root = tmp_path / "w"
    w = D.make_synthetic_world(root, n_train=4, n_eval=2, T=60, seed=0)
    model, b = _tiny_m2_and_batch(root, w)
    out = model.forward_infer(b)
    wm = model.wm
    gen_state = torch.get_rng_state()
    new = wm(out["adapt"], b["act"])["total"]
    torch.set_rng_state(gen_state)
    z1 = wm.proj(out["adapt"].mean(2))
    legacy = wm.stack.loss(z1, torch.stack([b["act"][..., 0] / 2.0, b["act"][..., 1] / wm.ch1_scale], -1)[:, :-1])["total"]
    assert torch.equal(new, legacy)


# ------------------------------------------------------------------------------------------------ end to end through the actual caller
def _tiny_m2_and_batch(root, w, wm="a", kin_mode="yaw", noisy_head=True):
    cfg = M.V8FMiniConfig(**{**TINY, "wm": wm, "hmax": 16, "kin_mode": kin_mode})
    J = D.load_join(Path(root) / "join_eval.npz")
    mu, sd = M.cons_stats(J)
    torch.manual_seed(0)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    if noisy_head:
        with torch.no_grad():
            model.head_o[2].weight.add_(torch.randn_like(model.head_o[2].weight) * 0.05)
    model.eval()
    tracks = D.load_native_tracks(Path(root) / "native_tracks.pt") if kin_mode == "yaw" else None
    store = D.ClipStore(root, "eval", lru=4, kin_mode=kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(Path(root) / "eval_grid_windows.json", J)
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    items = [ds.item(ci, t, 1, 3) for ci, t, _, _ in wins[:8]]
    return model, D.collate(items)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    w = D.make_synthetic_world(root, n_train=4, n_eval=3, T=60, seed=0)
    cks = {}
    for name, kw in (("m2_s0", dict(wm="a")), ("m2_s1", dict(wm="a")), ("m1", dict(wm="off")), ("m2_steer", dict(wm="a", kin_mode="steer"))):
        model, _ = _tiny_m2_and_batch(root, w, **kw)
        if name == "m2_s1":
            with torch.no_grad():
                model.head_o[2].weight.add_(0.02)
        p = root / f"{name}.pt"
        torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__, "grid": "pool4x10"}, p)
        cks[name] = str(p)
    return {"root": root, "cks": cks}


def _run(e2e, ck, tag, out, *extra, ev=None):
    r = Path(e2e["root"])
    return (ev or EV).main(["run", "--ckpt", e2e["cks"][ck], "--tag", tag, "--eval-cache", str(r), "--eval-grid", str(r / "eval_grid_windows.json"),
                    "--eval-kin", str(r / "eval_kin.npz"), "--eval-join", str(r / "join_eval.npz"), "--out-dir", str(out), "--n-boot", "40",
                    "--batch", "16", "--lru", "4", "--tracks", str(r / "native_tracks.pt"), "--beta-chunk", "8", *extra])


def test_beta_the_default_plan_is_none_and_runs_the_scorer_zero_times(e2e, tmp_path, monkeypatch):
    calls = []
    orig = HJEPAStack.score_candidates
    monkeypatch.setattr(HJEPAStack, "score_candidates", lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1])
    assert _run(e2e, "m2_s0", "none", tmp_path) == 0
    j = json.loads((tmp_path / "eval_none.json").read_text(encoding="utf-8"))
    assert j["plan"] == "none" and "plan_beta" not in j and not (tmp_path / "eval_none_beta.npz").exists()
    assert calls == [], "the default must not run the world-model scorer (and a default that disables a mechanism has to be named in argv)"


def test_beta_flag_reaches_score_candidates_from_the_run_command(e2e, tmp_path, monkeypatch):
    seen = []
    orig = HJEPAStack.score_candidates

    def spy(self, z1_t, blocks, goal, level=2, **kw):
        seen.append((tuple(blocks.shape[1:]), level))
        return orig(self, z1_t, blocks, goal, level, **kw)
    monkeypatch.setattr(HJEPAStack, "score_candidates", spy)
    assert _run(e2e, "m2_s0", "b", tmp_path, "--plan", "beta") == 0
    assert ((118, 30, 2), 2) in seen and ((1, 30, 2), 2) in seen, set(seen)
    j = json.loads((tmp_path / "eval_b.json").read_text(encoding="utf-8"))
    pb = j["plan_beta"]
    assert j["plan"] == "beta" and pb["score_calls"]["candidates"] >= 3 * 1 and pb["score_calls"]["goal"] >= 3
    assert pb["spec"]["tuned_parameters"] == 0 and pb["n_windows"] == len(np.load(tmp_path / "eval_b.npz")["eid"])
    z = np.load(tmp_path / "eval_b_beta.npz")
    n = pb["n_windows"]
    assert z["beta__pred"].shape == (n, 30, 2) and z["m2__pred"].shape == (n, 30, 2)
    assert np.isfinite(z["beta__pred"]).all()
    # the four families are all present for beta, per component, paired against M2, with the estimator named
    fam = pb["paired_vs_m2"]["families"]
    assert set(fam) == set(BE.COMPONENTS) and "paired episode-cluster bootstrap" in pb["paired_vs_m2"]["estimator"]
    assert {f for f, _ in BE.COMPONENTS.values()} == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC (plan)", "CONSISTENCY"}
    assert "beta" in pb["arms"] and all(k in pb["arms"] for k in ("beta_shuf", "beta_oracle", "fan_random", "fan_oracle"))
    assert "strategic" in pb["arms"]["beta"] and "tactical" in pb["arms"]["beta"]                  # S / T heads' own metrics are carried
    assert set(pb["decision_inputs"]) == {"ade6_beta_minus_m2", "ade6_shuf_minus_beta", "families", "replaced_frac", "signal"}
    assert 0.0 <= pb["wm_diag"]["replaced_frac"] <= 1.0


def _caller_mutant(edits, name):
    """``v8f_mini_eval.py`` with each edit applied exactly once, exec'd as a module (the real CLI entry, mutated)."""
    src = Path(EV.__file__).read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, f"caller-mutation anchor must occur exactly once, found {src.count(old)}: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = EV.__file__
    sys.modules[name] = mod
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


def _scorer_calls_from_run(e2e, out, ev, monkeypatch, tag):
    seen = []
    orig = HJEPAStack.score_candidates

    def spy(self, z1_t, blocks, goal, level=2, **kw):
        seen.append((tuple(blocks.shape[1:]), level))
        return orig(self, z1_t, blocks, goal, level, **kw)
    monkeypatch.setattr(HJEPAStack, "score_candidates", spy)
    _run(e2e, "m2_s0", tag, out, "--plan", "beta", ev=ev)
    monkeypatch.undo()
    return seen


def test_beta_deliberate_regression_of_the_caller_goes_red(e2e, tmp_path, monkeypatch):
    """The 'built, tested and unreachable' class: delete the caller's use of the flag and the reachability probe must see ZERO scorer calls."""
    real = _scorer_calls_from_run(e2e, tmp_path / "real", EV, monkeypatch, "r")
    assert ((118, 30, 2), 2) in real
    dead = _caller_mutant([('    if a.plan == "beta":\n        res["plan_beta"], beta_npz', '    if False:\n        res["plan_beta"], beta_npz'),
                           ('    if a.plan == "beta":\n        pb = res["plan_beta"]', '    if False:\n        pb = res["plan_beta"]')], "v8f_mini_eval_mut_dead")
    seen = _scorer_calls_from_run(e2e, tmp_path / "dead", dead, monkeypatch, "d")
    assert ((118, 30, 2), 2) not in seen and not (tmp_path / "dead" / "eval_d_beta.npz").exists()


def test_beta_constant_cost_reproduces_the_unplanned_output_end_to_end(e2e, tmp_path, monkeypatch):
    orig = HJEPAStack.score_candidates

    def const(self, z1_t, blocks, goal, level=2, **kw):
        out = orig(self, z1_t, blocks, goal, level, **kw)
        if isinstance(out, tuple):
            return torch.zeros_like(out[0]), out[1]
        return torch.zeros_like(out)
    monkeypatch.setattr(HJEPAStack, "score_candidates", const)
    assert _run(e2e, "m2_s0", "c", tmp_path, "--plan", "beta") == 0
    z, m2 = np.load(tmp_path / "eval_c_beta.npz"), np.load(tmp_path / "eval_c.npz")
    assert np.array_equal(z["beta__pred"], m2["pred"]) and not z["beta__replaced"].any()
    assert np.array_equal(z["beta__pred"], z["m2__pred"])


def test_beta_a_working_scorer_changes_the_output_end_to_end(e2e, tmp_path):
    assert _run(e2e, "m2_s0", "w", tmp_path, "--plan", "beta") == 0
    z, m2 = np.load(tmp_path / "eval_w_beta.npz"), np.load(tmp_path / "eval_w.npz")
    if z["beta__replaced"].any():
        assert not np.array_equal(z["beta__pred"], m2["pred"])
    keep = ~z["beta__replaced"]
    assert np.array_equal(z["beta__pred"][keep], m2["pred"][keep])                   # un-replaced windows ARE M2's output


def test_beta_refusals(e2e, tmp_path):
    with pytest.raises(SystemExit, match="REFUSED.*world model"):
        _run(e2e, "m1", "x", tmp_path, "--plan", "beta")
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(e2e, "m2_steer", "y", tmp_path, "--plan", "beta", "--kin-mode", "steer")


def test_beta_compare_reads_two_seeds_and_refuses_one_checkpoint_twice(e2e, tmp_path):
    for ck, tag in (("m2_s0", "s0"), ("m2_s1", "s1")):
        assert _run(e2e, ck, tag, tmp_path, "--plan", "beta") == 0
    r = Path(e2e["root"])
    common = ["--eval-grid", str(r / "eval_grid_windows.json"), "--eval-kin", str(r / "eval_kin.npz"), "--eval-join", str(r / "join_eval.npz")]
    out = tmp_path / "verdict.json"
    assert EV.main(["compare-beta", "--runs", f"s0={tmp_path / 'eval_s0.json'}", f"s1={tmp_path / 'eval_s1.json'}", "--out", str(out), *common]) == 0
    v = json.loads(out.read_text(encoding="utf-8"))
    assert v["verdict"] in BE.NEXT_ACTION and v["verdict"] != "UNDECIDABLE" and len(v["runs"]) == 2
    assert v["seed_floor_F_m"] == pytest.approx(abs(v["m2_ade6_means"][0] - v["m2_ade6_means"][1]))
    with pytest.raises(SystemExit, match="same checkpoint"):
        EV.main(["compare-beta", "--runs", f"a={tmp_path / 'eval_s0.json'}", f"b={tmp_path / 'eval_s0.json'}", "--out", str(out), *common])
    with pytest.raises(SystemExit, match="not a `run --plan beta`"):
        _run(e2e, "m2_s0", "plain", tmp_path)
        EV.main(["compare-beta", "--runs", f"a={tmp_path / 'eval_plain.json'}", f"b={tmp_path / 'eval_s1.json'}", "--out", str(out), *common])


# ------------------------------------------------------------------------------------------------ the instrument
def _paths(n=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.uniform(2, 20, (n, 1))
    t = np.arange(1, 31)[None] * 0.2
    yaw = rng.normal(0, 0.15, (n, 1)) * t
    x = np.cumsum(v * np.cos(yaw) * 0.2 * (1 + rng.normal(0, .05, (n, 30))), 1)
    y = np.cumsum(v * np.sin(yaw) * 0.2, 1)
    return np.stack([x, y], -1).astype(np.float32)


def test_beta_family_windows_reproduce_the_family_blocks_printed_numbers():
    pred, gt = _paths(60, 1), _paths(60, 2)
    fw = BE.family_windows(pred, gt, 0.2)
    P, G = torch.as_tensor(pred), torch.as_tensor(gt)
    lon, lat = ff.longitudinal(P, G, 0.2, n_boot=20), ff.lateral(P, G, 0.2, n_boot=20)
    assert fw["lon_speed_mae"].mean() == pytest.approx(lon["speed_mae_mps"], abs=2e-4)
    assert fw["lon_accel_mae"].mean() == pytest.approx(lon["accel_mae_mps2"], abs=2e-4)
    assert fw["lon_along_mae"].mean() == pytest.approx(lon["along_mae_m"], abs=2e-4)
    assert fw["lat_cross_mae"].mean() == pytest.approx(lat["cross_mae_m"], abs=2e-4)
    pooled = math.degrees(fw["lat_heading_sum_rad"].sum() / fw["lat_heading_n"].sum())
    assert pooled == pytest.approx(lat["heading_mae_deg"], abs=2e-4)
    assert math.degrees(fw["lat_yaw_rate_sum_rps"].sum() / fw["lat_pair_n"].sum()) == pytest.approx(lat["yaw_rate_mae_degps"], abs=2e-4)
    assert fw["lat_curvature_sum_1pm"].sum() / fw["lat_pair_n"].sum() == pytest.approx(lat["curvature_mae_1pm"], abs=2e-6)
    assert np.nanmean(fw["lat_yaw_rate_mae_degps"]) > 0 and np.nanmean(fw["lat_curvature_mae_1pm"]) > 0
    tac = ff.tactical_from_trajectory(P, G, 0.2, n_boot=20)
    assert fw["tac_lat_acc"].mean() == pytest.approx(tac["lateral_decision"]["accuracy"], abs=1e-4)
    assert fw["tac_lon_acc"].mean() == pytest.approx(tac["longitudinal_decision"]["accuracy"], abs=1e-4)
    # CONTROL: GT against itself reads exactly zero error and a perfect decision
    z = BE.family_windows(gt, gt, 0.2)
    assert z["lon_speed_mae"].max() == 0 and z["lat_cross_mae"].max() == 0 and z["tac_lat_acc"].min() == 1.0
    assert z["lat_yaw_rate_mae_degps"].max() == 0 and z["lat_curvature_mae_1pm"].max() == 0
    # a path that stands still has NO tangent: its heading / yaw-rate / curvature components are NaN (refused), never 0 (perfect agreement)
    still = BE.family_windows(np.zeros((2, 30, 2)), gt[:2], 0.2)
    assert np.isnan(still["lat_heading_mae_deg"]).all() and np.isnan(still["lat_yaw_rate_mae_degps"]).all()
    assert np.isnan(still["lat_curvature_mae_1pm"]).all() and np.isfinite(still["lat_cross_mae"]).all()


def test_beta_route_adherence_known_values():
    x = np.arange(1, 31) * 2.0                                           # 10 m/s straight: 60 m in 6 s
    plan = np.stack([x, np.zeros(30)], -1)[None].repeat(3, 0)
    rc = np.array([[[30., 0., 0.], [50., 0., 0.], [80., 0., 0.]],
                   [[30., 1.0, 0.], [50., 0., 0.], [80., 0., 0.]],
                   [[30., 0., 0.], [50., 0., 0.], [80., 0., 0.]]])
    ok = np.array([[True, True, True], [True, True, True], [False, True, True]])
    e = BE.route_adherence(plan, rc, ok)
    assert e[0, 0] == pytest.approx(0.0, abs=1e-9) and e[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(e[0, 2]), "the plan is 60 m long: it never reaches 80 m of arc"
    assert e[1, 0] == pytest.approx(1.0) and np.isnan(e[2, 0]) and e[2, 1] == pytest.approx(0.0, abs=1e-9)


def test_beta_paired_row_direction_and_masking():
    rng = np.random.default_rng(0)
    eid = np.repeat(np.arange(20), 10)
    a = rng.normal(1.0, 0.3, 200)
    b = a + 0.5 + rng.normal(0, 0.05, 200)                    # b is 0.5 higher in EVERY episode
    r = BE.paired_row(a, b, eid, 200)
    assert r["separated"] and r["mean"] < 0 and r["better"] and not r["worse"]
    r2 = BE.paired_row(a, b, eid, 200, lower_is_better=False)
    assert r2["worse"] and not r2["better"]
    a2 = a.copy()
    a2[:50] = np.nan
    assert BE.paired_row(a2, b, eid, 200)["n_windows"] == 150
    assert BE.paired_row(a[:1], b[:1], eid[:1], 10)["status"] == "UNAVAILABLE"


def test_beta_signal_recovery_known_values():
    assert BE.signal_recovery(10.0, 2.0, 2.0) == 1.0 and BE.signal_recovery(10.0, 10.0, 2.0) == 0.0
    assert BE.signal_recovery(10.0, 14.0, 2.0) < 0 and math.isnan(BE.signal_recovery(5.0, 5.0, 5.0))


def _row(mean, better=None, worse=None, sep=True, lib=True):
    neg = mean < 0 if lib else mean > 0                      # a delta toward "better" is negative for an error, positive for an accuracy
    return {"mean": mean, "separated": sep, "lower_is_better": lib, "better": (neg and sep) if better is None else better,
            "worse": (not neg and mean != 0 and sep) if worse is None else worse}


def _seed(d6=-0.8, shuf=+0.5, repl=0.4, rec=0.8, orc=-3.0, fam=None):
    return {"ade6_beta_minus_m2": _row(d6), "ade6_shuf_minus_beta": _row(shuf), "replaced_frac": repl,
            "signal": {"recovery": rec, "oracle_minus_random": _row(orc)},
            "families": fam or {c: _row(-0.1 if lib else 0.1, lib=lib) for c, (_, lib) in BE.COMPONENTS.items()}}


def test_beta_decision_table_reaches_every_verdict_in_the_committed_order():
    s = lambda **k: [_seed(**k), _seed(**k)]
    m2 = [2.50, 2.52]                                                      # F = 0.02, 2F = 0.04
    assert BE.decide_beta(s(), m2)["verdict"] == "PASS"
    assert BE.decide_beta(s(repl=0.01), m2)["verdict"] == "FAIL-INERT"
    assert BE.decide_beta(s(rec=0.2), m2)["verdict"] == "FAIL-RANKING-DEAD"                          # the oracle goal cannot rank the fan
    assert BE.decide_beta(s(orc=+1.0), m2)["verdict"] == "FAIL-RANKING-DEAD"
    assert BE.decide_beta(s(d6=+0.3), m2)["verdict"] == "FAIL-GOAL"                                  # ranks, but S's route is not a goal
    assert BE.decide_beta(s(d6=-0.03), m2)["verdict"] == "FAIL-GOAL"                                 # separated but inside 2F
    assert BE.decide_beta(s(shuf=-0.2), m2)["verdict"] == "FAIL-CONTROL"                             # the shuffled goal does NOT lose
    bad = {c: _row(-0.1 if lib else 0.1, lib=lib) for c, (_, lib) in BE.COMPONENTS.items()}
    bad["lat_cross_mae"] = _row(+0.2)
    r = BE.decide_beta(s(fam=bad), m2)
    assert r["verdict"] == "FAIL-NR" and r["families_separated_worse"] == ["lat_cross_mae"]
    # a higher-is-better component is worse when its delta is NEGATIVE
    bad2 = dict(bad, lat_cross_mae=_row(-0.1), tac_lat_acc=_row(-0.1, lib=False))
    assert BE.decide_beta(s(fam=bad2), m2)["families_separated_worse"] == ["tac_lat_acc"]
    # one seed good, one bad: not a pass
    assert BE.decide_beta([_seed(), _seed(d6=+0.1)], m2)["verdict"] == "FAIL-GOAL"
    assert BE.decide_beta([_seed()], m2[:1])["verdict"] == "UNDECIDABLE"
    for v in BE.NEXT_ACTION.values():
        assert isinstance(v, str) and len(v) > 40
