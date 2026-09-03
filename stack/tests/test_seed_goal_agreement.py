"""L0 / BACKLOG R38 — THE SEED IS NOT THE GOAL, and this pins both states.

⛔ THE DEFECT, MEASURED (`TanitAD Research Lab/Architecture & Inference/Research/
2026-09-03-tactical-decoder/raw/seed_goal_mismatch.json`): **62 of the 64
(lat, lon) token pairs** give a DIFFERENT tactical action sequence to the GOAL
and to the SEED that chases it — under BOTH `COST_TIME_GRIDS` — with `TURN_L`
over-rotating its own goal by **82.506 deg**. The only agreeing pairs are
`(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)`: the all-zero control.

⭐ THE CAUSE IS THE TRUNCATION, NOT THE REGRID:

    refa_v1.py `_imagine_tactical_goal`   acts = _model_actions(ctrl, ...)
                                                 [:, ::stride][:, :tac_steps]
                                          -> operative indices [0,3,...,27]
    refa_v1.py (the seed)                 seed = controls[:cfg.plan_steps]
                                          -> operative indices [0..9] ONLY

The goal's actions at operative indices 12, 15, 18, 21, 24 and 27 **are not in
the seed at all**, so no `cost_time_grid` can address them and the `4139203`
regrid does not close it: `"dense"` reads `[0..9]`, `"tactical"` reads
`[0,3,6,9,9,9,9,9,9,9]`.

⭐ WHAT IS PINNED HERE
  * the DEFAULT (`goal_time_grid="full"`) still reads **2/64** agreeing pairs
    and **82.506 deg** — this is the DELIBERATE-REGRESSION half: a gate that
    cannot still see the defect proves nothing about the fix;
  * the REPAIR (`goal_time_grid="plan"`) reads **64/64** and **exactly 0.0 deg**;
  * and both are checked on the REAL `plan()` code path, over all 64 token
    pairs, by capturing the action feed the goal rollout actually receives and
    comparing it element-for-element with the feed the cost gives the seed —
    which is `seed_goal_mismatch.py`'s own metric, applied to running code.

⚠️ WHAT THE REPAIR DOES **NOT** BUY, asserted so no reader can overclaim: under
`"plan"` the goal is the manoeuvre's first `plan_horizon_s`, NOT the 6 s
manoeuvre the token names. `test_h_*` pins that the two goals genuinely differ,
i.e. the repair buys IDENTITY and not HORIZON.

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import (COST_TIME_GRIDS, GOAL_TIME_GRIDS, RefAV1,
                                  RefAV1Config, canonical_controls)
from tanitad.refs.refa_v1_plan import PlanConfig

#: the BANKED numbers this test exists to hold, with their source
BANKED_N_AGREE_DEFAULT = 2            #: raw/seed_goal_mismatch.json by_grid.*
BANKED_AGREEING_PAIRS = {("LANE_KEEP", "CRUISE"), ("ABORT_LC", "CRUISE")}
BANKED_WORST_EXCESS_DEG = 82.506      #: TURN_L, both grids, v0 = 10.0
V0_METRIC = 10.0                      #: seed_goal_mismatch.py's own --v0 default

#: float32 cosine resolution near 1: `1 - cos(x, x)` cannot be smaller than
#: this, and a difference below it is not an effect. spacing(1f) = 1.1920929e-07.
COS32_TOL = 4.0 * 1.1920928955078125e-07


# --------------------------------------------------------------------------- #
#  the metric, verbatim in structure from tools/seed_goal_mismatch.py           #
# --------------------------------------------------------------------------- #
def _goal_op_indices(cfg: RefAV1Config) -> list[int]:
    """`_imagine_tactical_goal`: `[:, ::stride][:, :tac_steps]`."""
    stride = int(round(cfg.tac_dt / cfg.op_dt))
    return list(range(0, cfg.op_steps, stride))[:cfg.tac_steps]


def _seed_op_indices(cfg: RefAV1Config, cost_time_grid: str) -> list[int]:
    """The operative action each TACTICAL step of a candidate consumes —
    `refa_v1.py` `tac_idx`, gated on `cost_time_grid`."""
    stride = int(round(cfg.tac_dt / cfg.op_dt))
    h = cfg.plan_steps
    if cost_time_grid == "tactical":
        return [min(j * stride, h - 1) for j in range(cfg.tac_steps)]
    if cost_time_grid == "dense":
        return [min(j, h - 1) for j in range(cfg.tac_steps)]
    raise AssertionError(f"unhandled cost_time_grid {cost_time_grid!r} — this "
                         f"test must be extended before it can report on it")


def _indices(cfg: RefAV1Config, goal_time_grid: str,
             cost_time_grid: str) -> tuple[list[int], list[int]]:
    """(goal indices, seed indices). Under `"plan"` the goal is re-rolled from
    the seed's own feed, so the two index sets are the same set by
    construction — which is exactly the claim under test, and it is checked
    against RUNNING CODE in `test_c_*` rather than trusted here."""
    seed = _seed_op_indices(cfg, cost_time_grid)
    goal = seed if goal_time_grid == "plan" else _goal_op_indices(cfg)
    return goal, seed


def _pair_table(cfg: RefAV1Config, goal_time_grid: str, cost_time_grid: str,
                v0: float = V0_METRIC) -> dict:
    """seed_goal_mismatch.py's own table over the whole vocabulary."""
    g_idx, s_idx = _indices(cfg, goal_time_grid, cost_time_grid)
    agree, worst = [], ("", "", 0.0)
    n_mismatch = n_kappa = 0
    excess = {}
    for lat in tactical_lat_actions(cfg.tac_vocab_version):
        for lon in tactical_lon_actions_v(cfg.tac_vocab_version):
            ctrl = canonical_controls(lat, lon, v0, cfg.op_steps, cfg.op_dt)
            k, acc = ctrl[:, 1].tolist(), ctrl[:, 0].tolist()
            k_g, k_s = [k[i] for i in g_idx], [k[i] for i in s_idx]
            a_g, a_s = [acc[i] for i in g_idx], [acc[i] for i in s_idx]
            mism = (k_g != k_s) or (a_g != a_s)
            n_mismatch += int(mism)
            n_kappa += int(k_g != k_s)
            if not mism:
                agree.append((lat, lon))
            psi_g = sum(x * v0 * cfg.tac_dt for x in k_g)
            psi_s = sum(x * v0 * cfg.tac_dt for x in k_s)
            exc = math.degrees(psi_s - psi_g)
            excess[(lat, lon)] = exc
            if abs(exc) > abs(worst[2]):
                worst = (lat, lon, exc)
    n = len(tactical_lat_actions(cfg.tac_vocab_version)) * \
        len(tactical_lon_actions_v(cfg.tac_vocab_version))
    return {"n_pairs": n, "n_mismatch": n_mismatch, "n_agree": n - n_mismatch,
            "n_kappa_mismatch": n_kappa, "agreeing": set(agree),
            "worst": worst, "excess": excess,
            "goal_op_indices": g_idx, "seed_op_indices": s_idx}


# --------------------------------------------------------------------------- #
#  the tiny rig — the fixture of tests/test_refa_v1_plan_goal.py                #
# --------------------------------------------------------------------------- #
def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
                op_layers=1, op_heads=2, op_window=2, tac_layers=1,
                tac_queries=4, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_ctx=16,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                  n_heads=2, d_intent=16))
    base.update(kw)
    return RefAV1Config(**base)


def _model(seed: int = 0, **kw) -> RefAV1:
    torch.manual_seed(seed)
    m = RefAV1(_cfg(**kw)).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    return m


def _pc(c: RefAV1Config) -> PlanConfig:
    """Deliberately tiny: this test is about the GOAL, not about the search."""
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=4, n_iters=1, n_elites=2)


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


class _OneHotHead(torch.nn.Module):
    """A head whose ARGMAX is a fixed class — the only thing
    `_imagine_tactical_goal` reads off `lat_head` / `lon_head`."""

    def __init__(self, i: int, n: int):
        super().__init__()
        self.i, self.n = int(i), int(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.shape[0], self.n), -10.0, dtype=x.dtype,
                         device=x.device)
        out[:, self.i] = 10.0
        return out


def _force_token(m: RefAV1, lat: str, lon: str) -> None:
    """Pin the decoded (lat, lon) so every token pair can be exercised on a
    random-init head."""
    vv = m.cfg.tac_vocab_version
    m.lat_head = _OneHotHead(list(tactical_lat_actions(vv)).index(lat), m.n_lat)
    m.lon_head = _OneHotHead(list(tactical_lon_actions_v(vv)).index(lon),
                             m.n_lon)


def _plan_capturing_goal(m: RefAV1, feats, v0, nav, *, goal_time_grid,
                         cost_time_grid):
    """Run a REAL `plan()` and capture every tactical rollout it performs.

    The goal rollout is the LAST tactical rollout before the search begins:
    under `"full"` that is `_imagine_tactical_goal`'s, under `"plan"` it is the
    repair's re-roll (which deliberately runs after it). We detect the search
    by its batch size — a candidate chunk carries `n_samples` rows, the goal
    carries exactly 1."""
    seen = []
    orig = m.tactical.rollout

    def spy(field, actions, intent=None, last_only=False):
        out = orig(field, actions, intent=intent, last_only=last_only)
        seen.append({"acts": actions.detach().clone(),
                     "z": out.detach().clone() if last_only else None,
                     "n": int(actions.shape[0])})
        return out

    m.tactical.rollout = spy
    try:
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
                     cost_time_grid=cost_time_grid,
                     goal_time_grid=goal_time_grid)
    finally:
        m.tactical.rollout = orig
    pre = [r for r in seen if r["n"] == 1]
    assert pre, "no single-row tactical rollout: the goal was never built"
    return res, pre[-1], seen


def _z0_and_intent(m: RefAV1, feats, nav):
    """`plan()`'s own ``search_z`` / ``intent`` — the two things a candidate
    rollout is given besides its actions. Rebuilding a rollout WITHOUT the
    intent silently compares two different conditionings and reads as a
    ~1e-06 identity failure; that is a test bug, not a code one."""
    with torch.no_grad():
        field = m.encode(feats)
        last = m._last_state(field)
        brains = m._run_brains(field.mean(dim=-2), nav)
    intent = None if brains is None else brains["intent"]
    return last, intent


def _seed_feed(m: RefAV1, ctrl: torch.Tensor, v0: float, *, cost_time_grid,
               units: str = "kappa") -> torch.Tensor:
    """The action feed `_cost_chunk` gives the canonical SEED — the four lines
    of `refa_v1.py` `_cost_chunk` that build `acts`, and nothing else."""
    cfg = m.cfg
    seed = ctrl[:cfg.plan_steps][None]
    acts = m._model_actions(seed, torch.tensor([v0], dtype=torch.float32), units)
    if cost_time_grid == "tactical":
        stride = int(round(cfg.tac_dt / cfg.op_dt))
        idx = torch.tensor([min(j * stride, cfg.plan_steps - 1)
                            for j in range(cfg.tac_steps)], dtype=torch.long)
        acts = acts.index_select(1, idx)
    return acts


# =========================================================================== #
#  (a) THE DELIBERATE REGRESSION — the default must STILL read the defect      #
# =========================================================================== #
@pytest.mark.parametrize("grid", COST_TIME_GRIDS)
def test_a_the_default_still_reproduces_the_banked_62_of_64_mismatch(grid):
    cfg = _cfg()
    t = _pair_table(cfg, "full", grid)
    assert t["n_pairs"] == 64, t["n_pairs"]
    assert t["n_agree"] == BANKED_N_AGREE_DEFAULT, t["n_agree"]
    assert t["n_mismatch"] == 62, t["n_mismatch"]
    assert t["n_kappa_mismatch"] == 48, t["n_kappa_mismatch"]
    assert t["agreeing"] == BANKED_AGREEING_PAIRS, t["agreeing"]
    assert t["worst"][0] == "TURN_L", t["worst"]
    assert abs(abs(t["worst"][2]) - BANKED_WORST_EXCESS_DEG) < 0.01, t["worst"]
    # and the index sets are the ones the diagnosis names, from the shipped cfg
    assert t["goal_op_indices"] == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert t["seed_op_indices"] == (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if grid == "dense"
        else [0, 3, 6, 9, 9, 9, 9, 9, 9, 9])


# =========================================================================== #
#  (b) THE FIX — 64/64 and exactly 0.0 deg, both grids                         #
# =========================================================================== #
@pytest.mark.parametrize("grid", COST_TIME_GRIDS)
def test_b_the_repair_agrees_on_all_64_token_pairs_and_rotates_by_zero(grid):
    cfg = _cfg()
    t = _pair_table(cfg, "plan", grid)
    assert t["n_agree"] == 64 and t["n_mismatch"] == 0, t["n_mismatch"]
    assert t["n_kappa_mismatch"] == 0
    # ⛔ EXACTLY 0.0 — this is an IDENTITY, not an approximation: both feeds
    # are the same `list` of floats, so the tolerance is 0 and not "small".
    for pair, exc in t["excess"].items():
        assert exc == 0.0, (pair, exc)
    for lat in ("TURN_L", "TURN_R"):
        assert t["excess"][(lat, "CRUISE")] == 0.0
        assert t["excess"][(lat, "FOLLOW")] == 0.0


# =========================================================================== #
#  (c) THE SAME METRIC ON RUNNING CODE, over all 64 pairs                      #
# =========================================================================== #
@pytest.mark.parametrize("grid", COST_TIME_GRIDS)
def test_c_running_plan_feeds_the_goal_the_seeds_own_actions_on_64_of_64(grid):
    """The load-bearing one: capture the feed the GOAL rollout really gets
    inside `plan()`, and compare it element-for-element with the feed
    `_cost_chunk` gives the canonical seed."""
    m = _model(0)
    c = m.cfg
    feats, v0, nav = _window(c)
    n_agree_fix = n_agree_default = 0
    lats = list(tactical_lat_actions(c.tac_vocab_version))
    lons = list(tactical_lon_actions_v(c.tac_vocab_version))
    for lat in lats:
        for lon in lons:
            _force_token(m, lat, lon)
            for gtg, bucket in (("plan", "fix"), ("full", "default")):
                res, goal, _ = _plan_capturing_goal(
                    m, feats, v0, nav, goal_time_grid=gtg, cost_time_grid=grid)
                assert res.goal_time_grid == gtg           # provenance stamped
                assert res.goal_action["lat"] == lat
                assert res.goal_action["lon"] == lon
                want = _seed_feed(m, res.goal_action["controls"], v0,
                                  cost_time_grid=grid)
                same = (goal["acts"].shape == want.shape
                        and torch.equal(goal["acts"], want))
                if bucket == "fix":
                    n_agree_fix += int(same)
                else:
                    n_agree_default += int(same)
    assert n_agree_fix == len(lats) * len(lons) == 64, n_agree_fix
    # the deliberate regression: the default must STILL fail on 62 of them
    assert n_agree_default == BANKED_N_AGREE_DEFAULT, n_agree_default


# =========================================================================== #
#  (d) the numerical consequence — 1 - cos == 0 for the canonical seed         #
# =========================================================================== #
@pytest.mark.parametrize("grid", COST_TIME_GRIDS)
@pytest.mark.parametrize("lat", ["TURN_L", "TURN_R", "NUDGE_R",
                                 "LANE_CHANGE_R"])
def test_d_the_canonical_seed_pays_a_zero_goal_term_under_the_repair(lat, grid):
    m = _model(0)
    c = m.cfg
    feats, v0, nav = _window(c)
    _force_token(m, lat, "CRUISE")
    _, goal, _ = _plan_capturing_goal(m, feats, v0, nav,
                                      goal_time_grid="plan",
                                      cost_time_grid=grid)
    # roll the seed exactly as `_cost_chunk` does, against the captured goal
    last, intent = _z0_and_intent(m, feats, nav)
    with torch.no_grad():
        zk = m.tactical.rollout(m._tac_field(last), goal["acts"],
                                intent=intent, last_only=True)
        gt = float(1.0 - F.cosine_similarity(zk.flatten(1),
                                             goal["z"].flatten(1), dim=-1))
    assert abs(gt) <= COS32_TOL, (lat, grid, gt)


# =========================================================================== #
#  (e) the default path is BYTE-IDENTICAL — the flag is really default-OFF     #
# =========================================================================== #
def test_e_the_default_flag_changes_nothing():
    m = _model(0)
    c = m.cfg
    feats, v0, nav = _window(c)
    a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c),
               goal_time_grid="full")
    assert torch.equal(a.controls, b.controls)
    assert a.cost == b.cost and a.source == b.source
    assert a.baseline_costs == b.baseline_costs
    assert a.goal_source == b.goal_source == "tactical_imagined"
    assert a.goal_time_grid == "full"


# =========================================================================== #
#  (f) the flag is validated, and its vocabulary is the shipped one            #
# =========================================================================== #
def test_f_an_unknown_goal_time_grid_is_refused_by_name():
    assert GOAL_TIME_GRIDS == ("full", "plan")
    m = _model(0)
    feats, v0, nav = _window(m.cfg)
    with pytest.raises(ValueError, match="goal_time_grid must be one of"):
        m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
               goal_time_grid="tactical")


# =========================================================================== #
#  (g) the repair holds at plan_level="operative" too                          #
# =========================================================================== #
def test_g_the_repair_holds_when_the_search_rolls_the_operative_predictor():
    m = _model(0, plan_level="operative")
    c = m.cfg
    feats, v0, nav = _window(c)
    _force_token(m, "TURN_L", "CRUISE")
    seen = []
    orig_op, orig_tac = m.operative.rollout, m.tactical.rollout

    def spy_op(field, actions, intent=None, last_only=False):
        out = orig_op(field, actions, intent=intent, last_only=last_only)
        if int(actions.shape[0]) == 1 and last_only:
            seen.append((actions.detach().clone(), out.detach().clone()))
        return out

    m.operative.rollout = spy_op
    try:
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c),
                     goal_time_grid="plan")
    finally:
        m.operative.rollout = orig_op
        m.tactical.rollout = orig_tac
    assert res.goal_time_grid == "plan"
    assert seen, "the goal was not rolled by the OPERATIVE predictor"
    acts, z = seen[-1]
    # the goal's feed is the seed's feed, with NO tactical regrid (the regrid
    # applies only to `self.tactical` rollouts — `refa_v1.py` `_cost_chunk`)
    want = m._model_actions(res.goal_action["controls"][:c.plan_steps][None],
                            torch.tensor([v0], dtype=torch.float32), "kappa")
    assert torch.equal(acts, want)
    last, intent = _z0_and_intent(m, feats, nav)
    with torch.no_grad():
        zk = orig_op(last, want, intent=intent, last_only=True)
        gt = float(1.0 - F.cosine_similarity(
            m._tac_field(zk).flatten(1), m._tac_field(z).flatten(1), dim=-1))
    assert abs(gt) <= COS32_TOL, gt


# =========================================================================== #
#  (h) IDENTITY, NOT HORIZON — the honest limit, asserted so nobody overclaims #
# =========================================================================== #
def test_h_the_repair_buys_identity_not_horizon():
    """Under `"plan"` the goal is the manoeuvre's first `plan_horizon_s`, NOT
    the 6 s manoeuvre the token names. If these two goals were ever equal the
    repair would be free — they are not, and the register row must say so."""
    m = _model(0)
    c = m.cfg
    assert c.plan_steps == 10 and c.op_steps == 30      # 2.0 s of a 6.0 s goal
    feats, v0, nav = _window(c)
    _force_token(m, "TURN_L", "CRUISE")
    _, g_full, _ = _plan_capturing_goal(m, feats, v0, nav,
                                        goal_time_grid="full",
                                        cost_time_grid="dense")
    _, g_plan, _ = _plan_capturing_goal(m, feats, v0, nav,
                                        goal_time_grid="plan",
                                        cost_time_grid="dense")
    assert not torch.equal(g_full["acts"], g_plan["acts"])
    assert not torch.equal(g_full["z"], g_plan["z"])
    # and the arithmetic reason, in one line: the 6 s goal integrates 7 turn
    # steps, the 2 s plan window integrates 10 — the 82.5 deg excess.
    t = _pair_table(c, "full", "dense")
    assert t["excess"][("TURN_L", "CRUISE")] > 80.0
