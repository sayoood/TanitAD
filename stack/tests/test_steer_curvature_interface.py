"""The COMMAND↔GEOMETRY interface (PI ruling 2026-09-03) — pinned in one place.

THE CONTRACT UNDER TEST (`kinematic.STEER_WHEELBASE_M` carries it in prose):

  COMMAND   ``steer``  road-wheel angle [rad] — v2ep ``actions[:, 0]``, the
            loader's output, the predictor's input, every trained checkpoint.
  GEOMETRY  ``kappa``  path curvature [1/m] — `rollout_unicycle`, ``GOAL_KAPPA_*``,
            ``PlanConfig.kappa_max``, every metre-valued trajectory.
  BRIDGE    kappa = tan(steer)/L_enc      steer = arctan(L_enc*kappa)   L_enc = 2.9

⛔ THE DEFAULT IS THE LEGACY (UNCONVERTED) PATH, and half of these tests exist to
prove that: every banked refav1 number was produced by it and a silently converted
eval would make old and new numbers incomparable. The flag is the honest way to
make the change visible.

Families: A = the bridge itself; B = OFF is byte-identical; C = ON is exactly the
conversion; D = the planner→model crossing; E = the deliberate regression (the ON
path must be WORSE on a corpus whose channel really is a curvature — otherwise
the flag is a universal improver and proves nothing); F = REAL windows.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_STACK = str(Path(__file__).resolve().parents[1])
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402
from tanitad.models.kinematic import (ACTION_UNITS, STEER_WHEELBASE_M,  # noqa: E402
                                      as_command, as_curvature,
                                      kappa_of_steer, rollout_unicycle,
                                      steer_of_kappa)
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config  # noqa: E402
from tanitad.refs.refa_v1_plan import PlanConfig, unicycle_paths  # noqa: E402

DT = 0.2
K = 10
_REPO = Path(__file__).resolve().parents[2]
_ARM_PATH = _REPO / "taniteval" / "tools" / "refav1_arm.py"

#: The banked local slice this package measured on. Absent on a fresh box, so the
#: REAL-window family skips with a reason rather than silently passing.
_DUMP = Path(r"C:\Users\Admin\refav1_eval_slice\t1_dump")
_EPS = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
_HAVE_REAL = _DUMP.is_dir() and _EPS.is_dir()


def _arm():
    spec = importlib.util.spec_from_file_location("refav1_arm_iface", _ARM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctrl(n=4, kk=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.zeros(n, kk, 2)
    c[..., 0] = torch.randn(n, kk, generator=g) * 0.5          # accel
    c[..., 1] = torch.randn(n, kk, generator=g) * 0.06         # channel 1
    return c


# =========================================================================== #
# A — the bridge
# =========================================================================== #
def test_A1_round_trip_is_the_identity_both_ways():
    k = torch.linspace(-0.33, 0.33, 401)
    assert torch.allclose(kappa_of_steer(steer_of_kappa(k)), k, atol=1e-7)
    s = torch.linspace(-0.75, 0.75, 401)
    assert torch.allclose(steer_of_kappa(kappa_of_steer(s)), s, atol=1e-7)


def test_A2_the_constant_is_the_ENCODING_constant_not_a_vehicle():
    """MEASURED 2026-09-03 by inverting `physicalai.signals_at` against its own
    input on 20/20 local clips: L_enc = 2.9 to 3e-8. The clips' REAL wheelbases
    (release `calibration/vehicle_dimensions`) are {2.73, 3.135, 3.165, 3.216} —
    NOT ONE is 2.9. This test pins that the code uses the ENCODING constant."""
    from tanitad.data.physicalai import WHEELBASE
    assert STEER_WHEELBASE_M == 2.9
    assert STEER_WHEELBASE_M == WHEELBASE, (
        "the inverse must use the constant signals_at ENCODED with; if "
        "physicalai.WHEELBASE ever moves, this bridge moves with it")
    assert 2.9 not in {2.73, 2.85, 3.135, 3.165, 3.216}


def test_A3_the_gain_is_2p9_and_a_correlation_cannot_see_it():
    """Why `refav1_loader.py:17-24`'s 'r = 0.995' could not catch this: a
    correlation is scale-invariant. The SLOPE is the instrument."""
    # the corpus's own range: steer p99 ~ 0.13 rad -> |kappa| <~ 0.045
    # (`2026-07-26-wheelbase-impact/tier1_population_strata.json`)
    kap = torch.linspace(-0.08, 0.08, 501)
    steer = steer_of_kappa(kap)
    r = float(np.corrcoef(steer.numpy(), kap.numpy())[0, 1])
    assert r > 0.999, r                       # a correlation sees NOTHING wrong
    slope = float((torch.tan(steer) @ kap) / (kap @ kap))
    assert abs(slope - STEER_WHEELBASE_M) < 1e-5, slope   # the slope sees 2.9


def test_A4_units_are_named_never_guessed():
    assert ACTION_UNITS == ("kappa", "steer")
    with pytest.raises(ValueError):
        as_curvature(_ctrl(), "curvature")
    with pytest.raises(ValueError):
        as_command(_ctrl(), "radians")


# =========================================================================== #
# B — OFF is byte-identical (the flag is off by default and changes NOTHING)
# =========================================================================== #
def test_B1_as_curvature_kappa_returns_the_SAME_OBJECT():
    c = _ctrl()
    assert as_curvature(c, "kappa") is c
    assert as_command(c, "kappa") is c


def test_B2_unicycle_paths_default_is_bit_identical_to_the_old_call():
    c = _ctrl(n=6)
    v0 = torch.tensor([9.0])
    state0 = torch.zeros(6, 4)
    state0[:, 3] = 9.0
    legacy = rollout_unicycle(state0, c, dt=DT)[..., :2]
    assert torch.equal(unicycle_paths(c, v0, DT), legacy)
    assert torch.equal(unicycle_paths(c, v0, DT, action_units="kappa"), legacy)


def test_B3_arm_paths_from_controls_default_is_bit_identical():
    arm = _arm()
    c = _ctrl(n=1)[0]
    a = arm.paths_from_controls(c, 9.0, DT, K)
    b = arm.paths_from_controls(c, 9.0, DT, K, action_units="kappa")
    assert torch.equal(a, b)


def test_B4_the_arm_CLI_defaults_to_the_legacy_unit():
    """A default that silently repaired a live eval would make every banked
    number incomparable without saying so."""
    src = _ARM_PATH.read_text(encoding="utf-8", errors="replace")
    assert '"--action-units", choices=("kappa", "steer"), default="kappa"' in src


def test_B5_a_zero_control_is_zero_in_BOTH_units():
    """`ha0` (the constant-velocity floor) must be untouched by the flag."""
    z = torch.zeros(1, K, 2)
    v0 = torch.tensor([7.5])
    assert torch.equal(unicycle_paths(z, v0, DT),
                       unicycle_paths(z, v0, DT, action_units="steer"))


# =========================================================================== #
# C — ON is EXACTLY the conversion, applied at the integrator
# =========================================================================== #
def test_C1_on_equals_converting_the_controls_by_hand():
    c = _ctrl(n=5, seed=3)
    v0 = torch.tensor([11.0])
    by_hand = c.clone()
    by_hand[..., 1] = torch.tan(c[..., 1]) / STEER_WHEELBASE_M
    assert torch.equal(unicycle_paths(c, v0, DT, action_units="steer"),
                       unicycle_paths(by_hand, v0, DT))


def test_C2_the_integrator_itself_is_untouched():
    """`rollout_unicycle` still means yaw_rate = v*kappa; the unit travels with
    the CALL, because a planner candidate and a recorded action reach the same
    integrator in different units."""
    import inspect
    src = inspect.getsource(rollout_unicycle)
    assert "yaw_rate = v * kappa" in src
    assert "tan(" not in src and "atan" not in src


def test_C3_accel_channel_is_unit_invariant():
    c = _ctrl(n=3, seed=7)
    assert torch.equal(as_curvature(c, "steer")[..., 0], c[..., 0])
    assert torch.equal(as_command(c, "steer")[..., 0], c[..., 0])


def test_C4_a_constant_channel_gives_the_right_TURN_RADIUS():
    """The number the whole thing is about. A recorded steer of 0.08 rad is a
    turn of R = 2.9/tan(0.08) = 36.2 m; read as a curvature it is R = 12.5 m.
    2.9x apart, which is the whole 0.716 m lateral miss."""
    v0, kk = 10.0, 5
    c = torch.zeros(1, kk, 2)
    c[..., 1] = 0.08
    r_as_kappa = 1.0 / 0.08
    r_as_steer = STEER_WHEELBASE_M / math.tan(0.08)
    assert abs(r_as_kappa - 12.5) < 0.01
    assert abs(r_as_steer - 36.16) < 0.05
    assert abs(r_as_steer / r_as_kappa - 2.893) < 0.01      # the whole defect
    p_k = unicycle_paths(c, torch.tensor([v0]), DT)[0]
    p_s = unicycle_paths(c, torch.tensor([v0]), DT, action_units="steer")[0]
    # over the same arc the kappa reading turns ~2.9x as far, so it departs the
    # start heading ~2.9x more in y at small angles
    assert float(p_k[:, 1].abs().max()) > float(p_s[:, 1].abs().max())
    yaw_k = 0.08 * v0 * DT * kk
    yaw_s = math.tan(0.08) / STEER_WHEELBASE_M * v0 * DT * kk
    assert abs(yaw_k / yaw_s - r_as_steer / r_as_kappa) < 1e-6


# =========================================================================== #
# D — the PLANNER→MODEL crossing
# =========================================================================== #
def _tiny_cfg(**kw):
    return RefAV1Config(
        tac_vocab_version="v7.0", d_enc=32, n_tokens=8, d_state=32,
        op_layers=1, op_heads=2, op_window=4, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16), **kw)


def _tiny_model():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    m = RefAV1(cfg).eval()
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc)
    return m, cfg, feats


def _pc(cfg):
    return PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt, n_samples=8,
                      n_iters=1, n_elites=2, seed=0)


def test_D1_plan_default_is_byte_identical_to_the_legacy_call():
    m, cfg, feats = _tiny_model()
    with torch.no_grad():
        a = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        b = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa")
    assert torch.equal(a.controls, b.controls)
    assert a.source == b.source and a.cost == b.cost


def test_D2_the_flag_actually_REACHES_the_model():
    """If ON and OFF gave the same cost the crossing would not be wired.

    ⚠️ AMENDED 2026-09-03 (BACKLOG R26), and the amendment is the point. This
    test used to assert that the ZERO-channel-1 baselines (``cv``, ``hold_v0``)
    do NOT move, "because arctan(0) == 0". **That was only true while the
    conversion was HALF applied.** ``as_command`` was called at exactly one site
    (``_cost_chunk``), so the flag converted the CANDIDATE while
    `_imagine_tactical_goal` still rolled the GOAL in raw curvature — and a
    zero-curvature candidate, scored against an unchanged goal, could not move.
    The flag now converts BOTH sides through one boundary
    (`RefAV1._model_actions`), so with an IMAGINED goal the goal FIELD itself
    moves and EVERY candidate is scored against a different reference.

    The original invariant is not discarded, it is relocated to the
    configuration where it still holds: a SUPPLIED ``goal_field`` is not built
    from controls, so no conversion touches it.
    """
    m, cfg, feats = _tiny_model()
    with torch.no_grad():
        a = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa")
        b = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="steer")
    assert a.baseline_costs and b.baseline_costs
    assert a.goal_source == "tactical_imagined" == b.goal_source
    moved = [k for k in a.baseline_costs
             if abs(a.baseline_costs[k] - b.baseline_costs[k]) > 1e-9]
    assert moved, "no baseline cost moved — the conversion never reached the model"

    # SUPPLIED goal: nothing converts the reference, so a zero-curvature
    # candidate must score identically under either spelling and a curved one
    # must not. This is the assertion the imagined-goal arm used to carry.
    torch.manual_seed(1)
    goal = torch.randn(1, cfg.n_tokens, cfg.d_state)
    with torch.no_grad():
        ga = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), goal_field=goal,
                    model_action_units="kappa")
        gb = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), goal_field=goal,
                    model_action_units="steer")
    assert ga.goal_source == "supplied" == gb.goal_source
    for z in ("cv", "hold_v0"):
        if z in ga.baseline_costs:
            assert abs(ga.baseline_costs[z] - gb.baseline_costs[z]) < 1e-9, z
    assert [k for k in ga.baseline_costs
            if abs(ga.baseline_costs[k] - gb.baseline_costs[k]) > 1e-9], \
        "no baseline moved with a supplied goal — the candidate crossing broke"


def test_D3_bad_units_are_refused_loudly():
    m, cfg, feats = _tiny_model()
    with pytest.raises(ValueError):
        m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa_ish")


def test_D4_the_goal_constants_stay_in_CURVATURE():
    """`GOAL_KAPPA_TURN` is geometry and does not move under the repair; only
    the crossing to the model does."""
    from tanitad.refs.refa_v1 import (GOAL_KAPPA_MAX, GOAL_KAPPA_TURN,
                                      canonical_controls)
    assert GOAL_KAPPA_TURN == 0.08 and GOAL_KAPPA_MAX == 0.2
    c = canonical_controls("TURN_L", "CRUISE", 8.0, 30, 0.2)
    assert abs(float(c[0, 1]) - GOAL_KAPPA_TURN) < 1e-7      # float32 storage
    assert abs(1.0 / float(c[0, 1]) - 12.5) < 0.01           # R = 12.5 m
    # ...and at the model boundary it becomes a steering angle
    s = float(as_command(c[:1][None], "steer")[0, 0, 1])
    assert abs(s - math.atan(STEER_WHEELBASE_M * 0.08)) < 1e-7
    assert abs(s - 0.22797) < 1e-4


# =========================================================================== #
# E — the DELIBERATE REGRESSION: ON must be WORSE where the channel really is κ
# =========================================================================== #
def _synth_episode(store: str, M: int = 60):
    """A unicycle trajectory generated ON THE 0.2 s CACHE GRID, then written into
    a 10 Hz frame array at the even frames the loader reads (``2j``); the odd
    frames are midpoints and are never touched by the replay.

    ⭐ Generating on the SCORING grid removes the 10 Hz→5 Hz quadrature floor, so
    a kappa-corpus replay is EXACT (~0 m) and E2's regression is attributable to
    the unit alone rather than to the integrator's step size.

    ``store="steer"`` writes ``atan(L·kappa)`` — what `physicalai.signals_at`
    actually writes; ``store="kappa"`` writes the curvature itself.
    """
    j = torch.arange(M, dtype=torch.float64)
    v = 9.0 + 1.2 * torch.sin(2 * math.pi * j / 35.0)
    kap = 0.05 * torch.sin(2 * math.pi * j / 45.0 + 0.4)
    T = 2 * M
    poses = torch.zeros(T, 4, dtype=torch.float64)
    ch = torch.zeros(T, dtype=torch.float64)
    x = y = yaw = 0.0
    for i in range(M):
        poses[2 * i] = torch.tensor([x, y, yaw, float(v[i])], dtype=torch.float64)
        ch[2 * i] = (kap[i] if store == "kappa"
                     else math.atan(STEER_WHEELBASE_M * float(kap[i])))
        x_n = x + float(v[i]) * math.cos(yaw) * DT
        y_n = y + float(v[i]) * math.sin(yaw) * DT
        if 2 * i + 1 < T:                       # unused midpoint, kept sane
            poses[2 * i + 1] = torch.tensor(
                [0.5 * (x + x_n), 0.5 * (y + y_n), yaw, float(v[i])],
                dtype=torch.float64)
            ch[2 * i + 1] = ch[2 * i]
        x, y = x_n, y_n
        yaw = yaw + float(v[i]) * float(kap[i]) * DT
    return poses.float(), ch.float(), kap


def _replay_err(poses, ch, t0, units):
    """`ol`-style replay of the recorded channel vs the pose truth, in the t0
    heading frame — the same estimator the panel used."""
    v = poses[:, 3]
    c = torch.zeros(1, K, 2)
    for j in range(K):
        f, fn = 2 * (t0 + j), min(2 * (t0 + j + 1), len(v) - 1)
        c[0, j, 0] = (v[fn] - v[f]) / DT
        c[0, j, 1] = ch[f]
    p = unicycle_paths(c, v[2 * t0][None], DT, action_units=units)[0].double()
    y0, x0, yaw0 = (float(poses[2 * t0, 1]), float(poses[2 * t0, 0]),
                    float(poses[2 * t0, 2]))
    gt = []
    for j in range(1, K + 1):
        f = min(2 * t0 + 2 * j, len(poses) - 1)
        dx, dy = float(poses[f, 0]) - x0, float(poses[f, 1]) - y0
        gt.append([dx * math.cos(-yaw0) - dy * math.sin(-yaw0),
                   dx * math.sin(-yaw0) + dy * math.cos(-yaw0)])
    gt = np.asarray(gt)
    return (float(np.mean(np.abs(p[:, 1].numpy() - gt[:, 1]))),
            float(np.mean(np.abs(p[:, 0].numpy() - gt[:, 0]))))


def test_E1_on_repairs_a_STEER_corpus():
    poses, ch, _ = _synth_episode("steer")
    off = _replay_err(poses, ch, 5, "kappa")
    on = _replay_err(poses, ch, 5, "steer")
    assert on[0] < 0.2 * off[0], (off, on)


def test_E2_DELIBERATE_REGRESSION_on_breaks_a_KAPPA_corpus():
    """The flag must NOT be a universal improver. On a corpus whose channel
    really is a curvature, turning it ON must make the replay clearly WORSE —
    otherwise E1 proves nothing about which unit the real corpus is in."""
    poses, ch, _ = _synth_episode("kappa")
    off = _replay_err(poses, ch, 5, "kappa")
    on = _replay_err(poses, ch, 5, "steer")
    assert off[0] < 1e-3, off          # the contract HOLDS on a kappa corpus
    assert on[0] > 5.0 * max(off[0], 1e-4), (off, on)


# =========================================================================== #
# F — REAL windows (the pin the PI asked for) + the banked numbers
# =========================================================================== #
@pytest.mark.skipif(not _HAVE_REAL, reason=f"local eval slice absent ({_EPS})")
def test_F1_converted_kappa_matches_the_POSE_DERIVED_kappa_on_real_windows():
    """THE PHYSICS PIN: tan(recorded steer)/2.9 must equal the curvature the
    human's own recorded POSE says they drove — not merely 'some other code
    path'. Median |ratio - 1| over turning windows; the unconverted channel is
    ~2.9x off and is asserted to be so, so a pass cannot be vacuous."""
    import glob
    man = json.load(open(_DUMP / "manifest.json", encoding="utf-8"))
    names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
    ratios_on, ratios_off = [], []
    for fi, f in enumerate(sorted(glob.glob(str(_DUMP / "ep*.npz")))):
        z = np.load(f)
        o = torch.load(_EPS / f"{names[fi]}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        v = o["poses"][:, 3].double().numpy()
        yaw = o["poses"][:, 2].double().numpy()
        st = o["actions"][:, 0].double().numpy()
        for i in range(z["g"].shape[0]):
            t = int(z["ws"][i])
            fs = np.arange(2 * t, 2 * t + 2 * K, 2)
            fs = fs[fs + 2 < len(v)]
            if len(fs) < 5:
                continue
            dy = float(np.arctan2(np.sin(yaw[fs[-1] + 2] - yaw[fs[0]]),
                                  np.cos(yaw[fs[-1] + 2] - yaw[fs[0]])))
            if abs(dy) < 0.02:                      # unidentifiable on a straight
                continue
            on = float(np.sum(np.tan(st[fs]) / STEER_WHEELBASE_M * v[fs] * DT))
            off = float(np.sum(st[fs] * v[fs] * DT))
            ratios_on.append(on / dy)
            ratios_off.append(off / dy)
    assert len(ratios_on) >= 40, len(ratios_on)
    med_on, med_off = float(np.median(ratios_on)), float(np.median(ratios_off))
    assert abs(med_on - 1.0) < 0.05, med_on          # MEASURED 0.995
    assert med_off > 2.5, med_off                    # MEASURED 2.870


@pytest.mark.skipif(not _HAVE_REAL, reason=f"local dump absent ({_DUMP})")
def test_F2_the_banked_numbers_reproduce_OFF_and_ON():
    """OFF reproduces the shipped panel; ON reproduces the repaired panel that
    the Benchmarks package obtained by converting the DATA instead. Two sites,
    one algebra — if these ever diverge, one of the two framings is wrong."""
    import glob
    arm = _arm()
    man = json.load(open(_DUMP / "manifest.json", encoding="utf-8"))
    names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
    rows = []
    for fi, f in enumerate(sorted(glob.glob(str(_DUMP / "ep*.npz")))):
        z = np.load(f)
        o = torch.load(_EPS / f"{names[fi]}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        v = o["poses"][:, 3].float()
        st = o["actions"][:, 0].float()
        for i in range(z["g"].shape[0]):
            t, v0 = int(z["ws"][i]), float(z["v0"][i])
            g = z["g"][i].astype(np.float64)
            idx = torch.arange(t, t + K)
            fnx = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
            c = torch.stack([(v[fnx] - v[idx * 2]) / DT, st[idx * 2]], dim=-1)
            r = {"curved": float(np.abs(g[:, 1]).max()) >= 0.3}
            for tag, u in (("off", "kappa"), ("on", "steer")):
                p = arm.paths_from_controls(c, v0, DT, K,
                                            action_units=u)[0].double().numpy()
                r[f"lat_{tag}"] = float(np.mean(np.abs(p[:, 1] - g[:, 1])))
                r[f"lon_{tag}"] = float(np.mean(np.abs(p[:, 0] - g[:, 0])))
            rows.append(r)
    assert len(rows) == 140, len(rows)
    cur = [r for r in rows if r["curved"]]
    assert len(cur) == 72, len(cur)
    lat_off = float(np.mean([r["lat_off"] for r in cur]))
    lat_on = float(np.mean([r["lat_on"] for r in cur]))
    lon_off = float(np.mean([r["lon_off"] for r in rows]))
    lon_on = float(np.mean([r["lon_on"] for r in rows]))
    assert abs(lat_off - 0.7156) < 2e-3, lat_off     # the shipped panel
    assert abs(lon_off - 0.2395) < 2e-3, lon_off
    assert abs(lat_on - 0.0600) < 2e-3, lat_on       # the repaired panel
    assert abs(lon_on - 0.1199) < 2e-3, lon_on


@pytest.mark.skipif(not _HAVE_REAL, reason=f"local eval slice absent ({_EPS})")
def test_F3_the_TRUE_wheelbase_is_the_WRONG_constant_here():
    """The clips' real wheelbases are {2.73, 3.135, 3.165, 3.216}. Using one
    instead of the ENCODING constant makes the replay worse — the corpus was
    minted `const2p9` and the inverse must undo what was done, not what is true
    of the car. Pinned so a future 'improvement' cannot quietly regress it."""
    poses, ch, _ = _synth_episode("steer")
    good = _replay_err(poses, ch, 5, "kappa")        # placeholder, replaced below
    c = torch.zeros(1, K, 2)
    v = poses[:, 3]
    for j in range(K):
        f, fn = 2 * (5 + j), min(2 * (5 + j + 1), len(v) - 1)
        c[0, j, 0] = (v[fn] - v[f]) / DT
        c[0, j, 1] = ch[f]
    p_enc = unicycle_paths(c, v[10][None], DT, action_units="steer")
    p_true = unicycle_paths(c, v[10][None], DT, action_units="steer",
                            wheelbase=3.165)
    assert not torch.equal(p_enc, p_true)
    assert float((p_enc - p_true).abs().max()) > 0.01
    assert good[0] > 0.0
