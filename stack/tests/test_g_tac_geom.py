"""Tests for the `g_tac` Engine-A geometry deriver.

⭐ **THE LOAD-BEARING ONE IS `test_a_constant_curvature_bend_reads_zero_offset`.**
`LANE_TARGET` was ruled out by the PI on 2026-08-16 because a lateral-displacement
gate *"cannot distinguish a lane change from road curvature — a pure road curve
clears the gate at any highway speed"*. This module's whole claim is that a
CONSTANT-CURVATURE reference cancels road curvature exactly. That claim is worth
nothing asserted and everything measured, so it is measured here at four radii
and four speeds.

Every other property here is one whose failure would emit a *label* rather than
raise — the only failure mode that reaches a trained model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import g_tac_geom as G                            # noqa: E402

DT = 0.1


# --------------------------------------------------------------------------- #
# synthetic corpora — each an ANALYTIC path, so the expected value is known     #
# --------------------------------------------------------------------------- #
def _straight(T: int = 120, v: float = 10.0) -> torch.Tensor:
    s = torch.arange(T, dtype=torch.float64) * v * DT
    return torch.stack([s, torch.zeros(T, dtype=torch.float64),
                        torch.zeros(T, dtype=torch.float64),
                        torch.full((T,), v, dtype=torch.float64)], dim=-1)


def _arc(T: int = 120, v: float = 10.0, radius: float = 200.0,
         left: bool = True) -> torch.Tensor:
    """Exact constant-curvature path: kappa = +-1/radius, |v| constant."""
    kap = (1.0 if left else -1.0) / radius
    s = torch.arange(T, dtype=torch.float64) * v * DT
    th = kap * s
    r = 1.0 / kap
    x = r * torch.sin(th)
    y = r * (1.0 - torch.cos(th))
    return torch.stack([x, y, th, torch.full((T,), v, dtype=torch.float64)],
                       dim=-1)


def _straight_then_step(T: int = 120, v: float = 10.0, step_m: float = 3.5,
                        t0: int = 20, dur: int = 40) -> torch.Tensor:
    """Straight, then a smooth lateral displacement of EXACTLY ``step_m``."""
    s = torch.arange(T, dtype=torch.float64) * v * DT
    y = torch.zeros(T, dtype=torch.float64)
    for i in range(T):
        if i <= t0:
            y[i] = 0.0
        elif i >= t0 + dur:
            y[i] = step_m
        else:                                   # raised cosine — C1 continuous
            u = (i - t0) / dur
            y[i] = step_m * 0.5 * (1.0 - math.cos(math.pi * u))
    yaw = torch.zeros(T, dtype=torch.float64)
    yaw[1:] = torch.atan2(y[1:] - y[:-1], s[1:] - s[:-1])
    return torch.stack([s, y, yaw, torch.full((T,), v, dtype=torch.float64)],
                       dim=-1)


def _decel_to_stop(T: int = 120, v0: float = 10.0, a: float = -2.0,
                   t_start: int = 10) -> torch.Tensor:
    """Constant deceleration to rest — the stopping distance is analytic."""
    v = torch.zeros(T, dtype=torch.float64)
    x = torch.zeros(T, dtype=torch.float64)
    cur_v, cur_x = v0, 0.0
    for i in range(T):
        v[i], x[i] = cur_v, cur_x
        if i >= t_start:
            cur_v = max(cur_v + a * DT, 0.0)
        cur_x += cur_v * DT
    return torch.stack([x, torch.zeros(T, dtype=torch.float64),
                        torch.zeros(T, dtype=torch.float64), v], dim=-1)


# --------------------------------------------------------------------------- #
# the contract with the v6 vocabulary                                          #
# --------------------------------------------------------------------------- #
def test_every_token_is_either_emitted_or_refused():
    """⛔ A token in neither set is a BUG, not an abstention: it would be
    silently absent from the label stream with nobody able to say why."""
    from tanitad.models.v6 import TACTICAL_GOAL_TOKENS
    emitted = {"STOP_POINT"}   # CORRIDOR_OFFSET moved to REFUSED 2026-08-22
    covered = emitted | set(G.REFUSED)
    assert covered == set(TACTICAL_GOAL_TOKENS), (
        f"uncovered: {set(TACTICAL_GOAL_TOKENS) - covered}; "
        f"unknown: {covered - set(TACTICAL_GOAL_TOKENS)}")


def test_the_axis_abstains_are_the_factorings_own():
    from tanitad.models.v6 import (GOAL_AXIS_ABSTAIN, TACTICAL_GOAL_TOKENS_LAT,
                                   TACTICAL_GOAL_TOKENS_LON)
    assert "LAT_UNCONSTRAINED" in TACTICAL_GOAL_TOKENS_LAT
    assert "LON_UNCONSTRAINED" in TACTICAL_GOAL_TOKENS_LON
    assert set(GOAL_AXIS_ABSTAIN) == {"LAT_UNCONSTRAINED", "LON_UNCONSTRAINED"}


def test_module_has_no_situation_classifier_path():
    """Goal/situation disjointness (Sayed 2026-08-03), enforced structurally."""
    src = Path(G.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = code.split('"""')[0] + '"""'.join(code.split('"""')[2:])
    assert "import situations" not in code
    assert "data.situations" not in code
    assert "from tanitad.data import situations" not in code


def test_every_refusal_carries_a_reason():
    for tok, why in G.REFUSED.items():
        assert len(why) > 40, f"{tok} refusal is too thin to act on"


# --------------------------------------------------------------------------- #
# ⭐ THE CONTROL — a bend must read ZERO, at every radius and every speed       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [50.0, 100.0, 200.0, 500.0])
@pytest.mark.parametrize("v", [5.0, 10.0, 20.0, 30.0])
@pytest.mark.parametrize("left", [True, False])
def test_a_constant_curvature_bend_reads_zero_offset(radius, v, left):
    """⭐ THE TEST `LANE_TARGET` FAILED. A vehicle tracking a constant-radius
    bend is LANE-KEEPING; a deriver that calls that a corridor offset is the
    refuted gate wearing new clothes."""
    T = int(8.0 / DT) + 40
    poses = _arc(T=T, v=v, radius=radius, left=left)
    t = 20
    lab = G.g_tac_geom(poses, t, lat_arm="refuted-diagnostic")
    assert lab.lat.token == "LAT_UNCONSTRAINED", (
        f"R={radius} v={v} left={left} -> {lab.lat.token} "
        f"offset={lab.audit['lat_offset_m']}")
    assert abs(lab.audit["lat_offset_m"]) < 0.10, lab.audit


@pytest.mark.parametrize("v", [5.0, 10.0, 20.0, 30.0])
def test_a_straight_road_reads_zero_offset(v):
    poses = _straight(T=int(8.0 / DT) + 40, v=v)
    lab = G.g_tac_geom(poses, 20, lat_arm="refuted-diagnostic")
    assert lab.lat.token == "LAT_UNCONSTRAINED"
    assert abs(lab.audit["lat_offset_m"]) < 1e-6


# --------------------------------------------------------------------------- #
# and the POSITIVE control — a real displacement must be MEASURED, not missed   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("step", [1.0, 2.0, 3.5])
def test_a_lateral_step_is_recovered_to_within_10cm(step):
    """A control that only ever reads zero is a broken instrument, not a strict
    one. The offset must recover the displacement it was built from."""
    poses = _straight_then_step(T=140, v=10.0, step_m=step, t0=20, dur=40)
    lab = G.g_tac_geom(poses, 18, lat_arm="refuted-diagnostic")
    assert lab.lat.token == "CORRIDOR_OFFSET", lab.audit
    assert lab.audit["lat_offset_m"] == pytest.approx(step, abs=0.10), lab.audit
    assert lab.lat.args["arg0"] == pytest.approx(step, abs=0.10)


def test_the_sign_is_plus_left():
    """+1 = left, the DECLARED convention (`s2_derive` §sign, `refb_labels
    .ego_frame` +y = left). A sign flip here mirrors every lateral label."""
    poses = _straight_then_step(step_m=3.0)
    assert G.g_tac_geom(poses, 18, lat_arm="refuted-diagnostic").audit["lat_offset_m"] > 0
    mirrored = poses.clone()
    mirrored[:, 1] *= -1.0
    mirrored[:, 2] *= -1.0
    assert G.g_tac_geom(mirrored, 18, lat_arm="refuted-diagnostic").audit["lat_offset_m"] < 0


# --------------------------------------------------------------------------- #
# LON — STOP_POINT                                                              #
# --------------------------------------------------------------------------- #
def test_stop_point_recovers_the_analytic_stopping_distance():
    """v0=10, a=-2 -> the ego is at rest after 5 s having covered v0^2/(2|a|)
    = 25 m. The label must read that distance, not a plausible one."""
    poses = _decel_to_stop(T=140, v0=10.0, a=-2.0, t_start=0)
    lab = G.g_tac_geom(poses, 0)
    assert lab.lon.token == "STOP_POINT", lab.lon.reason
    assert lab.lon.args["arg0"] == pytest.approx(25.0, abs=1.5)


def test_stop_point_leaves_reason_unset_because_geometry_cannot_see_a_cause():
    """The IGNORE discipline: an unset categorical slot, never a plausible one."""
    poses = _decel_to_stop(T=140, v0=10.0, a=-2.0, t_start=0)
    lab = G.g_tac_geom(poses, 0)
    assert "reason" not in lab.lon.cat_args
    vals, mask = lab.lon.arg_vector()
    from tanitad.models.v6 import GOAL_ARG_NAMES
    for name, v, m in zip(GOAL_ARG_NAMES, vals, mask):
        if m == 0:
            assert math.isnan(v), f"{name} unset but not NaN — a quiet default"


def test_an_already_stopped_ego_abstains_rather_than_emitting_an_echo():
    """A STOP_POINT derived where v0~0 is predictable from v0, which the model
    is handed. That is the ego-speed echo family; it must abstain WITH a reason."""
    T = 140
    poses = torch.zeros(T, 4, dtype=torch.float64)
    lab = G.g_tac_geom(poses, 0)
    assert lab.lon.token == G.ABSTAIN
    assert "echo" in lab.lon.reason


def test_a_cruising_ego_is_lon_unconstrained():
    lab = G.g_tac_geom(_straight(T=140, v=12.0), 20)
    assert lab.lon.token == "LON_UNCONSTRAINED"


# --------------------------------------------------------------------------- #
# refusals and honesty                                                          #
# --------------------------------------------------------------------------- #
def test_a_band_that_runs_off_the_episode_abstains_rather_than_clamping():
    """`goal_tac_targets` CLAMPS by contract; a GOAL TOKEN must not — a clamped
    endpoint labels a shorter horizon than the token claims."""
    poses = _straight(T=40, v=10.0)
    lab = G.g_tac_geom(poses, 20, lat_arm="refuted-diagnostic")
    assert lab.lat.token == G.ABSTAIN
    assert "beyond the episode" in lab.lat.reason


def test_a_standstill_window_abstains_on_lat():
    poses = torch.zeros(140, 4, dtype=torch.float64)
    lab = G.g_tac_geom(poses, 20, lat_arm="refuted-diagnostic")
    assert lab.lat.token == G.ABSTAIN


def test_an_abstention_without_a_reason_is_refused_at_construction():
    with pytest.raises(ValueError, match="MUST carry a reason"):
        G.GTacField(G.ABSTAIN, "lat")


def test_a_token_on_the_wrong_axis_is_refused():
    with pytest.raises(ValueError, match="not on axis"):
        G.GTacField("STOP_POINT", "lat")


def test_unknown_arg_slots_are_refused():
    with pytest.raises(ValueError, match="GOAL_ARG_NAMES"):
        G.GTacField("CORRIDOR_OFFSET", "lat", args={"lat_offset_m": 1.0})


# --------------------------------------------------------------------------- #
# the census carries its own degeneracy control                                 #
# --------------------------------------------------------------------------- #
def test_census_flags_a_constant_only_emitter():
    """⛔ The control that must read a KNOWN value: a corpus of identical
    straight paths IS degenerate, and the census must say so."""
    labs = [G.g_tac_geom(_straight(T=140, v=10.0), t,
                         lat_arm="refuted-diagnostic") for t in range(20, 40)]
    cen = G.census(labs)
    assert cen["degenerate_lat"] is True
    assert cen["degenerate_lon"] is True
    assert cen["n_windows"] == 20


def test_census_is_not_degenerate_on_a_mixed_corpus():
    D = dict(lat_arm="refuted-diagnostic")
    labs = ([G.g_tac_geom(_straight(T=140, v=10.0), t, **D) for t in range(20, 30)]
            + [G.g_tac_geom(_straight_then_step(T=140, step_m=3.5), t, **D)
               for t in range(16, 26)])
    cen = G.census(labs)
    assert cen["degenerate_lat"] is False
    assert cen["abs_lat_offset_m"]["max"] > 1.0
    assert set(cen["lat"]) >= {"LAT_UNCONSTRAINED", "CORRIDOR_OFFSET"}


def test_census_reports_the_refused_set_so_absence_is_never_silent():
    cen = G.census([G.g_tac_geom(_straight(T=140), 20)])
    assert set(cen["refused_tokens"]) == set(G.REFUSED)
    assert "SPEED_BAND" in cen["refused_tokens"]


# --------------------------------------------------------------------------- #
# ⛔ THE REFUTATION, PINNED — so the LAT axis cannot silently come back         #
# --------------------------------------------------------------------------- #
def test_the_lat_axis_abstains_by_default():
    """`CORRIDOR_OFFSET` was REFUTED on real poses 2026-08-22. The default arm
    must abstain, and it must NOT claim `LAT_UNCONSTRAINED` — that token is a
    positive claim we cannot support, not a neutral placeholder."""
    poses = _straight_then_step(T=140, v=10.0, step_m=3.5)
    lab = G.g_tac_geom(poses, 18)                      # default lat_arm
    assert lab.lat.token == G.ABSTAIN
    assert lab.lat.token != "LAT_UNCONSTRAINED"
    assert "REFUTED" in lab.lat.reason
    assert lab.audit["lat_arm"] == "abstain"


def test_corridor_offset_is_in_the_refused_set_with_its_evidence():
    why = G.REFUSED["CORRIDOR_OFFSET"]
    for token in ("REFUTED", "ORACLE", "1.80", "9.92", "Engine C"):
        assert token in why, f"the refusal lost its evidence: {token!r} missing"


def test_the_refuted_arm_must_be_asked_for_explicitly():
    with pytest.raises(ValueError, match="lat_arm"):
        G.g_tac_geom(_straight(T=140), 20, lat_arm="on")


def test_the_lon_axis_still_ships_under_the_default_arm():
    """The refutation is LATERAL only. The LON axis is independent of the
    corridor reference and must be unaffected — that separation is the point of
    the LAT x LON factoring."""
    lab = G.g_tac_geom(_decel_to_stop(T=140, v0=10.0, a=-2.0, t_start=0), 0)
    assert lab.lon.token == "STOP_POINT"
    assert lab.lon.args["arg0"] == pytest.approx(25.0, abs=1.5)
    assert G.g_tac_geom(_straight(T=140, v=12.0), 20).lon.token == \
        "LON_UNCONSTRAINED"
