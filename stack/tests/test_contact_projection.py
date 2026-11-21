"""`contact_projection` must make a COLLIDING path unrepresentable, not merely rare.

⛔ Every test here asserts on the OUTPUT tensor through ``rewards._collision`` — the
scorer of record — never through the projection's own bookkeeping. RETRACTION #30: a
control that checks the arithmetic it just performed cannot see that it performed it
on the wrong object; a control that re-implements the predicate it checks cannot see
that both share a convention error.

⚠️ Two of these tests are DELIBERATE-REGRESSION controls: they assert that the thing
being fixed really is broken in the fixture. Without them the suite proves nothing.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.refs import contact_projection as CP        # noqa: E402
from tanitad.refs import feasible_decode as FD           # noqa: E402
from tanitad.rl.rewards import _collision                # noqa: E402

R = CP.CONTACT_R_M


def _hit(paths, lead, r=R):
    return (_collision(paths, {"lead_path": lead.expand_as(paths),
                               "ego_radius_m": r / 2.0, "obs_radius_m": r / 2.0}) < 0)


def _fan(seed, B=6, N=24, S=5, scale=3.0):
    """A fan of random candidates against a PLAUSIBLE lead: ahead of the ego at t=0
    and moving forward, so the fixture's contacts are the ego's fault rather than an
    agent teleporting onto a parked car. ⚠️ The unavoidable class is real and is
    asserted separately; mixing it into this fixture would make every clearance test
    read a number that is not about the projection."""
    g = torch.Generator().manual_seed(seed)
    p = torch.cumsum(torch.randn(B, N, S, 2, generator=g) * scale, dim=-2)
    p = (p - p[:, :, :1, :]).double()
    x0 = 2.5 + torch.rand(B, 1, 1, generator=g) * 4.5
    v = torch.rand(B, 1, 1, generator=g) * 4.0
    t = torch.arange(S, dtype=torch.float32).reshape(1, 1, S) * 0.5
    lead = torch.stack([x0 + v * t, torch.zeros(B, 1, S)], dim=-1).double()
    return p, lead


# ---------------------------------------------------- the predicate is THE predicate
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_min_rel_distance_is_exactly_the_scorers_flag(seed):
    """The search needs a DISTANCE where the scorer has a FLAG. If the two ever
    disagree the projection is optimising a different constraint from the one that
    is scored — the `df` / units family, in a geometry costume."""
    p, lead = _fan(seed)
    d = CP.min_rel_distance(p, lead.expand_as(p))
    assert bool(((d < R) == _hit(p, lead)).all()), \
        "min_rel_distance < r must equal rewards._collision exactly"


# ------------------------------------------------------------ the structural zero
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_colliding_candidate_becomes_unrepresentable(seed):
    p, lead = _fan(seed)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    n_before = int((_hit(p, lead) ).sum())
    assert n_before > 0, ("the fixture must CONTAIN colliders or the test below "
                          "proves nothing — this is the deliberate-regression control")
    out, info = CP.project_contact_free(p, lead, has, margin_m=0.0)
    assert int(_hit(out, lead).sum()) == 0, \
        f"{int(_hit(out, lead).sum())} colliders survived the projection"
    a = CP.assert_no_contact(out, lead.expand_as(out), has)
    assert a["contact_count"] == 0 and a["min_clearance_m"] >= R


@pytest.mark.parametrize("margin_m", [0.0, 0.5, 2.0])
def test_the_margin_is_the_clearance_the_output_actually_has(margin_m):
    p, lead = _fan(7)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    out, info = CP.project_contact_free(p, lead, has, margin_m=margin_m)
    d = CP.min_rel_distance(out, lead.expand_as(out))
    assert float(d.min()) >= float(info["r_need_m"]) - 1e-9, \
        "the output must clear the margin the caller asked for, not merely the scorer's r"
    assert int(_hit(out, lead, r=R + margin_m).sum()) == 0, \
        "a margin m must give exact immunity to an agent-position error of m"


# --------------------------------------------------------------------- the controls
def test_C1_disabled_returns_the_input_object():
    p, lead = _fan(0)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    out, info = CP.project_contact_free(p, lead, has, enabled=False)
    assert out is p, "the disabled lever must SHORT-CIRCUIT, not re-derive"
    assert float((out - p).abs().max()) == 0.0


def test_C2_a_window_with_no_agent_is_untouched_exactly():
    p, lead = _fan(1)
    has = torch.zeros(p.shape[0], dtype=torch.bool)
    has[0] = True
    out, info = CP.project_contact_free(p, lead, has, margin_m=0.0)
    assert float((out[1:] - p[1:]).abs().max()) == 0.0, \
        "no agent track means NO INFORMATION about contact, never a free pass"
    assert bool((info["reason"][1:] == CP.REASON_NO_AGENT).all())


def test_C3_only_the_colliding_candidates_move():
    p, lead = _fan(2)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    out, info = CP.project_contact_free(p, lead, has, margin_m=0.0)
    moved = ((out - p).norm(dim=-1).amax(dim=-1) > 0)
    d_in = CP.min_rel_distance(p, lead.expand_as(p))
    should = d_in < float(info["r_need_m"])
    assert bool((moved == should).all()), \
        ("the moved set must be exactly the set violating the projection's own "
         "clearance; anything else means the projection is touching something "
         "other than collision")


def test_C4_an_already_clear_path_is_a_fixed_point():
    """⭐ The ARITHMETIC control C1 cannot give: the search RUNS on every candidate,
    and one it leaves at sigma = 1 must come back through the integrator to itself."""
    g = torch.Generator().manual_seed(3)
    p = torch.cumsum(torch.randn(4, 8, 5, 2, generator=g) * 2.0, dim=-2)
    p = (p - p[:, :, :1, :]).double()
    lead = torch.full((4, 1, 5, 2), 1.0e4, dtype=torch.float64)   # nothing to hit
    has = torch.ones(4, dtype=torch.bool)
    out, info = CP.project_contact_free(p, lead, has, margin_m=0.0)
    assert float(info["roundtrip_max_m"]) < 1e-9, \
        f"round-trip residual {float(info['roundtrip_max_m'])} is not a fixed point"
    assert float((out - p).abs().max()) == 0.0, \
        "the SHIPPED form returns the input slice where nothing was retracted"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_C5_the_retraction_can_never_worsen_friction(seed):
    """The property that makes the composition safe in EITHER order."""
    p, lead = _fan(seed)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    out, _ = CP.project_contact_free(p, lead, has, margin_m=0.0)
    before = FD.assert_feasible(p)
    after = FD.assert_feasible(out)
    assert after["peak_g_max"] <= before["peak_g_max"] + 1e-9
    assert after["max_abs_accel"] <= before["max_abs_accel"] + 1e-9
    assert after["max_abs_kappa"] <= before["max_abs_kappa"] + 1e-6


def test_the_composed_decode_is_simultaneously_feasible_and_clear():
    """⭐ The headline: friction AND contact unrepresentable on the SAME tensor."""
    p, lead = _fan(5)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    fric = FD.project_feasible(p, mu=FD.MU_KAMM)
    out, _ = CP.project_contact_free(fric, lead, has, margin_m=0.0,
                                     clamp_friction=True)
    f = FD.assert_feasible(out)
    assert f["envelope_rate"] == 0.0, f"envelope leaked: {f}"
    assert f["kamm_over_rate"] == 0.0, f"kamm leaked: {f}"
    assert int(_hit(out, lead).sum()) == 0, "contact leaked"


# ---------------------------------------------------- the integrator and its inverse
def test_integrate_controls_inverts_recover_controls():
    g = torch.Generator().manual_seed(11)
    p = torch.cumsum(torch.randn(5, 7, 5, 2, generator=g) * 2.0, dim=-2)
    p = (p - p[:, :, :1, :]).double()
    sp, hd, ac, la = FD.recover_controls(p)
    q = CP.integrate_controls(sp[..., 0], hd[..., 0], ac, la, clamp=False)
    assert float((q - p).abs().max()) < 1e-9, \
        "a second finite-difference convention is how a projection and its scorer " \
        "silently disagree"


# ------------------------------------------------------- the three-way taxonomy
def test_a_static_obstacle_outside_the_disc_is_ALWAYS_clearable():
    """⭐ THE CLEARABILITY THEOREM. sigma -> 0 leaves the ego at its origin, so any
    agent that stays outside the ego's disc has a collision-free member. Nothing may
    be reported UNAVOIDABLE in that case."""
    ego = torch.tensor([[0., 0.], [5., 0.], [10., 0.], [15., 0.], [20., 0.]])
    for x in (2.5, 4.0, 8.0, 12.0):
        lead = torch.tensor([[x, 0.]] * 5)
        p = ego[None, None].double()
        ld = lead[None, None].double()
        out, info = CP.project_contact_free(p, ld, torch.ones(1, dtype=torch.bool),
                                            margin_m=0.0, skip_first=False)
        assert int(info["reason"][0, 0]) != CP.REASON_UNAVOIDABLE, \
            f"a parked car {x} m ahead must be clearable by stopping short"
        assert float(CP.min_rel_distance(out, ld.expand_as(out),
                                         skip_first=False)) >= R


def test_an_agent_inside_the_ego_disc_at_rest_is_reported_UNAVOIDABLE():
    """⛔ And it must be reported, never silently 'cleared'. This is the residual a
    projection provably cannot constrain — the honest RL / perception objective."""
    ego = torch.tensor([[0., 0.], [5., 0.], [10., 0.], [15., 0.], [20., 0.]])
    lead = torch.tensor([[0.4, 0.]] * 5)                 # parked ON the ego
    p, ld = ego[None, None].double(), lead[None, None].double()
    out, info = CP.project_contact_free(p, ld, torch.ones(1, dtype=torch.bool),
                                        margin_m=0.0, skip_first=False)
    assert int(info["reason"][0, 0]) == CP.REASON_UNAVOIDABLE
    assert float(CP.min_rel_distance(out, ld.expand_as(out), skip_first=False)) < R, \
        "an unavoidable case must NOT be reported as cleared"


# ------------------------------- the DELIBERATE-REGRESSION control for the predicate
def test_the_scorers_predicate_never_sweeps_the_FIRST_segment():
    """⛔ `rewards._collision`'s moving-lead branch slices `lead[..., 1:] - traj[..., 1:]`,
    so the relative segment from t = 0 to t = 0.5 s is NEVER swept — the D-SWEPT-1
    corridor, at the step where the ego's displacement is largest.

    ⚠️ This test asserts the DEFECT, so that a future fix breaks it loudly rather than
    silently changing every banked contact number. The fixture is a car parked 1.5 m
    dead ahead and a candidate that drives straight through it at 10 m/s."""
    ego = torch.tensor([[0., 0.], [5., 0.], [10., 0.], [15., 0.], [20., 0.]])[None, None]
    lead = torch.tensor([[1.5, 0.]] * 5)[None, None]
    assert not bool(_hit(ego.double(), lead.double()).any()), \
        "if the scorer catches this, the hole is closed and this test must be updated"
    assert float(CP.min_rel_distance(ego.double(), lead.double().expand_as(ego.double()),
                                     skip_first=False)) < R, \
        "the full sweep must catch what the scorer's slice misses"


def test_the_projection_can_enforce_a_STRICTER_predicate_than_the_scorer():
    """⭐ The thing a reward cannot do without retraining: change what is
    unrepresentable. `skip_first=False` closes the hole above by construction."""
    p, lead = _fan(4)
    has = torch.ones(p.shape[0], dtype=torch.bool)
    out, info = CP.project_contact_free(p, lead, has, margin_m=0.0, skip_first=False)
    d = CP.min_rel_distance(out, lead.expand_as(out), skip_first=False)
    clear = (info["reason"] != CP.REASON_UNAVOIDABLE)
    assert float(d[clear].min()) >= R, \
        "the strict arm must clear every SEGMENT, including t=0 -> t=0.5 s"


# ------------------------------------------------------------------- input guards
def test_it_refuses_a_path_whose_first_point_is_not_the_origin():
    p = torch.ones(2, 3, 5, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        CP.project_contact_free(p, p[:, :1], torch.ones(2, dtype=torch.bool))


def test_it_refuses_a_wrongly_shaped_fan():
    with pytest.raises(ValueError):
        CP.project_contact_free(torch.zeros(4, 5, 2, dtype=torch.float64),
                                torch.zeros(4, 1, 5, 2, dtype=torch.float64),
                                torch.ones(4, dtype=torch.bool))
