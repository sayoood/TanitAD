"""Does the planner still silently drive a parked car? MUTATION TEST, no GPU, no data needed.

⛔ THE DEFECT THIS PINS, MEASURED 2026-09-20 by the conformance review. `REFePlanner` declared
`requires_scenario = False`, so it never received the only object that carries a `log_name`.
`PlannerInitialization` holds only ['route_roadblock_ids', 'mission_goal', 'map_api'] and
`mission_goal` is a `StateSE2` of ['x','y','heading'] -- neither has one. `_log_hint` was a class
attribute that was never assigned. The camera lookup therefore could never resolve, and
`compute_planner_trajectory` returned `_hold(ego)` on EVERY step. A closed-loop run would have
scored a STATIONARY CAR and published it as a REFe result.

⭐ WHY "IT HOLDS" IS THE WORST POSSIBLE FAILURE. A crash is visible. A hold is a legal trajectory:
the benchmark scores it, the number looks like a weak policy rather than a broken one, and nothing
in the log says the planner never saw an image. That is why the fix is not only "resolve the name"
but "REFUSE after N consecutive holds" -- the refusal is the part that makes the failure visible.

THE ARMS
  1 no scenario at all       -> must RAISE within max_consec_holds  (the old behaviour, now loud)
  2 scenario with a log_name -> must record the hint and NOT raise on the first step
  3 CONTROL: one unresolvable frame -> must HOLD, not raise (holding is correct in the small)
A test that only checked arm 1 would pass on a planner that raises unconditionally, which is why
arms 2 and 3 exist.

Usage:  python diag_planner_holds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Goal:            # a StateSE2 stand-in: no log_name, exactly as measured
    x = y = heading = 0.0


class _Init:            # PlannerInitialization stand-in: the three real fields, no log_name
    route_roadblock_ids: list = []
    mission_goal = _Goal()
    map_api = None


class _Scenario:
    def __init__(self, log_name):
        self.log_name = log_name


def main() -> int:
    import planner as P
    print("Does the planner still silently drive a parked car?\n")
    results = {}

    # build without touching CUDA or a checkpoint; the model is irrelevant to this defect
    # ⚠️ THIS STUB HAS NOW DRIFTED FROM THE REAL OBJECT TWICE, AND THE SECOND TIME WAS TONIGHT:
    # `_hold` reads `self.cfg.horizon_steps` and the stub did not set `cfg`, so the arm died with
    # `AttributeError: 'REFePlanner' object has no attribute 'cfg'` instead of testing anything.
    # ⛔ That is the SAME FAMILY as the defect this file was written for -- a harness that stands in
    # for the object under test drifts away from it -- so the assertion below is the real fix: it
    # refuses to run at all unless the stub carries every attribute the exercised methods touch.
    NEEDED = ("_init", "_route_poly", "_scenario", "_log_hint", "_log_hint_source",
              "n_steps", "n_no_frame", "_consec_holds", "max_consec_holds", "cfg", "frames")

    class _Frames:                      # FrameResolver stand-in: only miss_reason is read here
        miss_reason = ""

    def make(scenario):
        p = P.REFePlanner.__new__(P.REFePlanner)
        p._init = None
        p._route_poly = None
        p._scenario = scenario
        p._log_hint = None
        p._log_hint_source = "unset"
        p.n_steps = p.n_no_frame = p._consec_holds = 0
        p.max_consec_holds = 20
        p.cfg = P.REFeConfig()          # `_hold` needs horizon_steps; the real ctor sets this
        p.frames = _Frames()            # the refusal message quotes frames.miss_reason
        missing = [a for a in NEEDED if not hasattr(p, a)]
        assert not missing, (
            f"the stub is missing {missing} -- it has drifted from REFePlanner and would test "
            f"the harness rather than the planner")
        return p

    # --- arm 2 first, because it establishes that the hint mechanism works at all ---------
    p = make(_Scenario("2021.05.12.23.36.44_veh-35_01133_01535"))
    p.initialize(_Init())
    ok2 = p._log_hint == "2021.05.12.23.36.44_veh-35_01133_01535" and \
        p._log_hint_source == "scenario.log_name"
    results["2 scenario with a log_name is recorded"] = ok2
    print(f"  arm 2: hint {p._log_hint!r} from {p._log_hint_source!r}")

    # --- arm 1: no scenario => the hint cannot resolve, and the planner must go LOUD ------
    p = make(None)
    p.initialize(_Init())
    print(f"  arm 1: hint {p._log_hint!r} from {p._log_hint_source!r}")
    # ⛔ THIS ARM USED TO SIMULATE THE COUNTER INSTEAD OF CALLING THE PLANNER. It incremented
    # `_consec_holds` itself and asserted its own loop -- so it would pass even if
    # `compute_planner_trajectory` had no refusal in it at all. Drive the REAL method with a stub
    # input whose frame can never resolve, and catch the refusal it is supposed to raise.
    # ⛔ A REAL EgoState, NOT A STUB WITH A time_point ON IT. The stub version got as far as
    # `_hold`, which needs `rear_axle`, `dynamic_car_state`, `car_footprint` and a `TimePoint` that
    # supports arithmetic -- and a stub thin enough to write by hand cannot supply them. That is
    # the whole lesson of this file: the broken `_hold` survived three reviews precisely because a
    # stub stood in for it. Building the real object is what makes arm 3 an actual test.
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2, StateVector2D, TimePoint)
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    real_ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(1_000_000),
        vehicle_parameters=get_pacifica_parameters())

    class _Hist:
        ego_states = [real_ego]

    class _In:
        history = _Hist()

    raised, holds = False, 0
    p._image_for = lambda ego: None          # the frame genuinely cannot resolve
    for i in range(p.max_consec_holds + 5):
        try:
            p.compute_planner_trajectory(_In())
        except RuntimeError:
            raised, holds = True, p._consec_holds
            break
    results["1 no scenario raises within max_consec_holds"] = raised and holds == p.max_consec_holds
    print(f"  arm 1: would refuse after {holds} consecutive holds")

    # --- arm 3 CONTROL: a single unresolvable frame must NOT raise ------------------------
    # ⚠️ THIS ARM WAS ALSO TAUTOLOGICAL: it set `_consec_holds = 1` and then asserted 1 < 20,
    # which is arithmetic about the test, not behaviour of the planner. Call the real method once
    # and require it to RETURN (a hold) rather than raise.
    p = make(_Scenario("some_log"))
    p.initialize(_Init())
    p._image_for = lambda ego: None
    # ⛔ THIS ARM USED TO STUB `_hold`, WHICH IS EXACTLY WHY A BROKEN `_hold` SURVIVED THREE
    # REVIEWS. It emitted 21 states with ONE distinct timestamp and `get_state_at_time` raises on
    # that -- the stub replaced the thing under test. Call the REAL `_hold` with a real EgoState.
    try:
        r3 = p.compute_planner_trajectory(_In()) is not None
    except RuntimeError:
        r3 = False
    results["3 CONTROL: one hold RETURNS, does not raise"] = r3

    # --- arm 4 CONTROL: the class no longer declares requires_scenario = False -----------
    results["4 CONTROL: requires_scenario is True"] = P.REFePlanner.requires_scenario is True

    print()
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    good = all(results.values())
    print("\n" + ("PLANNER_HOLDS_GUARDED" if good else "PLANNER_HOLDS_UNGUARDED"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
