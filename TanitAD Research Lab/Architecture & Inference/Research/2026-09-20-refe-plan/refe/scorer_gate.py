"""THE SCORER GATE -- the only check that has ever told the truth about this pipeline.

⛔ WHY THIS FILE EXISTS. Four successive estimates of the scorer gap were each greener than the
last and each wrong (name grep -> wrong object -> imports-only self-test -> "it runs and returns
real numbers"). The fourth was the worst, because the pipeline genuinely ran and genuinely returned
plausible PDM numbers -- while scoring a 25 m sideways veer IDENTICALLY to the teacher's own path.

⭐ THE ACCEPTANCE CRITERION IS A DISCRIMINATING CONTROL, NOT A SUCCESSFUL RUN.
   PASS  = the deliberately bad candidates score DIFFERENTLY from the teacher's realised path.
   FAIL  = every candidate scores the same, however healthy the log looks.
A control that MUST read a different value, and does not, is the whole instrument.

The three candidates (the last two are deliberate-regression arms):
   good      the teacher's own realised 4 s trajectory, straight out of the simulation log
   veer      the same path pushed 25 m sideways -- it MUST leave the drivable area
   stopped   a stationary ego -- it MUST fail goal progress

Usage:
  python scorer_gate.py --log <simulation_log.msgpack.xz> [--step 40] [--no-enrich]

`--no-enrich` is the deliberate-regression arm for the FIX ITSELF: it reruns with the lane graph
left empty and must reproduce the historical inert behaviour. A gate that cannot go red on the
known defect is not a gate.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

TEACHER_YAML = os.environ.get(   # pod portability: teacher_env.sh sets this; dev-box path is the fallback
    "DRIVERL_EVAL_CONFIG_PATH", "C:/Users/Admin/dz/DriveZero/DriveRL/release/configs/driverl_teacher.yaml")


def load_domain_randomization():
    """The yaml's own deterministic reward weights -- WITHOUT this the targets are not reproducible.

    ⛔ MEASURED 2026-09-20. `DriveRLNuPlanFeatureBuilderConfig.domain_randomization` defaults to
    None, and the ScenarioData then carries a RandomizedFeatureDict that DRAWS its weights: the same
    candidate scored -1.5194 in one process and -1.7086 in the next, with
    `collision_reward_weight` reading -1.0415 rather than the configured -1.0. Within a process it
    is stable (5/5 identical), so a single run looks perfectly deterministic and the drift only
    appears when you compare two runs -- exactly the shape that silently poisons a target bank
    built in shards.
    ⭐ Their released `driverl_teacher.yaml` already carries the fix and says so in a comment:
    `domain_randomization: {enabled: false, collision_reward_weight: {default: -1.0}, ...}` are the
    "Deterministic reward values used by the inference/TTS runtime". We were simply not passing it.
    ⚠️ The `info` channels were reproducible throughout; only the WEIGHTED `reward` channels moved.
    """
    import yaml
    from driverl.env.domain_randomization.config import (
        DomainRandomizationConfig, RewardFeatureConfig)
    import dataclasses
    y = yaml.safe_load(open(TEACHER_YAML, encoding="utf-8")) or {}
    found = {}

    def walk(d):
        for k, v in (d or {}).items():
            if k == "domain_randomization" and isinstance(v, dict):
                found.update(v)
            elif isinstance(v, dict):
                walk(v)
    walk(y)
    if not found:
        return None, {}
    names = {f.name for f in dataclasses.fields(DomainRandomizationConfig)}
    kw = {"enabled": bool(found.get("enabled", False))}
    for k, v in found.items():
        if k in names and isinstance(v, dict):
            kw[k] = RewardFeatureConfig(**{kk: vv for kk, vv in v.items()
                                           if kk in {"min", "max", "enabled", "default",
                                                     "as_feature", "calculator_default"}})
    return DomainRandomizationConfig(**kw), found


def build_engine_config(device="cpu", batch_size=1, max_agents=128, num_steps=20):
    """EngineMergedConfig from THEIR released teacher yaml + the six positional args it needs."""
    import yaml
    from driverl.env.engine.config import EngineMergedConfig
    import dataclasses
    y = yaml.safe_load(open(TEACHER_YAML, encoding="utf-8")) or {}
    flat: dict = {}

    def walk(d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                walk(v)
            else:
                flat.setdefault(k, v)
    walk(y)
    fields = {f.name for f in dataclasses.fields(EngineMergedConfig)}
    kw = {k: v for k, v in flat.items() if k in fields and k not in
          ("batch_size", "max_agents", "frame_time_interval", "dynamics_model", "device",
           "num_steps")}
    kw.pop("domain_randomization", None)
    dr, _raw = load_domain_randomization()
    if dr is not None:
        kw["domain_randomization"] = dr
    # ⛔ TWO TIME BASES. `frame_time_interval` is the step the CALCULATORS use for their own
    # derivatives, and it was hardcoded 0.1 while our candidate poses sit on a 0.2 s grid. Every
    # second derivative they compute was therefore 2x too large, and the driving-direction window
    # -- defined in frames -- was 2-4x too wide, with its width depending on `--stride`, which is a
    # cost knob and has no business changing a metric's definition.
    # ⭐ It is the same error as the velocity `dt` fixed earlier today, one level down: there I
    # divided by the wrong step, here I TOLD THEM the wrong step. Import the one constant that
    # names the candidate grid instead of restating it.
    import score_proposals as _SP
    cfg = EngineMergedConfig(batch_size=batch_size, max_agents=max_agents,
                             frame_time_interval=_SP.TRAJ_DT, dynamics_model="nuplan_bicycle",
                             device=device, num_steps=num_steps, **kw)
    return cfg, sorted(kw)


def scenario_data_from_log(log_path: str, step: int, device="cpu"):
    """Build a real ScenarioData OFFLINE from a banked simulation log. No live simulation."""
    from nuplan.planning.simulation.simulation_log import SimulationLog
    from nuplan.planning.simulation.planner.abstract_planner import (
        PlannerInitialization, PlannerInput)
    from nuplan.planning.simulation.history.simulation_history_buffer import (
        SimulationHistoryBuffer)
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration)
    from driverl.nuplan.feature_builder import DriveRLNuPlanFeatureBuilder

    log = SimulationLog.load_data(file_path=Path(log_path))
    sc = log.scenario
    smps = log.simulation_history.data
    step = min(max(step, 8), len(smps) - 25)

    egos = [s.ego_state for s in smps[: step + 1]]
    obs = [s.observation for s in smps[: step + 1]]
    keep = min(len(egos), 8)
    buf = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=keep, ego_states=egos[-keep:], observations=obs[-keep:],
        sample_interval=0.1)
    init = PlannerInitialization(
        route_roadblock_ids=list(sc.get_route_roadblock_ids()),
        mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
    pin = PlannerInput(
        iteration=SimulationIteration(time_point=egos[-1].time_point, index=step),
        history=buf, traffic_light_data=list(sc.get_traffic_light_status_at_iteration(step)))

    dr, _raw = load_domain_randomization()
    # ⛔ pass it to the BUILDER too: the calculators read scenario_data.randomized_features, which
    # the builder creates -- setting it only on the engine config leaves the draw in place.
    builder = (DriveRLNuPlanFeatureBuilder(domain_randomization=dr) if dr is not None
               else DriveRLNuPlanFeatureBuilder())
    sd = builder.build(pin, init)
    try:
        log_sd = builder.build_log_scenario_data(sd, sc, step, 20)
    except Exception:
        log_sd = sd
    return sd, log_sd, sc, smps, step, init


def teacher_future(smps, step, horizon=20, stride=2):
    """The teacher's realised future in the EGO frame at `step` -- the same frame inject_proposal
    demands and the same construction build_targets.py banks."""
    import math
    P = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y, s.ego_state.rear_axle.heading]
                  for s in smps])
    px, py, pyaw = P[step]
    sel = slice(step + stride, step + horizon * stride + 1, stride)
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    dx, dy = P[sel, 0] - px, P[sel, 1] - py
    xy = np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)
    yaw = ((P[sel, 2] - pyaw + math.pi) % (2 * math.pi)) - math.pi
    return torch.tensor(xy, dtype=torch.float32), torch.tensor(yaw, dtype=torch.float32)


def nearest_curb_target(sd, over_m: float = 8.0):
    """A point `over_m` BEYOND the nearest curb -- an ANALYTIC target whose verdict is known.

    ⛔ THE CONTROL THAT WAS WRONG BEFORE. The original regression arm pushed the teacher's path
    25 m sideways and expected off-road. MEASURED in this scene, +y moves the ego AWAY from the
    nearest curb (corner distance 10.66 m at 0 m -> 19.06 m at +14 m), reaching a minimum of 0.87 m
    only at +40 m and never touching. So OffRoad staying silent was CORRECT and my control was the
    defect -- I was one step from filing "their off-road detector is broken".
    ⭐ Constructing the violation from the map removes the guesswork: aim at the curb we measured.
    """
    from driverl.datatypes.data_enums import LaneType
    from driverl.utils.geometry import project_points_to_segments
    lp, lm = sd.lanes_points, sd.lanes_points_mask
    curb = (lm & (lp[..., 0, 2] == LaneType.CURB.code))[0]
    if int(curb.sum()) == 0:
        return None, None, 0
    s0, s1 = lp[0, curb, 0, :2], lp[0, curb, 1, :2]
    _, _, d = project_points_to_segments(torch.zeros(2).view(1, 1, 1, 2), s0.view(1, 1, -1, 2),
                                         s1.view(1, 1, -1, 2), return_dist=True)
    j = int(d.reshape(-1).argmin())
    mid = (s0[j] + s1[j]) / 2
    tgt = mid * (1 + over_m / max(float(mid.norm()), 1e-6))
    return tgt, float(d.reshape(-1)[j]), int(curb.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--no-enrich", action="store_true",
                    help="regression arm: leave the lane graph empty and see if it matters")
    a = ap.parse_args(argv)

    import math
    import score_proposals as SP
    import augment_routes as AR

    print("THE SCORER GATE -- can it tell a good path from one that drives over a curb?\n")
    cfg, used = build_engine_config()
    print(f"  engine config      OK  ({len(used)} values from driverl_teacher.yaml, "
          f"domain randomization PINNED)")
    calcs = SP.build_calculators(cfg)
    sd, log_sd, sc, smps, step, init = scenario_data_from_log(a.log, a.step)
    print(f"  calculators        {len(calcs)}/6   ScenarioData OK from {Path(a.log).name} step {step}")

    if not a.no_enrich:
        sd, stats = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                                         init.route_roadblock_ids)
        lg = sd.lanes_centers_groups
        print(f"  lane graph         ENRICHED  {tuple(lg.shape)} nonzero {int((lg != 0).sum())}"
              f"  segments {stats.get('center_segments')}")
    else:
        print("  lane graph         NOT ENRICHED (regression arm)")

    good_xy, good_yaw = teacher_future(smps, step)
    tgt, curb_d, n_curb = nearest_curb_target(sd, over_m=8.0)
    if tgt is None:
        print("\n  NO CURB IN RANGE -- this scene cannot host the control. Pick another log.")
        return 2
    cross_xy = torch.stack([tgt * f for f in torch.linspace(1.0 / 20, 1.0, 20)], 0)
    cross_yaw = torch.full((20,), math.atan2(float(tgt[1]), float(tgt[0])))
    stop_xy = torch.zeros_like(good_xy)
    print(f"  nearest curb       {curb_d:.2f} m from the ego, {n_curb} curb segments in range")

    cands = {"teacher": (good_xy, good_yaw),
             "over-curb": (cross_xy, cross_yaw),
             "stopped": (stop_xy, torch.zeros_like(good_yaw))}
    ends = "  ".join(f"{k} [{v[0][-1,0]:.1f},{v[0][-1,1]:.1f}]" for k, v in cands.items())
    print(f"  endpoints (ego frame):  {ends}")

    end = {k: SP.score_proposal(sd, log_sd, calcs, xy, yw) for k, (xy, yw) in cands.items()}
    roll = {k: SP.score_proposal_rollout(sd, log_sd, calcs, xy, yw) for k, (xy, yw) in cands.items()}

    print(f"\n  {'signal':38s} {'teacher':>9s} {'over-curb':>10s} {'stopped':>9s}    "
          f"{'teacher':>9s} {'over-curb':>10s} {'stopped':>9s}")
    print(f"  {'':38s} {'------ ENDPOINT ONLY ------':^30s}    "
          f"{'----- STEP-WISE ROLLOUT -----':^30s}")
    keys = sorted(k for k in set(end["teacher"]) if not k.endswith(".error"))
    n_end = n_roll = 0
    for k in keys:
        e = [end[c].get(k, float("nan")) for c in cands]
        r = [roll[c].get(k, float("nan")) for c in cands]
        de = (not any(np.isnan(e))) and (max(e) - min(e) > 1e-6)
        dr = (not any(np.isnan(r))) and (max(r) - min(r) > 1e-6)
        n_end += int(de); n_roll += int(dr)
        if de or dr:
            print(f"  {k:38s} {e[0]:9.4f} {e[1]:10.4f} {e[2]:9.4f}    "
                  f"{r[0]:9.4f} {r[1]:10.4f} {r[2]:9.4f}")

    offroad_t = roll["teacher"].get("off_road.OffRoad.info", float("nan"))
    offroad_c = roll["over-curb"].get("off_road.OffRoad.info", float("nan"))
    end_c = end["over-curb"].get("off_road.OffRoad.info", float("nan"))
    print(f"\n  differing signals:  endpoint {n_end}   rollout {n_roll}")
    print(f"  OffRoad on the teacher path (must be 0):        {offroad_t:.4f}")
    print(f"  OffRoad on the over-curb path (must be 1):      {offroad_c:.4f}")
    print(f"  ... same path scored at its ENDPOINT only:      {end_c:.4f}"
          f"   <- 0 here is the POINT: the crossing happens mid-horizon")

    checks = {
        "teacher path is clean": offroad_t == 0.0,
        "over-curb path is caught by the rollout": offroad_c == 1.0,
        "rollout separates more signals than the endpoint": n_roll >= n_end,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok = all(checks.values())
    print("\n" + ("SCORER_DISCRIMINATES" if ok else "SCORER_INERT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
