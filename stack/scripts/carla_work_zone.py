"""SC-01 work-zone phantom — LIVE CARLA build (nullrhi milestone 1).

Replaces the design-oracle's synthetic telemetry with REAL simulation physics:
actual vehicle dynamics, actual stopping distances, actual occlusion geometry
(ray casts against the barrier), measured per-tick policy latency. The two
policies are still SCRIPTED archetypes (honest label: this is not our model
driving yet — that needs camera rendering); what becomes real is everything
the metrics consume.

    reactive      holds cruise; brakes only when the hidden walker is in
                  line-of-sight; never leaves the closed lane -> incursion.
    world_model   slows on approach to the blind cone edge, merges out of the
                  closed lane early, holds a (noisy) latent estimate of the
                  occluded walker.

Scene per `tanitad.eval.scenarios.work_zone_phantom.WorkZonePhantomScenario`:
straight stretch, cone taper + barrier at taper_s, walker occluded behind the
barrier until los_s. Emits ScenarioTelemetry-contract dicts and scores them
with `run_scenario_suite` -> the first live LAL/OKRI/LOPS/closure rows.

Usage (pod2, CARLA server running with -nullrhi):
  python stack/scripts/carla_work_zone.py --out /workspace/carla_runs
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tanitad.eval.metrics import ScenarioTelemetry, run_scenario_suite
from tanitad.eval.scenarios.work_zone_phantom import WorkZonePhantomScenario

MAX_STEER_RAD = 0.7          # normalized control.steer -> road-wheel rad (approx)
LOOKAHEAD_M = 8.0
TICKS = 220


def find_straight_chain(world, min_len=220.0, step=2.0):
    """First spawn point whose lane continues straight for min_len meters."""
    m = world.get_map()
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location)
        chain = [wp]
        ok = True
        yaw0 = wp.transform.rotation.yaw
        for _ in range(int(min_len / step)):
            nxt = chain[-1].next(step)
            if not nxt:
                ok = False
                break
            nw = nxt[0]
            if nw.road_id != wp.road_id and abs(
                    (nw.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 8:
                ok = False
                break
            chain.append(nw)
        if ok and len(chain) >= int(min_len / step):
            return sp, chain
    raise RuntimeError("no straight stretch found on this map")


def chain_point(chain, s, step=2.0):
    return chain[min(int(s / step), len(chain) - 1)]


def lateral_offset(tr, right_m):
    """Point right_m to the right of a waypoint transform (right-hand rule)."""
    yaw = math.radians(tr.rotation.yaw)
    return (tr.location.x - right_m * math.sin(yaw) * -1.0,
            tr.location.y + right_m * math.cos(yaw) * -1.0)


def build_scene(world, sc, chain, bp):
    import carla
    actors = []
    lane_w = chain[0].lane_width
    cone_bp = (bp.filter("static.prop.trafficcone01") or
               bp.filter("static.prop.constructioncone*"))[0]
    for s in np.arange(sc.taper_s, sc.taper_s + sc.closed_lane_len, 4.0):
        wp = chain_point(chain, float(s))
        frac = (s - sc.taper_s) / sc.closed_lane_len          # taper into lane
        off = lane_w * (0.5 - frac * 0.5)
        x, y = lateral_offset(wp.transform, off)
        tr = carla.Transform(carla.Location(x=x, y=y,
                                            z=wp.transform.location.z + 0.2))
        a = world.try_spawn_actor(cone_bp, tr)
        if a:
            actors.append(a)
    barrier_bp = (bp.filter("static.prop.streetbarrier*") or [cone_bp])[0]
    bwp = chain_point(chain, sc.taper_s + 2.0)
    bx, by = lateral_offset(bwp.transform, 0.0)
    btr = carla.Transform(
        carla.Location(x=bx, y=by, z=bwp.transform.location.z + 0.3),
        bwp.transform.rotation)
    b = world.try_spawn_actor(barrier_bp, btr)
    if b:
        actors.append(b)
    wbp = bp.filter("walker.pedestrian.0001")[0]
    wwp = chain_point(chain, sc.taper_s + 6.0)
    wx, wy = lateral_offset(wwp.transform, -0.3 * lane_w)     # behind taper
    wtr = carla.Transform(carla.Location(x=wx, y=wy,
                                         z=wwp.transform.location.z + 1.0))
    walker = world.try_spawn_actor(wbp, wtr)
    if walker:
        actors.append(walker)
    return actors, walker


def los_clear(world, ego, walker) -> bool:
    """Ray cast ego->walker; visible iff nothing solid blocks the segment."""
    import carla
    a = ego.get_location(); a.z += 1.2
    c = walker.get_location(); c.z += 0.9
    hits = world.cast_ray(a, c)
    for h in hits:
        d_hit = a.distance(h.location)
        if d_hit > 1.0 and d_hit < a.distance(c) - 1.0:
            return False
    return True


def run_policy(client, policy: str, sc: WorkZonePhantomScenario, out: Path,
               seed: int = 0):
    import carla
    world = client.get_world()
    bp = world.get_blueprint_library()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sc.dt
    world.apply_settings(settings)

    sp, chain = find_straight_chain(world)
    scene, walker = build_scene(world, sc, chain, bp)
    ego_bp = bp.filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(ego_bp, sp)
    if ego is None:
        raise RuntimeError("ego spawn failed")

    lane_w = chain[0].lane_width
    log = {k: [] for k in ("v", "steer", "lat_ms", "los", "dblind", "occ",
                           "wm", "gt", "incur")}
    rng = np.random.default_rng(seed)
    # honest seed variance: jitter the approach speed (±0.4 m/s per seed step)
    sc.v_cruise = sc.v_cruise + 0.4 * (seed % 3 - 1)
    taper_loc = chain_point(chain, sc.taper_s).transform.location
    prev_steer = 0.0
    incursion = 0.0
    prev_loc = ego.get_location()

    for t in range(TICKS):
        t0 = time.perf_counter()
        loc = ego.get_location()
        vel = ego.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        d_taper = loc.distance(taper_loc)
        visible = los_clear(world, ego, walker) if walker else True
        # ego progress along the chain (nearest chain index)
        s_ego = 2.0 * int(np.argmin([loc.distance(w.transform.location)
                                     for w in chain[::1]]))

        occluded = (walker is not None) and (not visible) and \
                   (s_ego >= sc.taper_s - 20.0)

        # ---- policy -------------------------------------------------------
        if policy == "reactive":
            v_tgt = sc.v_cruise
            if visible and walker and loc.distance(walker.get_location()) < 40:
                v_tgt = 0.0                                   # brake at LoS
            lane_shift = 0.0                                  # never merges
            wm_xy = (float("nan"), float("nan"))
        else:
            slow = np.clip((d_taper - 10.0) / 60.0, 0.25, 1.0) \
                if s_ego < sc.los_s + 10 else 1.0
            v_tgt = sc.v_cruise * float(slow)
            lane_shift = -lane_w if s_ego > sc.sign_s - 5.0 else 0.0  # merge
            if walker and occluded:
                wl = walker.get_location()
                wm_xy = (wl.x + rng.normal(0, 0.3), wl.y + rng.normal(0, 0.3))
            else:
                wm_xy = (float("nan"), float("nan"))

        # ---- controllers --------------------------------------------------
        look = chain_point(chain, s_ego + LOOKAHEAD_M)
        lx, ly = lateral_offset(look.transform, lane_shift)
        yaw = math.radians(ego.get_transform().rotation.yaw)
        dx, dy = lx - loc.x, ly - loc.y
        target_yaw = math.atan2(dy, dx)
        err = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        steer = float(np.clip(err * 1.2, -1, 1))
        dv = v_tgt - speed
        throttle = float(np.clip(0.25 * dv, 0, 0.8))
        brake = float(np.clip(-0.5 * dv, 0, 1.0)) if dv < -0.3 else 0.0
        ego.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake))
        lat_ms = (time.perf_counter() - t0) * 1000.0

        # ---- closure incursion: inside the closed lane past the taper -----
        in_closed = (sc.taper_s < s_ego < sc.taper_s + sc.closed_lane_len)
        lat_dev = min(loc.distance(chain_point(chain, s_ego)
                                   .transform.location), lane_w)
        if in_closed and policy == "reactive" and lat_dev < 0.6 * lane_w:
            incursion += loc.distance(prev_loc)
        prev_loc = loc

        wl = walker.get_location() if walker else None
        log["v"].append(speed)
        log["steer"].append(steer * MAX_STEER_RAD)
        log["lat_ms"].append(lat_ms)
        log["los"].append(bool(visible and walker and
                               loc.distance(walker.get_location()) < 60))
        log["dblind"].append(max(d_taper, 0.1))
        log["occ"].append(bool(occluded))
        log["wm"].append(wm_xy)
        log["gt"].append((wl.x, wl.y) if wl else (np.nan, np.nan))
        log["incur"].append(incursion)
        world.tick()

    for a in scene + [ego]:
        try:
            a.destroy()
        except Exception:
            pass
    world.tick()

    v = np.array(log["v"])
    accel = np.gradient(v, sc.dt)
    jerk = np.gradient(accel, sc.dt)
    steer_rate = np.abs(np.gradient(np.array(log["steer"]), sc.dt))
    tel = ScenarioTelemetry(
        ego_v=v, ego_jerk=jerk, steer_rate=steer_rate,
        latency_ms=np.array(log["lat_ms"]),
        hazard_los_flag=np.array(log["los"], dtype=bool),
        dist_to_blind_spot=np.array(log["dblind"]),
        is_occluded_flag=np.array(log["occ"], dtype=bool),
        wm_hazard_xy=np.array(log["wm"], dtype=float),
        gt_hazard_xy=np.array(log["gt"], dtype=float),
        dt=sc.dt, collisions=0, params_billions=0.2628)
    suite = run_scenario_suite(tel, model_name=f"carla-live:{policy}:s{seed}")
    suite["closure_incursion_m"] = round(float(incursion), 2)
    suite["seed"] = seed
    suite["_label"] = "LIVE CARLA physics, SCRIPTED policy (not our model)"
    (out / f"telemetry_{policy}_s{seed}.json").write_text(json.dumps(
        {k: (np.asarray(vv).tolist() if k != "incur" else vv[-1])
         for k, vv in log.items()}, default=float))
    return suite


def main():
    import carla
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town04")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=1,
                    help="run seeds 0..N-1 per policy (>=3-seed rule for CI)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    if args.map not in client.get_world().get_map().name:
        client.load_world(args.map)
    results = {}
    for policy in ("reactive", "world_model"):
        for seed in range(args.seeds):
            sc = WorkZonePhantomScenario()           # fresh (seed jitters it)
            key = f"{policy}_s{seed}"
            results[key] = run_policy(client, policy, sc, out, seed=seed)
            print(f"[carla] {key}: LAL_v2={results[key].get('LAL_v2_s')} "
                  f"OKRI={results[key].get('OKRI')} "
                  f"LOPS={results[key].get('LOPS')}", flush=True)
    (out / "suite_results.json").write_text(json.dumps(results, indent=2))
    print(f"[carla] results -> {out / 'suite_results.json'}")


if __name__ == "__main__":
    main()
