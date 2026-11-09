"""PREREG_REWARD_2 AMENDMENT 1: the F class from the log and the state build only, before any rollout.  usage: presim_v6.py OUT.json fresh|used --workers N
fresh = the pinned REWARD-2 list (exists only after the chain's draw); used = the 39 eligible seed-20261110 states (instrument check).  No rollout, no reward, no return."""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # the chain runs python -I

WINDOW_CYCLES, CYCLE_STEPS = 17, 5
MIN_FREE_V0 = 1.0
HALF_WIDTH, MIN_L, RANGE_S, NEAR_M, MOVER_MPS = 1.5, 20.0, 2.5, 60.0, 0.3
TTC_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _local(ego, p):
    c, s = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading)
    dx, dy = p[0] - ego.rear_axle.x, p[1] - ego.rear_axle.y
    return dx * c + dy * s


def detect(cl, ego, objs):
    """(lead, conflict) at one logged pose: the reward evaluator's geometry (gym_env cycle), with no ego rollout."""
    from shapely import affinity
    from shapely.geometry import Point
    v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    s_end = float(cl.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
    ahead = cl.substring(s_end, min(s_end + max(RANGE_S * v, MIN_L), cl.length)).buffer(HALF_WIDTH, cap_style=2)
    near = [o for o in objs if (o.center.x - ego.center.x) ** 2 + (o.center.y - ego.center.y) ** 2 < NEAR_M ** 2]
    lead = any(_local(ego, (o.center.x, o.center.y)) > 0 and ahead.intersects(o.box.geometry) for o in near)
    conflict = False
    for o in near:
        ov = getattr(o, "velocity", None)
        if ov is None or math.hypot(ov.x, ov.y) < MOVER_MPS or ahead.intersects(o.box.geometry):
            continue
        if any(ahead.intersects(affinity.translate(o.box.geometry, ov.x * t, ov.y * t)) for t in TTC_GRID):
            conflict = True
            break
    return lead, conflict


def is_free(v0, flags):
    """flags: [(lead, conflict), ...] over the window; pure so a test can mutate it"""
    return bool(v0 >= MIN_FREE_V0 and flags and not any(l or c for l, c in flags))


def classify(gs):
    """gs: a built gym_env.GymState (no step taken)."""
    sc, cl = gs.sc, gs.evaluator.cl
    n_it = int(sc.get_number_of_iterations())
    its = [k for k in range(CYCLE_STEPS, CYCLE_STEPS * WINDOW_CYCLES + 1, CYCLE_STEPS) if k < n_it]
    flags = [detect(cl, sc.get_ego_state_at_iteration(k), list(sc.get_tracked_objects_at_iteration(k).tracked_objects.tracked_objects)) for k in its]
    v0 = float(gs.ego0.dynamic_car_state.rear_axle_velocity_2d.x)
    return dict(free_presim=is_free(v0, flags), v0=v0, n_window_iterations=len(its), n_lead_its=sum(l for l, _ in flags), n_conflict_its=sum(c for _, c in flags), limit=float(gs.limit))


def do_state(args):
    idx, st, curve_min_seg_m = args
    try:
        import gym_env as g
        return dict(idx=idx, **classify(g.GymState(st, curve_min_seg_m=curve_min_seg_m)))
    except Exception as e:
        return dict(idx=idx, error=repr(e), free_presim=None)


def main():
    import ramguard
    import reward1 as r1
    import reward2 as r2
    out_path, which = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1])
    assert which in ("fresh", "used")
    if which == "fresh":
        assert os.environ.get("MGYM6_DRY") != "1" or os.path.exists(r2.STATES_V6), "no pinned list in a dry run"
        sts = r2.load_states_v6()
        todo = list(range(len(sts)))
    else:
        import r4_v6
        sts = r1.load_states_v5()
        todo = list(r4_v6.eligible_idx())
    rows, t0 = {}, time.time()
    def on_result(r, task):
        rows[r["idx"]] = r
        print(time.strftime("%H:%M:%S"), "state", r["idx"], "free" if r.get("free_presim") else ("ERROR" if r.get("error") else "not free"), flush=True)
    ramguard.run_tasks(do_state, [(i, sts[i], r2.CURVE_MIN_SEG_M) for i in todo], workers, on_result, label_fn=lambda t: f"presim state {t[0]}", log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    res = dict(prereg="PREREG_REWARD_2_AMENDMENT_1.md", which=which, window_cycles=WINDOW_CYCLES, n=len(rows), n_failed=sum(1 for r in rows.values() if r.get("error")), n_free=sum(1 for r in rows.values() if r.get("free_presim")),
               wall_s=time.time() - t0, states={str(k): v for k, v in sorted(rows.items())})
    assert not os.path.exists(out_path) or which == "used", "free_presim.json is written once"
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
    print("PRESIM_DONE", json.dumps({k: res[k] for k in ("which", "n", "n_failed", "n_free")}), flush=True)


if __name__ == "__main__":
    main()
