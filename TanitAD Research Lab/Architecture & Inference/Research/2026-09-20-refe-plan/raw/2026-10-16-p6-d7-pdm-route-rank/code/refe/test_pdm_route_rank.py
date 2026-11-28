"""test_pdm_route_rank.py -- D7: a lane-rank twin is scored on ITS OWN route (pdm_route_rank.py + the Pool initializer / guard in build_metric_cache.py), with FAKES
(no NAVSIM, no nuPlan, no DB).

    python test_pdm_route_rank.py        -> ZZTEST_PASS <n>        (the build_metric_cache half needs numpy and is skipped, loudly, without it: run it on the pod)

Pins 2026-10-16: P6's 32,929 rank-1 twin parts were byte-identical to the logged route because (H1) the route patch lived in the PARENT only and `spawn` workers start fresh, and
(H2) PDM-Closed never calls DriveRL's `_choose_route_edge`.  Every arm marked REGRESSION is a deliberate regression that must make its check FAIL."""
import math
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import pdm_route_rank as R  # noqa: E402

n = 0
SKIPPED = []


def check(cond, what):
    global n
    if not cond:
        print("FAIL:", what)
        sys.exit(1)
    n += 1


def raises(fn, exc=SystemExit, text=None):
    try:
        fn()
    except exc as e:
        return text is None or text in str(e)
    return False


# ------------------------------------------------------------------------------------------------ fakes
class Lane:
    def __init__(self, id_, coords, block):
        self.id, self.coords, self.block = id_, coords, block

    def get_roadblock_id(self):
        return self.block


class Block:
    def __init__(self, id_, lanes):
        self.id, self.interior_edges = id_, lanes


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class FrameRow:                                            # field-for-field DriveRL's frozen dataclass
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_M():
    """DriveRL's score, verbatim (driverl_runtime_map_features.py:801-815): distance to the nearest centreline point + 5 * |heading error| of the chord."""
    M = types.ModuleType("nuplan.planning.script.driverl_runtime_map_features")
    M.FrameRow = FrameRow
    M._edge_centerline_coords = lambda e: list(e.coords)

    def heading(e):
        c = M._edge_centerline_coords(e)
        return 0.0 if len(c) < 2 else math.atan2(c[-1][1] - c[0][1], c[-1][0] - c[0][0])

    def score(e, anchor):
        c = M._edge_centerline_coords(e)
        if not c:
            return float("inf")
        return min(math.hypot(x - anchor.x, y - anchor.y) for x, y in c) + 5.0 * abs(wrap(heading(e) - anchor.yaw))
    M._route_edge_anchor_score = score

    def choose(edges, anchor, prev_edge):                  # their rule at prev_edge=None
        cands = [e for e in edges if M._edge_centerline_coords(e)]
        return min(cands, key=lambda e: score(e, anchor)) if cands else None
    M._choose_route_edge = choose
    return M


class Planner:                                             # AbstractPDMPlanner
    def __init__(self, lane0, blocks):
        self._lane0, self._route_roadblock_dict = lane0, {b.id: b for b in blocks}

    def _get_starting_lane(self, ego_state):
        return self._lane0


class Closed(Planner):                                     # PDMClosedPlanner: does NOT override
    pass


def ego(x=0.0, y=0.0, yaw=0.0):
    dyn = types.SimpleNamespace(rear_axle_velocity_2d=types.SimpleNamespace(x=5.0, y=0.0), rear_axle_acceleration_2d=types.SimpleNamespace(x=0.0, y=0.0), angular_velocity=0.0)
    return types.SimpleNamespace(time_us=123, rear_axle=types.SimpleNamespace(x=x, y=y, heading=yaw), center=types.SimpleNamespace(x=x + 1.4, y=y, heading=yaw),
                                 dynamic_car_state=dyn)


def straight(id_, y, block, x0=-5.0, x1=60.0):
    return Lane(id_, [(x0 + i * (x1 - x0) / 13, y) for i in range(14)], block)


def world(k_lanes=3):
    """A roadblock B0 with `k_lanes` parallel straight lanes at y = 0, 3.5, 7, ... (ego on y=0.4, heading 0) plus a second roadblock B1 whose lane must never be picked."""
    lanes = [straight(f"L{i}", 3.5 * i, "B0") for i in range(k_lanes)]
    b0, b1 = Block("B0", lanes), Block("B1", [straight("X", 0.0, "B1", 70, 130)])
    return lanes, b0, b1


def fresh(monkey_lane0_idx=0, k_lanes=3):
    """-> (Planner class stand-in, M, planner object).  Resets the class so each case starts from NAVSIM's own method."""
    lanes, b0, b1 = world(k_lanes)

    cls = type("AbstractPDMPlanner", (), {"_get_starting_lane": Planner._get_starting_lane, "__init__": Planner.__init__})
    obj = cls(lanes[monkey_lane0_idx], [b0, b1])
    M = fake_M()
    R._planner_cls, R._driverl = (lambda: cls), (lambda: M)
    return cls, M, obj, lanes


os.environ.pop(R.ENV, None)

# ---- the ordering is DriveRL's: nearest-first by distance + 5 * heading error, ego on y = 0.4 so L0 is nearest, then L1, then L2
cls, M, obj, lanes = fresh()
a = R.anchor_from_ego(ego(0.0, 0.4, 0.0), M)
check([e.id for e in R.ordered_lanes(lanes, a, M)] == ["L0", "L1", "L2"], "ordered nearest-first")
check(R.rank_k_lane(lanes, a, 0, M).id == "L0" and R.rank_k_lane(lanes, a, 1, M).id == "L1" and R.rank_k_lane(lanes, a, 2, M).id == "L2", "rank k -> ordered[k]")
check(R.rank_k_lane(lanes, a, 3, M).id == "L2" and R.rank_k_lane(lanes, a, 9, M).id == "L2", "rank >= n CLAMPS to the last lane, as route_lane_rank_patch.patched does")
check(R.rank_k_lane([], a, 1, M) is None and R.rank_k_lane([Lane("e", [], "B0")], a, 1, M) is None, "no lane with a baseline path -> None")
# the heading term decides: a nearer lane pointing the wrong way ranks AFTER a farther aligned one
wrong = Lane("W", [(60.0, 0.6), (-5.0, 0.6)], "B0")           # 0.2 m away but reversed (heading error pi -> +15.7)
check([e.id for e in R.ordered_lanes([wrong] + lanes, a, M)] == ["L0", "L1", "L2", "W"], "the 5x heading-error term is part of the order (a reversed lane is last)")
# the anchor is the REAR AXLE, not the centre (ego.center is 1.4 m ahead)
ax = R.anchor_from_ego(ego(3.0, 0.4, 0.2), M)
check((ax.x, ax.y, ax.yaw, ax.timestamp_us, ax.vx, ax.token_hex) == (3.0, 0.4, 0.2, 123, 5.0, "current"), "anchor == DriveRL _frame_from_ego_state (rear axle x, y, heading)")

# ---- rank 0 installs NOTHING, by construction
cls, M, obj, lanes = fresh()
orig = cls._get_starting_lane
msg = R.apply(0)
check(cls._get_starting_lane is orig and R.installed_rank(cls) is None and "no patch installed" in msg, "rank 0: NAVSIM's own method, untouched")
check(obj._get_starting_lane(ego(0, 0.4)).id == "L0", "rank 0 behaviour == NAVSIM's (lane0)")
R.assert_active(0, obj)
check(True, "assert_active(0) on an unpatched object")
# rank 0 refuses a stale patch
cls, M, obj, lanes = fresh()
R.apply(1)
check(raises(lambda: R.apply(0), SystemExit, "ALREADY installed"), "REGRESSION: rank 0 requested while a rank-1 patch is installed -> refused")
check(raises(lambda: R.assert_active(0, obj), SystemExit, "patched"), "assert_active(0) on a patched class -> refused")

# ---- rank k picks ordered[min(k, n-1)] of the STARTING lane's roadblock, from PDM's own lane0
for k, want in ((1, "L1"), (2, "L2"), (3, "L2"), (5, "L2")):
    cls, M, obj, lanes = fresh()
    R.ensure(k, obj)
    got = obj._get_starting_lane(ego(0, 0.4, 0.0))
    check(got.id == want, f"rank {k} -> {want}, got {got.id}")
    check(obj._d7_last["lane0"] == "L0" and obj._d7_last["lane"] == want and obj._d7_last["n_lanes"] == 3 and obj._d7_last["clamped"] == (k > 2), f"rank {k}: diagnostics")
    check(got.get_roadblock_id() == "B0", "the twin's lane is in the SAME roadblock as PDM's starting lane (B1's lane is never picked)")
# PDM's lane0 is the middle lane: rank is still relative to DRIVERL's ordering of that roadblock, not to lane0
cls, M, obj, lanes = fresh(monkey_lane0_idx=1)
R.ensure(1, obj)
check(obj._get_starting_lane(ego(0, 0.4)).id == "L1" and obj._d7_last["lane0_pos"] == 1, "PDM lane0 == DriveRL ordered[1]: the twin's lane equals PDM's own (reported as lane0_pos)")
# a single-lane roadblock clamps to itself
cls, M, obj, lanes = fresh(k_lanes=1)
R.ensure(2, obj)
check(obj._get_starting_lane(ego(0, 0.4)).id == "L0" and obj._d7_last["clamped"] is True, "1-lane roadblock: rank 2 clamps to the only lane")

# ---- one process serves ONE rank; idempotent
cls, M, obj, lanes = fresh()
R.apply(1)
check("already active" in R.apply(1), "apply(1) twice is idempotent")
check(raises(lambda: R.apply(2), SystemExit, "ONE rank"), "REGRESSION: a second rank in one process -> refused")

# ---- env var: a mismatch is refused (as route_rank.apply does), a match is fine
cls, M, obj, lanes = fresh()
os.environ[R.ENV] = "2"
check(raises(lambda: R.apply(1), SystemExit, "label it another"), "env names rank 2 but rank 1 requested -> refused")
check(R.installed_rank(cls) is None, "...and nothing was installed")
os.environ[R.ENV] = "1"
R.apply(1)
check(R.installed_rank(cls) == 1, "env == rank -> installs")
os.environ.pop(R.ENV)
check(raises(lambda: R.apply(-1), SystemExit), "negative rank refused")

# ---- failures are LOUD
cls, M, obj, lanes = fresh()
R.ensure(1, obj)
obj._lane0 = None
check(raises(lambda: obj._get_starting_lane(ego()), RuntimeError, "no starting lane"), "PDM found no starting lane -> raises (the job fails, nothing is written)")
obj._lane0 = Lane("Z", [(0, 0), (1, 0)], "NOT_IN_ROUTE")
check(raises(lambda: obj._get_starting_lane(ego()), RuntimeError, "not in its route dict"), "starting roadblock not in the route dict -> raises")
obj._lane0 = lanes[0]
obj._route_roadblock_dict["B0"].interior_edges = [Lane("e", [], "B0")]
check(raises(lambda: obj._get_starting_lane(ego()), RuntimeError, "no lane with a baseline"), "no lane with a baseline -> raises")

# ---- assert_active reads the OBJECT, not a flag
cls, M, obj, lanes = fresh()
R.apply(1)
obj2 = cls(lanes[0], [])
obj2._get_starting_lane = lambda ego_state: lanes[0]                           # an INSTANCE attribute shadows the class patch
check(raises(lambda: R.assert_active(1, obj2), SystemExit, "NOT wrapped"), "REGRESSION: an instance attribute shadowing the class patch -> refused")
check(raises(lambda: R.ensure(1, obj2), SystemExit), "ensure() refuses the shadowed object")
Sub = type("Sub", (cls,), {"_get_starting_lane": lambda self, e: lanes[0]})      # a subclass override shadows it too
check(raises(lambda: R.assert_active(1, Sub(lanes[0], [])), SystemExit, "NOT wrapped"), "REGRESSION: a subclass override -> refused")
cls._get_starting_lane = cls._get_starting_lane.__wrapped__                     # MUTATION: the patch removed
check(raises(lambda: R.assert_active(1), SystemExit, "NOT wrapped"), "MUTATION: patch removed -> assert_active(1) FAILS")

# ---- parity with the DriveRL patch the bank was made with: route_lane_rank_patch.patched == rank_k_lane, for every rank, on a sweep of anchors
import importlib  # noqa: E402
pk = None
try:
    import route_lane_rank_patch as pk
except Exception as e:                                                          # noqa: BLE001
    SKIPPED.append(f"route_lane_rank_patch parity ({type(e).__name__})")
if pk is not None:
    cls, M, obj, lanes = fresh()
    for name in ("nuplan", "nuplan.planning", "nuplan.planning.script"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["nuplan.planning.script.driverl_runtime_map_features"] = M
    sys.modules["nuplan.planning.script"].driverl_runtime_map_features = M
    pk._INSTALLED, pk._RANK = False, 0
    pk.install(1)
    pairs = 0
    for k in (0, 1, 2, 3, 4):
        pk.set_rank(k)
        for (x, y, yaw) in ((0, 0.4, 0.0), (10, 3.6, 0.05), (20, 7.2, -0.1), (5, -2.0, 0.3), (0, 100.0, 0.0)):
            a = R.anchor_from_ego(ego(x, y, yaw), M)
            want = M._choose_route_edge(lanes, a, None)                       # DriveRL's own patched function
            got = R.rank_k_lane(lanes, a, k, M)
            check(want is got, f"parity k={k} anchor=({x},{y},{yaw}): DriveRL patched -> {want.id}, ours -> {got.id}")
            pairs += 1
    check(pairs == 25, "25 (rank, anchor) pairs agree with route_lane_rank_patch.patched")

# ================================================================================================ build_metric_cache: initializer + guard (needs numpy)
try:
    import numpy  # noqa: F401
    import build_metric_cache as B
except Exception as e:                                                          # noqa: BLE001
    B = None
    SKIPPED.append(f"build_metric_cache initializer/guard ({type(e).__name__}: {e}) -- RUN THIS FILE ON THE POD")

if B is not None:
    def fake_pt(obj, fail_setup=False):
        pt = types.ModuleType("navsim_pdm_targets")

        def _setup():
            if fail_setup:
                raise ImportError("navsim is not importable")
            return {"proc": types.SimpleNamespace(_pdm_closed=obj)}
        pt._setup = _setup
        pt.metric_cache_parts = lambda sc: ("traj", "ego", "obs", ("centerline", sc), ["lane"], "dam")
        sys.modules["navsim_pdm_targets"] = pt
        return pt

    def reset():
        B._W.clear()
        os.environ.pop(R.MUTATE_ENV, None)

    def job(out, rank, toks=("t0", "t1")):
        aug = None if rank is None else {"kind": "lane_rank", "value": rank}
        return ("LOG", [(t, "null" if rank is None else __import__("json").dumps(aug, sort_keys=True)) for t in toks], out, None, None, rank)

    B._scenarios = lambda db_root, map_root, log, toks: {t: f"sc:{t}" for t in toks}

    # rank 1 twins: the initializer runs, the guard passes, files are written at the twin path
    reset()
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    with tempfile.TemporaryDirectory() as td:
        B._worker_init(1)
        check("init_err" not in B._W and B._W["rank"] == 1 and R.installed_rank(cls) == 1, "initializer(1) installed the patch and recorded the worker state")
        done, skipped, failed, msgs = B._build_log(job(td, 1))
        files = sorted(os.listdir(os.path.join(td, "LOG")))
        check((done, failed) == (2, 0) and files == ["t0__lane_rank_1.pkl.xz", "t1__lane_rank_1.pkl.xz"], f"rank-1 build writes the twin paths: {files} {msgs}")

    # rank 0 / no flag: the initializer ASSERTS no patch; logged paths
    reset()
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    with tempfile.TemporaryDirectory() as td:
        B._worker_init(0)
        check("init_err" not in B._W and B._W["rank"] == 0 and R.installed_rank(cls) is None, "initializer(0) installs NOTHING")
        done, skipped, failed, msgs = B._build_log(job(td, None))
        check((done, failed) == (2, 0) and sorted(os.listdir(os.path.join(td, "LOG"))) == ["t0.pkl.xz", "t1.pkl.xz"], "rank-0 build writes the logged paths")

    # REGRESSION: a stale patch in a rank-0 worker -> the initializer records the failure, NEVER raises, and no file is written
    reset()
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    R.apply(1)
    with tempfile.TemporaryDirectory() as td:
        try:
            B._worker_init(0)
            raised = False
        except BaseException:                                                   # noqa: BLE001
            raised = True
        check(not raised and "ALREADY installed" in B._W.get("init_err", ""), "REGRESSION: rank-0 worker with a stale patch: the initializer records the error and does not raise (a raising initializer hangs the Pool)")
        done, skipped, failed, msgs = B._build_log(job(td, None))
        check((done, failed) == (0, 2) and not os.path.exists(os.path.join(td, "LOG")) and "initializer FAILED" in msgs[0], "...and every job FAILS LOUDLY and writes nothing")

    # REGRESSION: the patch cannot be installed (navsim not importable) -> same
    reset()
    cls, M, obj, lanes = fresh()
    fake_pt(obj, fail_setup=True)
    with tempfile.TemporaryDirectory() as td:
        B._worker_init(1)
        done, skipped, failed, msgs = B._build_log(job(td, 1))
        check((done, failed) == (0, 2) and "navsim is not importable" in msgs[0] and not os.path.exists(os.path.join(td, "LOG")), "REGRESSION: the patch cannot be installed -> failed jobs, nothing written")

    # MUTATION 1: the initializer REMOVED (cmd_build would pass initializer=None): the per-job guard refuses -> the build fails loudly
    reset()
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    os.environ[R.MUTATE_ENV] = "no_initializer"
    with tempfile.TemporaryDirectory() as td:
        done, skipped, failed, msgs = B._build_log(job(td, 1))
        check((done, failed) == (0, 2) and "did not run for rank 1" in msgs[0] and not os.path.exists(os.path.join(td, "LOG")), "MUTATION: no initializer -> the guard FAILS every job (nothing written)")
    # MUTATION 2: initializer AND guard removed = P6's defect: the build "succeeds" with NO patch -- the parts are the logged route's, and ONLY the (a)/(b) checks can catch it
    os.environ[R.MUTATE_ENV] = "no_initializer_no_guard"
    with tempfile.TemporaryDirectory() as td:
        done, skipped, failed, msgs = B._build_log(job(td, 1))
        check((done, failed) == (2, 0) and R.installed_rank(cls) is None, "MUTATION: no initializer, no guard -> parts are written with NO patch (this is P6's defect; the distinctness / lane checks must catch it)")
    reset()

    # the guard re-reads the OBJECT: a patch removed mid-run is caught on the next job
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    B._worker_init(1)
    cls._get_starting_lane = cls._get_starting_lane.__wrapped__
    with tempfile.TemporaryDirectory() as td:
        done, skipped, failed, msgs = B._build_log(job(td, 1))
        check((done, failed) == (0, 2) and "does not match rank 1" in msgs[0], "MUTATION: the patch vanishes after the initializer -> the per-job guard refuses")
    reset()

    # existing parts are skipped BEFORE any scenario is built (the chain's lanes stage re-runs over a cache the standalone D7 rebuild filled)
    seen_calls = []

    def rec_scenarios(db_root, map_root, log, toks):
        seen_calls.append(sorted(toks))
        return {t: f"sc:{t}" for t in toks}
    B._scenarios = rec_scenarios
    cls, M, obj, lanes = fresh()
    fake_pt(obj)
    B._worker_init(1)
    with tempfile.TemporaryDirectory() as td:
        done, skipped, failed, msgs = B._build_log(job(td, 1, toks=("t0", "t1", "t2")))
        check((done, failed) == (3, 0) and seen_calls == [["t0", "t1", "t2"]], "first build: all three built, one scenario build")
        p0 = os.path.join(td, "LOG", "t0__lane_rank_1.pkl.xz")
        before = (os.stat(p0).st_mtime_ns, open(p0, "rb").read())
        seen_calls.clear()
        done, skipped, failed, msgs = B._build_log(job(td, 1, toks=("t0", "t1", "t2")))
        check((done, skipped, failed) == (3, 0, 0) and seen_calls == [], "all parts exist -> counted done, NO scenario built (the skip costs a stat per file)")
        os.remove(os.path.join(td, "LOG", "t1__lane_rank_1.pkl.xz"))
        done, skipped, failed, msgs = B._build_log(job(td, 1, toks=("t0", "t1", "t2")))
        check((done, failed) == (3, 0) and seen_calls == [["t1"]], "one part missing -> only that token is handed to the scenario builder")
        check((os.stat(p0).st_mtime_ns, open(p0, "rb").read()) == before, "an existing part is never rewritten")
        os.remove(os.path.join(td, "LOG", "t2__lane_rank_1.pkl.xz"))
        B._scenarios = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("db gone"))
        done, skipped, failed, msgs = B._build_log(job(td, 1, toks=("t0", "t1", "t2")))
        check((done, failed) == (2, 1) and "scenario build RuntimeError" in msgs[0], "a scenario failure fails only the MISSING token; the existing ones still count as done")
    reset()

print("ZZTEST_PASS %d" % n)
for s in SKIPPED:
    print("SKIPPED:", s)
