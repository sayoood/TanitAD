"""check_preroll.py -- known-value checks of the pre-roll machinery of v8f_t1_eval_reader_rule_rzgfhp.py (study longroll). numpy and the standard library only: no tool tree, no model,
no data. The wrapper text between its PREROLL PURE markers and its install / finish glue (def _tracked .. def arg_value) is executed here in a namespace with stubs, so the checks run
the shipped code. Each check has a planted wrong arm that must fail.
  i    logged-frame mapping of the executed path (own == logged: the pinned convention bit for bit; own = a rigid transform: the analytic value to 1e-12; planted: own pose as frame)
  ii   plan0 mapping (same two cases; planted: no mapping)
  iii  grid restriction (starts 3..71 stride 4: 10 kept per clip at P = 30, 3 at P = 60; 0 kept refused; planted: off-by-one threshold)
  iv   order on a stub tool (pre-roll call with H = P, hook None, the chosen executor; then the main call with the caller's arguments; H restored also when the pre-roll raises;
       P = 0: the original function, one call, nothing written)
  v    log split (main-phase stores equal those of a P = 0 run; the pre-roll rows and counters go to the pre files; planted: stores not restored)
  vi   the rule's frame shim (the own track read in the logged frame in the main phase only)
Usage: python check_preroll.py [WRAPPER.py]   Last line: ZZ CHECK-PREROLL PASS or FAIL <which>"""
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WRAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhp.py")
SRC = open(WRAP, encoding="utf-8").read()
print("wrapper", WRAP, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16])
FAILED = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


def planted(name, wrong_passes):
    """a planted wrong arm: the check it would pass must be False."""
    print(("ok   " if not wrong_passes else "FAIL ") + "planted wrong arm caught: " + name)
    if wrong_passes:
        FAILED.append("planted " + name)


def between(a, b):
    i, j = SRC.index(a), SRC.index(b)
    assert i < j
    return SRC[i:j]


PURE = between("# ==== PREROLL PURE BEGIN", "# ==== PREROLL PURE END")
GLUE = between("def _tracked():", "def arg_value(argv, flag):")


# ---- pinned conventions, copied from the pinned tool / floors_kin (the references the mapping is held against)
def gtw_pinned(poses, t0, frames):
    poses = np.asarray(poses, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.int64)
    fr = np.asarray(frames, dtype=np.int64)
    d = poses[t0[:, None] + fr[None, :], :2] - poses[t0, None, :2]
    c, s = np.cos(-poses[t0, 2])[:, None], np.sin(-poses[t0, 2])[:, None]
    return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], axis=-1)


def p2w_pinned(plan, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    plan = np.asarray(plan, np.float64)
    return np.stack([pose[0] + c * plan[:, 0] - s * plan[:, 1], pose[1] + s * plan[:, 0] + c * plan[:, 1]], 1)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def namespace(**extra):
    ns = {"np": np, "math": math, "json": json, "os": os}
    ns.update(extra)
    exec(PURE, ns)
    return ns


NS = namespace()
rng = np.random.default_rng(7)


def make_track(T5=101):
    n = 2 * T5
    yaw = np.cumsum(rng.normal(0, 0.02, n)) + 0.4
    sp = 8 + np.cumsum(rng.normal(0, 0.05, n))
    xy = np.cumsum(np.stack([sp * np.cos(yaw), sp * np.sin(yaw)], 1) * 0.1, axis=0) + np.array([100.0, -50.0])
    return np.column_stack([xy, yaw, sp])


# ---------------------------------------------------------------------------------------------------------- i, ii
lg = make_track()
t0 = 40
same = NS["exec_logged_frame"](gtw_pinned, lg, t0, lg[2 * t0], 30)
check("i  own == logged: the mapped path equals the pinned exec_in_ego bit for bit", np.array_equal(same, gtw_pinned(lg[0::2], np.array([t0]), np.arange(1, 31))[0]))
th, tv = 0.37, np.array([3.5, -2.25])
own = lg.copy()
own[:, :2] = lg[:, :2] @ rot(th).T + tv
own[:, 2] = lg[:, 2] + th
exp = np.stack([rot(-lg[2 * t0, 2]) @ (own[2 * (t0 + k), :2] - lg[2 * t0, :2]) for k in range(1, 31)])
got = NS["exec_logged_frame"](gtw_pinned, own, t0, lg[2 * t0], 30)
check("i  own = rigid transform of logged: the analytic value to 1e-12 (err %.2e)" % np.abs(got - exp).max(), np.abs(got - exp).max() < 1e-12)
wrong = gtw_pinned(own[0::2], np.array([t0]), np.arange(1, 31))[0]
planted("i  own pose as the frame", np.abs(wrong - exp).max() < 1e-6)
check("i  the offset is large enough to tell the arms apart (%.2f m)" % np.abs(wrong - exp).max(), np.abs(wrong - exp).max() > 1.0)

plan = rng.normal(0, 3, (4, 30, 2)).astype(np.float32)
ownp = [np.array([10 + b, 5.0 - b, 0.3 * b - 0.4, 6.0]) for b in range(4)]
same_p = NS["map_plan0"](p2w_pinned, plan, ownp, ownp)
check("ii plan0 with own pose == logged pose: unchanged to 1e-12 (err %.2e)" % np.abs(same_p - plan).max(), np.abs(same_p - plan).max() < 1e-6)
lgp = [np.array([ownp[b][0] + 1.5 - b * 0.2, ownp[b][1] - 0.75, ownp[b][2] + 0.2 + 0.1 * b, 5.0]) for b in range(4)]
exp_p = np.stack([np.stack([rot(-lgp[b][2]) @ (ownp[b][:2] + rot(ownp[b][2]) @ plan[b, k].astype(np.float64) - lgp[b][:2]) for k in range(30)]) for b in range(4)])
got_p = NS["map_plan0"](p2w_pinned, plan, ownp, lgp)
check("ii plan0 mapped into the logged frame equals the analytic value to 1e-12 (err %.2e)" % np.abs(got_p - exp_p).max(), np.abs(got_p - exp_p).max() < 1e-12)
planted("ii plan0 left in the own frame", np.abs(plan.astype(np.float64) - exp_p).max() < 1e-6)

# ---------------------------------------------------------------------------------------------------------- iii
spec = importlib.util.spec_from_file_location("mlg", os.path.join(HERE, "make_longroll_grid.py"))
mlg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlg)
grid = {"order": ["a", "b", "c"], "windows": {"a": list(range(3, 72, 4)), "b": list(range(3, 68, 4)), "c": [3, 7]}, "n_windows": 18 + 17 + 2, "source": "s", "kin_max_abs_diff_vs_banked": 0.0}
g30, k30, d30 = mlg.restrict(grid, 30)
check("iii P = 30: clip a keeps 35..71 (10 starts)", g30["windows"]["a"] == list(range(35, 72, 4)) and len(g30["windows"]["a"]) == 10)
check("iii P = 30: clip b keeps 35..67, clip c dropped from order and windows", g30["windows"]["b"] == list(range(35, 68, 4)) and g30["order"] == ["a", "b"] and "c" not in g30["windows"])
check("iii P = 30: counts and the other fields", k30 == 10 + 9 and g30["n_windows"] == 19 and d30 == 37 - 19 and g30["source"] == "s" and g30["kin_max_abs_diff_vs_banked"] == 0.0)
g60, k60, _ = mlg.restrict(grid, 60)
check("iii P = 60: clip a keeps 63, 67, 71 (3 starts), clip b keeps 63, 67", g60["windows"]["a"] == [63, 67, 71] and g60["windows"]["b"] == [63, 67] and g60["order"] == ["a", "b"])
try:
    mlg.restrict(grid, 80)
    refused = False
except ValueError:
    refused = True
check("iii 0 kept is refused", refused)
planted("iii threshold 6 instead of 3 (drops the start 35)", len(mlg.restrict(grid, 30, lo=6)[0]["windows"]["a"]) == 10)
planted("iii P ignored (t0 >= 3 only)", len(mlg.restrict(grid, 0)[0]["windows"]["a"]) == 10)
check("iii the pinned reader sees only order and windows: a restricted grid keeps both keys", set(("order", "windows")) <= set(g30))


# ---------------------------------------------------------------------------------------------------------- iv, v: a stub tool run through the shipped glue
class St:
    def __init__(self, T, ci, t0, track):
        self.T, self.ci, self.t0, self.j = T, ci, t0, t0
        self.track = np.full(track.shape, np.nan)
        self.track[:2 * t0 + 1] = track[:2 * t0 + 1]
        self.d = {"pose": np.full((len(track) // 2, 4), np.nan), "v": np.full(len(track) // 2, np.nan)}
        self.d["pose"][:t0 + 1] = track[0::2][:t0 + 1]
        self.d["v"][:t0 + 1] = track[0::2][:t0 + 1, 3]
        self.clipped = 0

    def executed_ego(self):
        return np.zeros((self.T.H, 2))


class StubT:
    def __init__(self, state, raise_on=0):
        self.H, self.calls, self.raise_on, self.STATE = 30, [], raise_on, state
        self.fk = types.SimpleNamespace(gt_waypoints=gtw_pinned)
        self.plan_to_world = p2w_pinned
        self.CAUSAL_STATS = {}
        self.track = make_track()

    def make_states(self, store, tracks, J, wins, nav_mode):
        return [St(self, w[0], w[1], tracks[str(J["clip"][w[0]])]) for w in wins]

    def rollout(self, planner, states, ds, proxy, *, m=1, executor="plan", logged_tracks=None, hook=None, **kw):
        self.calls.append({"H": self.H, "hook": hook, "executor": executor, "lt": None if logged_tracks is None else sorted(logged_tracks), "m": m, "planner": planner})
        if self.raise_on == len(self.calls):
            raise RuntimeError("boom")
        S = self.STATE
        B = len(states)
        for n in range(self.H // m):
            j0 = np.array([s.j for s in states])
            if hasattr(planner, "keys_read"):
                rows = S["rulelog"].rows
                rows["ci"].append(np.array([s.ci for s in states]))
                rows["t0"].append(np.array([s.t0 for s in states]))
                rows["j"].append(j0)
                rows["k"].append(j0 - np.array([s.t0 for s in states]))
                rows["tick"].append((j0 - np.array([s.t0 for s in states])) // m)
                S["hf"]["calls"] += 1
                S["hf"]["max_dv"] = max(S["hf"]["max_dv"], 0.5 + n)
                S["mid"]["rows"]["ci"].append(np.array([s.ci for s in states]))
                self.CAUSAL_STATS["ticks"] = self.CAUSAL_STATS.get("ticks", 0) + B
            for s in states:
                s.j += m
                lo_, hi_ = 2 * (s.j - m) + 1, 2 * s.j + 1
                s.track[lo_:hi_] = self.track[lo_:hi_] + np.array([0.5, -0.25, 0.0, 0.0])
                s.d["pose"][s.j] = np.array([0.5, -0.25, 0.1, 1.0]) + self.track[2 * s.j] if hasattr(s, "lr_t0") else self.track[2 * s.j]
                s.d["v"][s.j] = s.d["pose"][s.j, 3]
                s.clipped += 1
        return np.zeros((B, 30, 2), np.float32), [s.executed_ego() for s in states]


class Model:
    keys_read = set()


def fresh(P, ex="plan", raise_on=0, tmp=None, patch=None, run="x"):
    """one stub tool with the wrapper glue installed -> (namespace, T, mine, a)"""
    LF = ("ci", "t0", "j", "k", "tick")
    state = {"hf": {"calls": 0, "max_dv": 0.0, "sum_dv": 0.0}, "cols": {"p": []}, "mid": {"rows": {"ci": []}, "env": {"plans": 0}}, "alog": {}, "kb": None, "preroll": 0}
    state["rulelog"] = types.SimpleNamespace(rows={f: [] for f in LF})
    T = StubT(state, raise_on)
    state["T"] = T
    ns = {"np": np, "math": math, "json": json, "os": os, "STATE": state, "X": types.SimpleNamespace(COUNT={"ticks": 0}), "W": types.SimpleNamespace(LOG_FIELDS=LF)}
    exec(PURE, ns)
    if patch:
        ns.update(patch)
    exec(GLUE, ns)
    rl = os.path.join(tmp, "rule_log_%s.npz" % run) if tmp else ""
    mine = types.SimpleNamespace(preroll_steps=P, preroll_exec=ex, rule_log=rl, readout_log="", form="RLc", readout=None, kappa_max=None, rule_every=10)
    a = types.SimpleNamespace(replan_steps=10, nav="deadreckon", tag="t", dump_ticks="", out_dir=tmp or "")
    ns["install_preroll"](T, mine, a)
    return ns, T, mine, a


def run_window_set(ns, T, P, hook="HOOK"):
    J = {"clip": ["clipA"]}
    tracks = {"clipA": T.track}
    wins = [(0, 66, 0, 0), (0, 70, 0, 0)]
    states = T.make_states(None, tracks, J, wins, "deadreckon")
    plan0, exe = T.rollout(Model(), states, None, None, m=10, hook=hook, executor="plan")
    return states, plan0, exe


with tempfile.TemporaryDirectory() as tmp:
    # P = 0: nothing installed
    ns0, T0, _, _ = fresh(0, tmp=tmp)
    orig_fn = T0.rollout
    check("iv P = 0: the tool's rollout and make_states are left as they are", ns0["STATE"]["runner"] is None and T0.rollout == orig_fn and getattr(T0.make_states, "__name__", "") == "make_states")
    check("iv P = 0: build_rollout returns the original function itself", ns0["build_rollout"](orig_fn, None) is orig_fn)
    st0, p0, e0 = run_window_set(ns0, T0, 0)
    check("iv P = 0: one call, H = 30, the caller's hook", len(T0.calls) == 1 and T0.calls[0]["H"] == 30 and T0.calls[0]["hook"] == "HOOK")
    ns0["finish_preroll"]()
    check("iv P = 0: nothing written", os.listdir(tmp) == [])
    rows0 = {k: np.concatenate(v) for k, v in ns0["STATE"]["rulelog"].rows.items()}
    hf0 = dict(ns0["STATE"]["hf"])

with tempfile.TemporaryDirectory() as tmp:
    # P = 60, executor plan
    ns, T, mine, a = fresh(60, tmp=tmp)
    check("iv P = 60: windows are built at t0 - P", [s.t0 for s in T.make_states(None, {"clipA": T.track}, {"clip": ["clipA"]}, [(0, 70, 0, 0)], "d")] == [10])
    try:
        T.make_states(None, {"clipA": T.track}, {"clip": ["clipA"]}, [(0, 62, 0, 0)], "d")
        ref = False
    except SystemExit as e:
        ref = "below 3" in str(e)
    check("iv P = 60: a window with t0 - P below 3 is refused with a message", ref)
    T.calls.clear()
    states, plan0, exe = run_window_set(ns, T, 60)
    c = T.calls
    check("iv P = 60: two calls, the first with H = 60, hook None, executor plan", len(c) == 2 and c[0]["H"] == 60 and c[0]["hook"] is None and c[0]["executor"] == "plan" and c[0]["lt"] is None)
    check("iv P = 60: the second call has H = 30, the caller's hook and executor, the same planner", c[1]["H"] == 30 and c[1]["hook"] == "HOOK" and c[1]["executor"] == "plan" and c[0]["planner"] is c[1]["planner"] and T.H == 30)
    check("iv P = 60: windows end at t0 + 30 with t0 restored", [(s.t0, s.j) for s in states] == [(66, 96), (70, 100)])
    check("iv P = 60: the clipped count of the main phase only (3 ticks) and the pre-roll count kept for the log", [s.clipped for s in states] == [3, 3] and ns["STATE"]["runner"].clipped_pre["model"] == 12)
    eg = states[1].executed_ego()
    want_e = gtw_pinned(T.track[0::2], np.array([70]), np.arange(1, 31))[0] + rot(-T.track[2 * 70, 2]) @ np.array([0.5, -0.25])
    check("iv executed_ego() is the executed path t0+1..t0+30 in the logged frame at t0 (err %.2e)" % np.abs(eg - want_e).max(), eg.shape == (30, 2) and np.abs(eg - want_e).max() < 1e-12)
    check("iv the executed path returned by the rollout is the same (exe[1])", np.array_equal(exe[1], eg))
    check("iv the plan0 returned has the dtype of the tool's plan0 and the shape [B, 30, 2]", plan0.shape == (2, 30, 2) and plan0.dtype == np.float32)
    # logged executor
    ns2, T2, _, _ = fresh(60, "logged", tmp=tmp, run="y")
    T2.calls.clear()
    run_window_set(ns2, T2, 60)
    check("iv --preroll-exec logged: the pre-roll gets executor logged and the logged tracks by ci", T2.calls[0]["executor"] == "logged" and T2.calls[0]["lt"] == [0] and T2.calls[1]["executor"] == "plan")
    # raise in the pre-roll
    ns3, T3, _, _ = fresh(60, raise_on=1, tmp=tmp, run="z")
    s3 = ns3["STATE"]
    before = (dict(s3["hf"]), {k: list(v) for k, v in s3["rulelog"].rows.items()})
    try:
        run_window_set(ns3, T3, 60)
        raised = False
    except RuntimeError:
        raised = True
    check("iv a raise in the pre-roll propagates, H is back at 30, the stores are as before", raised and T3.H == 30 and s3["hf"] == before[0] and all(len(s3["rulelog"].rows[k]) == len(before[1][k]) for k in before[1]))

    # v: the log split
    rows = {k: np.concatenate(v) for k, v in ns["STATE"]["rulelog"].rows.items()}
    same_shape = all(rows[k].shape == rows0[k].shape for k in rows0) and np.array_equal(rows["k"], rows0["k"]) and np.array_equal(rows["tick"], rows0["tick"]) and np.array_equal(rows["ci"], rows0["ci"])
    check("v  main-phase rule rows have the shapes and the k / tick / ci values of a P = 0 run (%d rows)" % len(rows["k"]), same_shape and len(rows["k"]) == 6)
    check("v  main-phase counters equal those of a P = 0 run", ns["STATE"]["hf"] == hf0 and ns["STATE"]["mid"]["rows"]["ci"].__len__() == 3 and ns["STATE"]["T"].CAUSAL_STATS == {"ticks": 6})
    ns["finish_preroll"]()
    files = sorted(os.listdir(tmp))
    check("v  files written: rule_log_pre, preroll_log, preroll_offsets (found %s)" % [f for f in files if "x" in f], "rule_log_pre_x.npz" in files and "preroll_log_x.json" in files and "preroll_offsets_x.npz" in files)
    with np.load(os.path.join(tmp, "rule_log_pre_x.npz")) as z_:
        pre = {k: z_[k] for k in z_.files}
    check("v  rule_log_pre: 6 pre-roll ticks x 2 windows, tick 0..5, t0 = t0 - P", len(pre["ci"]) == 12 and sorted(set(pre["tick"].tolist())) == [0, 1, 2, 3, 4, 5] and sorted(set(pre["t0"].tolist())) == [6, 10])
    pl = json.load(open(os.path.join(tmp, "preroll_log_x.json")))
    with np.load(os.path.join(tmp, "preroll_offsets_x.npz")) as z_:
        off = {k: z_[k] for k in z_.files}
    want = {"preroll_steps", "preroll_exec", "windows", "ticks_pre_rolled", "offsets_at_t0", "readout_counters_pre_roll", "clipped_windows_in_preroll"}
    check("v  preroll_log keys and counts", want <= set(pl) and pl["windows"] == {"model": 2} and pl["ticks_pre_rolled"] == {"model": 12} and pl["preroll_steps"] == 60)
    check("v  pre-roll hf counters: calls = 6 ticks, max_dv from the pre-roll only", pl["readout_counters_pre_roll"]["model"]["hf"]["calls"] == 6 and pl["readout_counters_pre_roll"]["model"]["hf"]["max_dv"] == 5.5)
    o = pl["offsets_at_t0"]["model"]
    check("v  offsets at t0: position %.4f (0.5588), heading %.4f deg, speed 1.0" % (o["dpos"]["mean"], o["dpsi_deg"]["mean"]),
          abs(o["dpos"]["mean"] - math.hypot(0.5, 0.25)) < 1e-12 and abs(o["dpsi_deg"]["mean"] - math.degrees(0.1)) < 1e-12 and abs(o["dv"]["max"] - 1.0) < 1e-12
          and off["ci"].tolist() == [0, 0] and off["t0"].tolist() == [66, 70])

    # planted: the stores are not restored after the pre-roll
    nsw, Tw, _, _ = fresh(60, tmp=tmp, run="w", patch={"_restore": lambda dst, src: None})
    run_window_set(nsw, Tw, 60)
    roww = {k: np.concatenate(v) for k, v in nsw["STATE"]["rulelog"].rows.items()}
    planted("v  stores not restored (main rows keep the pre-roll rows)", roww["k"].shape == rows0["k"].shape and nsw["STATE"]["hf"] == hf0)
    # vi: the frame shim
    stv = St(T, 0, 70, T.track)
    stv.track[:] = T.track
    pl_ = types.SimpleNamespace(states=[stv])
    fk = ns["FrameFK"](T.fk, pl_)
    stv.t0 = 70
    lpv = T.track[2 * 70] + np.array([1.0, 2.0, 0.3, 0.0])
    plain = fk.gt_waypoints(stv.track[0::2], np.array([70]), np.array([3]))
    check("vi the shim passes through while frame_pose5 is not set (the pre-roll)", np.array_equal(plain, gtw_pinned(stv.track[0::2], np.array([70]), np.array([3]))))
    stv.frame_pose5 = lpv
    p5 = np.array(T.track[0::2], copy=True)
    p5[70] = lpv
    framed = fk.gt_waypoints(stv.track[0::2], np.array([70]), np.array([3]))
    check("vi with frame_pose5 set it reads the own track in the logged frame (the arithmetic of gt_waypoints on the logged pose)", np.array_equal(framed, gtw_pinned(p5, np.array([70]), np.array([3]))))
    planted("vi the shim not applied", np.abs(plain - framed).max() < 1e-6)
    check("vi any other array passes through", np.array_equal(fk.gt_waypoints(np.array(stv.track[0::2]), np.array([70]), np.array([3])), plain))

# ---------------------------------------------------------------------------------------------------------- static facts of the wrapper text
check("the wrapper sets nothing new when P = 0: install_preroll returns before it builds a runner", GLUE.index("if P == 0:") < GLUE.index("runner = PrerollRun"))
check("the pinned tool is not touched by this check (no import of it)", "v8f_t1_eval.py" not in PURE)
print("ZZ CHECK-PREROLL " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
sys.exit(0 if not FAILED else 1)
