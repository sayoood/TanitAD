"""check_preroll_b.py -- known-value checks of v8f_t1_eval_reader_rule_rzgfhq.py (study longroll, second build: R1 first-tick adjustment, R2 segments, R3 tick-recorder frames) and re-run of the
items of check_preroll.py against its functions. numpy and the standard library only: no tool tree, no model, no data. The wrapper text between its PREROLL PURE markers, its install / finish
glue (def _tracked .. def arg_value) and the class ReaderPlanner is executed here in namespaces with stubs, so the checks run the shipped code. dist_rule.py (numpy only; found beside this file or
in ../distrule) supplies the real progress geometry. Each check has a planted wrong arm that must fail.
  i-iii   (as check_preroll.py) logged-frame mapping of the executed path, plan0 mapping, grid restriction
  iv-v    (as check_preroll.py, for the segment build) order of the calls on a stub tool, H restored, P = 0 untouched, log split of main phase and pre-roll
  vi      the rule's frame shim
  R1      a_m0: +d / -d on a straight logged path to 1e-9, exact 0.0 without frame pose / at the logged pose / for k above 0, the saturation at the ends of the path extension (1000 m);
          the adjustment reaches exactly the first distance_rule call after the planner call (the second look gets none, the next tick gets none); the main-phase value in the offsets file
  R2      P = 60: two segments with a re-base after each, then the main call; P = 45 refused; the rule-log pre file names the segment
  R3      the hook shim: own == logged leaves the recorded rows as they are; a rigid offset gives pose_w0 = logged pose and s_ckpt_w0 = the analytic value to 1e-12
Usage: python check_preroll_b.py [WRAPPER.py]   Last line: ZZ CHECK-PREROLL-B PASS or FAIL <which>"""
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
WRAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py")
SRC = open(WRAP, encoding="utf-8").read()
print("wrapper", WRAP, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16])
FAILED = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


def planted(name, wrong_passes):
    print(("ok   " if not wrong_passes else "FAIL ") + "planted wrong arm caught: " + name)
    if wrong_passes:
        FAILED.append("planted " + name)


def between(a, b):
    i, j = SRC.index(a), SRC.index(b)
    assert i < j
    return SRC[i:j]


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


PURE = between("# ==== PREROLL PURE BEGIN", "# ==== PREROLL PURE END")
GLUE = between("def _tracked():", "def arg_value(argv, flag):")
RPSRC = between("class ReaderPlanner(X.MixPlanner):", "# ==== PREROLL PURE BEGIN")
for cand in (os.path.join(HERE, "dist_rule.py"), os.path.join(HERE, "..", "distrule", "dist_rule.py")):
    if os.path.isfile(cand):
        _sp = importlib.util.spec_from_file_location("dist_rule_real", cand)
        RD = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(RD)
        break
else:
    raise SystemExit("dist_rule.py not found beside this file or in ../distrule")


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


def namespace(pure=PURE, **extra):
    ns = {"np": np, "math": math, "json": json, "os": os}
    ns.update(extra)
    exec(pure, ns)
    return ns


NS = namespace()
rng = np.random.default_rng(7)


def make_track(T5=101):
    n = 2 * T5
    yaw = np.cumsum(rng.normal(0, 0.02, n)) + 0.4
    sp = 8 + np.cumsum(rng.normal(0, 0.05, n))
    xy = np.cumsum(np.stack([sp * np.cos(yaw), sp * np.sin(yaw)], 1) * 0.1, axis=0) + np.array([100.0, -50.0])
    return np.column_stack([xy, yaw, sp])


# ---------------------------------------------------------------------------------------------------------- i, ii (as check_preroll.py)
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
check("ii plan0 with own pose == logged pose: unchanged (err %.2e)" % np.abs(same_p - plan).max(), np.abs(same_p - plan).max() < 1e-6)
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
check("iii P = 30: clip b keeps 35..67, clip c dropped", g30["windows"]["b"] == list(range(35, 68, 4)) and g30["order"] == ["a", "b"] and "c" not in g30["windows"])
check("iii P = 30: counts and the other fields", k30 == 10 + 9 and g30["n_windows"] == 19 and d30 == 37 - 19 and g30["source"] == "s" and g30["kin_max_abs_diff_vs_banked"] == 0.0)
g60, k60, _ = mlg.restrict(grid, 60)
check("iii P = 60: clip a keeps 63, 67, 71, clip b keeps 63, 67", g60["windows"]["a"] == [63, 67, 71] and g60["windows"]["b"] == [63, 67] and g60["order"] == ["a", "b"])
try:
    mlg.restrict(grid, 80)
    refused = False
except ValueError:
    refused = True
check("iii 0 kept is refused", refused)
planted("iii threshold 6 instead of 3", len(mlg.restrict(grid, 30, lo=6)[0]["windows"]["a"]) == 10)
planted("iii P ignored", len(mlg.restrict(grid, 0)[0]["windows"]["a"]) == 10)


# ---------------------------------------------------------------------------------------------------------- R1: a_m0 on a straight logged path
class S1:
    """a state at its re-base step: logged path straight along +x of the logged frame (1 m steps), logged pose lp, own position = lp + d along the logged heading + lateral e."""

    def __init__(self, lp, d, e=0.0, k=0, with_frame=True, t0=20):
        self.t0, self.j, self.ci = t0, t0 + k, 0
        own_xy = lp[:2] + rot(lp[2]) @ np.array([d, e])
        self.track = np.zeros((2 * t0 + 2, 4))
        self.track[2 * t0] = np.array([own_xy[0], own_xy[1], lp[2] + 0.3, lp[3]])
        self.d = {"gt": np.zeros((t0 + 1, 30, 2))}
        self.d["gt"][t0] = np.stack([np.arange(1, 31, dtype=np.float64), np.zeros(30)], 1)
        if with_frame:
            self.frame_pose5 = np.array(lp, copy=True)


LP = np.array([100.0, -50.0, 0.6, 8.0])
am0 = NS["a_m0_first_tick"]
dd = 4.0
vals = am0(RD, gtw_pinned, [S1(LP, dd), S1(LP, -dd), S1(LP, dd, e=0.7), S1(LP, 0.0)])
check("R1 a_m0 +d ahead / -d behind along a straight logged path = +4 / -4 to 1e-9 (%s)" % np.round(vals[:2], 12), abs(vals[0] - dd) < 1e-9 and abs(vals[1] + dd) < 1e-9)
check("R1 a lateral offset of 0.7 m does not change it (%.2e)" % abs(vals[2] - dd), abs(vals[2] - dd) < 1e-9)
check("R1 own at the logged pose (frame pose set) is exactly 0.0", vals[3] == 0.0 and not np.signbit(vals[3]) or vals[3] == 0.0)
z = am0(RD, gtw_pinned, [S1(LP, dd, with_frame=False), S1(LP, dd, k=3), S1(LP, 0.0, with_frame=False)])
check("R1 exactly 0.0 without a frame pose and for k above 0 (a state displaced by 4 m)", np.array_equal(z, np.zeros(3)) and z.dtype == np.float64)
far = am0(RD, gtw_pinned, [S1(LP, -1500.0), S1(LP, 1500.0), S1(LP, 15.0)])
check("R1 path extension: RAY = %.0f m each side; 1500 m behind saturates at -1000, 1500 m ahead at 1030 (%s), 15 m on the path is 15" % (RD.RAY, np.round(far, 6)),
      RD.RAY == 1000.0 and abs(far[0] + 1000.0) < 1e-9 and abs(far[1] - 1030.0) < 1e-9 and abs(far[2] - 15.0) < 1e-9)
bad_flip = namespace(sub(PURE, "gtw(np.stack([fp, own])", "gtw(np.stack([own, fp])"))["a_m0_first_tick"]
vb = bad_flip(RD, gtw_pinned, [S1(LP, dd), S1(LP, -dd)])
planted("R1 frames swapped (own pose as the frame)", abs(vb[0] - dd) < 1e-6 and abs(vb[1] + dd) < 1e-6)
bad_k = namespace(sub(PURE, "if fp is None or int(s.j) != int(s.t0):", "if fp is None:"))["a_m0_first_tick"]
planted("R1 k above 0 not excluded", np.array_equal(bad_k(RD, gtw_pinned, [S1(LP, dd, k=3)]), np.zeros(1)))


# ---------------------------------------------------------------------------------------------------------- R1: the shim reaches exactly the next distance_rule call
def mk_rule(calls):
    def rec_fn(plan, v0, present, g_src, g0, v_L, m=10, **kw):
        calls.append(np.array(g0, copy=True))
        return "out"
    return rec_fn


S_ = {}
calls = []
shim = NS["make_g0_shim"](mk_rule(calls), S_)
S_["g0_corr"] = np.array([4.0, 0.0, -2.0])
base = np.array([10.0, 20.0, 30.0])
shim(None, None, None, base, base, base, m=10)
shim(None, None, None, base, base, base, m=5)
check("R1 shim: the first call after the adjustment gets g0 - a_m0, the second call gets g0 unchanged, the slot is cleared", np.array_equal(calls[0], base - np.array([4.0, 0.0, -2.0])) and np.array_equal(calls[1], base) and S_["g0_corr"] is None)
check("R1 shim: stats count one consumed call, two non-zero entries, largest 4.0", S_["g0_stats"] == {"calls": 1, "nonzero": 2, "max_abs": 4.0})
badshim = namespace(sub(PURE, "            state[\"g0_corr\"] = None\n            c = np.asarray(c, np.float64)", "            c = np.asarray(c, np.float64)"))["make_g0_shim"]
calls2, S2_ = [], {}
sh2 = badshim(mk_rule(calls2), S2_)
S2_["g0_corr"] = np.array([4.0, 0.0, -2.0])
sh2(None, None, None, base, base, base)
sh2(None, None, None, base, base, base)
planted("R1 shim that does not clear (the second look would be corrected)", np.array_equal(calls2[1], base))
badsign = namespace(sub(PURE, "            g0 = g0 - c\n", "            g0 = g0 + c\n"))["make_g0_shim"]
calls3, S3_ = [], {"g0_corr": np.array([4.0, 0.0, -2.0])}
badsign(mk_rule(calls3), S3_)(None, None, None, base, base, base)
planted("R1 shim with the wrong sign", np.array_equal(calls3[0], base - np.array([4.0, 0.0, -2.0])))


def planner_world(preroll, shim_factory=None):
    """the shipped class ReaderPlanner on stubs of the parent classes; R = a stub rule module with the real progress and a recording distance_rule."""
    calls = []
    rec_fn = mk_rule(calls)
    Rn = types.SimpleNamespace(progress=RD.progress, logged_arc=RD.logged_arc, distance_rule=rec_fn)
    S = {"preroll": preroll, "alim": None, "every": 5, "form": "RLc"}
    if preroll:
        Rn.distance_rule = (shim_factory or NS["make_g0_shim"])(rec_fn, S)

    class StubRule:
        def __init__(self, base, model, mode, m, fk, label=None, log=None, floor="card"):
            self.base, self.model, self.mode, self.m, self.fk, self.label, self.log, self.floor = base, model, mode, int(m), fk, label, log, floor
            self.source, self.states, self.kw = "mix", None, {}

    class StubMix(StubRule):
        def __call__(self, batch):
            B = len(self.states)
            g0 = np.array([10.0, 20.0, 30.0][:B])
            Rn.distance_rule(np.zeros((B, 30, 2)), np.zeros(B), np.ones(B, bool), g0, g0, g0, m=self.m)
            return np.zeros((B, 30, 2))

    def mid_looks(pl, plan):
        B = len(pl.states)
        g0 = np.array([10.0, 20.0, 30.0][:B])
        Rn.distance_rule(np.zeros((B, 30, 2)), np.zeros(B), np.ones(B, bool), g0, g0, g0, m=5)
        return plan

    ns = {"np": np, "X": types.SimpleNamespace(MixPlanner=StubMix, STATE={"label": "L"}), "_RULE_PLANNER": StubRule, "STATE": S, "R": Rn, "mid_looks": mid_looks,
          "FrameFK": NS["FrameFK"], "a_m0_first_tick": NS["a_m0_first_tick"]}
    exec(RPSRC, ns)
    fk = types.SimpleNamespace(gt_waypoints=gtw_pinned)
    return ns["ReaderPlanner"](None, None, "mix", 10, fk), calls, S, ns


pl, calls_p, S_p, _ = planner_world(60)
st_a, st_b, st_c = S1(LP, dd), S1(LP, dd, with_frame=False), S1(LP, dd, k=3)
pl.states = [st_a, st_b, st_c]
pl(None)
check("R1 planner: tick-level call g0 = [10 - 4, 20, 30] (the re-based state only), the second look gets [10, 20, 30]",
      len(calls_p) == 2 and np.allclose(calls_p[0], [6.0, 20.0, 30.0], atol=1e-9, rtol=0) and np.array_equal(calls_p[1], [10.0, 20.0, 30.0]))
st_a.j = st_a.t0 + 10
pl(None)
check("R1 planner: the next tick (k = 10) gets no adjustment, and nothing is left set", np.array_equal(calls_p[2], [10.0, 20.0, 30.0]) and np.array_equal(calls_p[3], [10.0, 20.0, 30.0]) and S_p["g0_corr"] is None)
pl0, calls_0, S_0, ns_0 = planner_world(0)
ns_0["a_m0_first_tick"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached at P = 0"))
pl0.states = [S1(LP, dd)]
pl0(None)
check("R1 P = 0: no adjustment code is reached (stub raises if it is), g0 unchanged, nothing written into STATE", np.array_equal(calls_0[0], [10.0]) and "g0_corr" not in S_0 and not isinstance(pl0.fk, NS["FrameFK"]))
plb, calls_b, _, _ = planner_world(60, badshim)
plb.states = [S1(LP, dd)]
plb(None)
check("R1 planner with the shim that does not clear: planner clears after the tick-level call itself, so the second look stays plain (the planner's own guard)", np.array_equal(calls_b[1], [10.0]))

# ---------------------------------------------------------------------------------------------------------- stub tool for R2 / R3 / log split
OFF = np.array([0.5, -0.25, 0.1, 1.0])


class St:
    def __init__(self, T, ci, t0, track):
        self.T, self.ci, self.t0, self.j = T, ci, t0, t0
        self.track = np.full(track.shape, np.nan)
        self.track[:2 * t0 + 1] = track[:2 * t0 + 1]
        n5 = len(track) // 2
        self.d = {"pose": np.full((n5, 4), np.nan), "v": np.full(n5, np.nan), "gt": np.full((n5, 30, 2), np.nan)}
        for t in range(n5 - 30):
            self.d["gt"][t] = gtw_pinned(track[0::2], np.array([t]), np.arange(1, 31))[0]
        self.d["pose"][:t0 + 1] = track[0::2][:t0 + 1]
        self.d["v"][:t0 + 1] = track[0::2][:t0 + 1, 3]
        self.clipped = 0

    def executed_ego(self):
        return np.zeros((self.T.H, 2))


class StubT:
    def __init__(self, state, raise_on=0, off=OFF):
        self.H, self.calls, self.raise_on, self.STATE, self.off = 30, [], raise_on, state, np.asarray(off, np.float64)
        self.fk = types.SimpleNamespace(gt_waypoints=gtw_pinned)
        self.plan_to_world = p2w_pinned
        self.CAUSAL_STATS = {}
        self.track = make_track()

    def make_states(self, store, tracks, J, wins, nav_mode):
        return [St(self, w[0], w[1], tracks[str(J["clip"][w[0]])]) for w in wins]

    def rollout(self, planner, states, ds, proxy, *, m=1, executor="plan", logged_tracks=None, hook=None, **kw):
        self.calls.append({"H": self.H, "hook": hook, "executor": executor, "lt": None if logged_tracks is None else sorted(logged_tracks), "m": m, "planner": planner,
                           "t0s": [s.t0 for s in states], "fp": [None if getattr(s, "frame_pose5", None) is None else np.array(s.frame_pose5, copy=True) for s in states]})
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
            if callable(hook):
                hook(n, None, None, states)                  # before the commit, as the pinned rollout does
            for s in states:
                s.j += m
                lo_, hi_ = 2 * (s.j - m) + 1, 2 * s.j + 1
                s.track[lo_:hi_] = self.track[lo_:hi_] + np.array([self.off[0], self.off[1], 0.0, 0.0])
                s.d["pose"][s.j] = self.off + self.track[2 * s.j] if hasattr(s, "lr_t0") else self.track[2 * s.j]
                s.d["v"][s.j] = s.d["pose"][s.j, 3]
                s.clipped += 1
        return np.zeros((B, 30, 2), np.float32), [s.executed_ego() for s in states]


class Model:
    keys_read = set()


class Rec:
    """the part of the pinned TickRecorder that records pose_w0: one row per state per call, in state order (st.d['pose'][st.t0][:3]); raw keeps what the recorder wrote."""

    def __init__(self):
        self.rows = {"pose_w0": [], "tick": []}
        self.raw = []

    def __call__(self, n, batch, plan, states):
        for st in states:
            self.rows["pose_w0"].append(np.array(st.d["pose"][st.t0][:3], np.float64))
            self.rows["tick"].append(n)
            self.raw.append(self.rows["pose_w0"][-1].copy())


def fresh(P, ex="plan", raise_on=0, tmp=None, patch=None, run="x", pure=PURE, off=OFF, m=10):
    """one stub tool with the wrapper glue installed -> (namespace, T, mine, a, R stub)"""
    LF = ("ci", "t0", "j", "k", "tick")
    state = {"hf": {"calls": 0, "max_dv": 0.0, "sum_dv": 0.0}, "cols": {"p": []}, "mid": {"rows": {"ci": []}, "env": {"plans": 0}}, "alog": {}, "kb": None, "preroll": 0}
    state["rulelog"] = types.SimpleNamespace(rows={f: [] for f in LF})
    T = StubT(state, raise_on, off)
    state["T"] = T
    drc = []
    Rn = types.SimpleNamespace(progress=RD.progress, logged_arc=RD.logged_arc, distance_rule=mk_rule(drc))
    ns = {"np": np, "math": math, "json": json, "os": os, "STATE": state, "R": Rn, "X": types.SimpleNamespace(COUNT={"ticks": 0}), "W": types.SimpleNamespace(LOG_FIELDS=LF)}
    exec(pure, ns)
    if patch:
        ns.update(patch)
    exec(GLUE, ns)
    rl = os.path.join(tmp, "rule_log_%s.npz" % run) if tmp else ""
    mine = types.SimpleNamespace(preroll_steps=P, preroll_exec=ex, rule_log=rl, readout_log="", form="RLc", readout=None, kappa_max=None, rule_every=10)
    a = types.SimpleNamespace(replan_steps=m, nav="deadreckon", tag="t", dump_ticks="", out_dir=tmp or "")
    ns["install_preroll"](T, mine, a)
    return ns, T, mine, a, Rn


def NOOP(n, b, p, s):
    return None


def run_window_set(ns, T, P, hook=NOOP, wins=None):
    J = {"clip": ["clipA"]}
    tracks = {"clipA": T.track}
    wins = wins or [(0, 66, 0, 0), (0, 70, 0, 0)]
    states = T.make_states(None, tracks, J, wins, "deadreckon")
    plan0, exe = T.rollout(Model(), states, None, None, m=10, hook=hook, executor="plan")
    return states, plan0, exe


with tempfile.TemporaryDirectory() as tmp:
    # P = 0: nothing installed
    ns0, T0, _, _, R0 = fresh(0, tmp=tmp)
    orig_fn = T0.rollout
    orig_dr = R0.distance_rule
    check("iv P = 0: rollout, make_states and the rule function are left as they are", ns0["STATE"]["runner"] is None and T0.rollout == orig_fn and getattr(T0.make_states, "__name__", "") == "make_states" and R0.distance_rule is orig_dr)
    check("iv P = 0: build_rollout returns the original function itself", ns0["build_rollout"](orig_fn, None) is orig_fn)
    st0, p0, e0 = run_window_set(ns0, T0, 0, hook="HOOK")
    check("iv P = 0: one call, H = 30, the caller's hook object itself", len(T0.calls) == 1 and T0.calls[0]["H"] == 30 and T0.calls[0]["hook"] == "HOOK")
    ns0["finish_preroll"]()
    check("iv P = 0: nothing written", os.listdir(tmp) == [])
    rows0 = {k: np.concatenate(v) for k, v in ns0["STATE"]["rulelog"].rows.items()}
    hf0 = dict(ns0["STATE"]["hf"])

with tempfile.TemporaryDirectory() as tmp:
    # P = 30: one segment, then the main call
    ns1, T1, _, _, R1 = fresh(30, tmp=tmp, run="p30")
    T1.calls.clear()
    st1, _, _ = run_window_set(ns1, T1, 30, wins=[(0, 66, 0, 0), (0, 70, 0, 0)])
    check("R2 P = 30: two calls (one segment + main), H = 30 each, t0 at the first call 36 / 40 without frame pose, at the main call 66 / 70 with it",
          len(T1.calls) == 2 and [c["H"] for c in T1.calls] == [30, 30] and T1.calls[0]["t0s"] == [36, 40] and all(f is None for f in T1.calls[0]["fp"]) and T1.calls[1]["t0s"] == [66, 70]
          and all(f is not None for f in T1.calls[1]["fp"]))

with tempfile.TemporaryDirectory() as tmp:
    # P = 60, executor plan
    ns, T, mine, a, Rn = fresh(60, tmp=tmp)
    orig_dr60 = ns["STATE"]["pre_shim"][1]
    check("R1 P = 60: R.distance_rule is the shim (not the original) while the run is installed", Rn.distance_rule is not orig_dr60 and Rn.distance_rule.__name__ == "distance_rule")
    check("iv P = 60: windows are built at t0 - P", [s.t0 for s in T.make_states(None, {"clipA": T.track}, {"clip": ["clipA"]}, [(0, 70, 0, 0)], "d")] == [10])
    try:
        T.make_states(None, {"clipA": T.track}, {"clip": ["clipA"]}, [(0, 62, 0, 0)], "d")
        ref = False
    except SystemExit as e:
        ref = "below 3" in str(e)
    check("iv P = 60: a window with t0 - P below 3 is refused with a message", ref)
    T.calls.clear()
    rec = Rec()
    states, plan0, exe = run_window_set(ns, T, 60, hook=rec)
    c = T.calls
    check("R2 P = 60: three calls (two segments, main), each with H = 30, the pre-roll calls with hook None and executor plan", len(c) == 3 and [x["H"] for x in c] == [30, 30, 30] and c[0]["hook"] is None and c[1]["hook"] is None
          and c[0]["executor"] == c[1]["executor"] == c[2]["executor"] == "plan" and c[0]["lt"] is None and T.H == 30)
    check("R2 P = 60: a re-base after each segment: t0 at the calls = [6, 10], [36, 40], [66, 70]", [x["t0s"] for x in c] == [[6, 10], [36, 40], [66, 70]])
    check("R2 P = 60: the first segment has no frame pose; the later calls have the LOGGED pose at their t0",
          all(f is None for f in c[0]["fp"]) and all(np.array_equal(c[1]["fp"][b], T.track[2 * t]) for b, t in enumerate([36, 40])) and all(np.array_equal(c[2]["fp"][b], T.track[2 * t]) for b, t in enumerate([66, 70])))
    check("R2 P = 60: the main call gets the hook through HookFrame (wrapping the caller's hook), the same planner", c[2]["hook"].__class__.__name__ == "HookFrame" and c[2]["hook"]._h is rec and c[0]["planner"] is c[2]["planner"])
    check("iv P = 60: windows end at t0 + 30 with t0 restored", [(s.t0, s.j) for s in states] == [(66, 96), (70, 100)])
    check("iv P = 60: the clipped count of the main phase only (3 ticks) and the pre-roll count kept for the log", [s.clipped for s in states] == [3, 3] and ns["STATE"]["runner"].clipped_pre["model"] == 12)
    eg = states[1].executed_ego()
    want_e = gtw_pinned(T.track[0::2], np.array([70]), np.arange(1, 31))[0] + rot(-T.track[2 * 70, 2]) @ np.array([0.5, -0.25])
    check("iv executed_ego() is the executed path t0+1..t0+30 in the logged frame at t0 (err %.2e)" % np.abs(eg - want_e).max(), eg.shape == (30, 2) and np.abs(eg - want_e).max() < 1e-12)
    check("iv the executed path returned by the rollout is the same (exe[1])", np.array_equal(exe[1], eg))
    check("iv the plan0 returned has the dtype of the tool's plan0 and the shape [B, 30, 2]", plan0.shape == (2, 30, 2) and plan0.dtype == np.float32)
    # R3 end to end: the recorder rows of the main phase hold the logged pose at t0
    logged_w0 = [T.track[2 * 66][:3], T.track[2 * 70][:3]]
    rw = rec.rows["pose_w0"]
    check("R3 rows: 3 ticks x 2 windows; pose_w0 = the LOGGED pose[:3] at t0 (err %.2e), the recorder itself wrote the own pose (offset %.3f m)" % (max(np.abs(rw[i] - logged_w0[i % 2]).max() for i in range(6)), np.hypot(*(rec.raw[0][:2] - rw[0][:2]))),
          len(rw) == 6 and all(np.abs(rw[i] - logged_w0[i % 2]).max() < 1e-12 for i in range(6)) and abs(np.hypot(*(rec.raw[0][:2] - rw[0][:2])) - math.hypot(0.5, 0.25)) < 1e-12)
    check("R3 the mapping counters: 6 rows, largest shift %.4f m" % ns["STATE"]["runner"].hook_max_shift, ns["STATE"]["runner"].hook_rows == 6 and abs(ns["STATE"]["runner"].hook_max_shift - math.hypot(0.5, 0.25)) < 1e-12)
    # logged executor
    ns2, T2, _, _, _ = fresh(60, "logged", tmp=tmp, run="y")
    T2.calls.clear()
    run_window_set(ns2, T2, 60)
    check("iv --preroll-exec logged: both segments get executor logged and the logged tracks by ci, the main call executor plan", [x["executor"] for x in T2.calls] == ["logged", "logged", "plan"] and T2.calls[0]["lt"] == [0] and T2.calls[2]["lt"] is None)
    # raise in the pre-roll (first, and second segment)
    for rz in (1, 2):
        ns3, T3, _, _, _ = fresh(60, raise_on=rz, tmp=tmp, run="z%d" % rz)
        s3 = ns3["STATE"]
        before = (dict(s3["hf"]), {k: list(v) for k, v in s3["rulelog"].rows.items()})
        try:
            run_window_set(ns3, T3, 60)
            raised = False
        except RuntimeError:
            raised = True
        check("iv a raise in pre-roll segment %d propagates, H is back at 30, the stores are as before" % rz, raised and T3.H == 30 and s3["hf"] == before[0] and all(len(s3["rulelog"].rows[k]) == len(before[1][k]) for k in before[1]))

    # v: the log split
    rows = {k: np.concatenate(v) for k, v in ns["STATE"]["rulelog"].rows.items()}
    same_shape = all(rows[k].shape == rows0[k].shape for k in rows0) and np.array_equal(rows["k"], rows0["k"]) and np.array_equal(rows["tick"], rows0["tick"]) and np.array_equal(rows["ci"], rows0["ci"])
    check("v  main-phase rule rows have the shapes and the k / tick / ci values of a P = 0 run (%d rows)" % len(rows["k"]), same_shape and len(rows["k"]) == 6)
    check("v  main-phase counters equal those of a P = 0 run", ns["STATE"]["hf"] == hf0 and len(ns["STATE"]["mid"]["rows"]["ci"]) == 3 and ns["STATE"]["T"].CAUSAL_STATS == {"ticks": 6})
    ns["finish_preroll"]()
    check("R1 finish_preroll puts the original rule function back", Rn.distance_rule is orig_dr60)
    files = sorted(os.listdir(tmp))
    check("v  files written: rule_log_pre, preroll_log, preroll_offsets", "rule_log_pre_x.npz" in files and "preroll_log_x.json" in files and "preroll_offsets_x.npz" in files)
    with np.load(os.path.join(tmp, "rule_log_pre_x.npz")) as z_:
        pre = {k: z_[k] for k in z_.files}
    check("R2 rule_log_pre: 2 segments x 3 ticks x 2 windows = 12 rows, tick 0..2 inside the segment, t0 = segment start {6, 10, 36, 40}, pre_segment {0, 1}",
          len(pre["ci"]) == 12 and sorted(set(pre["tick"].tolist())) == [0, 1, 2] and sorted(set(pre["t0"].tolist())) == [6, 10, 36, 40] and pre["pre_segment"].tolist() == [0] * 6 + [1] * 6)
    pl_ = json.load(open(os.path.join(tmp, "preroll_log_x.json")))
    with np.load(os.path.join(tmp, "preroll_offsets_x.npz")) as z_:
        off = {k: z_[k] for k in z_.files}
    want = {"preroll_steps", "preroll_exec", "windows", "ticks_pre_rolled", "offsets_at_t0", "readout_counters_pre_roll", "clipped_windows_in_preroll", "a_m0", "a_m0_used_in_rule", "segments_per_window", "tick_recorder_pose_w0_mapped"}
    check("v  preroll_log keys and counts (windows counted once, ticks over both segments)", want <= set(pl_) and pl_["windows"] == {"model": 2} and pl_["ticks_pre_rolled"] == {"model": 12} and pl_["preroll_steps"] == 60 and pl_["segments_per_window"] == 2 and pl_["rollout_calls"] == 2)
    check("v  pre-roll hf counters over the two segments: calls = 6 ticks, max_dv from the pre-roll only", pl_["readout_counters_pre_roll"]["model"]["hf"]["calls"] == 6 and pl_["readout_counters_pre_roll"]["model"]["hf"]["max_dv"] == 2.5)
    o = pl_["offsets_at_t0"]["model"]
    check("v  offsets at t0 (main-phase start): position %.4f (0.5590), heading %.4f deg, speed 1.0" % (o["dpos"]["mean"], o["dpsi_deg"]["mean"]),
          abs(o["dpos"]["mean"] - math.hypot(0.5, 0.25)) < 1e-12 and abs(o["dpsi_deg"]["mean"] - math.degrees(0.1)) < 1e-12 and abs(o["dv"]["max"] - 1.0) < 1e-12 and off["ci"].tolist() == [0, 0] and off["t0"].tolist() == [66, 70])
    exp_am = []
    for t_ in (66, 70):
        lp_ = T.track[2 * t_]
        v_ = rot(-lp_[2]) @ (T.track[2 * t_, :2] + np.array([0.5, -0.25]) - lp_[:2])
        exp_am.append(RD.progress(gtw_pinned(T.track[0::2], np.array([t_]), np.arange(1, 31)), v_[None, None, :])[0, 0])
    check("R1 offsets file: a_m0_first_tick per main window = %s (the arc of the own position on the logged path, err %.1e); seg_* rows for the first re-base (t0 36, 40)" % (np.round(off["a_m0_first_tick"], 6), np.abs(off["a_m0_first_tick"] - exp_am).max()),
          np.abs(off["a_m0_first_tick"] - exp_am).max() < 1e-12 and np.abs(off["a_m0_first_tick"]).min() > 0.1 and off["seg_t0"].tolist() == [36, 40] and off["seg_index"].tolist() == [1, 1] and off["seg_a_m0_first_tick"].shape == (2,))
    check("R1 preroll_log a_m0: the largest |a_m0| per run is reported for the main phase and for the segments", abs(pl_["a_m0"]["model"]["main_max_abs"] - max(abs(x) for x in exp_am)) < 1e-12 and pl_["a_m0"]["model"]["segments_max_abs"] > 0.1)

    # R2 planted: no re-base after the segment
    nsb, Tb, _, _, _ = fresh(60, tmp=tmp, run="bad2", pure=sub(PURE, "            st.t0 = t_new\n", "            pass\n"))
    Tb.calls.clear()
    run_window_set(nsb, Tb, 60)
    planted("R2 st.t0 not advanced after a segment", [x["t0s"] for x in Tb.calls] == [[6, 10], [36, 40], [66, 70]])

    # planted: the stores are not restored after the pre-roll
    nsw, Tw, _, _, _ = fresh(60, tmp=tmp, run="w", patch={"_restore": lambda dst, src: None})
    run_window_set(nsw, Tw, 60)
    roww = {k: np.concatenate(v) for k, v in nsw["STATE"]["rulelog"].rows.items()}
    planted("v  stores not restored (main rows keep the pre-roll rows)", roww["k"].shape == rows0["k"].shape and nsw["STATE"]["hf"] == hf0)

    # R2: refusals
    for bad in (45, 20, 15):
        try:
            fresh(bad, tmp=tmp, run="r%d" % bad, m=5 if bad == 15 else 10)
            ok_ = False
        except SystemExit as e:
            ok_ = "REFUSED" in str(e) and "multiple of" in str(e)
        check("R2 P = %d is refused (not a multiple of 30)" % bad, ok_)
    try:
        fresh(60, tmp=tmp, run="m20", m=20)
        ok_ = False
    except SystemExit as e:
        ok_ = "replan-steps" in str(e)
    check("R2 P = 60 with --replan-steps 20 is refused (20 does not divide 30)", ok_)
    try:
        NS["PrerollRun"](T, 45, "plan", lambda: [], RD)
        ok_ = False
    except ValueError:
        ok_ = True
    check("R2 PrerollRun itself refuses 45", ok_)
    for okp in (30, 90):
        try:
            fresh(okp, tmp=tmp, run="o%d" % okp)
            ok_ = True
        except SystemExit:
            ok_ = False
        check("R2 P = %d is accepted" % okp, ok_)

with tempfile.TemporaryDirectory() as tmp:
    # R3 with own == logged (zero offset): the recorded rows are unchanged
    nsz, Tz, _, _, _ = fresh(60, tmp=tmp, run="zero", off=np.zeros(4))
    recz = Rec()
    sz, _, _ = run_window_set(nsz, Tz, 60, hook=recz)
    check("R3 own == logged: the recorded pose_w0 rows are unchanged bit for bit (6 rows), no shift counted", len(recz.rows["pose_w0"]) == 6 and all(np.array_equal(a_, b_) for a_, b_ in zip(recz.rows["pose_w0"], recz.raw)) and nsz["STATE"]["runner"].hook_max_shift == 0.0)
    nsz["finish_preroll"]()
    with np.load(os.path.join(tmp, "preroll_offsets_zero.npz")) as z_:
        offz = {k: z_[k] for k in z_.files}
    pz = json.load(open(os.path.join(tmp, "preroll_log_zero.json")))
    check("R1 own == logged: a_m0 is exactly 0.0 for every window and every re-base, the log says so (max %s)" % pz["a_m0"]["model"]["main_max_abs"],
          np.array_equal(offz["a_m0_first_tick"], np.zeros(2)) and np.array_equal(offz["seg_a_m0_first_tick"], np.zeros(2)) and pz["a_m0"]["model"]["main_max_abs"] == 0.0 and pz["a_m0"]["model"]["main_exact_zero"] == 2)

with tempfile.TemporaryDirectory() as tmp:
    # R3 with a rigid offset: pose_w0 = logged pose, s_ckpt_w0 = the analytic value (the arithmetic of the tool's transform_ckpt in numpy, held against an independent expectation)
    def ckpt_w0(ck, sp, dp):
        """s_ckpt (x, y, psi in the frame of the world pose sp) -> the frame of the world pose dp (the formula of strat_roll_read.s1_row_check)."""
        cs, ss = np.cos(sp[:, 2])[:, None], np.sin(sp[:, 2])[:, None]
        wx = sp[:, 0:1] + cs * ck[..., 0] - ss * ck[..., 1]
        wy = sp[:, 1:2] + ss * ck[..., 0] + cs * ck[..., 1]
        dx, dy = wx - dp[:, 0:1], wy - dp[:, 1:2]
        cd, sd = np.cos(dp[:, 2])[:, None], np.sin(dp[:, 2])[:, None]
        wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
        return np.stack([cd * dx + sd * dy, -sd * dx + cd * dy, wrap(ck[..., 2] + sp[:, 2:3] - dp[:, 2:3])], -1)

    nsr, Tr, _, _, _ = fresh(60, tmp=tmp, run="rigid")
    recr = Rec()
    run_window_set(nsr, Tr, 60, hook=recr)
    pw = np.stack(recr.rows["pose_w0"])
    raw = np.stack(recr.raw)
    logged = np.stack([Tr.track[2 * t_][:3] for t_ in (66, 70)] * 3)
    sp_ = rng.normal(0, 5, (6, 3)) + pw
    sp_[:, 2] = pw[:, 2] + rng.normal(0, 0.2, 6)
    ck_ = rng.normal(0, 20, (6, 3, 3))
    # analytic expectation: the checkpoints in the world, then into the LOGGED frame at t0
    exp_ck = np.zeros_like(ck_)
    for i in range(6):
        for q in range(3):
            w = sp_[i, :2] + rot(sp_[i, 2]) @ ck_[i, q, :2]
            exp_ck[i, q, :2] = rot(-logged[i, 2]) @ (w - logged[i, :2])
            exp_ck[i, q, 2] = (ck_[i, q, 2] + sp_[i, 2] - logged[i, 2] + math.pi) % (2 * math.pi) - math.pi
    got_ck = ckpt_w0(ck_, sp_, pw)
    check("R3 rigid offset: pose_w0 = the logged pose (err %.1e); s_ckpt_w0 from it = the analytic expectation to 1e-12 (err %.1e)" % (np.abs(pw - logged).max(), np.abs(got_ck - exp_ck).max()),
          np.abs(pw - logged).max() < 1e-12 and np.abs(got_ck - exp_ck).max() < 1e-12)
    wrong_ck = ckpt_w0(ck_, sp_, raw)
    check("R3 the offset is large enough to tell the arms apart (own-frame arm differs by %.2f m)" % np.abs(wrong_ck - exp_ck)[..., :2].max(), np.abs(wrong_ck - exp_ck)[..., :2].max() > 0.3)
    planted("R3 the own pose left as pose_w0 (first-build behaviour)", np.abs(wrong_ck - exp_ck).max() < 1e-9)
    nsx, Tx, _, _, _ = fresh(60, tmp=tmp, run="nohook", pure=sub(PURE, "                rows[\"pose_w0\"][-B + b] = new\n", "                pass\n"))
    recx = Rec()
    run_window_set(nsx, Tx, 60, hook=recx)
    planted("R3 hook shim that does not write the rows", np.abs(np.stack(recx.rows["pose_w0"]) - logged).max() < 1e-9)
    # a hook without rows passes through; a recorder that did not record pose_w0 (no --dump-heads) is left alone
    nsy, Ty, _, _, _ = fresh(30, tmp=tmp, run="plain")
    seen = []
    run_window_set(nsy, Ty, 30, hook=lambda n, b, p, s: seen.append(n))
    check("R3 a hook without a rows dict (or without pose_w0) is called and left alone", seen == [0, 1, 2])

# ---------------------------------------------------------------------------------------------------------- vi: the frame shim (unchanged from the first build)
tt = StubT({}, 0)
stv = types.SimpleNamespace(track=np.array(tt.track), t0=70, frame_pose5=None)
pl_v = types.SimpleNamespace(states=[stv])
fk = NS["FrameFK"](tt.fk, pl_v)
lpv = tt.track[2 * 70] + np.array([1.0, 2.0, 0.3, 0.0])
plain = fk.gt_waypoints(stv.track[0::2], np.array([70]), np.array([3]))
check("vi the shim passes through while frame_pose5 is not set", np.array_equal(plain, gtw_pinned(stv.track[0::2], np.array([70]), np.array([3]))))
stv.frame_pose5 = lpv
p5 = np.array(tt.track[0::2], copy=True)
p5[70] = lpv
framed = fk.gt_waypoints(stv.track[0::2], np.array([70]), np.array([3]))
check("vi with frame_pose5 set it reads the own track in the logged frame", np.array_equal(framed, gtw_pinned(p5, np.array([70]), np.array([3]))))
planted("vi the shim not applied", np.abs(plain - framed).max() < 1e-6)
check("vi a two-row array (the R1 call) passes through the shim", np.array_equal(fk.gt_waypoints(np.stack([lpv, tt.track[2 * 70]]), np.array([0]), np.array([1])), gtw_pinned(np.stack([lpv, tt.track[2 * 70]]), np.array([0]), np.array([1]))))

# ---------------------------------------------------------------------------------------------------------- static facts of the wrapper text
check("the wrapper builds no runner, no shim and no hook wrapper when P = 0: install_preroll returns first", GLUE.index("if P == 0:") < GLUE.index("runner = PrerollRun") < GLUE.index("make_g0_shim"))
lines_g0 = [ln for ln in RPSRC.splitlines() if "g0_corr" in ln]
check("P = 0 text path of ReaderPlanner: every g0_corr line sits inside `if pre:` (%d lines)" % len(lines_g0), len(lines_g0) == 2 and all(ln.startswith("            STATE") for ln in lines_g0) and "pre = STATE.get(\"preroll\", 0) > 0" in RPSRC)
check("the pinned tool is not touched by this check (no import of it)", "v8f_t1_eval.py" not in PURE)
print("ZZ CHECK-PREROLL-B " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
sys.exit(0 if not FAILED else 1)
