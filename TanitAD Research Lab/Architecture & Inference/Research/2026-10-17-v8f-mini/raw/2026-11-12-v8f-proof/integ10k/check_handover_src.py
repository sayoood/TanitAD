"""check_handover_src.py -- checks of v8f_t1_eval_reader_rule_rzgfhqs.py (study "handover-src": --t0-state own|logged|speed_logged|path_logged).
Part A (always, numpy and the standard library only): the wrapper text between its HANDOVER markers is executed here with stubs and held against known values (the mixtures of the three
  modes, the group differences, the gating of the build_item patch), the undo of the wrapper's frame mapping against an analytic value, and static facts of the text (own installs nothing,
  the refusal, the parser). Each check has a planted wrong arm that must fail.
Part B (on the pod, needs the tool tree, the checkpoint and the data; skipped with --pure-only): runs the parent wrapper and the new wrapper on a tiny restricted grid and compares.
  1  own with the new wrapper equals the parent wrapper on the same aged run: every dump array (largest difference 0.0), every tick-file array, the pre-roll offsets file
  2  logged: the first replan segment of the scored track of the aged run, expressed in the own frame at t0 (the wrapper's rigid mapping undone with the pose_w0 and pose columns of the
     tick-0 rows), against the fresh run's track for the same windows: largest difference in metres, bar 1e-3 m; the tick-0 plan, tau, P and the head outputs identical (bar 1e-6);
     a planted arm (own, nothing handed over) must differ
  3  speed_logged and path_logged: only their group differs from own (v0 in the tick-0 rows, s_pose / t_pose, the group differences of the log line in the preroll log), each equals logged on
     its group; their pre-roll offsets equal those of own exactly
  4  refusal: --t0-state logged with --preroll-steps 0 is REFUSED
Nothing here removes, moves or stops anything; part B writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folders already there).
Usage on the pod (part B):
  python check_handover_src.py --parent PARENT.py --new NEW.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid GRID_P30.json --workdir NEWDIR
         [--distrule-dir DIR] [--replan-steps 10] [--preroll-steps 30] [--device cuda] [--python PY] [--dry]
  GRID_P30.json = a tiny grid already restricted for the pre-roll (make_longroll_grid.py --preroll-steps 30), the same grid for every run.
Usage anywhere: python check_handover_src.py --pure-only [--new NEW.py]
Last line: ZZ CHECK-HANDOVER-SRC PASS  or  ZZ CHECK-HANDOVER-SRC FAIL <names>   (--pure-only: ZZ CHECK-HANDOVER-SRC-PURE PASS or FAIL <names>)"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
    if not cond:
        FAILED.append(name)


def planted(name, wrong_passes, detail=""):
    print(("PASS " if not wrong_passes else "FAIL ") + "planted wrong arm caught: " + name + (" -- " + detail if detail else ""), flush=True)
    if wrong_passes:
        FAILED.append("planted " + name)


ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--new", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqs.py"))
ap.add_argument("--parent", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py"))
ap.add_argument("--pure-only", action="store_true")
ap.add_argument("--analyze-only", action="store_true")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--tree")
ap.add_argument("--ckpt")
ap.add_argument("--tracks")
ap.add_argument("--eval-cache")
ap.add_argument("--eval-join")
ap.add_argument("--grid")
ap.add_argument("--workdir")
ap.add_argument("--distrule-dir", default="")
ap.add_argument("--replan-steps", type=int, default=10)
ap.add_argument("--preroll-steps", type=int, default=30)
ap.add_argument("--device", default="cuda")
ap.add_argument("--python", default=sys.executable)
ARGS = ap.parse_args()

SRC = open(ARGS.new, encoding="utf-8").read()
print("wrapper", ARGS.new, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16], flush=True)


def between(a, b):
    i, j = SRC.index(a), SRC.index(b)
    assert i < j
    return SRC[i:j]


# ==================================================================================================================== part A
BLOCK = between("# ==== HANDOVER BEGIN", "# ==== HANDOVER END")
NS = {"np": np, "math": math}
exec(BLOCK, NS)
rng = np.random.default_rng(11)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def kappa_stub(om, v):                                   # the documented form of _pose_kappa0 (stand-in: the real one is the programme's)
    return np.clip(np.asarray(om, np.float64) / np.maximum(np.asarray(v, np.float64), 2.0), -0.3, 0.3)


def kin_feat_stub(k3):
    v, a, k = k3[:, 0], k3[:, 1], k3[:, 2]
    return np.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], 1)


def kdx_stub(k3):
    return np.stack([k3[:, 0] * 2.0 + k3[:, 1] * 3.0 + k3[:, 2] * 5.0] * 4, 1)


T_ROWS, T0, HMAX = 60, 40, 10


def make_clip(seed, own=False):
    r = np.random.default_rng(seed)
    v = np.abs(8.0 + np.cumsum(r.normal(0, 0.1, T_ROWS)))
    a = np.gradient(v) / 0.2
    om = r.normal(0, 0.05, T_ROWS)
    kap = kappa_stub(om, v)
    kin3 = np.stack([v, a, kap], 1)
    kin3[0] = np.nan
    x = np.cumsum(v * 0.2) + (5.0 if own else 0.0)
    pose = np.stack([x, np.cumsum(r.normal(0, 0.3, T_ROWS)), np.cumsum(om * 0.2) + (0.2 if own else 0.0), v], 1)
    p10f = kin_feat_stub(np.nan_to_num(kin3))
    p10f[0] = np.nan
    return {"pose": pose.astype(np.float64), "kin3": kin3, "p10f": p10f, "base": np.zeros((T_ROWS, 4)), "a": a, "ch1": om, "v": v}


LG = make_clip(1)
OWN = make_clip(2, own=True)
NAV_L = rng.normal(0, 1, (T_ROWS, 5))
NAV_O = rng.normal(0, 1, (T_ROWS, 5))
snap_own = {k: np.array(v, copy=True) for k, v in OWN.items()}
snap_lg = {k: np.array(v, copy=True) for k, v in LG.items()}
HIST = np.arange(T0 - HMAX + 1, T0 + 1)


def shadow(mode, own=OWN, lg=LG):
    return NS["handover_shadow"](mode, dict(own), lg, NAV_O, NAV_L, T0, HMAX, kappa_stub, kin_feat_stub, kdx_stub)


dl, nl, il = shadow("logged")
check("A1 logged: every ego array equals the logged cache on the rows up to t0 and the own array after; nav likewise",
      all(np.array_equal(dl[k][:T0 + 1], LG[k][:T0 + 1], equal_nan=True) and np.array_equal(dl[k][T0 + 1:], OWN[k][T0 + 1:], equal_nan=True) for k in NS["HO_EGO"])
      and np.array_equal(nl[:T0 + 1], NAV_L[:T0 + 1]) and np.array_equal(nl[T0 + 1:], NAV_O[T0 + 1:]))
ds_, ns_, is_ = shadow("speed_logged")
check("A2 speed_logged: pose x, y, yaw and the yaw rate are the own rows; v, a, kin3 v and a, pose column 3 are the logged rows (up to t0)",
      np.array_equal(ds_["pose"][:, :3], OWN["pose"][:, :3]) and np.array_equal(ds_["ch1"], OWN["ch1"]) and np.array_equal(ds_["v"][:T0 + 1], LG["v"][:T0 + 1]) and np.array_equal(ds_["a"][:T0 + 1], LG["a"][:T0 + 1])
      and np.array_equal(ds_["kin3"][:T0 + 1, 0:2], LG["kin3"][:T0 + 1, 0:2], equal_nan=True) and np.array_equal(ds_["pose"][:T0 + 1, 3], LG["pose"][:T0 + 1, 3]) and np.array_equal(ns_, NAV_O))
exp_k = kappa_stub(OWN["ch1"][HIST], LG["v"][HIST])
check("A2 speed_logged: kappa on the history rows = clamp(own omega / max(logged v, 2)); p10f rows follow kin3; base at t0 follows kin3 (err %.1e)" % np.abs(ds_["kin3"][HIST, 2] - exp_k).max(),
      np.array_equal(ds_["kin3"][HIST, 2], exp_k) and np.array_equal(ds_["p10f"][HIST], kin_feat_stub(ds_["kin3"][HIST])) and np.array_equal(ds_["base"][T0], kdx_stub(ds_["kin3"][T0:T0 + 1])[0]))
dp_, np_, ip_ = shadow("path_logged")
check("A3 path_logged: pose x, y, yaw and the yaw rate are the logged rows (up to t0); v, a, kin3 v and a, pose column 3 and nav are the own rows",
      np.array_equal(dp_["pose"][:T0 + 1, :3], LG["pose"][:T0 + 1, :3]) and np.array_equal(dp_["ch1"][:T0 + 1], LG["ch1"][:T0 + 1]) and np.array_equal(dp_["v"], OWN["v"]) and np.array_equal(dp_["a"], OWN["a"])
      and np.array_equal(dp_["kin3"][:, 0:2], OWN["kin3"][:, 0:2], equal_nan=True) and np.array_equal(dp_["pose"][:, 3], OWN["pose"][:, 3]) and np.array_equal(np_, NAV_O))
exp_k = kappa_stub(LG["ch1"][HIST], OWN["v"][HIST])
check("A3 path_logged: kappa on the history rows = clamp(logged omega / max(own v, 2)); p10f follows (err %.1e)" % np.abs(dp_["kin3"][HIST, 2] - exp_k).max(),
      np.array_equal(dp_["kin3"][HIST, 2], exp_k) and np.array_equal(dp_["p10f"][HIST], kin_feat_stub(dp_["kin3"][HIST])))
check("A4 the inputs (own d, logged cache) are not changed by any mode", all(np.array_equal(OWN[k], snap_own[k], equal_nan=True) and np.array_equal(LG[k], snap_lg[k], equal_nan=True) for k in OWN))
check("A4 the recomposition residual is reported for the partial modes (and is 0 for the stub, which has the form it reproduces)",
      "kappa_recompose_max_abs_own" in is_ and is_["kappa_recompose_max_abs_own"] < 1e-12 and ip_["kappa_recompose_max_abs_logged"] < 1e-12 and il == {})
planted("A2/A3 speed_logged and path_logged are the same mixture", np.array_equal(ds_["pose"], dp_["pose"]) and np.array_equal(ds_["v"], dp_["v"]))
planted("A1/A2 speed_logged built as logged", np.array_equal(ds_["pose"][:T0 + 1, :3], dl["pose"][:T0 + 1, :3]))


def item(d, nav):
    idc = HIST
    return {"pose": d["pose"][idc][:, :3].astype(np.float32), "pose_now": d["pose"][T0][:3].astype(np.float32), "pose_t": d["pose"][T0][:3].astype(np.float32), "pose_s": d["pose"][T0][:3].astype(np.float32),
            "p10f": np.nan_to_num(d["p10f"][idc]), "base": d["base"][T0], "nav": nav[idc], "limit": np.zeros(4), "frame_ok": np.ones(len(idc), bool)}


i_own, i_l, i_s, i_p = item(OWN, NAV_O), item(dl, nl), item(ds_, ns_), item(dp_, np_)
df_s, df_p, df_l = NS["item_diff"](i_s, i_own), NS["item_diff"](i_p, i_own), NS["item_diff"](i_l, i_own)
check("A5 item_diff speed_logged vs own: positions and headings exactly 0.0, speed above 0, nav 0.0", df_s["positions_m"] == 0.0 and df_s["headings_rad"] == 0.0 and df_s["speed_ms"] > 0.0 and df_s["nav"] == 0.0,
      "pos %.3g head %.3g speed %.3g" % (df_s["positions_m"], df_s["headings_rad"], df_s["speed_ms"]))
check("A5 item_diff path_logged vs own: speed exactly 0.0, positions and headings above 0, nav 0.0", df_p["speed_ms"] == 0.0 and df_p["positions_m"] > 0.0 and df_p["headings_rad"] > 0.0 and df_p["nav"] == 0.0,
      "pos %.3g head %.3g speed %.3g" % (df_p["positions_m"], df_p["headings_rad"], df_p["speed_ms"]))
check("A5 item_diff logged vs own: all three groups and nav above 0", df_l["positions_m"] > 0 and df_l["headings_rad"] > 0 and df_l["speed_ms"] > 0 and df_l["nav"] > 0)
check("A5 item_diff of an item with itself: every group 0.0", all(v == 0.0 for v in NS["item_diff"](i_own, i_own).values()))
planted("A5 item_diff on positions only (speed column ignored)", NS["item_diff"](i_s, i_own)["speed_ms"] == 0.0)
stats = {}
NS["ho_stats_add"](stats, "model", df_s, is_)
NS["ho_stats_add"](stats, "model", df_p, ip_)
check("A5 ho_stats_add keeps the largest value per group over the states", stats["model"]["states"] == 2 and stats["model"]["max_abs"]["speed_ms"] == df_s["speed_ms"] and stats["model"]["max_abs"]["positions_m"] == df_p["positions_m"])

# the gating of the patch (logged mode: no torch needed)
calls = []


class StubSt:
    def __init__(self, with_pose=True, base=True):
        self.ci, self.j, self.T, self.t0, self.lr_t0, self.scene0 = 0, T0, T_ROWS, T0, T0, None
        self.d, self.nav = dict(OWN), NAV_O
        self.frame_pose5 = np.zeros(4) if with_pose else None
        if base:
            self.lr_base, self.lr_nav = LG, NAV_L


def orig_bi(ds, proxy, st, tick, a_t, a_s, scene, limit, stats_=None):
    calls.append((st.__class__.__name__, int(tick), stats_ is None))
    return item(st.d, st.nav)


runner = type("R", (), {"main_kind": None})()
stt = {}
bi = NS["make_handover_build_item"](orig_bi, None, runner, "logged", stt)
DS = type("DS", (), {"hmax": HMAX})()
bi(DS, None, StubSt(), 0, 0, 0, "logged", "causal", {})
check("A6 main_kind None (the pre-roll segments): passes through to the original build_item with the state itself", calls[-1][0] == "StubSt" and not stt)
runner.main_kind = "model"
n0 = len(calls)
bi(DS, None, StubSt(), 1, 0, 0, "logged", "causal", {})
bi(DS, None, StubSt(with_pose=False), 0, 0, 0, "logged", "causal", {})
bi(DS, None, StubSt(base=False), 0, 0, 0, "logged", "causal", {})
check("A6 later ticks, a state without frame pose, a state without logged cache: pass through (3 calls, no statistics)", len(calls) == n0 + 3 and all(c[0] == "StubSt" for c in calls[n0:]) and not stt)
bi(DS, None, StubSt(), 0, 0, 0, "logged", "causal", {})
check("A6 main phase, tick 0: two builds (the handed view with the counters, the own reference without them), statistics for the kind",
      [c[0] for c in calls[-2:]] == ["HandoverView", "HandoverView"] and calls[-2][2] is False and calls[-1][2] is True and stt["model"]["states"] == 1 and stt["model"]["states_item_differs"] == 1)

# the undo of the wrapper's frame mapping against the analytic value
plan = rng.normal(0, 3, (10, 2))
po, pl = np.array([30.0, -12.0, 0.7]), np.array([29.1, -11.4, 0.55])
W = po[:2] + (rot(po[2]) @ plan.T).T
c_logged = (rot(-pl[2]) @ (W - pl[:2]).T).T                 # what the wrapper writes (frame of the logged pose at t0)


def undo_to_own(c, pose_own, pose_logged):
    w = pose_logged[None, :2] + (rot(pose_logged[2]) @ np.asarray(c, np.float64).T).T
    return (rot(-pose_own[2]) @ (w - pose_own[None, :2]).T).T


err = np.abs(undo_to_own(c_logged, po, pl) - plan).max()
check("A7 undoing the wrapper's mapping with (own pose, logged pose) returns the plan knots in the own frame (err %.1e)" % err, err < 1e-12)
planted("A7 the logged frame read as the own frame", np.abs(c_logged - plan).max() < 1e-3, "offset %.2f m" % np.abs(c_logged - plan).max())

# static facts of the text
i_inst = SRC.index("# handover-src: not installed for own")
check("S1 the build_item patch is installed only when the mode is not own (the guard line sits right before the assignment)", SRC[SRC.rindex(chr(10), 0, i_inst) + 1:i_inst].strip().startswith("if STATE[") and i_inst < SRC.index("T.build_item = make_handover_build_item(") and SRC.count("T.build_item = make_handover_build_item(") == 1)
check("S2 the refusal sits in the P = 0 branch of install_preroll, after the preroll-exec refusal",
      SRC.index('REFUSED: --preroll-exec goes with') < SRC.index("REFUSED: --t0-state %s goes with --preroll-steps above 0") < SRC.index("runner = PrerollRun("))
check("S3 the parser takes the four values with default own", 'ap.add_argument("--t0-state", choices=T0_STATES, default="own")' in SRC and 'T0_STATES = ("own", "logged", "speed_logged", "path_logged")' in SRC)
BANNED = ("vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "closed" + " loop")
bad = [w for w in BANNED if w in BLOCK]
check("S5 the new block has none of the banned words", not bad, str(bad))

if ARGS.pure_only:
    print("ZZ CHECK-HANDOVER-SRC-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
    sys.exit(0 if not FAILED else 1)

# ==================================================================================================================== part B (pod)
for k in ("tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "workdir"):
    if not getattr(ARGS, k):
        print("ZZ CHECK-HANDOVER-SRC STOP missing --" + k.replace("_", "-"))
        sys.exit(3)
W = ARGS.workdir
M = ARGS.replan_steps
P = ARGS.preroll_steps
DISTRULE = ARGS.distrule_dir or os.path.dirname(os.path.abspath(ARGS.new))
TIMEOUT = shutil.which("timeout")
if os.path.exists(W) and not ARGS.analyze_only:
    print("ZZ CHECK-HANDOVER-SRC STOP OUTPUT-EXISTS " + W)
    sys.exit(3)
if not ARGS.dry and not os.path.exists(W):
    os.makedirs(W)

RUNS = [  # name, wrapper, extra wrapper args
    ("fresh", ARGS.new, []),
    ("parent_aged", ARGS.parent, ["--preroll-steps", str(P)]),
    ("own", ARGS.new, ["--preroll-steps", str(P), "--t0-state", "own"]),
    ("logged", ARGS.new, ["--preroll-steps", str(P), "--t0-state", "logged"]),
    ("speed_logged", ARGS.new, ["--preroll-steps", str(P), "--t0-state", "speed_logged"]),
    ("path_logged", ARGS.new, ["--preroll-steps", str(P), "--t0-state", "path_logged"]),
]


def tag_of(name):
    return name + "_m%d_causal_heads" % M


def run_cmd(name, wrapper, extra):
    out = os.path.join(W, name)
    cmd = [ARGS.python, wrapper, "--tree", ARGS.tree, "--form", "R0"] + extra + [
        "run", "--ckpt", ARGS.ckpt, "--tag", tag_of(name), "--tracks", ARGS.tracks, "--eval-cache", ARGS.eval_cache, "--eval-grid", ARGS.grid, "--eval-join", ARGS.eval_join,
        "--out-dir", out, "--replan-steps", str(M), "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal",
        "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze", "--device", ARGS.device]
    if TIMEOUT:
        cmd = [TIMEOUT, "1800"] + cmd
    return out, cmd


def run_one(name, wrapper, extra):
    out, cmd = run_cmd(name, wrapper, extra)
    if ARGS.dry:
        print("CMD: DISTRULE_DIR=%s PYTHONPATH=%s %s" % (DISTRULE, os.path.join(ARGS.tree, "stack"), " ".join(cmd)))
        return 0
    if os.path.exists(out):
        if ARGS.analyze_only:
            return 0
        raise SystemExit("OUTPUT-EXISTS " + out)
    os.makedirs(os.path.join(out, "ticks"))
    env = dict(os.environ, DISTRULE_DIR=DISTRULE, PYTHONPATH=os.path.join(ARGS.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("RUN %s wrapper=%s rc=%d" % (name, os.path.basename(wrapper), rc), flush=True)
    return rc


for name, wrapper, extra in RUNS:
    rc = run_one(name, wrapper, extra)
    if rc != 0:
        print("ZZ CHECK-HANDOVER-SRC FAIL run-%s-rc-%d (see %s)" % (name, rc, os.path.join(W, name, "t1.log")))
        sys.exit(1)

# refusal run: no output folder is needed, only the return code and the text
REF_OUT = os.path.join(W, "refusal")
ref_cmd = [ARGS.python, ARGS.new, "--tree", ARGS.tree, "--form", "R0", "--t0-state", "logged", "--preroll-steps", "0", "run", "--ckpt", ARGS.ckpt, "--tag", "refusal", "--tracks", ARGS.tracks,
           "--eval-cache", ARGS.eval_cache, "--eval-grid", ARGS.grid, "--eval-join", ARGS.eval_join, "--out-dir", REF_OUT, "--replan-steps", str(M), "--nav", "deadreckon", "--scene", "logged",
           "--schedule", "periodic", "--limit", "causal", "--no-analyze", "--device", ARGS.device]
if TIMEOUT:
    ref_cmd = [TIMEOUT, "600"] + ref_cmd
if ARGS.dry:
    print("CMD: " + " ".join(ref_cmd))
    print("ZZ CHECK-HANDOVER-SRC dry run: nothing executed beyond part A")
    sys.exit(0)
if os.path.exists(os.path.join(W, "refusal.log")) and ARGS.analyze_only:
    ref_text, ref_rc = open(os.path.join(W, "refusal.log"), encoding="utf-8").read(), None
else:
    env = dict(os.environ, DISTRULE_DIR=DISTRULE, PYTHONPATH=os.path.join(ARGS.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    cp = subprocess.run(ref_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ref_text, ref_rc = cp.stdout.decode("utf-8", "replace"), cp.returncode
    with open(os.path.join(W, "refusal.log"), "w", encoding="utf-8") as fh:
        fh.write(ref_text)


def ticks_path(name):
    return os.path.join(W, name, "ticks", "ticks_%s.npz" % tag_of(name))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def tick0(name):
    z = load_npz(ticks_path(name))
    m = z["tick"] == 0
    o = np.lexsort((z["t0"][m], z["ci"][m]))
    return {k: z[k][m][o] for k in z if z[k].ndim >= 1 and len(z[k]) == len(m)}


def dumps(name):
    d = os.path.join(W, name, "dump_" + tag_of(name))
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    return {os.path.basename(f): load_npz(f) for f in files}


def maxdiff_dicts(a, b):
    """-> (largest absolute difference over the common numeric arrays, list of problems)"""
    worst, prob = 0.0, []
    if sorted(a) != sorted(b):
        prob.append("keys differ %s %s" % (sorted(set(a) ^ set(b))[:4], ""))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            prob.append("shape of %s" % k)
            continue
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            same_nan = np.array_equal(np.isnan(xf), np.isnan(yf))
            if not same_nan:
                prob.append("nan pattern of %s" % k)
                continue
            if xf.size:
                worst = max(worst, float(np.nanmax(np.abs(xf - yf))) if np.isfinite(xf).any() else 0.0)
        elif not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob


# ---------------------------------------------------------------- 1 own == parent
DUMP_P, DUMP_O = dumps("parent_aged"), dumps("own")
worst, prob = 0.0, []
if sorted(DUMP_P) != sorted(DUMP_O) or not DUMP_P:
    prob.append("dump file lists differ or empty")
for f in sorted(set(DUMP_P) & set(DUMP_O)):
    w_, p_ = maxdiff_dicts(DUMP_P[f], DUMP_O[f])
    worst, prob = max(worst, w_), prob + p_
check("1a own with the new wrapper vs the parent wrapper (aged run): every dump array identical, largest difference %.3g (%d files)" % (worst, len(DUMP_P)), worst == 0.0 and not prob, str(prob[:3]))
wt, pt = maxdiff_dicts(load_npz(ticks_path("parent_aged")), load_npz(ticks_path("own")))
check("1b the tick files: every array identical, largest difference %.3g" % wt, wt == 0.0 and not pt, str(pt[:3]))
OFF = {n: load_npz(os.path.join(W, n, "ticks", "preroll_offsets_%s.npz" % tag_of(n))) for n in ("parent_aged", "own", "logged", "speed_logged", "path_logged")}
wo, po_ = maxdiff_dicts(OFF["parent_aged"], OFF["own"])
check("1c the pre-roll offsets file of own equals the parent's: largest difference %.3g" % wo, wo == 0.0 and not po_, str(po_[:3]))
PL = {n: json.load(open(os.path.join(W, n, "ticks", "preroll_log_%s.json" % tag_of(n)), encoding="utf-8")) for n in ("own", "logged", "speed_logged", "path_logged")}
check("1d the preroll log records the mode for each run", all(PL[n].get("t0_state") == n for n in PL), str({n: PL[n].get("t0_state") for n in PL}))
check("1e own: nothing recorded as replaced (empty difference record)", PL["own"].get("t0_state_first_tick_max_abs_diff") == {})

# ---------------------------------------------------------------- 2 logged against fresh
TF, TO, TL, TS, TP = (tick0(n) for n in ("fresh", "own", "logged", "speed_logged", "path_logged"))
same_rows = all(np.array_equal(TF["ci"], t["ci"]) and np.array_equal(TF["t0"], t["t0"]) for t in (TO, TL, TS, TP))
check("2 the tick-0 rows of the five runs are the same windows (%d windows)" % len(TF["ci"]), same_rows and len(TF["ci"]) > 0)
HEADF = [k for k in ("plan", "tau", "P", "v0", "s_ckpt", "s_pose", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred", "t_pose") if k in TF and k in TL]


def maxd(a, b, keys):
    return {k: float(np.nanmax(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)))) if np.asarray(a[k]).size else 0.0 for k in keys}


d_fl = maxd(TF, TL, HEADF)
d_plan = max(d_fl[k] for k in ("plan", "tau", "P") if k in d_fl)
check("2a logged vs fresh, tick 0: plan, tau, P identical (largest %.3g, bar 1e-6)" % d_plan, d_plan <= 1e-6, str({k: "%.2g" % v for k, v in d_fl.items()}))
d_head = max(v for k, v in d_fl.items() if k not in ("plan", "tau", "P"))
check("2a logged vs fresh, tick 0: v0, s_pose, t_pose and the head outputs identical (largest %.3g, bar 1e-6)" % d_head, d_head <= 1e-6)
d_fo = maxd(TF, TO, ["plan", "tau", "P", "v0"])
planted("2a own (nothing handed over) against fresh: the same measurement must differ", max(d_fo.values()) <= 1e-6, "largest %.3g" % max(d_fo.values()))

DF, DLG, DOWN = dumps("fresh"), dumps("logged"), dumps("own")


def first_segment_in_own_frame(dump_aged, ticks_aged, arm):
    """the aged run's track (frame of the logged pose at t0) for the first replan segment, in the own frame at t0; keyed (ci, t0)."""
    out = {}
    pose_o = {(int(c), int(t)): p for c, t, p in zip(ticks_aged["ci"], ticks_aged["t0"], ticks_aged["pose"])}
    pose_l = {(int(c), int(t)): p for c, t, p in zip(ticks_aged["ci"], ticks_aged["t0"], ticks_aged["pose_w0"])}
    for f, z in dump_aged.items():
        for i, (e, w) in enumerate(zip(z["eid"], z["ws"])):
            key = (int(e), int(w))
            out[key] = undo_to_own(z[arm][i][:M], np.asarray(pose_o[key], np.float64)[:3], np.asarray(pose_l[key], np.float64)[:3])
    return out


def first_segment_fresh(dump_fresh, arm):
    return {(int(e), int(w)): z[arm][i][:M].astype(np.float64) for z in dump_fresh.values() for i, (e, w) in enumerate(zip(z["eid"], z["ws"]))}


for arm, label in (("cl", "model arm"), ("fl", "kinematic-prior arm")):
    a = first_segment_in_own_frame(DLG, TL, arm)
    b = first_segment_fresh(DF, arm)
    keys = sorted(set(a) & set(b))
    dd = max(float(np.abs(a[k] - b[k]).max()) for k in keys) if keys else float("inf")
    check("2b logged, %s: first replan segment (%d steps) of the aged track in the own frame vs the fresh track, largest difference %.3g m over %d windows (bar 1e-3 m)" % (label, M, dd, len(keys)),
          len(keys) == len(a) == len(b) and dd <= 1e-3)
a_own = first_segment_in_own_frame(DOWN, TO, "cl")
b_fresh = first_segment_fresh(DF, "cl")
dd_own = max(float(np.abs(a_own[k] - b_fresh[k]).max()) for k in sorted(set(a_own) & set(b_fresh)))
planted("2b own against fresh in the same measurement (the state part is visible)", dd_own <= 1e-3, "largest %.3g m" % dd_own)

# ---------------------------------------------------------------- 3 the partial modes
d_ls = maxd(TL, TO, ["v0", "s_pose", "t_pose"])
check("3 the offsets are large enough to tell the groups apart: logged differs from own in v0 (%.3g) and in s_pose (%.3g)" % (d_ls["v0"], d_ls["s_pose"]), d_ls["v0"] > 1e-4 and d_ls["s_pose"] > 1e-4)
s_vs_own, s_vs_log = maxd(TS, TO, ["v0", "s_pose", "t_pose"]), maxd(TS, TL, ["v0", "s_pose", "t_pose"])
check("3a speed_logged: positions and headings (s_pose, t_pose) equal own exactly (%.3g, %.3g); v0 equals logged exactly (%.3g) and differs from own (%.3g)"
      % (s_vs_own["s_pose"], s_vs_own["t_pose"], s_vs_log["v0"], s_vs_own["v0"]), s_vs_own["s_pose"] == 0.0 and s_vs_own["t_pose"] == 0.0 and s_vs_log["v0"] == 0.0 and s_vs_own["v0"] > 1e-4)
p_vs_own, p_vs_log = maxd(TP, TO, ["v0", "s_pose", "t_pose"]), maxd(TP, TL, ["v0", "s_pose", "t_pose"])
check("3b path_logged: v0 equals own exactly (%.3g); s_pose and t_pose equal logged exactly (%.3g, %.3g) and differ from own (%.3g)"
      % (p_vs_own["v0"], p_vs_log["s_pose"], p_vs_log["t_pose"], p_vs_own["s_pose"]), p_vs_own["v0"] == 0.0 and p_vs_log["s_pose"] == 0.0 and p_vs_log["t_pose"] == 0.0 and p_vs_own["s_pose"] > 1e-4)
planted("3 speed_logged built as path_logged (v0 equal to own)", s_vs_own["v0"] == 0.0)


def grp(name, kind="model"):
    return PL[name]["t0_state_first_tick_max_abs_diff"].get(kind, {}).get("max_abs", {})


gl, gs, gp = grp("logged"), grp("speed_logged"), grp("path_logged")
check("3c log line, speed_logged: positions 0, headings 0, speed above 0 (%s)" % {k: gs.get(k) for k in ("positions_m", "headings_rad", "speed_ms")}, bool(gs) and gs["positions_m"] == 0.0 and gs["headings_rad"] == 0.0 and gs["speed_ms"] > 0.0)
check("3c log line, path_logged: speed 0, positions above 0 or headings above 0 (%s)" % {k: gp.get(k) for k in ("positions_m", "headings_rad", "speed_ms")}, bool(gp) and gp["speed_ms"] == 0.0 and (gp["positions_m"] > 0.0 or gp["headings_rad"] > 0.0))
check("3c log line, logged: all three groups above 0 (%s)" % {k: gl.get(k) for k in ("positions_m", "headings_rad", "speed_ms")}, bool(gl) and gl["positions_m"] > 0.0 and gl["headings_rad"] > 0.0 and gl["speed_ms"] > 0.0)
for n in ("logged", "speed_logged", "path_logged"):
    txt = open(os.path.join(W, n, "t1.log"), encoding="utf-8", errors="replace").read()
    check("3c the run log of %s carries the ZZ T0-STATE line for mode %s" % (n, n), ("ZZ T0-STATE mode=%s kind=model" % n) in txt)
for n in ("logged", "speed_logged", "path_logged"):
    wo_, po2 = maxdiff_dicts(OFF["own"], OFF[n])
    check("3d pre-roll offsets of %s equal those of own exactly (largest difference %.3g)" % (n, wo_), wo_ == 0.0 and not po2, str(po2[:3]))

# ---------------------------------------------------------------- 4 the refusal
check("4 --t0-state logged with --preroll-steps 0 is REFUSED (return code %s)" % ref_rc, (ref_rc is None or ref_rc != 0) and "REFUSED: --t0-state logged goes with --preroll-steps above 0" in ref_text, ref_text.strip().splitlines()[-1][:160] if ref_text.strip() else "no output")

print("ZZ CHECK-HANDOVER-SRC " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
sys.exit(0 if not FAILED else 1)
