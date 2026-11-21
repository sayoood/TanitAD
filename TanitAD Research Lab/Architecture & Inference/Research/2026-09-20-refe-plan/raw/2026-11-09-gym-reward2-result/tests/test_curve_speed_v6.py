"""curve_speed fix tests (CURVE_SPEED_DEFECT.md). CPU only. Pattern as pilot_v5/scorer_gym/tests/test_v5.py:
check_<rule>(variant) must PASS for the real code and RAISE AssertionError for a deliberately mutated one (regression arm, must be RED).
Runner: python test_curve_speed_v6.py [names]  ->  PASS/FAIL lines + TESTS_DONE fails=N
Inputs: raw/cl/cl_<idx>.npz (centrelines of the real nuPlan map, written by curve_speed_probe.py from the same calls GymState makes),
the stored v5 records (pilot_v5/raw/*/states.jsonl.gz), and for the reachability test one real state build (state 108, ~3 min, nuPlan maps on F:)."""
import glob, gzip, importlib.util, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # pilot_v6
V5 = os.path.join(os.path.dirname(HERE), "pilot_v5")
sys.path.insert(0, os.path.join(HERE, "scorer_gym"))
import gym_env as g6                                                          # the v6 copy (switch)


def _load_v5():
    spec = importlib.util.spec_from_file_location("gym_env_v5", os.path.join(V5, "scorer_gym", "gym_env.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


def centrelines():
    out = {}
    for sub in ("cl_fast", "cl"):          # cl = full GymPDM.plan_all build (26 states); cl_fast = same centreline without proposal simulation (158 states; identical on all 26 overlaps)
        for f in sorted(glob.glob(os.path.join(HERE, "raw", sub, "cl_*.npz"))):
            z = np.load(f)
            out[int(os.path.basename(f)[3:-4])] = dict(pts=z["pts"], s=z["s"], s0=float(z["s0"]), limit=float(z["limit"]))
    return out


def stored(idx):
    sub = "run_main" if idx < 20 else "ladder_20_160"
    for l in gzip.open(os.path.join(V5, "raw", sub, "states.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(l)
        if r["idx"] == idx:
            return r


def s_ends(rec, s0, limit):
    """cycle-end path position per rung, from stored progress_legacy (approximation; exact unless progress clipped at 1.5)"""
    out = []
    for name, d in rec["ladder"].items():
        cum = 0.0
        for r in d["logs"]:
            cum += r["progress_legacy"] * limit * 0.5
            out.append((name, r["cycle"], s0 + cum, r))
    return out


def ev(mod, cl, arc):
    """a bare object carrying exactly what RewardEvaluator.curve_speed reads (no map/scenario needed)"""
    ns = types.SimpleNamespace(cl_pts=cl["pts"], cl_s=cl["s"])
    if arc != "v5attr_missing":
        ns.curve_min_seg_m = arc
    return ns


def cs(mod, cl, s0, arc=None):
    return mod.RewardEvaluator.curve_speed(ev(mod, cl, arc), s0)


# ---------------------------------------------------------------- T1: switch OFF == v5, bit for bit
def check_off_identical_to_v5(variant="real"):
    g5 = _load_v5()
    CL = centrelines()
    assert CL, "no centrelines: run curve_speed_probe.py first"
    n = 0
    for idx, cl in CL.items():
        grid = list(np.arange(cl["s0"], min(cl["s"][-1], cl["s0"] + 90.0), 0.37))
        rec = stored(idx)
        if rec is not None:
            grid += [se for _, _, se, _ in s_ends(rec, cl["s0"], cl["limit"])]
        for s0 in grid:
            a = g5.RewardEvaluator.curve_speed(types.SimpleNamespace(cl_pts=cl["pts"], cl_s=cl["s"]), s0)
            b = cs(g6, cl, s0, arc=None)
            if variant == "mutant_default_on":                 # mutant: the default switches the fix ON
                b = cs(g6, cl, s0, arc=0.05)
            assert a == b, (idx, s0, a, b)                      # exact float equality
            n += 1
    return n


# ---------------------------------------------------------------- T2: switch ON removes the defect on affected cycles, leaves clean cycles alone
SENSIBLE_KMAX = 0.5          # rad/m: implied curvature (k = a_lat / vstar^2, a_lat = 2) must be <= 0.5, i.e. radius >= 2 m
CTRL_MAX_REL_MOVE = 0.01     # stated bound: on a cycle whose 30 m window holds no segment shorter than 0.05 m, |vstar_fix - vstar_v5| / vstar_v5 <= 1 %
MIN_SEG = 0.05


def per_state_vstars(cl, rec, arc):
    out = []
    for name, c, se, r in s_ends(rec, cl["s0"], cl["limit"]):
        v5 = min(cl["limit"], cs(g6, cl, se, arc=None))
        v6 = min(cl["limit"], cs(g6, cl, se, arc=arc))
        sd = cl["s"][1:][np.diff(cl["s"]) < MIN_SEG]
        short = bool(((sd >= se) & (sd <= se + 30.0)).any())          # window holds a segment shorter than the fix's threshold
        out.append((name, c, se, v5, v6, r, short))
    return out


def check_fix_on(arc=MIN_SEG, switch_connected=True):
    CL = centrelines()
    a = arc if switch_connected else None
    n_aff = n_ctrl = n_ctrl_states = 0
    worst = 0.0
    for idx, cl in CL.items():
        rec = stored(idx)
        if rec is None:
            continue
        had_ctrl = False
        for name, c, se, v5, v6, r, short in per_state_vstars(cl, rec, a):
            if v5 < 1.0 and r["lead_gap"] is None and r["conf_ttc"] is None:       # the cycles where v5 skips q_free
                n_aff += 1
                assert v6 >= 1.0, ("affected cycle still < 1.0 m/s with fix on", idx, name, c, v5, v6)
                k6 = 2.0 / (v6 * v6) if v6 < 99 else 0.0
                assert k6 <= SENSIBLE_KMAX, ("implied curvature not sensible", idx, name, c, k6)
            if not short:                                                          # control cycles
                n_ctrl += 1; had_ctrl = True
                rel = abs(v6 - v5) / v5
                worst = max(worst, rel)
                assert rel <= CTRL_MAX_REL_MOVE, ("control cycle moved more than the stated bound", idx, name, c, v5, v6)
        n_ctrl_states += had_ctrl
    assert n_aff >= 500 and n_ctrl >= 300, (n_aff, n_ctrl)
    return dict(affected_cycles=n_aff, control_cycles=n_ctrl, control_states=n_ctrl_states, control_worst_rel_move=worst)


def check_fix_on_connected_mutant():
    """mutant: the switch is disconnected (value ignored) -> the 'on' test must go RED"""
    return check_fix_on(switch_connected=False)


# ---------------------------------------------------------------- T3: stored applicability is reproduced by the OFF path (R1-style control) and
#                              the cause is what the probe says: a zero-length segment
def check_off_reproduces_stored_applicability():
    n = bad = 0
    for idx, cl in centrelines().items():
        rec = stored(idx)
        if rec is None:
            continue
        for name, c, se, v5, v6, r, short in per_state_vstars(cl, rec, 0.05):
            if r["lead_gap"] is None and r["conf_ttc"] is None and not r["hard"]:
                n += 1
                bad += int((v5 >= 1.0) != bool(r["active"]["q_free"]))
    assert n > 200 and bad <= 0.01 * n, (n, bad)           # 1 % allowance for the reconstructed s_end
    return n, bad


# ---------------------------------------------------------------- T4: reachability, real caller -> q_free (state 108 build + one real rollout)
def check_reachability_real_caller(connected=True):
    """run_v5._do_state -> g.GymState(curve_min_seg_m=) -> RewardEvaluator(curve_min_seg_m=) -> .cycle -> vstar = min(limit, self.curve_speed(s_end))
    -> `if lead is None and conf_ttc is None and vstar >= 1.0 ...: act['q_free']=True`.  Runs the REAL crawl rung of state 108 (stored: q_free never
    applicable); with the switch connected q_free must be applicable on every free cycle, with it disconnected not."""
    import run_v5
    sts = json.load(open(os.path.join(V5, "PREREG_REWARD_1_STATES.json"), encoding="utf-8"))
    S = g6.GymState(sts[108], fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=0.20, curve_min_seg_m=0.05 if connected else None)
    if not connected:
        S.evaluator.curve_min_seg_m = None                          # mutant: flag stored but not read by the evaluator
    assert (S.evaluator.curve_min_seg_m == 0.05) == connected
    name, vt = S.ladder_targets()[0]
    assert name == "crawl1.5"
    logs = S.ladder_rollout(vt, "log")
    free = [r for r in logs if r["lead_gap"] is None and r["conf_ttc"] is None and not r["hard"]]
    act = [r["active"]["q_free"] for r in free]
    assert len(free) >= 15
    assert all(act), ("q_free not applicable on every free cycle", act)
    ret = g6.episode_return(logs)
    return ret


def check_reachability_off_matches_stored():
    """switch OFF through the same caller reproduces the stored v5 crawl return of state 108 bit for bit (determinism control)"""
    sts = json.load(open(os.path.join(V5, "PREREG_REWARD_1_STATES.json"), encoding="utf-8"))
    S = g6.GymState(sts[108], fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=0.20)          # curve_min_seg_m default
    assert S.evaluator.curve_min_seg_m is None
    name, vt = S.ladder_targets()[0]
    logs = S.ladder_rollout(vt, "log")
    st = stored(108)["ladder"]["crawl1.5"]["ret"]
    ret = g6.episode_return(logs)
    assert ret == st, (ret, st)
    assert not any(r["active"]["q_free"] for r in logs)
    return ret


def check_fix_on_source_mutant():
    """source-level mutant: delete the single branch in RewardEvaluator.curve_speed that reads the switch (module text patched in memory)."""
    global g6
    src = open(os.path.join(HERE, "scorer_gym", "gym_env.py"), encoding="utf-8").read()
    old = "        if self.curve_min_seg_m is not None:\n            return curve_speed_dedup("
    assert src.count(old) == 1
    mut = types.ModuleType("gym_env_mut"); mut.__file__ = os.path.join(HERE, "scorer_gym", "gym_env.py")
    exec(compile(src.replace(old, "        if False:\n            return curve_speed_dedup("), mut.__file__, "exec"), mut.__dict__)
    keep, g6 = g6, mut
    try:
        return check_fix_on()
    finally:
        g6 = keep


TESTS = {
    "fix_on_source_mutant_red": lambda: raises(check_fix_on_source_mutant) or (_ for _ in ()).throw(AssertionError("mutant not RED")),
    "off_identical": lambda: (check_off_identical_to_v5("real"), None)[0],
    "off_identical_mutant_red": lambda: raises(check_off_identical_to_v5, "mutant_default_on") or (_ for _ in ()).throw(AssertionError("mutant not RED")),
    "off_reproduces_stored": check_off_reproduces_stored_applicability,
    "fix_on": check_fix_on,
    "fix_on_mutant_red": lambda: raises(check_fix_on_connected_mutant) or (_ for _ in ()).throw(AssertionError("mutant not RED")),
    "reach_off_matches_stored": check_reachability_off_matches_stored,
    "reach_on": lambda: check_reachability_real_caller(True),
    "reach_mutant_red": lambda: raises(check_reachability_real_caller, False) or (_ for _ in ()).throw(AssertionError("mutant not RED")),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    fails = 0
    for n in names:
        t0 = time.perf_counter()
        try:
            res = TESTS[n]()
            print(f"PASS {n} ({time.perf_counter() - t0:.0f}s) {res}", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {n}: {e!r}", flush=True)
    print(f"TESTS_DONE fails={fails}", flush=True)
