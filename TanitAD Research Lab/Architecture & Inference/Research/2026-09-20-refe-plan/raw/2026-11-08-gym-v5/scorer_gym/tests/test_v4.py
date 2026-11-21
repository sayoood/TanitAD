"""M-GYM-4 tests. USED states only (main / stall sets of M-GYM-2): the fresh set is never touched here.
Pattern per fix: check_<fix>(switch) must PASS with the fix on and must raise AssertionError with the fix off (deliberate-regression arm),
and the fix-off path must equal the pilot_v2 behaviour (control).  Runner: python test_v4.py -> PASS/FAIL lines + TESTS_DONE fails=N."""
import gzip, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import gym_env as g

V2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2"
V3 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v3_diag/raw"


def stored(run):
    return {json.loads(l)["idx"]: json.loads(l) for l in gzip.open(f"{V2}/{run}/states.jsonl.gz", "rt", encoding="utf-8")}


def raises(fn, *a):
    try:
        fn(*a)
    except AssertionError:
        return True
    return False


_ST = {}


def state(sampler, idx, **kw):
    key = (sampler, idx, tuple(sorted(kw.items())))
    if key not in _ST:
        sel = g.select_states if sampler == "main" else g.select_states_stall
        _ST[key] = g.GymState(sel(20, 10)[idx], **kw)
    return _ST[key]


# ---------------------------------------------------------------- fix 1: oracle = fewest hard events, then return
def check_oracle(rule):
    st = stored("run_stall")[7]                         # stall 7: argmax arm 23 collides at cycle 16; 6 arms are clean
    rets = [st["arms"][f"log:{k}"]["ret"] for k in range(24)]
    hards = [st["arms"][f"log:{k}"]["logs"][-1]["hard"] is not None for k in range(24)]
    assert any(not h for h in hards) and any(h for h in hards)
    k = g.oracle_pick(rets, hards, rule)
    assert not hards[k], ("oracle arm has a hard event", k, rets[k])


def test_fix1_oracle_lexicographic_and_regression_arm():
    check_oracle("lex")                                 # on: clean
    assert raises(check_oracle, "argmax")               # off (= PREREG wording): picks hard arm 23 -> RED, as M-GYM-2 measured


def test_fix1_control_identical_when_argmax_arm_is_clean():
    for run in ("run_main", "run_stall"):
        for idx, st in stored(run).items():
            for mode in ("log", "idm"):
                rets = [st["arms"][f"{mode}:{k}"]["ret"] for k in range(24)]
                hards = [st["arms"][f"{mode}:{k}"]["logs"][-1]["hard"] is not None for k in range(24)]
                a = g.oracle_pick(rets, hards, "argmax")
                if not hards[a]:
                    assert g.oracle_pick(rets, hards, "lex") == a, (run, idx, mode)


# ---------------------------------------------------------------- fix 4: all-tie cycle-0 cells label-free
def check_tie(tie_rule):
    st = stored("run_main")[19]                         # main 19: all 24 arms hard at cycle 0, return exactly 0.0
    rets = [st["arms"][f"log:{k}"]["ret"] for k in range(24)]
    hards = [st["arms"][f"log:{k}"]["logs"][-1]["hard"] is not None for k in range(24)]
    ncy = [len(st["arms"][f"log:{k}"]["logs"]) for k in range(24)]
    assert set(rets) == {0.0} and set(ncy) == {1}
    c = g.cell_status(rets, hards, ncy, tie_rule)
    assert c["label_free_tie"] and c["unavoidable"]


def test_fix4_all_tie_cell_label_free_and_regression_arm():
    check_tie("label_free")
    assert raises(check_tie, "none")
    st = stored("run_stall")[10]                        # all 24 hard but NOT a cycle-0 tie: unavoidable, not label-free
    rets = [st["arms"][f"log:{k}"]["ret"] for k in range(24)]
    hards = [True] * 24
    ncy = [len(st["arms"][f"log:{k}"]["logs"]) for k in range(24)]
    c = g.cell_status(rets, hards, ncy)
    assert c["unavoidable"] and not c["label_free_tie"]
    st = stored("run_main")[1]
    rets = [st["arms"][f"log:{k}"]["ret"] for k in range(24)]
    c = g.cell_status(rets, [False] * 24, [17] * 24)
    assert not c["unavoidable"] and not c["label_free_tie"]


# ---------------------------------------------------------------- fix 3: map pre-screen
def check_prescreen(tol):
    old = g.PRESCREEN_TOL_M
    g.PRESCREEN_TOL_M = tol
    try:
        S = g.GymState(g.select_states(20, 10)[10], fixes="on")          # main 10: 0.254 m at s=21 m (M_GYM_3_DIAG)
        ev = S.evaluator
        assert 0.20 < ev.centreline_overshoot_m < 0.30, ev.centreline_overshoot_m
        assert ev.map_unreliable_prescreen, (ev.centreline_overshoot_m, tol)
        assert ev.map_slack_m == 0.0 and g.MAX_MAP_SLACK_M == 1.0           # slack cap NOT widened, no slack granted
    finally:
        g.PRESCREEN_TOL_M = old


def test_fix3_map_prescreen_flags_main10_and_regression_arm():
    check_prescreen(g.PRESCREEN_TOL_M)
    assert raises(check_prescreen, 0.30)                # tolerance above the measured 0.254 m -> state not flagged -> RED
    S_off = g.GymState(g.select_states(20, 10)[10], fixes="off")
    assert S_off.evaluator.centreline_overshoot_m == 0.0 and not S_off.evaluator.map_unreliable_prescreen   # control: off = pilot_v2


def test_fix3_prescreen_does_not_flag_clean_state():
    S = state("main", 1)
    assert not S.evaluator.map_unreliable_prescreen, S.evaluator.centreline_overshoot_m


# ---------------------------------------------------------------- fix 2: ttc predicate needs the object inside the ego corridor
def first_ttc(S, k, mode="log"):
    logs = S.rollout(k, mode)
    return [(r["cycle"], r["hard_step"]) for r in logs if r["hard"] == "ttc"], logs


def check_ttc_corridor(half_w):
    old = g.TTC_CORR_HALF_W
    g.TTC_CORR_HALF_W = half_w
    try:
        S = state("stall", 15, fixes="on")
        for k in (2, 6, 18):                            # stall 15: pure false alarms (legacy flags these arms; no collision with ttc off)
            h, _ = first_ttc(S, k)
            assert not h, (k, h)
    finally:
        g.TTC_CORR_HALF_W = old


def test_fix2_ttc_corridor_clears_stall15_false_alarms_and_regression_arm():
    S_leg = state("stall", 15, fixes="off")
    for k in (2, 6, 18):
        h, _ = first_ttc(S_leg, k)
        assert h, ("control: legacy predicate must flag arm", k)       # M_GYM_3_DIAG: stall 15 baseline 13/24 arms ttc
    check_ttc_corridor(g.TTC_CORR_HALF_W)
    assert raises(check_ttc_corridor, 100.0)            # corridor so wide it equals legacy -> flags again -> RED


def check_hazard_kept(mode):
    # derivation (explore_corridor, ttc OFF): stall 10 arm 9 collides at global tick 42 (cycle 8) and the corridor predicate flags it at tick 31;
    # stall 7 arm 3 collides at tick 74 (cycle 14) and is flagged at tick 69 (cycle 13).
    for sampler_idx, k, coll_cycle in ((10, 9, 8), (7, 3, 14)):
        S = state("stall", sampler_idx, fixes="on", ttc_mode=mode)
        logs = S.rollout(k, "log")
        assert logs[-1]["hard"] == "ttc" and logs[-1]["cycle"] < coll_cycle, (sampler_idx, k, logs[-1]["hard"], logs[-1]["cycle"])


def test_fix2_corridor_keeps_stall10_and_stall7_collisions_flagged():
    check_hazard_kept("corridor")                       # real hazards still flagged as ttc, before the collision cycle
    assert raises(check_hazard_kept, "off")             # predicate off -> the same arms end in a COLLISION, not ttc -> RED


# ---------------------------------------------------------------- fix 6: constant-speed ladder as regression arm
def test_fix6_ladder_known_function_and_regression_arms():
    d = json.load(open(f"{V3}/q3_stall9.json", encoding="utf-8"))["runs"]
    rets = {k: v["ret"] for k, v in d.items()}
    hards = {k: any(l["hard"] for l in v["logs"]) for k, v in d.items()}
    assert g.ladder_known(rets, hards) == (True, True)
    flat = {k: 5.0 for k in rets}                                        # reward that does not pay speed -> must NOT read known
    assert g.ladder_known(flat, {k: False for k in rets}) == (True, False)
    rev = dict(zip(rets, list(rets.values())[::-1]))                     # reversed ladder -> not known
    assert g.ladder_known(rev, {k: False for k in rets})[1] is False
    h2 = dict(hards); h2["f0.7"] = True                                  # hard event on a ladder rung -> uninformative, not a pass
    assert g.ladder_known(rets, h2) == (False, None)


def test_fix6_ladder_policy_reproduces_pilot_v3_q3_when_fixes_off():
    S = state("stall", 9, fixes="off")
    d = json.load(open(f"{V3}/q3_stall9.json", encoding="utf-8"))["runs"]
    names = dict(S.ladder_targets())
    assert list(names) == ["crawl1.5", "f0.3", "f0.5", "f0.7", "f0.9", "f1.0"]
    for n in ("crawl1.5", "f0.5", "f1.0"):
        r = g.episode_return(S.ladder_rollout(names[n], "log"))
        assert r == d[n]["ret"], (n, r, d[n]["ret"])


# ---------------------------------------------------------------- control: all fixes off == pilot_v2, bit for bit
def test_control_fixes_off_reproduces_pilot_v2_returns():
    st = stored("run_main")
    S = state("main", 1, fixes="off")
    for k, mode in ((5, "log"), (12, "idm")):
        assert g.episode_return(S.rollout(k, mode)) == st[1]["arms"][f"{mode}:{k}"]["ret"], (k, mode)
    S7 = state("stall", 7, fixes="off")
    sst = stored("run_stall")[7]
    assert g.episode_return(S7.rollout(23, "log")) == sst["arms"]["log:23"]["ret"]


def test_v4_deterministic_same_process_fixes_on():
    S = state("main", 1, fixes="on")
    a = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "log", horizon_s=2.0)]
    b = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "log", horizon_s=2.0)]
    assert a == b


def test_v4_state_selection_seeded_disjoint_navtest_free():
    ex = g.used_logs_m_gym_2()
    a, b = g.select_states_v4(exclude_logs=ex), g.select_states_v4(exclude_logs=ex)
    assert a == b and len(a) == 20
    assert a == g.load_pinned_states()                  # the list pinned after the PREREG sha is exactly what the seed draws
    assert sum(s["sampler"] == "random" for s in a) == 10 and sum(s["sampler"] != "random" for s in a) == 10
    assert len({s["log"] for s in a}) == 15 and not ({s["log"] for s in a} & set(ex))
    assert not any(s["log"] in g.navtest_log_names() for s in a)
    old = {s["token"] for s in g.select_states(20, 10)} | {s["token"] for s in g.select_states_stall(20, 10)}
    assert not ({s["token"] for s in a} & old)
    # control: old samplers unchanged by this module
    assert g.select_states(20, 10) == g.select_states(20, 10) and len(g.select_states_stall(20, 10)) == 20


if __name__ == "__main__":
    fails = 0
    only = set(sys.argv[1:])
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and (not only or name in only):
            t = time.time()
            try:
                fn()
                print("PASS", name, "%.1fs" % (time.time() - t), flush=True)
            except Exception as e:
                import traceback
                fails += 1
                print("FAIL", name, repr(e), flush=True)
                traceback.print_exc()
    print("TESTS_DONE fails=%d" % fails, flush=True)
    sys.exit(fails)
