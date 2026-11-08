"""REWARD-1 tests. USED states and synthetic rows only: seed 20261110 is never drawn here (selection is tested with a dummy seed).
Pattern: check_<rule>(variant) must PASS for the real rule and must raise AssertionError for a deliberately mutated rule (regression arm, must be RED).
Runner: python test_v5.py [test names] -> PASS/FAIL lines + TESTS_DONE fails=N."""
import json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import gym_env as g
import reward1 as r1

V4 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4"


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


# ------------------------------------------------------------ constants named in the card
def test_card_constants():
    assert r1.SEED_V5 == 20261110 and r1.FLOOR == 0.20 and r1.MIN_LIMIT == 4.0 and r1.MIN_ELIGIBLE == 5
    assert (r1.N_FULL, r1.N_FIRST, r1.BLOCK, r1.CAP) == (20, 160, 40, 240)
    assert g.Q_FREE_FLOOR_M_GYM_4 == 0.30 and g.Q_FREE_FLOOR_REWARD_1 == 0.20


# ------------------------------------------------------------ the q_free term: floor is the only thing that moved
def check_qfree(fn):
    v = 10.0
    # below the knee, deep: the floor decides (vbar = 0.34 v*, state 3's crawl)
    assert fn(3.4, v, 0.30) == 0.30 and fn(3.4, v, 0.20) == 0.20 and fn(3.4, v, 0.15) == 0.15
    # above the floor the value must NOT depend on the floor (slope 0.7, knee 0.9 v* unchanged): vbar = 0.7 v* -> 1 - 0.7 * 0.2/0.4 = 0.65
    assert abs(fn(7.0, v, 0.30) - 0.65) < 1e-12 and abs(fn(7.0, v, 0.20) - 0.65) < 1e-12
    assert fn(9.0, v, 0.20) == 1.0 and fn(10.0, v, 0.20) == 1.0
    # floor 1.0 holds the term at 1.0 (R2 arm)
    assert fn(3.4, v, 1.0) == 1.0 and fn(7.0, v, 1.0) == 1.0


def test_qfree_floor_value_and_regression_arm():
    check_qfree(g.q_free_value)
    mutant_ignores_floor = lambda vbar, vstar, floor: g.q_free_value(vbar, vstar, 0.30)
    assert raises(check_qfree, mutant_ignores_floor), "mutant that ignores the floor was not caught"
    mutant_slope = lambda vbar, vstar, floor: float(np.clip(1.0 - 0.8 * (0.9 * vstar - vbar) / (0.4 * vstar), floor, 1.0))
    assert raises(check_qfree, mutant_slope), "mutant slope was not caught"


# ------------------------------------------------------------ reachability: flag -> evaluator -> reward of a real rollout (USED state 3, crawl rung)
_S = {}


def used_state(idx, floor):
    if (idx, floor) not in _S:
        _S[(idx, floor)] = g.GymState(g.load_pinned_states()[idx], fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=floor)
    return _S[(idx, floor)]


def crawl(S):
    logs = S.ladder_rollout(1.5, "log")
    return g.episode_return(logs), logs


def check_floor_changes_crawl(S30, S20, ret30_stored):
    r30, l30 = crawl(S30)
    r20, l20 = crawl(S20)
    assert r30 == ret30_stored, ("floor 0.30 must equal the stored M-GYM-4 value exactly", r30, ret30_stored)
    assert r20 != r30 and r20 < r30, ("floor 0.20 must lower the crawl's return", r20, r30)
    assert abs(r20 - 0.993665162893293) / 0.993665162893293 < 0.01       # b3_decomp/attrib.json floor_cf['3']['0.2']
    q30 = [x["q"]["q_free"] for x in l30 if x["active"]["q_free"]]
    q20 = [x["q"]["q_free"] for x in l20 if x["active"]["q_free"]]
    assert min(q30) == 0.30 and min(q20) == 0.20, (min(q30), min(q20))


def test_floor_reachable_from_flag_to_crawl_return_and_regression_arm():
    stored = json.load(open(V4 + "/raw/results_v4.json", encoding="utf-8"))["ladder"]["per_state"][3]["rets"]["crawl1.5"]
    assert stored == 1.4904977443399394
    S30, S20 = used_state(3, 0.30), used_state(3, 0.20)
    assert S30.evaluator.q_free_floor == 0.30 and S20.evaluator.q_free_floor == 0.20
    assert g.GymState.__init__.__defaults__[-1] == 0.30 and g.RewardEvaluator.__init__.__defaults__[-1] == 0.30      # default = M-GYM-4
    check_floor_changes_crawl(S30, S20, stored)
    # regression arm: a flag that never reaches the term (evaluator floor pinned back to 0.30) must be caught as RED
    Sdead = g.GymState(g.load_pinned_states()[3], fixes="on", ttc_mode=None, ttc_shadow=True, q_free_floor=0.20)
    Sdead.evaluator.q_free_floor = 0.30
    assert raises(check_floor_changes_crawl, S30, Sdead, stored), "a disconnected floor flag was not caught"


# ------------------------------------------------------------ eligibility + B3 verdict + block extension (pure logic, each with mutants)
def R(idx, inf, known, limit):
    return dict(idx=idx, informative=inf, known=known, limit=limit)


def check_eligibility(elig):
    assert elig(True, 4.47) and elig(True, 4.0) and elig(True, 13.4)
    assert not elig(True, 3.99) and not elig(True, 2.24) and not elig(False, 13.4) and not elig(False, 2.0)


def check_b3(verdict_fn):
    good = [R(i, True, True, 8.0) for i in range(5)]
    # an informative low-limit state that is NOT known must not enter the verdict (and must not turn PASS into FAIL)
    assert verdict_fn(good + [R(9, True, False, 2.24)], True)["verdict"] == "PASS"
    assert verdict_fn(good + [R(9, True, False, 2.24)], True)["n_eligible"] == 5
    # one eligible state not known -> FAIL
    assert verdict_fn(good[:4] + [R(5, True, False, 4.47)], False)["verdict"] == "FAIL"
    # 4 eligible: CONTINUE before the cap, UNDETERMINED at the cap, never PASS/FAIL
    four = good[:4] + [R(7, False, None, 9.0), R(8, True, True, 3.0)]
    assert verdict_fn(four, False)["verdict"] == "CONTINUE" and verdict_fn(four, True)["verdict"] == "UNDETERMINED"
    assert verdict_fn(four + [R(10, True, False, 5.0)], True)["verdict"] == "FAIL"


def test_eligibility_rule_and_regression_arms():
    check_eligibility(r1.eligible)
    assert raises(check_eligibility, lambda inf, lim: bool(inf)), "mutant: no limit restriction not caught"
    assert raises(check_eligibility, lambda inf, lim: bool(inf) and lim > 4.0), "mutant: strict > 4.0 not caught"
    assert raises(check_eligibility, lambda inf, lim: True), "mutant: everything eligible not caught"


def test_b3_verdict_and_regression_arms():
    check_b3(r1.b3_verdict)
    no_restriction = lambda rows, at_cap: r1.b3_verdict(rows, at_cap, elig=lambda inf, lim: bool(inf))
    assert raises(check_b3, no_restriction), "mutant: low-limit states entering the verdict not caught"
    min2 = lambda rows, at_cap: r1.b3_verdict(rows, at_cap, min_eligible=2)
    assert raises(check_b3, min2), "mutant: minimum of 2 eligible not caught"
    min0 = lambda rows, at_cap: r1.b3_verdict(rows, at_cap, min_eligible=1)
    assert raises(check_b3, min0), "mutant: minimum of 1 eligible not caught"


def check_blocks(fn):
    assert fn(4, 160) == (160, 200) and fn(4, 200) == (200, 240)
    assert fn(0, 160) == (160, 200)
    assert fn(4, 240) is None and fn(5, 160) is None and fn(9, 200) is None


def test_block_extension_and_regression_arms():
    check_blocks(r1.next_block)
    assert raises(check_blocks, lambda n, k: r1.next_block(n, k, cap=10 ** 6)), "mutant: no cap not caught"
    assert raises(check_blocks, lambda n, k: r1.next_block(n, k, min_eligible=0)), "mutant: never extend not caught"
    assert raises(check_blocks, lambda n, k: r1.next_block(n, k, block=10)), "mutant: block of 10 not caught"
    assert raises(check_blocks, lambda n, k: r1.next_block(n, k, min_eligible=6)), "mutant: threshold 6 not caught"


# ------------------------------------------------------------ the fixed ordered list (dummy seed; 20261110 is NOT drawn here)
def test_selection_v5_structure_with_dummy_seed():
    seed = 7
    ex = r1.excluded_logs()
    assert len(ex) >= 30
    a = r1.select_states_v5(seed=seed, exclude_logs=ex)
    assert len(a) == 240 and len({s["token"] for s in a}) == 240
    base = g.select_states_v4(seed=seed, n_random=10, n_stall_logs=5, exclude_logs=ex)
    assert [{k: v for k, v in s.items() if k not in ("idx", "tier", "block")} for s in a[:20]] == base      # first 20 = exactly the M-GYM-4 recipe
    assert sum(s["sampler"] == "random" for s in a[:160]) == 80 and sum(s["sampler"] != "random" for s in a[:160]) == 80
    assert sum(s["sampler"] == "random" for s in a[160:]) == 40 and sum(s["sampler"] != "random" for s in a[160:]) == 40
    assert [s["idx"] for s in a] == list(range(240)) and [s["tier"] for s in a[:20]] == ["full"] * 20 and {s["tier"] for s in a[20:]} == {"ladder"}
    assert {s["block"] for s in a[:160]} == {0} and {s["block"] for s in a[160:200]} == {1} and {s["block"] for s in a[200:]} == {2}
    assert not ({s["log"] for s in a} & set(ex))
    assert not ({s["log"] for s in a if s["sampler"] == "random"} & {s["log"] for s in a if s["sampler"] != "random"})
    assert not any(s["log"] in g.navtest_log_names() for s in a)
    assert a == r1.select_states_v5(seed=seed, exclude_logs=ex)                    # deterministic
    # stall states come in (stop_and_go, junction) pairs of one log, so a block boundary never splits a log
    st = [s for s in a if s["sampler"] != "random"]
    assert all(st[i]["log"] == st[i + 1]["log"] and st[i]["sampler"] == "stop_and_go" and st[i + 1]["sampler"] == "junction" for i in range(0, 120, 2))
    # the section-0 exclusion is real: the M-GYM-4 logs and the M-GYM-2 logs are in it
    v4 = {s["log"] for s in json.load(open(r1.V4_STATES, encoding="utf-8"))}
    assert v4 <= set(ex) and set(g.used_logs_m_gym_2()) <= set(ex)
    # regression arm: the exclusion check must go RED on a list that contains a state from an excluded log
    def check_excl(lst):
        assert not ({s["log"] for s in lst} & set(ex)), "state on an excluded log"
    check_excl(a)
    bad = [dict(a[0], log=ex[0])] + a[1:]
    assert raises(check_excl, bad), "exclusion check did not go RED on a leaked log"


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
