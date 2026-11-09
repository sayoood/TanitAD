"""REWARD-2 tests (pilot_v6). USED states and synthetic rows only: seed 20261111 is NEVER drawn or read here (the guard is exercised with an EMPTY database directory, so even
a mutant that removes the guard cannot load a state; selection capacity is tested with the dummy seed 7).
Pattern: check_<rule>(variant) must PASS for the real code and RAISE AssertionError for a deliberately mutated one (regression arm, must be RED).
Runner: python test_v6.py [test names] -> PASS/FAIL lines + TESTS_DONE fails=N."""
import contextlib, gzip, json, os, sys, tempfile, time, types
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # pilot_v6/scorer_gym
P6 = os.path.dirname(HERE)
P5 = os.path.join(os.path.dirname(P6), "pilot_v5")
sys.path.insert(0, HERE)
os.environ["MGYM6_DRY"] = "1"                         # test mode for this whole process (the draw gate refuses it)
import numpy as np
import gym_env as g
import draw_gate as dg
import reward1 as r1
import reward2 as r2
import ramguard
import chain_v6 as chain
import draw_v6
import r4_v6
import run_v6

SCRATCH = r"F:\tanitad_scratch\mgym6"


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


def tmpdir():
    os.makedirs(SCRATCH + r"\tmp_tests", exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH + r"\tmp_tests")


def mutated_module(path, old, new, name, count=1):
    src = open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    assert src.count(old) == count, (path, old, src.count(old))
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src.replace(old, new), path, "exec"), mod.__dict__)
    return mod


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------ constants named in the card
def test_card_constants_v6():
    assert r2.SEED_V6 == 20261111 and r2.CURVE_MIN_SEG_M == 0.05 and r2.FLOOR == 0.20 and r2.MIN_FREE_V0 == 1.0 and r2.CRAWL_RATIO == 0.10
    assert (r1.N_FULL, r1.N_FIRST, r1.BLOCK, r1.CAP, r1.MIN_ELIGIBLE, r1.MIN_LIMIT) == (20, 160, 40, 240, 5, 4.0)
    assert ramguard.MIN_AVAIL_GIB == 7.0 and ramguard.MAX_WORKERS == 4
    assert chain.ARC == "0.05" and chain.FLOOR == "0.20" and float(chain.ARC) == r2.CURVE_MIN_SEG_M
    assert dg.GUARDED_SEEDS == frozenset({20261111})
    # the frozen card file equals its sha256 file (the chain checks the same at start)
    want = open(os.path.join(P6, "PREREG_REWARD_2.sha256"), encoding="utf-8").read().split()[0]
    assert dg.sha_file(os.path.join(P6, "PREREG_REWARD_2.md")) == want
    # run_v6 refuses a missing or a wrong curve_min_seg_m BEFORE it reads any state list
    old = sys.argv
    try:
        sys.argv = ["run_v6.py", "ladder", "--floor", "0.20", "--out", "x", "--range", "20:21"]
        try:
            run_v6.main()
            raise RuntimeError("accepted a missing --curve-min-seg-m")
        except SystemExit:
            pass
        for bad in ("0.0", "0.1", "0.06"):
            sys.argv = ["run_v6.py", "ladder", "--floor", "0.20", "--curve-min-seg-m", bad, "--out", "x", "--range", "20:21"]
            assert raises(run_v6.main), bad
    finally:
        sys.argv = old


# ------------------------------------------------------------ the draw gate: dry/test mode cannot reach seed 20261111
def make_gate(tmp, **over):
    b = {}
    for k, name in (("prereg_sha", "p.md"), ("code_pin_sha", "pin.json"), ("exclusion_sha", "ex.json")):
        p = os.path.join(tmp, name)
        open(p, "w", encoding="utf-8").write("x" + name)
        b[k] = p
    gate = dict(r_all_pass=True, R1=True, R2=True, R3=True, R4=True, G_REPRO=True, **{k: dg.sha_file(p) for k, p in b.items()})
    gate.update(over)
    gp = os.path.join(tmp, "gate.json")
    json.dump(gate, open(gp, "w", encoding="utf-8"))
    return gp, b


def check_gate(gmod, dgmod=dg):
    """gmod: gym_env (real or mutated). Every case below uses an EMPTY DB_DIR, so nothing of the seed can be loaded even if a case wrongly passes."""
    with tmpdir() as t, tmpdir() as emptydb:
        assert os.listdir(emptydb) == []
        gmod.DB_DIR = emptydb
        old_gp, old_b = dgmod.GATE_PATH, dgmod.BOUND
        try:
            gp, b = make_gate(t)
            dgmod.GATE_PATH, dgmod.BOUND = gp, b
            S = 20261111

            def refused(label, fn, **e):
                with env(**e):
                    try:
                        fn()
                    except AssertionError:
                        return
                raise AssertionError(f"case {label} reached the draw")
            sel = lambda: gmod.select_states_v4(seed=S, n_random=2, n_stall_logs=1, exclude_logs=[])
            hs = lambda: gmod._h_seed(S, "", "x")
            refused("A dry mode with every other condition valid (select)", sel, MGYM6_DRY="1", MGYM6_DRAW_STAGE="1")
            refused("A2 dry mode (_h_seed)", hs, MGYM6_DRY="1", MGYM6_DRAW_STAGE="1")
            refused("B not the draw stage", sel, MGYM6_DRY=None, MGYM6_DRAW_STAGE=None)
            refused("B2 not the draw stage (_h_seed)", hs, MGYM6_DRY=None, MGYM6_DRAW_STAGE=None)
            dgmod.GATE_PATH = os.path.join(t, "absent.json")
            refused("C no gate file", sel, MGYM6_DRY=None, MGYM6_DRAW_STAGE="1")
            gp2, _ = make_gate(t, r_all_pass=False)
            dgmod.GATE_PATH = gp2
            refused("D gate records a failure", sel, MGYM6_DRY=None, MGYM6_DRAW_STAGE="1")
            gp3, _ = make_gate(t, R4=False)
            dgmod.GATE_PATH = gp3
            refused("D2 R4 not recorded", sel, MGYM6_DRY=None, MGYM6_DRAW_STAGE="1")
            gp4, _ = make_gate(t, prereg_sha="0" * 64)
            dgmod.GATE_PATH = gp4
            refused("E gate bound to another prereg", sel, MGYM6_DRY=None, MGYM6_DRAW_STAGE="1")
            # positive control: the one valid combination passes the gate (and returns [] because the database is empty)
            gp5, _ = make_gate(t)
            dgmod.GATE_PATH = gp5
            with env(MGYM6_DRY=None, MGYM6_DRAW_STAGE="1"):
                assert sel() == []
            # other seeds are never gated
            with env(MGYM6_DRY="1", MGYM6_DRAW_STAGE=None):
                assert gmod.select_states_v4(seed=7, n_random=1, n_stall_logs=0, exclude_logs=[]) == []
        finally:
            dgmod.GATE_PATH, dgmod.BOUND = old_gp, old_b
            gmod.DB_DIR = g.DB_DIR if gmod is not g else gmod.DB_DIR


def test_draw_gate_dry_and_mutants():
    real_db = g.DB_DIR
    try:
        check_gate(g)
    finally:
        g.DB_DIR = real_db
    # mutant 1: the call is a no-op at run time
    saved = g._draw_gate
    g._draw_gate = types.SimpleNamespace(require_open=lambda *a, **k: True)
    try:
        assert raises(check_gate, g), "mutant: no-op gate not caught"
    finally:
        g._draw_gate = saved
        g.DB_DIR = real_db
    # mutant 2: the two guard lines deleted from gym_env.py's source
    gsrc = os.path.join(HERE, "gym_env.py")
    gm = mutated_module(gsrc, "    _draw_gate.require_open(seed)\n    excl = set(exclude_logs or [])", "    excl = set(exclude_logs or [])", "gym_env_mut1")
    assert raises(check_gate, gm), "mutant: guard deleted from select_states_v4 not caught"
    gm2 = mutated_module(gsrc, "    _draw_gate.require_open(seed)         # v6: seed 20261111 is spent once read; only the chain's draw stage passes (draw_gate.py)\n",
                         "", "gym_env_mut2")
    # deleting only the _h_seed guard must be caught by the _h_seed cases (select's own guard still present, so the select cases pass)
    assert raises(check_gate, gm2), "mutant: guard deleted from _h_seed not caught"
    # mutant 3: draw_gate itself without the dry-mode assertion
    dsrc = os.path.join(HERE, "draw_gate.py")
    dm = mutated_module(dsrc, '    assert not os.environ.get("MGYM6_DRY"), "dry/test mode (MGYM6_DRY): the seed 20261111 states are spent once read, this process may not draw"\n', "", "draw_gate_mut")
    saved = g._draw_gate
    g._draw_gate = dm
    try:
        assert raises(check_gate, g, dm), "mutant: draw_gate without the dry assertion not caught"
    finally:
        g._draw_gate = saved
        g.DB_DIR = real_db
    assert dg.require_open(7) is True


# ------------------------------------------------------------ draw_v6 and the chain: a dry or test run cannot reach the draw
class Reached(Exception):
    pass


def check_draw_v6(mod):
    """mod: draw_v6 (real or mutant). In dry mode main(['draw']) must stop with an AssertionError BEFORE it calls the selector; the selector here is a sentinel (never the real draw)."""
    called = []

    def sentinel(**k):
        called.append(k)
        raise Reached()
    with tmpdir() as t:
        old = (r2.STATES_V6, r2.load_exclusion)
        r2.STATES_V6 = os.path.join(t, "none.json")
        r2.load_exclusion = lambda *a, **k: ["log"]
        try:
            with env(MGYM6_DRY="1"):
                try:
                    mod.main(["draw"], select=sentinel)
                except AssertionError:
                    assert not called
                    return
                except Reached:
                    pass
        finally:
            r2.STATES_V6, r2.load_exclusion = old
    raise AssertionError("dry mode reached the draw")


def check_plan(plan_fn):
    d, r = plan_fn(True), plan_fn(False)
    assert "DRAW" not in d and "STATES_PINNED" not in d and "MAIN" not in d and d[-1] == "DRY_STOP_BEFORE_DRAW"
    ix = {s: i for i, s in enumerate(r)}
    order = ["PREREG_OK", "PIN_OK", "TESTS", "EXCL", "R123", "R4", "GATE", "DRAW", "MAIN", "LADDER", "DET", "TTCOFF", "SUMMARY", "ALL_DONE"]
    assert all(ix[a] < ix[b] for a, b in zip(order, order[1:])), r
    assert ix["EXCL"] < ix["DRAW"] and ix["R4"] < ix["GATE"] < ix["DRAW"]


def test_draw_v6_and_chain_cannot_draw_in_dry():
    check_draw_v6(draw_v6)
    srcp = os.path.join(HERE, "draw_v6.py")
    mut = mutated_module(srcp, '    assert not os.environ.get("MGYM6_DRY"), "draw_v6 refuses in dry/test mode"\n', "", "draw_v6_mut")
    assert raises(check_draw_v6, mut), "mutant: draw_v6 without the dry assertion not caught"
    check_plan(chain.plan)
    assert raises(check_plan, lambda dry: list(chain.STAGES_REAL)), "mutant: plan ignoring dry not caught"
    swapped = list(chain.STAGES_REAL)
    i, j = swapped.index("R4"), swapped.index("DRAW")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    assert raises(check_plan, lambda dry: list(chain.STAGES_DRY) if dry else swapped), "mutant: draw before R4 not caught"
    with env(MGYM6_DRY="1", MGYM6_DRY_CHAIN=None):
        c = chain.Chain(True, 1)
        assert raises(c.s_draw), "a dry chain's draw stage did not refuse"
        assert os.environ.get("MGYM6_DRY") == "1"
    # the dry chain's children are in test mode and cannot become the draw stage
    assert c.env["MGYM6_DRY"] == "1" and "MGYM6_DRAW_STAGE" not in c.env
    # source level: the only place the draw stage variable is set is the draw stage of chain_v6
    src = open(os.path.join(HERE, "chain_v6.py"), encoding="utf-8").read()
    assert src.count('env["MGYM6_DRAW_STAGE"] = "1"') == 1 and src.count("assert not self.dry") >= 2
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") and f not in ("chain_v6.py", "draw_gate.py"):
            assert "MGYM6_DRAW_STAGE" not in open(os.path.join(HERE, f), encoding="utf-8").read() or f in ("draw_v6.py",), f


# ------------------------------------------------------------ the RAM guard
def check_ramguard(rule):
    assert rule(7.0, 7.0) and rule(8.0, 7.0) and not rule(6.99, 7.0) and not rule(0.0, 7.0)
    logs, seq = [], iter([3.0, 5.5, 6.9, 7.5])
    got = ramguard.wait_for_ram("unit", logs.append, min_gib=7.0, avail_fn=lambda: next(seq), poll_s=0, sleep=lambda s: None, rule=rule)
    assert got == 7.5 and len(logs) == 3 and all(l.startswith("RAM_WAIT") and "unit" in l for l in logs), logs
    # the pool dispatcher: nothing starts while RAM is low, every wait is logged, results come back
    logs2, res, n = [], [], [0]

    def avail():
        n[0] += 1
        return 3.0 if n[0] <= 3 else 8.0
    ramguard.run_tasks(ramguard._echo, [1, 2], 1, lambda r, t: res.append(r), label_fn=lambda t: f"task {t}", log=logs2.append, min_gib=7.0, avail_fn=avail,
                       poll_s=0, settle_s=0.0, sleep=lambda s: time.sleep(0.02), rule=rule)
    assert sorted(res) == [1, 2] and len(logs2) == 3 and all("task 1" in l for l in logs2), logs2


def test_ramguard_and_mutants():
    check_ramguard(ramguard.may_start)
    assert raises(check_ramguard, lambda a, m: True), "mutant: guard that always admits not caught"
    assert raises(check_ramguard, lambda a, m: a > m + 5), "mutant: threshold shifted not caught"
    assert raises(check_ramguard, lambda a, m: a >= 6.0), "mutant: 6 GiB instead of 7 not caught"
    assert raises(ramguard.run_tasks, ramguard._echo, [1], 5, lambda r, t: None), "mutant: 5 workers not refused"
    a = ramguard.avail_gib()
    assert 0.1 < a < 1024, a
    with env(MGYM6_DRY_CHAIN=None, MGYM6_MIN_RAM_GIB="1"):
        assert ramguard.min_gib_now() == 7.0                       # a real launch cannot lower the guard
    with env(MGYM6_DRY_CHAIN="1", MGYM6_MIN_RAM_GIB="5"):
        assert ramguard.min_gib_now() == 5.0
    with env(MGYM6_DRY_CHAIN=None, MGYM6_MIN_RAM_GIB=None):
        assert ramguard.min_gib_now() == 7.0
    src = open(os.path.join(HERE, "run_v6.py"), encoding="utf-8").read()
    assert "ramguard.run_tasks" in src and "pool.imap" not in src and "mp.get_context" not in src       # run_v6 starts states only through the guard


# ------------------------------------------------------------ B3-F: free-road class
def rec(v0, leads=(), confs=()):
    cyc = lambda lg, ct: dict(lead_gap=lg, conf_ttc=ct)
    lad = {n: dict(logs=[cyc(None, None) for _ in range(4)]) for n in ("crawl1.5", "f0.3", "f0.5", "f0.7", "f0.9", "f1.0")}
    for n in leads:
        lad[n]["logs"][2] = cyc(5.0, None)
    for n in confs:
        lad[n]["logs"][3] = cyc(None, 2.0)
    return dict(v0=v0, ladder=lad)


def check_free(fn):
    assert fn(rec(2.0))["free"] is True and fn(rec(1.0))["free"] is True
    assert fn(rec(0.99))["free"] is False and fn(rec(0.0))["free"] is False
    assert fn(rec(5.0, leads=("f1.0",)))["free"] is False            # an agent seen only on the fastest rung still disqualifies ("any rung")
    assert fn(rec(5.0, confs=("crawl1.5",)))["free"] is False
    assert fn(rec(5.0, leads=("f0.5",), confs=("f0.9",)))["agent_in_window"] is True and fn(rec(5.0))["agent_in_window"] is False


def test_free_road_class_and_mutants():
    check_free(r2.classify_free)
    base = r2.classify_free

    def mut_no_agent(r):
        d = base(r); d["free"] = r["v0"] >= 1.0; return d
    def mut_thr15(r):
        d = base(r); d["free"] = r["v0"] >= 1.5 and not d["agent_in_window"]; return d
    def mut_crawl_only(r):
        d = base(r); a = any(c["lead_gap"] is not None or c["conf_ttc"] is not None for c in r["ladder"]["crawl1.5"]["logs"]); d["free"] = r["v0"] >= 1.0 and not a; return d
    for m in (mut_no_agent, mut_thr15, mut_crawl_only):
        assert raises(check_free, m), m.__name__
    # real data control: REWARD_2_COUNTERFACTUALS Q2 row "v0 >= 1.0 and no agent, any rung": 5 of the 39 eligible states, 5 KNOWN, 0 not KNOWN (stored REWARD-1 records)
    res = json.load(open(os.path.join(P5, "raw", "results_v5.json"), encoding="utf-8"))["B3"]["eligible_detail"]
    el = {e["idx"]: e["known"] for e in res}
    free = []
    for sub in ("run_main", "ladder_20_160"):
        for l in gzip.open(os.path.join(P5, "raw", sub, "states.jsonl.gz"), "rt", encoding="utf-8"):
            r = json.loads(l)
            if r["idx"] in el and base(r)["free"]:
                free.append(r["idx"])
    assert len(free) == 5 and all(el[i] for i in free), (free, [el[i] for i in free])


# ------------------------------------------------------------ kinematic crawl share
def check_kin(fn):
    tab = {4.47: 0.369, 8.0: 0.234, 11.18: 0.191, 13.41: 0.176, 15.0: 0.170, 17.0: 0.168, 20.0: 0.168, 30.0: 0.168}      # REWARD_2_COUNTERFACTUALS Q4 (launch from rest)
    for L, want in tab.items():
        assert abs(fn(0.0, L) - want) < 6e-4, (L, fn(0.0, L), want)
    assert fn(0.0, 11.18) > r2.CRAWL_RATIO                      # 0.10 is unreachable from progress alone on a launch from rest
    q4 = json.load(open(os.path.join(P6, "reward2_counterfactuals.json"), encoding="utf-8"))["Q4_per_state"]
    for k, d in q4.items():
        assert abs(fn(d["v0"], d["limit"]) - d["kin_crawl_over_best"]) < 1e-9, k


def test_kinematic_crawl_share_and_mutants():
    check_kin(r2.kin_crawl_over_best)
    src = os.path.join(HERE, "reward2.py")
    m1 = mutated_module(src, "vs = vs + _clip(vt - vs, -3.0 * DT, 2.0 * DT)", "vs = vs + _clip(vt - vs, -3.0 * DT, 3.0 * DT)", "r2_mut1")
    m2 = mutated_module(src, "pc = kin_progress(v0, 1.5, limit)", "pc = kin_progress(v0, 3.0, limit)", "r2_mut2")
    m3 = mutated_module(src, "pb = max(kin_progress(v0, 0.9 * limit, limit), kin_progress(v0, limit, limit))", "pb = kin_progress(v0, 0.5 * limit, limit)", "r2_mut3")
    m4 = mutated_module(src, "for cyc in range(17):", "for cyc in range(10):", "r2_mut4")
    for m in (m1, m2, m3, m4):
        assert raises(check_kin, m.kin_crawl_over_best), m.__name__
    rows = [dict(kin_crawl_over_best=x) for x in (0.05, 0.12, 0.17, 0.37)]
    s = r2.kin_share(rows)
    assert s["n"] == 4 and s["n_kin_above_bar"] == 3 and abs(s["share_above_bar"] - 0.75) < 1e-12 and s["min"] == 0.05 and s["max"] == 0.37
    assert r2.kin_share([])["share_above_bar"] is None


# ------------------------------------------------------------ B3-F verdict
def R(idx, known, limit=8.0, free=True, inf=True):
    return dict(idx=idx, informative=inf, known=known, limit=limit, free=free)


def check_b3f(fn):
    five = [R(i, True) for i in range(5)]
    assert fn(five)["verdict"] == "PASS" and fn(five)["n_free_eligible"] == 5
    assert fn(five[:4] + [R(9, False)])["verdict"] == "FAIL" and fn(five[:4] + [R(9, False)])["not_known_idx"] == [9]
    assert fn(five[:4])["verdict"] == "UNDETERMINED_LT5"                                              # fewer than 5 F states
    # not-free states (known or not) never enter; ineligible free states (low limit, not informative) never enter
    assert fn(five[:4] + [R(11, True, free=False)])["verdict"] == "UNDETERMINED_LT5"
    assert fn(five + [R(12, False, free=False)])["verdict"] == "PASS"
    assert fn(five + [R(13, False, limit=3.0), R(14, False, inf=False)])["verdict"] == "PASS"
    assert fn(five + [R(13, False, limit=3.0), R(14, False, inf=False)])["n_free_eligible"] == 5


def test_b3f_verdict_and_mutants():
    check_b3f(r2.b3f_verdict)
    assert raises(check_b3f, lambda rows: r2.b3f_verdict(rows, free_of=lambda r: True)), "mutant: all states counted as free not caught"
    src = os.path.join(HERE, "reward2.py")
    m1 = mutated_module(src, "v = r1.b3_verdict(f_rows, at_cap=True)", "v = r1.b3_verdict(f_rows, at_cap=False)", "r2_b3f1")
    assert raises(check_b3f, m1.b3f_verdict), "mutant: not at cap not caught"
    m2 = mutated_module(src, "f_rows = [r for r in rows if r1.eligible(r[\"informative\"], r[\"limit\"]) and free_of(r)]", "f_rows = [r for r in rows if free_of(r)]", "r2_b3f2")
    assert raises(check_b3f, m2.b3f_verdict), "mutant: eligibility dropped not caught"
    m3 = mutated_module(src, "v = r1.b3_verdict(f_rows, at_cap=True)", "v = r1.b3_verdict(f_rows, at_cap=True, min_eligible=4)", "r2_b3f3")
    assert raises(check_b3f, m3.b3f_verdict), "mutant: minimum 4 not caught"


# ------------------------------------------------------------ the code pin
def check_pin(verify):
    with tmpdir() as t:
        os.makedirs(os.path.join(t, "scorer_gym", "tests"))
        os.makedirs(os.path.join(t, "tests"))
        a, b = os.path.join(t, "scorer_gym", "a.py"), os.path.join(t, "scorer_gym", "tests", "t.py")
        data = os.path.join(t, "data.json")
        for p, c in ((a, "x=1\n"), (b, "y=2\n"), (data, "{}")):
            open(p, "w", encoding="utf-8").write(c)
        pin = {"files": {"scorer_gym/a.py": dg.sha_file(a), "scorer_gym/tests/t.py": dg.sha_file(b), "data.json": dg.sha_file(data)}}
        pp = os.path.join(t, "CODE_PIN.json")
        json.dump(pin, open(pp, "w", encoding="utf-8"))
        assert verify(t, pp) == []
        open(a, "w", encoding="utf-8").write("x=2\n")
        assert verify(t, pp) == ["SHA scorer_gym/a.py"], verify(t, pp)
        open(a, "w", encoding="utf-8").write("x=1\n")
        open(data, "w", encoding="utf-8").write("{ }")
        assert verify(t, pp) == ["SHA data.json"]
        open(data, "w", encoding="utf-8").write("{}")
        open(os.path.join(t, "scorer_gym", "new.py"), "w", encoding="utf-8").write("z=3\n")
        assert verify(t, pp) == ["UNPINNED scorer_gym/new.py"]
        os.remove(os.path.join(t, "scorer_gym", "new.py"))
        os.remove(b)
        assert verify(t, pp) == ["MISSING scorer_gym/tests/t.py"]


def test_pin_check_and_mutant():
    check_pin(chain.verify_pin)

    def mut_exists_only(p6, pin_path=None):
        pin = json.load(open(pin_path or os.path.join(p6, "CODE_PIN.json"), encoding="utf-8"))
        return [f"MISSING {r}" for r in pin["files"] if not os.path.exists(os.path.join(p6, r))]
    assert raises(check_pin, mut_exists_only), "mutant: existence-only pin check not caught"

    def mut_no_unpinned(p6, pin_path=None):
        return [b for b in chain.verify_pin(p6, pin_path) if not b.startswith("UNPINNED")]
    assert raises(check_pin, mut_no_unpinned), "mutant: unpinned file allowed not caught"


# ------------------------------------------------------------ flag -> evaluator -> run_v6 record (USED state 108 of the seed-20261110 list; real caller)
def check_reach(run_mod):
    st = r1.load_states_v5()[108]
    res = run_mod._do_state(108, st, "ladder", 0.20, 0.05)
    assert res["curve_min_seg_m"] == 0.05
    free = [c for c in res["ladder"]["crawl1.5"]["logs"] if c["lead_gap"] is None and c["conf_ttc"] is None and not c["hard"]]
    assert len(free) >= 15 and all(c["active"]["q_free"] for c in free), "q_free not applicable on every free crawl cycle"
    assert res["ladder"]["crawl1.5"]["ret"] == r4_v6.reference(108)[0]["crawl1.5"], res["ladder"]["crawl1.5"]["ret"]      # raw/resim = CURVE_SPEED_DEFECT.md (0.3973)
    return res["ladder"]["crawl1.5"]["ret"]


def test_flag_to_evaluator_and_run_meta():
    check_reach(run_v6)
    assert raises(run_v6._do_state, 108, r1.load_states_v5()[108], "ladder", 0.20, None), "None passed silently"
    orig = run_v6.g.GymState

    class Dropped(orig):                          # mutant: the switch is lost between run_v6 and GymState
        def __init__(self, *a, **k):
            k["curve_min_seg_m"] = None
            super().__init__(*a, **k)
    run_v6.g.GymState = Dropped
    try:
        assert raises(check_reach, run_v6), "mutant: dropped flag not caught"
    finally:
        run_v6.g.GymState = orig


# ------------------------------------------------------------ R4 / G-REPRO judges (table = raw/resim, the exact re-simulation behind CURVE_SPEED_DEFECT.md)
DEFECT_TABLE = {20: 0.089, 73: 0.069, 108: 0.040, 118: 0.044, 119: 0.042, 31: 0.084, 46: 0.122, 121: 0.054, 125: 0.017, 134: 0.153}     # CURVE_SPEED_DEFECT.md part 2b


def ref_rows(perturb=None):
    rows = []
    for i in r4_v6.eligible_idx():
        rets, hards = r4_v6.reference(i)
        if perturb:
            rets, hards = perturb(i, dict(rets), dict(hards))
        rows.append(r4_v6.judge(i, rets, hards))
    return rows


def check_r4(decide, perturb=None, expect=True):
    r4, gr = decide(ref_rows(perturb), [])
    assert (r4 and gr) == expect, (perturb, r4, gr)


def test_r_arms_judges_and_mutants():
    el = r4_v6.eligible_idx()
    assert len(el) == 39
    rows = ref_rows()
    assert sum(r["known"] for r in rows) == 34 and sorted(r["idx"] for r in rows if not r["known"]) == sorted(r4_v6.NOT_KNOWN_EXPECTED)
    for r in rows:
        assert r["bit_identical_all_rungs"] and r["err_crawl_over_best"] == 0.0 and r["err_f09"] == 0.0 and r["err_f10"] == 0.0 and r["hard_equal_stored"] and r["verdict_equals_table"]
    for i, want in DEFECT_TABLE.items():                               # the table of CURVE_SPEED_DEFECT.md to its printed 3 decimals
        got = next(r for r in rows if r["idx"] == i)["crawl_over_best"]
        assert abs(got - want) < 6e-4, (i, got, want)
    check_r4(r4_v6.decide)                                              # the table itself passes
    # regression arms: each perturbation must make the decision FAIL
    shift = lambda key, f: (lambda i, rets, hards: (dict(rets, **{key: rets[key] * f}) if i == 20 else rets, hards))
    for name, pert in (("crawl +2 %", shift("crawl1.5", 1.02)), ("f1.0 -2 %", shift("f1.0", 0.98)), ("f0.9 +2 %", shift("f0.9", 1.02)),
                       ("hard flag flipped", lambda i, rets, hards: (rets, dict(hards, **{"f0.5": not hards["f0.5"]}) if i == 7 else hards)),
                       ("a KNOWN state turns not KNOWN", lambda i, rets, hards: (dict(rets, **{"f0.7": rets["f1.0"] * 2}) if i == 16 else rets, hards)),
                       ("a not-KNOWN state turns KNOWN", lambda i, rets, hards: (dict(rets, **{"crawl1.5": rets["crawl1.5"] * 0.1}) if i == 134 else rets, hards))):
        assert raises(check_r4, r4_v6.decide, pert), name
    # within the 1 % gate passes the gate (not R4's verdict test, which is exact on the set)
    ok1 = shift("crawl1.5", 1.005)
    check_r4(r4_v6.decide, ok1)
    # a missing state or an error also fails
    r4, gr = r4_v6.decide(ref_rows()[:-1], [])
    assert not (r4 or gr)
    r4, gr = r4_v6.decide(ref_rows(), [(3, "boom")])
    assert not (r4 or gr)
    # R2/R1 constants against the stored M-GYM-5 record
    from grepro_v6 import R2_EXPECT, USED
    st5 = json.load(open(os.path.join(P5, "raw", "grepro_v5.json"), encoding="utf-8"))["states"]
    for i in USED:
        rr = st5[str(i)]["rets_floor100"]
        assert abs(rr["crawl1.5"] / max(rr["f0.9"], rr["f1.0"]) - R2_EXPECT[i]) / R2_EXPECT[i] <= 0.01
        assert "slowest_ret" in st5[str(i)] and "pdm_ret" in st5[str(i)] and len(st5[str(i)]["rets_floor020"]) == 6


# ------------------------------------------------------------ the chain's test groups name real tests (a misspelled name would otherwise be skipped silently)
def test_chain_test_groups_name_real_tests():
    import re
    n = 0
    for f, names in chain.test_groups():
        src = open(f, encoding="utf-8").read()
        for name in names:
            ok = re.search(rf"^def {re.escape(name)}\(", src, re.M) or (f.endswith("test_curve_speed_v6.py") and f'"{name}":' in src)
            assert ok, (f, name)
            n += 1
    assert n >= 40, n


# ------------------------------------------------------------ the exclusion list and the capacity of the draw (dummy seed 7; NOT the real seed)
def check_excl(ex):
    v5 = {s["log"] for s in r1.load_states_v5()}
    assert len(v5) == 180 and v5 <= set(ex), "an excluded seed-20261110 log is missing"
    v4 = {s["log"] for s in json.load(open(r1.V4_STATES, encoding="utf-8"))}
    assert v4 <= set(ex) and set(g.used_logs_m_gym_2()) <= set(ex)
    assert len(ex) == len(set(ex))


def test_selection_v6_capacity_with_dummy_seed():
    ex = r2.excluded_logs_v6()
    check_excl(ex)
    assert raises(check_excl, r1.excluded_logs()), "mutant: exclusion without the 240 REWARD-1 logs not caught"
    rec_ = r2.exclusion_record()
    assert rec_["logs"] == ex and rec_["n_logs"] == len(ex) and rec_["n_reward1_240_logs"] == 180
    a = r2.select_states_v6(seed=7, exclude_logs=ex)                      # dummy seed: the same rule, a seed that is not spent
    assert len(a) == 240 and len({s["token"] for s in a}) == 240 and not ({s["log"] for s in a} & set(ex))
    assert [s["idx"] for s in a] == list(range(240)) and sum(s["sampler"] == "random" for s in a[:160]) == 80
    assert a == r2.select_states_v6(seed=7, exclude_logs=ex)
    # the frozen-list round trip, in a temp dir
    with tmpdir() as t:
        p = os.path.join(t, "EX.json")
        assert r2.freeze_exclusion(p) == "FROZEN" and r2.freeze_exclusion(p) == "REUSED" and r2.load_exclusion(p) == ex
        open(p, "ab").write(b" ")
        assert raises(r2.load_exclusion, p), "a modified frozen list was accepted"
    # (the real seed is deliberately NOT called here, not even to show a refusal: the guard is shown on an empty database in test_draw_gate_dry_and_mutants)


def test_presim_detector_and_mutants():
    import types
    import presim_v6 as P
    from shapely.geometry import LineString, box
    from shapely.ops import substring
    class CL:
        length = 200.0
        l = LineString([(0, 0), (200, 0)])
        def project(self, p): return self.l.project(p)
        def substring(self, a, b): return substring(self.l, a, b)
    N = types.SimpleNamespace
    ego = lambda v=5.0: N(rear_axle=N(x=0.0, y=0.0, heading=0.0), center=N(x=1.5, y=0.0), dynamic_car_state=N(rear_axle_velocity_2d=N(x=v)))
    obj = lambda x, y, vx=0.0, vy=0.0: N(center=N(x=x, y=y), box=N(geometry=box(x - 2, y - 1, x + 2, y + 1)), velocity=N(x=vx, y=vy))
    cl = CL()
    def cases(M):
        assert M.detect(cl, ego(), []) == (False, False)
        assert M.detect(cl, ego(), [obj(15, 0)]) == (True, False), "lead"
        assert M.detect(cl, ego(), [obj(-15, 0)]) == (False, False), "behind"
        assert M.detect(cl, ego(), [obj(15, 8)]) == (False, False), "static off corridor"
        assert M.detect(cl, ego(), [obj(15, 8, vy=-3.0)]) == (False, True), "crossing mover"
        assert M.detect(cl, ego(), [obj(15, 8, vy=3.0)]) == (False, False), "moving away"
        assert M.detect(cl, ego(5.0), [obj(45, 0)]) == (False, False), "beyond the 20 m corridor"
        assert M.detect(cl, ego(20.0), [obj(45, 0)]) == (True, False), "inside the 50 m corridor"
        assert M.is_free(1.0, [(False, False)]) and not M.is_free(0.99, [(False, False)])
        assert not M.is_free(5.0, [(False, False), (True, False)]) and not M.is_free(5.0, [(False, True)]) and not M.is_free(5.0, [])
    cases(P)
    src = open(P.__file__, encoding="utf-8").read()
    for old, new in (("not any(l or c for l, c in flags)", "not any(l for l, c in flags)"), ("v0 >= MIN_FREE_V0 and flags", "flags"), ("TTC_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)", "TTC_GRID = ()"),
                     ("max(RANGE_S * v, MIN_L)", "MIN_L"), ("and flags and not", "and not"), ("math.hypot(ov.x, ov.y) < MOVER_MPS", "True")):
        assert src.count(old) == 1, old
        M = types.ModuleType("mut")
        M.__file__ = P.__file__
        exec(compile(src.replace(old, new).replace('if __name__ == "__main__":\n    main()', ""), "mut", "exec"), M.__dict__)
        assert raises(cases, M), "mutant not caught: " + old
    # the fresh list is reachable only through the pinned-list loader, and classify never steps the simulator
    assert "rollout" not in src.split("def classify")[1].split("def do_state")[0] and "load_states_v6" in src


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
