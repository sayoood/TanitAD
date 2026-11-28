"""Pins force_s2_reroll_r8.py: the COMMITTED outcome table (`decide`), the set / class constants, the pure selection logic, the F_slow rebinding of
force_lon's module global (and its guards), the chunked driver wiring, and the analysis against ANALYTIC targets.

Run:  python -m pytest -q -p no:cacheprovider test_force_s2_r8.py      (tanitad venv; PYTHONPATH = <PKG>/code; no GPU, no dataset, no model, < 1 GB)

Literal expectations only -- never an expression over the code under test -- and every guard is paired with a value that must NOT trigger it, so that a
mutation of a threshold or a class index flips one of these RED (mutation_check_force_s2_r8.py proves it)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import force_lon_reroll_r8 as FL  # noqa: E402
import force_s2_reroll_r8 as F  # noqa: E402
import s6_lon_decomp as SD  # noqa: E402


def C(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


# ====================================================================================================== the committed table (verbatim from the brief)
@pytest.mark.parametrize("f1,fs,want", [
    # TIMING-CONTROLLABLE: closure_Fslow >= 0.50 with lo > 0 AND closure_F1 < 0.50
    (C(0.10, -0.2, 0.4), C(0.50, 0.01, 0.9), "TIMING-CONTROLLABLE"),         # exactly on the F_slow point threshold, lower bound just above 0
    (C(0.49, -0.2, 0.9), C(0.80, 0.30, 1.1), "TIMING-CONTROLLABLE"),         # F1 just below 0.50
    (C(0.50, -0.2, 0.9), C(0.80, 0.30, 1.1), "INCONCLUSIVE"),                # F1 exactly 0.50 but lo <= 0: not IDENTITY, and TIMING needs F1 < 0.50
    (C(0.10, -0.2, 0.4), C(0.49, 0.10, 0.9), "INCONCLUSIVE"),                # F_slow just below 0.50
    (C(0.10, -0.2, 0.4), C(0.80, 0.00, 1.1), "INCONCLUSIVE"),                # F_slow lower bound exactly 0 is NOT > 0
    (C(0.10, -0.2, 0.4), C(0.80, -0.20, 1.1), "INCONCLUSIVE"),
    # IDENTITY: closure_F1 >= 0.50 with lo > 0
    (C(0.50, 0.01, 0.9), C(0.10, -0.2, 0.4), "IDENTITY"),
    (C(0.49, 0.10, 0.9), C(0.10, -0.2, 0.4), "INCONCLUSIVE"),                # F1 just below 0.50
    (C(0.80, 0.00, 1.1), C(0.10, -0.2, 0.4), "INCONCLUSIVE"),                # F1 lower bound exactly 0
    (C(0.80, 0.30, 1.1), C(0.80, 0.30, 1.1), "IDENTITY"),                    # both obey: F1 >= 0.50 so the defect is not the class's timing
    # DECODER: both upper bounds < 0.25
    (C(0.05, -0.10, 0.2499), C(0.05, -0.10, 0.2499), "DECODER"),
    (C(0.05, -0.10, 0.25), C(0.05, -0.10, 0.2499), "INCONCLUSIVE"),          # F1 upper exactly 0.25 is NOT < 0.25
    (C(0.05, -0.10, 0.2499), C(0.05, -0.10, 0.25), "INCONCLUSIVE"),          # F_slow upper exactly 0.25
    (C(-0.40, -0.90, 0.10), C(-0.30, -0.80, 0.10), "DECODER"),               # forcing makes it worse
    (C(0.30, 0.10, 0.60), C(0.30, 0.10, 0.60), "INCONCLUSIVE"),
])
def test_decide_is_the_committed_table(f1, fs, want):
    assert F.decide(f1, fs) == want


@pytest.mark.parametrize("f1,fs", [(None, C(0.9, 0.5, 1.0)), (C(0.1, 0.0, 0.2), None), (C(None, None, None), C(0.9, 0.5, 1.0)),
                                   (C(0.1, 0.0, 0.2), C(0.9, None, None)), ({}, {})])
def test_decide_refuses_an_undefined_closure(f1, fs):
    assert F.decide(f1, fs).startswith("INCONCLUSIVE (closure undefined")


def test_pre_registered_constants_are_literal():
    assert (F.CLOSE_POINT, F.CLOSE_LO_GT, F.DEC_HI_LT) == (0.50, 0.0, 0.25)
    assert F.SETS == ("S2A", "S2B")
    assert F.SET_DEFS == {"S2A": ("slow_both", "stopping"), "S2B": ("decelerating",)}
    assert F.SET_EXPECTED_N == {"S2A": 434, "S2B": 424}
    assert (F.MAX_PER_SET, F.KB_N, F.CHUNKS_PER_SET, F.S2_NSLOT) == (220, 100, 2, 4)
    assert F.KA_BAR_M == 0.05
    assert F.E1_LITERAL == {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138}


def test_slow_classes_are_hold_on_S2A_and_brake_to_on_S2B():
    # refcv8_conditioning.py:63  LON6 = (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE)
    assert F.SLOW_CLASS6 == {"S2A": 4, "S2B": 2}
    assert FL.LON6 == ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")
    assert FL.LON6[F.SLOW_CLASS6["S2A"]] == "HOLD" and FL.LON6[F.SLOW_CLASS6["S2B"]] == "BRAKE_TO"
    F.assert_fl_internals()                                                  # the real force_lon source still has the spec line this relies on


# ====================================================================================================== a synthetic ref (480 rows = 24 episodes x 20 windows)
V, T = 5.0, np.array([0.5, 1.0, 1.5, 2.0])
DV2_CYCLE = ["slow_both", "stopping", "decelerating", "cruising", "accelerating"]
LON_CYCLE = [-100, 1, 3, 5, 2, 7]


def make_ref(a_gpu=1.02, b_echo=0.4, n_ep=24, per=20):
    n = n_ep * per
    row = np.arange(n)
    ep = row // per
    k = row % per
    G = np.zeros((n, 4, 2))
    G[:, :, 0] = V * T
    ha = G.copy()
    ha[:, :, 0] += b_echo
    os0 = G.copy()
    os0[:, :, 0] += a_gpu
    os1 = os0.copy()
    os1[:, :, 0] += 0.01
    return {"ep_index": ep, "ws": 10 + k, "mask": ~((ep % 2 == 0) & (k == 1)), "dv2": np.asarray([DV2_CYCLE[i % 5] for i in row]),
            "lon_v7": np.asarray([LON_CYCLE[i % 6] for i in row]), "tac": row % 8, "v0": np.full(n, 3.0),
            "csha": np.asarray([f"sha{e:09d}" for e in ep]), "eid": np.asarray([f"e{e:03d}" for e in ep]), "G": G,
            "os0": os0, "os1": os1, "ha": ha}, {int(e): f"clip-{int(e):03d}" for e in range(n_ep)}


# ====================================================================================================== pure selection logic
def test_assign_sets_applies_the_dv2_rule_on_the_compliant_windows_only():
    ref, _ = make_ref()
    s = F.assign_sets(ref["dv2"], ref["mask"])
    assert int(s["S2A"].sum()) == 180 and int(s["S2B"].sum()) == 96          # 24 eps x 8 / 4, minus 12 non-compliant slow_both windows (hand count)
    assert not (s["S2A"] & s["S2B"]).any()                                  # the sets are disjoint
    s2 = F.assign_sets(ref["dv2"], np.ones(len(ref["mask"]), bool))
    assert int(s2["S2A"].sum()) == 192 and int(s2["S2B"].sum()) == 96       # the mask matters: 12 more without it
    assert not s2["S2A"][np.asarray([d in ("cruising", "accelerating") for d in ref["dv2"]])].any()


def test_build_plan_refuses_when_the_regenerated_sets_are_not_the_analysts_434_424():
    ref, cid = make_ref()
    with pytest.raises(SystemExit):
        F.build_plan(ref, cid)                                              # synthetic counts are 180 / 96, not 434 / 424
    F.build_plan(ref, cid, expect_n={"S2A": 180, "S2B": 96})                # the same plan passes when the expectation is right


def test_build_plan_is_stratified_proportional_single_set_per_chunk_and_deterministic():
    ref, cid = make_ref()
    kw = dict(max_per_set=60, kb_n=20, chunks_per_set=2, expect_n=None)
    ent, st = F.build_plan(ref, cid, **kw)
    ent2, _ = F.build_plan(ref, cid, **kw)
    assert [(e["set"], e["row"], e["chunk"]) for e in ent] == [(e["set"], e["row"], e["chunk"]) for e in ent2]
    a = [e for e in ent if e["set"] == "S2A"]
    b = [e for e in ent if e["set"] == "S2B"]
    assert (len(a), len(b)) == (60, 60) and st["S2A"]["n_used"] == 60 and st["S2A"]["n_in_compliant_S2_set"] == 180
    # proportional: even episodes hold 7 S2A windows, odd 8 -> 60 of 180 = 2 per even episode, 3 per odd one (largest remainder, hand-derived)
    per = {}
    for e in a:
        per[e["ep_index"]] = per.get(e["ep_index"], 0) + 1
    assert sorted(set(per.values())) == [2, 3] and all(per[e] == (2 if e % 2 == 0 else 3) for e in per)
    # S2B: 4 per episode, 60 of 96 -> 2.5 per episode: 12 episodes get 3, 12 get 2
    perb = {}
    for e in b:
        perb[e["ep_index"]] = perb.get(e["ep_index"], 0) + 1
    assert sorted(perb.values()).count(3) == 12 and sorted(perb.values()).count(2) == 12
    # every chunk holds ONE set: S2A -> {0, 1}, S2B -> {2, 3}; balanced to within one window
    assert {e["chunk"] for e in a} == {0, 1} and {e["chunk"] for e in b} == {2, 3}
    assert abs(sum(e["chunk"] == 0 for e in a) - sum(e["chunk"] == 1 for e in a)) <= 1
    # the control subset: exactly kb_n per set, and exactly those windows carry N2 + Fown
    for es in (a, b):
        assert sum(e["control_subset"] for e in es) == 20
        assert all(("N2" in e["arms"] and "Fown" in e["arms"]) == e["control_subset"] for e in es)
    # the slow class is the SET's, not the window's
    assert {e["slow_class6"] for e in a} == {4} and {e["slow_class6"] for e in b} == {2}
    assert all("F2" in e["arms"] for e in ent)                              # F_slow runs on every window
    # sets <= the cap are taken whole (never padded, never dropped)
    ent3, _ = F.build_plan(ref, cid, max_per_set=500, kb_n=0, chunks_per_set=2, expect_n=None)
    assert sum(e["set"] == "S2A" for e in ent3) == 180 and sum(e["set"] == "S2B" for e in ent3) == 96
    assert all(e["control_subset"] for e in ent3)                           # kb_n = 0 means "every window" (force_lon's convention)


def test_f1_is_excluded_and_counted_for_unlabelled_and_masked_classes_only():
    ref, cid = make_ref()
    ent, st = F.build_plan(ref, cid, max_per_set=60, kb_n=20, expect_n=None)
    for e in ent:
        if e["lon_v7"] in (-100, 2):
            assert e["f1_class"] is None and e["f1_excluded_reason"] and "F1" not in e["arms"]
        else:
            assert e["f1_class"] == {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5}[e["lon_v7"]] and "F1" in e["arms"] and e["f1_excluded_reason"] is None
    for s in ("S2A", "S2B"):
        es = [e for e in ent if e["set"] == s]
        assert st[s]["n_with_F1"] == sum(e["f1_class"] is not None for e in es)
        assert sum(st[s]["F1_excluded_by_reason"].values()) == len(es) - st[s]["n_with_F1"]
        assert st[s]["n_with_F1"] < len(es)                                 # the synthetic cycle really contains excluded windows


# ====================================================================================================== the F_slow rebinding (force_lon's module global)
def test_slow_class_bound_rebinds_and_always_restores():
    before = FL.CRUISE6
    with F.slow_class_bound("S2A"):
        assert FL.CRUISE6 == 4
        assert FL.stage_roll.__globals__["CRUISE6"] == 4                    # the function resolves the name in FL's own dict
    assert FL.CRUISE6 == before == 1
    with pytest.raises(RuntimeError):
        with F.slow_class_bound("S2B"):
            assert FL.CRUISE6 == 2
            raise RuntimeError("boom")
    assert FL.CRUISE6 == 1


def test_roll_chunk_binds_the_chunks_set_class_and_refuses_mixed_or_missing_chunks(tmp_path, monkeypatch):
    plan = {"entries": [{"set": "S2A", "chunk": 0, "clip": "c0", "ws": 1}, {"set": "S2A", "chunk": 1, "clip": "c1", "ws": 1},
                        {"set": "S2B", "chunk": 2, "clip": "c2", "ws": 1}, {"set": "S2B", "chunk": 3, "clip": "c3", "ws": 1},
                        {"set": "S2A", "chunk": 9, "clip": "c4", "ws": 1}, {"set": "S2B", "chunk": 9, "clip": "c5", "ws": 1}]}
    (tmp_path / "plan_raw.json").write_text(json.dumps(plan))
    monkeypatch.setattr(F, "SCR", tmp_path)
    monkeypatch.setattr(FL, "SCR", tmp_path)
    monkeypatch.setattr(F, "assert_fl_internals", lambda: None)             # tested on the real function above
    seen = []
    monkeypatch.setattr(FL, "stage_roll", lambda a: seen.append((a.chunk, FL.CRUISE6)) or 0)
    ns = lambda **k: type("A", (), {"limit_windows": 0, "driver": 0, "chunk": -1, "tag": "t", "save_every": 2, **k})()
    for c, want in ((0, 4), (1, 4), (2, 2), (3, 2)):
        assert F.stage_roll(ns(chunk=c)) == 0
        assert seen[-1] == (c, want)
    assert FL.CRUISE6 == 1                                                  # restored after every chunk
    with pytest.raises(SystemExit):
        F.stage_roll(ns(chunk=9))                                           # a chunk holding two sets is refused (the class is bound per chunk)
    with pytest.raises(SystemExit):
        F.stage_roll(ns(chunk=-1))                                          # neither --driver nor --chunk
    with pytest.raises(SystemExit):
        F.stage_roll(ns(chunk=0, limit_windows=3))


def test_roll_driver_spawns_children_of_THIS_file_over_this_scratch_and_resumes(tmp_path, monkeypatch):
    entries = [{"set": s, "chunk": c, "clip": f"clip{c}", "ws": 7, "arms": ["F2"]} for c, s in ((0, "S2A"), (1, "S2A"), (2, "S2B"), (3, "S2B"))]
    (tmp_path / "plan_raw.json").write_text(json.dumps({"entries": entries}))
    monkeypatch.setattr(F, "SCR", tmp_path)
    monkeypatch.setattr(FL, "SCR", tmp_path)
    monkeypatch.setattr(FL, "__file__", FL.__file__)                        # recorded so teardown restores the module attribute the driver rebinding touches
    monkeypatch.setattr(F, "assert_fl_internals", lambda: None)
    import subprocess
    calls = []

    def fake_run(cmd, check=False, **kw):
        calls.append(list(cmd))
        c = int(cmd[cmd.index("--chunk") + 1])
        p = tmp_path / "roll_full.json"
        res = json.loads(p.read_text()) if p.exists() else {"windows": {}}
        res["windows"][f"clip{c}|7"] = {"arms": {}}
        p.write_text(json.dumps(res))

    monkeypatch.setattr(subprocess, "run", fake_run)
    ns = type("A", (), {"limit_windows": 0, "driver": 4, "chunk": -1, "tag": "full", "save_every": 2})()
    assert F.stage_roll(ns) == 0
    assert [c[c.index("--chunk") + 1] for c in calls] == ["0", "1", "2", "3"]
    assert all(Path(c[4]).name == "force_s2_reroll_r8.py" for c in calls)    # NOT force_lon_reroll_r8.py: the child must run this file's chunk wrapper
    assert all(c[5:7] == ["roll", "--tag"] for c in calls)
    n_before = len(calls)
    assert F.stage_roll(ns) == 0
    assert len(calls) == n_before                                           # resumable: every chunk is already in the artifact -> nothing is re-run
    for wrong in (3, 5):                                                    # a driver that does not match the plan's chunk count would skip / invent a chunk
        with pytest.raises(SystemExit):
            F.stage_roll(type("A", (), {"limit_windows": 0, "driver": wrong, "chunk": -1, "tag": "full", "save_every": 2})())


def test_fl_internals_guard_goes_red_when_the_spec_line_changes(monkeypatch):
    import inspect
    src = inspect.getsource(FL.stage_roll)
    monkeypatch.setattr(inspect, "getsource", lambda f: src.replace('"F2": (0, CRUISE6)', '"F2": (0, 1)'))
    with pytest.raises(SystemExit):
        F.assert_fl_internals()
    monkeypatch.setattr(inspect, "getsource", lambda f: src + "\n    CRUISE6 = 1\n")
    with pytest.raises(SystemExit):
        F.assert_fl_internals()


# ====================================================================================================== the analysis against ANALYTIC targets
def synth_roll(ref, ents, *, a=1.0, f1_c=None, fs_c=None, own_c=0.5, n2_c=1.0, bad_forced=False):
    """Roll artifact with exact errors: echo error b=0.4 (from make_ref), N error `a`, F1 error `f1_c`, F_slow error `fs_c` (default: = N), x-offsets on G."""
    win = {}
    for e in ents:
        r = e["row"]
        G = np.asarray(ref["G"][r], np.float64)

        def traj(c):
            P = np.zeros((8, 2))
            P[:4] = G
            P[:4, 0] += c
            P[4:, 0] = 30.0
            return P.tolist()
        arms = {"N": {"traj": traj(a), "sel_idx": 0}}
        if "F1" in e["arms"]:
            arms["F1"] = {"traj": traj(a if f1_c is None else f1_c), "sel_idx": 0, "forced6": e["f1_class"]}
        arms["F2"] = {"traj": traj(a if fs_c is None else fs_c), "sel_idx": 0, "forced6": 1 if bad_forced else e["slow_class6"]}
        own6 = 5
        if "N2" in e["arms"]:
            arms["N2"] = {"traj": traj(n2_c), "sel_idx": 0}
        if "Fown" in e["arms"]:
            arms["Fown"] = {"traj": traj(own_c), "sel_idx": 0, "forced6": own6}
        win[f"{e['clip']}|{e['ws']}"] = {"set": e["set"], "arms": arms, "own6": own6, "head_argmax_v7": int(e["head_argmax_v7"]),
                                          "head_logits_v7": [0.0] * 8, "wall_s": 1.0}
    return win


def plan(ref, cid):
    return F.build_plan(ref, cid, max_per_set=60, kb_n=20, chunks_per_set=2, expect_n=None)[0]


def analyze(ref, ents, win, **kw):
    return F.analyze_core(ref, ents, win, SD, **kw)


def test_timing_controllable_when_F_slow_recovers_the_echo_and_F1_does_nothing():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    # echo error b = 0.4 (make_ref); N error 1.0; F_slow error 0.4 -> closure = (1.0 - 0.4) / (1.0 - 0.4) = 1.0 exactly; F1 == N -> closure 0.0
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0))
    p = rep["PRIMARY"]
    assert p["closure_Fslow"]["point"] == pytest.approx(1.0) and p["closure_Fslow"]["lo"] == pytest.approx(1.0) and p["closure_Fslow"]["hi"] == pytest.approx(1.0)
    assert p["closure_F1"]["point"] == pytest.approx(0.0) and p["closure_F1"]["hi"] == pytest.approx(0.0)
    assert p["verdict"] == "TIMING-CONTROLLABLE"
    assert rep["content_check"]["PASS"] and not rep["coverage"]["PARTIAL"] and rep["n_rolled"] == 120


def test_identity_when_F1_recovers_the_echo_and_F_slow_does_nothing():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, f1_c=0.4, fs_c=1.0))
    assert rep["PRIMARY"]["closure_F1"]["point"] == pytest.approx(1.0) and rep["PRIMARY"]["closure_Fslow"]["point"] == pytest.approx(0.0)
    assert rep["PRIMARY"]["verdict"] == "IDENTITY"


def test_decoder_when_neither_force_moves_the_plan_and_inconclusive_in_between():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, f1_c=1.0, fs_c=1.0))
    assert rep["PRIMARY"]["verdict"] == "DECODER"
    # closure 0.3 exactly on both: (1.0 - 0.82) / (1.0 - 0.4) = 0.30 -> between 0.25 and 0.50 with a point-mass CI: INCONCLUSIVE
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, f1_c=0.82, fs_c=0.82))
    assert rep["PRIMARY"]["closure_Fslow"]["point"] == pytest.approx(0.30) and rep["PRIMARY"]["verdict"] == "INCONCLUSIVE"
    assert "to_reach_closure_ge_0.50_with_lo_gt_0" in rep["PRIMARY"]["n_needed"]["F_slow"]               # n_needed is reported for the non-decision
    # F_slow closure 0.70 and F1 closure 0.70: F1 >= 0.50 -> IDENTITY (the committed table checks F1's own bound; F_slow cannot rescue it)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, f1_c=0.58, fs_c=0.58))
    assert rep["PRIMARY"]["closure_F1"]["point"] == pytest.approx(0.70) and rep["PRIMARY"]["verdict"] == "IDENTITY"


def test_f1_closure_is_computed_on_the_labelled_windows_only_and_the_like_for_like_is_reported():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, f1_c=0.4, fs_c=0.4))
    n_f1 = sum("F1" in e["arms"] for e in ents)
    assert 0 < n_f1 < len(ents)
    assert rep["PRIMARY"]["closure_F1"]["n_windows"] == n_f1 and rep["PRIMARY"]["closure_Fslow"]["n_windows"] == len(ents)
    assert rep["PRIMARY"]["closure_Fslow_like_for_like_on_F1_windows"]["n_windows"] == n_f1
    assert sum(v["with_F1"] for v in rep["sample"].values()) == n_f1
    assert sum(v["F1_excluded"] for v in rep["sample"].values()) == len(ents) - n_f1


def test_content_check_refuses_a_verdict_when_forced6_is_not_what_the_plan_says():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    good = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0))
    assert good["content_check"]["PASS"] and good["content_check"]["n_forced6_mismatch"] == 0
    bad = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0, bad_forced=True))      # the rebinding silently did nothing: F2 forced CRUISE (1)
    assert not bad["content_check"]["PASS"] and bad["content_check"]["n_forced6_mismatch"] == 120
    assert bad["PRIMARY"]["verdict"].startswith("REFUSED") and bad["PRIMARY"]["verdict_per_committed_table"] == "TIMING-CONTROLLABLE"


def test_partial_roll_is_labelled_and_a_missing_arm_record_fails_the_content_check():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    win = synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0)
    keys = sorted(win)
    part = {k: win[k] for k in keys[:70]}
    rep = analyze(ref, ents, part)
    assert rep["coverage"]["PARTIAL"] and rep["n_rolled"] == 70 and rep["n_planned"] == 120
    k0 = keys[0]
    del win[k0]["arms"]["F2"]
    assert analyze(ref, ents, win)["content_check"]["n_missing_arm_records"] == 1
    assert not analyze(ref, ents, win)["content_check"]["PASS"]


def test_ka_kb_and_the_banked_floor_read_their_analytic_values():
    ref, cid = make_ref(a_gpu=1.02)                                          # banked GPU os error 1.02; CPU N error 1.00 -> KA diff exactly -0.02
    ents = plan(ref, cid)
    # own-class force moves the plan by 0.5 from N (own_c=0.5 -> displacement 0.5); F_slow error 0.4 -> displacement 0.6; N2 error 1.0 -> displacement 0
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0, own_c=0.5, n2_c=1.0))
    ka = rep["controls"]["KA"]
    assert ka["ADE_diff_CPU_N_minus_GPU_os"]["all"]["mean"] == pytest.approx(-0.02)
    assert ka["bar_exceeded"]["all"] is False and ka["bar_abs_mean_ADE_diff_m"] == 0.05
    assert ka["BANKED_INFERENCE_FLOOR_GPU_seed1_minus_seed0_os_same_windows"]["all"]["mean"] == pytest.approx(0.01, abs=1e-6)
    assert ka["CPU_N2_minus_CPU_N_two_CPU_draws (control subset)"]["all"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert ka["median_L2_waypoint_diff_S2_slots_m"] == pytest.approx(0.02, abs=1e-6)
    kb = rep["controls"]["KB"]
    assert kb["own_minus_Fslow"]["n_windows"] == 40 and kb["own_minus_Fslow"]["delta"] == pytest.approx(-0.1)      # |0.5| - |0.6| displacement
    assert kb["PASS (own-class displacement < F_slow displacement, CI upper < 0)"] is True
    # the other direction: own-class force moves the plan MORE than F_slow -> KB must FAIL
    rep2 = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.9, f1_c=1.0, own_c=0.2))
    assert rep2["controls"]["KB"]["PASS (own-class displacement < F_slow displacement, CI upper < 0)"] is False
    # a CPU-vs-GPU gap beyond the bar is reported as exceeded
    big = analyze(*make_ref(a_gpu=1.2)[:1], ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0))
    assert big["controls"]["KA"]["bar_exceeded"]["all"] is True


def test_e1_arithmetic_shift_form_has_the_hand_computed_value():
    ref, cid = make_ref(a_gpu=1.02, b_echo=0.4)
    ents = plan(ref, cid)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.7, f1_c=1.0))
    ar = rep["ARITHMETIC_E1_if_S2A_u_S2B_took_the_Fslow_plans"]
    assert "NOT A RESULT" in ar["LABEL"]
    # banked d = 1.02 - 0.4 = 0.62 on every window; each set's rolled shift = (0.7 - 1.0) = -0.3
    assert ar["shift_per_set_m"] == {"S2A": pytest.approx(-0.3), "S2B": pytest.approx(-0.3)}
    assert ar["E1_native_banked_GPU_os"]["mean"] == pytest.approx(0.62)
    # compliant = 480 - 12 = 468 windows; S2A u S2B = 180 + 96 = 276 windows move by -0.3: 0.62 - 0.3 * 276 / 468
    assert ar["E1_if_Fslow"]["mean"] == pytest.approx(0.62 - 0.3 * 276 / 468)
    assert ar["n_windows"] == 468


def test_control_R_reproduces_a_literal_E1_cell_and_can_fail():
    ref, cid = make_ref(a_gpu=1.0, b_echo=1.0085)                            # d = 1.0 - 1.0085 = -0.0085 on every compliant window
    ents = plan(ref, cid)
    win = synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0)
    lit = {"delta": -0.0085, "lo": -0.0085, "hi": -0.0085, "n_windows": 468, "n_episodes": 24}
    r = analyze(ref, ents, win, expect_e1=lit)["controls"]["R_E1_cell_from_saved_arrays"]
    assert r["PASS"] is True and r["got"] == lit
    r2 = analyze(ref, ents, win, expect_e1={**lit, "delta": -0.0086})["controls"]["R_E1_cell_from_saved_arrays"]
    assert r2["PASS"] is False


def test_null_intervention_n2_closure_is_reported_as_the_noise_floor_of_the_statistic():
    ref, cid = make_ref()
    ents = plan(ref, cid)
    rep = analyze(ref, ents, synth_roll(ref, ents, a=1.0, fs_c=0.4, f1_c=1.0, n2_c=0.82))     # N2 = another draw that happens to be 0.18 m better
    n2 = rep["secondary"]["N2_as_the_numerator_arm (a NULL intervention, another sampler draw: the noise floor of this statistic)"]
    assert n2["n_windows"] == 40 and n2["point"] == pytest.approx(0.30)


# ====================================================================================================== the stages end to end on the synthetic ref (tmp dirs)
def test_select_then_analyze_stages_write_the_artifacts_without_raw_clip_ids_in_the_package(tmp_path, monkeypatch, capsys):
    ref, cid = make_ref()
    scr, out = tmp_path / "scr", tmp_path / "out"
    monkeypatch.setattr(F, "SCR", scr)
    monkeypatch.setattr(F, "OUT", out)
    monkeypatch.setattr(FL, "SCR", scr)
    F.save_ref(ref, cid, tmp_path / "ref.npz")
    assert F.main(["select", "--ref-npz", str(tmp_path / "ref.npz"), "--max-per-set", "60", "--kb-n", "20", "--no-expect-counts"]) == 0
    assert "ZZFS2SELECTOKZZ" in capsys.readouterr().out
    raw = json.loads((scr / "plan_raw.json").read_text())["entries"]
    assert len(raw) == 120 and all(isinstance(e["row"], int) and isinstance(e["f1_class"], (int, type(None))) for e in raw)
    pk = (out / "plan_sha12.json").read_text() + (out / "select_stats.json").read_text()
    assert "clip-" not in pk and "sha000" in pk                              # the package view is sha12-keyed; the raw ids live in the scratch plan only
    assert (out / "select_stats.json").exists()
    win = synth_roll(ref, [dict(e) for e in raw], a=1.0, fs_c=0.4, f1_c=1.0)
    (scr / "roll_full.json").write_text(json.dumps({"windows": win}))
    assert F.main(["status"]) == 0
    assert "ZZFS2STATUSZZ rolled=120/120" in capsys.readouterr().out
    assert F.main(["analyze", "--ref-npz", str(tmp_path / "ref.npz"), "--no-expect-e1"]) == 0
    assert "ZZFS2ANALYZEDONEZZ" in capsys.readouterr().out
    rep = json.loads((out / "FORCE_S2.json").read_text())
    assert rep["PRIMARY"]["verdict"] == "TIMING-CONTROLLABLE" and rep["headline"]["content_check_PASS"] is True
    md = (out / "FORCE_S2.md").read_text()
    assert "TIMING-CONTROLLABLE" in md and "ARITHMETIC, NOT A RESULT" in md
    assert "clip-" not in json.dumps(rep) and "clip-" not in md


def test_json_cleaner_turns_numpy_scalars_into_python_numbers_not_strings():
    out = F.clean({"a": np.int64(3), "b": np.float32(0.5), "c": np.array([1, 2]), "d": np.bool_(True), "e": (np.int32(1), None)})
    assert json.loads(json.dumps(out)) == {"a": 3, "b": 0.5, "c": [1, 2], "d": True, "e": [1, None]}


# ====================================================================================================== the real-data loader glue, against FAKE e1_decomp / clamp_posthoc modules
# (the real path needs the commit headroom the launcher waits for and could not be smoked tonight; the key names below are those of
#  e1_decomp_r8.load_seed's ctx dict and panel_layout's (man, files, ep_of), read from source)
def _fake_world(tmp_path, monkeypatch, *, seed1_shuffled=False, bad_sha=False, bad_slots=False):
    import types
    n, n_ep = 4754, 138
    ep = np.concatenate([np.repeat(np.arange(n_ep - 1), 34), np.full(n - 34 * (n_ep - 1), n_ep - 1)])
    assert len(ep) == n
    ws = np.concatenate([10 + np.arange((ep == e).sum()) for e in range(n_ep)])
    mask = np.ones(n, bool)
    mask[::8] = False
    mask[np.nonzero(mask)[0][4159:]] = False                              # exactly 4159 compliant
    cid = {e: f"clip-{e:03d}" for e in range(n_ep)}
    sha12 = lambda c: "s" + c[-3:]                                         # noqa: E731
    G = np.zeros((n, 4, 2), np.float32)
    G[:, :, 0] = V * T
    dv2 = np.asarray([DV2_CYCLE[i % 5] for i in range(n)], dtype=object)

    def cx_of(seed):
        e, w = ep.copy(), ws.copy()
        if seed1_shuffled and seed == 1:
            w[:2] = w[:2][::-1]
        return {"seed": seed, "sha": "MASKSHA", "mask_full": mask, "eid": np.asarray([f"e{x:03d}" for x in ep], dtype=object), "ep_index": e, "ws": w,
                "csha": np.asarray([sha12("bad" if (bad_sha and i == 0) else cid[x]) for i, x in enumerate(ep)], dtype=object),
                "lon_v7": np.full(n, 1), "tac": np.full(n, 3), "v0": np.full(n, 2.0),
                "A": {"g": G, "os": G + 1.0 + 0.01 * seed, "ha0_ext": G + 0.4}}
    comp = tmp_path / "compliant.json"
    comp.write_text(json.dumps({"controls": {"ALL_PASS": True}}))
    e1 = types.SimpleNamespace(COMPLIANT_JSON=comp, MASK_SHA="MASKSHA", ARCHIVE="X", TREE="t", load_seed=lambda seed, tools, rc, cj: cx_of(seed),
                               build_partitions=lambda cx, m, fl: ({}, {"dv2_all": dv2}))
    grid = {"slots": [0, 1, 2, 5] if bad_slots else [0, 1, 2, 3], "dt_s": 0.5}
    nr = types.SimpleNamespace(panel_layout=lambda pdir: ({"grid": grid, "episodes": [{"file_index": e, "episode_index": e, "clip_id": c} for e, c in cid.items()]}, [], {}))
    cp = types.SimpleNamespace(get_readers=lambda: (nr, None, None, None, sha12), run_cfg=lambda: {})
    monkeypatch.setitem(sys.modules, "e1_decomp_r8", e1)
    monkeypatch.setitem(sys.modules, "clamp_posthoc_r8", cp)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    return ep, ws, mask, cid


def test_load_s2_arrays_assembles_the_ref_from_the_loader_keys_and_survives_a_save_load_round_trip(tmp_path, monkeypatch):
    ep, ws, mask, cid = _fake_world(tmp_path, monkeypatch)
    ref, cmap = F.load_s2_arrays()
    assert len(ref["ep_index"]) == 4754 and int(ref["mask"].sum()) == 4159 and cmap == cid
    assert ref["G"].shape == (4754, 4, 2) and ref["os0"].shape == ref["os1"].shape == ref["ha"].shape == (4754, 4, 2)
    assert float(ref["os1"][0, 0, 0] - ref["os0"][0, 0, 0]) == pytest.approx(0.01, abs=1e-6)       # os1 comes from seed 1, os0 from seed 0
    assert float(ref["ha"][0, 0, 0] - ref["G"][0, 0, 0]) == pytest.approx(0.4, abs=1e-6)
    assert ref["dv2"].dtype.kind == "U" and ref["csha"].dtype.kind == "U" and ref["eid"].dtype.kind == "U"
    F.save_ref(ref, cmap, tmp_path / "r.npz")
    ref2, cmap2 = F.load_ref(tmp_path / "r.npz")
    assert cmap2 == cmap and sorted(ref2) == sorted(ref) and all(np.array_equal(ref[k], ref2[k]) for k in ref)
    ents, st = F.build_plan(ref2, cmap2, expect_n=None)                    # the plan builds from the loaded ref (counts differ from 434/424 on this fake)
    assert all(e["clip"] == cid[e["ep_index"]] and e["clip_sha12"] == "s" + e["clip"][-3:] for e in ents)


@pytest.mark.parametrize("kw", [{"seed1_shuffled": True}, {"bad_sha": True}, {"bad_slots": True}])
def test_load_s2_arrays_refuses_a_misordered_seed_a_clip_sha_mismatch_and_a_non_s2_grid(tmp_path, monkeypatch, kw):
    _fake_world(tmp_path, monkeypatch, **kw)
    with pytest.raises(SystemExit):
        F.load_s2_arrays()


def test_load_s2_arrays_refuses_a_compliant_mask_that_is_not_the_banked_one(tmp_path, monkeypatch):
    _fake_world(tmp_path, monkeypatch)
    sys.modules["e1_decomp_r8"].MASK_SHA = "SOMETHINGELSE"
    with pytest.raises(SystemExit):
        F.load_s2_arrays()


def test_load_s2_arrays_needs_cpu_only(tmp_path, monkeypatch):
    _fake_world(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        F.load_s2_arrays()
