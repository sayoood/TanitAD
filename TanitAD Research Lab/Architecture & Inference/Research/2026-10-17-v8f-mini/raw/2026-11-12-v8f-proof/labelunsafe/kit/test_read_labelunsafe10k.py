"""Tests of read_labelunsafe10k.py on tiny synthetic runs written here (portable to Linux). Child processes run in the folder named by the environment variable LABELUNSAFE_TEST_CWD
(default: this folder). The engine files are imported by path, as the tool does; nothing here reads a real run.

The synthetic world: window i is one clip of 31 label rows with a constant lead gap G0 and a logged ego moving along the x axis at v m/s; the model of a run moves at v + dv m/s on the same
line, so the model gap is G0 - dv * 0.2 * k and the model time gap (G0 - dv * 0.2 * k) / (v + dv). Four recipes (known by construction, table RECIPE below) and the quantities hand-derived:
  R1 = (G0 11.25, v 10, dv 0.2): lowest tau_m 0.98529 (k 30), first unsafe step 27, M1 T, M2 T, M3 T (tau_l 1.125), M4 F (0.98529 is not below 0.925)
  R2 = (11.25, 10, 0.5): lowest tau_m 0.78571, first 8, M1 T, M2 F, M3 T, M4 T
  R3 = (30, 10, 3.0): lowest tau_m 0.92308, first 29, M1 T, M2 T, M3 F (tau_l 3.0), M4 T
  R4 = (30, 1, 6.0): lowest tau_m -0.85714, first 20, gap below 2.0 m from step 24, M1 F, M2 F, M3 F, M4 T
Window 6 is (15, 20): the logged driver has tau_l 0.75, so U is not defined there.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import read_labelunsafe10k as LU  # noqa: E402

CWD = os.environ.get("LABELUNSAFE_TEST_CWD") or HERE
BASE = [(11.25, 10), (11.25, 10), (30, 10), (30, 1), (11.25, 10), (30, 10), (15, 20), (30, 10), (11.25, 10), (30, 10), (11.25, 10), (30, 10)]
DV = {"R1": 0.2, "R2": 0.5, "R3": 3.0, "R4": 6.0}
FLAGS = {"R1": (1, 1, 1, 0), "R2": (1, 0, 1, 1), "R3": (1, 1, 0, 1), "R4": (0, 0, 0, 1)}                  # M1 M2 M3 M4 by recipe (hand-derived above)
FIRST = {"R1": 27, "R2": 8, "R3": 29, "R4": 20}
SPEED = {"R1": 10.2, "R2": 10.5, "R3": 13.0, "R4": 7.0}
MINTAU = {"R1": 10.05 / 10.2, "R2": 8.25 / 10.5, "R3": 12.0 / 13.0, "R4": -6.0 / 7.0}
RLC = {0: {0: "R1", 1: "R2", 2: "R3", 3: "R4", 8: "R1"}, 1: {0: "R1", 1: "R2", 3: "R4", 5: "R3"}, 2: {0: "R1", 2: "R3", 4: "R2", 8: "R1"}, 3: {0: "R1", 1: "R2", 2: "R3", 3: "R4"}}
GEC = {0: {0: "R1", 4: "R1", 3: "R4"}, 1: {0: "R1", 9: "R3"}, 2: {2: "R3"}, 3: {}}
A90 = {0: {1: "R2"}, 1: {}, 2: {}, 3: {}}
W1 = {"RLc": RLC, "GEc": GEC, "A90": A90}


def tiny_cfg(expected=None, **kw):
    c = dict(LU.CFG)
    c.update({"expect_files": 6, "expect_lead": 12, "n_boot": 2000, "expected": expected if expected is not None else {
        "RLc": {"per_seed": [5 / 11, 4 / 11, 4 / 11, 4 / 11], "pooled": 17 / 44}, "GEc": {"pooled": 6 / 44}, "A90": {"pooled": 1 / 44}}})
    c.update(kw)
    return c


def build(tmp, assign, base=BASE, per_ep=2, skip_done=(), extra_file=None):
    """writes the twelve runs, the label join and the grid of a tiny world; returns the paths"""
    tmp = str(tmp)
    os.makedirs(tmp, exist_ok=True)
    root = os.path.join(tmp, "t1")
    n = len(base)
    names = ["clip%04d" % i for i in range(n)]
    gap = np.concatenate([np.full(31, b[0], np.float32) for b in base])
    np.savez(os.path.join(tmp, "join.npz"), clip=np.array(names), off=np.arange(n, dtype=np.int64) * 31, n=np.full(n, 31, np.int64), lead_valid=np.ones(31 * n, np.int8), lead_gap=gap)
    with open(os.path.join(tmp, "grid.json"), "w", encoding="utf-8") as f:
        json.dump({"order": names, "windows": {c: [0] for c in names}}, f)
    t = np.arange(1, 31) * 0.2
    for arm, tag in LU.TAGS.items():
        for s in (0, 1, 2, 3):
            folder = LU.run_folder(root, tag, s)
            dd = LU.dump_folder(root, tag, s)
            os.makedirs(dd, exist_ok=True)
            if (arm, s) not in skip_done:
                with open(os.path.join(folder, "T1_DONE"), "w", encoding="utf-8") as f:
                    f.write("done\n")
            for e in range(n // per_ep):
                ws_ = list(range(e * per_ep, (e + 1) * per_ep))
                g = np.zeros((len(ws_), 30, 2))
                cl = np.zeros((len(ws_), 30, 2))
                for r, i in enumerate(ws_):
                    v = base[i][1]
                    dv = DV[assign[arm][s][i]] if i in assign[arm][s] else 0.0
                    g[r, :, 0] = v * t
                    cl[r, :, 0] = (v + dv) * t
                np.savez(os.path.join(dd, "ep%03d.npz" % e), g=g, cl=cl, ws=np.zeros(len(ws_), np.int64), eid=np.array(["epi%05d" % e] * len(ws_)), clip_index=np.array(ws_, np.int64),
                         v0=np.array([base[i][1] for i in ws_], np.float64))
    if extra_file:
        np.savez(os.path.join(LU.dump_folder(root, LU.TAGS["RLc"], 0), "ep999.npz"), g=np.zeros((1, 30, 2)), cl=np.zeros((1, 30, 2)), ws=np.zeros(1, np.int64), eid=np.array(["epi99999"]), clip_index=np.zeros(1, np.int64), v0=np.zeros(1))
    return {"root": root, "join": os.path.join(tmp, "join.npz"), "grid": os.path.join(tmp, "grid.json")}


@pytest.fixture(scope="module")
def E():
    return LU.bind_engine(LU.default_paths())


@pytest.fixture(scope="module")
def w1(E, tmp_path_factory):
    p = build(tmp_path_factory.mktemp("w1"), W1)
    return LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"]), p


def pytest_text(n=7):
    return "test_read_labelunsafe10k.py .......\n\n%d passed in 3.21s\n" % n


# ---------------------------------------------------------------------------------------------- M1 to M5 by construction
def expected_measures():
    """hand-derived: 17 unsafe windows of RLc pooled over four seeds"""
    rec = [r for s in RLC for r in RLC[s].values()]
    n = len(rec)
    assert n == 17
    return {k: sum(FLAGS[r][j] for r in rec) / n for j, k in enumerate(("M1", "M2", "M3", "M4"))}


def test_world_is_the_one_derived_by_hand(w1):
    w, _ = w1
    assert w.complete and not w.problems
    assert w.n_ep == 6 and all(c == 12 for c in w.lead_counts.values())
    for s in (0, 1, 2, 3):
        S = w.S[("RLc", s)]
        assert sorted(np.flatnonzero(S["sel"]).tolist()) == sorted(RLC[s])
        assert not S["ud"][6]
        for i, r in RLC[s].items():
            assert S["first"][i] == FIRST[r]
            assert S["min_tm"][i] == pytest.approx(MINTAU[r], abs=1e-9)
            assert S["vm_first"][i] == pytest.approx(SPEED[r], abs=1e-9)


def test_measures_on_a_world_with_known_answers(w1):
    w, _ = w1
    cfg = tiny_cfg()
    m, beside, allm = LU.read_all(cfg, w)
    exp = expected_measures()
    for k in ("M1", "M2", "M3", "M4"):
        assert m[k]["value"] == pytest.approx(exp[k], abs=1e-12)
        assert m[k]["n"] == 17
        lo, hi = m[k]["interval"]
        assert 0.0 <= lo <= hi <= 1.0
    assert exp["M1"] == pytest.approx(14 / 17) and exp["M2"] == pytest.approx(10 / 17) and exp["M3"] == pytest.approx(10 / 17) and exp["M4"] == pytest.approx(11 / 17)
    assert m["M5"]["value"] == pytest.approx(4 / 7) and m["M5"]["n"] == 7
    c = beside["counts"]["RLc"]
    assert c["median_lowest_tau_m_s"] == pytest.approx(12 / 13, abs=1e-9)
    assert c["M3_over_all_scored_windows"]["value"] == pytest.approx(20 / 44) and c["M3_over_all_scored_windows"]["n"] == 44
    assert c["n_unsafe_windows_per_seed"] == [5, 4, 4, 4] and c["n_scored_windows_per_seed"] == [11] * 4
    assert c["distinct_windows_by_number_of_seeds_unsafe"] == {"1": 2, "2": 1, "3": 3, "4": 1}
    fu = beside["first_unsafe"]
    assert fu["first_unsafe_step_index"]["median"] == 27 and fu["first_unsafe_step_index"]["min"] == 8 and fu["first_unsafe_step_index"]["max"] == 29
    bands = fu["by_label_gap_at_first_unsafe_step"]
    assert [b["n"] for b in bands] == [0, 10, 7, 0]
    assert bands[1]["median_label_gap_m"] == 11.25 and bands[1]["median_model_speed_mps"] == pytest.approx(10.2)
    assert bands[2]["median_label_gap_m"] == 30.0 and bands[2]["median_model_speed_mps"] == pytest.approx(13.0)
    o = beside["GEc_unsafe_windows_also_unsafe_for_RLc_same_model_seed"]
    assert o["pooled"] == pytest.approx(4 / 6) and o["n_unsafe_windows_of_arm"] == 6 and o["per_seed"][:3] == pytest.approx([2 / 3, 1 / 2, 1.0]) and math.isnan(o["per_seed"][3])
    a = beside["arms_M1_to_M5"]["A90"]
    assert (a["M1"]["value"], a["M2"]["value"], a["M3"]["value"], a["M4"]["value"], a["M5"]["value"]) == (1.0, 0.0, 1.0, 1.0, 0.0) and a["M1"]["n"] == 1
    g = beside["arms_M1_to_M5"]["GEc"]
    assert g["M1"]["n"] == 6


def test_bootstrap_is_the_cluster_bootstrap_over_episodes_with_seed_135(w1):
    """the interval of M4 against an independent loop over the draws of numpy's generator seeded 135"""
    w, _ = w1
    m, _, _ = LU.read_all(tiny_cfg(), w)
    n_ep = 6
    num, den = np.zeros(n_ep), np.zeros(n_ep)
    for s in RLC:
        for i, r in RLC[s].items():
            den[i // 2] += 1
            num[i // 2] += FLAGS[r][3]
    rng = np.random.default_rng(135)
    idx = rng.integers(0, n_ep, size=(2000, n_ep))
    vals = []
    for d in idx:
        dd = den[d].sum()
        if dd > 0:
            vals.append(num[d].sum() / dd)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    assert m["M4"]["interval"] == pytest.approx([lo, hi], abs=1e-12)
    assert m["M4"]["value"] == pytest.approx(num.sum() / den.sum())


# ---------------------------------------------------------------------------------------------- names: every one reached
def meas(m3, m4, m5v):
    return {"M3": {"value": 0.0, "interval": list(m3), "n": 5}, "M4": {"value": 0.0, "interval": list(m4), "n": 5}, "M5": {"value": m5v, "interval": [0.0, 1.0], "n": 5}}


def test_names_by_rule_and_the_two_routines_agree():
    cases = [((0.5, 0.9), (0.1, 0.5), 0.6, LU.MECH_THRESH, LU.REP_SAME), ((0.4, 0.9), (0.1, 0.5), 0.5, LU.MECH_MIXED, LU.REP_SAME), ((0.6, 0.9), (0.1, 0.51), 0.49, LU.MECH_MIXED, LU.REP_DIFF),
             ((0.0, 0.2), (0.5, 0.9), 0.1, LU.MECH_CLOSER, LU.REP_DIFF), ((0.6, 0.9), (0.5, 0.9), 0.9, LU.MECH_CLOSER, LU.REP_SAME), ((0.2, 0.9), (0.3, 0.6), 0.0, LU.MECH_MIXED, LU.REP_DIFF),
             ((float("nan"),) * 2, (float("nan"),) * 2, float("nan"), LU.MECH_MIXED, LU.REP_DIFF)]
    for m3, m4, m5v, mech, rep in cases:
        assert LU.derive_names(meas(m3, m4, m5v)) == [mech, rep]
    assert {c[3] for c in cases} == {LU.MECH_THRESH, LU.MECH_CLOSER, LU.MECH_MIXED} and {c[4] for c in cases} == {LU.REP_SAME, LU.REP_DIFF}


def test_a_disagreement_of_the_two_routines_is_refused(monkeypatch):
    monkeypatch.setattr(LU, "mechanism_name_b", lambda m: LU.MECH_MIXED)
    with pytest.raises(SystemExit):
        LU.derive_names(meas((0.6, 0.9), (0.1, 0.4), 0.9))


def test_names_reached_through_worlds(E, tmp_path):
    same_all = {0: {0: "R1", 1: "R1", 2: "R1"}}
    thr = {s: dict(same_all[0]) for s in range(4)}                    # windows 0,1,2 are (11.25, 10): R1 everywhere: M3 true, M4 false, the same windows at all four seeds
    p = build(tmp_path / "thr", {"RLc": thr, "GEc": {s: {} for s in range(4)}, "A90": {s: {} for s in range(4)}})
    w = LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"])
    m, _, _ = LU.read_all(tiny_cfg(), w)
    assert m["M3"]["interval"] == [1.0, 1.0] and m["M4"]["interval"] == [0.0, 0.0] and m["M5"]["value"] == 1.0
    assert LU.derive_names(m) == [LU.MECH_THRESH, LU.REP_SAME]
    cl = {0: {0: "R2"}, 1: {2: "R2"}, 2: {4: "R2"}, 3: {8: "R2"}}     # R2 at four different windows: M4 true, the windows differ by seed
    p = build(tmp_path / "clo", {"RLc": cl, "GEc": {s: {} for s in range(4)}, "A90": {s: {} for s in range(4)}})
    w = LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"])
    m, _, _ = LU.read_all(tiny_cfg(), w)
    assert m["M4"]["interval"] == [1.0, 1.0] and m["M5"]["value"] == 0.0
    assert LU.derive_names(m) == [LU.MECH_CLOSER, LU.REP_DIFF]


def test_name_of_the_mixed_world_follows_the_intervals(w1):
    w, _ = w1
    m, _, _ = LU.read_all(tiny_cfg(), w)
    lo3, (lo4, hi4) = m["M3"]["interval"][0], m["M4"]["interval"]
    want = LU.MECH_THRESH if (lo3 >= 0.5 and hi4 <= 0.5) else LU.MECH_CLOSER if lo4 >= 0.5 else LU.MECH_MIXED
    assert LU.derive_names(m) == [want, LU.REP_SAME]


# ---------------------------------------------------------------------------------------------- controls, each with a planted fault
def test_u0_identity_passes_and_each_fault_fails(w1):
    w, _ = w1
    cfg = tiny_cfg()
    assert LU.ctl_u0(cfg, w)[0]
    bad = tiny_cfg(expected={"RLc": {"per_seed": [5 / 11, 4 / 11, 4 / 11, 4 / 11], "pooled": 17 / 44 + 0.01}, "GEc": {"pooled": 6 / 44}, "A90": {"pooled": 1 / 44}})
    ok, why = LU.ctl_u0(bad, w)
    assert not ok and "0.0" not in why
    bad = tiny_cfg(expected={"RLc": {"per_seed": [5 / 11, 4 / 11 + 0.01, 4 / 11, 4 / 11], "pooled": 17 / 44}, "GEc": {"pooled": 6 / 44}, "A90": {"pooled": 1 / 44}})
    assert not LU.ctl_u0(bad, w)[0]
    bad = tiny_cfg(expected={"RLc": {"pooled": 17 / 44}, "GEc": {"pooled": 6 / 44 + 0.001}, "A90": {"pooled": 1 / 44}})
    assert not LU.ctl_u0(bad, w)[0]
    k = ("GEc", 1)
    saved = w.A[k]["any"].copy()
    w.A[k]["any"] = ~w.A[k]["any"]
    try:
        assert not LU.ctl_u0(cfg, w)[0]
    finally:
        w.A[k]["any"] = saved
    assert LU.ctl_u0(cfg, w)[0]


def test_u1_second_routine_agrees_on_the_world_and_each_fault_fails(E, w1):
    w, _ = w1
    cfg = tiny_cfg()
    assert LU.ctl_u1(E, cfg, w)[0]
    k = ("RLc", 0)
    S2 = LU.second_routine(E, w.D[k]["g"], w.D[k]["cl"], w.L)
    lead = w.L["lead"]
    assert LU.routines_agree(w.S[k], S2, lead)
    for field, delta in (("min_tm", 1e-6), ("min_tl", 1e-6), ("min_gm", 1e-6)):
        S3 = {x: np.array(y, copy=True) for x, y in S2.items()}
        S3[field][np.flatnonzero(w.S[k]["sel"])[0]] += delta
        assert not LU.routines_agree(w.S[k], S3, lead)
    S3 = {x: np.array(y, copy=True) for x, y in S2.items()}
    S3["first"][np.flatnonzero(w.S[k]["sel"])[0]] += 1
    assert not LU.routines_agree(w.S[k], S3, lead)
    # tolerance: a difference of 1e-10 is within 1e-9
    S3 = {x: np.array(y, copy=True) for x, y in S2.items()}
    S3["min_tm"][np.flatnonzero(w.S[k]["sel"])[0]] += 1e-10
    assert LU.routines_agree(w.S[k], S3, lead)


def test_second_routine_matches_the_engine_on_curved_noisy_and_degenerate_paths(E):
    H = E.H
    rng = np.random.default_rng(7)
    t = np.arange(1, 31) * 0.2
    n = 9
    th = 0.15 * t
    g = np.zeros((n, 30, 2))
    for i in range(n):
        v, wv = 6.0 + i, 0.05 + 0.03 * i
        g[i, :, 0] = np.sin(wv * t) * v / wv
        g[i, :, 1] = (1 - np.cos(wv * t)) * v / wv
    g[0] = 0.0                                                            # stationary logged track: no non-degenerate segment
    cl = g * rng.uniform(0.7, 1.6, size=(n, 1, 1)) + rng.normal(0, 0.4, size=g.shape)
    gap = np.tile(np.linspace(14.0, 9.0, 31), (n, 1))
    val = np.ones((n, 31), bool)
    val[4:, 20:] = False
    G, ns, lead = H.build_track(val, gap)
    L = {"G": G, "ns": ns, "lead": lead}
    S1 = LU.summarize(LU.step_arrays(E, g, cl, L))
    S2 = LU.second_routine(E, g, cl, L)
    assert lead.all() and LU.routines_agree(S1, S2, lead)
    assert (S1["first"] > 0).any()


def test_u2_planted_passes_and_each_fault_fails(E, w1):
    w, _ = w1
    assert LU.ctl_u2(E, w)[0]
    # fault 1: the track read as the model track is not the logged track (it is the planted one)
    H = E.H
    g = w.ref["g"]
    adv = H.point_at_arc(g, H.logged_arc(g) + 0.35 * H.step_speeds(g))
    ok, why = LU.ctl_u2(E, w, ident_cl=adv)
    assert not ok and "logged track" in why
    # fault 2: a plant of zero travel makes no window unsafe
    ok, why = LU.ctl_u2(E, w, plant_s=0.0)
    assert not ok and "no window unsafe" in why


def test_u3_tests_outputs(tmp_path):
    a, b = tmp_path / "dev.out", tmp_path / "pod.out"
    a.write_text(pytest_text(), encoding="utf-8")
    b.write_text(pytest_text(9), encoding="utf-8")
    assert LU.ctl_u3([str(a), str(b)])[0]
    b.write_text("test_read_labelunsafe10k.py ...F\n1 failed, 3 passed in 1s\n", encoding="utf-8")
    assert not LU.ctl_u3([str(a), str(b)])[0]
    b.write_text("test_read_labelunsafe10k.py ....\n4 passed, 1 error in 1s\n", encoding="utf-8")
    assert not LU.ctl_u3([str(a), str(b)])[0]
    b.write_text("\n", encoding="utf-8")
    assert not LU.ctl_u3([str(a), str(b)])[0]
    b.write_text("5 passed in 1s\n", encoding="utf-8")
    assert not LU.ctl_u3([str(a), str(b)])[0]
    b.write_text(pytest_text(), encoding="utf-8")
    assert not LU.ctl_u3([str(a)])[0]
    assert not LU.ctl_u3([str(a), str(tmp_path / "missing.out")])[0]
    kit = tmp_path / "kit"
    kit.mkdir()
    c = kit / "pod.out"
    c.write_text(pytest_text(), encoding="utf-8")
    assert not LU.ctl_u3([str(a), str(c)])[0]
    assert LU.pytest_ok(pytest_text()) and not LU.pytest_ok("")


def test_u4_counts_and_each_fault_fails(E, tmp_path):
    p = build(tmp_path / "u4", W1, skip_done=[("GEc", 2)])
    w = LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"])
    ok, why = LU.ctl_u4(tiny_cfg(), w)
    assert not ok and "done marker" in why
    p = build(tmp_path / "u4b", W1)
    w = LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"])
    assert LU.ctl_u4(tiny_cfg(), w)[0]
    ok, why = LU.ctl_u4(tiny_cfg(expect_files=7), w)
    assert not ok and "dump file count" in why
    ok, why = LU.ctl_u4(tiny_cfg(expect_lead=13), w)
    assert not ok and "lead window count" in why
    p = build(tmp_path / "u4c", W1, extra_file=True)
    w = LU.load_world(E, tiny_cfg(), p["root"], p["join"], p["grid"])
    assert not LU.ctl_u4(tiny_cfg(), w)[0]


# ---------------------------------------------------------------------------------------------- stages end to end (in process) and the refusals
def stage_args(p, out, tests, stage):
    return [stage, "--t1-root", p["root"], "--out-dir", str(out), "--join", p["join"], "--grid", p["grid"]] + sum([["--tests-out", t] for t in tests], [])


def test_stages_in_process_and_refusals(E, tmp_path, capsys):
    p = build(tmp_path / "w", W1)
    out = tmp_path / "out"
    cfg = tiny_cfg()
    dev, pod = tmp_path / "dev.out", tmp_path / "pod.out"
    dev.write_text(pytest_text(), encoding="utf-8")
    pod.write_text(pytest_text(), encoding="utf-8")
    tests = [str(dev), str(pod)]
    assert LU.main(stage_args(p, out, tests, "names"), cfg) == 3                      # no controls file yet
    assert LU.main(stage_args(p, out, tests, "controls"), cfg) == 0
    txt = capsys.readouterr().out
    lines = txt.strip().splitlines()
    assert [x for x in lines if x.startswith("ctl ")] == ["ctl u%d PASS" % i for i in range(5)]
    assert lines[-1] == "ZZ READ-LABELUNSAFE10K controls all_pass True"
    rec = json.load(open(out / LU.CONTROLS_NAME, encoding="utf-8"))
    assert rec["card"] == "PREREG_V8F_PROOF_ADD135" and rec["all_pass"] is True and all(rec["pass_u%d" % i] is True for i in range(5)) and "inputs" in rec
    assert "measures" not in rec and "M1" not in json.dumps(rec)
    assert LU.main(stage_args(p, out, tests, "controls"), cfg) == 3                      # never overwritten
    # an input that changed since the controls stage: refused
    pod.write_text(pytest_text(8), encoding="utf-8")
    assert LU.main(stage_args(p, out, tests, "names"), cfg) == 3
    assert not (out / LU.RESULT_NAME).exists()
    pod.write_text(pytest_text(), encoding="utf-8")
    capsys.readouterr()
    assert LU.main(stage_args(p, out, tests, "names"), cfg) == 0
    txt = capsys.readouterr().out
    assert txt.strip().splitlines()[-1] == "ZZ READ-LABELUNSAFE10K names done"
    res = json.load(open(out / LU.RESULT_NAME, encoding="utf-8"))
    assert res["card"] == "PREREG_V8F_PROOF_ADD135" and "T1-kin/hb0.2" in res["tier"] and len(res["names"]) == 2
    assert set(res["measures"]) == {"M1", "M2", "M3", "M4", "M5"} and all(set(res["measures"][k]) == {"value", "interval", "n"} for k in res["measures"])
    assert "beside" in res and len(res["families"]) == 4
    assert res["measures"]["M1"]["value"] == pytest.approx(14 / 17)
    md = (out / LU.VERDICT_NAME).read_text(encoding="utf-8")
    assert "not driving performance" in md.lower() and "T1-kin/hb0.2" in md
    for bad in ("epi0000", "clip00"):
        assert bad not in md and bad not in json.dumps(res) and bad not in txt
    assert LU.main(stage_args(p, out, tests, "names"), cfg) == 3                         # outputs exist
    for word in ("voi" + "d", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"):
        assert word not in md.lower()


def test_controls_stage_with_a_failed_control_prints_no_value(E, tmp_path, capsys):
    p = build(tmp_path / "w", W1)
    out = tmp_path / "out"
    dev = tmp_path / "dev.out"
    dev.write_text(pytest_text(), encoding="utf-8")
    bad = tiny_cfg(expected={"RLc": {"per_seed": [0.5, 4 / 11, 4 / 11, 4 / 11], "pooled": 17 / 44}, "GEc": {"pooled": 6 / 44}, "A90": {"pooled": 1 / 44}})
    assert LU.main(stage_args(p, out, [str(dev)], "controls"), bad) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("ctl u0 FAIL") and lines[3].startswith("ctl u3 FAIL")
    assert lines[-1] == "ZZ READ-LABELUNSAFE10K controls all_pass False"
    assert not any(ch.isdigit() for ch in lines[0].split("seed")[0].replace("u0", ""))
    rec = json.load(open(out / LU.CONTROLS_NAME, encoding="utf-8"))
    assert rec["all_pass"] is False and rec["pass_u0"] is False and rec["pass_u1"] is True
    assert LU.main(stage_args(p, out, [str(dev)], "names"), bad) == 3


def run_child(args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "read_labelunsafe10k.py")] + args, cwd=CWD, env=env, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout


@pytest.mark.parametrize("variant", ["no_file", "u2_false", "all_pass_false", "string_true", "no_inputs", "wrong_card"])
def test_names_stage_is_refused_in_a_child_process(tmp_path, variant):
    out = tmp_path / "o"
    out.mkdir()
    rec = {"card": "PREREG_V8F_PROOF_ADD135", "all_pass": True, "inputs": {"files": {"x": {"path": str(tmp_path / "nothing"), "sha256": None}}, "run_dumps": {"a": {"path": "", "sha256": "", "n_files": 0}}}}
    for i in range(5):
        rec["pass_u%d" % i] = True
    if variant == "u2_false":
        rec["pass_u2"] = False
    elif variant == "all_pass_false":
        rec["all_pass"] = False
    elif variant == "string_true":
        rec["pass_u1"] = "true"
    elif variant == "no_inputs":
        del rec["inputs"]
    elif variant == "wrong_card":
        rec["card"] = "PREREG_V8F_PROOF_ADD134"
    if variant != "no_file":
        (out / LU.CONTROLS_NAME).write_text(json.dumps(rec), encoding="utf-8")
    rc, txt = run_child(["names", "--out-dir", str(out)])
    assert rc == 3 and "REFUSED" in txt and "names done" not in txt
    assert not (out / LU.RESULT_NAME).exists() and not (out / LU.VERDICT_NAME).exists()


def test_parser_has_the_documented_options():
    a = LU.build_parser().parse_args(["controls", "--tests-out", "a", "--tests-out", "b"])
    assert a.tests_out == ["a", "b"] and a.t1_root.endswith("integ10k/t1") and a.out_dir.endswith("labelunsafe/read_out")
    with pytest.raises(SystemExit):
        LU.build_parser().parse_args(["other"])
